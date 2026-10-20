"""Persistent Qdrant vectors and a BM25 index over exactly the same chunks."""

import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from erp_copilot.config.settings import get_settings
from erp_copilot.embeddings.qwen import QwenEmbeddings
from erp_copilot.ingestion.chunking import load_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text.lower())


def join_passages(texts: list[str]) -> str:
    """Join neighboring chunks while removing an exact shared boundary."""
    merged = ""
    for text in texts:
        overlap = 0
        for size in range(min(200, len(merged), len(text)), 19, -1):
            if merged.endswith(text[:size]):
                overlap = size
                break
        merged += ("\n\n" if merged and not overlap else "") + text[overlap:]
    return merged


class SearchStore:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = QwenEmbeddings()
        self._client = None
        self.documents = []
        self.bm25 = None

    @property
    def client(self):
        if self._client is None:
            if self.settings.qdrant_url:
                self._client = QdrantClient(
                    url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key
                )
            else:
                path = self.settings.data_dir / "qdrant"
                path.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(path))
        return self._client

    @property
    def chunks_path(self) -> Path:
        return self.settings.data_dir / "processed" / f"{self.settings.collection_name}.json"

    def load(self):
        if not self.chunks_path.exists():
            raise RuntimeError("Build the knowledge index first in the Library tab.")
        artifact = json.loads(self.chunks_path.read_text())
        if artifact["embedding_model"] != self.settings.embedding_model:
            raise RuntimeError("The embedding model changed; rebuild the index.")
        if not self.client.collection_exists(self.settings.collection_name):
            raise RuntimeError("The Qdrant collection is missing; rebuild the index.")
        if self.client.count(self.settings.collection_name, exact=True).count != len(
            artifact["chunks"]
        ):
            raise RuntimeError("The index is incomplete; rebuild it before searching.")
        self.documents = [Document(**d) for d in artifact["chunks"]]
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.documents])

    def build(self, progress=None) -> int:
        if progress:
            progress(0, desc="Loading documentation and Qwen embedding model…")
        documents = load_chunks()
        if not documents:
            raise RuntimeError("No usable chunks. Download documentation first.")
        name = self.settings.collection_name
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.unlink(missing_ok=True)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dimensions, distance=models.Distance.COSINE
            ),
        )
        for start in range(0, len(documents), 8):
            batch = documents[start : start + 8]
            vectors = self.embeddings.embed_documents([d.page_content for d in batch])
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, d.metadata["chunk_id"])),
                    vector=v,
                    payload={"page_content": d.page_content, "metadata": d.metadata},
                )
                for d, v in zip(batch, vectors)
            ]
            self.client.upsert(name, points=points, wait=True)
            if progress:
                progress(
                    min((start + len(batch)) / len(documents), 1),
                    desc=f"Embedding {start + len(batch)}/{len(documents)} chunks",
                )
        artifact = {
            "embedding_model": self.settings.embedding_model,
            "dimensions": self.settings.embedding_dimensions,
            "chunks": [{"page_content": d.page_content, "metadata": d.metadata} for d in documents],
        }
        temporary = self.chunks_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(artifact))
        temporary.replace(self.chunks_path)
        self.load()
        return len(documents)

    def expand_context(self, ranked: list[Document], limit: int) -> list[Document]:
        """Return distinct, bounded parent sections instead of fragmented lists."""

        def section_key(doc):
            return (
                doc.metadata["file"],
                doc.metadata.get("section", ""),
                doc.metadata.get("subsection", ""),
            )

        contexts, seen = [], set()
        total_chars = 0
        for selected in ranked:
            key = section_key(selected)
            if key in seen:
                continue
            seen.add(key)
            siblings = [d for d in self.documents if section_key(d) == key]
            pivot = next(
                i
                for i, d in enumerate(siblings)
                if d.metadata["chunk_id"] == selected.metadata["chunk_id"]
            )
            chosen = [pivot]
            for index in sorted(range(len(siblings)), key=lambda i: abs(i - pivot)):
                if index in chosen:
                    continue
                proposed = sorted([*chosen, index])
                if len(join_passages([siblings[i].page_content for i in proposed])) <= 2400:
                    chosen = proposed
            text = join_passages([siblings[i].page_content for i in sorted(chosen)])
            if total_chars + len(text) > 6000:
                continue
            contexts.append(
                Document(
                    page_content=text,
                    metadata={
                        **selected.metadata,
                        "expanded_chunk_ids": [
                            siblings[i].metadata["chunk_id"] for i in sorted(chosen)
                        ],
                    },
                )
            )
            total_chars += len(text)
            if len(contexts) >= limit:
                break
        return contexts

    def search(self, query: str, mode: str = "Hybrid", topic: str = "All topics") -> list[Document]:
        if not self.documents:
            self.load()
        rankings = []
        if mode in ("Hybrid", "Dense"):
            hits = self.client.query_points(
                self.settings.collection_name,
                query=self.embeddings.embed_query(query),
                query_filter=None
                if topic == "All topics"
                else models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.topic", match=models.MatchValue(value=topic)
                        )
                    ]
                ),
                limit=self.settings.retrieval_k,
            ).points
            rankings.append(
                [
                    Document(page_content=h.payload["page_content"], metadata=h.payload["metadata"])
                    for h in hits
                ]
            )
        if mode in ("Hybrid", "Keyword"):
            scores = self.bm25.get_scores(tokenize(query))
            indices = sorted(
                (
                    i
                    for i, d in enumerate(self.documents)
                    if topic == "All topics" or d.metadata.get("topic") == topic
                ),
                key=lambda i: scores[i],
                reverse=True,
            )
            rankings.append(
                [self.documents[i] for i in indices[: self.settings.retrieval_k] if scores[i] > 0]
            )
        if not rankings:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        fused = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking, 1):
                key = doc.metadata["chunk_id"]
                if key not in fused:
                    fused[key] = [doc, 0.0]
                fused[key][1] += 1 / (60 + rank)
        output = []
        for doc, score in sorted(fused.values(), key=lambda pair: pair[1], reverse=True):
            output.append(
                Document(
                    page_content=doc.page_content, metadata={**doc.metadata, "fusion_score": score}
                )
            )
        return output[: self.settings.rerank_candidates]
