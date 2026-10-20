"""Heading-aware chunks with immutable provenance and stable IDs."""

import hashlib
import json
import re

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from erp_copilot.config.settings import get_settings


def load_chunks() -> list[Document]:
    settings = get_settings()
    manifest = json.loads((settings.data_dir / "source-manifest.json").read_text())
    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1100, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for entry in manifest["documents"]:
        text = (settings.data_dir / "raw" / entry["file"]).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['file']}; download the corpus again.")
        frontmatter = {}
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                text = parts[2]
        text = re.sub(r":::image[^\n]*", "", text)
        text = re.sub(r"\[!INCLUDE[^\n]*", "", text)
        for doc in splitter.split_documents(headers.split_text(text)):
            if len(doc.page_content.strip()) < 70:
                continue
            doc.metadata = {
                **entry,
                **doc.metadata,
                "commit": manifest["commit"],
                "title": str(
                    frontmatter.get("title") or doc.metadata.get("title") or entry["file"]
                ),
                "updated": str(frontmatter.get("ms.date", "")),
            }
            doc.metadata["chunk_id"] = hashlib.sha256(
                (entry["repository_path"] + doc.page_content).encode()
            ).hexdigest()
            chunks.append(doc)
    # Overlapping sections can produce identical content. Preserve one stable point per ID.
    return list({d.metadata["chunk_id"]: d for d in chunks}.values())
