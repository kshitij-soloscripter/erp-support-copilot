"""Qwen yes/no relevance scoring using its documented reranking template."""

from langchain_core.documents import Document

from erp_copilot.config.settings import get_settings


class QwenReranker:
    def __init__(self):
        self.settings = get_settings()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.settings.torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.reranker_model,
            cache_dir=str(self.settings.model_cache_dir),
            padding_side="left",
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.settings.reranker_model,
                cache_dir=str(self.settings.model_cache_dir),
                torch_dtype=torch.float16,
            )
            .to(device)
            .eval()
        )
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        import torch

        self.load()
        scored = []
        with torch.inference_mode():
            for doc in documents:
                body = f"<Instruct>: Retrieve ERP documentation answering the question.\n<Query>: {query}\n<Document>: {doc.page_content}"
                ids = self.tokenizer.encode(
                    body,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=768 - len(self.prefix_ids) - len(self.suffix_ids),
                )
                tokens = torch.tensor(
                    [self.prefix_ids + ids + self.suffix_ids], device=self.model.device
                )
                logits = (
                    self.model(input_ids=tokens, use_cache=False, logits_to_keep=1)
                    .logits[0, -1, [self.no_id, self.yes_id]]
                    .float()
                )
                score = torch.softmax(logits, dim=0)[1].item()
                scored.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, "rerank_score": score},
                    )
                )
        return sorted(scored, key=lambda d: d.metadata["rerank_score"], reverse=True)

    def select_units(self, query: str, units: list[dict], limit: int = 6) -> list[dict]:
        """Compress parent sections to a small set of question-relevant evidence units."""
        from rank_bm25 import BM25Okapi

        from erp_copilot.retrieval.store import tokenize

        if not units:
            return []
        texts = [f"{u.get('heading', '')}\n{u['text']}" for u in units]
        scores = BM25Okapi([tokenize(t) for t in texts]).get_scores(tokenize(query))
        indices = sorted(range(len(units)), key=lambda i: scores[i], reverse=True)[:12]
        candidates = [Document(page_content=texts[i], metadata={"unit": units[i]}) for i in indices]
        ranked = self.rerank(query, candidates)
        return [{**d.metadata["unit"], "id": i + 1} for i, d in enumerate(ranked[:limit])]
