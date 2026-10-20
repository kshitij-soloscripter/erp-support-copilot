"""Instruction-aware Qwen embeddings with Matryoshka dimensionality reduction."""

from langchain_core.embeddings import Embeddings

from erp_copilot.config.settings import get_settings


class QwenEmbeddings(Embeddings):
    def __init__(self):
        self.settings = get_settings()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.torch_device(),
                truncate_dim=self.settings.embedding_dimensions,
                model_kwargs={"torch_dtype": torch.float16},
            )
            self._model.max_seq_length = 512
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        query = (
            "Instruct: Retrieve Dynamics 365 Finance documentation that answers the ERP support question.\nQuery: "
            + text
        )
        return self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[
            0
        ].tolist()
