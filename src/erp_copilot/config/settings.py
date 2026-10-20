"""Central configuration; generated files stay inside the project."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimensions: int = Field(default=512, ge=256, le=512)
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    llm_repository: str = "unsloth/Qwen3-1.7B-GGUF"
    llm_filename: str = "Qwen3-1.7B-Q4_K_M.gguf"
    llm_base_url: str = "http://127.0.0.1:8081"
    llm_port: int = 8081
    llm_context_size: int = 4096
    llm_max_tokens: int = 384
    app_port: int = 7860
    device: str = "auto"
    retrieval_k: int = Field(default=16, ge=1)
    rerank_candidates: int = Field(default=8, ge=1)
    context_chunks: int = Field(default=4, ge=1)
    document_limit: int = Field(default=45, ge=1)
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    @property
    def model_cache_dir(self) -> Path:
        return ROOT / ".cache/huggingface/hub"

    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def collection_name(self) -> str:
        return f"finance_ap_qwen_{self.embedding_dimensions}"

    def torch_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.embedding_dimensions not in (256, 512):
        raise ValueError("EMBEDDING_DIMENSIONS must be 256 or 512.")
    return settings
