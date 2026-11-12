"""Runtime configuration, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Principal(BaseModel):
    """Who is calling. customer_id is None only for admins."""

    customer_id: str | None
    role: Literal["customer", "admin"]


# Synthetic demo keys so a fresh clone runs without configuration. They grant access only to
# invented data; real deployments must set API_KEYS.
DEMO_API_KEYS = {
    "demo-c1001": Principal(customer_id="C-1001", role="customer"),
    "demo-c1002": Principal(customer_id="C-1002", role="customer"),
    "demo-admin": Principal(customer_id=None, role="admin"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    qdrant_path: Path = ROOT / ".qdrant"
    qdrant_url: str | None = None
    qdrant_timeout_s: int = 10
    collection_alias: str = "kb_live"

    fusion: Literal["rrf", "legacy_naive"] = "rrf"
    rrf_k: int = 60
    naive_bm25_weight: float = 0.5
    candidates_per_retriever: int = 20

    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    embed_cache_dir: Path = ROOT / ".cache" / "fastembed"

    llm_backend: Literal["ollama", "fake"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_s: float = 180.0
    dependency_retries: int = 2

    api_keys: dict[str, Principal] = DEMO_API_KEYS
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
