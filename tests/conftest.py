"""Shared fixtures: one real embedded Qdrant index per test session, seeded exactly like prod.

Embeddings are real (fastembed, CPU) because retrieval behaviour is what we are testing. The LLM
is always FakeLLM, so no model server is needed.
"""

import os

import pytest

from app.clients.embeddings import Embedder
from app.clients.qdrant_store import QdrantStore
from app.config import Principal, Settings
from app.records import load_records
from app.schemas import SourceRecord
from scripts.build_index import V2, seed

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

C1001 = Principal(customer_id="C-1001", role="customer")
C1002 = Principal(customer_id="C-1002", role="customer")
ADMIN = Principal(customer_id=None, role="admin")


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(
        qdrant_path=tmp_path_factory.mktemp("qdrant"),
        qdrant_url=None,
        llm_backend="fake",
        dependency_retries=0,
        _env_file=None,
    )


@pytest.fixture(scope="session")
def embedder(settings) -> Embedder:
    return Embedder(settings)


@pytest.fixture(scope="session")
def store(settings, embedder):
    store = QdrantStore(settings)
    seed(store, embedder, settings)
    yield store
    store.close()


@pytest.fixture
def live_store(store, settings):
    """The seeded store, with the alias restored to kb_v2 after the test."""
    yield store
    store.point_alias(settings.collection_alias, V2)


@pytest.fixture(scope="session")
def records() -> dict[str, SourceRecord]:
    return {r.source_id: r for r in load_records()}
