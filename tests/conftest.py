"""Shared fixtures: one real embedded Qdrant index per test session, seeded exactly like prod.

Embeddings are real (fastembed, CPU) because retrieval behaviour is what we are testing. The LLM
is always FakeLLM, so no model server is needed.
"""

import os

import pytest
from fastapi.testclient import TestClient

from app.clients.embeddings import Embedder
from app.clients.llm import FakeLLM
from app.clients.qdrant_store import QdrantStore
from app.config import Principal, Settings
from app.container import build_services
from app.main import create_app
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


KEYS = {"C-1001": "demo-c1001", "C-1002": "demo-c1002", "admin": "demo-admin"}


def auth(who: str) -> dict[str, str]:
    return {"X-API-Key": KEYS[who]}


@pytest.fixture
def make_client(live_store, embedder, settings):
    """Build a TestClient around the real app; pass an LLM and/or a different store."""

    def make(llm=None, store=None, raise_server_exceptions: bool = True) -> TestClient:
        services = build_services(
            settings, store=store or live_store, embedder=embedder, llm=llm or FakeLLM()
        )
        client = TestClient(
            create_app(settings, services), raise_server_exceptions=raise_server_exceptions
        )
        clients.append(client)
        return client.__enter__()

    clients: list[TestClient] = []
    yield make
    for client in clients:
        client.__exit__(None, None, None)
