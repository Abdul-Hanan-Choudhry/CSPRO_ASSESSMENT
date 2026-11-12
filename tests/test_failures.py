"""Dependency failures map to clean JSON errors with no stack traces."""

import httpx
import pytest

from app.clients.llm import FakeLLM, OllamaClient
from app.clients.qdrant_store import QdrantStore
from tests.conftest import auth

QUESTION = {"question": "What is the API rate limit?"}


def _assert_clean(response, status: int, code: str) -> None:
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert "Traceback" not in response.text and 'File "' not in response.text


@pytest.fixture
def unreachable_store(settings):
    # A Qdrant server URL with nothing listening: the realistic "Qdrant is down" case.
    down = settings.model_copy(update={"qdrant_url": "http://127.0.0.1:9", "qdrant_timeout_s": 1})
    store = QdrantStore(down)
    yield store
    store.close()


def test_qdrant_down_returns_503(make_client, unreachable_store):
    client = make_client(store=unreachable_store)
    _assert_clean(
        client.post("/ask", json=QUESTION, headers=auth("C-1001")), 503, "vector_store_unavailable"
    )


def test_health_reports_degraded_when_qdrant_down(make_client, unreachable_store):
    body = make_client(store=unreachable_store).get("/health").json()
    assert body["status"] == "degraded" and body["vector_store"]["ok"] is False


def test_llm_timeout_returns_503(make_client, settings):
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)

    llm = OllamaClient(settings, transport=httpx.MockTransport(slow))
    response = make_client(llm=llm).post("/ask", json=QUESTION, headers=auth("C-1001"))
    _assert_clean(response, 503, "llm_unavailable")


def test_llm_down_returns_503_with_fix_hint(make_client, settings):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    llm = OllamaClient(settings, transport=httpx.MockTransport(refuse))
    response = make_client(llm=llm).post("/ask", json=QUESTION, headers=auth("C-1001"))
    _assert_clean(response, 503, "llm_unavailable")
    assert "ollama serve" in response.json()["error"]["message"]


def test_malformed_llm_output_returns_502(make_client):
    client = make_client(llm=FakeLLM(["definitely not json"]))
    _assert_clean(client.post("/ask", json=QUESTION, headers=auth("C-1001")), 502, "bad_llm_output")


def test_unexpected_exception_returns_generic_500(make_client):
    class Exploding(FakeLLM):
        def complete(self, messages):
            raise RuntimeError("secret internal detail")

    client = make_client(llm=Exploding(), raise_server_exceptions=False)
    response = client.post("/ask", json=QUESTION, headers=auth("C-1001"))
    _assert_clean(response, 500, "internal_error")
    assert "secret internal detail" not in response.text


def test_health_ok_when_dependencies_up(make_client):
    body = make_client().get("/health").json()
    assert body["status"] == "ok" and body["active_collection"] == "kb_v2"
