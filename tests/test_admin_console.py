"""Admin retrieval endpoint and the test console route."""

from tests.conftest import auth

CSV_Q = "What is the CSV export row limit?"


def test_admin_retrieve_requires_admin(make_client):
    client = make_client()
    body = {"question": CSV_Q}
    assert client.post("/admin/retrieve", json=body).status_code == 401
    assert client.post("/admin/retrieve", json=body, headers=auth("C-1001")).status_code == 403


def test_admin_retrieve_shows_fusion_bug_and_fix(make_client):
    client = make_client()

    def top(fusion: str) -> str:
        body = {"question": CSV_Q, "as_customer": "C-1001", "fusion": fusion}
        response = client.post("/admin/retrieve", json=body, headers=auth("admin"))
        assert response.status_code == 200
        return response.json()["chunks"][0]["source_id"]

    assert (top("rrf"), top("legacy_naive")) == ("KB-0003", "KB-000B")


def test_admin_retrieve_as_customer_applies_tenant_filter(make_client):
    body = {"question": "invoice charged twice", "as_customer": "C-1002", "top_k": 10}
    chunks = (
        make_client().post("/admin/retrieve", json=body, headers=auth("admin")).json()["chunks"]
    )
    assert not any(c["source_id"] in {"T-0002", "LOG-0002"} for c in chunks)


def test_console_is_served(make_client):
    response = make_client().get("/ui/")
    assert response.status_code == 200 and "Support RAG Console" in response.text
