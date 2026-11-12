"""Input validation: bad, empty and oversized payloads are rejected with clean 422 JSON."""

import pytest

from app.schemas import MAX_QUESTION_CHARS, MAX_TICKET_BODY_CHARS
from tests.conftest import auth


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": "valid question", "top_k": 0},
        {"question": "valid question", "top_k": 11},
        {"question": "valid question", "unexpected": True},
        {"question": 123},
    ],
)
def test_ask_rejects_invalid_payloads(client, payload):
    response = client.post("/ask", json=payload, headers=auth("C-1001"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    "payload",
    [
        {"subject": "ok subject"},
        {"subject": "", "body": "body text"},
        {"subject": "ok subject", "body": "x" * (MAX_TICKET_BODY_CHARS + 1)},
        {"subject": "ok subject", "body": "body text", "customer_id": "not-an-id"},
    ],
)
def test_ticket_rejects_invalid_payloads(client, payload):
    response = client.post("/tickets", json=payload, headers=auth("C-1001"))
    assert response.status_code == 422


def test_malformed_json_body_rejected(client):
    response = client.post(
        "/ask",
        content=b"{not json",
        headers={**auth("C-1001"), "Content-Type": "application/json"},
    )
    assert response.status_code == 422


def test_invalid_ticket_id_path_rejected(client):
    assert client.get("/tickets/../../etc", headers=auth("C-1001")).status_code in (404, 422)
    assert client.get("/tickets/t-lower", headers=auth("C-1001")).status_code == 422


def test_validation_error_does_not_echo_input(client):
    secret_like = "x" * (MAX_QUESTION_CHARS + 1)
    response = client.post("/ask", json={"question": secret_like}, headers=auth("C-1001"))
    assert secret_like not in response.text


def test_error_body_shape_and_request_id(client):
    response = client.post("/ask", json={}, headers={**auth("C-1001"), "X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
    assert response.json()["error"]["request_id"] == "abc-123"


def test_admin_ticket_requires_customer_id(client):
    response = client.post(
        "/tickets", json={"subject": "s s", "body": "b b"}, headers=auth("admin")
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
