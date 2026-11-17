"""API tests using FastAPI's TestClient."""
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["docs"] == 10
    assert body["chunks"] > 0


def test_ask_returns_citations():
    resp = client.post("/ask", json={"question": "What does the Chevening Scholarship fund?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["citations"], "no citations returned"
    assert all("source_url" in c and c["source_url"] for c in body["citations"])


def test_ask_abstains_on_unrelated_question():
    resp = client.post("/ask", json={"question": "How do I bake sourdough bread at home?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["answer"] == "Not found in the corpus."
    assert body["citations"] == []


def test_ask_validates_short_question():
    resp = client.post("/ask", json={"question": "hi"})
    assert resp.status_code == 422
