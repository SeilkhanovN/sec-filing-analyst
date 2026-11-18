"""API tests with FastAPI's TestClient, a fake LLM and an in-memory index (offline)."""

import json

import pytest
from fastapi.testclient import TestClient

from sec_analyst.api import Services, create_app
from sec_analyst.config import Settings
from sec_analyst.llm import FakeClient, LLMError, LLMResponse, ToolCall

GOOD = {"found": True, "answer": "Microsoft employed about 228,000 people.",
        "citations": [{"chunk_id": "MSFT-2025-item1-000", "quote": "we employed approximately 228,000 people"}]}


def make_client(retriever, tmp_path, script):
    settings = Settings(llm_provider="fake", data_dir=tmp_path, chroma_dir=tmp_path / "db")
    services = Services(settings=settings, retriever=retriever, llm=FakeClient(script))
    return TestClient(create_app(services))


def test_health_and_filings(retriever, tmp_path):
    client = make_client(retriever, tmp_path, [])
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["indexed_chunks"] == 4 and health["model"] == "fake"
    filings = client.get("/filings").json()
    assert {f["ticker"] for f in filings} == {"AAPL", "MSFT", "TSLA"}
    assert next(f for f in filings if f["ticker"] == "AAPL")["chunks"] == 2


def test_ask_rag_returns_answer_with_verified_citation_and_passage(retriever, tmp_path):
    client = make_client(retriever, tmp_path, [LLMResponse(text=json.dumps(GOOD))])
    resp = client.post("/ask", json={"question": "How many employees does Microsoft have?", "mode": "rag", "ticker": "msft"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] and data["grounded"] and data["mode"] == "rag"
    cite = data["citations"][0]
    assert cite["valid"] and cite["company"] == "Microsoft Corporation"
    assert "228,000 people" in cite["passage"]
    assert isinstance(data["latency_ms"], int)


def test_ask_agent_returns_steps(retriever, tmp_path):
    script = [
        LLMResponse(tool_calls=[ToolCall("search_filings", {"query": "Microsoft employees"})]),
        LLMResponse(text=json.dumps(GOOD)),
    ]
    data = make_client(retriever, tmp_path, script).post("/ask", json={"question": "Microsoft headcount?"}).json()
    assert data["mode"] == "agent" and data["grounded"]
    assert data["steps"][0]["tool"] == "search_filings" and data["steps"][-1]["type"] == "answer"


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "  a "},
    {"question": "x" * 501},
    {"question": "Valid question?", "ticker": "ZZZZ"},
    {"question": "Valid question?", "mode": "magic"},
    {},
])
def test_invalid_input_is_rejected_with_422(retriever, tmp_path, payload):
    resp = make_client(retriever, tmp_path, []).post("/ask", json=payload)
    assert resp.status_code == 422
    assert "detail" in resp.json()


@pytest.mark.parametrize("mode", ["rag", "agent"])
def test_llm_outage_returns_503_with_clear_message(retriever, tmp_path, mode):
    def failing(*args):
        raise LLMError("429 RESOURCE_EXHAUSTED secret-details")

    resp = make_client(retriever, tmp_path, failing).post("/ask", json={"question": "Anything?", "mode": mode})
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"] and "secret-details" not in resp.json()["detail"]


def test_unexpected_error_returns_500_without_internals(retriever, tmp_path):
    def broken(*args):
        raise RuntimeError("internal stack details")

    client = TestClient(create_app(Services(Settings(llm_provider="fake", data_dir=tmp_path), retriever, FakeClient(broken))),
                        raise_server_exceptions=False)
    resp = client.post("/ask", json={"question": "Anything?", "mode": "rag"})
    assert resp.status_code == 500 and "internal stack" not in resp.text


def test_static_page_is_served(retriever, tmp_path):
    client = make_client(retriever, tmp_path, [])
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert "SEC Filing Analyst" in page.text and 'aria-live="polite"' in page.text
    assert "<script src=\"http" not in page.text  # no external scripts or CDNs
    assert client.get("/static/app.js").status_code == 200
    assert "--accent" in client.get("/static/styles.css").text
