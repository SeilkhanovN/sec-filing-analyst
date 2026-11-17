"""Shared fixtures. Everything here is offline: fake embedder, in-memory store, fake LLM."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sec_analyst.chunking import Chunk
from sec_analyst.embeddings import HashingEmbedder
from sec_analyst.retrieval import Retriever
from sec_analyst.store import InMemoryStore


def make_chunk(chunk_id: str, text: str, ticker: str = "AAPL", section_id: str = "1A") -> Chunk:
    names = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "TSLA": "Tesla, Inc."}
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        ticker=ticker,
        company=names.get(ticker, ticker),
        fiscal_year=2025,
        section_id=section_id,
        section=f"Item {section_id}. Test Section",
        position=0,
    )


SAMPLE_CHUNKS = [
    make_chunk("AAPL-2025-item1-000", "Apple designs smartphones, personal computers, tablets and wearables. "
               "The iPhone is the Company's line of smartphones.", "AAPL", "1"),
    make_chunk("AAPL-2025-item1a-000", "The Company depends on component suppliers and outsourcing partners "
               "located primarily in Asia for manufacturing.", "AAPL", "1A"),
    make_chunk("MSFT-2025-item1-000", "Microsoft Azure is a comprehensive set of cloud computing services. "
               "As of June 30, 2025, we employed approximately 228,000 people.", "MSFT", "1"),
    make_chunk("TSLA-2025-item1-000", "Tesla designs, develops, manufactures and sells electric vehicles "
               "and energy generation and storage systems.", "TSLA", "1"),
]


@pytest.fixture
def store() -> InMemoryStore:
    s = InMemoryStore()
    embedder = HashingEmbedder()
    s.upsert(SAMPLE_CHUNKS, embedder.embed_documents([c.text for c in SAMPLE_CHUNKS]))
    return s


@pytest.fixture
def retriever(store) -> Retriever:
    return Retriever(HashingEmbedder(), store, top_k=2)


class _ScreenerHandler(BaseHTTPRequestHandler):
    """Tiny fake of the Stock Screener API: /api/screener/<TICKER> returns JSON."""

    def do_GET(self):  # noqa: N802 - name required by BaseHTTPRequestHandler
        if self.path == "/api/screener/AAPL":
            body = json.dumps({"ticker": "AAPL", "pe_ratio": 31.2, "signal": "hold"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path == "/api/screener/BAD":
            body = b"<html>not json</html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
        else:
            body = json.dumps({"detail": "not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def screener_server():
    """Start the fake screener on 127.0.0.1 with a random free port; yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScreenerHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
