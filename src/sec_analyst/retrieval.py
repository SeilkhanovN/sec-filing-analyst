"""Semantic search: embed the question, return the top-k most similar chunks."""

from __future__ import annotations

from .config import COMPANIES
from .embeddings import Embedder
from .store import SearchHit, VectorStore


def normalize_ticker(ticker: str | None) -> str | None:
    """Return an upper-case known ticker, None for 'no filter', or raise ValueError."""
    if ticker is None or not str(ticker).strip():
        return None
    value = str(ticker).strip().upper()
    if value not in COMPANIES:
        raise ValueError(f"Unknown ticker {value!r}; indexed tickers: {', '.join(COMPANIES)}")
    return value


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int = 5) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def search(self, query: str, k: int | None = None, ticker: str | None = None) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        vector = self.embedder.embed_query(query)
        return self.store.query(vector, k or self.top_k, normalize_ticker(ticker))
