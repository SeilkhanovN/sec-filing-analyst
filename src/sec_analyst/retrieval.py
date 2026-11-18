"""Search: embed the question and return the top-k chunks (optionally hybrid with BM25)."""

from __future__ import annotations

from .config import COMPANIES
from .embeddings import Embedder
from .keyword import BM25Index, reciprocal_rank_fusion
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
    """Vector search, or hybrid search when a BM25 keyword index is given.

    Hybrid: take the top `candidates` from each method, fuse the two rankings with
    reciprocal rank fusion, return the top k. The returned score is the RRF score.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int = 5,
        keyword_index: BM25Index | None = None,
        candidates: int = 20,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.keyword_index = keyword_index
        self.candidates = candidates

    @property
    def mode(self) -> str:
        return "hybrid" if self.keyword_index is not None else "vector"

    def search(self, query: str, k: int | None = None, ticker: str | None = None) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []
        k = k or self.top_k
        ticker = normalize_ticker(ticker)
        vector = self.embedder.embed_query(query)
        if self.keyword_index is None:
            return self.store.query(vector, k, ticker)

        n = max(k, self.candidates)
        vector_hits = self.store.query(vector, n, ticker)
        keyword_hits = self.keyword_index.search(query, n, ticker)
        by_id = {h.chunk.chunk_id: h.chunk for h in vector_hits}
        by_id.update({c.chunk_id: c for c, _ in keyword_hits})
        fused = reciprocal_rank_fusion(
            [[h.chunk.chunk_id for h in vector_hits], [c.chunk_id for c, _ in keyword_hits]]
        )
        return [SearchHit(by_id[chunk_id], score) for chunk_id, score in fused[:k]]
