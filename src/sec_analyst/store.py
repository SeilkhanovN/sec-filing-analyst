"""Vector stores: Chroma (persistent, used for real runs) and an in-memory one for tests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .chunking import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...

    def query(self, embedding: list[float], k: int, ticker: str | None = None) -> list[SearchHit]: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def count_by_ticker(self) -> dict[str, int]: ...


def _chunk_from(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        ticker=str(meta["ticker"]),
        company=str(meta["company"]),
        fiscal_year=int(meta["fiscal_year"]),
        section_id=str(meta["section_id"]),
        section=str(meta["section"]),
        position=int(meta["position"]),
    )


class InMemoryStore:
    """Brute-force cosine search over a dict. Fine for tests and a few thousand chunks."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks, embeddings) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for chunk, emb in zip(chunks, embeddings):
            self._items[chunk.chunk_id] = (chunk, emb)

    def query(self, embedding, k, ticker=None) -> list[SearchHit]:
        hits = [
            SearchHit(chunk, sum(a * b for a, b in zip(embedding, emb)))
            for chunk, emb in self._items.values()
            if ticker is None or chunk.ticker == ticker
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:k]

    def get(self, chunk_id) -> Chunk | None:
        item = self._items.get(chunk_id)
        return item[0] if item else None

    def count_by_ticker(self) -> dict[str, int]:
        return dict(Counter(chunk.ticker for chunk, _ in self._items.values()))

    def all_chunks(self) -> list[Chunk]:
        return sorted((chunk for chunk, _ in self._items.values()), key=lambda c: c.chunk_id)


class ChromaStore:
    """Persistent Chroma collection using cosine distance. Embeddings are always supplied by
    our own embedder, so Chroma never downloads or runs its default embedding model."""

    BATCH = 1000

    def __init__(self, path: Path, collection: str = "sec_filings") -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    def upsert(self, chunks, embeddings) -> None:
        for start in range(0, len(chunks), self.BATCH):
            batch = chunks[start : start + self.BATCH]
            self._col.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata() for c in batch],
                embeddings=embeddings[start : start + self.BATCH],
            )

    def query(self, embedding, k, ticker=None) -> list[SearchHit]:
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"ticker": ticker} if ticker else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        return [
            SearchHit(_chunk_from(i, doc, meta), 1.0 - dist)
            for i, doc, meta, dist in zip(
                ids, result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def get(self, chunk_id) -> Chunk | None:
        result = self._col.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return _chunk_from(result["ids"][0], result["documents"][0], result["metadatas"][0])

    def count_by_ticker(self) -> dict[str, int]:
        result = self._col.get(include=["metadatas"])
        return dict(Counter(str(m["ticker"]) for m in result["metadatas"]))

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk (used to build the in-memory BM25 keyword index)."""
        result = self._col.get(include=["documents", "metadatas"])
        chunks = [_chunk_from(i, d, m) for i, d, m in zip(result["ids"], result["documents"], result["metadatas"])]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def delete_ticker(self, ticker: str) -> None:
        self._col.delete(where={"ticker": ticker})
