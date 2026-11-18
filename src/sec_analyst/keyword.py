"""Keyword search (BM25) and reciprocal rank fusion, for hybrid retrieval.

Embeddings capture meaning but can rank an exact phrase such as "full-time equivalent
employees" or a street address below thematically similar passages. BM25 scores exact
term overlap, weighting rare terms higher and normalising for chunk length. Reciprocal
rank fusion (RRF) merges the two ranked lists using only ranks, so the very different
score scales of cosine similarity and BM25 never have to be compared.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .chunking import Chunk

STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its many much of on or "
    "s that the their this to was were what when where which who why with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    """Okapi BM25 over a fixed list of chunks (k1=1.5, b=0.75, the usual defaults)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(self._len)) if chunks else 0.0
        df = Counter(term for tf in self._tf for term in tf)
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, k: int, ticker: str | None = None) -> list[tuple[Chunk, float]]:
        terms = [t for t in set(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        scored = []
        for i, chunk in enumerate(self.chunks):
            if ticker and chunk.ticker != ticker:
                continue
            tf, length = self._tf[i], self._len[i]
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
                    score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return scored[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked id lists: score(id) = sum over lists of 1 / (k + rank). k=60 is the
    constant from the original RRF paper; it stops the top rank of one list dominating."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
