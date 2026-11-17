"""Text embedders: a local sentence-transformers model, and a deterministic fake for tests."""

from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Runs locally on CPU; no API key. Vectors are L2-normalised (cosine = dot product)."""

    # BGE models were trained with this instruction on the query side only.
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = 32):
        if cache_dir is not None:
            # Keep the model download inside the project folder (data/hf_cache).
            os.environ.setdefault("HF_HOME", str(cache_dir))
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device="cpu")
        self._query_prefix = self.BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 200
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([self._query_prefix + text], normalize_embeddings=True)[0].tolist()


class HashingEmbedder:
    """Fake embedder for tests: bag-of-words hashed into a fixed-size, normalised vector.

    Texts that share words get similar vectors, so retrieval tests are meaningful, and the
    result is identical on every run and machine (md5, not Python's salted hash()).
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
