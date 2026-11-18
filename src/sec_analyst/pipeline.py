"""Wiring: build the real components from settings. Used by the CLI, the API and the eval."""

from __future__ import annotations

from .agent import Agent
from .config import Settings
from .embeddings import SentenceTransformerEmbedder
from .keyword import BM25Index
from .llm import LLMClient
from .retrieval import Retriever
from .store import ChromaStore
from .tools import make_screener_tool, make_search_tool


def build_store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings.chroma_dir)


def build_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.embedding_model, cache_dir=settings.data_dir / "hf_cache")


def build_retriever(settings: Settings, store: ChromaStore | None = None) -> Retriever:
    """RETRIEVAL_MODE=hybrid (default): vectors + BM25 fused with RRF; =vector: vectors only."""
    store = store or build_store(settings)
    keyword_index = BM25Index(store.all_chunks()) if settings.retrieval_mode == "hybrid" else None
    return Retriever(build_embedder(settings), store, top_k=settings.top_k, keyword_index=keyword_index)


def build_agent(settings: Settings, retriever: Retriever, llm: LLMClient, max_steps: int = 6) -> Agent:
    tools = [
        make_search_tool(retriever, k=settings.top_k),
        make_screener_tool(settings.screener_url, settings.screener_path),
    ]
    return Agent(llm, tools, chunk_lookup=retriever.store.get, max_steps=max_steps)
