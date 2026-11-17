import pytest

from conftest import SAMPLE_CHUNKS
from sec_analyst.embeddings import HashingEmbedder
from sec_analyst.retrieval import Retriever, normalize_ticker
from sec_analyst.store import ChromaStore


def test_most_relevant_chunk_ranks_first(retriever):
    hits = retriever.search("How many people does Microsoft employ?")
    assert hits[0].chunk.chunk_id == "MSFT-2025-item1-000"
    assert hits[0].score >= hits[1].score


def test_ticker_filter_only_returns_that_company(retriever):
    hits = retriever.search("electric vehicles and smartphones", k=4, ticker="aapl")
    assert hits and {h.chunk.ticker for h in hits} == {"AAPL"}


def test_top_k_limits_results(retriever):
    assert len(retriever.search("company", k=3)) == 3
    assert len(retriever.search("company")) == 2  # default top_k from the fixture


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("   ") == []


def test_unknown_ticker_is_rejected():
    assert normalize_ticker(None) is None
    assert normalize_ticker(" msft ") == "MSFT"
    with pytest.raises(ValueError):
        normalize_ticker("ZZZZ")


def test_hashing_embedder_is_deterministic_and_normalised():
    a, b = HashingEmbedder().embed_query("cloud revenue"), HashingEmbedder().embed_query("cloud revenue")
    assert a == b
    assert abs(sum(v * v for v in a) - 1.0) < 1e-9


def test_chroma_store_round_trip(tmp_path):
    """The real vector DB, persisted to a temp folder, with fake embeddings (offline)."""
    embedder = HashingEmbedder()
    store = ChromaStore(tmp_path / "chroma", collection="test")
    store.upsert(SAMPLE_CHUNKS, embedder.embed_documents([c.text for c in SAMPLE_CHUNKS]))

    retriever = Retriever(embedder, store, top_k=2)
    hits = retriever.search("Tesla electric vehicles energy storage")
    assert hits[0].chunk.chunk_id == "TSLA-2025-item1-000"
    assert hits[0].chunk.fiscal_year == 2025 and hits[0].chunk.section == "Item 1. Test Section"

    assert store.count_by_ticker() == {"AAPL": 2, "MSFT": 1, "TSLA": 1}
    assert store.get("MSFT-2025-item1-000").text.startswith("Microsoft Azure")
    assert store.get("does-not-exist") is None
    only_aapl = retriever.search("manufacturing", k=5, ticker="AAPL")
    assert {h.chunk.ticker for h in only_aapl} == {"AAPL"}

    # Re-opening the same folder sees the persisted data; upsert is idempotent.
    reopened = ChromaStore(tmp_path / "chroma", collection="test")
    reopened.upsert(SAMPLE_CHUNKS, embedder.embed_documents([c.text for c in SAMPLE_CHUNKS]))
    assert sum(reopened.count_by_ticker().values()) == len(SAMPLE_CHUNKS)
