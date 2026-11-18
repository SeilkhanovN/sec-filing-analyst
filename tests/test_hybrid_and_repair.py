"""Added in the improvement phase: hybrid retrieval (BM25 + RRF) and citation repair."""

from conftest import SAMPLE_CHUNKS, make_chunk
from sec_analyst.citations import build_citations
from sec_analyst.keyword import BM25Index, reciprocal_rank_fusion, tokenize
from sec_analyst.retrieval import Retriever
from sec_analyst.store import InMemoryStore


class ConstantEmbedder:
    """Every text gets the same vector, so vector search alone cannot rank anything."""

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("How many full-time employees did Apple have?") == ["full", "time", "employees", "apple"]


def test_bm25_ranks_exact_terms_first_and_respects_ticker():
    index = BM25Index(SAMPLE_CHUNKS)
    hits = index.search("outsourcing partners in Asia", k=3)
    assert hits[0][0].chunk_id == "AAPL-2025-item1a-000"
    assert index.search("Azure cloud", k=3, ticker="AAPL") == []
    assert index.search("the of and", k=3) == []  # stopwords only


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    assert [item for item, _ in fused][:2] == ["b", "c"]  # in both lists
    assert fused[0][1] == 1 / 62 + 1 / 61


def test_hybrid_retriever_finds_keyword_match_that_vectors_miss():
    store = InMemoryStore()
    store.upsert(SAMPLE_CHUNKS, ConstantEmbedder().embed_documents([c.text for c in SAMPLE_CHUNKS]))
    vector_only = Retriever(ConstantEmbedder(), store, top_k=1)
    hybrid = Retriever(ConstantEmbedder(), store, top_k=1, keyword_index=BM25Index(store.all_chunks()))
    query = "energy generation and storage systems"
    assert vector_only.search(query)[0].chunk.chunk_id == "AAPL-2025-item1-000"  # arbitrary tie order
    assert hybrid.search(query)[0].chunk.chunk_id == "TSLA-2025-item1-000"
    assert hybrid.mode == "hybrid" and vector_only.mode == "vector"
    assert {h.chunk.ticker for h in hybrid.search(query, k=3, ticker="MSFT")} == {"MSFT"}


def test_citation_repair_reattributes_a_real_quote_to_the_right_passage():
    a = make_chunk("TSLA-2025-item8-013", "Total revenues | 94,827 | 97,690", "TSLA", "8")
    b = make_chunk("TSLA-2025-item8-024", "Total revenues | $94,827 | $97,690", "TSLA", "8")
    lookup = {a.chunk_id: a, b.chunk_id: b}.get
    raw = [{"chunk_id": a.chunk_id, "quote": "Total revenues | $94,827 | $97,690"}]

    without_repair = build_citations(raw, lookup)
    assert not without_repair[0].valid

    repaired = build_citations(raw, lookup, candidates=[a, b])[0]
    assert repaired.valid and repaired.chunk_id == b.chunk_id and repaired.corrected_from == a.chunk_id


def test_citation_repair_never_validates_an_invented_quote():
    raw = [{"chunk_id": "AAPL-2025-item1-000", "quote": "Apple sells electric trucks in Mars"}]
    cites = build_citations(raw, {c.chunk_id: c for c in SAMPLE_CHUNKS}.get, candidates=SAMPLE_CHUNKS)
    assert not cites[0].valid and cites[0].corrected_from is None
