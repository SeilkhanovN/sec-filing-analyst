import json

from conftest import SAMPLE_CHUNKS
from sec_analyst.citations import build_citations, format_citation, format_context, quote_in_text
from sec_analyst.llm import FakeClient, LLMResponse
from sec_analyst.rag import NOT_FOUND_ANSWER, answer_from_model_output, answer_question

BY_ID = {c.chunk_id: c for c in SAMPLE_CHUNKS}


def test_quote_matching_is_robust_to_formatting():
    text = "The Company’s  products   include the iPhone — its smartphone line."
    assert quote_in_text("the company's products include the iPhone", text)
    assert quote_in_text('"products include the iPhone - its smartphone"', text)
    assert quote_in_text("The Company's products ... smartphone line", text)  # ellipsis, in order


def test_quote_matching_rejects_wrong_or_vague_quotes():
    text = "The Company's products include the iPhone."
    assert not quote_in_text("products include the iPad", text)
    assert not quote_in_text("smartphone line ... The Company's", text)  # parts out of order
    assert not quote_in_text("the", text)  # too short to verify anything
    assert not quote_in_text("", text)


def test_build_citations_verifies_and_enriches():
    raw = [
        {"chunk_id": "MSFT-2025-item1-000", "quote": "we employed approximately 228,000 people"},
        {"chunk_id": "MSFT-2025-item1-000", "quote": "we employed approximately 228,000 people"},  # duplicate
        {"chunk_id": "AAPL-2025-item1-000", "quote": "Apple sells cars"},  # not in the chunk
        {"chunk_id": "NOPE-2025-item1-000", "quote": "anything at all here"},  # unknown id
        "not a dict",
    ]
    cites = build_citations(raw, BY_ID.get)
    assert [c.chunk_id for c in cites] == ["MSFT-2025-item1-000", "AAPL-2025-item1-000", "NOPE-2025-item1-000"]
    assert [c.valid for c in cites] == [True, False, False]
    assert cites[0].company == "Microsoft Corporation" and cites[0].section == "Item 1. Test Section"
    assert cites[0].fiscal_year == 2025


def test_format_citation_shows_source_and_flags_unverified():
    good, bad = build_citations(
        [{"chunk_id": "MSFT-2025-item1-000", "quote": "Microsoft Azure is a comprehensive set"},
         {"chunk_id": "MSFT-2025-item1-000", "quote": "made up words here"}],
        BY_ID.get,
    )
    line = format_citation(1, good)
    assert line.startswith("[1] Microsoft Corporation (MSFT) FY2025, Item 1. Test Section | MSFT-2025-item1-000")
    assert "UNVERIFIED" not in line
    assert "UNVERIFIED" in format_citation(2, bad)


def test_format_context_labels_every_chunk_with_its_id():
    context = format_context(SAMPLE_CHUNKS[:2])
    assert "[chunk_id: AAPL-2025-item1-000]" in context and "[chunk_id: AAPL-2025-item1a-000]" in context


def test_answer_from_model_output_handles_found_not_found_and_garbage():
    found = answer_from_model_output("q", json.dumps({
        "found": True, "answer": "About 228,000 people.",
        "citations": [{"chunk_id": "MSFT-2025-item1-000", "quote": "approximately 228,000 people"}],
    }), BY_ID.get)
    assert found.found and found.grounded and found.answer == "About 228,000 people."

    fenced = answer_from_model_output("q", '```json\n{"found": false, "answer": "", "citations": []}\n```', BY_ID.get)
    assert not fenced.found and fenced.answer == NOT_FOUND_ANSWER

    garbage = answer_from_model_output("q", "I think the answer is 42", BY_ID.get)
    assert not garbage.found and garbage.error


def test_answer_question_sends_context_and_verifies_citations(retriever):
    reply = {"found": True, "answer": "Microsoft employed about 228,000 people.",
             "citations": [{"chunk_id": "MSFT-2025-item1-000", "quote": "we employed approximately 228,000 people"}]}
    llm = FakeClient([LLMResponse(text=json.dumps(reply))])
    answer = answer_question("How many employees does Microsoft have?", retriever, llm)

    sent = llm.calls[0]
    assert sent["json_output"] is True
    assert "[chunk_id: MSFT-2025-item1-000]" in sent["messages"][0].text
    assert "ONLY" in sent["system"]  # grounding instruction is present
    assert answer.found and answer.grounded
    assert answer.retrieved_ids[0] == "MSFT-2025-item1-000"


def test_answer_question_rejects_citation_to_chunk_that_was_not_retrieved(retriever):
    reply = {"found": True, "answer": "Tesla makes EVs.",
             "citations": [{"chunk_id": "TSLA-2025-item1-000", "quote": "manufactures and sells electric vehicles"}]}
    llm = FakeClient([LLMResponse(text=json.dumps(reply))])
    answer = answer_question("How many employees does Microsoft have?", retriever, llm, ticker="MSFT")
    assert answer.citations and not answer.citations[0].valid
    assert not answer.grounded


def test_unanswerable_question_returns_not_found(retriever):
    llm = FakeClient([LLMResponse(text='{"found": false, "answer": "", "citations": []}')])
    answer = answer_question("What is the CEO's favourite colour?", retriever, llm)
    assert not answer.found and answer.answer.startswith("Not found in the filings")
    assert answer.citations == []
