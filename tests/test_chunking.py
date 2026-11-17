from sec_analyst.chunking import chunk_section, make_chunk_id, split_text
from sec_analyst.parsing import html_to_text, split_sections


def test_chunk_id_format_is_stable():
    assert make_chunk_id("aapl", 2025, "1A", 7) == "AAPL-2025-item1a-007"
    assert make_chunk_id("MSFT", 2024, "cover", 0) == "MSFT-2024-cover-000"


def test_split_text_respects_max_chars_and_keeps_all_paragraphs():
    paragraphs = [f"Paragraph {i} " + "word " * 40 for i in range(30)]
    chunks = split_text("\n".join(paragraphs), max_chars=600, overlap_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)
    joined = "\n".join(chunks)
    for i in range(30):
        assert f"Paragraph {i} " in joined


def test_split_text_overlaps_consecutive_chunks():
    paragraphs = [f"Sentence number {i} is short." for i in range(60)]
    chunks = split_text("\n".join(paragraphs), max_chars=300, overlap_chars=80)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split("\n")[-1] in nxt  # last unit of a chunk starts the next one


def test_split_text_splits_a_single_huge_paragraph():
    text = "This is one sentence. " * 200  # about 4400 chars, no newlines
    chunks = split_text(text, max_chars=500, overlap_chars=50)
    assert len(chunks) >= 9
    assert all(len(c) <= 500 for c in chunks)


def test_split_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        split_text("abc", max_chars=100, overlap_chars=100)


def test_chunk_section_attaches_metadata():
    chunks = chunk_section(
        "First paragraph.\nSecond paragraph.",
        ticker="tsla", company="Tesla, Inc.", fiscal_year=2025, section_id="7",
        section="Item 7. MD&A", max_chars=1500,
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert (c.chunk_id, c.ticker, c.fiscal_year, c.section_id, c.position) == ("TSLA-2025-item7-000", "TSLA", 2025, "7", 0)
    meta = c.metadata()
    assert "text" not in meta and meta["company"] == "Tesla, Inc."


def test_html_to_text_drops_hidden_xbrl_and_flattens_tables():
    html = """
    <html><body>
      <div style="display:none"><ix:header>SECRET XBRL FACTS</ix:header></div>
      <p>Net sales grew <span>strongly</span> this year.</p>
      <table><tr><td>Net sales</td><td>$</td><td>391,035</td><td></td></tr>
             <tr><td>Margin</td><td>46.2</td><td>%</td></tr></table>
    </body></html>"""
    text = html_to_text(html)
    assert "SECRET" not in text
    assert "Net sales grew strongly this year." in text  # inline tags do not break lines
    assert "Net sales | $391,035" in text
    assert "Margin | 46.2%" in text


def test_split_sections_finds_items_and_skips_table_of_contents():
    body = "x" * 400
    text = "\n".join([
        "Apple Inc. Form 10-K cover page " + "c" * 400,
        "Item 1. | Business | 1",          # table of contents rows: too short to keep
        "Item 1A. | Risk Factors | 5",
        "Item 7. | MD&A | 20",
        "PART I",
        "Item 1. Business",
        "The Company designs smartphones. " + body,
        "Item 1A. Risk Factors",
        "Risks are described below. See Item 7 for more. " + body,
        "Item 7 of this report is referenced here as a cross-reference, not a heading.",
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "Revenue increased. " + body,
    ])
    sections = split_sections(text)
    assert [s.section_id for s in sections] == ["cover", "1", "1A", "7"]
    assert sections[1].title == "Item 1. Business"
    assert sections[2].title == "Item 1A. Risk Factors"
    assert "cross-reference" in sections[2].text  # not treated as a heading
    assert sections[3].text.startswith("Revenue increased.")
