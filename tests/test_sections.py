"""Added in milestone 3 after inspecting real filings (see STATUS.md)."""

from sec_analyst.parsing import split_sections


def test_front_matter_after_table_of_contents_is_not_assigned_to_the_last_toc_item():
    body = "y" * 400
    text = "\n".join([
        "Cover " + "c" * 400,
        "Item 1. | Business | 1",
        "Item 16. | Form 10-K Summary | 62",
        "This report contains forward-looking statements. " + body,  # front matter, not Item 16
        "Item 1. Business",
        "The Company designs smartphones. " + body,
        "Item 16. Form 10-K Summary",
        "None. Signatures follow. " + body,
    ])
    sections = {s.section_id: s.text for s in split_sections(text)}
    assert "forward-looking statements" in sections["cover"]
    assert "forward-looking" not in sections["16"]
    assert sections["16"].startswith("None. Signatures follow.")
    assert list(sections) == ["cover", "1", "16"]
