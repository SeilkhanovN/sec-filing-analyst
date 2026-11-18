"""Citations: build them from model output, verify quotes, and format them for display."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from .chunking import Chunk

_TRANSLATE = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", " ": " ",
    }
)


@dataclass
class Citation:
    chunk_id: str
    quote: str
    ticker: str = ""
    company: str = ""
    section: str = ""
    fiscal_year: int | None = None
    valid: bool = False  # True only if the quote really appears in the cited chunk
    corrected_from: str | None = None  # set when the model cited the wrong chunk id (see build_citations)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """Make quote matching robust to curly quotes, dashes, case and whitespace."""
    text = text.translate(_TRANSLATE).lower()
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True if the quote appears in text. An ellipsis splits the quote into parts that must
    all appear, in order. Quotes shorter than 8 characters are rejected as too vague."""
    norm_text = normalize(text)
    stripped = normalize(quote).strip(" \"'")
    parts = [p.strip(" \"'") for p in re.split(r"\.\.\.|…", stripped)]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 8:
        return False
    position = 0
    for part in parts:
        found = norm_text.find(part, position)
        if found == -1:
            return False
        position = found + len(part)
    return True


def build_citations(
    raw: Iterable[dict] | None,
    lookup: Callable[[str], Chunk | None],
    candidates: Iterable[Chunk] = (),
) -> list[Citation]:
    """Turn the model's [{"chunk_id", "quote"}] list into verified Citation objects.

    `lookup` maps a chunk id to the stored chunk (or None if the id does not exist, which
    marks the citation invalid). Duplicate (chunk_id, quote) pairs are dropped.

    Citation repair: models sometimes copy a sentence correctly but label it with the id of a
    neighbouring passage. If the quote is not in the cited chunk but appears verbatim in one
    of `candidates` (the passages the model was shown), the citation is re-attributed to that
    chunk and `corrected_from` records the id the model gave. Nothing is invented: a quote
    found in no passage stays invalid.
    """
    candidates = list(candidates)
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id", "")).strip()
        quote = str(item.get("quote", "")).strip()
        if not chunk_id or (chunk_id, quote) in seen:
            continue
        seen.add((chunk_id, quote))
        chunk = lookup(chunk_id)
        valid = chunk is not None and quote_in_text(quote, chunk.text)
        corrected_from = None
        if not valid:
            match = next((c for c in candidates if c.chunk_id != chunk_id and quote_in_text(quote, c.text)), None)
            if match is not None:
                chunk, valid, corrected_from = match, True, chunk_id
        if chunk is None:
            citations.append(Citation(chunk_id=chunk_id, quote=quote, valid=False))
            continue
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                quote=quote,
                ticker=chunk.ticker,
                company=chunk.company,
                section=chunk.section,
                fiscal_year=chunk.fiscal_year,
                valid=valid,
                corrected_from=corrected_from,
            )
        )
    return citations


def format_citation(number: int, c: Citation) -> str:
    """One line for terminal output, e.g. [1] Apple Inc. (AAPL) FY2025, Item 1A ... | id: "quote"."""
    status = "" if c.valid else "  [UNVERIFIED: quote not found in chunk]"
    if c.corrected_from:
        status += f"  [re-attributed: model cited {c.corrected_from}]"
    source = f"{c.company} ({c.ticker}) FY{c.fiscal_year}, {c.section}" if c.company else "unknown source"
    return f'[{number}] {source} | {c.chunk_id}: "{c.quote}"{status}'


_DELIMITER_RE = re.compile(r"<\s*/?\s*context\s*>|\[chunk_id:", re.I)


def neutralize_delimiters(text: str) -> str:
    """Retrieved text is untrusted. Remove anything that looks like our own prompt
    delimiters, so a passage cannot close the <context> block early or fake a new chunk
    header and then pose as instructions outside the data."""
    return _DELIMITER_RE.sub("[removed delimiter]", text)


def format_context(chunks: Iterable[Chunk]) -> str:
    """Render retrieved chunks for the prompt. Each block starts with its chunk id."""
    blocks = [
        f"[chunk_id: {c.chunk_id}] {c.company} ({c.ticker}) | fiscal year {c.fiscal_year} | {c.section}\n"
        f"{neutralize_delimiters(c.text)}"
        for c in chunks
    ]
    return "\n\n".join(blocks)
