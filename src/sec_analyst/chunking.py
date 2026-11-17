"""Split section text into overlapping chunks that carry citation metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

DEFAULT_MAX_CHARS = 1500  # about 350 tokens: well inside the embedding model's 512-token limit
DEFAULT_OVERLAP_CHARS = 200


@dataclass
class Chunk:
    chunk_id: str  # e.g. "AAPL-2025-item1a-007": stable across re-ingestion
    text: str
    ticker: str
    company: str
    fiscal_year: int
    section_id: str  # "1A"
    section: str  # "Item 1A. Risk Factors"
    position: int  # index of the chunk within its section

    def metadata(self) -> dict:
        data = asdict(self)
        data.pop("text")
        return data


def make_chunk_id(ticker: str, fiscal_year: int, section_id: str, position: int) -> str:
    slug = "cover" if section_id == "cover" else f"item{section_id.lower()}"
    return f"{ticker.upper()}-{fiscal_year}-{slug}-{position:03d}"


def _split_long(unit: str, max_chars: int) -> list[str]:
    """Split one paragraph that is too long: first at sentence ends, then hard-split."""
    if len(unit) <= max_chars:
        return [unit]
    pieces: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", unit):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
    return pieces


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap_chars: int = DEFAULT_OVERLAP_CHARS) -> list[str]:
    """Greedy paragraph packing.

    Paragraphs (lines) are added to the current chunk until the next one would exceed
    max_chars. The next chunk starts with the last few units of the previous chunk (up to
    overlap_chars) so a fact that straddles a boundary is still retrievable.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    units: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if paragraph:
            units.extend(_split_long(paragraph, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and size + added > max_chars:
            chunks.append("\n".join(current))
            # carry over trailing units as overlap
            carry: list[str] = []
            carry_size = 0
            for prev in reversed(current):
                if carry_size + len(prev) + 1 > overlap_chars:
                    break
                carry.insert(0, prev)
                carry_size += len(prev) + 1
            if carry_size + len(unit) + 1 > max_chars:
                carry, carry_size = [], 0
            current, size = carry, carry_size
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        size += added
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_section(
    text: str,
    *,
    ticker: str,
    company: str,
    fiscal_year: int,
    section_id: str,
    section: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=make_chunk_id(ticker, fiscal_year, section_id, i),
            text=piece,
            ticker=ticker.upper(),
            company=company,
            fiscal_year=fiscal_year,
            section_id=section_id,
            section=section,
            position=i,
        )
        for i, piece in enumerate(split_text(text, max_chars, overlap_chars))
    ]
