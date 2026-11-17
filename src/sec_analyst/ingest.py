"""Ingestion pipeline: EDGAR HTML -> text -> sections -> chunks -> embeddings -> Chroma."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .chunking import Chunk, chunk_section
from .edgar import EdgarClient, FilingMeta
from .embeddings import Embedder
from .parsing import html_to_text, split_sections


def chunks_from_html(html: str, meta: FilingMeta) -> list[Chunk]:
    text = html_to_text(html)
    chunks: list[Chunk] = []
    for section in split_sections(text):
        chunks.extend(
            chunk_section(
                section.text,
                ticker=meta.ticker,
                company=meta.company,
                fiscal_year=meta.fiscal_year,
                section_id=section.section_id,
                section=section.title,
            )
        )
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Keep a readable JSONL copy of the chunks (used to write eval questions and to debug)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def ingest(tickers: list[str], edgar: EdgarClient, embedder: Embedder, store, data_dir: Path, log=print) -> dict[str, int]:
    """Download (cached), parse, chunk, embed and upsert each company's latest 10-K."""
    counts: dict[str, int] = {}
    for ticker in tickers:
        meta, html_path = edgar.download_latest_10k(ticker)
        log(f"{ticker}: {meta.form} for fiscal year {meta.fiscal_year} (filed {meta.filing_date}, {meta.accession_number})")
        chunks = chunks_from_html(html_path.read_text(encoding="utf-8", errors="replace"), meta)
        sections = sorted({c.section_id for c in chunks}, key=lambda s: (s != "cover", s))
        log(f"{ticker}: {len(chunks)} chunks across sections {', '.join(sections)}")
        save_chunks(chunks, data_dir / "processed" / f"{ticker}_chunks.jsonl")
        if hasattr(store, "delete_ticker"):
            store.delete_ticker(ticker)  # drop chunks of an older filing before re-indexing
        embeddings = embedder.embed_documents([c.text for c in chunks])
        store.upsert(chunks, embeddings)
        counts[ticker] = len(chunks)
    return counts
