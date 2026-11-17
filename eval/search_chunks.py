"""Helper used to write eval questions: find chunks whose text matches a regex.

    python eval/search_chunks.py AAPL "full-time equivalent employees" "Total net sales"

Prints each matching chunk id with a short window of text around the match, so every
expected fact in questions.jsonl can be traced to a real chunk of the downloaded filing.
Options: --window N (characters of context, default 160), --max N (matches per pattern).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sec_analyst.config import load_settings  # noqa: E402
from sec_analyst.ingest import load_chunks  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--window", type=int, default=160)
    parser.add_argument("--max", type=int, default=4)
    args = parser.parse_args()
    chunks = load_chunks(load_settings().data_dir / "processed" / f"{args.ticker.upper()}_chunks.jsonl")
    for raw in args.patterns:
        pattern = re.compile(raw, re.I)
        print(f"=== {raw}")
        shown = 0
        for c in chunks:
            m = pattern.search(c.text)
            if m and shown < args.max:
                start, end = max(0, m.start() - args.window), min(len(c.text), m.end() + args.window)
                print(f"{c.chunk_id}: ...{c.text[start:end]!r}...")
                shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
