"""Regression check: compare the latest eval results with a stored baseline.

    python scripts/compare_eval.py                                   # eval/results.json vs docs/eval_baseline_v1.json
    python scripts/compare_eval.py --current X.json --baseline Y.json

Prints each metric with its change and exits with code 1 if any metric got worse, so a
change can be rejected automatically ("keep it only if no metric gets worse").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ("retrieval_hit_rate", "Retrieval hit rate"),
    ("answer_correctness_answerable", "Answer correctness (answerable)"),
    ("unanswerable_correctly_refused", "Unanswerable refused"),
    ("citation_validity", "Citation validity"),
]


def as_number(value) -> float:
    """Metrics are floats, except refusals, which are stored as 'correct/total'."""
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/")
        return int(num) / int(den) if int(den) else 0.0
    return float(value)


def compare(current: dict, baseline: dict) -> tuple[list[str], bool]:
    """Return report lines and whether any metric regressed."""
    lines, regressed = [], False
    for key, label in METRICS:
        now, before = as_number(current[key]), as_number(baseline[key])
        delta = now - before
        status = "worse" if delta < -1e-9 else ("better" if delta > 1e-9 else "same")
        regressed |= status == "worse"
        lines.append(f"{label:<34} {before * 100:6.1f}% -> {now * 100:6.1f}%  ({delta * 100:+.1f} pts, {status})")
    return lines, regressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", default=str(ROOT / "eval" / "results.json"))
    parser.add_argument("--baseline", default=str(ROOT / "docs" / "eval_baseline_v1.json"))
    args = parser.parse_args()
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))["summary"]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["summary"]
    lines, regressed = compare(current, baseline)
    print("\n".join(lines))
    print("REGRESSION: at least one metric got worse" if regressed else "OK: no metric got worse")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
