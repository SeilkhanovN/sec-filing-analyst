"""Regression check: compare the latest eval results with a stored baseline.

    python scripts/compare_eval.py                                   # eval/results.json vs docs/eval_baseline_v1.json
    python scripts/compare_eval.py --current X.json --baseline Y.json

Prints each metric with its change and exits with code 1 if any metric got worse, so a
change can be rejected automatically ("keep it only if no metric gets worse").

Side-by-side report of two or more runs (for example Gemini vs a local Ollama model). It
reads the stored results only (no LLM calls), the first run is the reference, and the
report lists every question whose correctness differs from it:

    python scripts/compare_eval.py --side-by-side gemini=eval/results.json ollama=eval/results_ollama.json --out eval/comparison.md
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


def _pct(num: int, den: int) -> str:
    return f"{num / den * 100:.1f}% ({num}/{den})" if den else "n/a"


def run_metrics(run: dict) -> dict[str, str]:
    """Display values for one results.json. Counts come from the per-question rows, which
    are scored by eval/run_eval.py; nothing is re-scored here."""
    rows, summary, meta = run["rows"], run["summary"], run.get("meta", {})
    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in answerable if r.get("status") == "ok"]
    unanswerable = [r for r in rows if not r["answerable"] and r.get("status") == "ok"]
    cit_valid = sum(r.get("citations_valid", 0) for r in rows if r.get("status") == "ok")
    cit_total = sum(r.get("citations_total", 0) for r in rows if r.get("status") == "ok")
    seconds = summary.get("avg_seconds_per_question")
    return {
        "Model": f"`{meta.get('model', '?')}` ({meta.get('date', '?')})",
        "Retrieval hit rate (source chunk in top-k)": _pct(sum(bool(r["retrieval_hit"]) for r in answerable), len(answerable)),
        "Answer correctness (answerable)": _pct(sum(bool(r["correct"]) for r in answered), len(answered)),
        "Unanswerable correctly refused": f"{sum(bool(r['correct']) for r in unanswerable)}/{len(unanswerable)}",
        "Citation validity (quote in cited chunk)": _pct(cit_valid, cit_total),
        "Average seconds per question": f"{seconds:.1f} s" if seconds is not None else "not recorded",
        "LLM errors": str(summary.get("llm_errors", 0)),
    }


def side_by_side(runs: list[tuple[str, dict]]) -> str:
    """Markdown: one metrics table, then the questions whose correctness differs from the first run."""
    labels = [label for label, _ in runs]
    metrics = [run_metrics(run) for _, run in runs]
    lines = ["| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(runs)]
    for key in metrics[0]:
        lines.append(f"| {key} | " + " | ".join(m[key] for m in metrics) + " |")
    ref_label, ref = runs[0]
    ref_rows = {r["id"]: r for r in ref["rows"]}
    for label, run in runs[1:]:
        lines += ["", f"### {label} vs {ref_label}: questions with a different result", ""]
        changed = []
        for r in run["rows"]:
            base = ref_rows.get(r["id"])
            if base is None:
                continue
            now_ok, before_ok = bool(r.get("correct")), bool(base.get("correct"))
            bad_cites = r.get("citations_valid", 0) < r.get("citations_total", 0)
            if now_ok != before_ok or bad_cites or r.get("status") != "ok":
                verdict = "error" if r.get("status") != "ok" else ("correct" if now_ok else "wrong")
                verdict += f" (was {'correct' if before_ok else 'wrong'} with {ref_label})"
                answer = str(r.get("answer", r.get("error", ""))).replace("|", "/").replace("\n", " ")[:160]
                cites = f"{r.get('citations_valid', 0)}/{r.get('citations_total', 0)}"
                changed.append(f"| {r['id']} | {verdict} | {cites} | {answer} |")
        if changed:
            lines += [f"| id | {label} result | valid citations | {label} answer (truncated) |", "|---|---|---|---|"] + changed
        else:
            lines.append("No differences.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", default=str(ROOT / "eval" / "results.json"))
    parser.add_argument("--baseline", default=str(ROOT / "docs" / "eval_baseline_v1.json"))
    parser.add_argument("--side-by-side", nargs="+", metavar="LABEL=RESULTS.json",
                        help="print a comparison table of two or more runs instead of the regression check")
    parser.add_argument("--out", default=None, help="with --side-by-side: also write the markdown to this file")
    args = parser.parse_args()
    if args.side_by_side:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        runs = []
        for item in args.side_by_side:
            label, sep, path = item.partition("=")
            if not sep:
                parser.error(f"expected LABEL=path, got {item!r}")
            runs.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
        report = side_by_side(runs)
        print(report)
        if args.out:
            Path(args.out).write_text(report, encoding="utf-8")
        return 0
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))["summary"]
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))["summary"]
    lines, regressed = compare(current, baseline)
    print("\n".join(lines))
    print("REGRESSION: at least one metric got worse" if regressed else "OK: no metric got worse")
    return 1 if regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
