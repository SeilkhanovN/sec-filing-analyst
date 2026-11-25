"""Added in the improvement phase: the eval regression check (scripts/compare_eval.py)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval.py"
spec = importlib.util.spec_from_file_location("compare_eval", SCRIPT)
compare_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_eval)

BASE = {"retrieval_hit_rate": 0.72, "answer_correctness_answerable": 0.8,
        "unanswerable_correctly_refused": "3/3", "citation_validity": 1.0}


def test_improvement_passes_and_any_drop_fails():
    better = dict(BASE, retrieval_hit_rate=0.8)
    lines, regressed = compare_eval.compare(better, BASE)
    assert not regressed and "better" in lines[0]

    worse = dict(better, unanswerable_correctly_refused="2/3")
    assert compare_eval.compare(worse, BASE)[1]


def test_cli_exit_code(tmp_path):
    cur, base = tmp_path / "cur.json", tmp_path / "base.json"
    base.write_text(json.dumps({"summary": BASE}))
    cur.write_text(json.dumps({"summary": dict(BASE, citation_validity=0.9)}))
    result = subprocess.run([sys.executable, str(SCRIPT), "--current", str(cur), "--baseline", str(base)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "REGRESSION" in result.stdout


def _run(model, rows, seconds=None):
    return {"meta": {"model": model, "date": "today"}, "rows": rows,
            "summary": {"avg_seconds_per_question": seconds, "llm_errors": sum(r["status"] != "ok" for r in rows)}}


def _row(qid, answerable=True, hit=True, correct=True, valid=1, total=1, status="ok", answer="a"):
    return {"id": qid, "answerable": answerable, "retrieval_hit": hit if answerable else None, "correct": correct,
            "citations_valid": valid, "citations_total": total, "status": status, "answer": answer}


def test_side_by_side_reports_metrics_and_every_changed_question():
    gemini = _run("gemini-x", [_row("q1"), _row("q2"), _row("q3", hit=False, correct=False, total=0, valid=0),
                               _row("none-1", answerable=False, total=0, valid=0)])
    local = _run("llama-y", [_row("q1"), _row("q2", correct=False, answer="wrong number"),
                             _row("q3", hit=False, correct=True, valid=0, total=1),
                             _row("none-1", answerable=False, correct=False, total=0, valid=0)], seconds=41.26)
    report = compare_eval.side_by_side([("gemini", gemini), ("ollama", local)])
    assert "| Answer correctness (answerable) | 66.7% (2/3) | 66.7% (2/3) |" in report
    assert "| Retrieval hit rate (source chunk in top-k) | 66.7% (2/3) | 66.7% (2/3) |" in report
    assert "| Unanswerable correctly refused | 1/1 | 0/1 |" in report
    assert "| Citation validity (quote in cited chunk) | 100.0% (2/2) | 66.7% (2/3) |" in report
    assert "| Average seconds per question | not recorded | 41.3 s |" in report
    assert "| q2 | wrong (was correct with gemini) | 1/1 | wrong number |" in report
    assert "| q3 | correct (was wrong with gemini) | 0/1 |" in report
    assert "| none-1 | wrong (was correct with gemini)" in report
    assert "| q1 |" not in report
