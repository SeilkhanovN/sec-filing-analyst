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
