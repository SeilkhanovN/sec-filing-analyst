"""Smoke test of the real web server (cross-platform, no curl needed).

    python scripts/smoke_server.py            # uses the configured LLM (1 real call, mode=rag)
    python scripts/smoke_server.py --fake     # offline fake LLM, no API key needed

Starts `python -m sec_analyst.api` on 127.0.0.1 as a subprocess, waits for /health, calls
/filings and /ask, checks that the page and its assets are served, then stops the server.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="use LLM_PROVIDER=fake")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ, API_HOST="127.0.0.1", API_PORT=str(args.port), PYTHONPATH=str(ROOT / "src"))
    if args.fake:
        env["LLM_PROVIDER"] = "fake"
    server = subprocess.Popen([sys.executable, "-m", "sec_analyst.api"], cwd=ROOT, env=env)
    failures = 0
    try:
        deadline = time.monotonic() + args.timeout
        health = None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                print("FAIL: server exited early")
                return 1
            try:
                health = requests.get(f"{base}/health", timeout=60).json()
                break
            except requests.RequestException:
                time.sleep(1)
        print("GET /health ->", health)
        failures += health is None or health.get("status") != "ok"

        filings = requests.get(f"{base}/filings", timeout=30).json()
        print("GET /filings ->", [(f["ticker"], f["chunks"], f.get("fiscal_year")) for f in filings])
        failures += len(filings) != 5

        resp = requests.post(f"{base}/ask", json={"question": "What are Apple's reportable segments?", "mode": "rag"}, timeout=300)
        data = resp.json()
        print("POST /ask ->", resp.status_code, "|", str(data.get("answer"))[:200])
        for c in data.get("citations", []):
            print(f"   citation {c['chunk_id']} valid={c['valid']}: {c['quote'][:100]}")
        failures += resp.status_code != 200 or not data.get("citations")

        bad = requests.post(f"{base}/ask", json={"question": "x" * 600}, timeout=30)
        print("POST /ask (too long) ->", bad.status_code)
        failures += bad.status_code != 422

        page = requests.get(f"{base}/", timeout=30)
        js = requests.get(f"{base}/static/app.js", timeout=30)
        css = requests.get(f"{base}/static/styles.css", timeout=30)
        print("GET / ->", page.status_code, page.headers.get("content-type"), "| app.js", js.status_code, "| styles.css", css.status_code)
        failures += page.status_code != 200 or "SEC Filing Analyst" not in page.text or js.status_code != 200 or css.status_code != 200
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
    print("SMOKE TEST", "PASSED" if not failures else f"FAILED ({failures} checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
