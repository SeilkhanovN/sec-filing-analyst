# STATUS

Overnight build of SEC Filing Analyst, 2026-11-18. Every milestone passed. The final review
phase is finished (tags `v1-working` and `v2-polished`).

## Milestones

| # | Milestone | Result | Date |
|---|-----------|--------|------|
| 1 | Scaffold | PASSED | 2026-11-18 |
| 2 | Tests first | PASSED | 2026-11-18 |
| 3 | Ingestion | PASSED (AAPL 203, MSFT 276, NVDA 284, TSLA 343, AMZN 244 chunks) | 2026-11-18 |
| 4 | Retrieval and answers | PASSED (verified citation; unanswerable -> "Not found in the filings") | 2026-11-18 |
| 5 | Agent | PASSED (mock tests; real Gemini run used both tools; screener unreachable, see below) | 2026-11-18 |
| 6 | Evaluation | PASSED (see numbers below) | 2026-11-18 |
| 7 | Docs | PASSED (README commands were run) | 2026-11-18 |
| 8 | Web app | PASSED (API tests; real server on 127.0.0.1 checked) | 2026-11-18 |
| - | Final review | DONE: 5 improvements implemented and kept, `docs/IMPROVEMENTS.md` | 2026-11-18 |

## Evaluation numbers

28 questions (25 answerable, 3 unanswerable), `gemini-3.5-flash-lite`, top-k 5.

| Metric | v1 (tag v1-working) | v2 (tag v2-polished) |
|---|---|---|
| Retrieval hit rate | 72.0% (18/25) | 80.0% (20/25) |
| Answer correctness (answerable) | 80.0% (20/25) | 88.0% (22/25) |
| Unanswerable refused | 3/3 | 3/3 |
| Citation validity | 100% (23/23) | 100% (25/25), 1 citation re-attributed by citation repair |

Real LLM calls during the run: about 120 in total. That covers 3 eval runs of 28 calls (v1;
hybrid attempt 1; prompt fix) and the CLI, agent and smoke-server checks, plus failed
503/504 retries early on. Each eval run made 28 calls or fewer (cap 40).

## Final checks (run after the review phase)

- pytest: 73 passed (offline, fakes only).
- `.github/workflows/ci.yml` and `docker-compose.yml`: valid YAML. `.env` is git-ignored.
- `cli count`: AAPL 203, MSFT 276, NVDA 284, TSLA 343, AMZN 244 (all > 100).
- `cli ask "How many full-time equivalent employees did Apple have?"` -> 167,000, verified
  citation AAPL-2026-item1-016 (this question failed in v1).
- `cli ask` (cafeteria-chef question) -> "Not found in the filings".
- `cli agent` (Tesla revenue + screener): both tools called, verified citation
  TSLA-2025-item7-003, screener unreachable.
- `eval/run_eval.py`: end to end (from cache). `scripts/compare_eval.py`: "OK: no metric got worse".
- `scripts/smoke_server.py` (real Gemini): /health, /filings, /ask (verified citation), 422 on
  a too-long question, page and assets served. PASSED.

## What is done

- Ingestion from EDGAR (User-Agent, throttling, retries, cache), HTML-to-text, 10-K Item
  detection, chunking with stable ids, local embeddings, Chroma index.
- Hybrid retrieval (vectors + BM25 with RRF), grounded RAG with code-verified citations,
  citation repair, prompt-injection hardening.
- Provider-agnostic `LLMClient` (GeminiClient, FakeClient, marked OpenAIClient stub) with
  backoff/retry, fast failure on daily quota, and a configurable thinking level.
- Tool-using agent (search_filings, get_screener_result) with a step limit and tool-error handling.
- Evaluation (3 metrics), regression check against the v1 baseline.
- FastAPI web app plus an accessible static page (light and dark themes); Dockerfile and
  compose (web is the default service); GitHub Actions CI running pytest.
- Docs: README, docs/EXPLAINED.md (20 interview Q&As), docs/ARCHITECTURE.md (10 ADRs),
  docs/IMPROVEMENTS.md.

## What is not done / known problems

- **Docker was never executed** (Docker commands were not permitted). The files were reviewed
  by reading only. On Linux hosts the mounted `data/` and `chroma_db/` must be writable by
  the container user.
- **CI was never run on GitHub** (no remote may be added). The workflow is valid YAML, and its
  steps are the same commands run locally.
- **Stock screener unreachable:** SCREENER_URL is set in .env, but every call ended in
  ConnectionError (nothing was listening at that address during the run). The tool's
  success path is tested only against the fake HTTP server. The endpoint path
  `/api/screener/{ticker}` is an assumption; set `SCREENER_PATH` if the real API differs.
- **Free-tier quotas:** `gemini-3.8-flash` allows 20 requests/day per model. It was exhausted
  during M5. Hence the default `gemini-3.5-flash-lite` (see Decisions).
- Remaining retrieval misses: nvda-1, tsla-1, tsla-3, tsla-5, amzn-5. nvda-1 and amzn-5 are
  still answered "not found"; tsla-3 answers "Texas" without "Austin".
- Answer correctness is a strict string match. nvda-2 was once marked wrong for
  "$215.938 billion" against an expected "215.9 billion". The eval file was not changed.
- Microsoft's financial tables put cells on separate lines. Fixing the parser would change
  chunk ids and invalidate the eval set, so it was deferred.
- Improvements 6 to 12 in docs/IMPROVEMENTS.md were not implemented (ruff in CI, API answer
  cache, more eval questions, faithfulness check, second provider, reranker, table parsing).

## Decisions

- **Model:** the official docs pages on ai.google.dev redirect to aistudio.google.com, which
  this run's WebFetch allow-list blocked. The google-genai SDK docs (googleapis.github.io) were
  read instead, and `models.list` (no generation cost) confirmed the models available to
  the key. `gemini-3.8-flash` worked in M4 but has only 20 free requests/day and returned many
  503 "high demand" errors. Default changed to **`gemini-3.5-flash-lite`** (Flash-class,
  free tier, function calling). The agent check and every eval run used it.
- **Thinking level `low`:** one call with the default thinking level failed with `504
  DEADLINE_EXCEEDED` (the SDK sends the client timeout as the server deadline).
- **Gemini function results** are sent in a `user` turn, grouped; `role="tool"` was rejected
  with 400. The raw model turn is kept so Gemini 3 thought signatures are echoed back.
- **Embeddings:** `BAAI/bge-small-en-v1.5`, cached in `data/hf_cache` (nothing is written
  outside the project) and loaded offline once cached.
- **Chunking:** 1,500 characters with 200 overlap, split by Item first; ids are
  `TICKER-FY-itemX-NNN`. The table-of-contents tail is assigned to "Cover page" (M3 fix).
  NVDA puts its financial statements under Item 15; that is indexed as is.
- **Fiscal year** = year of the report date (matches each company's naming).
- **Eval facts** were located with `eval/search_chunks.py`, and a script confirmed each
  expected fact occurs in its source chunk. `also_in` lists overlapping chunks that contain
  the same fact.
- **Citation repair** (improvement 1, fix attempt 2): a real quote under the wrong chunk id is
  re-attributed to the retrieved passage that contains it (`corrected_from` keeps the model's
  id). This was needed because the first hybrid attempt lowered citation validity to 23/24.
- **Not committed:** BRIEF.md, run_night.ps1 and setup.ps1 (the author's overnight-run
  harness) are left untracked for the author to decide.
- Git identity: the global git user was already configured, so no local identity was set.
- The frontend-design skill was not invoked; the brief's detailed design specification was
  followed directly.
- No instructions addressed to the agent were found in any fetched web page or filing.

## Changes to tests/ or eval/ after they were committed

No existing test or eval case was edited or deleted. Added:
- tests: `test_sections.py` (M3), `test_gemini_conversion.py` (M5), `test_api.py` (M8),
  `test_hybrid_and_repair.py`, `test_prompt_injection.py`, `test_embedder_offline.py`,
  `test_compare_eval.py` (review phase).
- eval: only the generated outputs `eval/results.md` / `eval/results.json` were regenerated by
  `run_eval.py`. `questions.jsonl` and `run_eval.py` are unchanged since M6. The v1 results are
  kept in `docs/eval_baseline_v1.json`.

## Exact commands (Windows PowerShell, from the project folder)

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m sec_analyst.cli check-config
.venv\Scripts\python -m sec_analyst.cli ingest
.venv\Scripts\python -m sec_analyst.cli count
.venv\Scripts\python -m sec_analyst.cli ask "How many full-time equivalent employees did Apple have?"
.venv\Scripts\python -m sec_analyst.cli agent "What were Tesla's total revenues in 2025, and what does the screener say about TSLA?"
.venv\Scripts\python eval\run_eval.py
.venv\Scripts\python scripts\compare_eval.py
.venv\Scripts\python scripts\smoke_server.py            # add --fake to run without an API key
.venv\Scripts\python -m uvicorn sec_analyst.api:app --host 127.0.0.1 --port 8000   # open http://127.0.0.1:8000
docker compose up --build web                           # not executed during the build
```
