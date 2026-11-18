# STATUS

Overnight build log for SEC Filing Analyst. Newest state at the top of each section.

## Milestones

| # | Milestone | Result | Date/time |
|---|-----------|--------|-----------|
| 1 | Scaffold | PASSED | 2026-11-18 |
| 2 | Tests first | PASSED | 2026-11-18 |
| 3 | Ingestion | PASSED (AAPL 203, MSFT 276, NVDA 284, TSLA 343, AMZN 244 chunks) | 2026-11-18 |
| 4 | Retrieval and answers | PASSED (real Gemini: cited answer verified; unanswerable -> "Not found in the filings") | 2026-11-18 |
| 5 | Agent | PASSED (mock-LLM tests pass; real Gemini run used both tools; screener unreachable, see notes) | 2026-11-18 |
| 6 | Evaluation | PASSED (retrieval hit 72.0%, answer correctness 80.0% answerable + 3/3 refusals, citation validity 100% 23/23) | 2026-11-18 |
| 7 | Docs | PASSED (README commands run: pytest, check-config, ingest, count, ask --ticker, agent, eval; pip dry-run of requirements) | 2026-11-18 |
| 8 | Web app | PASSED (13 API tests; real server on 127.0.0.1: /health, /filings, /ask real Gemini, 422 check, page + assets served) | 2026-11-18 |

## Final re-run of all checks (before the review phase)

- M1: pytest exits cleanly; ci.yml and docker-compose.yml are valid YAML; `.env` is git-ignored. PASS
- M2: 58 tests pass (fakes only). PASS
- M3: AAPL 203, MSFT 276, NVDA 284, TSLA 343, AMZN 244 chunks (all > 100). PASS
- M4: `ask "What are Apple's reportable segments?"` -> verified citation AAPL-2026-item1-003;
  cafeteria-chef question -> "Not found in the filings". PASS
- M5: real agent run (AMZN AWS growth + screener): both tools called, verified citation
  AMZN-2025-item7-024; screener unreachable (ConnectionError). PASS
- M6: `eval/run_eval.py` end to end: hit rate 72.0%, correctness 80.0%, citation validity 100%. PASS
- M7: README / docs/EXPLAINED.md / STATUS.md exist; README commands were run in M7. PASS
- M8: `scripts/smoke_server.py` (real LLM) and `--fake`: all endpoint checks passed. PASS

## Next step

Final review: re-run all checks, tag v1-working, write docs/IMPROVEMENTS.md, implement the
top improvements one at a time.

## Evaluation (M6 baseline)

`python eval/run_eval.py` with `gemini-3.5-flash-lite`, top-k 5, 28 questions (25 answerable,
3 unanswerable). First run: 28 live LLM calls, 0 errors (later re-runs read the cache).

| Metric | Value |
|---|---|
| Retrieval hit rate (source chunk in top-5) | 72.0% (18/25) |
| Answer correctness, answerable | 80.0% (20/25) |
| Unanswerable correctly refused | 3/3 |
| Citation validity | 100% (23/23) |

All 5 wrong answers were "not found" after a retrieval miss (aapl-1, aapl-2, aapl-5, nvda-1,
amzn-5): the model refused rather than guessed. Two retrieval misses still got correct
answers from another chunk with the same fact (nvda-3, tsla-5).

## Filings indexed (latest 10-K as of 2026-11-18)

| Ticker | Fiscal year (report date) | Filed | Accession |
|---|---|---|---|
| AAPL | 2026 (2026-09-26) | 2026-11-03 | 0000320193-26-000026 |
| MSFT | 2026 (2026-06-30) | 2026-07-29 | 0001193125-26-323660 |
| NVDA | 2026 (2026-01-25) | 2026-02-25 | 0001045810-26-000021 |
| TSLA | 2025 (2025-12-31) | 2026-01-29 | 0001628280-26-003952 |
| AMZN | 2025 (2025-12-31) | 2026-02-06 | (see data/raw/AMZN/filing.json) |

## Decisions

- **LLM model default `gemini-3.8-flash`.** The official docs pages on ai.google.dev now
  301-redirect to aistudio.google.com, which the run's WebFetch allow-list blocks. The
  official google-genai SDK docs (googleapis.github.io/python-genai) use `gemini-3.8-flash`
  in their text and function-calling examples. To be verified against the live API's model
  list before the first real call (see M4 notes).
- **Package layout:** `src/sec_analyst`, installed with `pip install --no-deps -e .`; pytest
  also adds `src` to the path, so tests run without installing.
- **Embeddings:** `BAAI/bge-small-en-v1.5` (384-dim, 512-token limit, runs on CPU). Chunks are
  at most 1500 characters (~350 tokens) with 200 characters of overlap.
- **Hugging Face cache** is redirected to `data/hf_cache` so nothing is written outside the
  project folder.
- **Chroma** is always given our own embeddings (never its default model) and telemetry is off.
- **Gemini thought signatures:** assistant turns keep the provider's raw content object
  (`Message.raw`), so multi-step tool use works with Gemini 3 models.
- **Screener API shape is unknown** to this build. The tool calls
  `SCREENER_URL + SCREENER_PATH` (default `/api/screener/{ticker}`), both configurable.
- **Not committed:** BRIEF.md, run_night.ps1, setup.ps1 (the author's local overnight-run
  harness) are left untracked for the author to decide.
- Git user: the global git identity is configured, so no repository-local identity was set.

- **Section detection (M3):** inspecting AAPL showed the last table-of-contents entry
  ("Item 16") swallowing the forward-looking-statements preamble. Fix: a segment whose next
  heading goes backwards in item order is front matter ("Cover page"). New test file
  `tests/test_sections.py` (added, no existing test changed).
- NVDA puts its financial statements under Item 15, not Item 8 (Item 8 only refers to them);
  the index reflects that faithfully.
- Fiscal year = year of the filing's report date (matches each company's own naming).
- **Model verified (M4):** `models.list` (no generation cost) shows `gemini-3.8-flash` available
  to this key, and real free-tier calls succeed.
- **Thinking level `low` (M4):** the first unanswerable-question call failed 5 times with
  `504 DEADLINE_EXCEEDED` (the SDK passes the client timeout to the server as a deadline and the
  model's default thinking ran past it). Fix: `LLM_THINKING_LEVEL=low` (configurable) and a 120 s
  timeout. After the fix: answered in seconds. Real LLM calls so far: 4 successful requests in M4
  (+ the failed attempts above).
- **Gemini function results (M5):** the API rejected `role="tool"` (400 INVALID_ARGUMENT);
  function results are now sent in a `user` turn, parallel results grouped. Offline test added
  (`tests/test_gemini_conversion.py`, new file).
- **Free-tier quota (M5):** `gemini-3.8-flash` allows only **20 requests per day per model** on
  the free tier (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20). The quota
  ran out after the M4 checks plus 503/504 overload retries ("model is experiencing high
  demand"). The CLI and eval now take `--model`; quotas are per model.
- **M5 real run:** `python -m sec_analyst.cli --model gemini-3.5-flash-lite agent "What does
  NVIDIA's 10-K say about its two operating segments, and what does the stock screener
  currently report for NVDA?"` -> step 1 called `search_filings` and `get_screener_result` in
  parallel; step 2 answered with 2 verified citations (NVDA-2026-item15-073). 2 LLM calls.
- **Screener unreachable:** `get_screener_result` returned "Stock screener unreachable
  (ConnectionError)". SCREENER_URL is set in .env but nothing answered at that address during
  the run (the author's screener service was not running or is not reachable from this
  machine). The agent handled it as designed and said live data was unavailable. The
  endpoint path (`/api/screener/{ticker}`) is an assumption and may need `SCREENER_PATH`.
- **Default model changed to `gemini-3.5-flash-lite` (M6):** it is a current Flash-class
  (Flash-Lite) model, on the free tier, supports function calling, and is the model the agent
  run and the whole evaluation were verified with. `gemini-3.8-flash` (verified in M4) is
  limited to 20 free requests/day, less than one 28-question eval, and returned many 503
  "high demand" errors during the run. Either can be chosen with `LLM_MODEL`.
- **Eval questions** were written from facts located with `eval/search_chunks.py`; a script
  confirmed that every expected fact occurs verbatim (after normalisation) in its source
  chunk. `also_in` lists overlapping chunks that contain the same fact.
- **Web app (M8):** `create_app(services)` takes injected services (tests pass fakes);
  production builds them lazily on the first request. LLM outages map to HTTP 503, validation
  to 422, anything else to a generic 500. The front end is three static files; the frontend
  design skill was not invoked (the brief's detailed design spec was followed directly).
  Docker: `web` is the default service; `cli` is behind a compose profile.
- RAG's `answer_question` raises `LLMError` (the eval needs to tell errors from "not found");
  the CLI catches it and prints a clean error. Retries are logged as warnings.

## Changes to tests/ or eval/ after they were committed

- M3: added `tests/test_sections.py` (new test; no existing test edited).

## Notes

- Docker commands are not permitted in this run: Dockerfile and docker-compose.yml were
  reviewed by reading only, never built.
