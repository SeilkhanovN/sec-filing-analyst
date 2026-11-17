# STATUS

Overnight build log for SEC Filing Analyst. Newest state at the top of each section.

## Milestones

| # | Milestone | Result | Date/time |
|---|-----------|--------|-----------|
| 1 | Scaffold | PASSED | 2026-11-18 |
| 2 | Tests first | PASSED | 2026-11-18 |
| 3 | Ingestion | PASSED (AAPL 203, MSFT 276, NVDA 284, TSLA 343, AMZN 244 chunks) | 2026-11-18 |
| 4 | Retrieval and answers | not started | |
| 5 | Agent | not started | |
| 6 | Evaluation | not started | |
| 7 | Docs | not started | |
| 8 | Web app | not started | |

## Next step

Milestone 4: real-LLM check of `python -m sec_analyst.cli ask` (answerable + unanswerable).

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

## Changes to tests/ or eval/ after they were committed

- M3: added `tests/test_sections.py` (new test; no existing test edited).

## Notes

- Docker commands are not permitted in this run: Dockerfile and docker-compose.yml were
  reviewed by reading only, never built.
