# STATUS

Overnight build log for SEC Filing Analyst. Newest state at the top of each section.

## Milestones

| # | Milestone | Result | Date/time |
|---|-----------|--------|-----------|
| 1 | Scaffold | PASSED | 2026-11-18 |
| 2 | Tests first | PASSED | 2026-11-18 |
| 3 | Ingestion | not started | |
| 4 | Retrieval and answers | not started | |
| 5 | Agent | not started | |
| 6 | Evaluation | not started | |
| 7 | Docs | not started | |
| 8 | Web app | not started | |

## Next step

Milestone 3: download and index the five 10-Ks.

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

## Changes to tests/ or eval/ after they were committed

None yet.

## Notes

- Docker commands are not permitted in this run: Dockerfile and docker-compose.yml were
  reviewed by reading only, never built.
