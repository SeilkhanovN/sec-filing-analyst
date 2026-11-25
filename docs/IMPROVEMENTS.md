# Candidate improvements (critic pass)

Written after tagging `v1-working`. I reread the repository as a hiring manager for an AI
Engineer role (LLM, RAG, semantic search, agents, provider-agnostic design, Docker/CI,
vector DBs, clear communication). Every weakness below cites evidence from this repository.
Ranked by benefit / (risk x effort), where S/M/L = 1/2/3 and low/medium/high = 1/2/3.

| Rank | Improvement | Benefit | Risk | Effort | Status |
|---|---|---|---|---|---|
| 1 | Hybrid retrieval (BM25 + vectors, reciprocal rank fusion) | high (3) | low | M | implemented, kept |
| 2 | Prompt-injection hardening + tests | medium (2) | low | S | implemented, kept |
| 3 | Load the embedding model offline once cached | medium (2) | low | S | implemented, kept |
| 4 | Eval regression check against a stored baseline | medium (2) | low | S | implemented, kept |
| 5 | Architecture decision record (docs/ARCHITECTURE.md) | medium (2) | low | S | implemented, kept |
| 6 | Lint (ruff) in CI | low-medium | low | S | not done |
| 7 | Answer cache in the API | medium | low | M | not done |
| 8 | More eval questions, incl. unanswerable (questions_extra.jsonl) | medium | low | M | not done |
| 9 | Faithfulness check (does the quote support the claim?) | medium | medium | M | not done |
| 10 | Second LLM provider implemented against fakes | low-medium | low | M | done in v3 as `OllamaClient` (local model; mocked tests + real runs), see below |
| 11 | Cross-encoder reranker | medium | medium | M | not done |
| 12 | Better table extraction | low-medium | high | M | not done |

## 1. Hybrid retrieval
- **Weakness:** retrieval is the bottleneck. In `eval/results.md`, 7 of 25 source chunks are
  missing from the top 5 (aapl-1, aapl-2, aapl-5, nvda-1, nvda-3, tsla-5, amzn-5), and **all 5
  wrong answers** are "not found" after a miss. The misses are exact-wording facts
  ("full-time equivalent employees", a street address, "gross margin percentage") where a
  small embedding model ranks thematically similar passages higher.
- **Fix:** add a BM25 keyword index over the same chunks (about 60 lines, no new
  dependency) and merge the two rankings with reciprocal rank fusion (RRF, k=60). Ticker
  filtering still applies. Chunk ids do not change, so the eval stays valid.
- **Benefit:** a higher hit rate on exact terms, numbers and names. **Risk:** low, because
  fusion can also push a good vector hit down; the eval guards against that. **Effort:** M.

## 2. Prompt-injection hardening
- **Weakness:** `rag.build_prompt` wraps passages in `<context>` tags, but chunk text is
  inserted raw. A filing passage containing `</context>` plus instructions could close the data
  block early. No test covers injected instructions in retrieved text or in tool results.
- **Fix:** neutralise the delimiter tags inside chunk text, and add tests. A malicious
  chunk stays inside the data block. The system prompt keeps the "treat as data"
  rule. A fake LLM that obeys injected text still cannot produce a verified citation unless
  the quote is really in the chunk.
- **Benefit:** a documented, tested defence. **Risk:** low. **Effort:** S.

## 3. Offline model loading
- **Weakness:** the server smoke-test log shows four HTTP requests to huggingface.co on every
  start, even though the model is cached in `data/hf_cache`. Startup is slower, and it fails
  without internet.
- **Fix:** if the model is already in the cache, set `HF_HUB_OFFLINE=1` before loading.
- **Benefit:** faster, deterministic, offline-capable startup (useful in Docker). **Risk:** low.
  **Effort:** S.

## 4. Eval regression check
- **Weakness:** `eval/run_eval.py` prints metrics, but nothing flags a regression. Comparing
  before and after numbers is manual.
- **Fix:** a `--baseline eval/baseline.json` option that prints the deltas and exits non-zero if
  any metric drops; commit the v1 numbers as the baseline.
- **Benefit:** safe experimentation, and it shows the habit of evaluation-driven changes.
  **Risk:** low. **Effort:** S.

## 5. Architecture decision record
- **Weakness:** design decisions (chunk size, embedding model, Chroma, JSON citations,
  thinking level, model choice, tool error handling) are spread across STATUS.md and README.
- **Fix:** `docs/ARCHITECTURE.md` with context, decision, alternatives and trade-offs.
- **Benefit:** clear communication of trade-offs. **Risk:** none. **Effort:** S.

## 6 to 12 (not implemented)
6. **Ruff in CI.** CI runs only pytest (`.github/workflows/ci.yml`). Cheap, but it adds a
   dependency and has little effect on behaviour.
7. **Answer cache in the API.** The free tier allows 20 requests/day for `gemini-3.8-flash`,
   so repeated demo questions spend quota. The eval already has a cache; the API does not.
8. **More eval questions.** Only 28 questions and 3 unanswerable. More need LLM calls under
   free-tier quotas.
9. **Faithfulness check.** Citation validity proves the quote exists, not that it supports the
   answer. An LLM-judge pass doubles the calls.
10. **Second provider.** `OpenAIClient` is a stub. Implementing it against fakes only would add
    code that has never run against the real service.
11. **Reranker.** It needs a second model download and adds latency; try hybrid search first.
12. **Table extraction.** Microsoft's tables put cells on separate lines
    (`MSFT-2026-item7-017`). Fixing the parser changes chunk boundaries and ids, which would
    invalidate the committed eval set (high risk).

## Results of implemented improvements

All numbers come from `eval/run_eval.py` (28 questions, `gemini-3.5-flash-lite`, top-k 5).
The v1 baseline is saved in `docs/eval_baseline_v1.json`.

### 1. Hybrid retrieval: KEPT

| Metric | v1 (vector) | hybrid, attempt 1 | hybrid + quote rule + citation repair (kept) |
|---|---|---|---|
| Retrieval hit rate | 72.0% (18/25) | 80.0% (20/25) | 80.0% (20/25) |
| Answer correctness (answerable) | 80.0% (20/25) | 84.0% (21/25) | 88.0% (22/25) |
| Unanswerable refused | 3/3 | 3/3 | 3/3 |
| Citation validity | 100% (23/23) | **95.8% (23/24)** | 100% (25/25), 1 re-attributed |

- Hybrid fixed 4 retrieval misses (aapl-1, aapl-2, aapl-5, nvda-3) and lost 2 (tsla-1,
  tsla-3). Net +2. Still missed: nvda-1, tsla-1, tsla-3, tsla-5, amzn-5.
- Attempt 1 made citation validity worse, so by the rules it could not be kept as is. The
  failing citation (tsla-1) quoted `Total revenues | $94,827 | ...` under chunk
  `TSLA-2025-item8-013`, whose row has no `$`. The `$` version exists verbatim in
  `TSLA-2025-item8-024`, which the model was also shown.
- Fix attempt 1: a stricter quoting rule in the prompt ("never add, drop or reformat symbols
  or numbers"). Correctness rose to 88% (nvda-2 now matches), but the tsla-1 quote was still
  mislabelled.
- Fix attempt 2: **citation repair**. If a quote is not in the cited chunk but appears
  verbatim in another passage the model was shown, the citation is moved to that passage and
  `corrected_from` records the model's id (shown in the CLI and the UI). A quote found nowhere
  stays invalid (tested). In this run, 1 of 25 citations was re-attributed.
- LLM calls: 28 (attempt 1) + 28 (prompt change); the repair step reused cached replies.
- Not changed: chunking (chunk ids, and so the eval set, stay valid) and the eval files.

### 2. Prompt-injection hardening: KEPT

- `citations.neutralize_delimiters` replaces `<context>`, `</context>` and `[chunk_id:` inside
  retrieved text before it enters the prompt. A passage can no longer close the data block
  early or fake a chunk header.
- New `tests/test_prompt_injection.py` (4 tests): delimiter escape; both system prompts
  declare retrieved text and tool results untrusted; a fake model that obeys an injected
  instruction gets no verified citation (`grounded: false`); hostile tool output stays in the
  tool message and never changes the system prompt.
- Eval before/after: identical (hit 80.0%, correctness 88.0%, refusals 3/3, citations 100%).
  No real chunk contained a delimiter, so all 28 replies came from the cache (0 LLM calls).
- Honest limit: if an injected sentence is in the filing and the model quotes it, the quote
  is "valid" (it exists). Verification proves provenance, not truth.

### 3. Offline model loading: KEPT

- If `data/hf_cache/hub/models--BAAI--bge-small-en-v1.5` exists, the embedder loads with
  `local_files_only=True`. If that fails (incomplete cache), it falls back to a normal load.
- Before: 4 HTTP requests to huggingface.co on every server start (smoke-test log). After: 0.
- New `tests/test_embedder_offline.py` (3 tests, using a fake sentence_transformers module).
- Eval before/after: identical (same model files; 0 LLM calls).

### 4. Eval regression check: KEPT

- `scripts/compare_eval.py` compares `eval/results.json` with `docs/eval_baseline_v1.json`
  (the v1 results, taken from the `v1-working` tag). It prints the deltas and exits 1 if any
  metric dropped. It lives outside `eval/` so the committed eval files stay unchanged.
- New `tests/test_compare_eval.py` (2 tests).
- Current output: hit rate +8.0 pts, correctness +8.0 pts, refusals and citation validity
  unchanged. "OK: no metric got worse".

### 5. Architecture decision record: KEPT

- `docs/ARCHITECTURE.md`: 10 short ADRs (grounding, hybrid retrieval, embeddings, Chroma,
  chunking, provider interface, model and rate limits, agent loop, prompt injection,
  offline testing), each with context, decision, alternatives and trade-off.
- Documentation only: tests and eval unchanged.

### Not implemented

Items 6 to 12 were not implemented (the brief allows at most 5). The next ones by rank would
be ruff in CI and an answer cache in the API.

## v3: Privately hosted LLM via Ollama

Added on 2026-11-25 (tag `v3-local-llm`). Goal: run the whole question-answering path with
no data leaving the computer, keeping every safety check. Design and trade-offs: ADR-11 in
`docs/ARCHITECTURE.md`.

### What was done
- `OllamaClient` (`src/sec_analyst/ollama_client.py`) behind the existing `LLMClient`
  interface: Ollama's `/api/chat` through `requests`, with no new dependency. One attempt per call, 5 s
  connect and 180 s read timeouts, and clear errors for "not running", "model not installed" and
  "timed out".
- Localhost-only `OLLAMA_HOST`, overridable only with `ALLOW_REMOTE_LLM=1`; proxy environment
  variables are ignored.
- Agent tool use through the interface: native Ollama tools when the model supports them, or
  a strict JSON tool-call format. Every call is validated. A malformed call becomes a
  failed step (`LLMResponse.tool_error`) that still counts towards `max_steps`; Gemini and
  the fake never set it.
- Configuration: `LLM_PROVIDER=gemini|ollama` (default gemini), `OLLAMA_MODEL`,
  `OLLAMA_HOST`, `OLLAMA_TIMEOUT_S`, `OLLAMA_NUM_CTX`, `OLLAMA_TOOL_MODE`. CLI `--provider`
  (and `--model` for the active provider), the API and the eval runner respect it.
- `eval/run_eval.py`: `--provider`; seconds per question (excluding the free-tier pause);
  non-Gemini runs write `eval/results_<provider>.md` and use their own cache, so the Gemini
  results cannot be overwritten. Questions, scoring and thresholds unchanged. A
  `--no-cache` run no longer wipes the existing cache file.
- `scripts/compare_eval.py --side-by-side`: one table for several runs plus every question
  whose result differs.
- 49 new offline tests (mocked HTTP session: success, Ollama down, model missing, timeout,
  malformed tool output, non-local hosts, config, agent end to end through the client).
  Total: 122 passed.

### Measured results (28 questions, top-k 5, 2026-11-25)

| Metric | Gemini `gemini-3.5-flash-lite` (re-run) | Local `llama3.2:3b` via Ollama |
|---|---|---|
| Retrieval hit rate | 80.0% (20/25) | 80.0% (20/25) |
| Answer correctness (answerable) | 88.0% (22/25) | 76.0% (19/25) |
| Unanswerable correctly refused | 3/3 | 3/3 |
| Citation validity | 100.0% (25/25) | 86.4% (19/22) |
| Seconds per question: mean / median | 9.1 / 1.3 | 33.8 / 34.1 |
| LLM errors | 0 | 0 |

- Hardware for the local run: Intel Core Ultra 5 125U, 16 GB RAM, integrated graphics
  only. `ollama ps`: 3.1 GB, 100% CPU, context 8,192. Prompts were 1,453 to 2,614 tokens (no
  truncation warning).
- The Gemini re-run (28 live calls, separate cache) reproduced the stored v2 results
  question by question. It was needed because the v2 run had no timing. Its mean includes 13
  server-side 503 retries with backoff (slowest question 63 s), so the median is the fairer
  speed figure. The stored `eval/results.md` is unchanged.

### Where the local model fails and Gemini passes
| id | Local answer | Why it fails |
|---|---|---|
| aapl-4 | "297 million shares" | The sentence gives both the share count and the dollar amount ($85.8 billion). The model picked the wrong number. Gemini gave both. |
| msft-4 | "approximately 12 million square feet" | The passage lists 14 million total, of which 12 million is owned HQ space. The model picked the part, not the total. |
| nvda-5 | "72" | The number is right, but the unchanged strict check needs "72 GPUs" / "72 Blackwell GPUs". It counts as wrong by the rules, though a human would accept it. |

Citations that failed verification (the answers themselves were correct):
- msft-3 quoted `"$35,562"`: too short (under 8 characters) to verify, and the Microsoft
  table puts `$` and the number on separate lines.
- tsla-1 quoted `"$94,827"` under chunk TSLA-2025-item8-028, which does not contain it.
- tsla-5 quoted `"$12,771 million"`: the word "million" is not next to the number in the source.

Each was flagged as unverified. Citation repair could not apply because none of the quotes
is a verbatim passage. Same as Gemini: nvda-1, tsla-3 and amzn-5 are wrong after retrieval
misses. The local model answered "Not found" for all three (Gemini answered tsla-3 with
"Texas", also counted wrong).

### Likely reasons
- **Model size:** 3.2B parameters at 4-bit against a hosted Flash-Lite model. The errors
  are about following instructions and reading closely. The model answers in very few words
  (median 2 vs 13 for Gemini), quotes numbers instead of the 5-to-30-word sentence the prompt
  asks for, and confuses related numbers in one passage.
- **Long context:** not the cause in the eval. All prompts fit in the 8,192-token window.
  In the agent, each search adds about 2,000 tokens, so several searches can approach the
  window; the client warns at 95%.
- **Tool calling:** not used by the eval (RAG mode, one call per question). Real agent
  runs: "How many full-time equivalent employees did Apple have?" worked with native tool
  calls (1 search, verified citation, 70 s) and with the JSON tool mode (verified citation,
  64 s, plus an unnecessary screener call). "What were Tesla's total revenues in 2025, and
  what does the screener say about TSLA?" **failed**: the model called both tools in one
  step, the screener was unreachable, and it answered "Not found" although the retrieved
  chunk TSLA-2025-item8-013 contains the revenue row. Gemini answered it in the M5 check.
- **Speed:** CPU only. About 34 s per question, mostly reading the ~2,000-token prompt.

### Known limits
- One model measured (`llama3.2:3b`, the only one installed). Larger local models were not tried.
- The agent was checked with 3 real runs, not a full agent evaluation.
- No GPU, no concurrency, no multi-user access control, no monitoring. Docker + Ollama is
  not set up or tested.
