# Candidate improvements (critic pass)

Written after tagging `v1-working`. I reread the repository as a hiring manager for an AI
Engineer role (LLM, RAG, semantic search, agents, provider-agnostic design, Docker/CI,
vector DBs, clear communication). Every weakness below cites evidence from this repository.
Ranked by benefit / (risk x effort), where S/M/L = 1/2/3 and low/medium/high = 1/2/3.

| Rank | Improvement | Benefit | Risk | Effort | Status |
|---|---|---|---|---|---|
| 1 | Hybrid retrieval (BM25 + vectors, reciprocal rank fusion) | high (3) | low | M | see below |
| 2 | Prompt-injection hardening + tests | medium (2) | low | S | see below |
| 3 | Load the embedding model offline once cached | medium (2) | low | S | see below |
| 4 | Eval regression check against a stored baseline | medium (2) | low | S | see below |
| 5 | Architecture decision record (docs/ARCHITECTURE.md) | medium (2) | low | S | see below |
| 6 | Lint (ruff) in CI | low-medium | low | S | not done |
| 7 | Answer cache in the API | medium | low | M | not done |
| 8 | More eval questions, incl. unanswerable (questions_extra.jsonl) | medium | low | M | not done |
| 9 | Faithfulness check (does the quote support the claim?) | medium | medium | M | not done |
| 10 | Second LLM provider implemented against fakes | low-medium | low | M | not done |
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
