# Architecture decisions

Short decision records: the context, what was decided, the alternatives, and the trade-off.
Numbers refer to `eval/results.md` unless stated otherwise.

## ADR-1: Grounded answers with code-verified citations
- **Context:** an analyst must be able to check every number. LLMs can invent quotes.
- **Decision:** the model returns JSON (`found`, `answer`, `citations[{chunk_id, quote}]`) in
  JSON mode. Code checks that each quote occurs in the cited chunk (normalising case,
  whitespace, curly quotes and dashes; an ellipsis splits a quote into ordered parts). Invalid
  citations are shown as unverified, not dropped. If the model quotes real text under the
  wrong chunk id, the citation is re-attributed to the passage that contains it, and the
  model's id is kept in `corrected_from`.
- **Alternatives:** free-text answers with `[1]` markers (cannot be checked); an LLM judge
  (more calls, and itself fallible).
- **Trade-off:** this proves provenance (the quote exists), not entailment (the quote
  supports the claim). A faithfulness check is a candidate next step.

## ADR-2: Hybrid retrieval (vectors + BM25 with reciprocal rank fusion)
- **Context:** pure vector search missed 7 of 25 source chunks. All 5 wrong answers in v1
  followed a retrieval miss, and the misses were exact-wording facts.
- **Decision:** take the top 20 from Chroma (cosine similarity) and the top 20 from an
  in-memory BM25 index, fuse them with RRF (k=60), and keep the top 5. `RETRIEVAL_MODE=vector`
  switches back.
- **Alternatives:** a cross-encoder reranker (another model, more latency); a bigger embedding
  model (slower on CPU, re-indexing).
- **Trade-off:** BM25 is rebuilt in memory at start-up (~1,400 chunks, under a second). At
  millions of chunks it would belong in the database (for example a full-text index).
  Results: hit rate 72% to 80%, correctness 80% to 88%.

## ADR-3: Local embeddings (`BAAI/bge-small-en-v1.5`)
- **Decision:** a small sentence-transformers model on CPU, loaded offline once cached.
- **Why:** free, no API key, no data leaves the machine, fast enough for ~1,400 chunks, and
  512-token inputs fit 1,500-character chunks.
- **Trade-off:** weaker than large API embedding models; hybrid search makes up part of
  the difference.

## ADR-4: Chroma as the vector database
- **Decision:** an embedded, persistent Chroma collection (cosine, HNSW) with metadata
  filters by ticker. We always pass our own vectors.
- **Alternatives:** pgvector or Qdrant (need a server); FAISS (no metadata or persistence
  out of the box).
- **Trade-off:** a single process and modest scale. The `VectorStore` interface isolates a
  later move.

## ADR-5: Chunking by 10-K Item, 1,500 characters, 200 overlap, stable ids
- **Decision:** split by Item headings first (so every chunk knows its section), then pack
  paragraphs. Ids are `TICKER-FY-itemX-NNN`.
- **Why:** section labels make citations readable, and stable ids let the eval refer to chunks.
- **Trade-off:** changing the chunker changes ids and invalidates the eval set, which is why
  table-parsing improvements were deferred.

## ADR-6: Provider-agnostic LLM interface
- **Decision:** one method, `generate(messages, system, tools, json_output)`, with neutral
  `Message`, `ToolSpec`, `ToolCall` and `LLMResponse` types. Providers translate. Gemini's raw
  reply is kept opaque on assistant turns, so its "thought signatures" survive multi-step
  tool use.
- **Implemented:** `GeminiClient` (real, default), `OllamaClient` (real, local; see ADR-11)
  and `FakeClient` (tests and offline demo). `OpenAIClient` is a marked stub.
- **Trade-off:** provider-specific features (streaming, caching) are not exposed through the
  interface yet.

## ADR-7: Model and rate-limit policy
- **Decision:** default `gemini-3.5-flash-lite` with `LLM_THINKING_LEVEL=low`. Retry 429/5xx
  with exponential backoff and jitter (6 attempts, honouring server retry hints). Fail fast
  on daily-quota errors. The eval caps calls and caches replies.
- **Why:** on the free tier `gemini-3.8-flash` allows 20 requests/day per model and was often
  overloaded (503) during the build. Default thinking caused a 504 deadline once.
- **Trade-off:** Flash-Lite is the smaller model. `LLM_MODEL` switches models without code
  changes.

## ADR-8: A hand-written agent loop
- **Decision:** a ~100-line loop with two tools, `max_steps=6`, and tool errors returned as
  data.
- **Alternatives:** LangChain or LlamaIndex agents (faster to start, harder to explain and
  debug).
- **Trade-off:** no planning or memory features. Every step is visible in the API response
  and the UI.

## ADR-9: Treat retrieved text and tool output as untrusted
- **Decision:** passages sit inside `<context>` tags, delimiter look-alikes inside them are
  neutralised, the system prompts say the data must never be followed as instructions, the
  output format is constrained to JSON, and citations are verified. Tests cover each part.
- **Trade-off:** this reduces prompt-injection risk but cannot eliminate it for an LLM.

## ADR-10: Testing without the network
- **Decision:** all unit tests use fakes (LLM, embedder, store, HTTP screener server, and a fake
  sentence-transformers module). Real calls happen only in the eval and the manual checks.
- **Why:** CI runs with no secrets and costs no quota. Tests are deterministic and take about 6 seconds.

## ADR-11: Privately hosted LLM via Ollama
- **Context:** with Gemini, every question and the filing passages retrieved for it are sent
  to Google's API. That is acceptable for public 10-Ks, but not for confidential documents,
  and the free tier added quotas and 503 "overloaded" errors. The goal was a mode in which
  no data leaves the computer, without weakening any safety check.
- **Decision:** `OllamaClient` implements the same `LLMClient.generate()` against a local
  Ollama server (`POST /api/chat`, `stream=false`, temperature 0), using `requests`, which
  the project already depended on. `LLM_PROVIDER=ollama` selects it, and Gemini stays the
  default. Default model: `llama3.2:3b`, the model installed on the build machine.
  - **Localhost only:** `OLLAMA_HOST` must be a loopback address as written (no DNS lookup:
    `127.0.0.0/8`, `::1` or the name `localhost`). LAN addresses, `0.0.0.0`, other host names, URLs
    with credentials and non-HTTP schemes are refused before any request is sent, unless
    `ALLOW_REMOTE_LLM=1` is set (then a warning is logged). The HTTP session ignores proxy
    environment variables, so "local" traffic cannot be rerouted.
  - **Failure policy:** one attempt per call, no retries. Connect timeout 5 s, read timeout
    `OLLAMA_TIMEOUT_S` (180 s). "Not running", "model not installed" and "timed out" each map
    to a clear `LLMError`. A local server that is down will not recover by waiting, unlike a
    cloud API under load.
  - **Tool use:** native Ollama tool calling if `/api/show` lists the `tools` capability,
    otherwise a strict JSON tool-call format (`{"tool_call": {...}}` with Ollama's JSON mode).
    Every call is validated (known tool, object arguments, required arguments, no unknown
    ones). A malformed call becomes `LLMResponse.tool_error`, which the agent records as a
    failed step that counts towards `max_steps`.
  - **Same checks:** the same prompts and JSON answer format, code-verified citations with
    repair, the "Not found" refusal, delimiter neutralisation, and the agent's step limit.
    Nothing is relaxed for the local model.
- **Alternatives:** llama.cpp or vLLM directly (more setup; vLLM needs a GPU); the
  `ollama` Python package (a new dependency for one HTTP call); an OpenAI-compatible
  endpoint (Ollama offers one, but the native API reports token counts and capabilities).
- **Trade-offs (measured, 28 questions, 2026-11-25):** quality is lower. Correctness was
  76.0% vs 88.0%, and citation validity 86.4% (19/22) vs 100% (25/25). Refusals were 3/3 for
  both. The small model gives very short answers (median 2 words) and sometimes quotes a
  bare number instead of a sentence. Speed: 33.8 s per question on a laptop CPU (Intel Core
  Ultra 5 125U, no discrete GPU) vs 9.1 s mean / 1.3 s median for Gemini. Hardware: the 3B
  model at Q4 with an 8,192-token context uses 3.1 GB of RAM and 100% CPU. Context:
  `OLLAMA_NUM_CTX=8192` covers the RAG prompts (1,453 to 2,614 tokens). Ollama silently
  drops the oldest tokens beyond it, so the client logs a warning when a prompt reaches 95%
  of the window.
- **Not covered:** no GPU serving, batching or concurrency tuning (one Ollama process on one
  machine); no multi-user setup (no authentication, quotas, queue or TLS; the API is for one
  local user); no model monitoring (no latency or error dashboards, no drift or quality
  tracking beyond running `eval/run_eval.py` by hand); no pinning of the model by digest
  (a re-pulled tag can change); Docker + Ollama is not set up (a container cannot reach the
  host's loopback, and `host.docker.internal` counts as remote).
