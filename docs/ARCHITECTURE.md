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
- **Implemented:** `GeminiClient` (real) and `FakeClient` (tests and offline demo).
  `OpenAIClient` is a marked stub.
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
