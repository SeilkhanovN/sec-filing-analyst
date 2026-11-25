| Metric | gemini-v2-stored | gemini-rerun | ollama |
|---|---|---|---|
| Model | `gemini-3.5-flash-lite` (2026-11-18 02:18) | `gemini-3.5-flash-lite` (2026-11-25 19:14) | `llama3.2:3b` (2026-11-25 19:30) |
| Retrieval hit rate (source chunk in top-k) | 80.0% (20/25) | 80.0% (20/25) | 80.0% (20/25) |
| Answer correctness (answerable) | 88.0% (22/25) | 88.0% (22/25) | 76.0% (19/25) |
| Unanswerable correctly refused | 3/3 | 3/3 | 3/3 |
| Citation validity (quote in cited chunk) | 100.0% (25/25) | 100.0% (25/25) | 86.4% (19/22) |
| Average seconds per question | not recorded | 9.1 s | 33.8 s |
| LLM errors | 0 | 0 | 0 |

### gemini-rerun vs gemini-v2-stored: questions with a different result

No differences.

### ollama vs gemini-v2-stored: questions with a different result

| id | ollama result | valid citations | ollama answer (truncated) |
|---|---|---|---|
| aapl-4 | wrong (was correct with gemini-v2-stored) | 1/1 | 297 million shares |
| msft-3 | correct (was correct with gemini-v2-stored) | 0/1 | $35,562 million |
| msft-4 | wrong (was correct with gemini-v2-stored) | 1/1 | approximately 12 million square feet |
| nvda-5 | wrong (was correct with gemini-v2-stored) | 1/1 | 72 |
| tsla-1 | correct (was correct with gemini-v2-stored) | 0/1 | $94,827 million |
| tsla-5 | correct (was correct with gemini-v2-stored) | 0/1 | $12,771 million |
