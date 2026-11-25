"""End-to-end evaluation of the RAG pipeline on questions written from the real filings.

    python eval/run_eval.py                      # eval/questions.jsonl
    python eval/run_eval.py --extra              # also eval/questions_extra.jsonl

Metrics (written to eval/results.md and eval/results.json):
  (a) retrieval hit rate   - the question's source chunk is among the top-k retrieved chunks
                             (no LLM involved; ticker filter NOT used, so this is the hard case)
  (b) answer correctness   - the expected short fact (or an accepted alias) appears in the
                             answer; for unanswerable questions, the system must say "not found"
  (c) citation validity    - share of returned citations whose quote really appears in the
                             cited chunk (checked against the stored chunk text)

At most --max-llm-calls real LLM calls are made (default 40). LLM replies are cached in
data/eval_llm_cache.json, keyed by model and full prompt, so re-running the evaluation after
a quota error does not spend calls again; the report says how many answers came from cache.

Providers (LLM_PROVIDER or --provider): gemini (default) writes eval/results.md and uses
data/eval_llm_cache.json, exactly as before. Any other provider gets its own files
(eval/results_<provider>.md, data/eval_llm_cache_<provider>.json), so a local-model run
never overwrites the Gemini results. For example:

    python eval/run_eval.py --provider ollama    # -> eval/results_ollama.md

Speed: each question's seconds (retrieval + LLM call, excluding the free-tier pause between
calls) is recorded; the average counts only questions answered by a live call, not the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sec_analyst import pipeline  # noqa: E402
from sec_analyst.config import load_settings  # noqa: E402
from sec_analyst.llm import LLMError, LLMResponse, make_llm_client  # noqa: E402
from sec_analyst.rag import answer_question  # noqa: E402


def normalize_fact(text: str) -> str:
    text = text.lower().replace("’", "'").replace("$", "").replace(",", "")
    return re.sub(r"\s+", " ", text).strip()


def fact_matches(answer: str, expected: list[str]) -> bool:
    norm = normalize_fact(answer)
    return any(normalize_fact(e) in norm for e in expected if e)


@dataclass
class CachingLLM:
    """Wraps an LLM client: identical (model, system, prompt) requests are answered from a
    JSON cache file; at most `budget` live calls are allowed."""

    inner: object
    cache_path: Path
    budget: int
    delay_s: float
    use_cache: bool = True
    live_calls: int = 0
    cached_calls: int = 0
    quota_exhausted: bool = False
    slept_s: float = 0.0  # time spent in rate-limit pauses (excluded from per-question timing)

    def __post_init__(self):
        self.model = getattr(self.inner, "model", "unknown")
        self._cache = {}
        # Always load existing entries, even with --no-cache, so writing new replies does not
        # wipe the cache file; --no-cache only stops replies being *read* from it.
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def generate(self, messages, system=None, tools=None, json_output=False):
        key_src = json.dumps([self.model, system, [m.text for m in messages], json_output])
        key = hashlib.sha256(key_src.encode()).hexdigest()
        if self.use_cache and key in self._cache:
            self.cached_calls += 1
            return LLMResponse(text=self._cache[key]["text"], usage=self._cache[key].get("usage", {}))
        if self.live_calls >= self.budget:
            raise LLMError("LLM call budget for this evaluation run is used up")
        if self.quota_exhausted:
            raise LLMError("not run: the provider's daily quota was exhausted earlier in this run")
        if self.live_calls and self.delay_s > 0:
            time.sleep(self.delay_s)  # stay under the free tier's requests-per-minute limit
            self.slept_s += self.delay_s
        self.live_calls += 1
        try:
            response = self.inner.generate(messages, system=system, tools=tools, json_output=json_output)
        except LLMError as exc:
            if re.search(r"PerDay|RESOURCE_EXHAUSTED", str(exc)):
                self.quota_exhausted = True  # do not waste further requests today
            raise
        self._cache[key] = {"text": response.text, "usage": response.usage}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=1), encoding="utf-8")
        return response


def load_questions(paths: list[Path]) -> list[dict]:
    questions = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            questions.extend(json.loads(line) for line in f if line.strip())
    return questions


def evaluate(questions: list[dict], retriever, llm, k: int) -> list[dict]:
    rows = []
    for q in questions:
        answerable = q.get("answerable", True)
        hits = retriever.search(q["question"], k=k)
        retrieved = [h.chunk.chunk_id for h in hits]
        # "also_in": overlapping chunks that contain the same fact verbatim (chunk overlap).
        sources = [q["source_chunk_id"]] + q.get("also_in", [])
        ranks = [retrieved.index(s) + 1 for s in sources if s in retrieved]
        row = {
            "id": q["id"],
            "ticker": q.get("ticker", ""),
            "answerable": answerable,
            "retrieval_hit": bool(ranks) if answerable else None,
            "source_rank": min(ranks) if ranks else None,
        }
        cached_before, slept_before = getattr(llm, "cached_calls", 0), getattr(llm, "slept_s", 0.0)
        started = time.perf_counter()
        try:
            answer = answer_question(q["question"], retriever, llm, k=k)
        except LLMError as exc:
            row.update({"status": "llm_error", "error": str(exc)[:200]})
            rows.append(row)
            print(f"{q['id']}: LLM error: {str(exc)[:120]}")
            continue
        seconds = time.perf_counter() - started - (getattr(llm, "slept_s", 0.0) - slept_before)
        expected = [q.get("expected_fact", "")] + q.get("aliases", [])
        correct = fact_matches(answer.answer, expected) if answerable else not answer.found
        row.update({
            "status": "ok",
            "correct": correct,
            "found": answer.found,
            "citations_total": len(answer.citations),
            "citations_valid": sum(c.valid for c in answer.citations),
            "answer": answer.answer,
            "seconds": round(seconds, 2),
            "from_cache": getattr(llm, "cached_calls", 0) > cached_before,
        })
        rows.append(row)
        print(f"{q['id']}: hit={row['retrieval_hit']} correct={correct} "
              f"citations={row['citations_valid']}/{row['citations_total']} {seconds:.1f}s"
              + (" (cache)" if row["from_cache"] else ""), flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    answered = [r for r in rows if r.get("status") == "ok"]
    cit_total = sum(r["citations_total"] for r in answered)
    cit_valid = sum(r["citations_valid"] for r in answered)
    unans = [r for r in answered if not r["answerable"]]
    timed = [r["seconds"] for r in answered if "seconds" in r and not r.get("from_cache")]
    return {
        "questions": len(rows),
        "retrieval_hit_rate": sum(bool(r["retrieval_hit"]) for r in answerable) / max(1, len(answerable)),
        "answer_correctness": sum(r["correct"] for r in answered) / max(1, len(answered)),
        "answer_correctness_answerable": (
            sum(r["correct"] for r in answered if r["answerable"]) / max(1, len([r for r in answered if r["answerable"]]))
        ),
        "unanswerable_correctly_refused": f"{sum(r['correct'] for r in unans)}/{len(unans)}",
        "citation_validity": cit_valid / cit_total if cit_total else 0.0,
        "citations": f"{cit_valid}/{cit_total}",
        "answered": len(answered),
        "llm_errors": len(rows) - len(answered),
        # Timing does not affect any score. None when every answer came from the cache.
        "avg_seconds_per_question": round(sum(timed) / len(timed), 2) if timed else None,
        "timed_questions": len(timed),
    }


def write_report(rows: list[dict], summary: dict, meta: dict, out_md: Path) -> None:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    avg = summary.get("avg_seconds_per_question")
    speed = f"{avg:.1f} s ({summary['timed_questions']} live calls)" if avg is not None else "not measured (all answers from cache)"
    lines = [
        "# Evaluation results",
        "",
        f"Generated by `eval/run_eval.py` on {meta['date']}. Provider: `{meta.get('provider', 'gemini')}`. Model: `{meta['model']}`. "
        f"Embeddings: `{meta['embedding_model']}`. Top-k: {meta['k']}. "
        f"Live LLM calls: {meta['live_calls']}, answers from cache: {meta['cached_calls']}.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| (a) Retrieval hit rate (source chunk in top-{meta['k']}, answerable questions) | {pct(summary['retrieval_hit_rate'])} |",
        f"| (b) Answer correctness (all answered questions) | {pct(summary['answer_correctness'])} |",
        f"| (b) Answer correctness (answerable questions only) | {pct(summary['answer_correctness_answerable'])} |",
        f"| (b) Unanswerable questions correctly refused | {summary['unanswerable_correctly_refused']} |",
        f"| (c) Citation validity (quote found in cited chunk) | {pct(summary['citation_validity'])} ({summary['citations']}) |",
        f"| Questions / answered / LLM errors | {summary['questions']} / {summary['answered']} / {summary['llm_errors']} |",
        f"| Average seconds per question (retrieval + LLM; excludes rate-limit pauses) | {speed} |",
        "",
        "Answer correctness is a strict string check: the expected fact (or a listed alias) must",
        "appear in the answer after lower-casing and removing `$` and commas. It can mark a",
        "correct answer wrong if the model phrases a number differently (for example \"1.6 million\"",
        "instead of \"1,576,000\"). Retrieval hit counts only the listed source chunk(s), so it can",
        "miss when a different chunk also contains the fact (see rows marked hit=no, correct=yes).",
        "",
        "## Per question",
        "",
        "| id | ticker | retrieval hit (rank) | correct | valid citations | seconds | answer (truncated) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        hit = "n/a" if r["retrieval_hit"] is None else ("yes" if r["retrieval_hit"] else "no")
        if r.get("source_rank"):
            hit += f" ({r['source_rank']})"
        if r.get("status") != "ok":
            lines.append(f"| {r['id']} | {r['ticker']} | {hit} | error | - | - | {r.get('error', '')[:80]} |")
            continue
        answer = r["answer"].replace("|", "/").replace("\n", " ")[:140]
        seconds = "cache" if r.get("from_cache") else (f"{r['seconds']:.1f}" if "seconds" in r else "-")
        lines.append(
            f"| {r['id']} | {r['ticker']} | {hit} | {'yes' if r['correct'] else 'no'} | "
            f"{r['citations_valid']}/{r['citations_total']} | {seconds} | {answer} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extra", action="store_true", help="also run eval/questions_extra.jsonl")
    parser.add_argument("--max-llm-calls", type=int, default=40)
    parser.add_argument("--delay", type=float, default=None,
                        help="seconds between live LLM calls (default: 6 for gemini's free tier, 0 otherwise)")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="do not read cached replies (new replies are still saved)")
    parser.add_argument("--provider", default=None, choices=["gemini", "ollama", "fake"], help="override LLM_PROVIDER")
    parser.add_argument("--model", default=None, help="override the active provider's model for this run")
    parser.add_argument("--out", default=None, help="default: eval/results.md (gemini) or eval/results_<provider>.md")
    parser.add_argument("--cache", default=None,
                        help="default: data/eval_llm_cache.json (gemini) or data/eval_llm_cache_<provider>.json")
    args = parser.parse_args()

    settings = load_settings()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.set_model(args.model)
    provider = settings.llm_provider
    suffix = "" if provider == "gemini" else f"_{provider}"
    out_md = Path(args.out) if args.out else ROOT / "eval" / f"results{suffix}.md"
    cache_path = Path(args.cache) if args.cache else settings.data_dir / f"eval_llm_cache{suffix}.json"
    delay = args.delay if args.delay is not None else (6.0 if provider == "gemini" else 0.0)
    k = args.k or settings.top_k
    paths = [ROOT / "eval" / "questions.jsonl"]
    if args.extra:
        paths.append(ROOT / "eval" / "questions_extra.jsonl")
    questions = load_questions(paths)

    retriever = pipeline.build_retriever(settings)
    llm = CachingLLM(make_llm_client(settings), cache_path,
                     budget=args.max_llm_calls, delay_s=delay, use_cache=not args.no_cache)
    print(f"provider: {provider}, model: {llm.model}, cache: {cache_path.name}, report: {out_md}", flush=True)
    rows = evaluate(questions, retriever, llm, k)
    summary = summarize(rows)
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "provider": provider,
        "model": llm.model,
        "embedding_model": settings.embedding_model,
        "k": k,
        "live_calls": llm.live_calls,
        "cached_calls": llm.cached_calls,
    }
    write_report(rows, summary, meta, out_md)
    out_md.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"live LLM calls: {llm.live_calls}, cached: {llm.cached_calls}; report: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
