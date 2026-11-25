"""Command-line entry point: python -m sec_analyst.cli <command> --help."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .citations import format_citation
from .config import COMPANIES, load_settings


def _print_answer(answer, show_steps: bool = False) -> None:
    if show_steps:
        for s in answer.steps:
            if s["type"] == "tool":
                print(f"  step {s['step']}: {s['tool']}({json.dumps(s['args'])}) -> {s['summary'][:160]}")
            elif s["type"] == "error":
                print(f"  step {s['step']}: FAILED STEP -> {s['summary'][:160]}")
            else:
                print(f"  step {s['step']}: final answer")
    print("\nANSWER:\n" + answer.answer)
    if answer.citations:
        print("\nCITATIONS:")
        for i, c in enumerate(answer.citations, 1):
            print(format_citation(i, c))
    if answer.error:
        print(f"\nNOTE: {answer.error}")
    if answer.usage:
        print(f"\nusage: {answer.usage}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    parser = argparse.ArgumentParser(prog="python -m sec_analyst.cli", description="SEC Filing Analyst")
    parser.add_argument("--provider", default=None, choices=["gemini", "ollama", "fake"],
                        help="override LLM_PROVIDER for this command (default: gemini)")
    parser.add_argument("--model", default=None,
                        help="override the model of the active provider (LLM_MODEL or OLLAMA_MODEL)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="show which settings are configured (never shows secrets)")

    p_ingest = sub.add_parser("ingest", help="download, chunk, embed and index the latest 10-Ks")
    p_ingest.add_argument("--tickers", nargs="*", default=list(COMPANIES), help="default: all five companies")

    sub.add_parser("count", help="print the number of indexed chunks per company")

    p_ask = sub.add_parser("ask", help="answer a question with RAG (one LLM call)")
    p_ask.add_argument("question")
    p_ask.add_argument("--ticker", default=None)
    p_ask.add_argument("--k", type=int, default=None, help="number of passages to retrieve")

    p_agent = sub.add_parser("agent", help="answer a question with the tool-using agent")
    p_agent.add_argument("question")
    p_agent.add_argument("--ticker", default=None)
    p_agent.add_argument("--max-steps", type=int, default=6)

    sub.add_parser("list-models", help="list available models: Gemini models that support generateContent, "
                                       "or the models installed in local Ollama (no generation call)")

    args = parser.parse_args(argv)
    settings = load_settings()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.set_model(args.model)

    if args.command == "check-config":
        print(json.dumps(settings.describe(), indent=2))
        return 0

    from . import pipeline  # heavy imports only when needed

    if args.command == "ingest":
        from .edgar import EdgarClient
        from .ingest import ingest

        edgar = EdgarClient(settings.edgar_user_agent, settings.data_dir)
        store = pipeline.build_store(settings)
        counts = ingest([t.upper() for t in args.tickers], edgar, pipeline.build_embedder(settings), store, settings.data_dir)
        print(json.dumps(counts, indent=2))
        return 0

    if args.command == "count":
        counts = pipeline.build_store(settings).count_by_ticker()
        for ticker in COMPANIES:
            print(f"{ticker}: {counts.get(ticker, 0)} chunks")
        print(f"total: {sum(counts.values())} chunks")
        return 0 if counts else 1

    from .llm import LLMError, make_llm_client

    if args.command == "list-models":
        if settings.llm_provider == "ollama":
            try:
                print("\n".join(make_llm_client(settings).list_models()))
            except LLMError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            return 0
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print(m.name)
        return 0

    try:
        llm = make_llm_client(settings)
    except LLMError as exc:  # misconfiguration, e.g. a non-local OLLAMA_HOST or a missing key
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    retriever = pipeline.build_retriever(settings)
    print(f"[provider: {settings.llm_provider}, model: {llm.model}]")
    if args.command == "ask":
        from .rag import answer_question

        try:
            _print_answer(answer_question(args.question, retriever, llm, k=args.k, ticker=args.ticker))
        except LLMError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "agent":
        agent = pipeline.build_agent(settings, retriever, llm, max_steps=args.max_steps)
        _print_answer(agent.run(args.question, ticker=args.ticker), show_steps=True)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
