"""FastAPI backend: POST /ask, GET /health, GET /filings, and the static demo page.

Run locally (binds to 127.0.0.1 only):
    uvicorn sec_analyst.api:app --host 127.0.0.1 --port 8000
or  python -m sec_analyst.api
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent import LLM_UNAVAILABLE
from .config import COMPANIES, Settings, load_settings
from .llm import LLMClient, LLMError
from .rag import Answer, answer_question
from .retrieval import Retriever, normalize_ticker

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 500


@dataclass
class Services:
    """Everything a request needs. Tests build this with fakes; production builds it lazily."""

    settings: Settings
    retriever: Retriever
    llm: LLMClient
    max_agent_steps: int = 6

    def agent(self):
        from .pipeline import build_agent

        return build_agent(self.settings, self.retriever, self.llm, max_steps=self.max_agent_steps)


class AskRequest(BaseModel):
    question: str = Field(..., description="Plain-English question", max_length=MAX_QUESTION_CHARS)
    ticker: str | None = Field(None, description="Optional company filter: AAPL, MSFT, NVDA, TSLA or AMZN")
    mode: Literal["agent", "rag"] = Field("agent", description="agent = tool-using loop, rag = one retrieval + one LLM call")

    @field_validator("question")
    @classmethod
    def _question(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("question must be at least 3 characters")
        return value

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, value: str | None) -> str | None:
        return normalize_ticker(value)  # raises ValueError -> HTTP 422


def _citation_payload(answer: Answer, retriever: Retriever) -> list[dict]:
    """Citations plus the full source passage, so the page can show the quote in context."""
    items = []
    for c in answer.citations:
        chunk = retriever.store.get(c.chunk_id)
        item = c.to_dict()
        item["passage"] = chunk.text if chunk else ""
        items.append(item)
    return items


def _filing_years(settings: Settings) -> dict[str, dict]:
    info = {}
    for ticker in COMPANIES:
        path = settings.data_dir / "raw" / ticker / "filing.json"
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            info[ticker] = {"fiscal_year": int(meta["report_date"][:4]), "filing_date": meta["filing_date"],
                            "url": meta["url"]}
    return info


def create_app(services: Services | None = None) -> FastAPI:
    app = FastAPI(title="SEC Filing Analyst", version="0.1.0",
                  description="Grounded Q&A over SEC 10-K filings with verified citations.")
    lock = threading.Lock()
    state: dict[str, Services | None] = {"services": services}

    def get_services() -> Services:
        with lock:
            if state["services"] is None:
                from .llm import make_llm_client
                from .pipeline import build_retriever

                settings = load_settings()
                state["services"] = Services(settings, build_retriever(settings), make_llm_client(settings))
            return state["services"]

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):  # never leak internals to the client
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/health")
    def health() -> dict:
        svc = get_services()
        counts = svc.retriever.store.count_by_ticker()
        return {
            "status": "ok" if counts else "no_index",
            "indexed_chunks": sum(counts.values()),
            "llm_provider": svc.settings.llm_provider,
            "model": getattr(svc.llm, "model", "unknown"),
        }

    @app.get("/filings")
    def filings() -> list[dict]:
        svc = get_services()
        counts = svc.retriever.store.count_by_ticker()
        years = _filing_years(svc.settings)
        return [
            {"ticker": t, "company": v["name"], "chunks": counts.get(t, 0), **years.get(t, {})}
            for t, v in COMPANIES.items()
            if counts.get(t)
        ]

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        svc = get_services()
        started = time.perf_counter()
        try:
            if req.mode == "rag":
                answer = answer_question(req.question, svc.retriever, svc.llm, ticker=req.ticker)
            else:
                answer = svc.agent().run(req.question, ticker=req.ticker)
        except LLMError as exc:
            log.warning("LLM error: %s", exc)
            raise HTTPException(status_code=503, detail="The language model is unavailable right now (rate limit or outage). Please try again in a minute.")
        if answer.answer == LLM_UNAVAILABLE:  # the agent turns LLMError into this answer
            raise HTTPException(status_code=503, detail="The language model is unavailable right now (rate limit or outage). Please try again in a minute.")
        latency_ms = int((time.perf_counter() - started) * 1000)
        log.info("ask mode=%s ticker=%s latency_ms=%d usage=%s", req.mode, req.ticker, latency_ms, answer.usage)
        return {
            "question": answer.question,
            "answer": answer.answer,
            "found": answer.found,
            "grounded": answer.grounded,
            "citations": _citation_payload(answer, svc.retriever),
            "steps": answer.steps,
            "mode": req.mode,
            "model": getattr(svc.llm, "model", "unknown"),
            "usage": answer.usage,
            "latency_ms": latency_ms,
            "note": answer.error,
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # 127.0.0.1 by default; Docker sets API_HOST=0.0.0.0 inside the container.
    uvicorn.run(app, host=os.environ.get("API_HOST", "127.0.0.1"), port=int(os.environ.get("API_PORT", "8000")))


if __name__ == "__main__":
    main()
