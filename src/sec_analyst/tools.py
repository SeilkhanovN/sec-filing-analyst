"""The agent's tools. Each tool is a JSON-schema spec plus a Python function returning a dict.

Tools never raise: failures come back as {"error": "..."} so the model can read them and
decide what to do next (for example, answer from the filings alone).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .config import COMPANIES
from .llm import ToolSpec
from .retrieval import Retriever, normalize_ticker

MAX_TOOL_RESULT_CHARS = 4000


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., dict[str, Any]]


def make_search_tool(retriever: Retriever, k: int = 5) -> Tool:
    spec = ToolSpec(
        name="search_filings",
        description=(
            "Semantic search over the latest 10-K annual reports of "
            + ", ".join(f"{t} ({v['name']})" for t, v in COMPANIES.items())
            + ". Returns the most relevant passages with their chunk_id, company and section."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain English."},
                "ticker": {
                    "type": "string",
                    "description": "Optional: restrict the search to one company.",
                    "enum": list(COMPANIES),
                },
            },
            "required": ["query"],
        },
    )

    def search_filings(query: str = "", ticker: str | None = None) -> dict[str, Any]:
        try:
            hits = retriever.search(query, k=k, ticker=normalize_ticker(ticker))
        except ValueError as exc:
            return {"error": str(exc)}
        return {
            "results": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "company": h.chunk.company,
                    "ticker": h.chunk.ticker,
                    "fiscal_year": h.chunk.fiscal_year,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                }
                for h in hits
            ]
        }

    return Tool(spec, search_filings)


def make_screener_tool(
    base_url: str,
    path_template: str = "/api/screener/{ticker}",
    timeout_s: float = 5.0,
    session: Any = requests,
) -> Tool:
    spec = ToolSpec(
        name="get_screener_result",
        description=(
            "Get current market data and screening metrics for a stock ticker from the author's "
            "Stock Screener API (live data, not from the filings)."
        ),
        parameters={
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
            "required": ["ticker"],
        },
    )

    def get_screener_result(ticker: str = "") -> dict[str, Any]:
        symbol = str(ticker).strip().upper()
        if not symbol.isalnum() or len(symbol) > 10:
            return {"error": f"Invalid ticker {ticker!r}"}
        if not base_url:
            return {"error": "Stock screener is not configured (SCREENER_URL is empty); use the filings only."}
        url = base_url.rstrip("/") + path_template.format(ticker=symbol)
        try:
            resp = session.get(url, timeout=timeout_s, headers={"Accept": "application/json"})
        except requests.RequestException as exc:
            return {"error": f"Stock screener unreachable ({type(exc).__name__}); use the filings only."}
        if resp.status_code != 200:
            return {"error": f"Stock screener returned HTTP {resp.status_code} for {symbol}."}
        try:
            data = resp.json()
        except ValueError:
            return {"error": "Stock screener returned a response that is not JSON."}
        text = json.dumps(data)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            return {"ticker": symbol, "data_truncated": text[:MAX_TOOL_RESULT_CHARS]}
        return {"ticker": symbol, "data": data}

    return Tool(spec, get_screener_result)
