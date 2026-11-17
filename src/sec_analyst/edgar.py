"""Download the latest 10-K for a company from SEC EDGAR, politely.

SEC rules for automated access: identify yourself in the User-Agent header and stay under
10 requests per second. This client sends EDGAR_USER_AGENT on every request, waits at
least `min_interval_s` between requests, and backs off on HTTP 429/5xx. Downloads are cached
in data/raw/, so re-running ingestion does not hit EDGAR again.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .config import COMPANIES
from .llm import call_with_retry

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


@dataclass
class FilingMeta:
    ticker: str
    company: str
    cik: str
    form: str
    accession_number: str
    filing_date: str
    report_date: str  # end of the fiscal year the report covers
    primary_document: str
    url: str

    @property
    def fiscal_year(self) -> int:
        return int(self.report_date[:4])


class HTTPStatusError(RuntimeError):
    def __init__(self, status: int, url: str):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path, min_interval_s: float = 0.25, session=None, sleep=time.sleep):
        if not user_agent.strip():
            raise ValueError("EDGAR_USER_AGENT must be set to 'Your Name your.email@example.com' (SEC requirement)")
        self.cache_dir = cache_dir
        self.min_interval_s = min_interval_s
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self._sleep = sleep
        self._last_request = 0.0

    def _get(self, url: str) -> requests.Response:
        def attempt() -> requests.Response:
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
            self._last_request = time.monotonic()
            resp = self.session.get(url, timeout=60)
            if resp.status_code != 200:
                raise HTTPStatusError(resp.status_code, url)
            return resp

        def retryable(exc: Exception) -> bool:
            if isinstance(exc, HTTPStatusError):
                return exc.status in {429, 500, 502, 503, 504}
            return isinstance(exc, requests.RequestException)

        return call_with_retry(attempt, retryable, max_attempts=5, base_delay=2.0, sleep=self._sleep)

    def latest_10k(self, ticker: str) -> FilingMeta:
        ticker = ticker.upper()
        company = COMPANIES[ticker]
        data = self._get(SUBMISSIONS_URL.format(cik=company["cik"])).json()
        recent = data["filings"]["recent"]
        for i, form in enumerate(recent["form"]):
            if form == "10-K":
                accession = recent["accessionNumber"][i]
                document = recent["primaryDocument"][i]
                return FilingMeta(
                    ticker=ticker,
                    company=company["name"],
                    cik=company["cik"],
                    form=form,
                    accession_number=accession,
                    filing_date=recent["filingDate"][i],
                    report_date=recent["reportDate"][i],
                    primary_document=document,
                    url=ARCHIVE_URL.format(cik=company["cik"], accession=accession.replace("-", ""), document=document),
                )
        raise LookupError(f"No 10-K found in the recent filings of {ticker}")

    def download_latest_10k(self, ticker: str) -> tuple[FilingMeta, Path]:
        """Return (metadata, path to the cached HTML). Downloads only if not cached."""
        folder = self.cache_dir / "raw" / ticker.upper()
        meta_path = folder / "filing.json"
        if meta_path.exists():
            meta = FilingMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
            html_path = folder / meta.primary_document
            if html_path.exists():
                return meta, html_path
        meta = self.latest_10k(ticker)
        folder.mkdir(parents=True, exist_ok=True)
        html_path = folder / meta.primary_document
        html_path.write_bytes(self._get(meta.url).content)
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        return meta, html_path
