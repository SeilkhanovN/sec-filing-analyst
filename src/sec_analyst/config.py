"""Settings, read from environment variables (and a local .env file, if present).

Secrets are never printed: `describe()` only reports whether each value is set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The five companies this project indexes. CIKs are SEC's permanent company IDs.
COMPANIES: dict[str, dict[str, str]] = {
    "AAPL": {"cik": "320193", "name": "Apple Inc."},
    "MSFT": {"cik": "789019", "name": "Microsoft Corporation"},
    "NVDA": {"cik": "1045810", "name": "NVIDIA Corporation"},
    "TSLA": {"cik": "1318605", "name": "Tesla, Inc."},
    "AMZN": {"cik": "1018724", "name": "Amazon.com, Inc."},
}

DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Local LLM through Ollama (LLM_PROVIDER=ollama). The host must be on this machine unless
# ALLOW_REMOTE_LLM=1 is set explicitly (see ollama_client.check_local_host).
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _flag(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    gemini_api_key: str = ""
    llm_provider: str = "gemini"
    llm_model: str = DEFAULT_LLM_MODEL
    thinking_level: str = "low"
    edgar_user_agent: str = ""
    screener_url: str = ""
    screener_path: str = "/api/screener/{ticker}"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "chroma_db")
    top_k: int = 5
    retrieval_mode: str = "hybrid"
    ollama_host: str = DEFAULT_OLLAMA_HOST
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout_s: float = 180.0
    ollama_num_ctx: int = 8192
    ollama_tool_mode: str = "auto"
    allow_remote_llm: bool = False

    def set_model(self, model: str) -> None:
        """Apply a --model override to the active provider (LLM_MODEL for Gemini, OLLAMA_MODEL for Ollama)."""
        if self.llm_provider == "ollama":
            self.ollama_model = model
        else:
            self.llm_model = model

    def describe(self) -> dict[str, str]:
        """Safe summary for logs: secrets are reported only as set / not set."""
        return {
            "GEMINI_API_KEY": "set" if self.gemini_api_key else "NOT SET",
            "EDGAR_USER_AGENT": "set" if self.edgar_user_agent else "NOT SET",
            "SCREENER_URL": "set" if self.screener_url else "NOT SET",
            "LLM_PROVIDER": self.llm_provider,
            "LLM_MODEL": self.llm_model,
            "LLM_THINKING_LEVEL": self.thinking_level or "(model default)",
            "OLLAMA_HOST": self.ollama_host,
            "OLLAMA_MODEL": self.ollama_model,
            "OLLAMA_TOOL_MODE": self.ollama_tool_mode,
            "ALLOW_REMOTE_LLM": "yes" if self.allow_remote_llm else "no (local Ollama only)",
            "EMBEDDING_MODEL": self.embedding_model,
            "RETRIEVAL_MODE": self.retrieval_mode,
            "DATA_DIR": str(self.data_dir),
            "CHROMA_DIR": str(self.chroma_dir),
        }


def load_settings(env_file: Path | None = None) -> Settings:
    """Load settings. A real environment variable wins over the .env file."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    data_dir = _env("DATA_DIR")
    chroma_dir = _env("CHROMA_DIR")
    return Settings(
        gemini_api_key=_env("GEMINI_API_KEY"),
        llm_provider=_env("LLM_PROVIDER", "gemini").lower(),
        llm_model=_env("LLM_MODEL", DEFAULT_LLM_MODEL),
        thinking_level=_env("LLM_THINKING_LEVEL", "low"),
        edgar_user_agent=_env("EDGAR_USER_AGENT"),
        screener_url=_env("SCREENER_URL").rstrip("/"),
        screener_path=_env("SCREENER_PATH", "/api/screener/{ticker}"),
        embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        data_dir=Path(data_dir) if data_dir else PROJECT_ROOT / "data",
        chroma_dir=Path(chroma_dir) if chroma_dir else PROJECT_ROOT / "chroma_db",
        top_k=int(_env("TOP_K", "5")),
        retrieval_mode=_env("RETRIEVAL_MODE", "hybrid").lower(),
        ollama_host=_env("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST,
        ollama_model=_env("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
        ollama_timeout_s=float(_env("OLLAMA_TIMEOUT_S", "180")),
        ollama_num_ctx=int(_env("OLLAMA_NUM_CTX", "8192")),
        ollama_tool_mode=_env("OLLAMA_TOOL_MODE", "auto").lower(),
        allow_remote_llm=_flag("ALLOW_REMOTE_LLM"),
    )
