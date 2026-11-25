"""Provider-agnostic LLM interface.

The rest of the code only talks to `LLMClient.generate()` using the small neutral types
below (Message, ToolSpec, ToolCall, LLMResponse). Each provider translates them to its own
SDK format. Implementations:

- GeminiClient: real Google Gemini calls through the google-genai SDK (the default).
- OllamaClient: a privately hosted model through a local Ollama server (ollama_client.py).
- FakeClient:   scripted responses for tests and offline demos (no network).
- OpenAIClient: STUB ONLY. Not implemented, not tested.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)

@dataclass
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class Message:
    """One conversation turn.

    role is "user", "assistant" or "tool". Assistant turns may carry tool_calls; tool turns
    carry the result of one call. `raw` holds the provider's original object for that turn
    (opaque to everything except the provider that created it). Gemini needs this to echo
    back "thought signatures" on multi-step tool use.
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None
    raw: Any = None


@dataclass
class LLMResponse:
    """The model's turn. `tool_error` is set (and `tool_calls` left empty) when the model
    tried to call a tool but the call failed validation (unknown tool, arguments that are
    not a JSON object, missing required arguments). The agent then records a failed step
    instead of crashing. Gemini never sets it: its SDK returns structured calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None
    tool_error: str | None = None


class LLMError(RuntimeError):
    """Raised when the provider fails permanently (after retries) or is misconfigured."""


class LLMClient(Protocol):
    model: str

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_output: bool = False,
    ) -> LLMResponse:
        """Return the model's next turn: either text or one or more tool calls."""
        ...


# --------------------------------------------------------------------------- retries


def call_with_retry(
    fn: Callable[[], Any],
    is_retryable: Callable[[Exception], bool],
    max_attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    retry_after: Callable[[Exception], float | None] = lambda e: None,
    on_retry: Callable[[Exception, float, int], None] | None = None,
) -> Any:
    """Call fn(); on a retryable error wait with exponential backoff and try again.

    Gives up after `max_attempts` calls and re-raises the last error, so a failing
    provider can never cause an endless loop.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable errors
            if attempt == max_attempts or not is_retryable(exc):
                raise
            hinted = retry_after(exc)
            delay = hinted if hinted is not None else base_delay * 2 ** (attempt - 1)
            delay = min(max_delay, delay) + random.uniform(0, 0.5)
            if on_retry is not None:
                on_retry(exc, delay, attempt)
            sleep(delay)
    raise AssertionError("unreachable")


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating ```json fences and extra prose."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- fake


class FakeClient:
    """Deterministic stand-in for a real LLM.

    Pass a list of scripted LLMResponse objects (returned in order), or a function that
    receives (messages, system, tools) and returns an LLMResponse. Every call is recorded
    in `self.calls` so tests can assert on what the model was sent.
    """

    model = "fake"

    def __init__(
        self,
        script: list[LLMResponse] | Callable[..., LLMResponse] | None = None,
    ) -> None:
        self.script = script if script is not None else []
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages, system=None, tools=None, json_output=False) -> LLMResponse:
        self.calls.append(
            {"messages": list(messages), "system": system, "tools": tools, "json_output": json_output}
        )
        if callable(self.script):
            return self.script(messages, system, tools)
        if not self.script:
            raise LLMError("FakeClient script exhausted")
        return self.script.pop(0)


def echo_context_fake(messages, system, tools) -> LLMResponse:
    """A FakeClient behaviour for offline demos: quote the first retrieved chunk.

    It never invents facts: it copies a sentence from the context it was given, so the
    citation pipeline can be demonstrated without an API key.
    """
    prompt = messages[-1].text if messages else ""
    match = re.search(r"\[chunk_id: ([^\]]+)\][^\n]*\n(.+?)(?:\n\[chunk_id:|\Z)", prompt, re.S)
    if not match:
        return LLMResponse(text=json.dumps({"found": False, "answer": "", "citations": []}))
    chunk_id, body = match.group(1).strip(), match.group(2).strip()
    sentence = re.split(r"(?<=[.!?])\s+", body)[0][:200]
    payload = {
        "found": True,
        "answer": f"(offline fake model) The most relevant passage says: {sentence}",
        "citations": [{"chunk_id": chunk_id, "quote": sentence}],
    }
    return LLMResponse(text=json.dumps(payload))


# --------------------------------------------------------------------------- gemini


class GeminiClient:
    """Google Gemini through the official google-genai SDK, with retry on 429/5xx."""

    RETRYABLE_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        model: str,
        max_attempts: int = 6,
        timeout_s: float = 120.0,
        thinking_level: str = "low",
    ):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai  # imported lazily so tests never need the SDK
        from google.genai import types

        self._types = types
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        self.model = model
        self.max_attempts = max_attempts
        # Gemini 3 models "think" before answering. Extracting facts from given passages
        # needs little reasoning; "low" cuts latency and avoids server deadline errors.
        # Empty string = model default (needed for older models without thinking levels).
        self.thinking_level = thinking_level.strip().upper()
        self.calls_made = 0

    # -- error classification --------------------------------------------------
    @staticmethod
    def _code(exc: Exception) -> int | None:
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    def _is_retryable(self, exc: Exception) -> bool:
        code = self._code(exc)
        if code in self.RETRYABLE_CODES:
            # A daily quota will not recover by waiting a minute; fail fast instead.
            return not _is_daily_quota(exc)
        return isinstance(exc, (TimeoutError, ConnectionError))

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        match = re.search(r"retry(?:Delay)?['\"]?\s*[:in ]+\s*['\"]?(\d+(?:\.\d+)?)s", str(exc), re.I)
        return float(match.group(1)) + 1.0 if match else None

    # -- message conversion ----------------------------------------------------
    def _to_contents(self, messages: list[Message]) -> list[Any]:
        t = self._types
        contents = []
        for m in messages:
            if m.role == "assistant" and m.raw is not None:
                contents.append(m.raw)  # keeps thought signatures intact
            elif m.role == "assistant":
                parts = [t.Part.from_text(text=m.text)] if m.text else []
                parts += [t.Part.from_function_call(name=c.name, args=c.args) for c in m.tool_calls]
                contents.append(t.Content(role="model", parts=parts))
            elif m.role == "tool":
                # The Gemini API expects function results in a "user" turn; results of
                # parallel calls from one model turn go together in a single turn.
                part = t.Part.from_function_response(name=m.tool_name or "tool", response=m.tool_result or {})
                previous = contents[-1] if contents else None
                if previous is not None and previous.role == "user" and all(
                    p.function_response is not None for p in previous.parts or []
                ):
                    previous.parts.append(part)
                else:
                    contents.append(t.Content(role="user", parts=[part]))
            else:
                contents.append(t.Content(role="user", parts=[t.Part.from_text(text=m.text)]))
        return contents

    def generate(self, messages, system=None, tools=None, json_output=False) -> LLMResponse:
        t = self._types
        config_kwargs: dict[str, Any] = {"temperature": 0.0}
        if self.thinking_level:
            config_kwargs["thinking_config"] = t.ThinkingConfig(thinking_level=self.thinking_level)
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = [
                t.Tool(
                    function_declarations=[
                        t.FunctionDeclaration(
                            name=s.name, description=s.description, parameters_json_schema=s.parameters
                        )
                        for s in tools
                    ]
                )
            ]
            config_kwargs["automatic_function_calling"] = t.AutomaticFunctionCallingConfig(disable=True)
        if json_output:
            config_kwargs["response_mime_type"] = "application/json"
        config = t.GenerateContentConfig(**config_kwargs)
        contents = self._to_contents(messages)

        def _call():
            self.calls_made += 1
            return self._client.models.generate_content(model=self.model, contents=contents, config=config)

        def _log_retry(exc: Exception, delay: float, attempt: int) -> None:
            log.warning("Gemini %s (attempt %d/%d); retrying in %.0fs",
                        self._code(exc) or type(exc).__name__, attempt, self.max_attempts, delay)

        try:
            response = call_with_retry(
                _call, self._is_retryable, max_attempts=self.max_attempts, base_delay=3.0,
                retry_after=self._retry_after, on_retry=_log_retry,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Gemini call failed: {type(exc).__name__}: {str(exc)[:300]}") from exc

        tool_calls = [ToolCall(name=fc.name, args=dict(fc.args or {})) for fc in (response.function_calls or [])]
        text = ""
        if not tool_calls:
            try:
                text = response.text or ""
            except ValueError:
                text = ""
        usage = {}
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            usage = {
                "input_tokens": meta.prompt_token_count or 0,
                "output_tokens": meta.candidates_token_count or 0,
            }
        raw = response.candidates[0].content if response.candidates else None
        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage, raw=raw)


def _is_daily_quota(exc: Exception) -> bool:
    return bool(re.search(r"PerDay|per day|daily", str(exc), re.I))


# --------------------------------------------------------------------------- stub


class OpenAIClient:
    """STUB - NOT IMPLEMENTED AND NOT TESTED.

    This marks where a second provider plugs in: implement generate() by converting
    Message/ToolSpec to that provider's chat format and back, exactly like GeminiClient.
    Nothing else in the code base needs to change.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("OpenAIClient is a placeholder; only Gemini and Fake are implemented")


def make_llm_client(settings) -> LLMClient:
    """Factory used by the CLI, API and eval: choose the provider from settings."""
    if settings.llm_provider == "fake":
        return FakeClient(echo_context_fake)
    if settings.llm_provider == "gemini":
        return GeminiClient(settings.gemini_api_key, settings.llm_model, thinking_level=settings.thinking_level)
    if settings.llm_provider == "ollama":
        from .ollama_client import OllamaClient

        return OllamaClient(
            settings.ollama_model,
            host=settings.ollama_host,
            timeout_s=settings.ollama_timeout_s,
            num_ctx=settings.ollama_num_ctx,
            tool_mode=settings.ollama_tool_mode,
            allow_remote=settings.allow_remote_llm,
        )
    if settings.llm_provider == "openai":
        return OpenAIClient()
    raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
