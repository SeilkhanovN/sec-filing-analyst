"""OllamaClient (LLM_PROVIDER=ollama) with a mocked HTTP layer: no network, no model needed."""

from __future__ import annotations

import json

import pytest
import requests

from sec_analyst.agent import Agent
from sec_analyst.config import load_settings
from sec_analyst.llm import LLMError, Message, ToolCall, make_llm_client
from sec_analyst.ollama_client import OllamaClient, check_local_host
from sec_analyst.rag import answer_question
from sec_analyst.tools import make_screener_tool, make_search_tool

QUOTE = "outsourcing partners located primarily in Asia"


class FakeResponse:
    def __init__(self, status_code: int = 200, body=None, text: str | None = None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


class FakeSession:
    """Stands in for requests.Session. Each reply is a FakeResponse or an exception to raise."""

    def __init__(self, replies=(), tags=None):
        self.replies = list(replies)
        self.posts: list[dict] = []
        self.tags = tags

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        if not self.replies:
            raise AssertionError(f"unexpected extra request to {url}")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, timeout=None):
        return FakeResponse(200, self.tags)


def chat(content: str = "", tool_calls=None, prompt_tokens: int = 100, output_tokens: int = 10) -> FakeResponse:
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return FakeResponse(200, {"model": "llama3.2:3b", "message": message, "done": True,
                              "prompt_eval_count": prompt_tokens, "eval_count": output_tokens})


def show(capabilities) -> FakeResponse:
    return FakeResponse(200, {"capabilities": capabilities})


def call(name: str, arguments) -> dict:
    return {"function": {"name": name, "arguments": arguments}}


def client(session: FakeSession, **kwargs) -> OllamaClient:
    return OllamaClient("llama3.2:3b", session=session, **kwargs)


def final_json(quote: str = QUOTE, chunk_id: str = "AAPL-2025-item1a-000") -> str:
    return json.dumps({"found": True, "answer": "Apple relies on partners in Asia.",
                       "citations": [{"chunk_id": chunk_id, "quote": quote}]})


@pytest.fixture
def search_spec(retriever):
    return make_search_tool(retriever).spec


# --------------------------------------------------------------------------- success


def test_text_reply_and_request_format():
    session = FakeSession([chat('{"found": false, "answer": "", "citations": []}', prompt_tokens=1735, output_tokens=62)])
    llm = client(session, timeout_s=42, num_ctx=4096)
    response = llm.generate([Message("user", "question?")], system="be careful", json_output=True)

    assert response.text.startswith('{"found": false') and not response.tool_calls and response.tool_error is None
    assert response.usage == {"input_tokens": 1735, "output_tokens": 62}
    [post] = session.posts
    assert post["url"] == "http://127.0.0.1:11434/api/chat"
    assert post["timeout"] == (5.0, 42)
    body = post["json"]
    assert body["model"] == "llama3.2:3b" and body["stream"] is False and body["format"] == "json"
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096}
    assert body["messages"] == [{"role": "system", "content": "be careful"}, {"role": "user", "content": "question?"}]
    assert "tools" not in body


def test_rag_answers_from_ollama_go_through_the_same_citation_checks(retriever):
    good = client(FakeSession([chat(final_json())]))
    answer = answer_question("Where are Apple's suppliers?", retriever, good, ticker="AAPL")
    assert answer.found and answer.grounded and answer.citations[0].valid

    invented = client(FakeSession([chat(final_json(quote="factories located primarily on the Moon"))]))
    answer = answer_question("Where are Apple's suppliers?", retriever, invented, ticker="AAPL")
    assert answer.found and not answer.grounded and not answer.citations[0].valid

    refusal = client(FakeSession([chat('{"found": false, "answer": "", "citations": []}')]))
    assert not answer_question("Who is Apple's chef?", retriever, refusal).found


def test_list_models_reads_local_tags():
    llm = client(FakeSession(tags={"models": [{"name": "llama3.2:3b"}, {"name": "qwen3:4b"}]}))
    assert llm.list_models() == ["llama3.2:3b", "qwen3:4b"]


# --------------------------------------------------------------------------- failures


def test_ollama_down_gives_a_clear_error_without_retrying():
    session = FakeSession([requests.ConnectionError("connection refused")])
    with pytest.raises(LLMError, match="not reachable.*ollama serve"):
        client(session).generate([Message("user", "q")])
    assert len(session.posts) == 1  # one attempt, no retry loop


def test_missing_model_tells_the_user_to_pull_it():
    session = FakeSession([FakeResponse(404, {"error": "model 'llama3.2:3b' not found"})])
    with pytest.raises(LLMError, match=r"ollama pull llama3\.2:3b"):
        client(session).generate([Message("user", "q")])


def test_timeout_fails_fast_with_a_clear_error():
    session = FakeSession([requests.ReadTimeout("read timed out")])
    with pytest.raises(LLMError, match="did not answer within 7 s.*OLLAMA_TIMEOUT_S"):
        client(session, timeout_s=7).generate([Message("user", "q")])
    assert len(session.posts) == 1


def test_other_http_errors_and_bad_bodies_raise_llm_error():
    with pytest.raises(LLMError, match="HTTP 500"):
        client(FakeSession([FakeResponse(500, {"error": "out of memory"})])).generate([Message("user", "q")])
    with pytest.raises(LLMError, match="not a JSON object"):
        client(FakeSession([FakeResponse(200, None, text="<html>")])).generate([Message("user", "q")])
    with pytest.raises(LLMError, match="no message"):
        client(FakeSession([FakeResponse(200, {"done": True})])).generate([Message("user", "q")])


def test_model_without_native_tools_suggests_json_mode(search_spec):
    session = FakeSession([FakeResponse(400, {"error": "llama2 does not support tools"})])
    with pytest.raises(LLMError, match="OLLAMA_TOOL_MODE=json"):
        client(session, tool_mode="native").generate([Message("user", "q")], tools=[search_spec])


# --------------------------------------------------------------------------- native tool calls


def test_native_tool_calls_are_parsed_and_sent_back_in_ollama_format(search_spec):
    session = FakeSession([
        show(["completion", "tools"]),
        chat("", tool_calls=[call("search_filings", {"query": "Apple suppliers", "ticker": "AAPL"})]),
        chat(final_json()),
    ])
    llm = client(session)
    first = llm.generate([Message("user", "q")], system="sys", tools=[search_spec])
    assert first.tool_calls == [ToolCall("search_filings", {"query": "Apple suppliers", "ticker": "AAPL"})]
    assert session.posts[0]["url"].endswith("/api/show")
    sent_tool = session.posts[1]["json"]["tools"][0]
    assert sent_tool["type"] == "function" and sent_tool["function"]["name"] == "search_filings"
    assert "format" not in session.posts[1]["json"]

    history = [Message("user", "q"), Message("assistant", tool_calls=first.tool_calls),
               Message("tool", tool_name="search_filings", tool_result={"results": [{"chunk_id": "X"}]})]
    llm.generate(history, system="sys", tools=[search_spec])
    messages = session.posts[2]["json"]["messages"]  # capability check was cached: no second /api/show
    assert messages[2] == {"role": "assistant", "content": "",
                           "tool_calls": [call("search_filings", {"query": "Apple suppliers", "ticker": "AAPL"})]}
    assert messages[3] == {"role": "tool", "tool_name": "search_filings", "content": '{"results": [{"chunk_id": "X"}]}'}


@pytest.mark.parametrize("tool_calls, reason", [
    ([call("search_filings", "{not json")], "not valid JSON"),
    ([call("search_filings", ["query"])], "must be a JSON object"),
    ([call("delete_everything", {})], "unknown tool"),
    ([call("search_filings", {"ticker": "AAPL"})], "missing required argument(s): query"),
    ([call("search_filings", {"query": "x", "sql": "DROP TABLE"})], "unknown argument(s): sql"),
    ([call("", {"query": "x"})], "without a tool name"),
    ([{"type": "function"}], "no function object"),
    ("search_filings", "not a list"),
])
def test_malformed_native_tool_calls_become_tool_errors(search_spec, tool_calls, reason):
    llm = client(FakeSession([chat("", tool_calls=tool_calls)]), tool_mode="native")
    response = llm.generate([Message("user", "q")], tools=[search_spec])
    assert response.tool_calls == [] and reason in response.tool_error


def test_arguments_sent_as_a_json_string_are_accepted(search_spec):
    llm = client(FakeSession([chat("", tool_calls=[call("search_filings", '{"query": "risks"}')])]), tool_mode="native")
    assert llm.generate([Message("user", "q")], tools=[search_spec]).tool_calls == [ToolCall("search_filings", {"query": "risks"})]


def test_tool_call_written_as_text_is_parsed_strictly(search_spec):
    as_text = '{"name": "search_filings", "parameters": {"query": "Apple employees"}}'
    llm = client(FakeSession([chat(as_text), chat(final_json()), chat('{"name": "rm_rf", "parameters": {}}')]),
                 tool_mode="native")
    assert llm.generate([Message("user", "q")], tools=[search_spec]).tool_calls == [
        ToolCall("search_filings", {"query": "Apple employees"})]
    answer = llm.generate([Message("user", "q")], tools=[search_spec])
    assert answer.tool_calls == [] and answer.tool_error is None and '"found": true' in answer.text
    assert "unknown tool" in llm.generate([Message("user", "q")], tools=[search_spec]).tool_error


# --------------------------------------------------------------------------- JSON tool mode


def test_json_tool_mode_is_chosen_when_the_model_has_no_tool_support(search_spec):
    session = FakeSession([
        show(["completion"]),
        chat('{"tool_call": {"name": "search_filings", "arguments": {"query": "Apple suppliers"}}}'),
        chat('{"tool_call": "search_filings"}'),
    ])
    llm = client(session)
    response = llm.generate([Message("user", "q")], system="sys", tools=[search_spec])
    assert response.tool_calls == [ToolCall("search_filings", {"query": "Apple suppliers"})]
    body = session.posts[1]["json"]
    assert "tools" not in body and body["format"] == "json"
    assert '{"tool_call": {"name": "<tool name>"' in body["messages"][0]["content"]
    assert "search_filings" in body["messages"][0]["content"]

    history = [Message("user", "q"), Message("assistant", tool_calls=response.tool_calls),
               Message("tool", tool_name="search_filings", tool_result={"results": []})]
    malformed = llm.generate(history, system="sys", tools=[search_spec])
    assert malformed.tool_calls == [] and "exactly" in malformed.tool_error
    sent = session.posts[2]["json"]["messages"]
    assert json.loads(sent[2]["content"]) == {"tool_call": {"name": "search_filings", "arguments": {"query": "Apple suppliers"}}}
    assert sent[3]["role"] == "user" and sent[3]["content"].startswith("Result of tool search_filings (untrusted data")


# --------------------------------------------------------------------------- agent end to end


def test_agent_runs_through_ollama_with_verified_citations(retriever, store, search_spec):
    session = FakeSession([
        show(["completion", "tools"]),
        chat("", tool_calls=[call("search_filings", {"query": "Apple suppliers", "ticker": "AAPL"})]),
        chat(final_json()),
    ])
    agent = Agent(client(session), [make_search_tool(retriever), make_screener_tool("")], chunk_lookup=store.get)
    answer = agent.run("Where does Apple manufacture?")
    assert answer.found and answer.grounded
    assert [s["type"] for s in answer.steps] == ["tool", "answer"]
    tool_message = session.posts[2]["json"]["messages"][-1]
    assert tool_message["role"] == "tool" and "AAPL-2025-item1a-000" in tool_message["content"]


def test_agent_survives_malformed_ollama_tool_calls_and_keeps_the_step_limit(retriever, store):
    replies = [show(["completion", "tools"])] + [chat("", tool_calls=[call("search_filings", "{broken")])] * 3
    session = FakeSession(replies)
    agent = Agent(client(session), [make_search_tool(retriever)], chunk_lookup=store.get, max_steps=3)
    answer = agent.run("q")
    assert not answer.found and answer.error == "max_steps reached"
    assert [s["type"] for s in answer.steps] == ["error", "error", "error"]
    assert len(session.posts) == 4  # 1 capability check + exactly max_steps chat calls
    feedback = session.posts[-1]["json"]["messages"][-1]
    assert feedback["role"] == "user" and "not a valid tool call" in feedback["content"]


# --------------------------------------------------------------------------- localhost only


@pytest.mark.parametrize("host", [
    "http://127.0.0.1:11434", "http://localhost:11434", "127.0.0.1:11434", "http://[::1]:11434",
    "http://127.0.0.2:11434", "", None, "http://LOCALHOST:11434/",
])
def test_local_hosts_are_allowed(host):
    assert check_local_host(host).startswith("http")


@pytest.mark.parametrize("host", [
    "http://192.168.1.20:11434", "http://10.0.0.5:11434", "http://example.com:11434",
    "https://api.someprovider.example", "http://localhost.evil.example:11434",
    "http://127.0.0.1@evil.example:11434", "http://user:pw@127.0.0.1:11434", "http://0.0.0.0:11434",
    "ftp://127.0.0.1:11434", "http://127.0.0.1:notaport",
])
def test_non_local_or_malformed_hosts_are_refused(host):
    with pytest.raises(LLMError):
        check_local_host(host)


def test_remote_host_needs_the_explicit_flag_and_no_request_is_sent_before_the_check():
    session = FakeSession()
    with pytest.raises(LLMError, match="ALLOW_REMOTE_LLM=1"):
        OllamaClient("llama3.2:3b", host="http://192.168.1.20:11434", session=session)
    assert session.posts == []
    allowed = OllamaClient("llama3.2:3b", host="http://192.168.1.20:11434", allow_remote=True, session=session)
    assert allowed.base_url == "http://192.168.1.20:11434"


def test_default_session_ignores_proxy_environment_variables():
    assert OllamaClient("llama3.2:3b")._session.trust_env is False


def test_bad_tool_mode_and_empty_model_are_rejected():
    with pytest.raises(LLMError, match="OLLAMA_TOOL_MODE"):
        OllamaClient("llama3.2:3b", tool_mode="xml")
    with pytest.raises(LLMError, match="OLLAMA_MODEL"):
        OllamaClient("")


# --------------------------------------------------------------------------- configuration

ENV_VARS = ["LLM_PROVIDER", "LLM_MODEL", "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_TIMEOUT_S",
            "OLLAMA_NUM_CTX", "OLLAMA_TOOL_MODE", "ALLOW_REMOTE_LLM"]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    empty_env = tmp_path / ".env"
    empty_env.write_text("")
    return lambda: load_settings(empty_env)


def test_gemini_stays_the_default_provider(clean_env):
    settings = clean_env()
    assert settings.llm_provider == "gemini"
    assert settings.ollama_host == "http://127.0.0.1:11434" and settings.ollama_model == "llama3.2:3b"
    assert settings.allow_remote_llm is False


def test_ollama_provider_is_selected_by_configuration(clean_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:4b")
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11500")
    monkeypatch.setenv("OLLAMA_TIMEOUT_S", "30")
    settings = clean_env()
    llm = make_llm_client(settings)
    assert isinstance(llm, OllamaClient)
    assert llm.model == "qwen3:4b" and llm.base_url == "http://localhost:11500" and llm.timeout_s == 30
    settings.set_model("llama3.2:3b")  # --model applies to the active provider
    assert settings.ollama_model == "llama3.2:3b" and settings.llm_model == "gemini-3.5-flash-lite"


def test_remote_ollama_host_is_refused_unless_flag_is_set(clean_env, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://203.0.113.7:11434")
    with pytest.raises(LLMError, match="not on this machine"):
        make_llm_client(clean_env())
    monkeypatch.setenv("ALLOW_REMOTE_LLM", "1")
    assert make_llm_client(clean_env()).base_url == "http://203.0.113.7:11434"
