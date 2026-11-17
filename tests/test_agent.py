import json

import pytest

from sec_analyst.agent import Agent
from sec_analyst.llm import FakeClient, LLMError, LLMResponse, ToolCall
from sec_analyst.tools import Tool, make_screener_tool, make_search_tool
from sec_analyst.llm import ToolSpec


def final(answer: str, citations: list[dict]) -> LLMResponse:
    return LLMResponse(text=json.dumps({"found": True, "answer": answer, "citations": citations}))


def make_agent(retriever, store, script, screener_url="", max_steps=4):
    tools = [make_search_tool(retriever), make_screener_tool(screener_url)]
    llm = FakeClient(script)
    return Agent(llm, tools, chunk_lookup=store.get, max_steps=max_steps), llm


def test_agent_uses_both_tools_then_answers_with_verified_citation(retriever, store, screener_server):
    script = [
        LLMResponse(tool_calls=[ToolCall("search_filings", {"query": "Apple suppliers", "ticker": "AAPL"})]),
        LLMResponse(tool_calls=[ToolCall("get_screener_result", {"ticker": "AAPL"})]),
        final("Apple relies on suppliers in Asia; the screener signal is hold.",
              [{"chunk_id": "AAPL-2025-item1a-000", "quote": "outsourcing partners located primarily in Asia"}]),
    ]
    agent, llm = make_agent(retriever, store, script, screener_url=screener_server)
    answer = agent.run("Where does Apple manufacture and what does the screener say?")

    assert answer.found and answer.grounded
    tools_used = [s["tool"] for s in answer.steps if s["type"] == "tool"]
    assert tools_used == ["search_filings", "get_screener_result"]
    assert answer.steps[-1]["type"] == "answer"
    assert answer.usage["llm_calls"] == 3
    # The tool results were fed back to the model as tool messages.
    last_messages = llm.calls[2]["messages"]
    tool_msgs = [m for m in last_messages if m.role == "tool"]
    assert tool_msgs[0].tool_result["results"][0]["ticker"] == "AAPL"
    assert tool_msgs[1].tool_result["data"]["signal"] == "hold"
    assert "AAPL-2025-item1a-000" in answer.retrieved_ids


def test_agent_stops_at_max_steps():
    looping = FakeClient(lambda *a: LLMResponse(tool_calls=[ToolCall("search_filings", {"query": "x"})]))
    agent = Agent(looping, [Tool(ToolSpec("search_filings", "d", {}), lambda **kw: {"results": []})],
                  chunk_lookup=lambda _: None, max_steps=3)
    answer = agent.run("loop forever?")
    assert len(looping.calls) == 3
    assert not answer.found and answer.error == "max_steps reached"


def test_unknown_tool_and_crashing_tool_are_reported_to_the_model():
    def boom(**kwargs):
        raise RuntimeError("disk on fire")

    script = [
        LLMResponse(tool_calls=[ToolCall("delete_everything", {})]),
        LLMResponse(tool_calls=[ToolCall("search_filings", {"query": "x"})]),
        LLMResponse(tool_calls=[ToolCall("search_filings", {"unexpected": 1})]),
        LLMResponse(text='{"found": false, "answer": "", "citations": []}'),
    ]
    llm = FakeClient(script)
    agent = Agent(llm, [Tool(ToolSpec("search_filings", "d", {}), boom)], chunk_lookup=lambda _: None, max_steps=5)
    answer = agent.run("q")
    summaries = [s["summary"] for s in answer.steps if s["type"] == "tool"]
    assert summaries[0].startswith("error: Unknown tool")
    assert "disk on fire" in summaries[1]
    assert summaries[2].startswith("error:")
    assert not answer.found


def test_llm_failure_returns_a_clear_error_instead_of_crashing(retriever, store):
    def failing(*args):
        raise LLMError("429 quota exhausted")

    agent, _ = make_agent(retriever, store, failing)
    answer = agent.run("anything")
    assert not answer.found and "429" in answer.error


def test_prose_answer_is_kept_but_marked_uncited(retriever, store):
    agent, _ = make_agent(retriever, store, [LLMResponse(text="Apple makes phones.")])
    answer = agent.run("What does Apple make?")
    assert answer.answer == "Apple makes phones." and not answer.grounded and answer.error


def test_ticker_hint_is_passed_to_the_model(retriever, store):
    agent, llm = make_agent(retriever, store, [LLMResponse(text='{"found": false, "answer": "", "citations": []}')])
    agent.run("Risks?", ticker="TSLA")
    assert "TSLA" in llm.calls[0]["messages"][0].text
    assert {t.name for t in llm.calls[0]["tools"]} == {"search_filings", "get_screener_result"}


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError):
        Agent(FakeClient([]), [], chunk_lookup=lambda _: None, max_steps=0)


# ---- tools ---------------------------------------------------------------------------


def test_search_tool_returns_passages_and_rejects_unknown_ticker(retriever):
    tool = make_search_tool(retriever, k=2)
    result = tool.fn(query="cloud computing services", ticker="MSFT")
    assert result["results"][0]["chunk_id"] == "MSFT-2025-item1-000"
    assert "error" in tool.fn(query="x", ticker="ZZZZ")


def test_screener_tool_against_fake_http_server(screener_server):
    tool = make_screener_tool(screener_server)
    assert tool.fn(ticker="aapl") == {"ticker": "AAPL", "data": {"ticker": "AAPL", "pe_ratio": 31.2, "signal": "hold"}}
    assert "HTTP 404" in tool.fn(ticker="MSFT")["error"]
    assert "not JSON" in tool.fn(ticker="BAD")["error"]
    assert "Invalid ticker" in tool.fn(ticker="../etc")["error"]


def test_screener_tool_unreachable_or_unconfigured():
    # Port 9 on localhost: nothing listens there, so the connection is refused quickly.
    unreachable = make_screener_tool("http://127.0.0.1:9", timeout_s=2).fn(ticker="AAPL")
    assert "unreachable" in unreachable["error"]
    assert "not configured" in make_screener_tool("").fn(ticker="AAPL")["error"]
