"""Added in milestone 5: offline checks of GeminiClient's message conversion (no API call).

A real run showed the API rejects role="tool" for function results; they must be sent in a
"user" turn, with parallel results grouped together.
"""

from sec_analyst.llm import GeminiClient, Message, ToolCall


def test_tool_results_are_sent_as_one_user_turn():
    client = GeminiClient(api_key="dummy-key-not-used", model="gemini-test")
    messages = [
        Message(role="user", text="question"),
        Message(role="assistant", tool_calls=[ToolCall("a", {"x": 1}), ToolCall("b", {})]),
        Message(role="tool", tool_name="a", tool_result={"results": []}),
        Message(role="tool", tool_name="b", tool_result={"error": "down"}),
    ]
    contents = client._to_contents(messages)
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert [p.function_call.name for p in contents[1].parts] == ["a", "b"]
    assert [p.function_response.name for p in contents[2].parts] == ["a", "b"]
    assert contents[2].parts[1].function_response.response == {"error": "down"}


def test_raw_provider_content_is_passed_back_unchanged():
    client = GeminiClient(api_key="dummy-key-not-used", model="gemini-test")
    raw = object()
    contents = client._to_contents([Message(role="assistant", raw=raw)])
    assert contents[0] is raw
