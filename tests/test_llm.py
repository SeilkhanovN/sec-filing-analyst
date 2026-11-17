import pytest

from sec_analyst.llm import FakeClient, LLMError, call_with_retry, echo_context_fake, parse_json_object, _is_daily_quota


class Flaky:
    def __init__(self, failures: int, exc: Exception):
        self.failures, self.exc, self.calls = failures, exc, 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def test_retry_succeeds_after_transient_errors_with_growing_delays():
    delays: list[float] = []
    fn = Flaky(2, RuntimeError("429"))
    assert call_with_retry(fn, lambda e: True, max_attempts=5, base_delay=1, sleep=delays.append) == "ok"
    assert fn.calls == 3
    assert len(delays) == 2 and 1 <= delays[0] < 1.6 and 2 <= delays[1] < 2.6


def test_retry_gives_up_after_max_attempts():
    delays: list[float] = []
    fn = Flaky(100, RuntimeError("429"))
    with pytest.raises(RuntimeError):
        call_with_retry(fn, lambda e: True, max_attempts=4, sleep=delays.append)
    assert fn.calls == 4 and len(delays) == 3


def test_non_retryable_error_is_raised_immediately():
    fn = Flaky(1, ValueError("400 bad request"))
    with pytest.raises(ValueError):
        call_with_retry(fn, lambda e: False, sleep=lambda s: None)
    assert fn.calls == 1


def test_retry_after_hint_and_max_delay_are_respected():
    delays: list[float] = []
    call_with_retry(Flaky(1, RuntimeError()), lambda e: True, sleep=delays.append,
                    retry_after=lambda e: 500.0, max_delay=30)
    assert 30 <= delays[0] < 30.6


def test_daily_quota_detection():
    assert _is_daily_quota(RuntimeError("Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"))
    assert not _is_daily_quota(RuntimeError("GenerateRequestsPerMinutePerProjectPerModel"))


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 2} hope it helps') == {"a": 2}
    assert parse_json_object("[1, 2]") is None
    assert parse_json_object("no json") is None


def test_fake_client_records_calls_and_fails_when_exhausted():
    client = FakeClient([])
    with pytest.raises(LLMError):
        client.generate([])
    assert len(client.calls) == 1


def test_echo_fake_quotes_the_first_chunk_verbatim():
    from sec_analyst.llm import Message

    prompt = "<context>\n[chunk_id: X-1] Co | fy | Item 1\nFirst sentence here. Second one.\n</context>\n\nQuestion: q"
    reply = parse_json_object(echo_context_fake([Message("user", prompt)], None, None).text)
    assert reply["citations"] == [{"chunk_id": "X-1", "quote": "First sentence here."}]
