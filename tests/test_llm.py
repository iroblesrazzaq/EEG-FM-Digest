from eegfm_digest.llm import (
    LLMCallConfig,
    LLMRateLimitError,
    OpenAICall,
    load_api_key,
    parse_json_text,
    parse_retry_after_seconds,
    rate_limit_sleep_seconds,
)


def _fake_response(text: str):
    return type(
        "FakeResponse",
        (),
        {
            "choices": [
                type(
                    "FakeChoice",
                    (),
                    {
                        "message": type(
                            "FakeMessage",
                            (),
                            {
                                "content": text,
                            },
                        )()
                    },
                )()
            ]
        },
    )()


class _FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class _FakeClient:
    def __init__(self, completions):
        self.chat = type("FakeChat", (), {"completions": completions})()


def test_openai_call_extract_text_handles_dict_content_parts():
    call = OpenAICall.__new__(OpenAICall)
    response = type(
        "FakeResponse",
        (),
        {
            "choices": [
                type(
                    "FakeChoice",
                    (),
                    {
                        "message": type(
                            "FakeMessage",
                            (),
                            {
                                "content": [
                                    {"type": "text", "text": "{\"decision\":\"accept\"}"},
                                ]
                            },
                        )()
                    },
                )()
            ]
        },
    )()

    assert call._extract_text(response) == "{\"decision\":\"accept\"}"


def test_google_provider_omits_response_format():
    completions = _FakeCompletions(_fake_response("{\"decision\":\"accept\"}"))
    call = OpenAICall.__new__(OpenAICall)
    call.config = LLMCallConfig(
        provider="google",
        api_key="test-key",
        model="gemma-4-31b-it",
        temperature=0.2,
        max_output_tokens=256,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    call._client = _FakeClient(completions)

    result = call.call("prompt", schema={"type": "object"})

    assert result.text == "{\"decision\":\"accept\"}"
    assert "response_format" not in completions.request


def test_openrouter_provider_sets_json_object_response_format():
    completions = _FakeCompletions(_fake_response("{\"decision\":\"accept\"}"))
    call = OpenAICall.__new__(OpenAICall)
    call.config = LLMCallConfig(
        provider="openrouter",
        api_key="test-key",
        model="stepfun/step-3.5-flash:free",
        temperature=0.2,
        max_output_tokens=256,
        base_url="https://openrouter.ai/api/v1",
    )
    call._client = _FakeClient(completions)

    call.call("prompt", schema={"type": "object"})

    assert completions.request["response_format"] == {"type": "json_object"}


def test_load_api_key_uses_gemini_key_for_google_provider(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")

    assert load_api_key("google") == "gem-key"
    assert load_api_key() == "gem-key"


def test_parse_json_text_extracts_json_object_from_surrounding_text():
    text = "<thought>internal reasoning</thought>{\"decision\":\"accept\",\"confidence\":0.9,\"reasons\":[\"r1\",\"r2\"]}"

    assert parse_json_text(text) == {
        "decision": "accept",
        "confidence": 0.9,
        "reasons": ["r1", "r2"],
    }


def test_parse_retry_after_seconds_from_gemini_message():
    detail = (
        "You exceeded your current quota... limit: 16000, model: gemma-4-31b\n"
        "Please retry in 40.692078921s."
    )
    assert parse_retry_after_seconds(detail) == 40.692078921


def test_parse_retry_after_seconds_from_retry_delay_json():
    detail = '{"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "4s"}]}'
    assert parse_retry_after_seconds(detail) == 4.0


def test_rate_limit_sleep_prefers_provider_retry_info():
    assert rate_limit_sleep_seconds(
        "Please retry in 40s.",
        attempt=0,
        floor_backoff_seconds=5.0,
        max_sleep_seconds=120.0,
    ) == 40.0


def test_rate_limit_sleep_falls_back_to_exponential():
    assert rate_limit_sleep_seconds(
        "no retry hint",
        attempt=2,
        floor_backoff_seconds=5.0,
        max_sleep_seconds=120.0,
    ) == 20.0


def test_openai_call_honors_retry_info_before_succeeding(monkeypatch):
    sleeps: list[float] = []

    class RateLimitedOnce:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                body = (
                    '[{"error":{"code":429,"message":"Quota exceeded. Please retry in 3.5s.",'
                    '"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"3s"}]}}]'
                )
                exc = Exception("rate limited")
                exc.status_code = 429  # type: ignore[attr-defined]
                exc.response = type("R", (), {"text": body, "status_code": 429})()  # type: ignore[attr-defined]
                raise exc
            return _fake_response('{"decision":"accept"}')

    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    call = OpenAICall.__new__(OpenAICall)
    call.config = LLMCallConfig(
        provider="google",
        api_key="test-key",
        model="gemma-4-31b-it",
        temperature=0.2,
        max_output_tokens=256,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    completions = RateLimitedOnce()
    call._client = _FakeClient(completions)

    result = call.call("prompt")
    assert result.text == '{"decision":"accept"}'
    assert sleeps == [3.5]
    assert completions.calls == 2


def test_openai_call_raises_after_retries_exhausted(monkeypatch):
    monkeypatch.setenv("LLM_RATE_LIMIT_RETRIES", "1")
    monkeypatch.setenv("LLM_RATE_LIMIT_BACKOFF_SECONDS", "1")
    sleeps: list[float] = []

    class AlwaysLimited:
        def create(self, **kwargs):
            exc = Exception("rate limited")
            exc.status_code = 429  # type: ignore[attr-defined]
            exc.response = type("R", (), {"text": "Please retry in 2s.", "status_code": 429})()  # type: ignore[attr-defined]
            raise exc

    monkeypatch.setattr("time.sleep", lambda seconds: sleeps.append(seconds))
    call = OpenAICall.__new__(OpenAICall)
    call.config = LLMCallConfig(
        provider="google",
        api_key="test-key",
        model="gemma-4-31b-it",
        temperature=0.2,
        max_output_tokens=256,
    )
    call._client = _FakeClient(AlwaysLimited())

    try:
        call.call("prompt")
        raise AssertionError("expected LLMRateLimitError")
    except LLMRateLimitError:
        pass
    assert sleeps == [2.0]
