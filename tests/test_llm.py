from eegfm_digest.llm import LLMCallConfig, OpenAICall, load_api_key, parse_json_text


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
