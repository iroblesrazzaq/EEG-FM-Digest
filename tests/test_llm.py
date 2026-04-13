from eegfm_digest.llm import OpenAICall


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
