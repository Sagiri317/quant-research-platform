import json

import httpx
import pytest

from quant_research_platform.agents.client import AgentAction, OpenAIResponsesClient


@pytest.fixture
def llm_config():
    return {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna",
        "credential_env": "QRP_TEST_API_KEY",
        "max_output_tokens": 128000,
        "reasoning_effort": "high",
        "timeout_seconds": 300,
    }


@pytest.mark.parametrize(
    "effort,remaining,expected_timeout", [("none", 1, 1), ("max", 600, 300)]
)
def test_http_transport_and_response_schema(
    monkeypatch, llm_config, effort, remaining, expected_timeout
):
    received = []

    def respond(request):
        received.append(request)
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "reasoning", "summary": []},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"kind":"finish","summary":"Done"}',
                            }
                        ],
                    },
                ],
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setenv("QRP_TEST_API_KEY", "synthetic-key")
    llm_config["reasoning_effort"] = effort
    client = OpenAIResponsesClient(llm_config)
    messages = [{"role": "user", "content": "Return a finish action."}]
    assert client.complete(messages, timeout=remaining).kind == "finish"
    assert received[0].url.path == "/v1/responses"
    assert received[0].headers["authorization"] == "Bearer synthetic-key"
    assert set(received[0].extensions["timeout"].values()) == {expected_timeout}
    body = json.loads(received[0].content)
    assert body["input"] == messages
    assert body["model"] == "gpt-6-luna"
    assert body["max_output_tokens"] == 128000
    assert body["reasoning"] == {"effort": effort}
    assert body["store"] is False
    assert body["service_tier"] == "default"
    assert body["text"]["verbosity"] == "medium"
    assert not {"max_tokens", "temperature", "top_p", "messages"} & body.keys()
    response_format = body["text"]["format"]
    assert response_format["type"] == "json_schema"
    assert response_format["strict"] is True
    schema = response_format["schema"]
    for model in [schema, *schema.get("$defs", {}).values()]:
        assert model["additionalProperties"] is False
        assert set(model["required"]) == set(model["properties"])
        assert all("default" not in field for field in model["properties"].values())
    with pytest.raises(ValueError, match="agent_action_fields_invalid"):
        AgentAction(kind="finish", summary="Done", tool="python")


@pytest.mark.parametrize(
    "status,body,error",
    [
        (401, {}, "agent_model_http_status_401"),
        (
            200,
            {"status": "incomplete", "output": []},
            "agent_model_response_not_completed",
        ),
        (
            200,
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "No"}],
                    }
                ],
            },
            "agent_model_refused",
        ),
    ],
)
def test_responses_failure_is_not_an_action(
    monkeypatch, llm_config, status, body, error
):
    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(status, json=body)
            ),
            **kwargs,
        ),
    )
    monkeypatch.setenv("QRP_TEST_API_KEY", "synthetic-key")
    with pytest.raises(RuntimeError, match=error):
        OpenAIResponsesClient(llm_config).complete([], timeout=1)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("reasoning_effort", "ultra", "reasoning_effort"),
        ("max_output_tokens", True, "response_budget"),
        ("timeout_seconds", 0, "request_timeout"),
        ("timeout_seconds", float("inf"), "qrp_json_nonfinite_number"),
    ],
)
def test_llm_config_rejects_invalid_request_settings(llm_config, field, value, error):
    llm_config[field] = value
    with pytest.raises(ValueError, match=error):
        OpenAIResponsesClient(llm_config)


def test_action_rejects_duplicate_arguments_and_nonfinite_values():
    for value in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError):
            AgentAction(
                kind="tool",
                tool="search",
                summary="Try a formula.",
                arguments_json=value,
            )


def test_web_search_sends_only_query_and_retains_sources_and_usage(
    monkeypatch, llm_config
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "usage": {"input_tokens": 7, "output_tokens": 13},
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "A cited method.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://example.org/paper",
                                        "title": "Paper",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(respond), **kw),
    )
    monkeypatch.setenv("QRP_TEST_API_KEY", "synthetic-key")
    client = OpenAIResponsesClient(llm_config)
    result = client.search("factor robustness", timeout=20)
    assert requests[0]["input"] == [{"role": "user", "content": "factor robustness"}]
    assert requests[0]["tools"] == [{"type": "web_search"}]
    assert requests[0]["tool_choice"] == "required"
    assert result == {
        "text": "A cited method.",
        "sources": [{"url": "https://example.org/paper", "title": "Paper"}],
    }
    assert client.usage == {"input_tokens": 7, "output_tokens": 13}
