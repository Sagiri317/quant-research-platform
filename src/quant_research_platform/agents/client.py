"""Exchange structured research actions through the OpenAI Responses API.

Credentials stay in the host process. Responses must satisfy the action schema
before the runner can select an experiment or accept a conclusion.
"""

import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)


class AgentAction(BaseModel):
    """One tool call, question, evidence-backed conclusion."""

    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["tool", "ask_user", "finish"]
    tool: str | None = None
    arguments_json: str = "{}"
    summary: str = Field(
        min_length=1,
        description=(
            "For tool calls: a short progress description. For finish: the complete "
            "user-facing answer, including requested explanations and results, "
            "not a plan, heading, or description of what you would explain. "
            "For ask_user: the actual question."
        ),
    )
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action(self):
        if (self.kind == "tool") != bool(self.tool):
            raise ValueError("agent_action_fields_invalid")
        self.arguments()
        return self

    def arguments(self):
        return parse_qrp_json_object(self.arguments_json.encode("utf-8"))

    @staticmethod
    def response_schema():
        """Make all action fields explicit as required by strict Structured Outputs."""
        schema = _ModelResponse.model_json_schema()
        for model in [schema, *schema.get("$defs", {}).values()]:
            model["required"] = list(model["properties"])
            for field in model["properties"].values():
                field.pop("default", None)
        return schema


class _ToolAction(AgentAction):
    kind: Literal["tool"]
    tool: str = Field(min_length=1)


class _ControlAction(AgentAction):
    kind: Literal["ask_user", "finish"]
    tool: None = None


class _ModelResponse(BaseModel):
    """Keep kind/tool consistency inside the provider's supported nested union."""

    model_config = ConfigDict(extra="forbid", strict=True)
    action: _ToolAction | _ControlAction


class OpenAIResponsesClient:
    """Transport only; provider credentials are read outside experiment containers."""

    def __init__(self, config):
        self.usage = None
        self.config = _strict_config(
            config,
            {
                "base_url",
                "model",
                "credential_env",
                "max_output_tokens",
                "reasoning_effort",
                "timeout_seconds",
            },
            set(),
        )
        url = httpx.URL(self.config["base_url"])
        if (
            url.username
            or url.password
            or url.query
            or url.scheme not in {"http", "https"}
        ):
            raise ValueError("agent_api_url_invalid")
        if url.scheme == "http" and url.host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("agent_remote_api_requires_tls")
        if (
            type(config["max_output_tokens"]) is not int
            or config["max_output_tokens"] <= 0
        ):
            raise ValueError("agent_response_budget_invalid")
        if config["reasoning_effort"] not in (
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise ValueError("agent_reasoning_effort_invalid")
        if (
            type(config["timeout_seconds"]) not in (int, float)
            or config["timeout_seconds"] <= 0
        ):
            raise ValueError("agent_request_timeout_invalid")

    def complete(self, messages, *, timeout):
        payload = self._request(
            messages,
            timeout=timeout,
            text={
                "verbosity": "medium",
                "format": {
                    "type": "json_schema",
                    "name": "research_action",
                    "strict": True,
                    "schema": AgentAction.response_schema(),
                },
            },
        )
        response = _ModelResponse.model_validate(
            parse_qrp_json_object(self._text(payload).encode("utf-8"))
        )
        return response.action

    def search(self, query, *, timeout):
        """Search public sources without sending local research context."""
        payload = self._request(
            [{"role": "user", "content": query}],
            timeout=timeout,
            tools=[{"type": "web_search"}],
            tool_choice="required",
            include=["web_search_call.action.sources"],
        )
        citations = []
        for item in payload["output"]:
            for part in item.get("content", []):
                for annotation in part.get("annotations", []):
                    if annotation["type"] == "url_citation":
                        citations.append(
                            {key: annotation[key] for key in ("url", "title")}
                        )
        return {"text": self._text(payload), "sources": citations}

    def _request(self, messages, *, timeout, **options):
        self.usage = None
        credential = os.environ[self.config["credential_env"]]
        with httpx.Client(
            timeout=min(timeout, self.config["timeout_seconds"]), follow_redirects=False
        ) as client:
            response = client.post(
                self.config["base_url"].rstrip("/") + "/responses",
                headers={"Authorization": "Bearer " + credential},
                json={
                    "model": self.config["model"],
                    "input": messages,
                    "max_output_tokens": self.config["max_output_tokens"],
                    "reasoning": {"effort": self.config["reasoning_effort"]},
                    "store": False,
                    "service_tier": "default",
                    **options,
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"agent_model_http_status_{response.status_code}")
        payload = response.json()
        self.usage = payload.get("usage")
        if payload["status"] != "completed":
            raise RuntimeError("agent_model_response_not_completed")
        return payload

    @staticmethod
    def _text(payload):
        text = []
        for item in payload["output"]:
            if item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "refusal":
                        raise RuntimeError("agent_model_refused")
                    if part["type"] == "output_text":
                        text.append(part["text"])
        return "".join(text)
