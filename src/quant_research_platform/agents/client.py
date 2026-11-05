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


class ParameterChoice(BaseModel):
    """Select a zero-based entry from one tool parameter's declared choices."""

    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    choice: int = Field(ge=0)


class ArtifactInput(BaseModel):
    """Bind one declared tool input to a successful experiment's evidence ID."""

    model_config = ConfigDict(extra="forbid", strict=True)
    path: str
    evidence: str


class AgentAction(BaseModel):
    """One proposed experiment or conclusion; execution policy belongs to the runner."""

    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["run_tool", "run_python", "finish"]
    tool: str | None = None
    changes: list[ParameterChoice] = Field(default_factory=list)
    inputs: list[ArtifactInput] = Field(default_factory=list)
    code: str | None = None
    conclusion: str | None = None
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action(self):
        for selections in (self.changes, self.inputs):
            if len({item.path for item in selections}) != len(selections):
                raise ValueError("agent_action_duplicate_path")
        if self.kind == "run_tool":
            valid = (
                self.tool
                and self.code is None
                and self.conclusion is None
                and not self.evidence
            )
        elif self.kind == "run_python":
            valid = (
                self.code
                and self.tool is None
                and not self.changes
                and not self.inputs
                and self.conclusion is None
                and not self.evidence
            )
        else:
            valid = (
                self.conclusion
                and self.tool is None
                and self.code is None
                and not self.changes
                and not self.inputs
            )
        if not valid:
            raise ValueError("agent_action_fields_invalid")
        return self

    @classmethod
    def response_schema(cls):
        """Make all action fields explicit as required by strict Structured Outputs."""
        schema = cls.model_json_schema()
        for model in [schema, *schema["$defs"].values()]:
            model["required"] = list(model["properties"])
            for field in model["properties"].values():
                field.pop("default", None)
        return schema


class OpenAIResponsesClient:
    """Transport only; provider credentials are read outside experiment containers."""

    def __init__(self, config):
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
                    "text": {
                        "verbosity": "medium",
                        "format": {
                            "type": "json_schema",
                            "name": "research_action",
                            "strict": True,
                            "schema": AgentAction.response_schema(),
                        },
                    },
                    "store": False,
                    "service_tier": "default",
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"agent_model_http_status_{response.status_code}")
        payload = response.json()
        if payload["status"] != "completed":
            raise RuntimeError("agent_model_response_not_completed")
        text = []
        for item in payload["output"]:
            if item["type"] == "message":
                for part in item["content"]:
                    if part["type"] == "refusal":
                        raise RuntimeError("agent_model_refused")
                    if part["type"] == "output_text":
                        text.append(part["text"])
        content = "".join(text)
        return AgentAction.model_validate(
            parse_qrp_json_object(content.encode("utf-8"))
        )
