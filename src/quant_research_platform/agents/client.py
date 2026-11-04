"""Exchange research actions with an OpenAI-compatible chat-completions API.

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
    """One proposed experiment or conclusion; execution policy belongs to the runner."""

    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["run_tool", "run_python", "finish"]
    tool: str | None = None
    changes: dict = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    code: str | None = None
    conclusion: str | None = None
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_action(self):
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


class OpenAICompatibleClient:
    """Transport only; provider credentials are read outside experiment containers."""

    def __init__(self, config):
        self.config = _strict_config(
            config, {"base_url", "model", "credential_env", "max_output_tokens"}, set()
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

    def complete(self, messages, *, timeout):
        credential = os.environ[self.config["credential_env"]]
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.post(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + credential},
                json={
                    "model": self.config["model"],
                    "messages": messages,
                    "max_tokens": self.config["max_output_tokens"],
                    "response_format": {"type": "json_object"},
                },
            )
        if response.status_code != 200:
            raise RuntimeError(f"agent_model_http_status_{response.status_code}")
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return AgentAction.model_validate(
            parse_qrp_json_object(content.encode("utf-8"))
        )
