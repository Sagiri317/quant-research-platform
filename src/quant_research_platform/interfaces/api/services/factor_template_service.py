"""API helpers for controlled factor expression templates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from quant_research_platform.research.factor_factory.factor_registry import (
    FactorTemplateRegistry,
)

TEMPLATE_ROOT = Path("config/a_share/factor_templates")
FACTOR_TEMPLATE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class FactorTemplatePayload(BaseModel):
    """Factor expression template exposed to the local web UI."""

    id: str
    name: str
    version: str
    market: str = "a_share"
    expression: str
    description: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    grid_parameter: str | None = None
    grid_values: list[str] = Field(default_factory=list)


def list_factor_template_payloads(market: str = "a_share") -> list[dict[str, Any]]:
    """Return controlled factor templates for ``market``."""

    if market != "a_share":
        return []
    registry = FactorTemplateRegistry.from_directory(TEMPLATE_ROOT, market=market)
    payloads: list[FactorTemplatePayload] = []
    for template in registry.templates():
        payloads.append(
            FactorTemplatePayload(
                id=template.template_id,
                name=template.name,
                version=template.version,
                market=template.market,
                expression=template.expression,
                description=template.description,
                dependencies=list(template.dependencies),
                inputs=list(template.inputs),
                outputs=list(template.outputs),
                capabilities=list(template.capabilities),
                grid_parameter=template.grid_parameter,
                grid_values=list(template.grid_values),
            )
        )
    return [payload.model_dump() for payload in payloads]


def factor_template_metadata(
    factor_template_id: str | None,
    *,
    market: str,
) -> tuple[str, str | None]:
    """Resolve a controlled factor template id and portable source path."""

    template_id = factor_template_id or "custom"
    if template_id == "custom":
        return template_id, None
    if not FACTOR_TEMPLATE_ID_PATTERN.fullmatch(template_id):
        _raise_payload_validation_error(
            code="unsafe_factor_template_id",
            field="factor_template_id",
            message=(
                "factor_template_id may only contain ASCII letters, numbers, "
                "underscores, and hyphens"
            ),
        )
    if market == "a_share" and template_id not in _controlled_template_ids(market):
        _raise_payload_validation_error(
            code="unknown_factor_template_id",
            field="factor_template_id",
            message=f"unknown factor_template_id: {template_id}",
        )
    return template_id, f"{TEMPLATE_ROOT.as_posix()}/{template_id}.yaml"


def _controlled_template_ids(market: str) -> set[str]:
    registry = FactorTemplateRegistry.from_directory(TEMPLATE_ROOT, market=market)
    return {template.template_id for template in registry.templates()}


def _raise_payload_validation_error(*, code: str, field: str, message: str) -> None:
    detail = {
        "blocking": [
            {
                "severity": "blocking",
                "code": code,
                "message": message,
                "field": field,
            }
        ],
        "warnings": [],
    }
    raise HTTPException(status_code=422, detail=detail)
