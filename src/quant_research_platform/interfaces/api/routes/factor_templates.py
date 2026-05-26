"""Factor template catalog routes."""

from __future__ import annotations

from fastapi import APIRouter

from quant_research_platform.interfaces.api.services.factor_template_service import (
    list_factor_template_payloads,
)

router = APIRouter()


@router.get("/api/factor-templates")
def factor_templates(market: str = "a_share") -> list[dict[str, object]]:
    """Return editable factor expression templates."""

    return list_factor_template_payloads(market)


__all__ = ["router"]
