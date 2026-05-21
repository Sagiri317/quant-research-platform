"""Static market capability metadata for the local platform API."""

from __future__ import annotations

from copy import deepcopy

MARKET_CAPABILITIES: dict[str, dict[str, object]] = {
    "a_share": {
        "market": "a_share",
        "label": "A 股",
        "enabled": True,
        "status": "enabled",
        "capabilities": [
            "local_csmar_data",
            "factor_pipeline",
            "evaluation_artifacts",
            "evidence_workbench",
        ],
        "data_boundary": "local_private_csmar",
    },
    "crypto": {
        "market": "crypto",
        "label": "加密资产",
        "enabled": False,
        "status": "planned",
        "capabilities": [],
        "data_boundary": "not_configured",
    },
}


def list_market_capabilities() -> list[dict[str, object]]:
    """Return copy-safe market capability summaries."""

    return [deepcopy(MARKET_CAPABILITIES[key]) for key in ("a_share", "crypto")]


def get_market_capabilities(market: str) -> dict[str, object] | None:
    """Return copy-safe capabilities for one market."""

    capabilities = MARKET_CAPABILITIES.get(market)
    if capabilities is None:
        return None
    return deepcopy(capabilities)
