"""Diagnostics request and normalized backend spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REQUESTABLE_MATERIALIZED_VIEWS = frozenset(
    {
        "tradability_diagnostics",
        "tradability_summary",
        "portfolio_assumptions",
        "weights",
        "scenario_metrics",
        "exposure_capacity_summary",
        "portfolio_risk_summary",
        "diagnostic_observations",
    }
)
IMPLEMENTED_MATERIALIZED_VIEWS = frozenset(
    {
        "portfolio_assumptions",
        "weights",
        "tradability_diagnostics",
        "tradability_summary",
        "scenario_metrics",
        "exposure_capacity_summary",
        "portfolio_risk_summary",
    }
)


@dataclass(frozen=True)
class DiagnosticsSpec:
    enabled: bool = False
    policy_ids: tuple[str, ...] = ()
    materialized_views: tuple[str, ...] = ()

    @classmethod
    def disabled(cls) -> DiagnosticsSpec:
        return cls(enabled=False)

    @classmethod
    def from_request(cls, value: Any) -> DiagnosticsSpec:
        if value is None:
            return cls.disabled()
        if isinstance(value, DiagnosticsSpec):
            return value
        if not isinstance(value, dict):
            raise TypeError("diagnostics request must be an object.")
        enabled = bool(value.get("enabled", False))
        policy_ids = _string_tuple(value.get("policy_ids", ()), field="policy_ids")
        materialized_views = _string_tuple(
            value.get("materialized_views", ()),
            field="materialized_views",
        )
        unknown = sorted(
            set(materialized_views).difference(REQUESTABLE_MATERIALIZED_VIEWS)
        )
        if unknown:
            raise ValueError(f"unknown diagnostics materialized views: {unknown}.")
        unimplemented = sorted(
            set(materialized_views).difference(IMPLEMENTED_MATERIALIZED_VIEWS)
        )
        if unimplemented:
            views = ", ".join(unimplemented)
            raise ValueError(
                "diagnostic_view_not_implemented: "
                f"requested diagnostics materialized views are not implemented: "
                f"{views}."
            )
        if not enabled:
            return cls.disabled()
        return cls(
            enabled=True,
            policy_ids=policy_ids,
            materialized_views=materialized_views,
        )

    def to_public_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "policy_ids": list(self.policy_ids),
            "materialized_views": list(self.materialized_views),
        }


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{field} entries must be non-empty strings.")
        result.append(item)
    return tuple(result)


__all__ = [
    "DiagnosticsSpec",
    "IMPLEMENTED_MATERIALIZED_VIEWS",
    "REQUESTABLE_MATERIALIZED_VIEWS",
]
