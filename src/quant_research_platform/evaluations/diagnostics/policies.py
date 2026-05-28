"""Diagnostic policy declarations."""

from __future__ import annotations

from dataclasses import dataclass

DISABLED_DIAGNOSTICS_POLICY_ID = "__diagnostics_disabled__"
N5_V1_POLICY_IDS = (
    "risk_path_proxy_v1",
    "cost_proxy_v1",
    "capacity_proxy_v1",
    "size_proxy_v1",
    "regime_proxy_v1",
    "a_share_assumption_v1",
)


@dataclass(frozen=True)
class DiagnosticPolicy:
    policy_id: str
    policy_family: str
    enabled: bool
    requested: bool
    materialized: bool
    policy_fingerprint: str
    diagnostic_status: str
    status_reason: str
    warning: str = ""
    output_tables: tuple[str, ...] = ()


def disabled_policy() -> DiagnosticPolicy:
    return DiagnosticPolicy(
        policy_id=DISABLED_DIAGNOSTICS_POLICY_ID,
        policy_family="diagnostics_contract",
        enabled=False,
        requested=False,
        materialized=True,
        policy_fingerprint="disabled",
        diagnostic_status="disabled",
        status_reason="diagnostics disabled by request",
    )


def n5_v1_policies(*, requested_views: tuple[str, ...]) -> tuple[DiagnosticPolicy, ...]:
    views = tuple(requested_views)
    return tuple(
        DiagnosticPolicy(
            policy_id=policy_id,
            policy_family=policy_id.removesuffix("_v1"),
            enabled=True,
            requested=True,
            materialized=True,
            policy_fingerprint="n5_v1_descriptive_proxy",
            diagnostic_status="ok",
            status_reason="descriptive post-run proxy diagnostics",
            output_tables=views,
        )
        for policy_id in N5_V1_POLICY_IDS
    )


__all__ = [
    "DISABLED_DIAGNOSTICS_POLICY_ID",
    "DiagnosticPolicy",
    "N5_V1_POLICY_IDS",
    "disabled_policy",
    "n5_v1_policies",
]
