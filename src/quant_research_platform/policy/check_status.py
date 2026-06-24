"""Validation status tokens and aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    BLOCKED = "BLOCKED"


VALID_STATUS_TOKENS = tuple(status.value for status in CheckStatus)
SMOKE_STATUSES = frozenset(VALID_STATUS_TOKENS)
COMPARE_STATUSES = frozenset(VALID_STATUS_TOKENS)
FULL_HORIZON_STATUSES = frozenset(VALID_STATUS_TOKENS)
COMPARE_EVIDENCE_FIELDS = frozenset({"comparison_path", "verification_scope"})
FULL_HORIZON_EVIDENCE_FIELDS = frozenset({"evidence_path", "verification_scope"})


def require_status(value: object) -> str:
    status = str(value).strip().upper()
    if status not in VALID_STATUS_TOKENS:
        raise ValueError(f"invalid_check_status: {value}")
    return status


def aggregate_check_status(statuses: Iterable[object]) -> str:
    values = tuple(require_status(status) for status in statuses)
    if not values:
        return CheckStatus.NOT_RUN.value
    if CheckStatus.FAIL.value in values:
        return CheckStatus.FAIL.value
    if CheckStatus.BLOCKED.value in values:
        return CheckStatus.BLOCKED.value
    if CheckStatus.NOT_RUN.value in values:
        if CheckStatus.PASS.value in values:
            return CheckStatus.FAIL.value
        return CheckStatus.NOT_RUN.value
    return CheckStatus.PASS.value


def validate_verification_statuses(
    *,
    smoke_status: str,
    compare_status: str,
    full_horizon_status: str,
    compare_evidence: dict[str, object] | None,
    full_horizon_evidence: dict[str, object] | None,
) -> dict[str, object]:
    smoke = smoke_status.upper()
    compare = compare_status.upper()
    full_horizon = full_horizon_status.upper()
    if smoke not in SMOKE_STATUSES:
        raise ValueError("invalid_smoke_status")
    if compare not in COMPARE_STATUSES:
        raise ValueError("invalid_compare_status")
    if full_horizon not in FULL_HORIZON_STATUSES:
        raise ValueError("invalid_full_horizon_status")
    if smoke == "NOT_RUN" and compare != "NOT_RUN":
        raise ValueError("compare_status_requires_smoke")
    if smoke == "FAIL" and compare == "PASS":
        raise ValueError("compare_pass_requires_smoke_pass")
    if compare in {"PASS", "FAIL"} and not compare_evidence:
        raise ValueError("compare_evidence_required")
    if compare == "NOT_RUN" and compare_evidence:
        raise ValueError("compare_evidence_not_applicable")
    if compare in {"PASS", "FAIL"}:
        _require_evidence_fields(
            compare_evidence,
            required_fields=COMPARE_EVIDENCE_FIELDS,
            error_name="compare_evidence_missing_fields",
        )
    if full_horizon == "NOT_RUN" and full_horizon_evidence:
        raise ValueError("full_horizon_evidence_not_applicable")
    if full_horizon != "NOT_RUN" and not full_horizon_evidence:
        raise ValueError("full_horizon_evidence_required")
    if full_horizon != "NOT_RUN":
        _require_evidence_fields(
            full_horizon_evidence,
            required_fields=FULL_HORIZON_EVIDENCE_FIELDS,
            error_name="full_horizon_evidence_missing_fields",
        )
    return {
        "smoke_status": smoke,
        "compare_status": compare,
        "full_horizon_status": full_horizon,
        "compare_evidence": dict(compare_evidence or {}),
        "full_horizon_evidence": dict(full_horizon_evidence or {}),
    }


def overall_verification_status(
    validation_status: object,
    statuses: dict[str, object],
) -> str:
    child_statuses = [str(validation_status).upper()]
    child_statuses.extend(
        str(statuses[key]) for key in ("smoke_status", "full_horizon_status")
    )
    child_statuses.append(str(statuses["compare_status"]))
    return aggregate_check_status(child_statuses)


def _require_evidence_fields(
    evidence: dict[str, object] | None,
    *,
    required_fields: frozenset[str],
    error_name: str,
) -> None:
    payload = evidence or {}
    missing = [
        field
        for field in sorted(required_fields)
        if not str(payload.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(error_name)


__all__ = [
    "CheckStatus",
    "VALID_STATUS_TOKENS",
    "aggregate_check_status",
    "overall_verification_status",
    "require_status",
    "validate_verification_statuses",
]
