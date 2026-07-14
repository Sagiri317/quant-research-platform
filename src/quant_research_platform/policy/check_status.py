"""Validation status tokens and aggregation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import PurePosixPath


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
_VERIFICATION_SCOPE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def require_status(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_STATUS_TOKENS:
        raise ValueError(f"invalid_check_status: {value}")
    return str(value)


def aggregate_check_status(statuses: Iterable[object]) -> str:
    return aggregate_component_status(statuses)


def aggregate_component_status(statuses: Iterable[object]) -> str:
    """Aggregate checks that must all execute and pass."""

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


def aggregate_validation_matrix_status(statuses: Iterable[object]) -> str:
    """Aggregate independently runnable validation matrix entries."""

    values = tuple(require_status(status) for status in statuses)
    if not values:
        return CheckStatus.NOT_RUN.value
    if CheckStatus.FAIL.value in values:
        return CheckStatus.FAIL.value
    if CheckStatus.BLOCKED.value in values:
        return CheckStatus.BLOCKED.value
    if CheckStatus.NOT_RUN.value in values:
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
    if not isinstance(smoke_status, str) or smoke_status not in SMOKE_STATUSES:
        raise ValueError("invalid_smoke_status")
    if not isinstance(compare_status, str) or compare_status not in COMPARE_STATUSES:
        raise ValueError("invalid_compare_status")
    if (
        not isinstance(full_horizon_status, str)
        or full_horizon_status not in FULL_HORIZON_STATUSES
    ):
        raise ValueError("invalid_full_horizon_status")
    smoke = str(smoke_status)
    compare = str(compare_status)
    full_horizon = str(full_horizon_status)
    if smoke == "NOT_RUN" and compare != "NOT_RUN":
        raise ValueError("compare_status_requires_smoke")
    if smoke == "FAIL" and compare == "PASS":
        raise ValueError("compare_pass_requires_smoke_pass")
    if compare in {"PASS", "FAIL"} and compare_evidence is None:
        raise ValueError("compare_evidence_required")
    if compare == "NOT_RUN" and compare_evidence is not None:
        raise ValueError("compare_evidence_not_applicable")
    if compare_evidence is not None:
        _require_evidence_fields(
            compare_evidence,
            required_fields=COMPARE_EVIDENCE_FIELDS,
            error_name="compare_evidence_missing_fields",
        )
    if full_horizon == "NOT_RUN" and full_horizon_evidence is not None:
        raise ValueError("full_horizon_evidence_not_applicable")
    if full_horizon != "NOT_RUN" and full_horizon_evidence is None:
        raise ValueError("full_horizon_evidence_required")
    if full_horizon != "NOT_RUN":
        _require_evidence_fields(
            full_horizon_evidence,
            required_fields=FULL_HORIZON_EVIDENCE_FIELDS,
            error_name="full_horizon_evidence_missing_fields",
        )
    validation_matrix_status = aggregate_validation_matrix_status(
        (smoke, compare, full_horizon)
    )
    return {
        "smoke_status": smoke,
        "compare_status": compare,
        "full_horizon_status": full_horizon,
        "validation_matrix_status": validation_matrix_status,
        "compare_evidence": {} if compare_evidence is None else dict(compare_evidence),
        "full_horizon_evidence": (
            {} if full_horizon_evidence is None else dict(full_horizon_evidence)
        ),
    }


def overall_verification_status(
    validation_status: object,
    statuses: dict[str, object],
) -> str:
    return aggregate_validation_matrix_status(
        (
            validation_status,
            statuses["validation_matrix_status"],
        )
    )


def _require_evidence_fields(
    evidence: dict[str, object] | None,
    *,
    required_fields: frozenset[str],
    error_name: str,
) -> None:
    if type(evidence) is not dict or set(evidence) != required_fields:
        raise ValueError(error_name)
    for field in sorted(required_fields):
        value = evidence[field]
        if type(value) is not str or not value or value != value.strip():
            raise ValueError(error_name)
        if field == "verification_scope":
            if _VERIFICATION_SCOPE_RE.fullmatch(value) is None:
                raise ValueError(error_name)
        else:
            path = PurePosixPath(value)
            if (
                "\\" in value
                or path.is_absolute()
                or path.as_posix() != value
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(error_name)


__all__ = [
    "CheckStatus",
    "VALID_STATUS_TOKENS",
    "aggregate_component_status",
    "aggregate_check_status",
    "aggregate_validation_matrix_status",
    "overall_verification_status",
    "require_status",
    "validate_verification_statuses",
]
