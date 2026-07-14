from __future__ import annotations

import pytest

from quant_research_platform.policy.check_status import (
    CheckStatus,
    aggregate_check_status,
    aggregate_component_status,
    aggregate_validation_matrix_status,
    require_status,
    validate_verification_statuses,
)


def test_aggregate_status_keeps_not_run_visible_under_pass_parent() -> None:
    assert aggregate_check_status([CheckStatus.PASS, CheckStatus.NOT_RUN]) == "FAIL"


def test_component_status_treats_partial_execution_as_failure() -> None:
    assert aggregate_component_status(["PASS", "NOT_RUN"]) == "FAIL"


def test_validation_matrix_status_keeps_not_run_visible() -> None:
    assert aggregate_validation_matrix_status(["PASS", "NOT_RUN"]) == "NOT_RUN"


def test_aggregate_status_failure_order_is_explicit() -> None:
    assert aggregate_check_status(["PASS", "BLOCKED", "FAIL"]) == "FAIL"


def test_status_parser_rejects_unknown_tokens() -> None:
    with pytest.raises(ValueError, match="invalid_check_status"):
        require_status("SKIPPED")


@pytest.mark.parametrize("value", ("pass", " PASS", "PASS ", True, 1))
def test_status_parser_rejects_normalized_or_coerced_tokens(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_check_status"):
        require_status(value)


def test_verification_statuses_accept_exact_relative_evidence() -> None:
    result = validate_verification_statuses(
        smoke_status="PASS",
        compare_status="PASS",
        full_horizon_status="PASS",
        compare_evidence={
            "comparison_path": "outputs/compare/compare.json",
            "verification_scope": "same_window",
        },
        full_horizon_evidence={
            "evidence_path": "outputs/full/summary.json",
            "verification_scope": "full_horizon",
        },
    )

    assert result["validation_matrix_status"] == "PASS"


@pytest.mark.parametrize(
    "evidence",
    (
        {},
        {"comparison_path": "compare.json", "verification_scope": "scope", "x": 1},
        {"comparison_path": True, "verification_scope": "scope"},
        {"comparison_path": "/absolute/compare.json", "verification_scope": "scope"},
        {"comparison_path": "outputs\\compare.json", "verification_scope": "scope"},
        {"comparison_path": "outputs/../compare.json", "verification_scope": "scope"},
        {"comparison_path": " compare.json", "verification_scope": "scope"},
        {"comparison_path": "compare.json", "verification_scope": "Same-Window"},
    ),
)
def test_verification_statuses_reject_nonexact_compare_evidence(
    evidence: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="compare_evidence_missing_fields"):
        validate_verification_statuses(
            smoke_status="PASS",
            compare_status="PASS",
            full_horizon_status="NOT_RUN",
            compare_evidence=evidence,
            full_horizon_evidence=None,
        )


def test_not_run_rejects_even_empty_evidence_mapping() -> None:
    with pytest.raises(ValueError, match="compare_evidence_not_applicable"):
        validate_verification_statuses(
            smoke_status="PASS",
            compare_status="NOT_RUN",
            full_horizon_status="NOT_RUN",
            compare_evidence={},
            full_horizon_evidence=None,
        )
