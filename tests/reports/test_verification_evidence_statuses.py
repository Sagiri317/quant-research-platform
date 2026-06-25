from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.artifacts.evidence import write_run_summary_markdown
from quant_research_platform.reports.verification_evidence import (
    build_verification_evidence_payload,
    render_verification_evidence_markdown,
)


def _summary(**overrides):
    payload = {
        "run_id": "s1",
        "date_range": {"start_date": "2014-01-01", "end_date": "2016-12-31"},
        "manifests": {"dataset_count": 1},
        "timings": {"records": []},
        "validation_summary": {"status": "PASS"},
        "output_root": Path("verification-output") / "s1",
    }
    payload.update(overrides)
    return build_verification_evidence_payload(**payload)


def test_verification_status_allows_first_night_without_smoke_or_full_horizon() -> None:
    summary = _summary()

    assert summary["status"] == "NOT_RUN"
    assert summary["smoke_status"] == "NOT_RUN"
    assert summary["compare_status"] == "NOT_RUN"
    assert summary["full_horizon_status"] == "NOT_RUN"


def test_verification_status_rejects_smoke_without_compare_evidence() -> None:
    with pytest.raises(ValueError, match="compare_evidence_required"):
        _summary(smoke_status="PASS", compare_status="PASS")


def test_verification_status_rejects_empty_compare_evidence_shell() -> None:
    with pytest.raises(ValueError, match="compare_evidence_missing_fields"):
        _summary(
            smoke_status="PASS",
            compare_status="PASS",
            compare_evidence={"comparison_path": "smoke-compare.json"},
        )


def test_verification_status_accepts_smoke_with_real_compare_evidence() -> None:
    summary = _summary(
        smoke_status="PASS",
        compare_status="PASS",
        compare_evidence={
            "verification_scope": "aggregate_review_safe_parity",
            "comparison_path": "smoke-compare-s63-vs-s64.json",
        },
    )

    assert summary["smoke_status"] == "PASS"
    assert summary["compare_status"] == "PASS"
    assert summary["compare_evidence"]["comparison_path"].endswith(".json")


def test_verification_status_accepts_smoke_pass_with_compare_fail() -> None:
    summary = _summary(
        smoke_status="PASS",
        compare_status="FAIL",
        compare_evidence={
            "verification_scope": "aggregate_review_safe_parity",
            "comparison_path": "smoke-compare-old-baseline.json",
        },
    )

    assert summary["smoke_status"] == "PASS"
    assert summary["compare_status"] == "FAIL"
    assert summary["status"] == "FAIL"


def test_verification_status_rejects_smoke_fail_with_compare_pass() -> None:
    with pytest.raises(ValueError, match="compare_pass_requires_smoke_pass"):
        _summary(
            smoke_status="FAIL",
            compare_status="PASS",
            compare_evidence={
                "verification_scope": "aggregate_review_safe_parity",
                "comparison_path": "smoke-compare.json",
            },
        )


def test_verification_status_rejects_compare_when_smoke_not_run() -> None:
    with pytest.raises(ValueError, match="compare_status_requires_smoke"):
        _summary(compare_status="PASS", compare_evidence={"comparison_path": "x.json"})


def test_verification_status_rejects_full_horizon_pass_without_evidence() -> None:
    with pytest.raises(ValueError, match="full_horizon_evidence_required"):
        _summary(full_horizon_status="PASS")


def test_run_summary_markdown_renders_verification_statuses(tmp_path: Path) -> None:
    summary = _summary()

    path = write_run_summary_markdown(
        tmp_path / "summary.md",
        render_verification_evidence_markdown(summary),
    )

    text = path.read_text(encoding="utf-8")
    assert "- smoke_status: `NOT_RUN`" in text
    assert "- compare_status: `NOT_RUN`" in text
    assert "- full_horizon_status: `NOT_RUN`" in text
