from __future__ import annotations

from pathlib import Path

from quant_research_platform.policy.identifiers import unsafe_identifier_reason
from quant_research_platform.policy.share_safety import (
    local_run_root_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
    zip_member_unsafe_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert unsafe_for_share_reason(text) in {"local_path", "local_only_path"}
    assert "C:/Users" not in sanitize_shared_text(text)


def test_share_safety_rejects_forbidden_zip_member() -> None:
    assert zip_member_unsafe_reason("data/raw/source/file.parquet") == "local_only_path"
    assert zip_member_unsafe_reason("outputs/prediction_daily/result.parquet") == (
        "local_only_path"
    )


def test_share_safety_does_not_treat_output_identifier_as_path() -> None:
    assert unsafe_identifier_reason("prediction_outputs") is None


def test_share_safety_allows_review_safe_handoff_evidence_name() -> None:
    assert zip_member_unsafe_reason("handoff_evidence/validation-summary.json") is None


def test_local_run_root_accepts_explicit_locations(tmp_path: Path) -> None:
    assert (
        local_run_root_reason(tmp_path / "outputs" / "verification-run", tmp_path)
        is None
    )
    assert local_run_root_reason(tmp_path / ".run" / "smoke", tmp_path) is None
    assert local_run_root_reason(tmp_path / "reports" / "local", tmp_path) is None
    assert local_run_root_reason(tmp_path.parent / "external" / "run", tmp_path) is None


def test_local_run_root_rejects_repository_root(tmp_path: Path) -> None:
    assert local_run_root_reason(tmp_path, tmp_path) == "run_root_must_not_be_repo_root"
