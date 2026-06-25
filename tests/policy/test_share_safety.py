from __future__ import annotations

from pathlib import Path

from quant_research_platform.policy.share_safety import (
    local_output_root_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
    zip_member_unsafe_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert unsafe_for_share_reason(text) in {"local_path", "local_only_path"}
    assert "C:/Users" not in sanitize_shared_text(text)


def test_share_safety_rejects_forbidden_zip_member() -> None:
    assert (
        zip_member_unsafe_reason("data/raw/source/file.parquet")
        == "local_only_path"
    )


def test_share_safety_allows_review_safe_handoff_evidence_name() -> None:
    assert zip_member_unsafe_reason("handoff_evidence/generation.json") is None


def test_local_output_root_accepts_owned_output_locations(tmp_path: Path) -> None:
    assert (
        local_output_root_reason(tmp_path / "run_outputs" / "local", tmp_path)
        is None
    )
    assert local_output_root_reason(tmp_path / ".run" / "smoke", tmp_path) is None


def test_local_output_root_rejects_unowned_location(tmp_path: Path) -> None:
    assert (
        local_output_root_reason(tmp_path / "reports" / "local", tmp_path)
        == "unignored_local_output_root"
    )
