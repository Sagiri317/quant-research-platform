from __future__ import annotations

from pathlib import Path

from quant_research_platform.policy.share_safety import (
    local_run_root_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_share_safety_rejects_local_path_and_sanitizes_it() -> None:
    text = "created at C:/Users/example/project/data/raw/source/file.csv"

    assert unsafe_for_share_reason(text) in {"local_path", "local_only_path"}
    assert "C:/Users" not in sanitize_shared_text(text)


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
