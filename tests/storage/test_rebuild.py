from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.rebuild as rebuild
from quant_research_platform.storage.layout import StorageLayout


def test_rebuild_candidate_success_cleans_owner_scratch(tmp_path: Path) -> None:
    with rebuild.rebuild_candidate_workspace(
        StorageLayout(project_root=tmp_path)
    ) as candidate:
        root = candidate.project_root
        assert root.name.startswith(".r-")
        assert root.parent.name == ".run"
        (root / "evidence").write_text("candidate", encoding="utf-8")

    assert not root.exists()


def test_rebuild_candidate_explicit_preserve_keeps_evidence(tmp_path: Path) -> None:
    with rebuild.rebuild_candidate_workspace(
        StorageLayout(project_root=tmp_path)
    ) as candidate:
        root = candidate.project_root
        candidate.preserve()

    assert root.is_dir()


def test_rebuild_candidate_exception_preserves_primary_and_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PrimaryError(Exception):
        pass

    cleanup_calls: list[Path] = []

    def unexpected_cleanup(path: Path) -> None:
        cleanup_calls.append(path)
        raise AssertionError("primary path must preserve candidate")

    monkeypatch.setattr(rebuild, "remove_owned_scratch", unexpected_cleanup)
    primary = PrimaryError("producer failure")
    with pytest.raises(PrimaryError) as caught:
        with rebuild.rebuild_candidate_workspace(
            StorageLayout(project_root=tmp_path)
        ) as candidate:
            root = candidate.project_root
            (root / "evidence").write_text("candidate", encoding="utf-8")
            raise primary

    assert caught.value is primary
    assert cleanup_calls == []
    assert root.is_dir()


def test_rebuild_candidate_cleanup_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_cleanup(_path: Path) -> None:
        raise OSError("cleanup failure")

    monkeypatch.setattr(rebuild, "remove_owned_scratch", fail_cleanup)
    with pytest.raises(OSError, match="cleanup failure"):
        with rebuild.rebuild_candidate_workspace(
            StorageLayout(project_root=tmp_path)
        ) as candidate:
            root = candidate.project_root

    assert root.is_dir()
