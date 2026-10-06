from __future__ import annotations

import os
from pathlib import Path

import pytest

import quant_research_platform.storage.rebuild as rebuild
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout


@pytest.mark.skipif(os.name != "nt", reason="Windows UNC syntax")
def test_rebuild_uses_canonical_unc_paths_without_network_access(monkeypatch) -> None:
    created = []
    cleaned = []
    monkeypatch.setattr(
        rebuild, "require_unaliased_path", lambda path, **_kwargs: Path(path)
    )
    monkeypatch.setattr(
        rebuild,
        "filesystem_mkdir",
        lambda path, **_kwargs: created.append(str(native_filesystem_path(path))),
    )
    monkeypatch.setattr(
        rebuild, "remove_owned_scratch", lambda path: cleaned.append(str(path))
    )
    with rebuild.rebuild_candidate_workspace(
        StorageLayout(Path("\\\\server\\share\\project"))
    ) as candidate:
        assert str(candidate.project_root).startswith(
            "\\\\server\\share\\project\\.run\\.r-"
        )
    assert created[-1] == str(native_filesystem_path(candidate.project_root))
    assert cleaned == [str(candidate.project_root)]


@pytest.mark.parametrize("long_path", (False, True))
def test_rebuild_candidate_success_cleans_owner_scratch(
    tmp_path: Path, long_path: bool
) -> None:
    if long_path:
        while len(str(tmp_path)) < 280:
            tmp_path /= "segment_" + "a" * 32
    with rebuild.rebuild_candidate_workspace(
        StorageLayout(project_root=tmp_path)
    ) as candidate:
        root = candidate.project_root
        assert root.name.startswith(".r-")
        assert root.parent.name == ".run"
        filesystem_write_text(root / "evidence", "candidate", encoding="utf-8")
    assert not filesystem_exists(root)


def test_rebuild_candidate_explicit_preserve_keeps_evidence(tmp_path: Path) -> None:
    with rebuild.rebuild_candidate_workspace(
        StorageLayout(project_root=tmp_path)
    ) as candidate:
        root = candidate.project_root
        candidate.preserve()
    assert root.is_dir()


def test_rebuild_candidate_exception_preserves_primary_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
