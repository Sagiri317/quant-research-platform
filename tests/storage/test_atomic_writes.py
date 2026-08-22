from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.atomic as atomic
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
    publish_no_replace,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError


def test_atomic_directory_writer_failure_removes_only_operation_temp(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "nested" / "bundle"
    unrelated = tmp_path / ".a-user"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    def fail(stage: Path) -> None:
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("writer_failed")

    with pytest.raises(RuntimeError, match="writer_failed"):
        atomic_write_directory(final_root, fail)

    assert not final_root.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not list(tmp_path.glob(".a-[0-9a-f]*"))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_atomic_directory_preserves_control_flow_primary_and_cleans_temp(
    tmp_path: Path,
    exception: type[BaseException],
) -> None:
    final_root = tmp_path / "bundle"

    def cancel(stage: Path) -> None:
        (stage / "partial").write_text("partial", encoding="utf-8")
        raise exception("cancelled")

    with pytest.raises(exception, match="cancelled"):
        atomic_write_directory(final_root, cancel)

    assert not final_root.exists()
    assert not list(tmp_path.glob(".a-[0-9a-f]*"))


def test_atomic_write_text_creates_nested_parent_and_publishes(
    tmp_path: Path,
) -> None:
    final_path = tmp_path / "root" / "year=2014" / "part.txt"

    assert atomic_write_text(final_path, "ok\n") == final_path
    assert final_path.read_text(encoding="utf-8") == "ok\n"
    assert not list(final_path.parent.glob(".f-*"))


def test_atomic_json_writer_rejects_existing_target_without_overwrite(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "manifest.json"
    json_path.write_text("existing", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        atomic_write_json(json_path, {"ok": True})

    assert json_path.read_text(encoding="utf-8") == "existing"


def test_atomic_directory_rejects_existing_target_without_overwrite(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "bundle"
    final_root.mkdir()
    (final_root / "winner").write_text("winner", encoding="utf-8")

    with pytest.raises(OutputAlreadyExistsError):
        atomic_write_directory(final_root, lambda _stage: None)

    assert (final_root / "winner").read_text(encoding="utf-8") == "winner"


def test_publish_no_replace_preserves_concurrent_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "candidate.tmp"
    target = tmp_path / "output.txt"
    source.write_text("candidate", encoding="utf-8")

    def race(_source_path: Path, target_path: Path) -> None:
        target_path.write_text("winner", encoding="utf-8")
        raise FileExistsError(target_path)

    monkeypatch.setattr(atomic, "move_path_no_replace", race)
    with pytest.raises(OutputAlreadyExistsError):
        publish_no_replace(source, target, source_identity=source.lstat())

    assert source.read_text(encoding="utf-8") == "candidate"
    assert target.read_text(encoding="utf-8") == "winner"


def test_publish_rejects_replaced_claimed_source_without_deleting_foreign_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "candidate.tmp"
    target = tmp_path / "output.txt"
    source.write_text("owned", encoding="utf-8")
    identity = source.lstat()
    source.unlink()
    source.write_text("foreign", encoding="utf-8")

    with pytest.raises(ValueError, match="storage_scratch_identity_changed"):
        publish_no_replace(source, target, source_identity=identity)

    assert source.read_text(encoding="utf-8") == "foreign"
    assert not target.exists()


def test_atomic_write_text_rename_failure_cleans_operation_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_path = tmp_path / "output.txt"
    primary = PermissionError(5, "rename_primary")
    monkeypatch.setattr(
        atomic,
        "move_path_no_replace",
        lambda *_args: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(PermissionError, match="rename_primary") as caught:
        atomic_write_text(final_path, "owned")

    assert caught.value is primary
    assert not final_path.exists()
    assert not list(tmp_path.glob(".f-*"))


def test_atomic_directory_publishes_writer_result(tmp_path: Path) -> None:
    final_root = tmp_path / "bundle"

    def write(stage: Path) -> str:
        (stage / "done.txt").write_text("done", encoding="utf-8")
        return "published"

    assert atomic_write_directory(final_root, write) == "published"
    assert (final_root / "done.txt").read_text(encoding="utf-8") == "done"
