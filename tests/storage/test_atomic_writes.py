from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.atomic as atomic
from quant_research_platform.storage import _scratch_ownership as scratch
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
    publish_no_replace,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.identity import native_filesystem_path


def _long_parent(tmp_path: Path) -> Path:
    parent = tmp_path
    while len(str(parent / "file")) <= 260:
        parent /= "segment_" + "a" * 32
    return parent


def test_atomic_directory_writer_failure_removes_only_operation_temp(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "nested" / "bundle"
    final_root.parent.mkdir()
    unrelated = final_root.parent / ".a-user"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    def fail(stage: Path) -> None:
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("writer_failed")

    with pytest.raises(RuntimeError, match="writer_failed"):
        atomic_write_directory(final_root, fail)
    assert not final_root.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert set(final_root.parent.iterdir()) == {unrelated}


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_atomic_directory_preserves_control_flow_primary_and_cleans_temp(
    tmp_path: Path, exception: type[BaseException]
) -> None:
    final_root = tmp_path / "bundle"

    def cancel(stage: Path) -> None:
        (stage / "partial").write_text("partial", encoding="utf-8")
        raise exception("cancelled")

    with pytest.raises(exception, match="cancelled"):
        atomic_write_directory(final_root, cancel)
    assert not final_root.exists()
    assert not list(tmp_path.glob(".a-[0-9a-f]*"))


def test_atomic_write_text_creates_nested_parent_and_publishes(tmp_path: Path) -> None:
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


@pytest.mark.parametrize("long_path", (False, True))
def test_publish_no_replace_preserves_concurrent_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, long_path: bool
) -> None:
    if long_path:
        tmp_path = _long_parent(tmp_path)
        assert len(str(tmp_path / "candidate.tmp")) > 260
        assert len(str(tmp_path / "output.txt")) > 260
        Path(native_filesystem_path(tmp_path)).mkdir(parents=True)
    source = tmp_path / "candidate.tmp"
    target = tmp_path / "output.txt"
    Path(native_filesystem_path(source)).write_text("candidate", encoding="utf-8")

    def race(_source_path: Path, target_path: Path) -> None:
        Path(native_filesystem_path(target_path)).write_text("winner", encoding="utf-8")
        raise FileExistsError(target_path)

    monkeypatch.setattr(atomic, "move_path_no_replace", race)
    with pytest.raises(OutputAlreadyExistsError):
        publish_no_replace(source, target)
    assert (
        Path(native_filesystem_path(source)).read_text(encoding="utf-8") == "candidate"
    )
    assert Path(native_filesystem_path(target)).read_text(encoding="utf-8") == "winner"


def test_atomic_write_text_rename_failure_cleans_operation_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_path = tmp_path / "output.txt"
    primary = PermissionError(5, "rename_primary")
    monkeypatch.setattr(
        atomic, "move_path_no_replace", lambda *_args: (_ for _ in ()).throw(primary)
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


def test_long_scratch_and_existing_target_use_native_io(tmp_path):
    parent = _long_parent(tmp_path)
    native_parent = Path(native_filesystem_path(parent))
    native_parent.mkdir(parents=True)
    existing = parent / "existing.txt"
    assert len(str(existing)) > 260
    Path(native_filesystem_path(existing)).write_text("old", encoding="utf-8")
    with pytest.raises(OutputAlreadyExistsError):
        atomic.reject_if_exists(existing)
    candidate = parent / ".f-owned"
    assert len(str(candidate)) > 260
    with scratch.open_owned_scratch_file(candidate, encoding="utf-8") as handle:
        handle.write("candidate")
    assert (
        Path(native_filesystem_path(candidate)).read_text(encoding="utf-8")
        == "candidate"
    )
    scratch.remove_owned_scratch(candidate)
    assert Path(native_filesystem_path(existing)).read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_atomic_long_paths_create_parents_keep_old_good_and_preserve_primary(
    tmp_path, monkeypatch, kind
):
    parent = _long_parent(tmp_path)
    final = parent / kind
    assert len(str(final)) > 260
    assert len(str(parent / "failed")) > 260
    if kind == "file":
        assert atomic_write_text(final, "old") == final

        class ForbiddenChunks:
            def __iter__(self):
                raise AssertionError("input_iterated_before_existing_target_gate")

        with pytest.raises(OutputAlreadyExistsError):
            atomic.atomic_write_chunks(final, ForbiddenChunks())
        assert Path(native_filesystem_path(final)).read_text(encoding="utf-8") == "old"
    else:
        atomic_write_directory(
            final, lambda stage: atomic_write_text(stage / "value", "old")
        )
        with pytest.raises(OutputAlreadyExistsError):
            atomic_write_directory(
                final, lambda _stage: pytest.fail("existing_target_entered_writer")
            )
        assert (
            Path(native_filesystem_path(final / "value")).read_text(encoding="utf-8")
            == "old"
        )
    unrelated = parent / ".a-user"
    Path(native_filesystem_path(unrelated)).mkdir()
    primary = KeyboardInterrupt("writer_primary")
    cleanup = atomic.remove_owned_scratch

    def failing_cleanup(path):
        cleanup(path)
        raise OSError("cleanup_secondary")

    def fail(stage):
        atomic_write_text(stage / "partial", "partial")
        raise primary

    monkeypatch.setattr(atomic, "remove_owned_scratch", failing_cleanup)
    with pytest.raises(KeyboardInterrupt) as caught:
        atomic_write_directory(parent / "failed", fail)
    assert caught.value is primary
    assert "atomic scratch cleanup also failed: OSError" in primary.__notes__
    assert {path.name for path in Path(native_filesystem_path(parent)).iterdir()} == {
        kind,
        ".a-user",
    }


def test_atomic_chunk_failure_closes_source_and_preserves_primary(
    tmp_path, monkeypatch
):
    closed = []
    primary = KeyboardInterrupt("chunk_write_primary")
    open_file = atomic.open_owned_scratch_file

    class FailingHandle:
        def __init__(self, handle):
            self.handle = handle

        def write(self, _text):
            raise primary

        def close(self):
            self.handle.close()

    def source():
        try:
            yield "owned payload"
        finally:
            closed.append(True)
            raise OSError("iterator_close_secondary")

    monkeypatch.setattr(
        atomic,
        "open_owned_scratch_file",
        lambda *args, **kwargs: FailingHandle(open_file(*args, **kwargs)),
    )
    chunks = source()
    with pytest.raises(KeyboardInterrupt) as caught:
        atomic.atomic_write_chunks(tmp_path / "failed.txt", chunks)
    assert caught.value is primary
    assert closed == [True] and chunks.gi_frame is None
    assert "iterator close also failed: OSError" in primary.__notes__
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("message", (None, "", "custom_rejection"))
def test_existing_output_error_preserves_contract_and_exception_chain(
    tmp_path, message
):
    error = OutputAlreadyExistsError(message)
    assert isinstance(error, ValueError)
    assert error.failure_code == "output_already_exists"
    assert str(error) == (message or "output_already_exists")
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"candidate")
    target.write_bytes(b"old_good")
    with pytest.raises(
        OutputAlreadyExistsError, match="^output_already_exists$"
    ) as caught:
        publish_no_replace(source, target)
    assert isinstance(caught.value.__cause__, FileExistsError)
    assert source.read_bytes() == b"candidate"
    assert target.read_bytes() == b"old_good"
