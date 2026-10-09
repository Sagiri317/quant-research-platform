from __future__ import annotations

import io
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.storage.datasets.comparison as comparison
import quant_research_platform.storage.datasets.inventory as inventory_owner
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.identity as identity
import quant_research_platform.storage.parquet as physical
from quant_research_platform.storage import _scratch_ownership as scratch_owner
from quant_research_platform.storage import atomic, manifests

_PRIMARY_EXCEPTION_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


@pytest.mark.parametrize("primary_type", (*_PRIMARY_EXCEPTION_TYPES, OSError))
def test_content_hash_close_preserves_primary_and_closes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primary_type
) -> None:
    path = tmp_path / "hash.bin"
    identity.filesystem_write_bytes(path, b"synthetic")
    raw = identity.open_filesystem_file(path, "rb")
    primary = primary_type("hash_primary")
    closed = []

    class Handle:
        def fileno(self):
            return raw.fileno()

        def read(self, _size):
            raise primary

        def close(self):
            closed.append(True)
            raw.close()
            raise OSError("close_secondary")

    monkeypatch.setattr(
        identity, "open_filesystem_file", lambda *_args, **_kwargs: Handle()
    )
    try:
        with pytest.raises(primary_type) as raised:
            identity.content_hash_for_files(
                (("member", path),), error="identity_changed"
            )
        assert raised.value is primary
        assert raw.closed and closed == [True]
        assert raised.value.__notes__ == ["iterator close also failed: OSError"]
    finally:
        raw.close()


@pytest.mark.parametrize("long_path", (False, True))
def test_shared_path_move_is_fail_if_destination_exists(
    tmp_path: Path, long_path
) -> None:
    shared = __import__(
        "quant_research_platform.storage._scratch_ownership", fromlist=["*"]
    )
    parent = tmp_path

    def native(path):
        from quant_research_platform.storage.identity import native_filesystem_path

        return Path(native_filesystem_path(path))

    if long_path:
        parent = parent / ("nested" * 20) / ("nested" * 20)
        native(parent).mkdir(parents=True)
    source = parent / "source"
    destination = parent / "destination"
    native(source).write_text("source", encoding="utf-8")
    native(destination).write_text("destination", encoding="utf-8")
    with pytest.raises(FileExistsError):
        shared.move_path_no_replace(source, destination)
    assert native(source).read_text(encoding="utf-8") == "source"
    assert native(destination).read_text(encoding="utf-8") == "destination"
    moved = parent / "moved"
    shared.move_path_no_replace(source, moved)
    assert native(moved).read_text(encoding="utf-8") == "source"
    assert not native(source).exists()


def test_write_and_scratch_owners_reject_aliased_parent_before_access(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    identity.filesystem_mkdir(target)
    alias = tmp_path / "alias"
    try:
        Path(identity.native_filesystem_path(alias)).symlink_to(
            identity.native_filesystem_path(target), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")
    operations = (
        lambda: atomic.atomic_write_text(alias / "text.txt", "payload"),
        lambda: atomic.atomic_write_directory(alias / "tree", lambda _root: None),
        lambda: manifests.write_manifest(
            alias / "manifest.json", {"schema_id": "qrp-test"}
        ),
    )
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    assert not tuple(identity.filesystem_iterdir(target))


def test_unaliased_path_checks_lexical_component_before_parent_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    alias = tmp_path / "alias"
    visited: list[Path] = []

    def is_alias(path: Path) -> bool:
        visited.append(path)
        return path == alias

    monkeypatch.setattr(identity, "path_is_filesystem_alias", is_alias)
    with pytest.raises(ValueError, match="invalid_path"):
        identity.require_unaliased_path(alias / ".." / "artifact", error="invalid_path")
    assert alias in visited


class _BodyAndCloseFailingHandle:
    def __init__(self, primary: BaseException) -> None:
        self.primary = primary

    def __enter__(self) -> _BodyAndCloseFailingHandle:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def fileno(self) -> int:
        return 1

    def read(self, *_args: object) -> bytes:
        raise self.primary

    def write(self, *_args: object) -> int:
        raise self.primary

    def flush(self) -> None:
        pass

    def close(self) -> None:
        raise OSError("close_secondary")


@pytest.mark.parametrize("long_path", (False, True))
def test_dataset_write_read_long_path_preserves_existing_publication(
    tmp_path: Path, long_path: bool
) -> None:
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    parent = tmp_path
    if long_path:
        while len(str(parent)) < 280:
            parent /= "segment_" + "a" * 32
    root = make_dataset_artifact_root(parent, "native_io")
    native_root = Path(identity.native_filesystem_path(root))
    frame = make_default_dataset_frame().astype({"timestamp": "datetime64[ns]"})
    spec = make_processed_dataset_spec("native_io")
    dataset_writing.write_dataset(root, frame, spec=spec)
    actual = dataset_reading.read_dataset(root)
    pd.testing.assert_frame_equal(
        actual.sort_values(list(spec.primary_key)).reset_index(drop=True),
        frame.sort_values(list(spec.primary_key)).reset_index(drop=True),
    )
    original_bytes = {
        path.relative_to(native_root): path.read_bytes()
        for path in native_root.rglob("*")
        if path.is_file()
    }
    members = set(native_root.parent.iterdir())
    with pytest.raises(ValueError, match="dataset_exists"):
        dataset_writing.write_dataset(root, frame, spec=spec)
    assert set(native_root.parent.iterdir()) == members
    assert {
        path.relative_to(native_root): path.read_bytes()
        for path in native_root.rglob("*")
        if path.is_file()
    } == original_bytes
    pd.testing.assert_frame_equal(
        dataset_reading.read_dataset(root), actual, check_exact=True
    )


def test_opened_parquet_close_preserves_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def fileno(self) -> int:
            return 1

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\x00\x00\x00\x00PAR1"

    class Parquet:
        def close(self) -> None:
            raise RuntimeError("close_failed")

    metadata = type(
        "Stat", (), {"st_dev": 1, "st_ino": 2, "st_size": 3, "st_mtime_ns": 4}
    )()
    monkeypatch.setattr(physical, "open_filesystem_file", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(
        identity, "dataset_file_identity", physical.dataset_file_identity
    )
    monkeypatch.setattr(
        physical.pq, "ParquetFile", lambda _handle, **_kwargs: Parquet()
    )
    with pytest.raises(GeneratorExit):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise GeneratorExit()


@pytest.mark.parametrize(
    "payload", (b"\x00\x00\x00\x00NOPE", (9).to_bytes(4, "little") + b"PAR1")
)
def test_parquet_footer_rejects_invalid_trailer_without_constructor(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        physical._parquet_footer_length(io.BytesIO(payload))


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_write_dataset_activates_and_closes_caller_iterator(
    primary_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    primary = primary_type("body_primary")

    class InputIterator:
        def __iter__(self) -> InputIterator:
            return self

        def __next__(self) -> pd.DataFrame:
            raise StopIteration

        def close(self) -> None:
            events.append("close")

    class CallerIterable:
        def __iter__(self) -> InputIterator:
            events.append("iter")
            return InputIterator()

    monkeypatch.setattr(
        dataset_writing, "data_spec_payload", lambda _spec, **_kwargs: {}
    )
    monkeypatch.setattr(
        dataset_writing, "validate_dataset_artifact_root", lambda *_args: None
    )
    monkeypatch.setattr(
        dataset_writing,
        "atomic_write_directory",
        lambda *_args: (_ for _ in ()).throw(primary),
    )
    with pytest.raises(primary_type) as raised:
        dataset_writing.write_dataset(Path("dataset"), CallerIterable(), spec=object())
    assert raised.value is primary
    assert events == ["iter", "close"]


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_opened_parquet_raw_handle_close_preserves_active_primary(
    primary_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = primary_type("body_primary")

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\x00\x00\x00\x00PAR1"

        def close(self) -> None:
            raise OSError("raw_close_secondary")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(physical, "open_filesystem_file", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(
        identity, "dataset_file_identity", physical.dataset_file_identity
    )
    monkeypatch.setattr(
        physical.pq, "ParquetFile", lambda _handle, **_kwargs: Parquet()
    )
    with pytest.raises(primary_type) as raised:
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            raise primary
    assert raised.value is primary


def test_opened_parquet_raw_handle_close_propagates_without_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def fileno(self) -> int:
            return 1

        def tell(self) -> int:
            return getattr(self, "position", 0)

        def seek(self, offset: int, whence: int = 0) -> int:
            if whence == 2:
                self.position = 8 + offset
            elif whence == 0:
                self.position = offset
            else:
                self.position += offset
            return self.position

        def read(self, _size: int = -1) -> bytes:
            return b"\x00\x00\x00\x00PAR1"

        def close(self) -> None:
            raise OSError("raw_close_failed")

    class Parquet:
        def close(self) -> None:
            pass

    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(physical, "open_filesystem_file", lambda *_a, **_k: Handle())
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(
        physical,
        "dataset_file_identity",
        lambda _path: identity.dataset_file_identity_from_stat(metadata),
    )
    monkeypatch.setattr(
        identity, "dataset_file_identity", physical.dataset_file_identity
    )
    monkeypatch.setattr(
        physical.pq, "ParquetFile", lambda _handle, **_kwargs: Parquet()
    )
    with pytest.raises(OSError, match="raw_close_failed"):
        with physical._opened_parquet(Path("part.parquet"), expected_identity=None):
            pass


@pytest.mark.parametrize(
    "primary", (MemoryError(), KeyboardInterrupt(), SystemExit(), GeneratorExit())
)
def test_atomic_directory_cleanup_preserves_active_primary(
    primary: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        atomic,
        "remove_owned_scratch",
        lambda _path: (_ for _ in ()).throw(RuntimeError("cleanup_failed")),
    )

    def writer(_path: Path) -> None:
        raise primary

    with pytest.raises(type(primary)):
        atomic.atomic_write_directory(tmp_path / "result", writer)


@pytest.mark.parametrize("primary_type", _PRIMARY_EXCEPTION_TYPES)
def test_bounded_manifest_read_preserves_body_primary_over_handle_close(
    primary_type: type[BaseException], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    identity.filesystem_write_text(path, '{"schema_id":"qrp-test"}', encoding="utf-8")
    metadata = identity.filesystem_stat(path)
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        manifests,
        "open_filesystem_file",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )
    monkeypatch.setattr(manifests.os, "fstat", lambda _fd: metadata)
    with pytest.raises(primary_type, match="body_primary"):
        manifests.read_qrp_json_file(path)


def test_compare_expected_storage_validation_still_maps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    identity.filesystem_mkdir(left)
    identity.filesystem_mkdir(right)
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )
    assert comparison.compare_datasets(left, right).status == "INSPECTION_FAILED"


def test_artifact_root_alias_is_rejected_by_unique_root_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("dataset")
    absolute_root = Path.cwd() / root
    monkeypatch.setattr(
        identity,
        "path_is_filesystem_alias",
        lambda candidate: candidate == absolute_root,
    )
    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.require_artifact_root(root)


@pytest.mark.parametrize("position", ("root", "immediate", "deep"))
@pytest.mark.parametrize("alias_kind", ("symlink", "junction"))
def test_artifact_root_rejects_exact_or_ancestor_alias(
    tmp_path: Path, position: str, alias_kind: str
) -> None:
    target = tmp_path / "target"
    identity.filesystem_mkdir(target)
    alias = tmp_path / "alias"
    if alias_kind == "symlink":
        try:
            Path(identity.native_filesystem_path(alias)).symlink_to(
                identity.native_filesystem_path(target), target_is_directory=True
            )
        except OSError as exc:
            pytest.skip(f"symlink_not_available: {exc}")
    else:
        if os.name != "nt" or not hasattr(Path, "is_junction"):
            pytest.skip("junction_capability_required")
        created = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(alias), str(target)],
            check=False,
            capture_output=True,
        )
        if created.returncode != 0:
            pytest.skip("junction_not_available")
    root = (
        alias
        if position == "root"
        else alias / "artifact"
        if position == "immediate"
        else alias / "nested" / "artifact"
    )
    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.require_artifact_root(root)


def test_artifact_root_ordinary_chain_preserves_candidate_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("relative") / "dataset"
    monkeypatch.setattr(identity, "path_is_filesystem_alias", lambda _candidate: False)
    assert identity.require_artifact_root(root) == root


def test_dataset_file_identity_accepts_single_external_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    identity.filesystem_write_bytes(source, b"payload")
    try:
        os.link(
            identity.native_filesystem_path(source),
            identity.native_filesystem_path(alias),
        )
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")
    assert identity.dataset_file_identity(alias) == identity.dataset_file_identity(
        source
    )


def test_dataset_file_identity_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    identity.filesystem_mkdir(target)
    identity.filesystem_write_bytes(target / "part.bin", b"payload")
    alias = tmp_path / "alias"
    try:
        Path(identity.native_filesystem_path(alias)).symlink_to(
            identity.native_filesystem_path(target), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")
    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        identity.dataset_file_identity(alias / "part.bin")


def test_bounded_metadata_reader_rejects_ancestor_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    identity.filesystem_mkdir(target)
    identity.filesystem_write_text(
        target / "manifest.json", '{"schema_id":"test"}', encoding="utf-8"
    )
    alias = tmp_path / "alias"
    try:
        Path(identity.native_filesystem_path(alias)).symlink_to(
            identity.native_filesystem_path(target), target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")
    with pytest.raises(ValueError, match="^dataset_metadata_file_invalid$"):
        manifests.read_qrp_json_file(alias / "manifest.json")


def test_write_dataset_requires_artifact_root_before_writer_or_data_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dataset_writing, "data_spec_payload", lambda _spec, **_kwargs: {}
    )
    monkeypatch.setattr(
        dataset_writing, "validate_dataset_artifact_root", lambda *_args: None
    )
    monkeypatch.setattr(
        dataset_writing,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
        raising=False,
    )
    monkeypatch.setattr(
        dataset_writing,
        "_write_contents",
        lambda *_args, **_kwargs: pytest.fail("writer_before_artifact_root_gate"),
    )

    class ForbiddenChunks:
        def __iter__(self):
            raise AssertionError("input_iterated_before_artifact_root_gate")

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        dataset_writing.write_dataset(Path("dataset"), ForbiddenChunks(), spec=object())


def test_compare_rechecks_artifact_roots_before_each_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    validations = {left: 0, right: 0}
    probes = {left: 0, right: 0}
    original_exists = comparison.filesystem_exists

    def require_root(path: Path) -> Path:
        candidate = Path(path)
        validations[candidate] += 1
        return candidate

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate not in probes:
            return original_exists(path)
        probes[candidate] += 1
        assert validations[candidate] == probes[candidate]
        return True

    monkeypatch.setattr(
        comparison, "require_artifact_root", require_root, raising=False
    )
    monkeypatch.setattr(comparison, "filesystem_exists", exists)
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalid_dataset")),
    )
    result = comparison.compare_datasets(left, right)
    assert result.status == "INSPECTION_FAILED"
    assert validations == {left: 1, right: 1}
    assert probes == {left: 1, right: 1}


def test_compare_maps_aliased_root_before_exists_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = Path("left")
    right = Path("right")
    original_exists = comparison.filesystem_exists

    def reject_alias(path: Path) -> Path:
        if Path(path) == left:
            raise ValueError("invalid_dataset_part_files")
        return Path(path)

    monkeypatch.setattr(
        comparison, "require_artifact_root", reject_alias, raising=False
    )

    def exists(path: Path) -> bool:
        if Path(path) in (left, right):
            raise AssertionError("dataset_exists_before_artifact_root_alias_gate")
        return original_exists(path)

    monkeypatch.setattr(comparison, "filesystem_exists", exists)
    result = comparison.compare_datasets(left, right)
    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_maps_ancestor_alias_before_public_exists_probe(
    monkeypatch: pytest.MonkeyPatch, alias_side: str
) -> None:
    left = Path("left-parent") / "left"
    right = Path("right-parent") / "right"
    aliased_parent = Path.cwd() / (left if alias_side == "left" else right).parent
    monkeypatch.setattr(
        identity,
        "path_is_filesystem_alias",
        lambda candidate: candidate == aliased_parent,
    )
    original_exists = comparison.filesystem_exists

    def exists(path: Path) -> bool:
        candidate = Path(path)
        if candidate == (left if alias_side == "left" else right):
            raise AssertionError("dataset_exists_before_ancestor_alias_gate")
        if candidate in (left, right):
            return True
        return original_exists(path)

    monkeypatch.setattr(comparison, "filesystem_exists", exists)
    result = comparison.compare_datasets(left, right)
    assert result.status == "INSPECTION_FAILED"
    assert result.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("alias_side", ("left", "right"))
def test_compare_internal_recheck_blocks_new_ancestor_alias_before_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_side: str
) -> None:
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    left = make_dataset_artifact_root(tmp_path / "left-parent", "same")
    right = make_dataset_artifact_root(tmp_path / "right-parent", "same")
    for root in (left, right):
        dataset_writing.write_dataset(
            root, make_default_dataset_frame(), spec=make_processed_dataset_spec("same")
        )
    assert comparison.compare_datasets(left, right).status == "MATCH"
    target = left if alias_side == "left" else right
    aliased_parent = target.parent
    activated = False
    alias_hits = 0
    rejected = []
    original_require = dataset_reading.require_artifact_root
    original_alias = identity.path_is_filesystem_alias

    def require_root(path: Path) -> Path:
        nonlocal activated
        if path == target:
            activated = True
        try:
            return original_require(path)
        except ValueError as error:
            rejected.append(str(error))
            raise

    def is_alias(candidate: Path) -> bool:
        nonlocal alias_hits
        if activated and candidate == aliased_parent:
            alias_hits += 1
            return True
        return original_alias(candidate)

    monkeypatch.setattr(dataset_reading, "require_artifact_root", require_root)
    monkeypatch.setattr(identity, "path_is_filesystem_alias", is_alias)
    result = comparison.compare_datasets(left, right)
    assert result.status == "INSPECTION_FAILED"
    assert activated
    assert alias_hits == 1
    assert rejected == ["invalid_dataset_part_files"]


def test_shared_scratch_removes_only_exact_owned_paths(tmp_path: Path) -> None:
    owned_file = tmp_path / "owned-file"
    identity.filesystem_write_bytes(owned_file, b"x")
    owned_directory = tmp_path / "owned-directory"
    identity.filesystem_mkdir(owned_directory)
    identity.filesystem_write_bytes(owned_directory / "part", b"x")
    scratch_owner.remove_owned_scratch(owned_file)
    scratch_owner.remove_owned_scratch(owned_directory)
    assert not identity.filesystem_exists(owned_file)
    assert not identity.filesystem_exists(owned_directory)


def test_content_hash_rechecks_earlier_members_before_return(tmp_path, monkeypatch):
    first, last = (tmp_path / "first", tmp_path / "last")
    identity.filesystem_write_bytes(first, b"original")
    identity.filesystem_write_bytes(last, b"last")
    real_hash = identity._hash_file_content

    def mutate_earlier(path, *args, **kwargs):
        result = real_hash(path, *args, **kwargs)
        if path == last:
            identity.filesystem_write_bytes(
                first, b"changed after its own final verification"
            )
        return result

    monkeypatch.setattr(identity, "_hash_file_content", mutate_earlier)
    with pytest.raises(ValueError, match="changed"):
        identity.content_hash_for_files(
            (("first", first), ("last", last)), error="changed"
        )


@pytest.mark.parametrize("boundary", ("stat", "open"))
@pytest.mark.parametrize("failure", ("missing", "memory", "io", "interrupt"))
def test_content_identity_distinguishes_invalid_from_fatal(
    tmp_path, monkeypatch, boundary, failure
):
    import errno

    path = tmp_path / "member"
    identity.filesystem_write_bytes(path, b"synthetic")
    primary = {
        "missing": FileNotFoundError(errno.ENOENT, "missing"),
        "memory": MemoryError("primary"),
        "io": OSError(errno.EIO, "primary"),
        "interrupt": KeyboardInterrupt("primary"),
    }[failure]

    def fail(*args, **kwargs):
        raise primary

    with monkeypatch.context() as patch:
        if boundary == "stat":
            patch.setattr(identity.os, "stat", fail)
        else:
            patch.setattr(identity, "open_filesystem_file", fail)
        with pytest.raises(
            ValueError if failure == "missing" else type(primary)
        ) as caught:
            identity.content_hash_for_files((("member", path),), error="changed")
        if failure != "missing":
            assert caught.value is primary


@pytest.mark.parametrize("source", ("choice", "csmar", "reference"))
@pytest.mark.parametrize("boundary", ("stat", "open"))
def test_source_identity_preserves_serious_io(tmp_path, monkeypatch, source, boundary):
    import errno

    from quant_research_platform.sources.choice import interim as choice
    from quant_research_platform.sources.choice.raw_exports import choice_session
    from quant_research_platform.sources.csmar import interim as csmar
    from quant_research_platform.sources.reference import interim as reference
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.choice_raw import SDK, response, write_css
    from tests.sources.reference.test_interim import _write_reference_root

    layout = StorageLayout(tmp_path)
    if source == "choice":
        with choice_session(SDK(response())) as session:
            write_css(
                session,
                layout,
                period="2019-12-31",
                end_date="2025-12-31",
                indicators=("STMTACTDATE",),
                data={"000001.SZ": ["2020/1/30"]},
            )
        root, data_id = (layout.raw_source_root("choice"), "financial_balance")
        path = next(identity.filesystem_glob(root, "**/response.metadata.json"))
        owner = choice
    elif source == "csmar":
        root, data_id = (tmp_path, "trading_calendar")
        path = root / "native-calendar.csv"
        identity.filesystem_write_text(
            path, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
        )
        owner = csmar
    else:
        root = _write_reference_root(tmp_path / "raw") / "reference"
        data_id = reference.A_SHARE_FEE_SCHEDULE_DATA_ID
        path = next(identity.filesystem_glob(root, "**/*.csv"))
        owner = reference
    primary = OSError(errno.EIO, "primary")
    from quant_research_platform.sources.csmar import raw_schemas
    from quant_research_platform.sources.reference import a_share_fees

    target_owner = (
        identity
        if boundary == "stat"
        else manifests
        if source == "choice"
        else raw_schemas
        if source == "csmar"
        else a_share_fees
    )
    operation = "filesystem_stat" if boundary == "stat" else "open_filesystem_file"
    original = getattr(target_owner, operation)

    def fail(candidate, *args, **kwargs):
        if candidate == path or candidate == str(path):
            raise primary
        return original(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target_owner, operation, fail)
        with pytest.raises(OSError) as caught:
            owner.source_content_identity(root, data_id)
        assert caught.value is primary


@pytest.mark.parametrize(
    "boundary",
    (
        "dataset_inventory",
        "dataset_inventory_next",
        "dataset_instances_scan",
        "dataset_instances_stat",
        "dataset_instances_manifest",
        "semantic_instances_scan",
        "semantic_instances_stat",
        "semantic_members_scan",
        "semantic_member_stat",
        "semantic_member_hash",
    ),
)
@pytest.mark.parametrize("error_type", (OSError, PermissionError))
def test_persisted_storage_boundaries_preserve_serious_io(
    tmp_path, monkeypatch, boundary, error_type
):
    from quant_research_platform.storage import semantic_artifacts as semantic
    from quant_research_platform.storage.datasets import instances
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_processed_dataset_spec,
    )
    from tests.storage.test_dataset_instances import _envelope, _write_instance
    from tests.storage.test_semantic_artifacts import _write_artifact

    primary = error_type(5, "synthetic_storage_io")
    if boundary.startswith("dataset"):
        logical = make_dataset_artifact_root(tmp_path, "io")
        spec = make_processed_dataset_spec("io")
        root = _write_instance(logical, spec, _envelope(spec, seed=1))
    else:
        root = _write_artifact(tmp_path)
        logical = root.parent.parent
    before = {
        path: identity.filesystem_read_bytes(path)
        for path in identity.filesystem_glob(root, "**/*")
        if identity.filesystem_is_file(path)
    }

    def fail(*args, **kwargs):
        raise primary

    with monkeypatch.context() as patch:
        if boundary.endswith("instances_stat"):
            owner = instances if boundary.startswith("dataset") else semantic
            original = owner.filesystem_stat

            def selected_stat(path, *args, **kwargs):
                if path == logical / "instances":
                    raise primary
                return original(path, *args, **kwargs)

            patch.setattr(owner, "filesystem_stat", selected_stat)
        elif boundary == "dataset_inventory_next":
            original = inventory_owner.filesystem_scandir

            class BrokenEntries:
                def __init__(self, entries):
                    self.entries = entries

                def __next__(self):
                    raise primary

                def close(self):
                    self.entries.close()

            def scan(path):
                entries = original(path)
                return BrokenEntries(entries) if path == root / "data" else entries

            patch.setattr(inventory_owner, "filesystem_scandir", scan)
        elif boundary in {
            "dataset_inventory",
            "dataset_instances_scan",
            "semantic_instances_scan",
            "semantic_members_scan",
        }:
            owner = (
                inventory_owner
                if boundary == "dataset_inventory"
                else instances
                if boundary == "dataset_instances_scan"
                else semantic
            )
            original = owner.filesystem_scandir
            target = (
                root / "data"
                if boundary == "dataset_inventory"
                else root
                if boundary == "semantic_members_scan"
                else logical / "instances"
            )

            def selected(path):
                if path == target:
                    raise primary
                return original(path)

            patch.setattr(owner, "filesystem_scandir", selected)
        elif boundary == "dataset_instances_manifest":
            patch.setattr(instances, "read_dataset_manifest_evidence", fail)
        elif boundary == "semantic_member_hash":
            patch.setattr(semantic, "_hash_file_content", fail)
        else:
            assert boundary == "semantic_member_stat"
            original = semantic.filesystem_stat

            def selected(path, *args, **kwargs):
                if path == root / "model.pkl":
                    raise primary
                return original(path, *args, **kwargs)

            patch.setattr(semantic, "filesystem_stat", selected)
        with pytest.raises(type(primary)) as caught:
            if boundary in {"dataset_inventory", "dataset_inventory_next"}:
                read_dataset(root)
            elif boundary.startswith("dataset"):
                instances.discover_dataset_instances(logical)
            elif "instances" in boundary:
                semantic.discover_semantic_artifact_instances(logical)
            else:
                semantic.read_semantic_artifact(root)
        assert caught.value is primary
    assert all(
        (
            identity.filesystem_read_bytes(path) == value
            for path, value in before.items()
        )
    )


@pytest.mark.parametrize("operation", ("read", "inspect", "fingerprint"))
@pytest.mark.parametrize("failure", ("tell", "seek", "read"))
def test_public_parquet_footer_preserves_system_io(
    tmp_path, monkeypatch, operation, failure
):
    import io

    from quant_research_platform.storage import datasets
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    root = make_dataset_artifact_root(tmp_path, "footer_io")
    datasets.write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("footer_io"),
    )
    primary = OSError(5, "synthetic_footer_io")
    opened = []
    original = physical.open_filesystem_file

    class FailingHandle(io.BufferedReader):
        def tell(self):
            if failure == "tell":
                raise primary
            return super().tell()

        def seek(self, offset, whence=0):
            if failure == "seek" and whence == os.SEEK_END:
                raise primary
            return super().seek(offset, whence)

        def read(self, size=-1):
            if failure == "read" and size == 8:
                raise primary
            return super().read(size)

    def open_member(path):
        if path.suffix == ".parquet":
            handle = FailingHandle(io.FileIO(path, "rb"))
            opened.append(handle)
            return handle
        return original(path)

    monkeypatch.setattr(physical, "open_filesystem_file", open_member)
    with pytest.raises(OSError) as caught:
        getattr(datasets, operation + "_dataset")(root)
    assert caught.value is primary
    assert opened and all(handle.closed for handle in opened)


@pytest.mark.parametrize("phase", ("before", "after", "primary"))
def test_verified_file_handle_binds_decoder_and_preserves_primary(tmp_path, phase):
    from quant_research_platform.storage.identity import (
        dataset_file_identity,
        verified_file_handle,
    )

    path = tmp_path / "member.bin"
    replacement = tmp_path / "replacement.bin"
    identity.filesystem_write_bytes(path, b"trusted")
    identity.filesystem_write_bytes(replacement, b"changed")
    expected = dataset_file_identity(path)
    opened = replacement if phase == "before" else path
    primary = RuntimeError("decode failed")
    with identity.open_filesystem_file(opened, "rb") as handle:
        if phase == "before":
            with pytest.raises(ValueError, match="dataset_file_identity_changed"):
                with verified_file_handle(path, handle, expected_identity=expected):
                    pytest.fail("unverified handle reached decoder")
        else:
            with pytest.raises(
                RuntimeError if phase == "primary" else ValueError
            ) as caught:
                with verified_file_handle(path, handle, expected_identity=expected):
                    assert handle.read() == b"trusted"
                    with identity.open_filesystem_file(path, "r+b") as mutating:
                        mutating.write(b"changed-content")
                        mutating.truncate()
                    if phase == "primary":
                        raise primary
            if phase == "primary":
                assert caught.value is primary
                assert any(
                    "identity verification" in note for note in primary.__notes__
                )
        assert not handle.closed


def test_parquet_writer_finalizes_deferred_mtime_before_read(tmp_path, monkeypatch):
    import pyarrow as pa

    pending_update = False
    original_write = dataset_writing.pq.write_table
    original_utime = os.utime
    original_verify = physical.verified_file_handle

    def deferred_write(*args, **kwargs):
        nonlocal pending_update
        original_write(*args, **kwargs)
        pending_update = True

    def commit_metadata(*args, **kwargs):
        nonlocal pending_update
        original_utime(*args, **kwargs)
        pending_update = False

    @contextmanager
    def read_with_deferred_update(path, handle, **kwargs):
        nonlocal pending_update
        with original_verify(path, handle, **kwargs):
            yield handle
            if pending_update:
                before = identity.filesystem_stat(path)
                original_utime(
                    identity.native_filesystem_path(path),
                    ns=(before.st_atime_ns, before.st_mtime_ns + 1000000),
                )
                pending_update = False

    monkeypatch.setattr(dataset_writing.pq, "write_table", deferred_write)
    monkeypatch.setattr(dataset_writing.os, "utime", commit_metadata)
    monkeypatch.setattr(physical, "verified_file_handle", read_with_deferred_update)
    path = tmp_path / "part.parquet"
    table = pa.table({"value": [1.0, 2.0, 3.0]})
    dataset_writing.write_parquet_file(table, path)
    with physical._opened_parquet(path, expected_identity=None) as (parquet, _, _):
        assert parquet.read().equals(table)
    pending_update = True
    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        with physical._opened_parquet(path, expected_identity=None) as (parquet, _, _):
            parquet.read()


@pytest.mark.parametrize(
    "operation", ["file_identity", "csv_records", "choice_logout", "rebuild_cleanup"]
)
def test_cleanup_scope_ignores_handled_outer(tmp_path, monkeypatch, operation):
    path = tmp_path / "member.csv"
    identity.filesystem_write_bytes(path, b"value\n1\n")
    try:
        raise RuntimeError("handled_outer")
    except RuntimeError as outer:
        if operation == "file_identity":
            expected = identity.dataset_file_identity(path)
            with identity.open_filesystem_file(path, "rb") as handle:
                with pytest.raises(ValueError, match="dataset_file_identity_changed"):
                    with identity.verified_file_handle(
                        path, handle, expected_identity=expected
                    ):
                        identity.filesystem_write_bytes(path, b"changed-content\n")
                assert not handle.closed
        elif operation == "csv_records":
            from quant_research_platform.sources.csmar.raw_schemas import (
                iter_export_csv_records,
            )

            records = iter_export_csv_records(path)
            assert next(records) == ["value"]
            identity.filesystem_write_bytes(path, b"changed-content\n")
            with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
                records.close()
        elif operation == "choice_logout":
            from quant_research_platform.sources.choice.raw_exports import (
                choice_session,
            )
            from tests.helpers.choice_raw import SDK, response

            sdk = SDK(response())
            sdk.stop_error = 17
            with pytest.raises(RuntimeError, match="choice_logout_failed:17"):
                with choice_session(sdk) as session:
                    assert session.active
            assert sdk.stops == 1 and not session.active
        else:
            import quant_research_platform.storage.rebuild as rebuild
            from quant_research_platform.storage.layout import StorageLayout

            def fail_cleanup(_root):
                raise OSError("scratch_cleanup_failed")

            monkeypatch.setattr(rebuild, "remove_owned_scratch", fail_cleanup)
            with pytest.raises(OSError, match="scratch_cleanup_failed"):
                with rebuild.rebuild_candidate_workspace(StorageLayout(tmp_path)):
                    pass
        assert not getattr(outer, "__notes__", [])
