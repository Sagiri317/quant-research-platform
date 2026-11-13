from __future__ import annotations

import io
import os
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
from quant_research_platform.storage import atomic, manifests
from quant_research_platform.storage import atomic as scratch_owner
from tests.helpers.paths import (
    filesystem_glob,
    filesystem_write_bytes,
    filesystem_write_text,
)
from tests.helpers.source_preparation import prepared_source_evidence

_PRIMARY_EXCEPTION_TYPES = (
    RuntimeError,
    MemoryError,
    KeyboardInterrupt,
    SystemExit,
)


@pytest.mark.parametrize("long_path", (False, True))
def test_shared_path_move_is_fail_if_destination_exists(
    tmp_path: Path, long_path
) -> None:
    shared = __import__("quant_research_platform.storage.atomic", fromlist=["*"])
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


def _install_parquet_stubs(monkeypatch, handle, parquet):
    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4)
    monkeypatch.setattr(physical, "open_filesystem_file", lambda *_a, **_k: handle)
    monkeypatch.setattr(physical.os, "fstat", lambda _fd: metadata)
    monkeypatch.setattr(physical.pq, "ParquetFile", lambda _handle, **_kwargs: parquet)


def test_opened_parquet_close_surfaces_failure_on_cancellation(
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

    _install_parquet_stubs(monkeypatch, Handle(), Parquet())
    with pytest.raises(RuntimeError, match="close_failed"):
        with physical._opened_parquet(Path("part.parquet")):
            raise GeneratorExit()


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

    _install_parquet_stubs(monkeypatch, Handle(), Parquet())
    with pytest.raises(primary_type) as raised:
        with physical._opened_parquet(Path("part.parquet")):
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

    _install_parquet_stubs(monkeypatch, Handle(), Parquet())
    with pytest.raises(OSError, match="raw_close_failed"):
        with physical._opened_parquet(Path("part.parquet")):
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
    filesystem_write_text(path, '{"schema_id":"qrp-test"}', encoding="utf-8")
    primary = primary_type("body_primary")
    monkeypatch.setattr(
        manifests,
        "open_filesystem_file",
        lambda *_args, **_kwargs: _BodyAndCloseFailingHandle(primary),
    )
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


def test_shared_scratch_removes_only_exact_owned_paths(tmp_path: Path) -> None:
    owned_file = tmp_path / "owned-file"
    filesystem_write_bytes(owned_file, b"x")
    owned_directory = tmp_path / "owned-directory"
    identity.filesystem_mkdir(owned_directory)
    filesystem_write_bytes(owned_directory / "part", b"x")
    scratch_owner.remove_owned_scratch(owned_file)
    scratch_owner.remove_owned_scratch(owned_directory)
    assert not identity.filesystem_exists(owned_file)
    assert not identity.filesystem_exists(owned_directory)


@pytest.mark.parametrize("source", ["csmar", "reference"])
def test_source_reader_preserves_io_failure(tmp_path, monkeypatch, source):
    import errno

    from quant_research_platform.sources.csmar import interim as csmar
    from quant_research_platform.sources.reference import interim as reference
    from tests.sources.reference.test_interim import _write_reference_root

    if source == "csmar":
        root, data_id = (tmp_path, "trading_calendar")
        path = root / "native-calendar.csv"
        filesystem_write_text(
            path, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
        )
        owner = csmar
    else:
        root = _write_reference_root(tmp_path / "raw") / "reference"
        data_id = reference.A_SHARE_FEE_SCHEDULE_DATA_ID
        path = next(filesystem_glob(root, "**/*.csv"))
        owner = reference
    primary = OSError(errno.EIO, "primary")
    from quant_research_platform.sources.csmar import raw_schemas
    from quant_research_platform.sources.reference import a_share_fees

    target_owner = raw_schemas if source == "csmar" else a_share_fees
    operation = "open_filesystem_file"
    original = getattr(target_owner, operation)

    def fail(candidate, *args, **kwargs):
        if candidate == path or candidate == str(path):
            raise primary
        return original(candidate, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(target_owner, operation, fail)
        with pytest.raises(OSError) as caught:
            prepared_source_evidence(owner, root, data_id)
        assert caught.value is primary


@pytest.mark.parametrize("error_type", (OSError, PermissionError))
def test_dataset_inventory_preserves_serious_io(tmp_path, monkeypatch, error_type):
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    root = make_dataset_artifact_root(tmp_path, "io")
    dataset_writing.write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec("io")
    )
    primary = error_type(5, "synthetic_storage_io")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(inventory_owner, "filesystem_scandir", fail)
    with pytest.raises(error_type) as caught:
        dataset_reading.read_dataset(root)
    assert caught.value is primary


@pytest.mark.parametrize("operation", ("read", "fingerprint"))
@pytest.mark.parametrize("failure", ("tell", "seek", "read"))
def test_public_parquet_reader_preserves_system_io(
    tmp_path, monkeypatch, operation, failure
):

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
            if failure == "read":
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


def test_cleanup_scope_ignores_handled_outer(tmp_path, monkeypatch):
    from quant_research_platform.sources.csmar import raw_schemas

    path = tmp_path / "member.csv"
    filesystem_write_bytes(path, b"value\n1\n")
    original = raw_schemas._CsvRecords.close

    def close_then_fail(records):
        original(records)
        raise OSError("close_failed")

    monkeypatch.setattr(raw_schemas._CsvRecords, "close", close_then_fail)
    try:
        raise RuntimeError("handled_outer")
    except RuntimeError as outer:
        records = raw_schemas.iter_export_csv_records(path)
        assert next(records) == ["value"]
        with pytest.raises(OSError, match="close_failed"):
            records.close()
        assert not getattr(outer, "__notes__", [])
