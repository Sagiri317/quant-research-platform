from __future__ import annotations

import csv
import gc
import io
import os
import struct
import subprocess
import tempfile
import tracemalloc
import weakref
from array import array
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import (
    field_mappings,
    interim,
    raw_exports,
    raw_schemas,
)
from quant_research_platform.sources.csmar import (
    ids as csmar_ids,
)
from quant_research_platform.storage import errors as storage_errors
from quant_research_platform.storage._dataset_key_tracking import CompactPositionState
from quant_research_platform.storage._dataset_physical import DatasetFileIdentity
from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.runtime_resources import capture_test_policy


def test_csmar_spool_owners_reject_aliased_parent_before_disk_access(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="csmar_raw_scratch_parent_invalid"):
        raw_exports._IndexInfoContinuationSpool(
            record_limit_bytes=1024,
            scratch_parent=alias,
        )

    records = object.__new__(raw_schemas._BoundedCsvRecords)
    records._spool = None
    records._path = alias / "source.csv"
    records._slow_control_span_bytes = 1
    records._retained_payload_bytes = [0]
    records._task_memory_budget_bytes = 4096
    records._limit_bytes = 1024
    with pytest.raises(ValueError, match="csmar_raw_scratch_parent_invalid"):
        records._ensure_spool()

    assert not tuple(target.iterdir())


CSMAR_LOGICAL_RECORD_LIMIT_BYTES = 1024**2


@pytest.fixture(autouse=True)
def _synthetic_csv_handle_paths_have_stable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = DatasetFileIdentity(device=1, inode=2, size=3, modified_ns=4)
    real_identity = raw_schemas.dataset_file_identity

    def identify(path: Path) -> DatasetFileIdentity:
        try:
            candidate = Path(path)
        except TypeError:
            return identity
        if not candidate.exists() and candidate.suffix == ".csv":
            return identity
        return real_identity(candidate)

    monkeypatch.setattr(raw_schemas, "dataset_file_identity", identify)


@pytest.mark.parametrize("failure_type", (OSError, ValueError))
def test_raw_export_file_identity_has_one_error_translation_owner(
    failure_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = failure_type("identity_probe_failed")
    monkeypatch.setattr(
        raw_schemas,
        "dataset_file_identity",
        lambda _path: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed") as caught:
        raw_schemas.raw_export_file_identity(Path("raw.csv"))

    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "value",
    ("a\u00e9\u754c\U0001f642",),
)
def test_utf8_size_preserves_multibyte_lengths(value: str) -> None:
    assert raw_exports._utf8_size(value) == len(value.encode("utf-8"))


def test_utf8_size_does_not_materialize_string_slices() -> None:
    class SliceRejectingString(str):
        def __getitem__(self, key):  # noqa: ANN001, ANN204
            if isinstance(key, slice):
                raise AssertionError("utf8_size_materialized_string_slice")
            return super().__getitem__(key)

    value = SliceRejectingString("a\u00e9\u754c\U0001f642")

    assert raw_exports._utf8_size(value) == 10


def test_utf8_size_preserves_strict_surrogate_error() -> None:
    with pytest.raises(UnicodeEncodeError):
        raw_exports._utf8_size("\ud800")


@pytest.mark.parametrize("chunksize", (0, True, 1.5))
def test_private_export_chunks_require_exact_positive_int_chunksize(
    chunksize: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_csmar_chunksize"):
        raw_exports.iter_private_export_chunks(
            Path("export.csv"),
            dataset="daily",
            chunksize=chunksize,  # type: ignore[arg-type]
        )


def test_csv_owner_bounds_total_logical_record_and_restores_limit_before_yield(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wide-header.csv"
    path.write_text(",".join(["x" * 100] * 11_000) + "\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    previous_limit = csv.field_size_limit()

    rows = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    with pytest.raises(
        ValueError,
        match="csmar_logical_record_exceeds_source_limit",
    ):
        next(rows)
    assert csv.field_size_limit() == previous_limit


@pytest.mark.parametrize("extra_bytes", (0, 1))
def test_csv_owner_accepts_source_record_boundary_and_rejects_next_byte(
    tmp_path: Path,
    extra_bytes: int,
) -> None:
    path = tmp_path / "boundary.csv"
    path.write_bytes(
        b"x" * (CSMAR_LOGICAL_RECORD_LIMIT_BYTES - 1 + extra_bytes) + b"\n"
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    rows = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    if extra_bytes:
        with pytest.raises(
            ValueError,
            match="csmar_logical_record_exceeds_source_limit",
        ):
            next(rows)
    else:
        row, source_bytes = next(rows)
        assert len(row[0]) == CSMAR_LOGICAL_RECORD_LIMIT_BYTES - 1
        assert source_bytes == CSMAR_LOGICAL_RECORD_LIMIT_BYTES
        rows.close()


def test_csv_owner_accepts_exact_boundary_at_end_of_file(tmp_path: Path) -> None:
    path = tmp_path / "boundary-without-line-ending.csv"
    path.write_bytes(b"x" * CSMAR_LOGICAL_RECORD_LIMIT_BYTES)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    rows = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    row, source_bytes = next(rows)
    assert len(row[0]) == CSMAR_LOGICAL_RECORD_LIMIT_BYTES
    assert source_bytes == CSMAR_LOGICAL_RECORD_LIMIT_BYTES
    assert next(rows, None) is None


@pytest.mark.parametrize("position", ("first", "later"))
def test_csv_owner_rejects_oversize_data_record_with_larger_payload(
    tmp_path: Path,
    position: str,
) -> None:
    path = tmp_path / "data-boundary.csv"
    oversize = b"x" * CSMAR_LOGICAL_RECORD_LIMIT_BYTES + b"\n"
    rows = [oversize, b"ok\n"] if position == "first" else [b"ok\n", oversize]
    path.write_bytes(b"value\n" + b"".join(rows))
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    assert next(records) == (["value"], len(b"value\n"))
    if position == "later":
        assert next(records) == (["ok"], len(b"ok\n"))
    with pytest.raises(
        ValueError,
        match="csmar_logical_record_exceeds_source_limit",
    ):
        next(records)


def test_csv_owner_bounds_quoted_newline_as_one_source_record(tmp_path: Path) -> None:
    path = tmp_path / "quoted-newline.csv"
    content = b"x" * (CSMAR_LOGICAL_RECORD_LIMIT_BYTES - 4)
    path.write_bytes(b'value\n"' + content + b'\ny"\n')
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    assert next(records) == (["value"], len(b"value\n"))
    with pytest.raises(
        ValueError,
        match="csmar_logical_record_exceeds_source_limit",
    ):
        next(records)


def test_csv_owner_preserves_bom_utf8_crlf_quotes_and_escapes(tmp_path: Path) -> None:
    path = tmp_path / "grammar.csv"
    header_bytes = b"\xef\xbb\xbfname,note\r\n"
    row_bytes = '中文,"a,""b"""\r\n'.encode()
    path.write_bytes(header_bytes + row_bytes)

    records = raw_schemas.iter_export_csv_records(
        path,
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )

    assert list(records) == [
        (["name", "note"], len(header_bytes)),
        (["中文", 'a,"b"'], len(row_bytes)),
    ]


def test_csv_owner_restores_field_limit_while_row_is_suspended(tmp_path: Path) -> None:
    path = tmp_path / "row.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    previous_limit = csv.field_size_limit()

    rows = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    first_physical_line = path.read_bytes().splitlines(keepends=True)[0]
    assert next(rows) == (["a", "b"], len(first_physical_line))
    assert csv.field_size_limit() == previous_limit
    rows.close()


def test_csv_owner_rejects_retained_headroom_before_physical_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    read_calls = 0

    class Handle:
        def read(self, _limit: int = -1) -> bytes:
            nonlocal read_calls
            read_calls += 1
            return b"value\n"

        def close(self) -> None:
            pass

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())
    records = raw_schemas.iter_export_csv_records(
        Path("unused.csv"),
        resource_policy=policy,
        retained_payload_bytes=[policy.task_memory_budget_bytes],
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(records)

    assert read_calls == 0


def test_csv_owner_never_requests_bytes_beyond_retained_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    read_limits: list[int] = []

    class Handle:
        def read(self, limit: int = -1) -> bytes:
            read_limits.append(limit)
            return b"x" * limit

        def close(self) -> None:
            pass

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())
    records = raw_schemas.iter_export_csv_records(
        Path("unused.csv"),
        resource_policy=policy,
        retained_payload_bytes=[policy.task_memory_budget_bytes - 100],
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(records)

    assert read_limits == []


def test_csmar_csv_field_bound_is_derived_from_mapping_contract() -> None:
    expected = max(
        len(mapping)
        + sum(
            len(aliases)
            for aliases in field_mappings.COLUMN_ALIASES.get(dataset, {}).values()
        )
        for dataset, mapping in field_mappings.COLUMN_MAPPINGS.items()
    )

    assert raw_schemas._legal_provider_field_count_bound() == expected


def test_csmar_csv_field_bound_is_cached_after_mapping_derivation() -> None:
    bound = raw_schemas._legal_provider_field_count_bound
    bound.cache_clear()

    assert bound() == bound()
    assert bound.cache_info().misses == 1
    assert bound.cache_info().hits == 1


def test_csv_owner_task_equal_rejection_python_allocation_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task-equal.csv"
    path.write_bytes(b"value\n" + b"x" * (CSMAR_LOGICAL_RECORD_LIMIT_BYTES - 1) + b"\n")
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
    header = next(records)
    del header

    tracemalloc.start()
    try:
        with pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="csmar_raw_working_set_exceeds_task_memory_budget",
        ):
            next(records)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        records.close()

    assert peak_bytes < 2 * 1024**2


def test_csv_owner_fast_path_admits_before_content_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = CSMAR_LOGICAL_RECORD_LIMIT_BYTES

    class ObservedBytes(bytes):
        def __getitem__(self, key):  # noqa: ANN001, ANN204
            value = super().__getitem__(key)
            if isinstance(key, slice) and len(value) > io.DEFAULT_BUFFER_SIZE:
                pytest.fail("content_slice_before_capacity_admission")
            return value

        def decode(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
            pytest.fail("decode_before_capacity_admission")

        def split(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
            pytest.fail("split_before_capacity_admission")

    class Handle:
        returned = False

        def readline(self, _limit: int = -1) -> bytes:
            if self.returned:
                return b""
            self.returned = True
            return ObservedBytes(b"x" * (limit - 1) + b"\n")

        def close(self) -> None:
            pass

    monkeypatch.setattr(raw_schemas.io, "BufferedReader", Handle)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())
    records = raw_schemas.iter_export_csv_records(
        Path("source.csv"),
        resource_policy=capture_test_policy(
            available_memory_bytes=4 * 1024**2,
            explicit_cap_bytes=2 * 1024 * 1024,
        ),
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(records)


def test_index_info_high_field_unicode_python_allocation_diagnostic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index-info-high-fields.csv"
    path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        + ",".join(("000985", *("界" * 64 for _ in range(1004))))
        + "\n",
        encoding="utf-8",
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    rows = iter(raw_exports.iter_index_info_rows(path, resource_policy=policy))

    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
            next(rows)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        rows.close()

    assert peak_bytes < 1024**2


def test_csv_owner_closes_record_spool_on_consumer_abandon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "abandon.csv"
    path.write_text('value\n"1"\n', encoding="utf-8")
    opened: list[object] = []
    original = tempfile.TemporaryFile

    class TrackedFile:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._handle = original(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(self._handle, name)

        def close(self) -> None:
            self.closed = True
            self._handle.close()

    monkeypatch.setattr(raw_schemas.tempfile, "TemporaryFile", TrackedFile)
    records = raw_schemas.iter_export_csv_records(
        path,
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )

    next(records)
    assert opened == []
    next(records)
    records.close()

    assert opened
    assert all(item.closed for item in opened)


def test_csv_owner_spool_is_disk_gated_on_source_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "source-volume"
    source_parent.mkdir()
    path = source_parent / "quoted.csv"
    path.write_text('value\n"1"\n', encoding="utf-8")
    gated: list[Path] = []
    temporary_dirs: list[Path] = []
    original = tempfile.TemporaryFile

    def require_headroom(parent: Path, *, reserve_bytes: int) -> None:
        assert reserve_bytes > 0
        gated.append(Path(parent))

    def temporary_file(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        temporary_dirs.append(Path(kwargs["dir"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(raw_schemas, "require_free_disk_headroom", require_headroom)
    monkeypatch.setattr(raw_schemas.tempfile, "TemporaryFile", temporary_file)

    records = raw_schemas.iter_export_csv_records(
        path,
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )
    assert [row for row, _ in records] == [["value"], ["1"]]

    assert gated == [source_parent]
    assert temporary_dirs == [source_parent]


def test_csv_owner_admits_slow_controls_before_spool_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    retained = [policy.task_memory_budget_bytes - 1]
    monkeypatch.setattr(
        raw_schemas.tempfile,
        "TemporaryFile",
        lambda *_args, **_kwargs: pytest.fail("spool_created_before_capacity"),
    )
    records = raw_schemas.iter_export_csv_records(
        Path("quoted.csv"),
        resource_policy=policy,
        retained_payload_bytes=retained,
    )
    monkeypatch.setattr(
        Path,
        "open",
        lambda *_args, **_kwargs: SimpleNamespace(
            close=lambda: None,
        ),
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(records)


def test_csv_owner_admits_total_slow_materialization_before_decode(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    field_count = raw_schemas._legal_provider_field_count_bound()
    field = "x" * (680_000 // field_count)
    path = tmp_path / "many-small-fields.csv"
    path.write_text(
        ",".join(f'"{field}"' for _ in range(field_count)),
        encoding="utf-8",
    )
    records = raw_schemas.iter_export_csv_records(
        path,
        resource_policy=policy,
        retained_payload_bytes=[900_000],
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(records)


def test_csv_owner_admits_fast_decoded_list_copy() -> None:
    owner = SimpleNamespace(
        generation=1,
        _task_memory_budget_bytes=64,
        _retained_payload_bytes=[49],
    )
    record = raw_schemas.BoundedCsvRecord(
        owner,
        generation=1,
        field_count=2,
        source_bytes=4,
        canonical_bytes=2,
        decoded_fields=["a", "b"],
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        record.materialize()


def test_over_width_header_rejects_before_field_decode() -> None:
    record = SimpleNamespace(
        field_count=50_000,
        iter_fields=lambda: pytest.fail("over_width_header_decoded"),
    )

    with pytest.raises(ValueError, match="csmar_raw_export_schema_invalid"):
        raw_schemas.materialize_header_record(record)  # type: ignore[arg-type]


def test_index_info_continuation_spool_is_disk_gated_on_source_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "source-volume"
    source_parent.mkdir()
    path = source_parent / "index-info.csv"
    path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000905,mid,2014-01-06,2013-12-31,500,scope\n"
        "continued text,1,2,3,4,5,formula\n",
        encoding="utf-8",
    )
    gated: list[Path] = []
    temporary_dirs: list[Path] = []
    original = tempfile.TemporaryFile

    def require_headroom(parent: Path, *, reserve_bytes: int) -> None:
        assert reserve_bytes > 0
        gated.append(Path(parent))

    def temporary_file(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        temporary_dirs.append(Path(kwargs["dir"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(raw_exports, "require_free_disk_headroom", require_headroom)
    monkeypatch.setattr(raw_exports.tempfile, "TemporaryFile", temporary_file)

    rows = list(raw_exports.iter_index_info_rows(path))

    assert rows[0][1][5] == "scope\ncontinued text"
    assert gated == [source_parent]
    assert temporary_dirs == [source_parent]


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        (b"\xef\xbb\xbfname,note\r\n", [["name", "note"]]),
        (b'a,"b""c"\n', [["a", 'b"c']]),
        (b'a,"b\r\nc"\r\n', [["a", "b\r\nc"]]),
        (b'a,"b"x\n', [["a", "bx"]]),
        (b'a,b"c\r', [["a", 'b"c']]),
        (b"\n", [[]]),
        (b"a,b", [["a", "b"]]),
    ),
)
def test_bounded_csv_owner_preserves_supported_provider_grammar(
    tmp_path: Path,
    payload: bytes,
    expected: list[list[str]],
) -> None:
    path = tmp_path / "grammar.csv"
    path.write_bytes(payload)

    records = raw_schemas.iter_export_csv_records(
        path,
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )

    assert [row for row, _ in records] == expected


@pytest.mark.parametrize("payload", (b'a,"unterminated\n', b"\xff\n"))
def test_bounded_csv_owner_rejects_strict_malformed_input(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "malformed.csv"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        list(
            raw_schemas.iter_export_csv_records(
                path,
                resource_policy=capture_test_policy(
                    available_memory_bytes=64 * 1024**2,
                    explicit_cap_bytes=32 * 1024 * 1024,
                ),
            )
        )


@pytest.mark.parametrize("dataset", ("trading_calendar", "index_info"))
def test_private_export_flushes_publishable_batch_before_next_record(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    pulls: list[int] = []
    header = ["value"]

    def records(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        for position in range(3):
            pulls.append(position)
            yield header, [str(position)], 8

    owner = "iter_index_info_rows" if dataset == "index_info" else "_iter_csv_records"
    monkeypatch.setattr(raw_exports, owner, records)
    with runtime_resource_policy_scope(policy):
        chunks = raw_exports.iter_private_export_chunks_with_policy(
            Path("unused.csv"),
            dataset=dataset,
            chunksize=1,
            resource_policy=policy,
        )

        first = next(chunks)

        assert first["value"].tolist() == [0]
        assert pulls == [0]
        chunks.close()


@pytest.mark.parametrize("wide_position", ("first", "later"))
def test_private_export_chunks_reject_indivisible_wide_logical_record(
    tmp_path: Path,
    wide_position: str,
) -> None:
    path = tmp_path / "calendar.csv"
    wide = "x" * (2 * 1024**2)
    states = (wide, "O") if wide_position == "first" else ("O", wide)
    pd.DataFrame(
        {
            "Markettype": ["4", "4"],
            "Clddt": ["2024-01-01", "2024-01-02"],
            "Daywk": ["1", "2"],
            "State": states,
        }
    ).to_csv(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="csmar_logical_record_exceeds_source_limit"
        ):
            list(
                raw_exports.iter_private_export_chunks(
                    path,
                    dataset="trading_calendar",
                    chunksize=2,
                    resource_policy=policy,
                )
            )


def test_private_export_adapts_across_later_wide_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "later-wide.csv"
    pd.DataFrame(
        {
            "Markettype": ["4"] * 21,
            "Clddt": pd.date_range("2024-01-01", periods=21).astype(str),
            "Daywk": ["1"] * 21,
            "State": ["O", *("x" * 200_000 for _ in range(20))],
        }
    ).to_csv(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    observed = 0

    with runtime_resource_policy_scope(policy):
        chunks = raw_exports.iter_private_export_chunks(
            path,
            dataset="trading_calendar",
            chunksize=100,
            resource_policy=policy,
        )
        for chunk in chunks:
            observed += len(chunk)
            del chunk

    assert observed == 21


def test_private_export_parent_reservation_blocks_downstream_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    def records(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        yield (
            ["Markettype", "Clddt", "Daywk", "State"],
            [
                "4",
                "2024-01-02",
                "2",
                "O",
            ],
            10 * 1024**2,
        )
        yield (
            ["Markettype", "Clddt", "Daywk", "State"],
            [
                "4",
                "2024-01-03",
                "3",
                "O",
            ],
            10 * 1024**2,
        )

    monkeypatch.setattr(raw_exports, "_iter_csv_records", records)
    monkeypatch.setattr(
        raw_exports,
        "pandas_frame_payload_bytes",
        lambda _frame: 24 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        chunks = raw_exports.iter_private_export_chunks_with_policy(
            Path("unused.csv"),
            dataset="trading_calendar",
            chunksize=1,
            resource_policy=policy,
        )
        previous = next(chunks)
        assert policy.reserved_memory_bytes == 24 * 1024**2

        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            with policy.reserve("storage.dataset.arrow", 10 * 1024**2):
                pytest.fail("downstream overlap admitted")
        assert len(previous) == 1
        chunks.close()


def test_private_export_reserves_downstream_retained_before_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    def records(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        yield (
            ["Markettype", "Clddt", "Daywk", "State"],
            [
                "4",
                "2024-01-02",
                "2",
                "O",
            ],
            1024,
        )

    monkeypatch.setattr(raw_exports, "_iter_csv_records", records)
    monkeypatch.setattr(
        raw_exports,
        "_raw_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DataFrame allocated before downstream retained admission")
        ),
    )
    retained_payload_bytes = [policy.task_memory_budget_bytes]

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(
            raw_exports.iter_private_export_chunks_with_policy(
                Path("unused.csv"),
                dataset="trading_calendar",
                chunksize=1,
                resource_policy=policy,
                retained_payload_bytes=retained_payload_bytes,
            )
        )


def test_interim_transform_rejects_task_equal_payload_wide_row_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=1024**2,
    )
    raw = pd.DataFrame(
        {
            "Markettype": ["4"],
            "Clddt": ["2024-01-02"],
            "Daywk": ["2"],
            "State": ["O"],
            **{f"wide_{index}": ["x" * 1024] for index in range(32)},
        }
    )
    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda *_args, **_kwargs: iter((raw,)),
    )
    monkeypatch.setattr(
        interim,
        "pandas_frame_payload_bytes",
        lambda _frame: policy.remaining_memory_bytes,
    )

    def reject_mapping(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("mapping ran before transform overlap admission")

    monkeypatch.setattr(interim, "normalize_raw_frame", reject_mapping)
    with runtime_resource_policy_scope(policy):
        frames = interim._iter_pruned_interim_frames(
            data_id="trading_calendar",
            raw_dataset="trading_calendar",
            input_files=(Path("unused.csv"),),
            start_date=None,
            end_date=None,
            overrides={},
            chunksize=1,
            pruning_reports=None,
        )
        with pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="csmar_interim_working_set_exceeds_task_memory_budget",
        ):
            next(frames)


def test_private_export_chunks_keep_normal_wide_record_with_default_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calendar.csv"
    state = "x" * 200_000
    pd.DataFrame(
        {
            "Markettype": ["4"],
            "Clddt": ["2024-01-01"],
            "Daywk": ["1"],
            "State": [state],
        }
    ).to_csv(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        chunks = list(
            raw_exports.iter_private_export_chunks(
                path,
                dataset="trading_calendar",
                chunksize=1,
                resource_policy=policy,
            )
        )

    assert len(chunks) == 1
    assert chunks[0].loc[0, "State"] == state


def test_private_export_chunks_small_fit_has_no_hidden_budget_floor(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=31 * 1024 * 1024,
    )
    path = tmp_path / "calendar.csv"
    path.write_text(
        "Markettype,Clddt,Daywk,State\n4,2024-01-02,2,O\n",
        encoding="utf-8",
    )
    with runtime_resource_policy_scope(policy):
        chunks = list(
            raw_exports.iter_private_export_chunks(
                path,
                dataset="trading_calendar",
                chunksize=1,
                resource_policy=policy,
            )
        )
    assert len(chunks) == 1


def test_materialize_low_budget_fails_only_on_exact_working_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    export = raw_root / "TRDNEW_Cale.csv"
    export.write_text(
        "Markettype,Clddt,Daywk,State\n4,2024-01-02,2,O\n",
        encoding="utf-8",
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    monkeypatch.setattr(
        interim,
        "_require_interim_transform_headroom",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            storage_errors.DatasetReadCapacityExceeded(
                "csmar_interim_working_set_exceeds_task_memory_budget"
            )
        ),
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="csmar_interim_working_set_exceeds_task_memory_budget",
        ):
            interim.materialize_interim_data(
                layout=StorageLayout(project_root=tmp_path),
                raw_root=raw_root,
                data_ids=("trading_calendar",),
                resource_policy=policy,
            )


def test_export_header_bounds_logical_record(tmp_path: Path) -> None:
    path = tmp_path / "TRDNEW_Cale.csv"
    path.write_text("x" * (2 * 1024**2) + "\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="csmar_logical_record_exceeds_source_limit"
        ):
            raw_schemas.read_export_header(path, resource_policy=policy)


@pytest.mark.parametrize("payload", ("", '"unterminated\n'))
def test_export_header_rejects_empty_and_malformed_csv(
    tmp_path: Path,
    payload: str,
) -> None:
    path = tmp_path / "TRDNEW_Cale.csv"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        raw_schemas.read_export_header(path)


def test_header_only_export_has_no_data_chunks(tmp_path: Path) -> None:
    path = tmp_path / "TRDNEW_Cale.csv"
    path.write_text("Markettype,Clddt,Daywk,State\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        assert raw_schemas.read_export_header(path, resource_policy=policy) == (
            "Markettype",
            "Clddt",
            "Daywk",
            "State",
        )
        assert (
            list(
                raw_exports.iter_private_export_chunks(
                    path,
                    dataset="trading_calendar",
                    chunksize=1,
                    resource_policy=policy,
                )
            )
            == []
        )


def test_private_export_batch_target_adapts_to_measured_frame_payload() -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    logical_bytes = 1024**2

    compact_target = raw_exports._next_batch_target_bytes(
        logical_bytes=logical_bytes,
        frame_payload_bytes=logical_bytes,
        resource_policy=policy,
    )
    expanded_target = raw_exports._next_batch_target_bytes(
        logical_bytes=logical_bytes,
        frame_payload_bytes=15 * logical_bytes,
        resource_policy=policy,
    )

    assert compact_target == policy.remaining_memory_bytes // 2
    assert expanded_target == 2 * logical_bytes
    assert expanded_target < compact_target


def test_index_info_continuations_join_once_after_linear_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "index-info.csv"
    continuation_count = 50
    path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000905,mid,2014-01-06,2013-12-31,500,scope\n"
        + "".join(f"part {position}\n" for position in range(continuation_count - 1))
        + f"part {continuation_count - 1},1,2,3,4,5,formula\n",
        encoding="utf-8",
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    original_join = raw_exports._IndexInfoContinuationSpool.joined
    original_append = raw_exports._IndexInfoContinuationSpool.append_fields
    observed_fragment_counts: list[int] = []
    appended = 0

    def recording_append(
        spool: raw_exports._IndexInfoContinuationSpool,
        fields: Iterable[str],
        **kwargs,
    ) -> list[str]:
        nonlocal appended
        appended += 1
        return original_append(spool, fields, **kwargs)

    def recording_join(
        spool: raw_exports._IndexInfoContinuationSpool,
        current: str,
        **kwargs,
    ) -> str:
        observed_fragment_counts.append(appended)
        return original_join(spool, current, **kwargs)

    monkeypatch.setattr(
        raw_exports._IndexInfoContinuationSpool,
        "append_fields",
        recording_append,
    )
    monkeypatch.setattr(
        raw_exports._IndexInfoContinuationSpool,
        "joined",
        recording_join,
    )
    with runtime_resource_policy_scope(policy):
        frame = pd.concat(
            raw_exports.iter_private_export_chunks(
                path,
                dataset="index_info",
                chunksize=100,
                resource_policy=policy,
            ),
            ignore_index=True,
        )

    assert observed_fragment_counts == [continuation_count]
    assert frame.loc[0, "Idxinfo04"].count("\n") == continuation_count


def test_index_info_reconstructed_source_record_uses_owner_limit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index-info-limit.csv"
    header = (
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
    )
    start = "000905,mid,2014-01-06,2013-12-31,500," + "s" * 600_000 + "\n"
    continuation = "x " * 225_000 + ",1,2,3,4,5,formula\n"
    path.write_text(header + start + continuation, encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="csmar_logical_record_exceeds_source_limit",
        ):
            list(raw_exports.iter_index_info_rows(path, resource_policy=policy))


def test_index_reader_releases_consumed_record_before_next_parser_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index.csv"
    header = (
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
    )
    body = "".join(
        f"{code},index,2014-01-02,2013-12-31,1000,scope,1,2,3,4,5,formula\n"
        for code in ("000001", "000002", "000003")
    )
    path.write_text(header + body, encoding="utf-8")
    rows = iter(raw_exports.iter_index_info_rows(path))

    first_header, first, _ = next(rows)
    assert first_header[0] == "Indexcd"
    assert first[0] == "000001"
    del first_header, first
    gc.collect()
    _, second, _ = next(rows)
    assert second[0] == "000002"
    _, third, _ = next(rows)
    assert third[0] == "000003"
    rows.close()


@pytest.mark.parametrize(
    "payload",
    (
        b'{"unknown":{"keep":[],"drop":[]}}',
        b'{"stock_master":{"keep":"stock_name","drop":[]}}',
        b'{"stock_master":{"keep":["stock_name","stock_name"],"drop":[]}}',
        b'{"stock_master":{"keep":["stock_name"],"drop":["stock_name"]}}',
        b'{"stock_master":{"keep":[],"drop":[]},"stock_master":{"keep":[],"drop":[]}}',
    ),
)
def test_column_overrides_reject_noncanonical_payloads(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "overrides.json"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        interim.load_column_overrides(path)


def test_column_overrides_reject_oversize_before_full_file_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "overrides.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))

    def forbidden_read_bytes(_path: Path) -> bytes:
        raise AssertionError("full_override_file_read_before_size_rejection")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    with pytest.raises(ValueError, match="csmar_column_overrides_exceed_max_bytes"):
        interim.load_column_overrides(path)


class _EmittedItemsMapping(Mapping[str, object]):
    def __init__(self, items: Iterable[tuple[str, object]]) -> None:
        self._items = tuple(items)
        self.consumed = 0

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return 0

    def items(self):  # noqa: ANN202
        for item in self._items:
            self.consumed += 1
            yield item


def _oversized_emitted_mapping() -> _EmittedItemsMapping:
    return _EmittedItemsMapping((f"unknown_{index}", {}) for index in range(10_000))


@pytest.mark.parametrize("level", ("outer", "nested", "optional"))
def test_column_override_mapping_snapshot_stops_at_owner_bound(
    level: str,
) -> None:
    emitted = _oversized_emitted_mapping()
    if level == "outer":
        payload: Mapping[str, object] = emitted
        bound = len(interim.INTERIM_COLUMNS_BY_DATA_ID)
    elif level == "nested":
        payload = {"stock_master": emitted}
        bound = 3
    else:
        payload = {
            "stock_master": {
                "_available_optional_columns": emitted,
            }
        }
        bound = len(interim.OPTIONAL_COLUMNS["stock_master"])

    with pytest.raises(ValueError):
        interim._validated_column_overrides(payload)

    assert emitted.consumed == bound + 1


@pytest.mark.parametrize("level", ("outer", "nested", "optional"))
def test_column_override_mapping_rejects_duplicate_emitted_keys(level: str) -> None:
    if level == "outer":
        emitted = _EmittedItemsMapping((("stock_master", {}), ("stock_master", {})))
        payload: Mapping[str, object] = emitted
    elif level == "nested":
        emitted = _EmittedItemsMapping((("keep", []), ("keep", [])))
        payload = {"stock_master": emitted}
    else:
        optional = interim.OPTIONAL_COLUMNS["stock_master"]
        first = next(iter(optional.items()))
        emitted = _EmittedItemsMapping((*optional.items(), first))
        payload = {
            "stock_master": {
                "_available_optional_columns": emitted,
            }
        }

    with pytest.raises(ValueError, match="invalid_csmar_column_override"):
        interim._validated_column_overrides(payload)

    assert emitted.consumed >= 2


@pytest.mark.parametrize("caller", ("iter", "apply"))
def test_public_column_override_callers_bound_lying_mapping(
    caller: str,
) -> None:
    emitted = _oversized_emitted_mapping()

    with pytest.raises(ValueError):
        if caller == "iter":
            interim.iter_interim_frames(
                data_id="stock_master",
                input_files=(),
                start_date=None,
                end_date=None,
                overrides=emitted,  # type: ignore[arg-type]
            )
        else:
            interim.apply_output_column_rules(
                pd.DataFrame(),
                "stock_master",
                overrides=emitted,  # type: ignore[arg-type]
            )

    assert emitted.consumed == len(interim.INTERIM_COLUMNS_BY_DATA_ID) + 1


@pytest.mark.parametrize("key", ("keep", "drop"))
@pytest.mark.parametrize("container_type", (list, tuple))
def test_column_override_lists_stop_at_owner_column_bound(
    key: str,
    container_type: type[list] | type[tuple],
) -> None:
    allowed = interim.INTERIM_COLUMNS_BY_DATA_ID["stock_master"]
    values = container_type([allowed[0]] * (len(allowed) + 1))

    with pytest.raises(ValueError, match="invalid_csmar_column_override"):
        interim._validated_column_overrides({"stock_master": {key: values}})


@pytest.mark.parametrize("forged_outer_proxy", (False, True))
def test_interim_operation_validates_and_snapshots_overrides_once(
    monkeypatch: pytest.MonkeyPatch,
    forged_outer_proxy: bool,
) -> None:
    calls = 0
    original = interim._validated_column_overrides

    def validate(overrides):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        return original(overrides)

    observed: list[object] = []

    def frames(**kwargs):  # noqa: ANN003, ANN202
        observed.append(kwargs["overrides"])
        yield pd.DataFrame({"value": [1]})
        observed.append(kwargs["overrides"])
        yield pd.DataFrame({"value": [2]})

    mutable_overrides = {
        "trading_calendar": {"keep": ["source_day_of_week"], "drop": []}
    }
    overrides = (
        MappingProxyType(mutable_overrides) if forged_outer_proxy else mutable_overrides
    )
    monkeypatch.setattr(interim, "_validated_column_overrides", validate)
    monkeypatch.setattr(interim, "_iter_pruned_interim_frames", frames)

    output = interim.iter_interim_frames(
        data_id="trading_calendar",
        input_files=[Path("private.csv")],
        start_date=None,
        end_date=None,
        overrides=overrides,
    )
    assert next(output)["value"].tolist() == [1]
    mutable_overrides["trading_calendar"]["keep"].append("timestamp")
    assert next(output)["value"].tolist() == [2]
    output.close()

    assert calls == 1
    assert observed[0] is observed[1]
    assert observed[0]["trading_calendar"]["keep"] == ("source_day_of_week",)


def test_loaded_column_overrides_compose_with_public_apply(
    tmp_path: Path,
) -> None:
    path = tmp_path / "overrides.json"
    path.write_text(
        '{"trading_calendar":{"keep":["source_day_of_week"],"drop":[]}}',
        encoding="utf-8",
    )
    overrides = interim.load_column_overrides(path)

    _, report = interim.apply_output_column_rules(
        pd.DataFrame({"source_day_of_week": [1]}),
        "trading_calendar",
        overrides=overrides,
    )

    assert report["manual_keep"] == ["source_day_of_week"]


def _collect_interim_frames(**kwargs: object) -> pd.DataFrame:
    frames = list(interim.iter_interim_frames(**kwargs))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_raw_export(
    path: Path,
    dataset: str,
    *,
    chunksize: int = 1024,
) -> pd.DataFrame:
    chunks = list(
        raw_exports.iter_private_export_chunks(
            path,
            dataset=dataset,
            chunksize=chunksize,
        )
    )
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def test_csmar_interim_catalog_registers_source_owned_specs() -> None:
    specs = interim.interim_catalog().list()

    assert {spec.data_id for spec in specs} == {
        frequency_ids.MARKET_BARS_DATA_ID,
        "stock_master",
        "market_cap",
        "adjustment",
        "index_info",
        "trading_calendar",
    }
    assert {str(spec.source_id) for spec in specs} == {"csmar"}


def test_csmar_trading_calendar_interim_spec_uses_source_calendar_grain() -> None:
    catalog = interim.interim_catalog()
    spec = catalog.get("trading_calendar")

    assert spec.grain is DataGrain.SOURCE_MARKET_CALENDAR_DATE
    assert spec.primary_key == ("timestamp", "source_market_type")
    assert spec.partition_columns == ("year",)
    assert "market_id" not in spec.required_columns
    assert "source_id" not in spec.required_columns
    assert "asset_id" not in spec.required_columns

    market_bars_daily = catalog.get(frequency_ids.MARKET_BARS_DATA_ID)
    assert market_bars_daily.grain is DataGrain.ASSET_TIMESTAMP
    assert market_bars_daily.primary_key == ("source_asset_id", "timestamp")
    assert "source_asset_id" in market_bars_daily.required_columns
    assert "asset_id" not in market_bars_daily.required_columns

    index_info = catalog.get("index_info")
    assert index_info.grain is DataGrain.ASSET_REFERENCE_TIMESTAMP
    assert index_info.primary_key == ("source_asset_id", "timestamp")
    assert index_info.partition_columns == ("year",)
    assert "timestamp" in index_info.required_columns
    assert "asset_id" not in index_info.required_columns


@pytest.mark.parametrize(
    "entrypoint",
    (raw_exports.iter_raw_export_files, raw_exports.dataset_files_with_identity_by_id),
)
def test_raw_inventory_indexes_use_source_sibling_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
) -> None:  # noqa: ANN001
    raw_root = tmp_path / "raw" / "csmar"
    raw_root.mkdir(parents=True)
    observed: list[Path] = []

    class Index:
        def __init__(self, *, scratch_parent: Path, **_kwargs: object) -> None:
            observed.append(scratch_parent)

        def __enter__(self) -> Index:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def add_identity_payloads(self, **_kwargs: object) -> None:
            pass

        def identity_payload_values(self, **_kwargs: object):
            return iter(())

    monkeypatch.setattr(raw_exports, "ExactPrimaryKeyIndex", Index)

    assert list(entrypoint(raw_root)) == []
    assert observed == [raw_root.parent]


def test_reference_dedup_uses_selected_sibling_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    scratch_parent = tmp_path / "interim"
    observed: list[Path] = []

    class Index:
        def __init__(self, *, scratch_parent: Path, **_kwargs: object) -> None:
            observed.append(scratch_parent)

        def __enter__(self) -> Index:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", Index)

    with runtime_resource_policy_scope(policy):
        assert (
            list(
                interim._iter_deduped_reference_frames(
                    iter(()),
                    data_id="stock_master",
                    pruning_reports=None,
                    scratch_parent=scratch_parent,
                )
            )
            == []
        )

    assert observed == [scratch_parent]


def test_standalone_reference_dedup_uses_source_sibling_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_parent = tmp_path / "raw"
    source_parent.mkdir()
    source = source_parent / "reference.csv"
    observed: list[Path | None] = []

    monkeypatch.setattr(
        interim,
        "_iter_pruned_interim_frames",
        lambda **_kwargs: iter(()),
    )

    def dedup(_frames, **kwargs: object):  # noqa: ANN001
        observed.append(kwargs.get("scratch_parent"))
        return iter(())

    monkeypatch.setattr(interim, "_iter_deduped_reference_frames", dedup)

    assert (
        list(
            interim.iter_interim_frames(
                data_id="stock_master",
                input_files=iter((source,)),
                start_date=None,
                end_date=None,
            )
        )
        == []
    )
    assert observed == [source_parent]


def test_csmar_duplicate_header_fails_at_inspection_and_reader_boundaries(
    tmp_path: Path,
) -> None:
    export = tmp_path / "TRDNEW_Dalyr.csv"
    frame = _market_bars_daily_raw_frame().iloc[:1].copy()
    frame.insert(1, "Stkcd_duplicate", frame["Stkcd"])
    frame.columns = ("Stkcd", "Stkcd", *frame.columns[2:])
    frame.to_csv(export, index=False)

    with pytest.raises(ValueError, match="duplicate_csmar_provider_column"):
        raw_schemas.inspect_raw_export_file(export)
    with pytest.raises(ValueError, match="duplicate_csmar_provider_column"):
        list(
            raw_exports.iter_private_export_chunks(
                export,
                dataset=frequency_ids.MARKET_BARS_DATA_ID,
                chunksize=1,
            )
        )


def test_csmar_provider_superset_is_projected_before_dataframe_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "daily_market.csv"
    frame = _market_bars_daily_raw_frame().iloc[:2].copy()
    extra_columns = tuple(f"ProviderExtra{index}" for index in range(5))
    for column in extra_columns:
        frame[column] = "unused"
    frame.to_csv(export, index=False)

    inspection = raw_schemas.inspect_raw_export_file(export)
    assert inspection.dataset == frequency_ids.MARKET_BARS_DATA_ID
    assert inspection.columns == tuple(frame.columns)

    original_raw_frame = raw_exports._raw_frame

    def projected_raw_frame(rows: list[list[str]], header: list[str]) -> pd.DataFrame:
        assert not set(extra_columns).intersection(header)
        assert all(len(row) == len(header) for row in rows)
        return original_raw_frame(rows, header)

    monkeypatch.setattr(raw_exports, "_raw_frame", projected_raw_frame)
    chunks = list(
        raw_exports.iter_private_export_chunks(
            export,
            dataset=frequency_ids.MARKET_BARS_DATA_ID,
            chunksize=2,
        )
    )

    assert sum(len(chunk) for chunk in chunks) == len(frame)
    assert all(
        tuple(chunk.columns)
        == tuple(column for column in frame.columns if column not in extra_columns)
        for chunk in chunks
    )


def test_csmar_raw_schema_sidecar_and_missing_columns_are_reported(
    tmp_path: Path,
) -> None:
    export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    sidecar = export.with_suffix(".metadata.json")
    _write_trading_calendar_raw_export(export)
    sidecar.write_text("{}", encoding="utf-8")

    inspection = raw_schemas.inspect_raw_export_file(export)
    assert raw_schemas.expected_metadata_sidecar_path(export) == sidecar
    assert inspection.has_metadata is True
    sidecar.unlink()
    assert raw_schemas.inspect_raw_export_file(export).has_metadata is False

    broken = tmp_path / "TRDNEW_Cale.csv"
    pd.DataFrame({"Markettype": [1], "Daywk": [1]}).to_csv(broken, index=False)
    broken_inspection = raw_schemas.inspect_raw_export_file(broken)

    assert broken_inspection.dataset == "trading_calendar"
    assert broken_inspection.missing_required_columns == ["Clddt", "State"]


def test_csmar_intake_rejects_missing_required_columns(tmp_path: Path) -> None:
    broken = tmp_path / "TRDNEW_Cale.csv"
    pd.DataFrame({"Markettype": [1], "Daywk": [1]}).to_csv(broken, index=False)

    intake = raw_exports.inspect_raw_export(broken)

    assert not intake.ok
    with pytest.raises(ValueError, match="csmar_raw_export_schema_invalid"):
        list(raw_exports.dataset_files_with_identity_by_id(tmp_path))


def test_csmar_dataset_inventory_fails_closed_on_inspection_error(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "TRDNEW_Cale.csv"
    broken.write_text('"unterminated\n', encoding="utf-8")

    with pytest.raises(ValueError, match="csmar_raw_export_inspection_failed"):
        list(raw_exports.dataset_files_with_identity_by_id(tmp_path))


def test_csmar_inspection_rejects_unregistered_extremely_wide_schema(
    tmp_path: Path,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    required = ("Markettype", "Clddt", "Daywk", "State")
    columns = (*required, *(f"extra_{position}" for position in range(50_000)))
    export.write_text(",".join(columns) + "\n", encoding="utf-8")

    intake = raw_exports.inspect_raw_export(export)

    assert not intake.ok
    assert intake.error == "csmar_raw_export_schema_invalid"


def test_csmar_index_info_requires_timestamp_source_columns_at_header_stage(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "IDX_Idxinfo.csv"
    pd.DataFrame({"Indexcd": ["000985"]}).to_csv(broken, index=False)

    inspection = raw_schemas.inspect_raw_export_file(broken)

    assert inspection.dataset == "index_info"
    assert inspection.missing_required_columns == ["Idxinfo11", "Idxinfo02"]
    assert field_mappings.missing_raw_column_names(
        ("Indexcd",),
        "index_info",
    ) == ["Idxinfo11", "Idxinfo02"]


def test_csmar_txt_field_description_is_not_raw_export_table(
    tmp_path: Path,
) -> None:
    csv_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    txt_description = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.txt"
    )
    _write_trading_calendar_raw_export(csv_export)
    txt_description.write_text("字段\t说明\nClddt\t交易日期\n", encoding="utf-8")

    exports = list(raw_exports.iter_raw_export_files(tmp_path))

    assert exports == [csv_export]
    with pytest.raises(ValueError, match="Cannot infer CSMAR dataset"):
        raw_schemas.inspect_raw_export_file(txt_description)


def test_csmar_raw_export_contract_is_csv_only(tmp_path: Path) -> None:
    csv_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    xlsx_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.xlsx"
    )
    _write_trading_calendar_raw_export(csv_export)
    xlsx_export.write_bytes(b"not-a-supported-raw-export")

    assert raw_exports.SUPPORTED_RAW_EXPORT_SUFFIXES == {".csv"}
    assert list(raw_exports.iter_raw_export_files(tmp_path)) == [csv_export]
    with pytest.raises(ValueError, match="Invalid CSMAR raw filename"):
        raw_schemas.parse_raw_filename(xlsx_export)


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
@pytest.mark.parametrize("entrypoint", ("inventory", "identified"))
def test_csmar_raw_inventory_does_not_descend_nested_junction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    export = external / "TRDNEW_Cale.csv"
    export.write_text("Markettype,Clddt,Daywk,State\n", encoding="utf-8")
    junction = root / "nested"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")
    if entrypoint == "inventory":
        result = list(raw_exports.iter_raw_export_files(root))
    else:
        monkeypatch.setattr(
            raw_exports,
            "inspect_raw_export",
            lambda path, **_kwargs: raw_exports.RawExportIntakeResult(
                path=path,
                inspection=SimpleNamespace(
                    dataset="trading_calendar",
                    missing_required_columns=(),
                ),
            ),
        )
        result = list(raw_exports.dataset_files_with_identity_by_id(root))

    assert result == []


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
@pytest.mark.parametrize("entrypoint", ("inventory", "identified"))
def test_csmar_raw_inventory_rejects_junction_root(
    tmp_path: Path,
    entrypoint: str,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "raw"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(external)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")
    operation = (
        raw_exports.iter_raw_export_files(root)
        if entrypoint == "inventory"
        else raw_exports.dataset_files_with_identity_by_id(root)
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(operation)


def test_csmar_raw_inventory_maps_walk_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_walk(_self: Path, *, on_error, **_kwargs):  # noqa: ANN001, ANN202
        on_error(OSError("walk_failed"))
        yield

    monkeypatch.setattr(Path, "walk", fail_walk)

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(raw_exports.iter_raw_export_files(tmp_path))


def test_csmar_raw_inventory_rejects_file_symlink(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    root.mkdir()
    external = tmp_path / "TRDNEW_Cale.csv"
    external.write_text("Markettype,Clddt,Daywk,State\n", encoding="utf-8")
    alias = root / "alias.csv"
    try:
        alias.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(raw_exports.iter_raw_export_files(root))


def test_raw_export_inventory_keeps_case_distinct_exact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upper = tmp_path / "A.csv"
    lower = tmp_path / "a.csv"
    upper.touch()
    lower.touch()
    upper.touch()
    lower.touch()
    monkeypatch.setattr(
        Path,
        "walk",
        lambda self, **_kwargs: iter(((self, [], ["a.csv", "A.csv"]),)),
    )
    identities = {
        "A.csv": DatasetFileIdentity(device=1, inode=1, size=0, modified_ns=1),
        "a.csv": DatasetFileIdentity(device=1, inode=2, size=0, modified_ns=1),
    }
    monkeypatch.setattr(
        raw_exports,
        "raw_export_file_identity",
        lambda path: identities[path.name],
    )

    assert list(raw_exports.iter_raw_export_files(tmp_path)) == [upper, lower]


def test_dataset_inventory_keeps_case_distinct_exact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upper = tmp_path / "A.csv"
    lower = tmp_path / "a.csv"
    upper.touch()
    lower.touch()
    inspection = raw_schemas.RawExportInspection(
        path=upper,
        dataset="trading_calendar",
        table_code="TRDNEW_Cale",
        columns=("Markettype", "Clddt", "Daywk", "State"),
        missing_required_columns=[],
        has_metadata=False,
    )
    monkeypatch.setattr(
        Path,
        "walk",
        lambda self, **_kwargs: iter(((self, [], ["a.csv", "A.csv"]),)),
    )
    identities = {
        "A.csv": DatasetFileIdentity(device=1, inode=1, size=0, modified_ns=1),
        "a.csv": DatasetFileIdentity(device=1, inode=2, size=0, modified_ns=1),
    }
    monkeypatch.setattr(
        raw_exports,
        "raw_export_file_identity",
        lambda path: identities[path.name],
    )
    monkeypatch.setattr(
        raw_exports,
        "_inspect_raw_export_with_identity",
        lambda path, **_kwargs: raw_exports.RawExportIntakeResult(
            path=path,
            inspection=inspection,
        ),
    )

    assert list(raw_exports.dataset_files_with_identity_by_id(tmp_path)) == [
        ("trading_calendar", upper, identities["A.csv"]),
        ("trading_calendar", lower, identities["a.csv"]),
    ]


def test_csmar_dataset_files_with_identity_groups_raw_exports_once(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "provider_exports"
    raw_dir.mkdir()
    market_export = raw_dir / "TRDNEW_Dalyr.csv"
    calendar_export = raw_dir / "TRDNEW_Cale.csv"
    _market_bars_daily_raw_frame().to_csv(market_export, index=False)
    _write_trading_calendar_raw_export(calendar_export)

    grouped = list(raw_exports.dataset_files_with_identity_by_id(tmp_path))

    assert set(grouped) == {
        (
            frequency_ids.MARKET_BARS_DATA_ID,
            market_export,
            raw_schemas.raw_export_file_identity(market_export),
        ),
        (
            "trading_calendar",
            calendar_export,
            raw_schemas.raw_export_file_identity(calendar_export),
        ),
    }


def test_dataset_inventory_binds_identity_into_inspection_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    expected_identity = raw_schemas.raw_export_file_identity(export)
    observed: list[DatasetFileIdentity | None] = []
    original_inspect = raw_exports.inspect_raw_export_file

    def inspect_with_observation(path, **kwargs):  # noqa: ANN001, ANN003, ANN202
        observed.append(kwargs.get("expected_identity"))
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(
        raw_exports,
        "inspect_raw_export_file",
        inspect_with_observation,
    )

    assert list(raw_exports.dataset_files_with_identity_by_id(tmp_path)) == [
        ("trading_calendar", export, expected_identity)
    ]
    assert observed == [expected_identity]


def test_raw_export_inventory_is_a_deterministic_iterator(tmp_path: Path) -> None:
    later = tmp_path / "z" / "TRDNEW_Cale.csv"
    earlier = tmp_path / "a" / "TRDNEW_Cale.csv"
    later.parent.mkdir()
    earlier.parent.mkdir()
    _write_trading_calendar_raw_export(later)
    _write_trading_calendar_raw_export(earlier)

    inventory = raw_exports.iter_raw_export_files(tmp_path)

    assert iter(inventory) is inventory
    assert list(inventory) == [earlier, later]


def test_raw_export_inventory_rejects_identity_change_during_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    original = raw_exports._inspect_raw_export_with_identity

    def replace_after_inspection(path: Path, **kwargs):  # noqa: ANN003, ANN202
        result = original(path, **kwargs)
        stat = path.stat()
        payload = path.read_bytes()
        path.write_bytes(payload.replace(b",O", b",C", 1))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(
        raw_exports,
        "_inspect_raw_export_with_identity",
        replace_after_inspection,
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(raw_exports.dataset_files_with_identity_by_id(tmp_path))


def test_csv_owner_rejects_same_handle_size_mtime_change_at_close(
    tmp_path: Path,
) -> None:
    export = tmp_path / "calendar.csv"
    _write_trading_calendar_raw_export(export)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    rows = raw_schemas.iter_export_csv_records(export, resource_policy=policy)
    next(rows)
    with export.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(rows)


def test_csmar_field_mappings_describe_source_local_aliases() -> None:
    assert "source-local normalized column mappings" in (
        field_mappings.column_mapping.__doc__ or ""
    )
    assert "source-local normalized aliases" in (
        field_mappings.map_columns.__doc__ or ""
    )

    mapped = field_mappings.map_columns(
        pd.DataFrame(
            {
                "Markettype": [1],
                "Clddt": ["2024-01-06"],
                "Daywk": [6],
                "State": ["O"],
            }
        ),
        "trading_calendar",
    )

    assert mapped.columns.tolist() == [
        "source_market_type",
        "calendar_date",
        "source_day_of_week",
        "source_state",
    ]
    assert field_mappings.missing_raw_column_names(
        ("Markettype", "Daywk"),
        "trading_calendar",
    ) == ["Clddt", "State"]
    assert set(field_mappings.field_descriptions("trading_calendar")) == {
        "Markettype",
        "Clddt",
        "Daywk",
        "State",
    }


def test_csmar_raw_reader_preserves_code_strings_and_index_formula_commas(
    tmp_path: Path,
) -> None:
    daily_path = tmp_path / "daily.csv"
    pd.DataFrame(
        {
            "Stkcd": ["000001"],
            "Trddt": ["2024-01-02"],
            "Markettype": [4],
        }
    ).to_csv(daily_path, index=False)

    daily = _collect_raw_export(
        daily_path,
        frequency_ids.MARKET_BARS_DATA_ID,
    )
    assert daily.loc[0, "Stkcd"] == "000001"

    index_path = tmp_path / "index_info.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo07,Idxinfo10\n"
        '000985,benchmark,2014-01-02,2013-12-31,broad,"a,b,c"\n'
        "000300,large,,,,\n",
        encoding="utf-8",
    )
    index_info = _collect_raw_export(index_path, "index_info")

    assert index_info.loc[0, "Indexcd"] == "000985"
    assert index_info.loc[0, "Idxinfo10"] == "a,b,c"
    assert pd.isna(index_info.loc[1, "Idxinfo11"])


def test_csmar_raw_reader_rejects_nonnumeric_registered_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "daily.csv"
    raw = _market_bars_daily_raw_frame().iloc[:1].copy()
    raw["Opnprc"] = pd.Series(["not-a-number"], dtype="object")
    raw.to_csv(path, index=False)

    with pytest.raises(ValueError, match="^invalid_csmar_numeric_value$"):
        list(
            raw_exports.iter_private_export_chunks(
                path,
                dataset=frequency_ids.MARKET_BARS_DATA_ID,
                chunksize=1,
            )
        )


def test_csmar_dirty_index_info_csv_preserves_alignment_and_raw_semantics(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_dirty.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,all shares,"
        "1,2,3,4,5,weighted formula, with English comma，中文标点\n"
        "000300,large,2014-01-03,2013-12-31,1000,blue chips,"
        'W,P,C,T,M,"formula with embedded line\nbreak and ""quotes"""\n',
        encoding="utf-8",
    )

    raw = _collect_raw_export(index_path, "index_info")
    normalized = interim.normalize_raw_frame(raw, "index_info")
    pruned, _ = interim.apply_output_column_rules(normalized, "index_info")

    assert len(raw) == 2
    assert raw.loc[0, "Idxinfo10"] == ("weighted formula, with English comma，中文标点")
    assert "embedded line\nbreak" in raw.loc[1, "Idxinfo10"].replace("\r\n", "\n")
    benchmark = pruned.set_index("source_asset_id").loc["000985"]
    large = pruned.set_index("source_asset_id").loc["000300"]
    assert pd.isna(benchmark["base_point"])
    assert large["base_point"] == 1000.0
    assert benchmark["start_date"] == pd.Timestamp("2014-01-02")
    assert benchmark["base_date"] == pd.Timestamp("2013-12-31")
    assert benchmark["weighting_method"] == "1"
    assert benchmark["publisher"] == "2"
    assert benchmark["index_category"] == "3"
    assert benchmark["index_type"] == "4"
    assert benchmark["market_belonging"] == "5"
    assert benchmark["calculation_formula"].endswith("中文标点")


def test_csmar_index_info_unquoted_text_newlines_recover_when_aligned(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_unquoted_newlines.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,sample scope first line\n"
        "sample scope second line,1,2,3,4,5,formula text\n"
        "000300,large,2014-01-03,2013-12-31,1000,blue chips,"
        "1,2,3,4,5,formula first line\n"
        "formula second line with, comma\n",
        encoding="utf-8",
    )

    raw = _collect_raw_export(index_path, "index_info")

    assert len(raw) == 2
    assert raw.loc[0, "Idxinfo04"] == (
        "sample scope first line\nsample scope second line"
    )
    assert raw.loc[0, "Idxinfo05"] == "1"
    assert raw.loc[0, "Idxinfo09"] == "5"
    assert raw.loc[1, "Idxinfo05"] == "1"
    assert raw.loc[1, "Idxinfo09"] == "5"
    assert raw.loc[1, "Idxinfo10"] == (
        "formula first line\nformula second line with, comma"
    )


@pytest.mark.parametrize("extra_bytes", (0, 1))
def test_csmar_reconstructed_index_info_uses_exact_source_bytes_at_boundary(
    tmp_path: Path,
    extra_bytes: int,
) -> None:
    path = tmp_path / "index-info-wire-boundary.csv"
    header = (
        b"\xef\xbb\xbfIndexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        b"Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\r\n"
    )
    first_prefix = b'000985,benchmark,2014-01-02,2013-12-31,-9666,"a""'
    first_suffix = b'"\r\n'
    continuation = b'continued text,1,2,3,4,5,"formula ""quoted"""\r\n'
    filler_size = CSMAR_LOGICAL_RECORD_LIMIT_BYTES - sum(
        map(len, (first_prefix, first_suffix, continuation))
    )
    reconstructed = (
        first_prefix + b"x" * (filler_size + extra_bytes) + first_suffix + continuation
    )
    path.write_bytes(header + reconstructed)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    rows = iter(raw_exports.iter_index_info_rows(path, resource_policy=policy))
    if extra_bytes:
        with pytest.raises(
            ValueError, match="csmar_logical_record_exceeds_source_limit"
        ):
            next(rows)
    else:
        _, row, source_bytes = next(rows)
        assert row[0] == "000985"
        assert row[-1] == 'formula "quoted"'
        assert source_bytes == len(reconstructed) == CSMAR_LOGICAL_RECORD_LIMIT_BYTES
        rows.close()


def test_csmar_index_info_sample_scope_and_formula_commas_preserve_alignment(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_commas.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample scope, with comma,1,2,3,4,5,formula text, with comma\n",
        encoding="utf-8",
    )

    raw = _collect_raw_export(index_path, "index_info")

    assert len(raw) == 1
    assert raw.loc[0, "Idxinfo04"] == "sample scope, with comma"
    assert raw.loc[0, "Idxinfo05"] == "1"
    assert raw.loc[0, "Idxinfo09"] == "5"
    assert raw.loc[0, "Idxinfo10"] == "formula text, with comma"


def test_csmar_index_info_ignores_terminal_code_like_formula_candidate(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_code_like_formula.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample,scope,with comma,1,1,1,1,1,1\n",
        encoding="utf-8",
    )

    raw = _collect_raw_export(index_path, "index_info")

    assert raw.loc[0, "Idxinfo04"] == "sample,scope,with comma"
    assert raw.loc[0, "Idxinfo10"] == "1"
    assert str(raw["Idxinfo10"].dtype) == "string"


@pytest.mark.parametrize(
    ("filename", "row"),
    [
        (
            "index_info_ambiguous.csv",
            "000985,benchmark,2014-01-02,2013-12-31,-9666,"
            "1,2,3,4,1,1,2,3,4,5,formula text\n",
        ),
        (
            "index_info_invalid_enum_long_row.csv",
            "000985,benchmark,2014-01-02,2013-12-31,-9666,"
            "sample scope, with comma,5,1,1,1,1,formula text\n",
        ),
    ],
)
def test_csmar_index_info_malformed_long_rows_fail_closed(
    tmp_path: Path,
    filename: str,
    row: str,
) -> None:
    index_path = tmp_path / filename
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n" + row,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        _collect_raw_export(index_path, "index_info")


def test_index_info_ambiguity_fails_before_candidate_row_slices() -> None:
    class ObservedRow(list[str]):
        slice_count = 0

        def __getitem__(self, key):  # noqa: ANN001, ANN204
            if isinstance(key, slice):
                self.slice_count += 1
            return super().__getitem__(key)

    row = ObservedRow(
        [
            "000985",
            "benchmark",
            "2014-01-02",
            "2013-12-31",
            "1000",
            "ambiguous",
            "1",
            "2",
            "3",
            "4",
            "5",
            "tail",
            "1",
            "2",
            "3",
            "4",
            "5",
            "formula",
        ]
    )

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        raw_exports._normalize_long_index_info_row(row, 2)

    assert row.slice_count == 0


def test_index_info_reconstruction_admits_result_overlap_before_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    header = [f"column_{position}" for position in range(12)]
    row = [
        "000985",
        "benchmark",
        "2014-01-02",
        "2013-12-31",
        "1000",
        "sample",
        "scope",
        "1",
        "2",
        "3",
        "4",
        "5",
        "formula",
    ]
    record = SimpleNamespace(
        field_count=len(row),
        iter_fields=lambda: iter(row),
        materialize=lambda *, max_fields: row,
    )
    monkeypatch.setattr(
        raw_exports,
        "_iter_csv_records",
        lambda *_args, **_kwargs: iter(
            ((header, record, policy.task_memory_budget_bytes),)
        ),
    )

    rows = iter(
        raw_exports.iter_index_info_rows(
            Path("unused.csv"),
            resource_policy=policy,
        )
    )
    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(rows)


def test_index_info_continuation_admits_before_materialized_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "index-info-tight-headroom.csv"
    path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000905,mid,2014-01-06,2013-12-31,500,"
        + "s" * 500_000
        + "\n"
        + "x" * 100_000
        + ",1,2,3,4,5,formula\n",
        encoding="utf-8",
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    opened: list[object] = []
    original = tempfile.TemporaryFile

    class TrackedFile:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self._handle = original(*args, **kwargs)
            self.closed = False
            opened.append(self)

        def __getattr__(self, name: str):  # noqa: ANN204
            return getattr(self._handle, name)

        def close(self) -> None:
            self.closed = True
            self._handle.close()

    monkeypatch.setattr(raw_exports.tempfile, "TemporaryFile", TrackedFile)
    monkeypatch.setattr(
        raw_exports.io,
        "StringIO",
        lambda: pytest.fail("materialized_before_capacity_admission"),
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        list(raw_exports.iter_index_info_rows(path, resource_policy=policy))

    assert opened
    assert all(item.closed for item in opened)


def test_csmar_index_info_aligned_out_of_enum_code_is_cleaned_to_missing(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index_info_out_of_enum.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,"
        "sample scope,5,1,1,1,1,formula text\n",
        encoding="utf-8",
    )

    raw = _collect_raw_export(index_path, "index_info")

    assert pd.isna(raw.loc[0, "Idxinfo05"])
    assert raw.loc[0, "Idxinfo10"] == "formula text"


def test_csmar_index_info_dirty_csv_and_bad_dates_fail_closed(
    tmp_path: Path,
) -> None:
    short_row = tmp_path / "index_info_short.csv"
    short_row.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo10\n000985,benchmark,2014-01-02\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        _collect_raw_export(short_row, "index_info")

    bad_date = pd.DataFrame(
        {
            "Indexcd": ["000985"],
            "Idxinfo01": ["benchmark"],
            "Idxinfo11": ["bad-date"],
            "Idxinfo02": ["2013-12-31"],
        }
    )
    with pytest.raises(ValueError, match="invalid_index_info_date"):
        interim.normalize_raw_frame(
            bad_date,
            "index_info",
        )


def test_csmar_daily_interim_normalization_uses_source_asset_id_only() -> None:
    normalized = interim.normalize_raw_frame(
        _market_bars_daily_raw_frame(),
        frequency_ids.MARKET_BARS_DATA_ID,
    )
    pruned, report = interim.apply_output_column_rules(
        normalized,
        frequency_ids.MARKET_BARS_DATA_ID,
    )

    assert pruned["source_asset_id"].tolist() == ["000001", "000002"]
    assert pruned["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
    assert not {
        "market_id",
        "asset_id",
        "source_id",
        "eligible",
        "can_enter",
        "can_exit",
        "available_at",
        "source_exchange",
    }.intersection(pruned.columns)
    assert pruned.columns.tolist() == list(interim.MARKET_BARS_INTERIM_COLUMNS)
    assert pruned["turnover"].dtype == "float64"
    assert pruned["turnover"].isna().all()
    assert report["data_id"] == frequency_ids.MARKET_BARS_DATA_ID


def test_csmar_empty_daily_interim_preserves_turnover_float64_dtype() -> None:
    raw = _market_bars_daily_raw_frame().iloc[0:0]

    normalized = interim.normalize_raw_frame(
        raw,
        frequency_ids.MARKET_BARS_DATA_ID,
    )
    pruned, _ = interim.apply_output_column_rules(
        normalized,
        frequency_ids.MARKET_BARS_DATA_ID,
    )

    assert pruned.empty
    assert pruned.columns.tolist() == list(interim.MARKET_BARS_INTERIM_COLUMNS)
    assert pruned["turnover"].dtype == "float64"


@pytest.mark.parametrize("index_code", [None, pd.NA, "", [], {}, set(), 123])
def test_csmar_index_info_rejects_invalid_index_code(index_code: object) -> None:
    raw = pd.DataFrame(
        {
            "Indexcd": [index_code],
            "Idxinfo01": ["index"],
            "Idxinfo11": ["2014-01-01"],
            "Idxinfo02": ["2014-01-01"],
            "Idxinfo03": [1000.0],
            "Idxinfo04": ["scope"],
            "Idxinfo05": ["method"],
            "Idxinfo06": ["publisher"],
            "Idxinfo07": ["class"],
            "Idxinfo08": ["type"],
            "Idxinfo09": ["market"],
            "Idxinfo10": ["formula"],
        }
    )

    with pytest.raises(ValueError, match="invalid_csmar_index_code"):
        interim.normalize_raw_frame(raw, "index_info")


def test_csmar_narrow_market_bars_optional_raw_columns_do_not_block_mapping() -> None:
    data_id = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
    optional_columns = (
        "Dretnd",
        "Adjprcnd",
        "ChangeRatio",
        "LimitDown",
        "LimitUp",
    )
    raw = _market_bars_daily_raw_frame().drop(columns=list(optional_columns))

    inspection = raw_schemas.RawExportInspection(
        path=Path("narrow.csv"),
        dataset=data_id,
        table_code=raw_schemas.RAW_TABLE_CONTRACTS[data_id].table_code,
        columns=tuple(raw.columns),
        missing_required_columns=field_mappings.missing_raw_columns(raw, data_id),
        has_metadata=False,
    )
    normalized = interim.normalize_raw_frame(raw, data_id)
    pruned, report = interim.apply_output_column_rules(normalized, data_id)

    assert inspection.missing_required_columns == []
    assert "period_return" in pruned.columns
    assert "adj_close" in pruned.columns
    assert report["optional_available_columns"] == {}


def test_csmar_market_cap_optional_share_columns_do_not_block_mapping() -> None:
    data_id = "market_cap"
    raw = pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Shrchgdt": ["2024-01-02"],
            "Nshrttl": [1000.0],
            "Nshra": [800.0],
        }
    )

    assert field_mappings.missing_raw_columns(raw, data_id) == []
    normalized = interim.normalize_raw_frame(raw, data_id)
    pruned, report = interim.apply_output_column_rules(normalized, data_id)

    assert pruned["total_shares"].tolist() == [1000.0]
    assert pruned["a_float_shares"].tolist() == [800.0]
    assert report["optional_available_columns"] == {}


def test_csmar_missing_required_raw_columns_still_fails() -> None:
    data_id = csmar_ids.market_bars_data_id(DataFrequency.DAILY)
    raw = _market_bars_daily_raw_frame().drop(columns=["Dretwd"])

    assert field_mappings.missing_raw_columns(raw, data_id) == ["Dretwd"]
    with pytest.raises(
        ValueError,
        match=f"{frequency_ids.MARKET_BARS_DATA_ID} raw export is missing",
    ):
        field_mappings.map_columns(raw, data_id)


def test_csmar_trading_calendar_normalizer_outputs_source_contract() -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [4, 1, 4],
            "Clddt": ["2024-01-06", "2024-01-02", "2024-01-07"],
            "Daywk": [6, 2, 0],
            "State": ["O", "C", "C"],
        }
    )

    result = interim.normalize_raw_frame(raw, "trading_calendar")

    assert result.columns.tolist() == [
        "timestamp",
        "calendar_date",
        "source_market_type",
        "is_trading_day",
        "source_day_of_week",
        "source_state",
    ]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-06"),
        pd.Timestamp("2024-01-07"),
    ]
    assert result["is_trading_day"].dtype == bool
    assert result["is_trading_day"].tolist() == [False, True, False]
    assert not {"market_id", "source_id", "asset_id"}.intersection(result.columns)


@pytest.mark.parametrize("reverse_rows", (False, True))
def test_csmar_calendar_chunk_normalizes_date_first_with_market_tie_break(
    reverse_rows: bool,
) -> None:
    raw = pd.DataFrame(
        {
            "Markettype": [4, 1, 4, 1],
            "Clddt": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-03"],
            "Daywk": [3, 2, 2, 3],
            "State": ["O", "O", "C", "C"],
        }
    )
    if reverse_rows:
        raw = raw.iloc[::-1].reset_index(drop=True)

    result = interim.normalize_raw_frame(raw, "trading_calendar")

    assert list(
        result[["timestamp", "source_market_type"]].itertuples(
            index=False,
            name=None,
        )
    ) == [
        (pd.Timestamp("2024-01-02"), 1),
        (pd.Timestamp("2024-01-02"), 4),
        (pd.Timestamp("2024-01-03"), 1),
        (pd.Timestamp("2024-01-03"), 4),
    ]


def test_csmar_stock_master_missing_effective_dates_fails_closed() -> None:
    raw = pd.DataFrame(
        {
            "Stkcd": ["1"],
            "Stknme": ["demo"],
            "Listdt": [""],
            "DelistedDate": [""],
            "Markettype": [4],
            "Nnindcd": ["I1"],
            "Nnindnme": ["industry"],
            "Statco": ["normal"],
            "Statdt": [""],
        }
    )

    with pytest.raises(ValueError, match="missing_stock_master_effective_timestamp"):
        interim.normalize_raw_frame(raw, "stock_master")


def test_csmar_stock_master_dates_have_stable_cross_chunk_precision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stock_master.csv"
    first = _stock_master_raw_row("normal", "I1", "2020-01-01")
    second = _stock_master_raw_row("normal", "I1", "2020-01-02")
    second["Stkcd"] = "2"
    second["DelistedDate"] = "2020-06-01"
    pd.DataFrame([first, second]).to_csv(path, index=False)

    chunks = list(
        raw_exports.iter_private_export_chunks(
            path,
            dataset="stock_master",
            chunksize=1,
        )
    )
    normalized = [
        interim.normalize_raw_frame(chunk, "stock_master") for chunk in chunks
    ]

    assert len(normalized) == 2
    for frame in normalized:
        for column in (
            "timestamp",
            "listing_date",
            "delisting_date",
            "status_change_date",
        ):
            assert str(frame[column].dtype) == "datetime64[ns]"


def test_csmar_reference_interim_dedupes_exact_duplicates_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    exact_a = tmp_path / "stock_master_exact_a.csv"
    exact_b = tmp_path / "stock_master_exact_b.csv"
    conflict = tmp_path / "stock_master_conflict.csv"
    pd.DataFrame([_stock_master_raw_row("normal", "I1", "2020-01-01")]).to_csv(
        exact_a,
        index=False,
    )
    pd.DataFrame([_stock_master_raw_row("normal", "I1", "2020-01-01")]).to_csv(
        exact_b,
        index=False,
    )
    pd.DataFrame(
        [
            _stock_master_raw_row("normal", "I1", "2020-01-01"),
            _stock_master_raw_row("renamed", "I2", "2020-01-01"),
        ]
    ).to_csv(conflict, index=False)

    pruning_reports: list[dict[str, object]] = []
    deduped = _collect_interim_frames(
        data_id="stock_master",
        input_files=[exact_a, exact_b],
        start_date=None,
        end_date=None,
        chunksize=1,
        pruning_reports=pruning_reports,
    )

    assert len(deduped) == 1
    assert [
        report["reference_exact_duplicate_count"]
        for report in pruning_reports
        if report.get("report_type") == "reference_duplicate_summary"
    ] == [1]
    assert "exact duplicate primary-key rows deduped: 1" in (
        interim.render_column_pruning_report(pruning_reports)
    )
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: stock_master",
    ):
        _collect_interim_frames(
            data_id="stock_master",
            input_files=[conflict],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


@pytest.mark.parametrize(
    "error_type",
    (RuntimeError, GeneratorExit, KeyboardInterrupt, SystemExit),
)
def test_csmar_reference_dedup_summary_is_atomic_on_upstream_failure(
    error_type: type[BaseException],
) -> None:
    reports: list[dict[str, object]] = []
    closed: list[bool] = []

    def failing_frames():
        try:
            yield pd.DataFrame(
                {
                    "source_asset_id": ["000001", "000001"],
                    "timestamp": [pd.Timestamp("2020-01-01")] * 2,
                    "stock_name": [pd.NA, pd.NA],
                }
            )
            raise error_type("upstream_failed")
        finally:
            closed.append(True)

    with pytest.raises(error_type, match="upstream_failed"):
        list(
            interim._iter_deduped_reference_frames(
                failing_frames(),
                data_id="stock_master",
                pruning_reports=reports,
                scratch_parent=Path("."),
            )
        )

    assert not any(
        report.get("report_type") == "reference_duplicate_summary" for report in reports
    )
    assert closed == [True]


def test_csmar_reference_dedup_empty_success_has_atomic_zero_summary() -> None:
    reports: list[dict[str, object]] = []

    assert (
        list(
            interim._iter_deduped_reference_frames(
                iter(()),
                data_id="stock_master",
                pruning_reports=reports,
                scratch_parent=Path("."),
            )
        )
        == []
    )
    assert [
        report["reference_exact_duplicate_count"]
        for report in reports
        if report.get("report_type") == "reference_duplicate_summary"
    ] == [0]


def test_csmar_reference_dedup_is_typed_bounded_exact_across_chunks() -> None:
    reports: list[dict[str, object]] = []
    timestamp = pd.Timestamp("2020-01-01")
    exact_frames = iter(
        (
            pd.DataFrame(
                {
                    "source_asset_id": pd.Series(["000001"], dtype="string"),
                    "timestamp": [timestamp],
                    "nullable": pd.Series([pd.NA], dtype="Int64"),
                    "nonfinite": [float("inf")],
                }
            ),
            pd.DataFrame(
                {
                    "nonfinite": [float("inf")],
                    "nullable": pd.Series([pd.NA], dtype="Int64"),
                    "timestamp": [timestamp],
                    "source_asset_id": pd.Series(["000001"], dtype="string"),
                }
            ),
        )
    )

    result = list(
        interim._iter_deduped_reference_frames(
            exact_frames,
            data_id="stock_master",
            pruning_reports=reports,
            scratch_parent=Path("."),
        )
    )

    assert sum(len(frame) for frame in result) == 1
    assert reports[-1]["reference_exact_duplicate_count"] == 1
    conflict = iter(
        (
            result[0],
            result[0].assign(nonfinite=float("-inf")),
        )
    )
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: stock_master",
    ):
        list(
            interim._iter_deduped_reference_frames(
                conflict,
                data_id="stock_master",
                pruning_reports=[],
                scratch_parent=Path("."),
            )
        )


def test_csmar_reference_dedup_delegates_typed_classification_to_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    class RecordingIndex:
        retained_memory_bytes = 1024

        def __init__(self, *, primary_key, **_kwargs):
            self.primary_key = tuple(primary_key)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def classify_comparison_table(self, table, *, value_columns, **_kwargs):
            observed.append((self.primary_key, tuple(value_columns)))
            return CompactPositionState(
                array("Q", range(table.num_rows)), table.num_rows
            ), 0

    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", RecordingIndex, raising=False)
    frames = iter(
        (
            pd.DataFrame(
                {
                    "source_asset_id": ["000001"],
                    "timestamp": [pd.Timestamp("2020-01-01")],
                    "stock_name": [pd.NA],
                }
            ),
        )
    )

    assert (
        len(
            list(
                interim._iter_deduped_reference_frames(
                    frames,
                    data_id="stock_master",
                    pruning_reports=[],
                    scratch_parent=Path("."),
                )
            )
        )
        == 1
    )
    assert len(observed) == 1
    assert observed[0][0] == ("source_asset_id", "timestamp")
    assert observed[0][1] == ("stock_name",)


def test_csmar_reference_opaque_arrow_overrun_fails_before_index_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classified_sizes: list[int] = []

    class RecordingIndex:
        retained_memory_bytes = 0

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def classify_comparison_table(self, table, **_kwargs):
            classified_sizes.append(table.num_rows)
            return CompactPositionState(
                array("Q", range(table.num_rows)), table.num_rows
            ), 0

    class Table:
        def __init__(self, rows: int) -> None:
            self.num_rows = rows
            self.nbytes = 2 * 1024**2 if rows > 1 else 1

    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", RecordingIndex)
    monkeypatch.setattr(
        interim.pa,
        "Table",
        SimpleNamespace(from_pandas=lambda frame, **_kwargs: Table(len(frame))),
    )
    frame = pd.DataFrame(
        {
            "source_asset_id": [f"{position:06d}" for position in range(4)],
            "timestamp": [pd.Timestamp("2020-01-01")] * 4,
            "stock_name": ["name"] * 4,
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="csmar_reference_working_set_exceeds_task_memory_budget",
        ):
            list(
                interim._iter_deduped_reference_frames(
                    iter((frame,)),
                    data_id="stock_master",
                    pruning_reports=[],
                    scratch_parent=Path("."),
                )
            )

    assert classified_sizes == []


def test_csmar_interim_uses_one_observed_downshift_then_fixed_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=1000,
        explicit_cap_bytes=100,
    )
    frame = pd.DataFrame(
        {
            "source_asset_id": [f"{position:06d}" for position in range(8)],
            "timestamp": pd.date_range("2020-01-01", periods=8),
        }
    )
    calls: list[int] = []

    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda *_args, **_kwargs: iter((frame,)),
    )
    monkeypatch.setattr(interim, "pandas_frame_payload_bytes", lambda _frame: 10)

    def transform(candidate, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(len(candidate))
        if len(candidate) == len(frame):
            raise interim._InterimCapacityExceeded(200, 100)  # noqa: SLF001
        return candidate.copy(), {}

    monkeypatch.setattr(interim, "_transform_interim_chunk", transform)

    with runtime_resource_policy_scope(policy):
        output = tuple(
            interim._iter_pruned_interim_frames(
                data_id="stock_master",
                raw_dataset="stock_master",
                input_files=(Path("unused.csv"),),
                start_date=None,
                end_date=None,
                overrides={},
                chunksize=8,
                pruning_reports=None,
            )
        )

    assert calls == [8, 3, 3, 2]
    assert sum(map(len, output)) == len(frame)


def test_csmar_direct_lazy_iteration_captures_policy_before_first_raw_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    events: list[str] = []

    def active_policy() -> RuntimeResourcePolicy:
        events.append("policy")
        return policy

    def raw_frames(**_kwargs: object):
        events.append("raw")
        yield pd.DataFrame({"source_asset_id": ["000001"]})

    monkeypatch.setattr(interim, "active_runtime_resource_policy", active_policy)
    monkeypatch.setattr(interim, "_iter_pruned_interim_frames", raw_frames)

    frames = interim.iter_interim_frames(
        data_id=frequency_ids.MARKET_BARS_DATA_ID,
        input_files=[Path("private.csv")],
        start_date=None,
        end_date=None,
    )
    assert events == []

    next(frames)

    assert events == ["policy", "raw"]
    frames.close()


def test_csmar_pruning_reports_merge_repeated_chunk_facts_by_data_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tuple(
        pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(position, "D")],
                "value": [position],
            }
        )
        for position in range(100)
    )
    report = {
        "data_id": frequency_ids.MARKET_BARS_DATA_ID,
        "report_type": "column_pruning",
        "kept_columns": ["value"],
        "dropped_columns": [],
    }
    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda *_args, **_kwargs: iter(chunks),
    )
    monkeypatch.setattr(interim, "normalize_raw_frame", lambda frame, _data_id: frame)
    monkeypatch.setattr(
        interim,
        "_filter_normalized_frame",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(
        interim,
        "_apply_output_column_rules_validated",
        lambda frame, _data_id, **_kwargs: (frame.copy(), dict(report)),
    )
    reports: list[dict[str, object]] = []

    assert (
        len(
            list(
                interim._iter_pruned_interim_frames(
                    data_id=frequency_ids.MARKET_BARS_DATA_ID,
                    raw_dataset=frequency_ids.MARKET_BARS_DATA_ID,
                    input_files=[Path("private.csv")],
                    start_date=None,
                    end_date=None,
                    overrides=interim._validated_column_overrides({}),
                    chunksize=100,
                    pruning_reports=reports,
                )
            )
        )
        == 100
    )
    assert reports == [report]


def test_interim_releases_previous_frame_before_requesting_next_raw_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released_before_next: list[bool] = []

    def raw_chunks(*_args: object, **_kwargs: object):
        first = pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [pd.Timestamp("2024-01-01")],
            }
        )
        first_ref = weakref.ref(first)
        yield first
        del first
        gc.collect()
        released_before_next.append(first_ref() is None)
        yield pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [pd.Timestamp("2024-01-02")],
            }
        )

    report = {
        "data_id": frequency_ids.MARKET_BARS_DATA_ID,
        "report_type": "column_pruning",
    }
    monkeypatch.setattr(interim, "iter_private_export_chunks_with_policy", raw_chunks)
    monkeypatch.setattr(interim, "normalize_raw_frame", lambda frame, _data_id: frame)
    monkeypatch.setattr(
        interim,
        "_filter_normalized_frame",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(
        interim,
        "_apply_output_column_rules_validated",
        lambda frame, _data_id, **_kwargs: (frame, report),
    )
    frames = interim.iter_interim_frames(
        data_id=frequency_ids.MARKET_BARS_DATA_ID,
        input_files=[Path("private.csv")],
        start_date=None,
        end_date=None,
        chunksize=1,
    )

    first = next(frames)
    del first
    gc.collect()
    second = next(frames)
    del second
    frames.close()

    assert released_before_next == [True]


def test_csmar_source_order_validation_honors_inventory_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = {
        Path("a.csv"): pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-02")],
                "source_market_type": ["4"],
            }
        ),
        Path("b.csv"): pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2024-01-01")],
                "source_market_type": ["4"],
            }
        ),
    }
    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda path, **_kwargs: iter((frames[Path(path)].copy(),)),
    )
    monkeypatch.setattr(interim, "normalize_raw_frame", lambda frame, _data_id: frame)
    monkeypatch.setattr(
        interim, "_filter_normalized_frame", lambda frame, **_kwargs: frame
    )
    monkeypatch.setattr(
        interim,
        "_apply_output_column_rules_validated",
        lambda frame, _data_id, **_kwargs: (
            frame,
            {"data_id": "trading_calendar", "report_type": "column_pruning"},
        ),
    )

    def collect(paths: list[Path]) -> list[pd.DataFrame]:
        return list(
            interim._iter_pruned_interim_frames(
                data_id="trading_calendar",
                raw_dataset="trading_calendar",
                input_files=paths,
                start_date=None,
                end_date=None,
                overrides=interim._validated_column_overrides({}),
                chunksize=1,
                pruning_reports=[],
            )
        )

    with pytest.raises(ValueError, match="csmar_source_order_regression"):
        collect([Path("a.csv"), Path("b.csv")])
    ordered = collect([Path("b.csv"), Path("a.csv")])
    assert [frame.loc[0, "timestamp"] for frame in ordered] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]


@pytest.mark.parametrize("data_id", ("trading_calendar",))
@pytest.mark.parametrize("boundary", ("chunk", "file"))
def test_csmar_source_order_rejects_inverse_boundaries_for_every_data_id(
    monkeypatch: pytest.MonkeyPatch,
    data_id: str,
    boundary: str,
) -> None:
    primary_key = tuple(interim.interim_catalog().get(data_id).primary_key)

    def keyed(timestamp: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: [
                    pd.Timestamp(timestamp)
                    if column == "timestamp"
                    else (4 if column == "source_market_type" else "000001")
                ]
                for column in primary_key
            }
        )

    frames_by_path = (
        {Path("a.csv"): (keyed("2024-01-02"), keyed("2024-01-01"))}
        if boundary == "chunk"
        else {
            Path("a.csv"): (keyed("2024-01-02"),),
            Path("b.csv"): (keyed("2024-01-01"),),
        }
    )
    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda path, **_kwargs: iter(frames_by_path[Path(path)]),
    )
    monkeypatch.setattr(interim, "normalize_raw_frame", lambda frame, _data_id: frame)
    monkeypatch.setattr(
        interim, "_filter_normalized_frame", lambda frame, **_kwargs: frame
    )
    monkeypatch.setattr(
        interim,
        "_apply_output_column_rules_validated",
        lambda frame, _data_id, **_kwargs: (
            frame,
            {"data_id": data_id, "report_type": "column_pruning"},
        ),
    )

    with pytest.raises(ValueError, match="csmar_source_order_regression"):
        list(
            interim._iter_pruned_interim_frames(
                data_id=data_id,
                raw_dataset=data_id,
                input_files=list(frames_by_path),
                start_date=None,
                end_date=None,
                overrides=interim._validated_column_overrides({}),
                chunksize=1,
                pruning_reports=[],
            )
        )


@pytest.mark.parametrize("data_id", ("trading_calendar",))
def test_csmar_source_order_accepts_equal_and_empty_boundaries_for_every_data_id(
    monkeypatch: pytest.MonkeyPatch,
    data_id: str,
) -> None:
    primary_key = tuple(interim.interim_catalog().get(data_id).primary_key)
    row = pd.DataFrame(
        {
            column: [
                pd.Timestamp("2024-01-02")
                if column == "timestamp"
                else (4 if column == "source_market_type" else "000001")
            ]
            for column in primary_key
        }
    )
    empty = row.iloc[:0]
    frames_by_path = {
        Path("a.csv"): (row, empty),
        Path("b.csv"): (row.copy(),),
    }
    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda path, **_kwargs: iter(frames_by_path[Path(path)]),
    )
    monkeypatch.setattr(interim, "normalize_raw_frame", lambda frame, _data_id: frame)
    monkeypatch.setattr(
        interim, "_filter_normalized_frame", lambda frame, **_kwargs: frame
    )
    monkeypatch.setattr(
        interim,
        "_apply_output_column_rules_validated",
        lambda frame, _data_id, **_kwargs: (
            frame,
            {"data_id": data_id, "report_type": "column_pruning"},
        ),
    )

    result = list(
        interim._iter_pruned_interim_frames(
            data_id=data_id,
            raw_dataset=data_id,
            input_files=list(frames_by_path),
            start_date=None,
            end_date=None,
            overrides=interim._validated_column_overrides({}),
            chunksize=1,
            pruning_reports=[],
        )
    )
    assert sum(map(len, result)) == 2


def test_csmar_failed_iteration_publishes_no_pruning_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reports: list[dict[str, object]] = []

    def frames(**_kwargs):
        _kwargs["pruning_reports"].append(
            {"data_id": frequency_ids.MARKET_BARS_DATA_ID}
        )
        yield pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [pd.Timestamp("2024-01-01")],
            }
        )
        raise RuntimeError("source_primary")

    monkeypatch.setattr(interim, "_iter_pruned_interim_frames", frames)
    iterator = interim.iter_interim_frames(
        data_id=frequency_ids.MARKET_BARS_DATA_ID,
        input_files=[Path("private.csv")],
        start_date=None,
        end_date=None,
        pruning_reports=reports,
    )
    assert len(next(iterator)) == 1
    with pytest.raises(RuntimeError, match="source_primary"):
        next(iterator)
    assert reports == []


@pytest.mark.parametrize(
    "primary_type",
    (KeyboardInterrupt,),
)
def test_csmar_raw_close_failure_does_not_replace_primary(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    class PrimaryThenCloseFailure:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary_type("body_primary")

        def close(self) -> None:
            raise OSError("close_secondary")

    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda *_args, **_kwargs: PrimaryThenCloseFailure(),
    )
    output = interim._iter_pruned_interim_frames(
        data_id=frequency_ids.MARKET_BARS_DATA_ID,
        raw_dataset=frequency_ids.MARKET_BARS_DATA_ID,
        input_files=[Path("private.csv")],
        start_date=None,
        end_date=None,
        overrides=interim._validated_column_overrides({}),
        chunksize=1,
        pruning_reports=[],
    )

    with pytest.raises(primary_type, match="body_primary"):
        next(output)


@pytest.mark.parametrize(
    "primary_type",
    (KeyboardInterrupt,),
)
def test_csv_owner_open_failure_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("open_primary")
    open_calls = 0

    class FailingPath:
        def open(self, *_args, **_kwargs):
            nonlocal open_calls
            open_calls += 1
            raise primary

    monkeypatch.setattr(raw_schemas, "Path", lambda path: path)
    with pytest.raises(primary_type) as raised:
        next(
            raw_schemas.iter_export_csv_records(
                FailingPath(),  # type: ignore[arg-type]
                resource_policy=capture_test_policy(
                    available_memory_bytes=64 * 1024**2,
                    explicit_cap_bytes=32 * 1024 * 1024,
                ),
            )
        )
    assert raised.value is primary
    assert open_calls == 1


def test_csv_owner_open_primary_does_not_create_spool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    primary = KeyboardInterrupt("open_primary")
    spool_calls = 0
    open_calls = 0

    def temporary_file(*_args, **_kwargs):
        nonlocal spool_calls
        spool_calls += 1
        raise AssertionError("spool_created_after_open_failure")

    class FailingPath:
        def open(self, *_args, **_kwargs):
            nonlocal open_calls
            open_calls += 1
            raise primary

    monkeypatch.setattr(raw_schemas, "Path", lambda path: path)
    monkeypatch.setattr(raw_schemas.tempfile, "TemporaryFile", temporary_file)

    with pytest.raises(KeyboardInterrupt) as raised:
        next(
            raw_schemas.iter_export_csv_records(
                FailingPath(),  # type: ignore[arg-type]
                resource_policy=policy,
            )
        )

    assert raised.value is primary
    assert spool_calls == 0
    assert open_calls == 1


@pytest.mark.parametrize("probe", ("fileno", "fstat"))
@pytest.mark.parametrize(
    "primary_type",
    (KeyboardInterrupt,),
)
def test_csv_owner_identity_probe_failure_closes_once_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    probe: str,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("identity_probe_primary")
    close_calls = 0

    class Handle:
        def fileno(self) -> int:
            if probe == "fileno":
                raise primary
            return 7

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            raise OSError("close_secondary")

    class OpenedPath:
        def open(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return Handle()

    def fstat(_fileno: int) -> object:
        if probe == "fstat":
            raise primary
        raise AssertionError("fstat_ran_after_fileno_failure")

    monkeypatch.setattr(raw_schemas, "Path", lambda path: path)
    monkeypatch.setattr(raw_schemas.os, "fstat", fstat)

    with pytest.raises(primary_type) as raised:
        next(
            raw_schemas.iter_export_csv_records(
                OpenedPath(),  # type: ignore[arg-type]
                resource_policy=capture_test_policy(
                    available_memory_bytes=64 * 1024**2,
                    explicit_cap_bytes=32 * 1024 * 1024,
                ),
            )
        )

    assert raised.value is primary
    assert close_calls == 1


def test_csv_owner_rejects_unissued_policy_before_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls = 0

    class OpenedPath:
        def open(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal open_calls
            open_calls += 1
            pytest.fail("raw_file_opened_before_policy_validation")

    class Policy:
        payload_target_bytes = 1024 * 1024
        task_memory_budget_bytes = 1024 * 1024

    monkeypatch.setattr(raw_schemas, "Path", lambda path: path)
    with pytest.raises(TypeError, match="invalid_runtime_resource_policy"):
        raw_schemas.iter_export_csv_records(
            OpenedPath(),  # type: ignore[arg-type]
            resource_policy=Policy(),  # type: ignore[arg-type]
        )
    assert open_calls == 0


def test_csmar_total_pipeline_materializes_all_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "calendar.csv"
    row_count = 50_000
    pd.DataFrame(
        {
            "Markettype": ["4"] * row_count,
            "Clddt": pd.date_range("1900-01-01", periods=row_count).astype(str),
            "Daywk": ["1"] * row_count,
            "State": ["O"] * row_count,
        }
    ).to_csv(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    observed_rows = 0

    with runtime_resource_policy_scope(policy):
        frames = interim.iter_interim_frames(
            data_id="trading_calendar",
            input_files=[path],
            start_date=None,
            end_date=None,
            chunksize=row_count,
            resource_policy=policy,
        )
        for frame in frames:
            observed_rows += len(frame)
            del frame

    assert observed_rows == row_count


def test_csmar_direct_reference_iteration_uses_active_policy_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    observed_targets: list[int] = []

    class RecordingIndex:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def classify_comparison_table(self, table, *, payload_target_bytes, **_kwargs):
            observed_targets.append(payload_target_bytes)
            return CompactPositionState(
                array("Q", range(table.num_rows)), table.num_rows
            ), 0

    frame = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "stock_name": ["demo"],
        }
    )
    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", RecordingIndex)
    monkeypatch.setattr(
        interim,
        "_iter_pruned_interim_frames",
        lambda **_kwargs: iter((frame,)),
    )

    with runtime_resource_policy_scope(policy):
        result = list(
            interim.iter_interim_frames(
                data_id="stock_master",
                input_files=[Path("private.csv")],
                start_date=None,
                end_date=None,
                resource_policy=policy,
            )
        )

    assert len(result) == 1
    assert observed_targets == [policy.remaining_memory_bytes]


def test_csmar_materialization_forwards_captured_policy_target_to_exact_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    policy_calls: list[bool] = []
    dataset_policies: list[RuntimeResourcePolicy] = []
    observed_targets: list[int] = []
    write_policies: list[RuntimeResourcePolicy] = []

    class RecordingIndex:
        retained_memory_bytes = 1024

        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def classify_comparison_table(self, table, *, payload_target_bytes, **_kwargs):
            observed_targets.append(payload_target_bytes)
            return CompactPositionState(
                array("Q", range(table.num_rows)), table.num_rows
            ), 0

    raw_dataset = interim._raw_dataset_for_interim("stock_master")
    frame = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "stock_name": ["demo"],
        }
    )
    monkeypatch.setattr(
        interim,
        "active_runtime_resource_policy",
        lambda: policy_calls.append(True) or policy,
        raising=False,
    )
    monkeypatch.setattr(
        interim,
        "runtime_resource_policy_scope",
        runtime_resource_policy_scope,
        raising=False,
    )
    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", RecordingIndex)

    def dataset_files(_root, **kwargs):  # noqa: ANN001, ANN003
        dataset_policies.append(kwargs["resource_policy"])
        return iter(
            ((raw_dataset, Path("private.csv"), DatasetFileIdentity(1, 2, 3, 4)),)
        )

    monkeypatch.setattr(interim, "dataset_files_with_identity_by_id", dataset_files)
    monkeypatch.setattr(
        interim,
        "_iter_pruned_interim_frames",
        lambda **_kwargs: iter((frame,)),
    )

    def write_dataset(_root, frames, **_kwargs):  # noqa: ANN001, ANN202
        write_policies.append(active_runtime_resource_policy())
        list(frames)

    monkeypatch.setattr(interim, "write_dataset", write_dataset)
    layout = type(
        "Layout",
        (),
        {
            "raw_source_root": lambda *_args: Path("raw"),
            "interim_data_root": lambda *_args: Path("interim"),
        },
    )()

    interim.materialize_interim_data(
        layout=layout,
        data_ids=("stock_master",),
    )

    assert policy_calls == [True]
    assert dataset_policies == [policy]
    assert observed_targets == [policy.remaining_memory_bytes]
    assert write_policies == [policy]


def test_csmar_reference_dedup_closes_upstream_on_early_close() -> None:
    closed: list[bool] = []

    def frames():
        try:
            yield pd.DataFrame(
                {
                    "source_asset_id": ["000001"],
                    "timestamp": [pd.Timestamp("2020-01-01")],
                    "stock_name": ["demo"],
                }
            )
            yield pd.DataFrame()
        finally:
            closed.append(True)

    upstream = frames()
    output = interim._iter_deduped_reference_frames(
        upstream,
        data_id="stock_master",
        pruning_reports=None,
        scratch_parent=Path("."),
    )
    next(output)
    output.close()

    assert closed == [True]


def test_csmar_materializer_closes_prefetched_frames_when_writer_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []

    def frames(**_kwargs: object):
        try:
            yield pd.DataFrame({"source_asset_id": ["000001"]})
            yield pd.DataFrame({"source_asset_id": ["000002"]})
        finally:
            closed.append(True)

    monkeypatch.setattr(
        interim,
        "dataset_files_with_identity_by_id",
        lambda _root, **_kwargs: iter(
            (("stock_master", Path("raw"), DatasetFileIdentity(1, 2, 3, 4)),)
        ),
    )
    monkeypatch.setattr(interim, "_iter_interim_frames", frames)

    def failing_writer(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("writer_failed")

    monkeypatch.setattr(interim, "write_dataset", failing_writer)

    with pytest.raises(RuntimeError, match="writer_failed"):
        interim.materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            data_ids=("stock_master",),
        )

    assert closed == [True]


def test_csmar_materializer_requires_artifact_root_before_frame_pull(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dataset = interim._raw_dataset_for_interim("stock_master")
    monkeypatch.setattr(
        interim,
        "dataset_files_with_identity_by_id",
        lambda _root, **_kwargs: iter(
            ((raw_dataset, Path("raw"), DatasetFileIdentity(1, 2, 3, 4)),)
        ),
    )
    monkeypatch.setattr(
        interim,
        "require_artifact_root",
        lambda _root: (_ for _ in ()).throw(ValueError("invalid_dataset_part_files")),
    )
    monkeypatch.setattr(
        interim,
        "_iter_interim_frames",
        lambda **_kwargs: pytest.fail("frame_pull_before_artifact_root_gate"),
    )

    with pytest.raises(ValueError, match="^invalid_dataset_part_files$"):
        interim.materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            data_ids=("stock_master",),
        )


def test_csmar_materializer_consumes_spill_order_without_prefetch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    def indexed_files(_root: Path, **_kwargs):  # noqa: ANN003, ANN202
        for index, name in enumerate(("first", "second")):
            events.append(f"inventory:{name}")
            yield (
                "stock_master",
                Path(f"{name}.csv"),
                DatasetFileIdentity(1, index + 1, 3, 4),
            )

    def frames(*, input_files, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        for path, _identity in input_files:
            events.append(f"frame:{path.stem}")
            yield pd.DataFrame({"source_asset_id": [path.stem]})

    monkeypatch.setattr(interim, "dataset_files_with_identity_by_id", indexed_files)
    monkeypatch.setattr(interim, "_iter_interim_frames", frames)
    monkeypatch.setattr(
        interim,
        "write_dataset",
        lambda _root, data, **_kwargs: list(data),
    )

    interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("stock_master",),
    )

    assert events == [
        "inventory:first",
        "frame:first",
        "inventory:second",
        "frame:second",
    ]


def test_materializer_releases_prefetched_frame_before_next_writer_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_refs: list[weakref.ReferenceType[pd.DataFrame]] = []
    released_before_second: list[bool] = []

    def frames(**_kwargs: object):
        first = pd.DataFrame({"source_asset_id": ["000001"]})
        first_refs.append(weakref.ref(first))
        yield first
        del first
        yield pd.DataFrame({"source_asset_id": ["000002"]})

    def write_dataset(_root: Path, data, **_kwargs: object) -> None:  # noqa: ANN001
        writer_frames = iter(data)
        first = next(writer_frames)
        del first
        second = next(writer_frames)
        gc.collect()
        released_before_second.append(first_refs[0]() is None)
        del second

    monkeypatch.setattr(
        interim,
        "dataset_files_with_identity_by_id",
        lambda _root, **_kwargs: iter(
            (("stock_master", Path("raw"), DatasetFileIdentity(1, 2, 3, 4)),)
        ),
    )
    monkeypatch.setattr(interim, "_iter_interim_frames", frames)
    monkeypatch.setattr(interim, "write_dataset", write_dataset)

    interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("stock_master",),
    )

    assert released_before_second == [True]


def test_csmar_index_info_reference_interim_dedupes_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    exact_a = tmp_path / "index_info_exact_a.csv"
    exact_b = tmp_path / "index_info_exact_b.csv"
    history = tmp_path / "index_info_history.csv"
    conflict = tmp_path / "index_info_conflict.csv"
    pd.DataFrame([_index_info_raw_row("2014-01-02", "2013-12-31", "1")]).to_csv(
        exact_a,
        index=False,
    )
    pd.DataFrame([_index_info_raw_row("2014-01-02", "2013-12-31", "1")]).to_csv(
        exact_b,
        index=False,
    )
    pd.DataFrame(
        [
            _index_info_raw_row("2014-01-02", "2013-12-31", "1"),
            _index_info_raw_row("2014-02-03", "2013-12-31", "2"),
        ]
    ).to_csv(history, index=False)
    pd.DataFrame(
        [
            _index_info_raw_row("2014-01-02", "2013-12-31", "1"),
            _index_info_raw_row("2014-01-02", "2013-12-31", "2"),
        ]
    ).to_csv(conflict, index=False)

    pruning_reports: list[dict[str, object]] = []
    deduped = _collect_interim_frames(
        data_id="index_info",
        input_files=[exact_a, exact_b],
        start_date=None,
        end_date=None,
        chunksize=1,
        pruning_reports=pruning_reports,
    )

    assert len(deduped) == 1
    assert deduped["timestamp"].tolist() == [pd.Timestamp("2014-01-02")]
    assert "asset_id" not in deduped.columns
    assert [
        report["reference_exact_duplicate_count"]
        for report in pruning_reports
        if report.get("report_type") == "reference_duplicate_summary"
    ] == [1]
    historical = _collect_interim_frames(
        data_id="index_info",
        input_files=[history],
        start_date=None,
        end_date=None,
        chunksize=10,
    )
    assert len(historical) == 2
    assert historical["timestamp"].tolist() == [
        pd.Timestamp("2014-01-02"),
        pd.Timestamp("2014-02-03"),
    ]
    with pytest.raises(
        ValueError,
        match="conflicting_reference_primary_key_duplicates: index_info",
    ):
        _collect_interim_frames(
            data_id="index_info",
            input_files=[conflict],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


def test_csmar_index_info_missing_start_and_base_dates_fails_closed(
    tmp_path: Path,
) -> None:
    export = tmp_path / ("csmar__index_info__all_a_ex_bjse__full__export-20260515.csv")
    pd.DataFrame([_index_info_raw_row("", "", "1")]).to_csv(export, index=False)

    with pytest.raises(ValueError, match="missing_index_info_effective_timestamp"):
        _collect_interim_frames(
            data_id="index_info",
            input_files=[export],
            start_date=None,
            end_date=None,
            chunksize=10,
        )


def test_csmar_market_bars_daily_materializer_streams_chunks_to_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, dict[str, object]] = {}

    def fake_iter_interim_frames(**kwargs: object):
        data_id = kwargs["data_id"]
        if data_id == frequency_ids.MARKET_BARS_DATA_ID:
            yield pd.DataFrame(
                {
                    "source_asset_id": ["000001"],
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "trade_date": [pd.Timestamp("2024-01-02")],
                    "market_type": [4],
                    "trading_status_code": [1],
                    "filling_flag": ["0"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.8],
                    "close": [10.2],
                    "volume": [1000],
                    "amount": [10200.0],
                    "prev_close": [9.9],
                    "limit_status": [0],
                    "period_return": [0.03],
                    "adj_close": [10.2],
                    "adj_open": [10.0],
                    "float_market_cap": [100.0],
                    "total_market_cap": [200.0],
                    "turnover": [0.01],
                }
            )
            return
        if data_id == "market_cap":
            yield pd.DataFrame(
                {
                    "source_asset_id": ["000001"],
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "share_change_date": [pd.Timestamp("2024-01-02")],
                    "total_shares": [200.0],
                    "a_float_shares": [100.0],
                }
            )
            return
        raise AssertionError(f"unexpected data_id: {data_id}")

    def fake_write_dataset(root: Path, data, *, spec, **_kwargs) -> None:  # noqa: ANN003
        assert not isinstance(data, pd.DataFrame)
        captured[spec.data_id] = {
            "root": root,
            "chunks": list(data),
        }

    raw_index_calls = 0

    def fake_indexed_files(
        _: Path,
        *,
        resource_policy: RuntimeResourcePolicy,
    ):  # noqa: ANN202
        nonlocal raw_index_calls
        raw_index_calls += 1
        assert isinstance(resource_policy, RuntimeResourcePolicy)
        return iter(
            (
                (
                    frequency_ids.MARKET_BARS_DATA_ID,
                    Path("raw-market-bars"),
                    DatasetFileIdentity(1, 2, 3, 4),
                ),
                (
                    "market_cap",
                    Path("raw-market-cap"),
                    DatasetFileIdentity(1, 3, 3, 4),
                ),
            )
        )

    monkeypatch.setattr(
        interim, "dataset_files_with_identity_by_id", fake_indexed_files
    )
    monkeypatch.setattr(
        interim,
        "_iter_interim_frames",
        fake_iter_interim_frames,
        raising=False,
    )
    monkeypatch.setattr(interim, "write_dataset", fake_write_dataset)

    result = interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=(frequency_ids.MARKET_BARS_DATA_ID, "market_cap"),
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    assert set(captured) == {frequency_ids.MARKET_BARS_DATA_ID, "market_cap"}
    assert len(captured[frequency_ids.MARKET_BARS_DATA_ID]["chunks"]) == 1
    assert len(captured["market_cap"]["chunks"]) == 1
    assert (
        result.artifact_roots[frequency_ids.MARKET_BARS_DATA_ID]
        == captured[frequency_ids.MARKET_BARS_DATA_ID]["root"]
    )
    assert result.artifact_roots["market_cap"] == captured["market_cap"]["root"]
    assert raw_index_calls == 1


def test_csmar_interim_materializer_writes_source_path(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw" / "csmar" / "provider_exports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    export = raw_dir / "TRDNEW_Dalyr.csv"
    raw = _market_bars_daily_raw_frame()
    raw["LimitStatus"] = pd.Series([None, "0"], dtype="object")
    raw["Opnprc"] = pd.Series(["10", "20.5"], dtype="object")
    raw.to_csv(export, index=False)

    result = interim.materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=(frequency_ids.MARKET_BARS_DATA_ID,),
        start_date="2024-01-01",
        end_date="2024-12-31",
        chunksize=1,
    )

    artifact_root = result.artifact_roots[frequency_ids.MARKET_BARS_DATA_ID]
    read_back = read_dataset(artifact_root)
    inspection = inspect_dataset(artifact_root)

    assert artifact_root == frequency_ids.csmar_market_bars_interim_root(tmp_path)
    assert inspection.file_count >= 1
    assert inspection.row_count == 2
    assert len(read_back) == 2
    assert read_back.columns.tolist() == list(interim.MARKET_BARS_INTERIM_COLUMNS)
    assert read_back["limit_status"].dtype == "float64"
    assert read_back["limit_status"].isna().sum() == 1
    assert read_back["limit_status"].dropna().tolist() == [0.0]
    assert read_back["open"].dtype == "float64"
    assert read_back["open"].tolist() == [10.0, 20.5]
    assert read_back["turnover"].dtype == "float64"
    assert read_back["turnover"].isna().all()
    assert "asset_id" not in read_back.columns
    assert "market_id" not in read_back.columns
    parts = sorted(artifact_root.rglob("*.parquet"))
    assert parts
    assert all(
        pa.types.is_float64(pq.ParquetFile(part).schema_arrow.field("turnover").type)
        for part in parts
    )
    assert all(
        pa.types.is_float64(
            pq.ParquetFile(part).schema_arrow.field("limit_status").type
        )
        for part in parts
    )


def _market_bars_daily_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Stkcd": "1",
                "Trddt": "2024-01-02",
                "Markettype": 4,
                "Trdsta": 1,
                "Filling": "0",
                "Opnprc": 10.0,
                "Hiprc": 10.5,
                "Loprc": 9.8,
                "Clsprc": 10.2,
                "Dnshrtrd": 1000,
                "Dnvaltrd": 10200.0,
                "Dsmvosd": 100000.0,
                "Dsmvtll": 200000.0,
                "Dretwd": 0.03,
                "Dretnd": 0.028,
                "Adjprcwd": 10.2,
                "Adjprcnd": 10.1,
                "PreClosePrice": 9.9,
                "ChangeRatio": 0.0303,
                "LimitDown": 8.91,
                "LimitUp": 10.89,
                "LimitStatus": 0,
            },
            {
                "Stkcd": "2",
                "Trddt": "2024-01-02",
                "Markettype": 64,
                "Trdsta": 1,
                "Filling": "0",
                "Opnprc": 20.0,
                "Hiprc": 20.5,
                "Loprc": 19.8,
                "Clsprc": 20.2,
                "Dnshrtrd": 2000,
                "Dnvaltrd": 40400.0,
                "Dsmvosd": 200000.0,
                "Dsmvtll": 400000.0,
                "Dretwd": 0.01,
                "Dretnd": 0.008,
                "Adjprcwd": 20.2,
                "Adjprcnd": 20.1,
                "PreClosePrice": 20.0,
                "ChangeRatio": 0.01,
                "LimitDown": 18.0,
                "LimitUp": 22.0,
                "LimitStatus": 0,
            },
        ]
    )


def _write_trading_calendar_raw_export(path: Path) -> None:
    pd.DataFrame(
        [
            {"Markettype": 1, "Clddt": "1990-12-19", "Daywk": 3, "State": "O"},
            {"Markettype": 4, "Clddt": "2024-01-06", "Daywk": 6, "State": "O"},
            {"Markettype": 64, "Clddt": "2026-12-31", "Daywk": 4, "State": "C"},
        ]
    ).to_csv(path, index=False)


def _stock_master_raw_row(
    company_status: str,
    industry_code: str,
    status_change_date: str,
) -> dict[str, object]:
    return {
        "Stkcd": "1",
        "Stknme": "demo",
        "Listdt": "2010-01-01",
        "DelistedDate": "",
        "Markettype": 4,
        "Nnindcd": industry_code,
        "Nnindnme": f"industry {industry_code}",
        "Statco": company_status,
        "Statdt": status_change_date,
    }


def _index_info_raw_row(
    start_date: str,
    base_date: str,
    index_class: str,
) -> dict[str, object]:
    return {
        "Indexcd": "000985",
        "Idxinfo01": "benchmark",
        "Idxinfo11": start_date,
        "Idxinfo02": base_date,
        "Idxinfo07": index_class,
        "Idxinfo10": "formula",
    }


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_csmar_reference_frames_close_under_policy_and_preserve_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    primary = primary_type("reference_primary")
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Frames:
        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            raise primary

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("close_secondary")

    class Index:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Index:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    monkeypatch.setattr(
        interim,
        "interim_catalog",
        lambda: SimpleNamespace(
            get=lambda _data_id: SimpleNamespace(primary_key=("id",))
        ),
    )
    monkeypatch.setattr(interim, "ExactPrimaryKeyIndex", Index)

    with runtime_resource_policy_scope(policy):
        chunks = interim._iter_deduped_reference_frames(
            Frames(),
            data_id="reference",
            pruning_reports=None,
            scratch_parent=Path("."),
        )
        with pytest.raises(primary_type) as raised:
            next(chunks)

    assert raised.value is primary
    assert closed_under == [policy]


def test_private_export_first_frame_reserves_owner_known_decode_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    column_count = 4096

    def records(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        yield (
            [f"column_{index}" for index in range(column_count)],
            ["1"] * column_count,
            policy.task_memory_budget_bytes - 1,
        )

    monkeypatch.setattr(raw_exports, "_iter_csv_records", records)
    monkeypatch.setattr(
        raw_exports,
        "_raw_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("DataFrame allocated before decode-buffer admission")
        ),
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(
            raw_exports.iter_private_export_chunks_with_policy(
                Path("unused.csv"),
                dataset="trading_calendar",
                chunksize=1,
                resource_policy=policy,
            )
        )


@pytest.mark.parametrize("dataset", ("trading_calendar", "index_info"))
def test_private_export_retains_pending_decoded_owner_state_before_next_pull(
    dataset: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    if dataset == "trading_calendar":
        header = ["Markettype", "Clddt", "Daywk", "State"]
        rows = (
            ["4", "2014-01-01", "3", "O"],
            ["4", "2014-01-02", "4", "O"],
            ["4", "2014-01-03", "5", "O"],
        )
    else:
        header = ["Indexcd", "Idxinfo11", "Idxinfo02"]
        rows = (
            ["000985", "2014-01-01", "2013-12-31"],
            ["000986", "2014-01-02", "2013-12-31"],
            ["000987", "2014-01-03", "2013-12-31"],
        )
    row_bytes = 4
    observed: list[int] = []

    def records(*_args, retained_payload_bytes, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        for row in rows:
            observed.append(retained_payload_bytes[0])
            yield header, row, row_bytes

    owner = "iter_index_info_rows" if dataset == "index_info" else "_iter_csv_records"
    monkeypatch.setattr(raw_exports, owner, records)
    with runtime_resource_policy_scope(policy):
        chunks = iter(
            raw_exports.iter_private_export_chunks_with_policy(
                Path("unused.csv"),
                dataset=dataset,
                chunksize=100,
                resource_policy=policy,
            )
        )

        first = next(chunks)
        second = next(chunks)
        chunks.close()

    pointer_bytes = struct.calcsize("P")
    assert observed[2] - observed[1] == row_bytes + (1 + len(header)) * pointer_bytes
    assert len(first) == 1
    assert len(second) == 2


@pytest.mark.parametrize("require_exact_width", (True, False))
def test_csv_reader_counts_copied_fast_header_before_data_pull(
    tmp_path: Path,
    require_exact_width: bool,
) -> None:
    field_span = 425_000
    path = tmp_path / "wide-fast.csv"
    path.write_text(
        f"{'h' * field_span},{'i' * field_span}\n"
        f"{'x' * field_span},{'y' * field_span}\n",
        encoding="utf-8",
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    rows = raw_exports._iter_csv_records(
        path,
        resource_policy=policy,
        require_exact_width=require_exact_width,
    )

    with pytest.raises(
        storage_errors.DatasetReadCapacityExceeded,
        match="csmar_raw_working_set_exceeds_task_memory_budget",
    ):
        next(rows)


def test_csv_reader_avoids_fast_record_duplicate_before_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fast.csv"
    path.write_text("left,right\na,b\nc,d\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    references: list[weakref.ReferenceType[raw_schemas.BoundedCsvRecord]] = []
    original_materialize = raw_schemas.BoundedCsvRecord.materialize

    def materialize(record, **kwargs):  # noqa: ANN001, ANN003, ANN202
        references.append(weakref.ref(record))
        return original_materialize(record, **kwargs)

    monkeypatch.setattr(raw_schemas.BoundedCsvRecord, "materialize", materialize)
    rows = raw_exports._iter_csv_records(
        path,
        resource_policy=policy,
        require_exact_width=True,
    )

    _header, row, _source_bytes = next(rows)
    gc.collect()

    assert row == ["a", "b"]
    assert len(references) == 1
    assert references[0]() is None
    rows.close()


@pytest.mark.parametrize("existing_continuation", (False, True))
def test_index_continuation_unicode_write_admits_transient_before_append(
    tmp_path: Path,
    existing_continuation: bool,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    with raw_exports._IndexInfoContinuationSpool(
        record_limit_bytes=1024**2,
        scratch_parent=tmp_path,
    ) as spool:
        if existing_continuation:
            spool.append_fields(
                ("existing",),
                field_count=1,
                retained_payload_bytes=0,
                source_bytes=8,
                resource_policy=policy,
            )
        position = spool._handle.tell()
        with pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="csmar_raw_working_set_exceeds_task_memory_budget",
        ):
            spool.append_fields(
                ("界" * io.DEFAULT_BUFFER_SIZE,),
                field_count=1,
                retained_payload_bytes=policy.task_memory_budget_bytes - 1,
                source_bytes=1,
                resource_policy=policy,
            )
        assert spool._handle.tell() == position


@pytest.mark.parametrize(
    "entrypoint",
    (raw_exports.iter_raw_export_files, raw_exports.dataset_files_with_identity_by_id),
)
def test_csmar_raw_inventory_rejects_symlink_in_absolute_root_ancestor_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
) -> None:  # noqa: ANN001
    real_parent = tmp_path / "real-parent"
    raw_root = real_parent / "raw"
    raw_root.mkdir(parents=True)
    _write_trading_calendar_raw_export(raw_root / "TRDNEW_Cale.csv")
    alias_parent = tmp_path / "alias-parent"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")
    monkeypatch.setattr(
        raw_exports,
        "inspect_raw_export",
        lambda *_args, **_kwargs: pytest.fail("inspection_before_root_alias_gate"),
    )
    monkeypatch.setattr(
        raw_exports,
        "ExactPrimaryKeyIndex",
        lambda **_kwargs: pytest.fail("index_before_root_alias_gate"),
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(entrypoint(alias_parent / "raw"))


def test_csmar_raw_inventory_gates_walk_parent_before_candidate_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    parent = raw_root / "nested"
    candidate = parent / "TRDNEW_Cale.csv"

    def walk(_self: Path, **_kwargs):  # noqa: ANN202
        yield parent, [], [candidate.name]

    def require_path(path: Path, *, error: str) -> Path:
        assert error == "csmar_raw_export_identity_changed"
        if Path(path) == parent:
            raise ValueError(error)
        return Path(path)

    monkeypatch.setattr(Path, "walk", walk)
    monkeypatch.setattr(raw_exports, "require_unaliased_path", require_path)
    monkeypatch.setattr(
        raw_exports,
        "path_is_filesystem_alias",
        lambda _path: pytest.fail("candidate_probe_before_parent_gate"),
        raising=False,
    )

    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        with raw_exports._raw_export_path_session(raw_root) as (paths, _verify):
            list(paths)


def test_csmar_raw_inventory_gates_child_directory_before_walk_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    child = raw_root / "nested"
    gated: list[Path] = []

    def walk(_self: Path, **_kwargs):  # noqa: ANN202
        yield raw_root, [child.name], []
        assert child in gated, "walk_resumed_before_child_directory_gate"
        yield child, [], []

    def require_path(path: Path, *, error: str) -> Path:
        assert error == "csmar_raw_export_identity_changed"
        candidate = Path(path)
        gated.append(candidate)
        return candidate

    monkeypatch.setattr(Path, "walk", walk)
    monkeypatch.setattr(raw_exports, "require_unaliased_path", require_path)

    with raw_exports._raw_export_path_session(raw_root) as (paths, _verify):
        assert list(paths) == []


def test_csmar_raw_inventory_gates_supported_file_before_file_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    candidate = raw_root / "TRDNEW_Cale.csv"

    def walk(_self: Path, **_kwargs):  # noqa: ANN202
        yield raw_root, [], [candidate.name]

    def require_path(path: Path, *, error: str) -> Path:
        assert error == "csmar_raw_export_identity_changed"
        if Path(path) == candidate:
            raise ValueError(error)
        return Path(path)

    monkeypatch.setattr(Path, "walk", walk)
    monkeypatch.setattr(raw_exports, "require_unaliased_path", require_path)
    monkeypatch.setattr(
        raw_exports,
        "path_is_filesystem_alias",
        lambda _path: pytest.fail("filesystem_alias_probe_before_candidate_gate"),
        raising=False,
    )

    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        with raw_exports._raw_export_path_session(raw_root) as (paths, _verify):
            list(paths)


@pytest.mark.parametrize(
    "entrypoint",
    (raw_exports.iter_raw_export_files, raw_exports.dataset_files_with_identity_by_id),
)
def test_csmar_raw_inventory_rejects_duplicate_hardlink_identity_before_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint,
) -> None:  # noqa: ANN001
    first = tmp_path / "TRDNEW_Cale.csv"
    second = tmp_path / "TRDNEW_Cale-copy.csv"
    _write_trading_calendar_raw_export(first)
    try:
        os.link(first, second)
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")
    monkeypatch.setattr(
        raw_exports,
        "inspect_raw_export",
        lambda *_args, **_kwargs: pytest.fail(
            "inspection_before_physical_identity_gate"
        ),
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        list(entrypoint(tmp_path))


def test_raw_export_tree_binds_inventory_identity_through_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    replacement = tmp_path.parent / f"{tmp_path.name}-replacement.csv"
    _write_trading_calendar_raw_export(export)
    _write_trading_calendar_raw_export(replacement)
    original_inspect = raw_exports.inspect_raw_export_file
    replaced = False

    def replace_before_inspect(path, **kwargs):  # noqa: ANN001, ANN003, ANN202
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, path)
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(raw_exports, "inspect_raw_export_file", replace_before_inspect)

    result = list(raw_exports.inspect_raw_export_tree(tmp_path))

    assert len(result) == 1
    assert result[0].inspection is None
    assert result[0].error == "csmar_raw_export_identity_changed"


def test_raw_export_inspection_rejects_aliased_metadata_sidecar(
    tmp_path: Path,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    try:
        export.with_suffix(".metadata.json").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        raw_schemas.inspect_raw_export_file(export)


@pytest.mark.parametrize("sidecar_kind", ("directory", "hardlink"))
def test_raw_export_inspection_rejects_non_unique_regular_metadata_sidecar(
    tmp_path: Path,
    sidecar_kind: str,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    sidecar = export.with_suffix(".metadata.json")
    if sidecar_kind == "directory":
        sidecar.mkdir()
    else:
        external = tmp_path / "external.json"
        external.write_text("{}\n", encoding="utf-8")
        try:
            os.link(external, sidecar)
        except OSError as exc:
            pytest.skip(f"hardlink_not_available: {exc}")

    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        raw_schemas.inspect_raw_export_file(export)


def test_direct_raw_export_inspection_rejects_post_header_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    replacement = tmp_path / "replacement.csv"
    _write_trading_calendar_raw_export(replacement)
    read_header = raw_schemas.read_export_header

    def replace_after_header(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        header = read_header(*args, **kwargs)
        export.unlink()
        replacement.rename(export)
        return header

    monkeypatch.setattr(raw_schemas, "read_export_header", replace_after_header)

    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        raw_schemas.inspect_raw_export_file(export)


def test_raw_export_inspection_rechecks_metadata_sidecar_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    export = tmp_path / "TRDNEW_Cale.csv"
    _write_trading_calendar_raw_export(export)
    sidecar = export.with_suffix(".metadata.json")
    sidecar.write_text("{}\n", encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"changed":true}\n', encoding="utf-8")
    identify = raw_schemas.dataset_file_identity

    def replace_after_capture(path: Path):
        identity = identify(path)
        if Path(path) == sidecar and replacement.exists():
            sidecar.unlink()
            replacement.rename(sidecar)
        return identity

    monkeypatch.setattr(raw_schemas, "dataset_file_identity", replace_after_capture)

    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        raw_schemas.inspect_raw_export_file(export)


def test_raw_export_tree_releases_inventory_index_on_early_close(
    tmp_path: Path,
) -> None:
    _write_trading_calendar_raw_export(tmp_path / "TRDNEW_Cale.csv")
    inspections = raw_exports.inspect_raw_export_tree(tmp_path)

    next(inspections)
    inspections.close()

    assert not list(tmp_path.parent.glob(".k-*"))


@pytest.mark.parametrize("error_type", (KeyboardInterrupt,))
def test_raw_export_tree_releases_inventory_index_on_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    _write_trading_calendar_raw_export(tmp_path / "TRDNEW_Cale.csv")
    monkeypatch.setattr(
        raw_exports,
        "_inspect_raw_export_with_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type()),
    )

    with pytest.raises(error_type):
        list(raw_exports.inspect_raw_export_tree(tmp_path))

    assert not list(tmp_path.parent.glob(".k-*"))


def test_interim_transform_variable_width_preflight_precedes_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    raw = pd.DataFrame(
        {"Markettype": ["4"], "Clddt": ["2024-01-02"], "Daywk": ["2"], "State": ["O"]}
    )
    monkeypatch.setattr(
        interim,
        "pandas_frame_payload_bytes",
        lambda _frame: policy.task_memory_budget_bytes - 64,
    )
    monkeypatch.setattr(
        interim,
        "normalize_raw_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("normalize ran before variable-width admission")
        ),
    )

    with (
        runtime_resource_policy_scope(policy),
        pytest.raises(
            storage_errors.DatasetReadCapacityExceeded,
            match="csmar_interim_working_set_exceeds_task_memory_budget",
        ),
    ):
        interim._transform_interim_chunk(
            raw,
            data_id="trading_calendar",
            start_date=None,
            end_date=None,
            overrides={},
            retained_previous_bytes=0,
        )


def test_non_reference_publication_keeps_upstream_alive_until_next_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = pd.DataFrame({"raw": ["owner"]})
    pruned = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")],
            "source_market_type": [4],
        }
    )
    raw_bytes = 400_000
    current_bytes = 500_000
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    raw_reference = weakref.ref(raw)

    monkeypatch.setattr(
        interim,
        "iter_private_export_chunks_with_policy",
        lambda *_args, **_kwargs: iter((raw,)),
    )
    monkeypatch.setattr(
        interim,
        "_transform_interim_chunk",
        lambda *_args, **_kwargs: (pruned, {}),
    )
    monkeypatch.setattr(
        interim,
        "pandas_frame_payload_bytes",
        lambda frame: raw_bytes if frame is raw else current_bytes,
    )
    with runtime_resource_policy_scope(policy):
        publications = interim._iter_pruned_interim_frames(
            data_id="trading_calendar",
            raw_dataset="trading_calendar",
            input_files=(Path("unused.csv"),),
            start_date=None,
            end_date=None,
            overrides={},
            chunksize=1,
            pruning_reports=None,
        )
        publication = next(publications)

    assert publication is pruned
    assert raw_reference() is not None


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_csmar_index_info_handle_close_preserves_active_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = primary_type("csv_primary")

    class Handle:
        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def read(self, _limit: int = -1) -> bytes:
            raise primary

        def close(self) -> None:
            raise OSError("handle_close_secondary")

    monkeypatch.setattr(Path, "open", lambda *_a, **_k: Handle())

    rows = iter(raw_exports.iter_index_info_rows(Path("index.csv")))
    with pytest.raises(primary_type) as raised:
        next(rows)

    assert raised.value is primary


def test_index_info_rows_bind_creation_policy_before_delayed_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    creation_policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    consumer_policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    actions: list[str] = []

    def records(*_args: object, **_kwargs: object):
        actions.append("read")
        return iter(())

    monkeypatch.setattr(raw_exports, "_iter_csv_records", records)

    with runtime_resource_policy_scope(creation_policy):
        rows = iter(
            raw_exports.iter_index_info_rows(
                Path("index.csv"),
                resource_policy=creation_policy,
            )
        )
    with runtime_resource_policy_scope(consumer_policy):
        with pytest.raises(RuntimeError, match="runtime_resource_policy_conflict"):
            next(rows)

    assert actions == []


def test_index_info_rows_scope_covers_spool_read_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    primary = RuntimeError("row_primary")
    observed: list[tuple[str, RuntimeResourcePolicy | None]] = []

    class Spool:
        def __init__(self, **_kwargs: object) -> None:
            observed.append(("spool_init", current_runtime_resource_policy()))

        def __enter__(self) -> Spool:
            observed.append(("spool_enter", current_runtime_resource_policy()))
            return self

        def __exit__(self, *_args: object) -> None:
            observed.append(("spool_exit", current_runtime_resource_policy()))

    def records(*_args: object, **_kwargs: object):
        try:
            observed.append(("read", current_runtime_resource_policy()))
            raise primary
            yield  # pragma: no cover
        finally:
            observed.append(("reader_close", current_runtime_resource_policy()))

    monkeypatch.setattr(raw_exports, "_IndexInfoContinuationSpool", Spool)
    monkeypatch.setattr(raw_exports, "_iter_csv_records", records)

    rows = iter(
        raw_exports.iter_index_info_rows(
            Path("index.csv"),
            resource_policy=policy,
        )
    )
    with pytest.raises(RuntimeError) as raised:
        next(rows)

    assert raised.value is primary
    assert observed == [
        ("spool_init", policy),
        ("spool_enter", policy),
        ("read", policy),
        ("reader_close", policy),
        ("spool_exit", policy),
    ]


@pytest.mark.parametrize(
    "primary_type",
    (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit, GeneratorExit),
)
def test_csmar_text_reader_closes_under_policy_and_preserves_primary(
    primary_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    primary = primary_type("reader_primary")
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Handle:
        position = 0

        def read(self, _limit: int = -1) -> bytes:
            if self.position == 0:
                self.position += 1
                return b"value\n"
            raise primary

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("reader_close_secondary")

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())

    with runtime_resource_policy_scope(policy):
        records = raw_exports._iter_csv_records(
            Path("export.csv"),
            resource_policy=policy,
            require_exact_width=True,
        )
        with pytest.raises(primary_type) as raised:
            next(records)

    assert raised.value is primary
    assert closed_under == [policy]


def test_csmar_text_reader_close_preserves_consumer_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Handle:
        rows = iter((b"value\n", b"y\n"))

        def read(self, _limit: int = -1) -> bytes:
            return next(self.rows)

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("reader_close_secondary")

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Handle())

    with runtime_resource_policy_scope(policy):
        records = raw_exports._iter_csv_records(
            Path("export.csv"),
            resource_policy=policy,
            require_exact_width=True,
        )
        assert next(records) == (["value"], ["y"], len(b"y\n"))
        records.close()

    assert closed_under == [policy]


def test_csmar_nonreference_frames_close_preserves_consumer_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Frames:
        yielded = False

        def __iter__(self) -> Frames:
            return self

        def __next__(self) -> pd.DataFrame:
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return pd.DataFrame({"value": ["x"]})

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("frames_close_secondary")

    monkeypatch.setattr(interim, "_raw_dataset_for_interim", lambda _data_id: "raw")
    monkeypatch.setattr(
        interim,
        "_iter_pruned_interim_frames",
        lambda **_kwargs: Frames(),
    )

    with runtime_resource_policy_scope(policy):
        chunks = interim.iter_interim_frames(
            data_id="nonreference",
            input_files=[],
            start_date=None,
            end_date=None,
            resource_policy=policy,
        )
        next(chunks)
        chunks.close()

    assert closed_under == [policy]


def test_csmar_index_info_iterator_close_preserves_consumer_generator_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    closed_under: list[RuntimeResourcePolicy | None] = []

    class Rows:
        position = 0

        def __iter__(self) -> Rows:
            return self

        def __next__(self) -> tuple[list[str], list[str], int]:
            if self.position == 2:
                raise StopIteration
            self.position += 1
            header = ["Indexcd", "Idxinfo11", "Idxinfo02"]
            row = [
                f"00098{self.position}",
                f"2014-01-0{self.position}",
                "2013-12-31",
            ]
            return header, row, sum(len(value.encode("utf-8")) for value in row)

        def close(self) -> None:
            closed_under.append(current_runtime_resource_policy())
            raise OSError("rows_close_secondary")

    monkeypatch.setattr(
        raw_exports, "iter_index_info_rows", lambda _path, **_kwargs: Rows()
    )

    with runtime_resource_policy_scope(policy):
        chunks = iter(
            raw_exports.iter_private_export_chunks(
                Path("index.csv"),
                dataset="index_info",
                chunksize=1,
                resource_policy=policy,
            )
        )
        next(chunks)
        chunks.close()

    assert closed_under == [policy]


def test_csmar_csv_reader_rejects_hardlink_before_read(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    alias = tmp_path / "alias.csv"
    source.write_bytes(b"value\n1\n")
    try:
        os.link(source, alias)
    except OSError as exc:
        pytest.skip(f"hardlink_not_available: {exc}")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )

    rows = raw_schemas.iter_export_csv_records(alias, resource_policy=policy)
    with pytest.raises(ValueError, match="^csmar_raw_export_identity_changed$"):
        next(rows)


def test_csmar_direct_csv_reader_activates_explicit_policy_before_identity_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    observed: list[RuntimeResourcePolicy | None] = []

    def identity(_path: Path) -> DatasetFileIdentity:
        observed.append(current_runtime_resource_policy())
        raise ValueError("identity_probe_stop")

    monkeypatch.setattr(raw_schemas, "dataset_file_identity", identity)
    rows = raw_schemas.iter_export_csv_records(
        Path("export.csv"),
        resource_policy=policy,
    )

    with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
        next(rows)

    assert observed == [policy]


@pytest.mark.parametrize(
    "factory",
    (raw_exports.iter_raw_export_files,),
)
def test_csmar_lazy_inventory_policy_is_resolved_only_at_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory,
) -> None:  # noqa: ANN001
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    bind_calls: list[RuntimeResourcePolicy | None] = []
    activation_calls: list[str] = []
    gate_hits: list[RuntimeResourcePolicy | None] = []
    real_bind = raw_exports.bind_runtime_resource_policy

    def bind(
        value: RuntimeResourcePolicy | None,
        **kwargs,
    ) -> RuntimeResourcePolicy | None:  # noqa: ANN003
        bind_calls.append(value)
        return real_bind(value, **kwargs)

    def activate() -> RuntimeResourcePolicy:
        activation_calls.append("active")
        return policy

    def stop_at_first_filesystem_gate(path: Path, *, error: str) -> Path:
        assert error == "csmar_raw_export_identity_changed"
        gate_hits.append(current_runtime_resource_policy())
        raise RuntimeError("activation_gate_hit")

    monkeypatch.setattr(
        raw_exports, "require_unaliased_path", stop_at_first_filesystem_gate
    )
    monkeypatch.setattr(raw_exports, "bind_runtime_resource_policy", bind)
    monkeypatch.setattr(raw_exports, "active_runtime_resource_policy", activate)

    lazy = factory(tmp_path)
    assert bind_calls == [None]
    assert activation_calls == []
    assert gate_hits == []
    with pytest.raises(RuntimeError, match="^activation_gate_hit$"):
        next(lazy)

    assert activation_calls == ["active"]
    assert gate_hits == [policy]
    assert lazy.gi_frame is None


@pytest.mark.parametrize(
    "factory",
    (raw_exports.iter_raw_export_files,),
)
def test_csmar_lazy_inventory_abandon_before_activation_has_no_active_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory,
) -> None:  # noqa: ANN001
    bind_calls: list[RuntimeResourcePolicy | None] = []
    activation_calls: list[str] = []
    real_bind = raw_exports.bind_runtime_resource_policy

    def bind(
        value: RuntimeResourcePolicy | None,
        **kwargs,
    ) -> RuntimeResourcePolicy | None:  # noqa: ANN003
        bind_calls.append(value)
        return real_bind(value, **kwargs)

    monkeypatch.setattr(raw_exports, "bind_runtime_resource_policy", bind)
    monkeypatch.setattr(
        raw_exports,
        "active_runtime_resource_policy",
        lambda: (
            activation_calls.append("active") or pytest.fail("activation_after_abandon")
        ),
    )

    lazy = factory(tmp_path)
    lazy.close()

    assert bind_calls == [None]
    assert activation_calls == []
    assert lazy.gi_frame is None


@pytest.mark.parametrize("dataset", ("unknown_dataset",))
def test_csmar_chunk_reader_rejects_unknown_dataset_before_reader(
    monkeypatch: pytest.MonkeyPatch,
    dataset: object,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    reader_hits: list[Path] = []

    def unexpected_reader(path: Path, **_kwargs):  # noqa: ANN202
        reader_hits.append(Path(path))
        return iter(())

    monkeypatch.setattr(raw_exports, "_iter_csv_records", unexpected_reader)
    chunks = raw_exports.iter_private_export_chunks_with_policy(
        Path("unknown.csv"),
        dataset=dataset,  # type: ignore[arg-type]
        chunksize=1,
        resource_policy=policy,
    )

    with pytest.raises(ValueError, match="^unknown_csmar_dataset$"):
        next(chunks)

    assert reader_hits == []


_PHASE2_RAW_ROOT_FACTORIES = ("iter_raw_export_files",)


def _phase2_write_raw_exports(root: Path, count: int) -> tuple[Path, ...]:
    root.mkdir()
    paths = tuple(root / f"export-{index}.csv" for index in range(count))
    for path in paths:
        path.write_text("value\n1\n", encoding="utf-8")
    return paths


def _phase2_stub_raw_inspection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        raw_exports,
        "_inspect_raw_export_with_identity",
        lambda path, **_kwargs: raw_exports.RawExportIntakeResult(
            path=Path(path),
            inspection=SimpleNamespace(
                dataset="trading_calendar",
                missing_required_columns=(),
            ),
        ),
    )


def _phase2_replace_raw_root_preserving_files(root: Path) -> Path:
    displaced = root.with_name(f"{root.name}-displaced")
    root.rename(displaced)
    root.mkdir()
    for path in tuple(displaced.iterdir()):
        path.rename(root / path.name)
    return displaced


def _phase2_error_text(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}:{error}"


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_replacement_after_initial_gate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    real_gate = raw_exports.require_unaliased_path
    gate_hits = 0

    def replace_after_gate(path: Path, *, error: str) -> Path:
        nonlocal gate_hits
        result = real_gate(path, error=error)
        if Path(path) == raw_root and gate_hits == 0:
            gate_hits += 1
            _phase2_replace_raw_root_preserving_files(raw_root)
        return result

    monkeypatch.setattr(raw_exports, "require_unaliased_path", replace_after_gate)
    caught: BaseException | None = None
    results: list[object] = []
    try:
        results = list(getattr(raw_exports, factory_name)(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "gate_hits": gate_hits,
        "result_count": len(results),
        "error": _phase2_error_text(caught),
    }
    expected = {
        "gate_hits": 1,
        "result_count": 0,
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_replacement_during_walk_yield_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    paths = _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    walk_hits = 0

    def replacing_walk(self: Path, **_kwargs):  # noqa: ANN202
        nonlocal walk_hits
        assert self == raw_root
        walk_hits += 1
        _phase2_replace_raw_root_preserving_files(raw_root)
        yield raw_root, [], [paths[0].name]

    monkeypatch.setattr(Path, "walk", replacing_walk)
    caught: BaseException | None = None
    results: list[object] = []
    try:
        results = list(getattr(raw_exports, factory_name)(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "walk_hits": walk_hits,
        "result_count": len(results),
        "error": _phase2_error_text(caught),
    }
    expected = {
        "walk_hits": 1,
        "result_count": 0,
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_replacement_during_indexed_replay_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    real_values = raw_exports.ExactPrimaryKeyIndex.identity_payload_values
    replay_hits = 0

    def replacing_values(self, *, namespace: str):  # noqa: ANN001, ANN202
        nonlocal replay_hits
        if replay_hits == 0 and namespace.endswith("-paths"):
            replay_hits += 1
            _phase2_replace_raw_root_preserving_files(raw_root)
        yield from real_values(self, namespace=namespace)

    monkeypatch.setattr(
        raw_exports.ExactPrimaryKeyIndex,
        "identity_payload_values",
        replacing_values,
    )
    caught: BaseException | None = None
    results: list[object] = []
    try:
        results = list(getattr(raw_exports, factory_name)(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "replay_hits": replay_hits,
        "result_count": len(results),
        "error": _phase2_error_text(caught),
    }
    expected = {
        "replay_hits": 1,
        "result_count": 0,
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_replacement_between_public_yields_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 2)
    _phase2_stub_raw_inspection(monkeypatch)
    iterator = getattr(raw_exports, factory_name)(raw_root)
    first = next(iterator)
    _phase2_replace_raw_root_preserving_files(raw_root)
    caught: BaseException | None = None
    second: object | None = None
    try:
        second = next(iterator)
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc
    finally:
        iterator.close()

    actual = {
        "first_returned": first is not None,
        "second_returned": second is not None,
        "error": _phase2_error_text(caught),
    }
    expected = {
        "first_returned": True,
        "second_returned": False,
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_replacement_before_normal_exhaustion_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    iterator = getattr(raw_exports, factory_name)(raw_root)
    first = next(iterator)
    _phase2_replace_raw_root_preserving_files(raw_root)
    caught: BaseException | None = None
    try:
        next(iterator)
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "first_returned": first is not None,
        "error": _phase2_error_text(caught),
    }
    expected = {
        "first_returned": True,
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
def test_raw_root_close_runs_final_verify_without_replacing_generator_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    root_identity = raw_root.lstat()
    real_samestat = os.path.samestat
    root_compare_hits = 0

    def tracked_samestat(left: os.stat_result, right: os.stat_result) -> bool:
        nonlocal root_compare_hits
        if real_samestat(left, root_identity) or real_samestat(right, root_identity):
            root_compare_hits += 1
        return real_samestat(left, right)

    monkeypatch.setattr(os.path, "samestat", tracked_samestat)
    iterator = getattr(raw_exports, factory_name)(raw_root)
    next(iterator)
    _phase2_replace_raw_root_preserving_files(raw_root)
    before_close = root_compare_hits
    close_error: BaseException | None = None
    try:
        iterator.close()
    except BaseException as exc:  # pragma: no branch - asserted below
        close_error = exc

    actual = {
        "final_verify_hit": root_compare_hits > before_close,
        "close_error": _phase2_error_text(close_error),
    }
    expected = {"final_verify_hit": True, "close_error": None}
    assert actual == expected


@pytest.mark.parametrize("factory_name", _PHASE2_RAW_ROOT_FACTORIES)
@pytest.mark.parametrize("primary_type", (KeyboardInterrupt,))
def test_raw_root_final_verify_preserves_active_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    primary_type: type[BaseException],
) -> None:
    raw_root = tmp_path / "raw"
    paths = _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    root_identity = raw_root.lstat()
    real_samestat = os.path.samestat
    root_compare_hits = 0
    before_primary = 0
    primary = primary_type("walk_primary")

    def tracked_samestat(left: os.stat_result, right: os.stat_result) -> bool:
        nonlocal root_compare_hits
        if real_samestat(left, root_identity) or real_samestat(right, root_identity):
            root_compare_hits += 1
        return real_samestat(left, right)

    def failing_walk(self: Path, **_kwargs):  # noqa: ANN202
        nonlocal before_primary
        assert self == raw_root
        yield raw_root, [], [paths[0].name]
        before_primary = root_compare_hits
        _phase2_replace_raw_root_preserving_files(raw_root)
        raise primary

    monkeypatch.setattr(os.path, "samestat", tracked_samestat)
    monkeypatch.setattr(Path, "walk", failing_walk)
    caught: BaseException | None = None
    try:
        list(getattr(raw_exports, factory_name)(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "primary_preserved": caught is primary,
        "final_verify_hit": root_compare_hits > before_primary,
    }
    expected = {"primary_preserved": True, "final_verify_hit": True}
    assert actual == expected


def test_interim_root_replacement_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _write_trading_calendar_raw_export(raw_root / "TRDNEW_Cale.csv")
    publication_hits: list[int] = []
    replacement_hits = 0

    def frames(*, input_files, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        nonlocal replacement_hits
        inputs = iter(input_files)
        next(inputs)
        replacement_hits += 1
        _phase2_replace_raw_root_preserving_files(raw_root)
        yield pd.DataFrame({"value": [1]})
        for _item in inputs:
            pass

    def publish(_root: Path, data, **_kwargs) -> None:  # noqa: ANN001, ANN202
        materialized = list(data)
        publication_hits.append(len(materialized))

    monkeypatch.setattr(interim, "_iter_interim_frames", frames)
    monkeypatch.setattr(interim, "write_dataset", publish)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    caught: BaseException | None = None
    try:
        interim._materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            raw_root=raw_root,
            data_ids=("trading_calendar",),
            start_date=None,
            end_date=None,
            override_path=None,
            chunksize=1,
            resource_policy=policy,
        )
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    actual = {
        "replacement_hits": replacement_hits,
        "publication_hits": publication_hits,
        "error": _phase2_error_text(caught),
    }
    expected = {
        "replacement_hits": 1,
        "publication_hits": [],
        "error": "ValueError:csmar_raw_export_identity_changed",
    }
    assert actual == expected


def _phase2_mutate_second_replay_path(path: Path, mutation: str) -> None:
    if mutation == "ordinary":
        identity = path.stat()
        replacement = path.with_name(f"{path.name}.replacement")
        replacement.write_bytes(path.read_bytes())
        os.utime(replacement, ns=(identity.st_atime_ns, identity.st_mtime_ns))
        os.replace(replacement, path)
        return
    if mutation == "ancestor_alias":
        original_parent = path.parent
        moved_parent = original_parent.with_name(f"{original_parent.name}-real")
        original_parent.rename(moved_parent)
        try:
            original_parent.symlink_to(moved_parent, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlink unavailable: {exc}")
        return
    moved = path.with_name(f"{path.name}.real")
    path.rename(moved)
    try:
        if mutation == "file_alias":
            path.symlink_to(moved)
        elif mutation == "hardlink":
            os.link(moved, path)
        elif mutation == "directory":
            path.mkdir()
        else:  # pragma: no cover - parametrization owns the closed set
            raise AssertionError(f"unknown mutation: {mutation}")
    except OSError as exc:
        pytest.skip(f"{mutation} unavailable: {exc}")


@pytest.mark.parametrize(
    "mutation", ("ordinary", "ancestor_alias", "file_alias", "hardlink", "directory")
)
def test_dataset_second_replay_rejects_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    raw_root = tmp_path / "raw"
    raw_path = raw_root / "child" / "export.csv"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("value\n1\n", encoding="utf-8")
    _phase2_stub_raw_inspection(monkeypatch)
    real_values = raw_exports.ExactPrimaryKeyIndex.identity_payload_values
    namespaces: list[str] = []
    mutation_hits = 0

    def mutating_values(self, *, namespace: str):  # noqa: ANN001, ANN202
        nonlocal mutation_hits
        namespaces.append(namespace)
        if namespace == "csmar-dataset-paths":
            mutation_hits += 1
            _phase2_mutate_second_replay_path(raw_path, mutation)
        yield from real_values(self, namespace=namespace)

    monkeypatch.setattr(
        raw_exports.ExactPrimaryKeyIndex,
        "identity_payload_values",
        mutating_values,
    )
    caught: BaseException | None = None
    results: list[object] = []
    try:
        results = list(raw_exports.dataset_files_with_identity_by_id(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    assert namespaces == [
        "csmar-dataset-physical-exports-paths",
        "csmar-dataset-paths",
    ]
    assert mutation_hits == 1
    assert results == []
    assert _phase2_error_text(caught) == (
        "ValueError:csmar_raw_export_identity_changed"
    )


@pytest.mark.parametrize("primary_type", (KeyboardInterrupt,))
def test_dataset_second_replay_final_root_check_preserves_active_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    raw_root = tmp_path / "raw"
    _phase2_write_raw_exports(raw_root, 1)
    _phase2_stub_raw_inspection(monkeypatch)
    real_values = raw_exports.ExactPrimaryKeyIndex.identity_payload_values
    real_file_identity = raw_exports.raw_export_file_identity
    real_samestat = os.path.samestat
    second_replay_hits = 0
    identity_check_hits = 0
    final_root_check_hits = 0
    primary = primary_type("second_replay_primary")

    def tracked_values(self, *, namespace: str):  # noqa: ANN001, ANN202
        nonlocal second_replay_hits
        if namespace == "csmar-dataset-paths":
            second_replay_hits += 1
        yield from real_values(self, namespace=namespace)

    def failing_file_identity(path: Path):  # noqa: ANN202
        nonlocal identity_check_hits
        if second_replay_hits:
            identity_check_hits += 1
            raise primary
        return real_file_identity(path)

    def failing_final_samestat(left: os.stat_result, right: os.stat_result) -> bool:
        nonlocal final_root_check_hits
        if identity_check_hits:
            final_root_check_hits += 1
            raise OSError("root_final_secondary")
        return real_samestat(left, right)

    monkeypatch.setattr(
        raw_exports.ExactPrimaryKeyIndex,
        "identity_payload_values",
        tracked_values,
    )
    monkeypatch.setattr(raw_exports, "raw_export_file_identity", failing_file_identity)
    monkeypatch.setattr(os.path, "samestat", failing_final_samestat)
    caught: BaseException | None = None
    try:
        list(raw_exports.dataset_files_with_identity_by_id(raw_root))
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    assert caught is primary
    assert second_replay_hits == 1
    assert identity_check_hits == 1
    assert final_root_check_hits >= 1


def test_interim_second_replay_rejects_before_transform_or_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_path = raw_root / "TRDNEW_Cale.csv"
    raw_root.mkdir()
    _write_trading_calendar_raw_export(raw_path)
    _phase2_stub_raw_inspection(monkeypatch)
    real_values = raw_exports.ExactPrimaryKeyIndex.identity_payload_values
    second_replay_hits = 0
    transform_hits = 0
    publication_hits = 0

    def mutating_values(self, *, namespace: str):  # noqa: ANN001, ANN202
        nonlocal second_replay_hits
        if namespace == "csmar-dataset-paths":
            second_replay_hits += 1
            _phase2_mutate_second_replay_path(raw_path, "ordinary")
        yield from real_values(self, namespace=namespace)

    def frames(**_kwargs):  # noqa: ANN003, ANN202
        nonlocal transform_hits
        transform_hits += 1
        return iter(())

    def publish(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        nonlocal publication_hits
        publication_hits += 1

    monkeypatch.setattr(
        raw_exports.ExactPrimaryKeyIndex,
        "identity_payload_values",
        mutating_values,
    )
    monkeypatch.setattr(interim, "_iter_interim_frames", frames)
    monkeypatch.setattr(interim, "write_dataset", publish)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    caught: BaseException | None = None
    try:
        interim._materialize_interim_data(
            layout=StorageLayout(project_root=tmp_path),
            raw_root=raw_root,
            data_ids=("trading_calendar",),
            start_date=None,
            end_date=None,
            override_path=None,
            chunksize=1,
            resource_policy=policy,
        )
    except BaseException as exc:  # pragma: no branch - asserted below
        caught = exc

    assert second_replay_hits == 1
    assert transform_hits == 0
    assert publication_hits == 0
    assert _phase2_error_text(caught) == (
        "ValueError:csmar_raw_export_identity_changed"
    )
