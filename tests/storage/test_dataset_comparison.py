from __future__ import annotations

import gc
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.storage.datasets.comparison as dataset_comparison
import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_mkdir,
    filesystem_stat,
    native_filesystem_path,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_iter_frame_chunks_reuses_bounded_frame_storage() -> None:
    frame = pd.DataFrame({"value": np.arange(20000, dtype="int64")})
    chunks = tuple(dataset_reading.iter_frame_chunks(frame, batch_rows=10000))
    assert len(chunks) >= 2
    assert all(
        np.shares_memory(
            chunk["value"].to_numpy(copy=False), frame["value"].to_numpy(copy=False)
        )
        for chunk in chunks
    )


@pytest.mark.parametrize(
    "primary_type", (RuntimeError, KeyboardInterrupt, GeneratorExit)
)
def test_fingerprint_consumer_failure_retains_primary_and_reader_identity_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primary_type
) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_consumer")
    write_dataset(
        root,
        make_default_dataset_frame().iloc[:1],
        spec=make_processed_dataset_spec("fingerprint_consumer"),
    )
    part = next(filesystem_glob(root / "data", "**/*.parquet"))
    primary = primary_type("fingerprint_consumer_failed")
    unraisable = []

    def fail_hash(*_args, **_kwargs):
        before = filesystem_stat(part)
        os.utime(
            native_filesystem_path(part),
            ns=(before.st_atime_ns, before.st_mtime_ns + 100_000_000),
        )
        raise primary

    monkeypatch.setattr(dataset_fingerprinting, "_iter_fingerprint_hashes", fail_hash)
    monkeypatch.setattr(
        sys, "unraisablehook", lambda event: unraisable.append(event.exc_value)
    )
    with pytest.raises(primary_type) as caught:
        fingerprint_dataset(root)
    assert caught.value is primary
    assert primary.__notes__ == [
        "file identity verification also failed: ValueError",
        "dataset verification also failed: ValueError",
    ]
    gc.collect()
    assert unraisable == []


def test_compare_requires_exact_proof_after_forced_fingerprint_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = make_dataset_artifact_root(tmp_path, "collision_left")
    right = make_dataset_artifact_root(tmp_path, "collision_right")
    write_dataset(
        left,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("collision_left"),
    )
    changed = make_default_dataset_frame()
    changed.loc[changed.index[-1], "value"] = 99.0
    write_dataset(right, changed, spec=make_processed_dataset_spec("collision_right"))
    collision = fingerprint_dataset(left)

    def owned_collision(*_args, **_kwargs):
        return collision

    monkeypatch.setattr(
        dataset_comparison, "fingerprint_dataset_session", owned_collision
    )
    comparison = dataset_comparison.compare_datasets(left, right)
    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.summary["exact"] == {
        "missing_row_count": 0,
        "extra_row_count": 0,
        "changed_row_count": 1,
    }


def test_compare_treats_partition_and_file_boundaries_as_layout_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = make_dataset_artifact_root(tmp_path, "layout_left")
    right = make_dataset_artifact_root(tmp_path, "layout_right")
    frame = make_default_dataset_frame()
    write_dataset(
        left,
        (frame.iloc[:1], frame.iloc[1:]),
        spec=make_processed_dataset_spec("layout_left"),
    )
    write_dataset(right, frame, spec=make_processed_dataset_spec("layout_right"))
    fingerprint = fingerprint_dataset(left)
    right_fingerprint = replace(
        fingerprint,
        partition_fingerprints=(
            {
                "partition": "layout-only",
                "row_count": fingerprint.row_count,
                "content_hash": fingerprint.content_hash,
            },
        ),
        file_count=1,
    )
    fingerprints = iter((fingerprint, right_fingerprint))

    def owned_fingerprint(*_args, **_kwargs):
        return next(fingerprints)

    monkeypatch.setattr(
        dataset_comparison, "fingerprint_dataset_session", owned_fingerprint
    )
    comparison = compare_datasets(left, right)
    assert comparison.status == "MATCH"
    assert comparison.matches
    assert comparison.layout_delta["file_count"] == {"left": 3, "right": 1}
    assert comparison.layout_delta["partitions"]["left"]
    assert comparison.layout_delta["partitions"]["right"]
    assert all(
        set(partition) == {"partition", "row_count"}
        for side in ("left", "right")
        for partition in comparison.layout_delta["partitions"][side]
    )
    assert (
        comparison.layout_delta["partitions"]["left"]
        != comparison.layout_delta["partitions"]["right"]
    )


def test_compare_propagates_programmer_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left_programmer_error"
    right = tmp_path / "right_programmer_error"
    filesystem_mkdir(left)
    filesystem_mkdir(right)
    monkeypatch.setattr(
        dataset_comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("programmer_error")
        ),
    )
    with pytest.raises(RuntimeError, match="programmer_error"):
        dataset_comparison.compare_datasets(left, right)


def test_compare_rejects_mutation_between_fingerprint_and_exact_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = make_dataset_artifact_root(tmp_path, "mutation_left")
    right = make_dataset_artifact_root(tmp_path, "mutation_right")
    frame = make_default_dataset_frame()
    write_dataset(left, frame, spec=make_processed_dataset_spec("mutation_left"))
    write_dataset(right, frame, spec=make_processed_dataset_spec("mutation_right"))
    fingerprint = dataset_comparison.fingerprint_dataset_session
    calls = 0

    def changing_identity(context, **kwargs):
        nonlocal calls
        result = fingerprint(context, **kwargs)
        calls += 1
        if calls == 2:
            part = next(filesystem_glob(left / "data", "**/*.parquet"))
            stat = filesystem_stat(part)
            os.utime(
                native_filesystem_path(part),
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000),
            )
        return result

    monkeypatch.setattr(
        dataset_comparison, "fingerprint_dataset_session", changing_identity
    )
    comparison = compare_datasets(left, right)
    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize("mutated_side", ("left", "right"))
def test_compare_best_effort_observed_stage_drift_rejects_post_index_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated_side: str
) -> None:
    roots = {
        side: make_dataset_artifact_root(tmp_path, f"final_barrier_{side}")
        for side in ("left", "right")
    }
    frame = make_default_dataset_frame()
    for side, root in roots.items():
        write_dataset(
            root, frame, spec=make_processed_dataset_spec(f"final_barrier_{side}")
        )
    batches = dataset_comparison.iter_dataset_file_batches
    injected = False

    def index_then_mutate(context, **kwargs: object):
        nonlocal injected
        yield from batches(context, **kwargs)
        if context.root != roots[mutated_side]:
            return
        injected = True
        part = next(filesystem_glob(context.root / "data", "**/*.parquet"))
        stat = filesystem_stat(part)
        os.utime(
            native_filesystem_path(part),
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 10000000),
        )

    monkeypatch.setattr(
        dataset_comparison, "iter_dataset_file_batches", index_then_mutate
    )
    comparison = compare_datasets(roots["left"], roots["right"])
    assert injected
    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


@pytest.mark.parametrize(
    ("primary_type", "verify_type"),
    (
        (RuntimeError, OSError),
        (MemoryError, ValueError),
        (KeyboardInterrupt, OSError),
        (SystemExit, ValueError),
    ),
)
@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
def test_reader_final_verify_does_not_replace_active_primary(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    verify_type: type[BaseException],
    iterator_name: str,
) -> None:

    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1 if iterator_name == "file_batches" else 0

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise verify_type("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    context = Context()

    def fail_batches(*_args: object, **_kwargs: object):
        raise primary_type("primary_failure")
        yield

    if iterator_name == "file_batches":
        monkeypatch.setattr(
            dataset_reading,
            "pandas_dtype_overrides_from_logical_payload",
            lambda *_args, **_kwargs: {},
        )
        monkeypatch.setattr(
            dataset_reading, "iter_declared_parquet_member_batches", fail_batches
        )
        iterator = dataset_reading.iter_dataset_file_batches(
            context, columns=("value",), batch_rows=2
        )
    else:
        monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", fail_batches)
        iterator = dataset_reading.iter_dataset_chunks(context, batch_rows=2)
    with pytest.raises(primary_type, match="primary_failure"):
        next(iterator)


@pytest.mark.parametrize("completion", ("exhausted", "early_close"))
def test_reader_final_verify_failure_propagates_without_external_primary(
    monkeypatch: pytest.MonkeyPatch, completion: str
) -> None:
    frame = pd.DataFrame({"value": [1]})

    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise OSError("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    monkeypatch.setattr(
        dataset_reading,
        "pandas_dtype_overrides_from_logical_payload",
        lambda *_args, **_kwargs: {},
    )

    def declared(*_args: object, **_kwargs: object):
        yield frame

    monkeypatch.setattr(
        dataset_reading, "iter_declared_parquet_member_batches", declared
    )
    context = Context()
    iterator = dataset_reading.iter_dataset_file_batches(
        context, columns=("value",), batch_rows=2
    )
    if completion == "exhausted":
        with pytest.raises(OSError, match="verify_failure"):
            list(iterator)
    else:
        next(iterator)
        with pytest.raises(OSError, match="verify_failure"):
            iterator.close()


@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
@pytest.mark.parametrize(
    "primary_type", (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit)
)
def test_dataset_iterator_context_preserves_caller_primary(
    monkeypatch: pytest.MonkeyPatch,
    iterator_name: str,
    primary_type: type[BaseException],
) -> None:
    frame = pd.DataFrame({"value": [1]})

    class Context:
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {"value": {"kind": "numpy_int", "dtype": "int64"}}
        parquet_filters = None

        def __init__(self) -> None:
            self.verify_calls = 0
            self.verify_fail_after = 1 if iterator_name == "file_batches" else 0

        def verify(self) -> None:
            self.verify_calls += 1
            if self.verify_calls > self.verify_fail_after:
                raise OSError("verify_failure")

        def metadata_for(self, _path: Path) -> object:
            return SimpleNamespace(identity=object())

    context = Context()
    if iterator_name == "file_batches":
        monkeypatch.setattr(
            dataset_reading,
            "pandas_dtype_overrides_from_logical_payload",
            lambda *_args, **_kwargs: {},
        )

        def declared(*_args: object, **_kwargs: object):
            yield frame

        monkeypatch.setattr(
            dataset_reading, "iter_declared_parquet_member_batches", declared
        )
        iterator = dataset_reading.iter_dataset_file_batches(
            context, columns=("value",), batch_rows=2
        )
    else:

        def batches(*_args: object, **_kwargs: object):
            yield (Path("part.parquet"), frame)

        monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", batches)
        iterator = dataset_reading.iter_dataset_chunks(context, batch_rows=2)
    with pytest.raises(primary_type, match="primary_failure"):
        with closing_iterator(iterator):
            next(iterator)
            raise primary_type("primary_failure")


@pytest.mark.parametrize(
    "primary_type", (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit)
)
def test_materialized_iterator_close_does_not_replace_caller_primary(
    monkeypatch: pytest.MonkeyPatch, primary_type: type[BaseException]
) -> None:

    class FailingIterator:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary_type("primary_failure")

        def close(self) -> None:
            raise OSError("close_failure")

    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: FailingIterator(),
    )
    with pytest.raises(primary_type, match="primary_failure"):
        dataset_reading.materialize_dataset_session(SimpleNamespace())


def test_compare_float_value_zero_keeps_bit_identity(tmp_path: Path) -> None:
    left = make_dataset_artifact_root(tmp_path, "float_value_negative_zero")
    right = make_dataset_artifact_root(tmp_path, "float_value_positive_zero")
    for root, zero in ((left, -0.0), (right, 0.0)):
        write_dataset(
            root,
            make_default_dataset_frame(
                asset_ids=("asset_001",), dates=("2014-01-02",), values=(zero,)
            ),
            spec=make_processed_dataset_spec(root.name),
        )
    assert compare_datasets(left, right).status == "CONTENT_MISMATCH"


def test_aggregate_chunked_and_materialized_reads_match(tmp_path: Path) -> None:
    chunk_rows = 2000
    chunk_count = 16
    total_rows = chunk_rows * chunk_count
    asset_suffix = "x" * 512
    chunks: list[pd.DataFrame] = []
    for start in range(0, total_rows, chunk_rows):
        positions = np.arange(start, start + chunk_rows, dtype=np.int64)
        chunks.append(
            pd.DataFrame(
                {
                    "market_id": ["a_share"] * chunk_rows,
                    "asset_id": [
                        f"asset_{value % 8:02d}_{asset_suffix}" for value in positions
                    ],
                    "timestamp": pd.date_range(
                        pd.Timestamp("2020-01-01") + pd.Timedelta(minutes=int(start)),
                        periods=chunk_rows,
                        freq="min",
                    ),
                    "value": positions.astype("float64"),
                }
            )
        )
    left = make_dataset_artifact_root(tmp_path / "left_parent", "aggregate_streaming")
    right = make_dataset_artifact_root(tmp_path / "right_parent", "aggregate_streaming")
    spec = make_processed_dataset_spec("aggregate_streaming")
    for root in (left, right):
        inspection = dataset_writing.write_dataset(root, chunks, spec=spec)
        assert inspection.row_count == total_rows
    assert (
        len(dataset_reading.read_dataset(left, mode=DatasetReadMode.MATERIALIZED))
        == total_rows
    )
    streamed_rows = 0
    streamed = dataset_reading.read_dataset(left, mode=DatasetReadMode.CHUNKED)
    try:
        for chunk in streamed:
            streamed_rows += len(chunk)
            del chunk
    finally:
        close = getattr(streamed, "close", None)
        if close is not None:
            close()
    assert streamed_rows == total_rows
    inspections = []
    for root in (left, right):
        inspections.append(dataset_inspection.inspect_dataset(root))
    left_inspection, right_inspection = inspections
    assert left_inspection.row_count == right_inspection.row_count == total_rows
    fingerprints = []
    for root in (left, right):
        fingerprints.append(dataset_fingerprinting.fingerprint_dataset(root))
    left_fingerprint, right_fingerprint = fingerprints
    assert left_fingerprint == right_fingerprint
    comparison = dataset_comparison.compare_datasets(left, right)
    assert comparison.status == "MATCH"


@pytest.mark.parametrize("common_value, changed", ((0.0, 1), (float("nan"), 0)))
def test_exact_comparison_keeps_missing_extra_and_nullable_changed_counts(
    tmp_path, common_value, changed
):
    roots = {
        side: make_dataset_artifact_root(tmp_path, side) for side in ("left", "right")
    }
    frames = {
        "left": pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["a", "b"],
                "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "value": [1.0, float("nan")],
            }
        ),
        "right": pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["b", "c"],
                "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01"]),
                "value": [common_value, 2.0],
            }
        ),
    }
    for side, frame in frames.items():
        write_dataset(
            roots[side],
            (frame.iloc[:1], frame.iloc[1:]),
            spec=make_processed_dataset_spec(side),
        )
    sessions = {
        side: dataset_reading.open_dataset_session(root, columns=None, time_window=None)
        for side, root in roots.items()
    }
    assert dataset_comparison._exact_comparison_counts(
        sessions["left"],
        sessions["right"],
        primary_key=("market_id", "asset_id", "timestamp"),
    ) == {"missing_row_count": 1, "extra_row_count": 1, "changed_row_count": changed}
    for session in sessions.values():
        session.verify()
