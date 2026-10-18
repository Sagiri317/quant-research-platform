from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
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
    filesystem_mkdir,
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

    def collision_hashes(frame, **_kwargs):
        yield np.zeros(len(frame), dtype="uint64")

    monkeypatch.setattr(
        dataset_fingerprinting, "_iter_fingerprint_hashes", collision_hashes
    )
    assert (
        fingerprint_dataset(left).content_hash
        == fingerprint_dataset(right).content_hash
    )
    comparison = dataset_comparison.compare_datasets(left, right)
    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.summary["exact"] == {
        "missing_row_count": 0,
        "extra_row_count": 0,
        "changed_row_count": 1,
    }


def test_compare_treats_file_boundaries_as_layout_only(
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
    write_dataset(
        right,
        frame,
        spec=make_processed_dataset_spec("layout_right"),
    )
    comparison = compare_datasets(left, right)
    assert comparison.status == "MATCH"
    assert comparison.matches
    assert comparison.layout_delta["file_count"] == {"left": 3, "right": 2}
    assert comparison.layout_delta["partitions"]["left"]
    assert comparison.layout_delta["partitions"]["right"]
    assert all(
        set(partition) == {"partition", "row_count"}
        for side in ("left", "right")
        for partition in comparison.layout_delta["partitions"][side]
    )
    assert (
        comparison.layout_delta["partitions"]["left"]
        == comparison.layout_delta["partitions"]["right"]
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


@pytest.mark.parametrize(
    "primary_type", (RuntimeError, MemoryError, KeyboardInterrupt, SystemExit)
)
@pytest.mark.parametrize("iterator_name", ("file_batches", "chunks"))
def test_reader_propagates_decode_failure(
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
    iterator_name: str,
) -> None:

    class Context:
        data_root = Path(".")
        _expected_partition_rows = ()
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {}
        arrow_schema = pa.schema([("value", pa.int64())])
        parquet_filters = None

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
        data_root = Path(".")
        _expected_partition_rows = ()
        read_paths = (Path("part.parquet"),)
        physical_columns = ("value",)
        logical_columns = ("value",)
        logical_dtypes = {}
        arrow_schema = pa.schema([("value", pa.int64())])
        parquet_filters = None

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
    chunk_rows = 2
    chunk_count = 16
    total_rows = chunk_rows * chunk_count
    chunks: list[pd.DataFrame] = []
    for start in range(0, total_rows, chunk_rows):
        positions = np.arange(start, start + chunk_rows, dtype=np.int64)
        chunks.append(
            pd.DataFrame(
                {
                    "market_id": ["a_share"] * chunk_rows,
                    "asset_id": [f"asset_{value % 8:02d}" for value in positions],
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
        assert inspection.file_count == chunk_count
    expected = pd.concat(chunks, ignore_index=True)
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    materialized = dataset_reading.read_dataset(left, mode=DatasetReadMode.MATERIALIZED)
    pd.testing.assert_frame_equal(
        materialized.loc[:, expected.columns]
        .sort_values("timestamp")
        .reset_index(drop=True),
        expected,
        check_exact=True,
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
    comparison = compare_datasets(roots["left"], roots["right"])
    assert comparison.status == "CONTENT_MISMATCH"
    assert comparison.summary["exact"] == {
        "missing_row_count": 1,
        "extra_row_count": 1,
        "changed_row_count": changed,
    }


@pytest.mark.parametrize("changed", [False, True])
def test_compare_nested_values_and_float_keys_decodes_each_member_once(
    tmp_path, monkeypatch, changed
):
    roots = {
        side: make_dataset_artifact_root(tmp_path, side) for side in ("left", "right")
    }
    for side, root in roots.items():
        frame = pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": [-0.0 if side == "left" else 0.0, 1.0],
                "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "value": [[1, None], [2, 4 if changed and side == "right" else 3]],
            }
        )
        write_dataset(root, frame, spec=make_processed_dataset_spec(side))
    decoded = {}
    original = dataset_reading.iter_declared_parquet_member_batches

    def observe(path, **kwargs):
        decoded[path] = decoded.get(path, 0) + 1
        yield from original(path, **kwargs)

    monkeypatch.setattr(
        dataset_reading, "iter_declared_parquet_member_batches", observe
    )
    result = compare_datasets(roots["left"], roots["right"])
    assert result.status == ("CONTENT_MISMATCH" if changed else "MATCH")
    assert result.summary["exact"] == {
        "missing_row_count": 0,
        "extra_row_count": 0,
        "changed_row_count": int(changed),
    }
    assert decoded and set(decoded.values()) == {1}
    from quant_research_platform.contracts.time import TimeWindow

    decoded.clear()
    projected = compare_datasets(
        roots["left"],
        roots["right"],
        columns=("value",),
        time_window=TimeWindow("2020-01-01", "2020-01-01"),
    )
    assert projected.status == "MATCH"
    assert projected.summary["row_count"] == {"left": 1, "right": 1}
    assert decoded and set(decoded.values()) == {1}
