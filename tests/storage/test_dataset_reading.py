from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.reading as dataset_reading
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_read_dataset_materialized_chunked_projection_and_time_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    materialized = read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2014-01-01", "2014-12-31"),
    )
    consumed = []
    original = dataset_reading.iter_dataset_file_batches

    def measured(*args, **kwargs):
        for path, frame in original(*args, **kwargs):
            consumed.append(len(frame))
            yield path, frame

    monkeypatch.setattr(dataset_reading, "iter_dataset_file_batches", measured)
    stream = read_dataset(root, mode=DatasetReadMode.CHUNKED)
    assert consumed == []
    first = next(stream)
    assert consumed == [len(first)]
    chunks = [first, *stream]
    assert list(materialized.columns) == ["asset_id", "value"]
    assert materialized["asset_id"].tolist() == ["asset_001"]
    assert materialized["value"].tolist() == [1.0]
    expected = (
        make_default_dataset_frame()
        .sort_values("timestamp", kind="stable")
        .astype({"timestamp": "datetime64[ns]"})
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        pd.concat(chunks, ignore_index=True), expected, check_exact=True
    )
    assert sum(consumed) == len(expected) == 3


def test_internal_physical_reader_uses_bounded_record_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "bounded_reader", market_id="dummy")
    row_count = 7
    frame = pd.DataFrame(
        {
            "market_id": ["dummy"] * row_count,
            "asset_id": [f"asset_{index:04d}" for index in range(row_count)],
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "value": [float(index) for index in range(row_count)],
        }
    )
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("bounded_reader", market_id="dummy"),
    )
    context = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    assert len(context.read_paths) == 1
    physical_sizes = []

    physical_reader = dataset_reading.iter_dataset_part_batches

    def observe(path, **kwargs):
        assert kwargs["batch_rows"] == 2
        for batch in physical_reader(path, **kwargs):
            physical_sizes.append(batch.num_rows)
            yield batch

    monkeypatch.setattr(dataset_reading, "iter_dataset_part_batches", observe)
    chunks = list(dataset_reading.iter_dataset_chunks(context, batch_rows=2))
    assert physical_sizes == [2, 2, 2, 1]
    assert [len(chunk) for chunk in chunks] == [2, 2, 2, 1]
    pd.testing.assert_frame_equal(
        pd.concat(chunks, ignore_index=True), frame, check_exact=True
    )
    window = TimeWindow(frame.loc[1, "timestamp"], frame.loc[5, "timestamp"])
    projected_context = dataset_reading.open_dataset_session(
        root, columns=("value", "asset_id"), time_window=window
    )
    physical_sizes.clear()
    projected_chunks = list(
        dataset_reading.iter_dataset_chunks(projected_context, batch_rows=2)
    )
    assert physical_sizes and max(physical_sizes) <= 2
    assert all(len(chunk) <= 2 for chunk in projected_chunks)
    expected_projected = frame.loc[1:5, ["value", "asset_id"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        pd.concat(projected_chunks, ignore_index=True),
        expected_projected,
        check_exact=True,
    )


def test_internal_physical_reader_preserves_single_8kib_text_row(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path, "oversized_reader_row", market_id="dummy"
    )
    frame = pd.DataFrame(
        {
            "market_id": ["dummy"],
            "asset_id": ["asset_a"],
            "timestamp": [pd.Timestamp("2026-01-01")],
            "value": ["x" * 8192],
        }
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("oversized_reader_row", market_id="dummy"),
    )
    context = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    chunks = list(dataset_reading.iter_dataset_chunks(context, batch_rows=1))
    expected = frame.copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), expected)


def test_bounded_dataset_read_path_never_calls_full_file_pandas_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "no_full_file_reader")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("no_full_file_reader"),
    )

    def fail_read_parquet(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded_dataset_read_must_not_call_pd_read_parquet")

    monkeypatch.setattr(dataset_reading.pd, "read_parquet", fail_read_parquet)
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    assert sum(len(chunk) for chunk in chunks) == 3


@pytest.mark.parametrize("extra", [{}, {"schema_hash": "0" * 64}])
def test_storage_and_diagnostics_do_not_require_persisted_schema_hash(
    tmp_path: Path, extra: dict
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    expected = make_default_dataset_frame().astype({"timestamp": "datetime64[ns]"})
    write_dataset(root, expected, spec=make_processed_dataset_spec())
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "schema_hash" not in manifest
    manifest.update(extra)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    keys = ["timestamp", "asset_id"]
    pd.testing.assert_frame_equal(
        read_dataset(root).sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
    )
    assert inspect_dataset(root).row_count == len(expected)
    assert fingerprint_dataset(root).schema_hash


def test_read_dataset_rejects_data_root_missing_and_missing_projection_columns(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    with pytest.raises(ValueError, match="dataset_column_not_found"):
        read_dataset(root, columns=["asset_id", "missing"])


def test_inspect_dataset_returns_expanded_metadata(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    inspection = inspect_dataset(root)
    assert inspection.root == root
    assert inspection.data_root == root / "data"
    assert inspection.spec["data_id"] == "x"
    assert inspection.spec["kind"] == "factor"
    assert inspection.columns == ("market_id", "asset_id", "timestamp", "value")
    assert inspection.file_count == 2
    assert inspection.row_count == 3
    assert inspection.partitions == ("year=2014", "year=2015")
    assert inspection.partition_row_counts == {"year=2014": 1, "year=2015": 2}
    assert inspection.date_min == "2014-01-02"
    assert inspection.date_max == "2015-01-05"


def test_write_dataset_preserves_factor_and_label_nan_profiles(tmp_path: Path) -> None:
    factor_root = make_dataset_artifact_root(tmp_path, "factor_nan")
    label_root = make_dataset_artifact_root(
        tmp_path, "label_x", kind=ProcessedDataKind.LABEL
    )
    label_frame = make_default_dataset_frame().rename(columns={"value": "label"})
    write_dataset(
        factor_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("factor_nan"),
    )
    write_dataset(
        label_root,
        label_frame,
        spec=make_processed_dataset_spec(
            "label_x", kind=ProcessedDataKind.LABEL, value_column="label"
        ),
    )
    factor_manifest = json.loads(
        (factor_root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    label_manifest = json.loads(
        (label_root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert factor_manifest["factor_nan_profile"] == {"value": 1}
    assert factor_manifest["label_nan_profile"] == {}
    assert label_manifest["factor_nan_profile"] == {}
    assert label_manifest["label_nan_profile"] == {"label": 1}


def test_compare_datasets_rejects_schema_difference(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path, "left")
    schema_root = make_dataset_artifact_root(tmp_path, "schema")
    write_dataset(
        left_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("left"),
    )
    wider_spec = ProcessedDataSpec(
        data_id="schema",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **make_processed_dataset_spec().column_roles,
            "other": ColumnRole.FACTOR,
        },
        primary_key=make_processed_dataset_spec().primary_key,
        partition_columns=("year",),
    )
    write_dataset(
        schema_root, make_default_dataset_frame().assign(other=1.0), spec=wider_spec
    )
    schema_comparison = compare_datasets(left_root, schema_root)
    assert schema_comparison.status == "SCHEMA_MISMATCH"
    assert not schema_comparison.matches


def test_compare_datasets_reports_input_metadata_difference_for_same_content(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path, "left_metadata")
    right_root = make_dataset_artifact_root(tmp_path, "right_metadata")
    write_dataset(
        left_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("left_metadata"),
    )
    write_dataset(
        right_root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("right_metadata"),
        metadata={"inputs": {"demo_input": "supplied.parquet"}},
    )
    comparison = compare_datasets(left_root, right_root)
    assert comparison.status == "METADATA_MISMATCH"
    assert not comparison.matches


def test_empty_materialized_and_chunked_reads_preserve_stored_dtypes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_dtypes")
    expected = make_default_dataset_frame().iloc[:0].copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    expected["asset_id"] = pd.Categorical(
        expected["asset_id"], categories=["asset_001", "asset_002"]
    )
    write_dataset(root, expected, spec=make_processed_dataset_spec("empty_dtypes"))
    materialized = dataset_reading.read_dataset(root, mode=DatasetReadMode.MATERIALIZED)
    chunked = list(dataset_reading.read_dataset(root, mode=DatasetReadMode.CHUNKED))
    assert len(chunked) == 1
    assert (
        materialized.dtypes.astype(str).to_dict()
        == expected.dtypes.astype(str).to_dict()
    )
    assert (
        chunked[0].dtypes.astype(str).to_dict() == expected.dtypes.astype(str).to_dict()
    )


def test_time_window_filter_is_applied_before_arrow_to_pandas_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "physical_time_filter")
    row_count = 3
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": ["asset_a"] * row_count,
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="min"),
            "value": np.arange(row_count, dtype="float64"),
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("physical_time_filter"))
    converted_rows = 0
    original = dataset_reading._arrow_batch_to_bounded_pandas

    def record_rows(batch: pa.RecordBatch, **kwargs):
        nonlocal converted_rows
        converted_rows += batch.num_rows
        yield from original(batch, **kwargs)

    monkeypatch.setattr(dataset_reading, "_arrow_batch_to_bounded_pandas", record_rows)
    target = frame.loc[1, "timestamp"]
    result = read_dataset(
        root, time_window=TimeWindow(target.to_pydatetime(), target.to_pydatetime())
    )
    assert len(result) == 1
    assert converted_rows == 1
    assert result["timestamp"].iloc[0] == target
    assert result["value"].tolist() == [1.0]


def test_time_window_row_group_pruning_uses_parquet_leaf_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    original_writer = pq.ParquetWriter

    def writer_with_statistics(*args, **kwargs):
        kwargs["write_statistics"] = True
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetWriter", writer_with_statistics)
    root = tmp_path / "extra_vendor_fields.parquet"
    spec = ProcessedDataSpec(
        data_id="nested_before_timestamp",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "context": ColumnRole.CONTEXT,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    target = pd.Timestamp("2026-01-02")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "context": [
                {
                    "first": pd.Timestamp("2000-01-01"),
                    "second": pd.Timestamp("2000-01-02"),
                }
            ],
            "timestamp": [target],
            "value": [1.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), root)
    with pq.ParquetFile(root) as parquet:
        top_level_position = parquet.schema_arrow.get_field_index("timestamp")
        leaf_position = next(
            index
            for index in range(len(parquet.schema))
            if parquet.schema.column(index).path == "timestamp"
        )
        assert leaf_position != top_level_position
        assert parquet.schema.column(top_level_position).path == "context.second"
        group = parquet.metadata.row_group(0)
        correct_stats = group.column(leaf_position).statistics
        wrong_stats = group.column(top_level_position).statistics
        assert correct_stats is not None and correct_stats.has_min_max
        assert wrong_stats is not None and wrong_stats.has_min_max
        assert pd.Timestamp(correct_stats.min) == target
        assert pd.Timestamp(correct_stats.max) == target
        assert pd.Timestamp(wrong_stats.max) < target
    result = read_dataset(
        root,
        spec=spec,
        columns=["timestamp", "value"],
        time_window=TimeWindow(target.to_pydatetime(), target.to_pydatetime()),
    )
    assert len(result) == 1, "retained_target_row_missing"
    assert result.loc[0, "timestamp"] == target


def test_fingerprint_hashes_ordinary_frame_without_normalized_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame({"value": np.arange(3, dtype="int64")})
    expected = pd.util.hash_pandas_object(frame, index=False).to_numpy("uint64")
    values = frame["value"].to_numpy(copy=False)
    original = pd.util.hash_pandas_object
    slices = []

    def hash_view(current, *args, **kwargs):
        assert np.shares_memory(current["value"].to_numpy(copy=False), values)
        slices.append(len(current))
        return original(current, *args, **kwargs)

    monkeypatch.setattr(pd.util, "hash_pandas_object", hash_view)
    observed = tuple(dataset_fingerprinting._iter_fingerprint_hashes(frame))
    assert len(observed) == 1
    assert slices == [len(frame)]
    np.testing.assert_array_equal(observed[0], expected)


@pytest.mark.parametrize("data_type", (pa.float16(), pa.float32(), pa.float64()))
@pytest.mark.parametrize("values", ((-0.0, 0.0), (1.0, 2.0)))
def test_fingerprint_normalizes_float_primary_key_zero(
    data_type: pa.DataType, values, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"asset_id": values})
    schema = pa.schema([("asset_id", data_type)])
    monkeypatch.setattr(
        pd.util,
        "hash_pandas_object",
        lambda *_args, **_kwargs: pytest.fail("float_primary_key_was_hashed"),
    )
    (hashes,) = dataset_fingerprinting._iter_fingerprint_hashes(
        frame, arrow_schema=schema, primary_key=("asset_id",)
    )
    assert bool(hashes[0] == hashes[1]) == (values[0] == values[1])


@pytest.mark.parametrize(
    "selected, operation, expected",
    [
        ({}, {"start": "2021-01-02", "end": "2021-01-02"}, ["d", "e"]),
        (
            {"time_window": {"start": "2021-01-02", "end": "2021-01-02"}},
            {"start": "2021-01-02T23:59:59.999999999", "end": None},
            ["e"],
        ),
        ({}, {"start": "2021-01-03T01:00:00+01:00", "end": None}, ["f"]),
        ({"years": [2019]}, {"start": "2021-01-02", "end": "2021-01-02"}, []),
        (
            {"time_window": {"start": "2021-01-03", "end": "2021-01-03"}},
            {"start": "2021-01-02", "end": "2021-01-02"},
            [],
        ),
        (
            {"row_filters": [["asset_id", "in", ["e", "f"]]], "columns": ["asset_id"]},
            {"start": "2021-01-02", "end": "2021-01-02"},
            ["e"],
        ),
    ],
)
def test_shared_operation_input_selection_preserves_boundaries(
    tmp_path, selected, operation, expected
):
    from quant_research_platform.contracts.time import PredictionWindow

    root = make_dataset_artifact_root(tmp_path)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": list("abcdef"),
            "value": range(6),
            "timestamp": pd.to_datetime(
                [
                    "2019-01-01",
                    "2020-01-01",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-02T23:59:59.999999999",
                    "2021-01-03",
                ],
                format="mixed",
            ),
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec())
    (root / "data/year=2020/unselected.parquet").write_bytes(b"unselected defect")
    options = dataset_reading._dataset_input_options(
        {"path": root, **selected}, time_window=PredictionWindow(**operation)
    )
    result = dataset_reading.materialize_dataset_session(
        dataset_reading.open_dataset_session(**options)
    )
    assert result.asset_id.tolist() == expected
    if "columns" in selected:
        assert result.columns.tolist() == ["asset_id"]


def test_disjoint_discovery_skips_year_gap_without_bypassing_input_validation(tmp_path):
    root = make_dataset_artifact_root(tmp_path)
    frame = make_default_dataset_frame().copy()
    frame["timestamp"] = pd.to_datetime(["2019-01-02", "2020-01-02", "2021-01-02"])
    write_dataset(root, frame, spec=make_processed_dataset_spec())
    (root / "data/year=2020/unselected.parquet").write_bytes(b"unselected defect")
    windows = (
        TimeWindow("2019-01-02", "2019-01-02"),
        TimeWindow("2021-01-02", "2021-01-02"),
    )
    options = dataset_reading._dataset_input_options(root, discovery_windows=windows)
    session = dataset_reading.open_dataset_session(**options)
    assert {path.parent.name for path in session.read_paths} == {
        "year=2019",
        "year=2021",
    }
    actual = pd.concat(
        [
            dataset_reading.materialize_dataset_session(session.filtered_view(window))
            for window in windows
        ]
    )
    assert actual.timestamp.tolist() == [
        pd.Timestamp("2019-01-02"),
        pd.Timestamp("2021-01-02"),
    ]
    selected = {
        "path": root,
        "time_window": {"start": "2021-01-01", "end": "2021-01-01"},
    }
    for fields, error in [
        ({"years": [True]}, "dataset_years_invalid"),
        ({"columns": ["absent"]}, "dataset_column_not_found"),
        ({"row_filters": [["absent", "==", 1]]}, "dataset_filter_column_not_found"),
        ({"files": ["../outside.parquet"]}, "dataset_files_invalid"),
    ]:
        with pytest.raises(ValueError, match=error):
            dataset_reading.open_dataset_session(
                **dataset_reading._dataset_input_options(
                    {**selected, **fields}, time_window=windows[0]
                )
            )
    (root / "data/year=2021/selected.parquet").write_bytes(b"selected defect")
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        dataset_reading.open_dataset_session(**options)


def test_repeated_day_views_open_only_matching_native_payloads(tmp_path, monkeypatch):
    from collections import Counter

    from quant_research_platform.storage import parquet

    root = tmp_path / "daily"
    days = pd.date_range("2024-01-01", periods=6)
    frame = pd.DataFrame(
        {"market_id": "a_share", "asset_id": "a", "timestamp": days, "value": range(6)}
    )
    write_dataset(
        root,
        (frame.iloc[i : i + 1] for i in range(len(frame))),
        spec=make_processed_dataset_spec("daily"),
    )
    session = dataset_reading.open_dataset_session(root)
    opened = Counter()
    original = parquet.open_filesystem_file

    def measured(path, **kwargs):
        opened[Path(path)] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(parquet, "open_filesystem_file", measured)
    actual = pd.concat(
        [
            dataset_reading.materialize_dataset_session(
                session.filtered_view(TimeWindow(day, day))
            )
            for day in days
        ],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(actual, frame.astype({"timestamp": "datetime64[ns]"}))
    assert len(opened) == len(days)
    assert set(opened.values()) == {1}
