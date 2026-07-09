from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_inspection import (
    _inspection_runtime_metadata,
)
from quant_research_platform.storage._dataset_reading import DatasetReadContext
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
    make_processed_dataset_spec,
    make_time_window_dataset_frame,
)


def capture_read_parquet_paths(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> list[Path]:
    original_read_parquet = pd.read_parquet
    read_paths: list[Path] = []

    def record_read_parquet(path, *args, **kwargs):
        read_paths.append(Path(path))
        return original_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(target, record_read_parquet)
    return read_paths


def test_read_dataset_time_window_uses_closed_interval(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "windowed")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("windowed"),
    )

    result = read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2014-01-02", "2015-01-05"),
    )

    assert list(result["asset_id"]) == ["000001", "000001"]
    assert list(result["value"]) == [1.0, 2.0]


def test_read_dataset_time_window_prunes_year_partitions_and_keeps_output_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "read_pruned")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("read_pruned"),
    )
    full = read_dataset(root)
    read_paths = capture_read_parquet_paths(
        monkeypatch,
        "quant_research_platform.storage._dataset_reading.pd.read_parquet",
    )

    result = read_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-01-05"),
    )
    manual = full.loc[
        (full["timestamp"].dt.date >= pd.Timestamp("2015-01-01").date())
        & (full["timestamp"].dt.date <= pd.Timestamp("2015-01-05").date()),
        ["asset_id", "value"],
    ].reset_index(drop=True)

    assert list(result.columns) == ["asset_id", "value"]
    pd.testing.assert_frame_equal(result, manual)
    assert {path.parent.name for path in read_paths} == {"year=2015"}


def test_read_dataset_datetime_time_window_filters_intraday_precisely(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "intraday_window")
    frame = pd.DataFrame(
        {
            "market_id": ["crypto"] * 4,
            "asset_id": ["btc_usdt"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00:00",
                    "2026-07-06 00:01:00",
                    "2026-07-06 00:02:00",
                    "2026-07-06 00:03:00",
                ]
            ),
            "value": [0.0, 1.0, 2.0, 3.0],
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("intraday_window"))

    result = read_dataset(
        root,
        columns=["timestamp", "value"],
        time_window=TimeWindow("2026-07-06T00:01:00", "2026-07-06T00:02:00"),
    )

    assert result["value"].tolist() == [1.0, 2.0]


def test_date_window_includes_subsecond_end_of_day(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "subsecond_end")
    frame = pd.DataFrame(
        {
            "market_id": ["crypto", "crypto"],
            "asset_id": ["btc_usdt", "btc_usdt"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 23:59:59.999999", "2026-07-07 00:00:00"],
                format="mixed",
            ),
            "value": [1.0, 2.0],
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("subsecond_end"))

    result = read_dataset(
        root,
        columns=["timestamp", "value"],
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
    )

    assert result["value"].tolist() == [1.0]


def test_time_window_normalizes_timezone_aware_bounds_to_naive_utc(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "timezone_window")
    frame = pd.DataFrame(
        {
            "market_id": ["crypto", "crypto"],
            "asset_id": ["btc_usdt", "btc_usdt"],
            "timestamp": pd.to_datetime(
                ["2026-07-05 23:59:00", "2026-07-06 00:00:00"]
            ),
            "value": [1.0, 2.0],
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("timezone_window"))

    result = read_dataset(
        root,
        columns=["timestamp", "value"],
        time_window=TimeWindow(
            "2026-07-06T07:59:00+08:00",
            "2026-07-06T08:00:00+08:00",
        ),
    )

    assert result["value"].tolist() == [1.0, 2.0]


def test_time_window_invalid_string_raises_stable_error() -> None:
    with pytest.raises(ValueError, match="invalid_time_window_bound"):
        TimeWindow("not-a-date", "2026-07-06")


def test_chunked_datetime_window_preserves_empty_schema(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_datetime_chunked")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("empty_datetime_chunked"),
    )

    chunks = list(
        read_dataset(
            root,
            columns=["asset_id", "value"],
            time_window=TimeWindow(
                "2017-01-01T00:00:00",
                "2017-01-01T00:01:00",
            ),
            mode=DatasetReadMode.CHUNKED,
        )
    )

    assert len(chunks) == 1
    assert list(chunks[0].columns) == ["asset_id", "value"]
    assert chunks[0].empty


def test_inspect_dataset_time_window_reports_window_scoped_counts(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "inspect")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("inspect"),
    )

    inspection = inspect_dataset(
        root,
        time_window=TimeWindow("2015-01-01", "2015-12-31"),
    )

    assert inspection.row_count == 1
    assert inspection.date_min == "2015-01-05"
    assert inspection.date_max == "2015-01-05"
    assert inspection.file_count == 1
    assert inspection.partitions == ("year=2015",)
    assert inspection.partition_row_counts == {"year=2015": 1}


def test_inspection_runtime_metadata_rejects_invalid_timestamp_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    path = data_root / "part-000.parquet"
    context = DatasetReadContext(
        root=tmp_path,
        data_root=data_root,
        spec={
            "schema_id": "qrp-dataset-spec",
            "primary_key": ["market_id", "asset_id", "timestamp"],
            "column_roles": {
                "market_id": "key",
                "asset_id": "key",
                "timestamp": "timestamp",
                "value": "feature",
            },
        },
        manifest={},
        all_paths=(path,),
        read_paths=(path,),
        logical_columns=("market_id", "asset_id", "timestamp", "value"),
        physical_columns=("timestamp",),
        timestamp_column="timestamp",
        time_window=TimeWindow("2024-01-01", "2024-01-31"),
        schema=(("timestamp", "string"),),
        schema_hash="hash",
        parquet_filters=None,
    )

    monkeypatch.setattr(
        "quant_research_platform.storage._dataset_inspection.pd.read_parquet",
        lambda *_, **__: pd.DataFrame({"timestamp": ["2024-01-02", "bad-date"]}),
    )
    with pytest.raises(ValueError, match="invalid_time_window_timestamp_value"):
        _inspection_runtime_metadata(context)


def test_fingerprint_dataset_time_window_hashes_window_only(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("fingerprint"),
    )

    full = fingerprint_dataset(root)
    windowed = fingerprint_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-12-31"),
    )

    assert full.row_count == 3
    assert windowed.row_count == 1
    assert windowed.column_names == ("asset_id", "value")
    assert windowed.content_hash != full.content_hash


def test_fingerprint_and_inspect_time_window_file_count_match_contributing_files(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_window")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("empty_window"),
    )

    window = TimeWindow("2015-02-01", "2015-02-02")
    inspection = inspect_dataset(root, time_window=window)
    fingerprint = fingerprint_dataset(root, time_window=window)

    assert inspection.row_count == 0
    assert fingerprint.row_count == 0
    assert inspection.file_count == 0
    assert fingerprint.file_count == inspection.file_count


def test_chunked_read_time_window_without_matching_partitions_yields_empty_schema(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_chunked")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("empty_chunked"),
    )

    chunks = list(
        read_dataset(
            root,
            columns=["asset_id", "value"],
            time_window=TimeWindow("2017-01-01", "2017-01-02"),
            mode=DatasetReadMode.CHUNKED,
        )
    )

    assert len(chunks) == 1
    assert list(chunks[0].columns) == ["asset_id", "value"]
    assert chunks[0].empty


def test_fingerprint_dataset_time_window_prunes_year_partitions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_pruned")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("fingerprint_pruned"),
    )
    read_paths = capture_read_parquet_paths(
        monkeypatch,
        "quant_research_platform.storage._dataset_fingerprinting.pd.read_parquet",
    )

    fingerprint = fingerprint_dataset(
        root,
        columns=["asset_id", "value"],
        time_window=TimeWindow("2015-01-01", "2015-01-05"),
    )

    assert fingerprint.row_count == 1
    assert fingerprint.partition_fingerprints[0]["partition"] == "year=2015"
    assert {path.parent.name for path in read_paths} == {"year=2015"}


def test_compare_datasets_time_window_compares_window_only(tmp_path: Path) -> None:
    left = make_dataset_artifact_root(tmp_path, "left")
    right = make_dataset_artifact_root(tmp_path, "right")
    write_dataset(
        left,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("left"),
    )
    changed = make_time_window_dataset_frame()
    changed.loc[changed["timestamp"] == pd.Timestamp("2016-01-04"), "value"] = 99.0
    write_dataset(right, changed, spec=make_processed_dataset_spec("right"))

    comparison = compare_datasets(
        left,
        right,
        columns=["asset_id", "timestamp", "value"],
        time_window=TimeWindow("2014-01-01", "2015-12-31"),
    )

    assert comparison.matches is True
    assert comparison.status == "MATCH"


def test_compare_datasets_time_window_detects_window_differences(
    tmp_path: Path,
) -> None:
    left = make_dataset_artifact_root(tmp_path, "left_window_diff")
    right = make_dataset_artifact_root(tmp_path, "right_window_diff")
    write_dataset(
        left,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("left_window_diff"),
    )
    changed = make_time_window_dataset_frame()
    changed.loc[changed["timestamp"] == pd.Timestamp("2015-01-05"), "value"] = 99.0
    write_dataset(
        right,
        changed,
        spec=make_processed_dataset_spec("right_window_diff"),
    )

    comparison = compare_datasets(
        left,
        right,
        columns=["asset_id", "timestamp", "value"],
        time_window=TimeWindow("2015-01-01", "2015-01-05"),
    )

    assert comparison.matches is False
    assert comparison.status == "CONTENT_MISMATCH"


def test_time_window_without_timestamp_role_raises(tmp_path: Path) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "annotation" / "reference"
    spec = ProcessedDataSpec(
        data_id="reference",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.ANNOTATION,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "name": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id"),
        partition_columns=(),
    )
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["000001"],
                "name": ["demo"],
            }
        ),
        spec=spec,
    )

    with pytest.raises(ValueError, match="time_window_not_supported"):
        read_dataset(root, time_window=TimeWindow("2014-01-01", "2014-12-31"))


def test_write_dataset_existing_root_raises_dataset_exists(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "exists")
    write_dataset(
        root,
        make_time_window_dataset_frame(),
        spec=make_processed_dataset_spec("exists"),
    )

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(
            root,
            make_time_window_dataset_frame(),
            spec=make_processed_dataset_spec("exists"),
        )
