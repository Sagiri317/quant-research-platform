from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from tests.research.test_research_dataset_contract_closure import _processed_spec


def test_write_stats_fail_closed_paths() -> None:
    spec = _storage_spec()
    stats = datasets._WriteStats(spec=spec, data_root=Path("data"))

    with pytest.raises(ValueError, match="unregistered_output_columns"):
        stats.update_frame(_frame().assign(extra=1))
    with pytest.raises(ValueError, match="missing_required_columns"):
        stats.update_frame(_frame().drop(columns=["value"]))
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        stats.update_frame(pd.concat([_frame(), _frame().iloc[[0]]]))

    with pytest.raises(ValueError, match="dataset_schema_missing"):
        datasets._WriteStats(spec=spec, data_root=Path("data")).payload()


def test_write_stats_tracks_null_and_non_finite_failures() -> None:
    spec = _storage_spec()

    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        datasets._WriteStats(spec=spec, data_root=Path("data")).update_frame(
            _frame().assign(asset_id=["asset_a", None])
        )
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        datasets._WriteStats(spec=spec, data_root=Path("data")).update_frame(
            _frame().assign(value=[1.0, float("inf")])
        )


def test_dataset_compare_and_filter_fail_closed_paths(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"

    assert datasets.compare_datasets(left, right).status == "MISSING_LEFT"
    left.mkdir()
    assert datasets.compare_datasets(left, right).status == "MISSING_RIGHT"

    spec_payload = {
        "primary_key": ["market_id", "asset_id"],
        "column_roles": {
            "market_id": ColumnRole.KEY.value,
            "asset_id": ColumnRole.KEY.value,
        },
    }
    with pytest.raises(ValueError, match="time_window_not_supported"):
        datasets._filter_frame_by_time_window(
            _frame(),
            spec=spec_payload,
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    with pytest.raises(ValueError, match="time_window_timestamp_column_not_selected"):
        datasets._filter_frame_by_time_window(
            _frame().drop(columns=["timestamp"]),
            spec=datasets._data_spec_payload(_storage_spec()),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    assert datasets._filter_frame_by_time_window(
        _frame(),
        spec=datasets._data_spec_payload(_storage_spec()),
        time_window=TimeWindow("2025-01-03", "2025-01-31"),
    ).empty


def test_dataset_partition_and_manifest_shape_helpers(tmp_path: Path) -> None:
    frame = _frame()

    assert list(
        datasets._partition_frames(
            frame.iloc[0:0],
            timestamp_column="timestamp",
            partition_columns=("year",),
        )
    ) == []
    assert list(
        datasets._partition_frames(
            frame,
            timestamp_column="timestamp",
            partition_columns=(),
        )
    )[0][0] == ""
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        list(
            datasets._partition_frames(
                frame,
                timestamp_column="timestamp",
                partition_columns=("month",),
            )
    )
    with pytest.raises(ValueError, match="missing_partition_timestamp_column"):
        list(
            datasets._partition_frames(
                frame.drop(columns=["timestamp"]),
                timestamp_column="timestamp",
                partition_columns=("year",),
            )
        )

    manifest = {
        "columns": ["market_id"],
        "schema_hash": "abc",
        "file_count": 1,
        "row_count": 2,
        "partition_row_counts": {"year=2024": 2},
    }
    assert datasets._validate_manifest_runtime_metadata_shape(manifest)[0] == 1
    for bad_manifest, error in (
        ({**manifest, "columns": []}, "manifest_columns_invalid"),
        ({**manifest, "schema_hash": ""}, "manifest_schema_hash_invalid"),
        ({**manifest, "file_count": "bad"}, "manifest_counts_invalid"),
        ({**manifest, "row_count": -1}, "manifest_counts_invalid"),
        (
            {**manifest, "partition_row_counts": {}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"": 1}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"year=2024": "bad"}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"year=2024": -1}},
            "manifest_partition_row_counts_invalid",
        ),
        ({**manifest, "row_count": 3}, "manifest_row_count_mismatch"),
    ):
        with pytest.raises(ValueError, match=error):
            datasets._validate_manifest_runtime_metadata_shape(bad_manifest)


def test_dataset_timestamp_and_candidate_file_helpers(tmp_path: Path) -> None:
    spec = datasets._data_spec_payload(_storage_spec())
    assert datasets._read_columns_for_time_window(
        ("market_id",),
        spec=spec,
        time_window=TimeWindow("2024-01-01", "2024-01-31"),
    ) == ("market_id", "timestamp")
    with pytest.raises(ValueError, match="invalid_column_roles"):
        datasets._timestamp_column_from_spec_payload({"column_roles": []})
    with pytest.raises(ValueError, match="invalid_timestamp_primary_key"):
        datasets._timestamp_column_from_spec_payload(
            {
                "primary_key": ["timestamp", "as_of"],
                "column_roles": {
                    "timestamp": ColumnRole.TIMESTAMP.value,
                    "as_of": ColumnRole.TIMESTAMP.value,
                },
            }
        )

    data_root = tmp_path / "data"
    data_root.mkdir()
    assert datasets._candidate_data_files(
        data_root,
        spec={**spec, "partition_columns": []},
    ) == []
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        datasets._candidate_data_files(
            data_root,
            spec={**spec, "partition_columns": ["month"]},
        )
    with pytest.raises(ValueError, match="dataset_data_dir_missing"):
        datasets._validated_schema([])


def _storage_spec():
    return _processed_spec(
        "storage_demo",
        ProcessedDataKind.FACTOR,
        ("market_id", "asset_id", "timestamp", "value"),
        role=ColumnRole.FACTOR,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["dummy", "dummy"],
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "value": [1.0, 2.0],
        }
    )
