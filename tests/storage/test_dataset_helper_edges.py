from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from quant_research_platform.storage._dataset_manifest_validation import (
    validate_manifest_runtime_metadata_shape,
)
from quant_research_platform.storage._dataset_paths import (
    candidate_data_files,
    partition_frames,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_reading import (
    filter_frame_by_time_window,
    read_columns_for_time_window,
)
from quant_research_platform.storage._dataset_schema import (
    data_spec_payload,
    validated_schema,
)
from quant_research_platform.storage._dataset_writing import _WriteStats
from tests.storage.low_level_fixtures import (
    make_dataset_frame,
    make_processed_dataset_spec,
)


def test_write_stats_fail_closed_paths() -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    stats = _WriteStats(spec=spec, data_root=Path("data"))

    with pytest.raises(ValueError, match="unregistered_output_columns"):
        stats.update_frame(
            make_dataset_frame(scenario="helper_edges", market_id="dummy").assign(
                extra=1
            )
        )
    with pytest.raises(ValueError, match="missing_required_columns"):
        stats.update_frame(
            make_dataset_frame(scenario="helper_edges", market_id="dummy").drop(
                columns=["value"]
            )
        )
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        frame = make_dataset_frame(scenario="helper_edges", market_id="dummy")
        stats.update_frame(pd.concat([frame, frame.iloc[[0]]]))

    with pytest.raises(ValueError, match="dataset_schema_missing"):
        _WriteStats(spec=spec, data_root=Path("data")).payload()


def test_write_stats_tracks_null_and_non_finite_failures() -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")

    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        _WriteStats(spec=spec, data_root=Path("data")).update_frame(
            make_dataset_frame(scenario="helper_edges", market_id="dummy").assign(
                asset_id=["asset_a", None]
            )
        )
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        _WriteStats(spec=spec, data_root=Path("data")).update_frame(
            make_dataset_frame(scenario="helper_edges", market_id="dummy").assign(
                value=[1.0, float("inf")]
            )
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
        filter_frame_by_time_window(
            make_dataset_frame(scenario="helper_edges", market_id="dummy"),
            spec=spec_payload,
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    with pytest.raises(ValueError, match="time_window_timestamp_column_not_selected"):
        filter_frame_by_time_window(
            make_dataset_frame(scenario="helper_edges", market_id="dummy").drop(
                columns=["timestamp"]
            ),
            spec=data_spec_payload(
                make_processed_dataset_spec("storage_demo", market_id="dummy")
            ),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    assert filter_frame_by_time_window(
        make_dataset_frame(scenario="helper_edges", market_id="dummy"),
        spec=data_spec_payload(
            make_processed_dataset_spec("storage_demo", market_id="dummy")
        ),
        time_window=TimeWindow("2025-01-03", "2025-01-31"),
    ).empty


def test_dataset_partition_and_manifest_shape_helpers(tmp_path: Path) -> None:
    frame = make_dataset_frame(scenario="helper_edges", market_id="dummy")

    assert list(
        partition_frames(
            frame.iloc[0:0],
            timestamp_column="timestamp",
            partition_columns=("year",),
        )
    ) == []
    assert list(
        partition_frames(
            frame,
            timestamp_column="timestamp",
            partition_columns=(),
        )
    )[0][0] == ""
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        list(
            partition_frames(
                frame,
                timestamp_column="timestamp",
                partition_columns=("month",),
            )
    )
    with pytest.raises(ValueError, match="missing_partition_timestamp_column"):
        list(
            partition_frames(
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
    assert validate_manifest_runtime_metadata_shape(manifest)[0] == 1
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
            validate_manifest_runtime_metadata_shape(bad_manifest)


def test_dataset_timestamp_and_candidate_file_helpers(tmp_path: Path) -> None:
    spec = data_spec_payload(
        make_processed_dataset_spec("storage_demo", market_id="dummy")
    )
    assert read_columns_for_time_window(
        ("market_id",),
        spec=spec,
        time_window=TimeWindow("2024-01-01", "2024-01-31"),
    ) == ("market_id", "timestamp")
    with pytest.raises(ValueError, match="invalid_column_roles"):
        timestamp_column_from_spec_payload({"column_roles": []})
    with pytest.raises(ValueError, match="invalid_timestamp_primary_key"):
        timestamp_column_from_spec_payload(
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
    assert candidate_data_files(
        data_root,
        spec={**spec, "partition_columns": []},
    ) == []
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        candidate_data_files(
            data_root,
            spec={**spec, "partition_columns": ["month"]},
        )
    with pytest.raises(ValueError, match="dataset_data_dir_missing"):
        validated_schema([])
