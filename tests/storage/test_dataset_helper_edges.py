from __future__ import annotations

import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage._dataset_comparison as dataset_comparison
import quant_research_platform.storage._dataset_key_tracking as key_tracking
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from quant_research_platform.storage._dataset_key_tracking import (
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_manifest_validation import (
    validate_manifest_runtime_metadata_shape,
)
from quant_research_platform.storage._dataset_paths import (
    _require_only_expected_parquet_paths,
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
    make_helper_edges_dataset_frame,
    make_processed_dataset_spec,
)


def test_exact_key_index_cleans_scratch_when_open_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        key_tracking.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
            scratch_parent=tmp_path,
        ).__enter__()

    assert tuple(tmp_path.iterdir()) == ()


def test_dataset_comparison_failure_summary_does_not_expose_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(
        dataset_comparison,
        "fingerprint_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(str(tmp_path))),
    )

    comparison = datasets.compare_datasets(left, right)

    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


def test_write_stats_fail_closed_paths(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as tracker:
        stats = _WriteStats(
            spec=spec,
            data_root=Path("data"),
            key_tracker=tracker,
        )

        with pytest.raises(ValueError, match="unregistered_output_columns"):
            stats.update_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    extra=1
                )
            )
        with pytest.raises(ValueError, match="missing_required_columns"):
            stats.update_frame(
                make_helper_edges_dataset_frame(market_id="dummy").drop(
                    columns=["value"]
                )
            )
        with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
            frame = make_helper_edges_dataset_frame(market_id="dummy")
            stats.update_frame(pd.concat([frame, frame.iloc[[0]]]))

        with pytest.raises(ValueError, match="dataset_schema_missing"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=tracker,
            ).payload()


def test_exact_index_preserves_decimal_values_beyond_context_precision(
    tmp_path: Path,
) -> None:
    schema = pa.schema([pa.field("key", pa.decimal128(38, 0))])
    first = Decimal("12345678901234567890123456789012345678")
    second = Decimal("12345678901234567890123456789012345679")

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        key_schema=schema,
    ) as index:
        assert index.add_table(pa.table({"key": [first]})) == 0
        assert index.add_table(pa.table({"key": [second]})) == 0


def test_exact_index_preserves_unclaimed_key_scratch_directory(
    tmp_path: Path,
) -> None:
    unclaimed = tmp_path / ".k-user-cache"
    unclaimed.mkdir()
    sentinel = unclaimed / "sentinel.txt"
    sentinel.write_text("unclaimed\n", encoding="utf-8")

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ):
        pass

    assert sentinel.read_text(encoding="utf-8") == "unclaimed\n"


def test_write_stats_tracks_null_and_non_finite_failures(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as null_tracker:
        with pytest.raises(ValueError, match="dataset_primary_key_null"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=null_tracker,
            ).update_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    asset_id=["asset_a", pd.NA]
                )
            )
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as numeric_tracker:
        with pytest.raises(ValueError, match="dataset_infinite_value"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=numeric_tracker,
            ).update_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    value=[1.0, float("inf")]
                )
            )


def test_write_stats_rejects_invalid_non_null_timestamp(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as tracker:
        with pytest.raises(ValueError, match="invalid_dataset_timestamp_value"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=tracker,
            ).update_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    timestamp=["2024-01-02", "bad-date"]
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
            make_helper_edges_dataset_frame(market_id="dummy"),
            spec=spec_payload,
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    with pytest.raises(ValueError, match="time_window_timestamp_column_not_selected"):
        filter_frame_by_time_window(
            make_helper_edges_dataset_frame(market_id="dummy").drop(
                columns=["timestamp"]
            ),
            spec=data_spec_payload(
                make_processed_dataset_spec("storage_demo", market_id="dummy")
            ),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    assert filter_frame_by_time_window(
        make_helper_edges_dataset_frame(market_id="dummy"),
        spec=data_spec_payload(
            make_processed_dataset_spec("storage_demo", market_id="dummy")
        ),
        time_window=TimeWindow("2025-01-03", "2025-01-31"),
    ).empty

    with pytest.raises(ValueError, match="invalid_time_window_timestamp_value"):
        filter_frame_by_time_window(
            make_helper_edges_dataset_frame(market_id="dummy").assign(
                timestamp=["2024-01-02", "bad-date"]
            ),
            spec=data_spec_payload(
                make_processed_dataset_spec("storage_demo", market_id="dummy")
            ),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )


def test_dataset_partition_and_manifest_shape_helpers(tmp_path: Path) -> None:
    frame = make_helper_edges_dataset_frame(market_id="dummy")

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


def test_candidate_data_files_orders_numeric_part_indices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for index in (1001, 999, 1000, 998):
        pq.write_table(
            pa.table({"value": [index]}),
            data_root / f"part-{index}.parquet",
        )

    paths = candidate_data_files(
        data_root,
        spec={"partition_columns": []},
    )

    assert [path.name for path in paths] == [
        "part-998.parquet",
        "part-999.parquet",
        "part-1000.parquet",
        "part-1001.parquet",
    ]


@pytest.mark.parametrize(
    "names",
    [
        ("part-invalid.parquet",),
        ("part-1.parquet", "part-001.parquet"),
        ("part-000.parquet", "stray.parquet"),
    ],
    ids=["invalid", "duplicate-index", "stray"],
)
def test_candidate_data_files_rejects_invalid_or_ambiguous_parts(
    tmp_path: Path,
    names: tuple[str, ...],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in names:
        pq.write_table(pa.table({"value": [1]}), data_root / name)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(
            data_root,
            spec={"partition_columns": []},
        )


def test_candidate_data_files_rejects_duplicate_numeric_years(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    for year_name in ("year=1", "year=01"):
        year_root = data_root / year_name
        year_root.mkdir(parents=True)
        pq.write_table(pa.table({"value": [1]}), year_root / "part-000.parquet")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(
            data_root,
            spec={"partition_columns": ["year"]},
        )


def test_candidate_data_files_rejects_resolved_path_outside_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.parquet"
    pq.write_table(pa.table({"value": [1]}), external)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _require_only_expected_parquet_paths(data_root, [external])


def test_candidate_data_files_rejects_symlinked_part_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    part = data_root / "part-000.parquet"
    pq.write_table(pa.table({"value": [1]}), part)
    alias = data_root / "part-001.parquet"
    try:
        alias.symlink_to(part)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _require_only_expected_parquet_paths(data_root, [part, alias])


def test_candidate_data_files_rejects_symlinked_partition_alias(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    year = data_root / "year=2024"
    year.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1]}), year / "part-000.parquet")
    alias = data_root / "year=2025"
    try:
        alias.symlink_to(year, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": ["year"]})


@pytest.mark.skipif(not hasattr(Path, "is_junction"), reason="junction API required")
def test_candidate_data_files_rejects_junction_data_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pq.write_table(pa.table({"value": [1]}), external / "part-000.parquet")
    data_root = tmp_path / "data"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(data_root), str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert data_root.is_junction()
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})


def test_candidate_data_files_rejects_hardlinked_part_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = data_root / "part-000.parquet"
    alias = data_root / "part-001.parquet"
    pq.write_table(pa.table({"value": [1]}), first)
    try:
        os.link(first, alias)
    except OSError:
        pytest.skip("hardlink_not_available")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})


def test_candidate_data_files_rejects_case_variant_parquet_suffix(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    pq.write_table(pa.table({"value": [1]}), data_root / "part-000.parquet")
    pq.write_table(pa.table({"value": [2]}), data_root / "part-001.PARQUET")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})
