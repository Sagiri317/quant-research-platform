from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_correctness_dataset_frame,
    make_dataset_artifact_root,
    make_processed_dataset_spec,
)


def test_storage_duplicate_primary_key_fails_closed(tmp_path: Path) -> None:
    frame = pd.concat(
        [
            make_correctness_dataset_frame(value_column="factor_value")
            .iloc[:1],
            make_correctness_dataset_frame(value_column="factor_value")
            .iloc[:1],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "correctness_demo"),
            frame,
            spec=make_processed_dataset_spec(
                "correctness_demo",
                value_column="factor_value",
            ),
        )


def test_storage_chunked_write_read_keeps_schema_keys_and_sort_stable(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "correctness_demo")
    frame = make_correctness_dataset_frame(value_column="factor_value")
    write_dataset(
        root,
        (frame.iloc[:2], frame.iloc[2:]),
        spec=make_processed_dataset_spec(
            "correctness_demo",
            value_column="factor_value",
        ),
    )

    result = read_dataset(root)
    expected = frame.copy()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")

    assert list(result.columns) == list(frame.columns)
    assert result.duplicated(["market_id", "asset_id", "timestamp"]).sum() == 0
    pd.testing.assert_frame_equal(result, expected)


def test_inspect_dataset_time_window_reports_only_scope_contributors(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "correctness_demo")
    write_dataset(
        root,
        make_correctness_dataset_frame(value_column="factor_value"),
        spec=make_processed_dataset_spec(
            "correctness_demo",
            value_column="factor_value",
        ),
    )

    inspection = inspect_dataset(
        root,
        time_window=TimeWindow("2020-01-03", "2020-01-06"),
    )

    assert inspection.row_count == 2
    assert inspection.date_min == "2020-01-06"
    assert inspection.date_max == "2020-01-06"
    assert inspection.file_count >= 1
