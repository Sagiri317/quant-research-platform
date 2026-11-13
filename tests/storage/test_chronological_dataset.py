"""Chronological reads retain selected values and keep member-order reads intact."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_time_chunks,
    open_dataset_session,
)


@pytest.mark.parametrize("batch_rows", [2, 10])
@pytest.mark.parametrize("one_timestamp", [False, True])
def test_chronological_sort_retains_native_dtypes_values_and_selection(
    tmp_path, batch_rows, one_timestamp, monkeypatch
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03", "2024-01-01", "2024-01-02"] * 2
            ).as_unit("ns"),
            "asset_id": ["a"] * 3 + ["b"] * 3,
            "quantity": pd.Series(
                [2**63 + 1, None, 2**63 + 2, 2**63 + 3, 0, 2**63 + 4], dtype="UInt64"
            ),
            "observed": pd.Series([True, False, None] * 2, dtype="boolean"),
            "score": pd.Series([1.0, None, 3.0, 4.0, 5.0, 6.0], dtype="Float32"),
            "industry": pd.Categorical(["x", "y", "x"] * 2, categories=["x", "y", "z"]),
        }
    )
    frame["score"] = pd.arrays.FloatingArray(
        np.array([np.nan, 0, 3, 4, 5, 6], dtype="float32"),
        np.array([False, True, False, False, False, False]),
    )
    roles = {name: "context" for name in frame}
    roles.update(timestamp="timestamp", asset_id="key")
    spec = DataSpec(
        data_id="unordered",
        frequency="1d",
        grain="asset_timestamp",
        column_roles=roles,
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
    )
    root = tmp_path / "data"
    write_dataset(root, [frame.iloc[:3], frame.iloc[3:]], spec=spec)
    pd.testing.assert_frame_equal(read_dataset(root), frame)
    columns = ("timestamp", "asset_id", "quantity", "score", "industry")
    session = open_dataset_session(
        root, time_window=TimeWindow("2024-01-02", "2024-01-03"), columns=columns
    )
    from quant_research_platform.storage.datasets import reading

    consumed = []
    original = reading.iter_dataset_file_batches

    def observe(*args, **kwargs):
        for path, chunk in original(*args, **kwargs):
            consumed.extend(zip(chunk.timestamp, chunk.asset_id, strict=True))
            yield path, chunk

    monkeypatch.setattr(reading, "iter_dataset_file_batches", observe)
    chunks = list(
        iter_dataset_time_chunks(
            session, batch_rows=batch_rows, one_timestamp=one_timestamp
        )
    )
    assert len(consumed) == len(set(consumed)) == 4
    actual = pd.concat(chunks, ignore_index=True)
    expected = (
        frame.loc[frame.timestamp.ge("2024-01-02"), list(columns)]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
    seen = set()
    for chunk in chunks:
        assert chunk.timestamp.is_monotonic_increasing
        current = set(chunk.timestamp)
        if one_timestamp:
            assert len(current) == 1
        assert not seen & current
        seen.update(current)
        assert len(chunk) <= batch_rows + 2


def test_native_time_sorted_hot_path_needs_no_temporary_sort(tmp_path, monkeypatch):
    import sqlite3

    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(
                pd.date_range("2024-01-01", periods=4).as_unit("ns"), 2
            ),
            "asset_id": ["a", "b"] * 4,
        }
    )
    spec = DataSpec(
        data_id="sorted",
        frequency="1d",
        grain="asset_timestamp",
        column_roles={"timestamp": "timestamp", "asset_id": "key"},
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
    )
    write_dataset(tmp_path / "data", [frame.iloc[:4], frame.iloc[4:]], spec=spec)

    def unexpected(*args, **kwargs):
        pytest.fail("ordered bounded row groups must stay on the streaming path")

    monkeypatch.setattr(sqlite3, "connect", unexpected)
    actual = pd.concat(
        iter_dataset_time_chunks(open_dataset_session(tmp_path / "data")),
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(actual, frame)


@pytest.mark.parametrize(
    "window",
    [
        TimeWindow("2024-01-03", "2024-01-03"),
        TimeWindow("2024-01-03T01:30:00", "2024-01-03T01:30:03"),
        TimeWindow("2024-01-04", "2024-01-04"),
    ],
)
def test_window_excluded_unordered_groups_do_not_require_sort(
    tmp_path, monkeypatch, window
):
    import sqlite3

    from quant_research_platform.contracts.time import prediction_window_mask

    times = pd.to_datetime(
        [
            "2024-01-02",
            "2024-01-01",
            "2024-01-03T01:29:59.999999999",
            "2024-01-03T01:30:00",
            "2024-01-03T01:30:03",
            "2024-01-03T01:30:03.000000001",
        ],
        format="mixed",
    ).as_unit("ns")
    frame = pd.DataFrame({"timestamp": times, "asset_id": "a"})
    spec = DataSpec(
        data_id="selected_order",
        frequency="tick",
        grain="asset_timestamp",
        column_roles={"timestamp": "timestamp", "asset_id": "key"},
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
    )
    root = tmp_path / "data"
    write_dataset(root, [frame.iloc[:1], frame.iloc[1:2], frame.iloc[2:]], spec=spec)

    def unexpected(*args, **kwargs):
        pytest.fail("groups outside this selection cannot require a temporary sort")

    monkeypatch.setattr(sqlite3, "connect", unexpected)
    actual = pd.concat(
        iter_dataset_time_chunks(open_dataset_session(root, time_window=window)),
        ignore_index=True,
    )
    expected = frame.loc[prediction_window_mask(frame.timestamp, window)].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("statistics", [True, False])
def test_matching_unordered_or_unknown_statistics_still_require_sort(
    tmp_path, monkeypatch, statistics
):
    import sqlite3

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-03T01:30:03", "2024-01-03T01:30:00"]
            ).as_unit("ns"),
            "asset_id": ["a", "a"],
        }
    )
    path = tmp_path / "unordered.parquet"
    frame.to_parquet(path, index=False, row_group_size=1, write_statistics=statistics)
    spec = DataSpec(
        data_id="external_order",
        frequency="tick",
        grain="asset_timestamp",
        column_roles={"timestamp": "timestamp", "asset_id": "key"},
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
    )
    calls = []
    connect = sqlite3.connect

    def observed(*args, **kwargs):
        calls.append(True)
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed)
    actual = pd.concat(
        iter_dataset_time_chunks(
            open_dataset_session(
                path, spec=spec, time_window=TimeWindow("2024-01-03", "2024-01-03")
            )
        ),
        ignore_index=True,
    )
    assert len(calls) == 1
    pd.testing.assert_frame_equal(
        actual, frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    )
