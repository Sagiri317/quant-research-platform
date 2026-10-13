"""Explicit paths and selected-input reads without physical snapshot admission."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.atomic import atomic_write_bytes
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_explicit_output_path_is_atomic_and_non_overwriting(tmp_path):
    root = tmp_path / "chosen-result"
    frame = make_default_dataset_frame()
    write_dataset(root, frame, spec=make_processed_dataset_spec())
    expected = read_dataset(root)
    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(
            root, frame.assign(value=10.0), spec=make_processed_dataset_spec()
        )
    pd.testing.assert_frame_equal(read_dataset(root), expected)
    assert not list(tmp_path.glob(".a-*"))


def test_window_does_not_open_unselected_footer_or_hash_bytes(tmp_path, monkeypatch):
    root = tmp_path / "chosen-result"
    frame = make_default_dataset_frame()
    write_dataset(root, frame, spec=make_processed_dataset_spec())
    excluded = root / "data" / "year=2014" / "part-000.parquet"
    excluded.write_bytes(b"not parquet")
    import quant_research_platform.storage.identity as identity

    monkeypatch.setattr(
        identity,
        "_hash_file_content",
        lambda *a, **k: pytest.fail("default full-file hash"),
    )
    actual = read_dataset(
        root, columns=("value",), time_window=TimeWindow("2015-01-01", "2015-12-31")
    )
    assert list(actual.columns) == ["value"]
    assert len(actual) == 2
    with pytest.raises((ValueError, pa.ArrowInvalid)):
        read_dataset(root, time_window=TimeWindow("2014-01-01", "2014-12-31"))


@pytest.mark.parametrize("mode", list(DatasetReadMode))
def test_bare_parquet_requires_roles_and_projects_nanosecond_window(
    tmp_path: Path, mode
):
    path = tmp_path / "supplied.parquet"
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta([0, 1, 2], unit="ns")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": pd.Series([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype="uint64"),
            "timestamp": times,
            "value": [1.0, 2.0, 3.0],
        }
    )
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    with pytest.raises(ValueError, match="parquet_semantic_spec_required"):
        read_dataset(path)
    result = read_dataset(
        path,
        spec=make_processed_dataset_spec(),
        columns=("value", "asset_id"),
        time_window=TimeWindow(times[1], times[1]),
        mode=mode,
    )
    actual = (
        pd.concat(list(result), ignore_index=True)
        if mode is DatasetReadMode.CHUNKED
        else result
    )
    assert actual.columns.tolist() == ["value", "asset_id"]
    assert actual.value.tolist() == [2.0]
    assert actual.asset_id.tolist() == [2**63 + 2]


def test_atomic_binary_publication_does_not_replace(tmp_path):
    path = tmp_path / "figure.png"
    atomic_write_bytes(path, b"first")
    with pytest.raises(ValueError, match="output_already_exists"):
        atomic_write_bytes(path, b"second")
    assert path.read_bytes() == b"first"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("dtype", ["Float64", "Int64", "boolean", "category"])
@pytest.mark.parametrize("mode", list(DatasetReadMode))
def test_empty_bare_projection_preserves_declared_pandas_dtype(tmp_path, dtype, mode):
    frame = make_default_dataset_frame().iloc[:2].copy()
    values = [True, False] if dtype == "boolean" else [1, 2]
    frame["value"] = pd.Series(values, dtype=dtype, index=frame.index)
    path = tmp_path / "bare.parquet"
    frame.to_parquet(path, index=False)
    result = read_dataset(
        path,
        spec=make_processed_dataset_spec(),
        columns=("value",),
        time_window=TimeWindow("2090-01-01", "2090-01-01"),
        mode=mode,
    )
    actual = (
        pd.concat(list(result), ignore_index=True)
        if mode is DatasetReadMode.CHUNKED
        else result
    )
    expected = (
        read_dataset(path, spec=make_processed_dataset_spec(), columns=("value",))
        .iloc[:0]
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected)
