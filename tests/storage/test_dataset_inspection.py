from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_session_consuming_inspection_does_not_reopen_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    monkeypatch.setattr(
        dataset_inspection,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("session_inspection_reopened_dataset")
        ),
    )
    monkeypatch.setattr(
        dataset_inspection,
        "iter_dataset_file_batches",
        lambda *_args, **_kwargs: pytest.fail("published counts were rescanned"),
    )
    inspection = dataset_inspection.inspect_dataset_session(session)
    assert inspection.root == root
    assert inspection.row_count == 3
    assert inspection.file_count == 2


def test_session_inspection_counts_only_selected_members_and_rows(tmp_path):
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    selected_file = next((root / "data" / "year=2015").glob("*.parquet"))
    for options in (
        {"years": [2015]},
        {"files": [selected_file.relative_to(root).as_posix()]},
    ):
        session = dataset_reading.open_dataset_session(root, **options)
        inspection = dataset_inspection.inspect_dataset_session(session)
        assert (inspection.row_count, inspection.file_count) == (2, 1)
        assert inspection.partition_row_counts == {"year=2015": 2}
        assert inspection.date_min == inspection.date_max == "2015-01-05"

    selected_asset = dataset_reading.open_dataset_session(
        root, row_filters=(("asset_id", "==", "asset_002"),)
    )
    view = selected_asset.filtered_view(TimeWindow("2014-01-01", "2015-01-04"))
    for session, expected in ((selected_asset, 1), (view, 0)):
        inspection = dataset_inspection.inspect_dataset_session(session)
        assert (inspection.row_count, inspection.file_count) == (expected, expected)
        assert inspection.partitions == (("year=2015",) if expected else ())
        assert (
            inspection.date_min
            == inspection.date_max
            == ("2015-01-05" if expected else None)
        )
