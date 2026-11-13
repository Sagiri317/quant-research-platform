from __future__ import annotations

from pathlib import Path

import pytest

import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
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
