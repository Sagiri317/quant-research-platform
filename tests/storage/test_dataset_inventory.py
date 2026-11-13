from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import DatasetReadMode, write_dataset
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_rename,
    native_filesystem_path,
)
from tests.helpers.paths import (
    filesystem_glob,
    filesystem_write_bytes,
    filesystem_write_text,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_chunked_read_defers_session_inventory_until_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_session(*_args, **_kwargs):
        events.append("inventory")
        return Session()

    monkeypatch.setattr(dataset_reading, "open_dataset_session", open_session)
    monkeypatch.setattr(
        dataset_reading, "iter_dataset_chunks", lambda *_args, **_kwargs: iter(())
    )
    chunks = dataset_reading.read_dataset(Path("unused"), mode=DatasetReadMode.CHUNKED)
    assert events == []
    next(chunks, None)
    assert events == ["inventory"]


def test_dataset_session_inventory_preserves_numeric_part_order(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "inventory_numeric_order")
    base = make_default_dataset_frame()
    later = base.copy()
    later["timestamp"] = later["timestamp"] + pd.Timedelta(days=2)
    frame = pd.concat((base, later), ignore_index=True)
    write_dataset(
        root,
        tuple(frame.iloc[index : index + 1].copy() for index in range(len(frame))),
        spec=make_processed_dataset_spec("inventory_numeric_order"),
    )
    parts = sorted(filesystem_glob(root / "data", "**/part-*.parquet"))
    part_indices = tuple(range(999, 999 + len(parts)))
    for path, part_index in zip(parts, part_indices, strict=True):
        filesystem_rename(path, path.with_name(f"part-{part_index}.parquet"))
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    expected = [f"part-{part_index}.parquet" for part_index in part_indices]
    assert [path.name for path in session.all_paths] == expected
    assert isinstance(session.all_paths, tuple)
    assert session.all_paths[0].name == expected[0]
    assert [path.name for path in session.all_paths[1:]] == expected[1:]


def test_inspection_window_file_count_counts_consecutive_file_batches_once(tmp_path):
    root = tmp_path / "dataset"
    frame = make_default_dataset_frame()
    frame["timestamp"] = pd.date_range("2026-01-01", periods=len(frame))
    write_dataset(root, frame, spec=make_processed_dataset_spec("batches"))
    context = dataset_reading.open_dataset_session(
        root,
        time_window=TimeWindow("2026-01-01", "2026-01-31"),
    )
    runtime = dataset_inspection._inspection_runtime_metadata(context, batch_rows=1)
    assert runtime["file_count"] == 1
    assert runtime["row_count"] == 3


def test_private_writer_releases_result_when_publish_loses_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "publish_race")
    filesystem_mkdir(root.parent, parents=True, exist_ok=True)
    unrelated = root.parent / ".a-unrelated"
    filesystem_mkdir(unrelated)
    filesystem_write_bytes(unrelated / "old-good", b"old-good")
    before_staging = set(filesystem_glob(root.parent, ".a-*"))
    staging_paths = []
    real_atomic_write_directory = dataset_writing.atomic_write_directory

    def publish_after_writer(final_root, writer):

        def race_writer(temp_root):
            assert temp_root.parent == root.parent
            staging_paths.append(temp_root)
            result = writer(temp_root)
            filesystem_mkdir(final_root)
            filesystem_write_text(final_root / "winner", "winner", encoding="utf-8")
            return result

        return real_atomic_write_directory(final_root, race_writer)

    monkeypatch.setattr(dataset_writing, "atomic_write_directory", publish_after_writer)
    with pytest.raises(ValueError, match="dataset_exists"):
        dataset_writing.write_dataset(
            root,
            (make_default_dataset_frame(),),
            spec=make_processed_dataset_spec("publish_race"),
        )
    assert (
        Path(native_filesystem_path(root / "winner")).read_text(encoding="utf-8")
        == "winner"
    )
    assert (
        Path(native_filesystem_path(unrelated / "old-good")).read_bytes() == b"old-good"
    )
    assert set(filesystem_glob(root.parent, ".a-*")) == before_staging, (
        "publication_staging_not_cleaned"
    )
    assert staging_paths and all(not filesystem_exists(path) for path in staging_paths)
