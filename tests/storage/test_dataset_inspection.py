from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import quant_research_platform.storage.datasets.inspection as dataset_inspection
import quant_research_platform.storage.datasets.reading as dataset_reading
from quant_research_platform.storage.datasets import (
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


@pytest.mark.parametrize(
    ("failure", "error"),
    (
        ("duplicate", "dataset_primary_key_duplicate"),
        ("null", "dataset_primary_key_null"),
        ("infinite", "dataset_infinite_value"),
    ),
)
def test_explicit_inspection_checks_actual_values_without_persisted_zero_facts(
    tmp_path: Path, failure: str, error: str
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = make_dataset_artifact_root(tmp_path)
    result = write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    assert not {
        "duplicate_primary_key_count",
        "null_count_summary",
        "non_finite_count_summary",
    }.intersection(result.manifest)
    path = next((root / "data").rglob("*.parquet"))
    with pq.ParquetFile(path) as member:
        table = member.read()
    if failure == "duplicate":
        table = pa.concat_tables([table, table.slice(0, 1)])
    else:
        column = "asset_id" if failure == "null" else "value"
        values = table[column].to_pylist()
        values[0] = None if failure == "null" else float("inf")
        field = table.schema.field(column)
        table = table.set_column(
            table.schema.get_field_index(column),
            field,
            pa.array(values, type=field.type),
        )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match=error):
        inspect_dataset(root)


def test_inspection_ignores_unconsumed_old_integrity_metadata(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        duplicate_primary_key_count=1,
        null_count_summary={"asset_id": 1},
        non_finite_count_summary={"value": 1},
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert len(read_dataset(root)) == 3
    assert fingerprint_dataset(root).row_count == 3
    assert inspect_dataset(root).row_count == 3


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
    inspection = dataset_inspection.inspect_dataset_session(session)
    assert inspection.root == root
    assert inspection.row_count == 3
    assert inspection.file_count == 2


@pytest.mark.parametrize("consumer", ("public-partition", "frame-observer"))
def test_inspection_closes_native_iterator_and_preserves_consumer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    assert len(read_dataset(root)) == 3
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    path = session.read_paths[0]
    original_stat = path.stat()
    original_batches = dataset_inspection.iter_dataset_file_batches
    created = []
    unraisable = []
    primary = RuntimeError("synthetic_inspection_consumer_failure")

    def capture_native(*args, **kwargs):
        iterator = original_batches(*args, **kwargs)
        created.append(iterator)
        return iterator

    def fail_consumer(*_args):
        raise primary

    monkeypatch.setattr(dataset_inspection, "iter_dataset_file_batches", capture_native)
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    try:
        with pytest.raises(RuntimeError) as observed:
            if consumer == "public-partition":
                monkeypatch.setattr(
                    dataset_inspection, "relative_partition_label", fail_consumer
                )
                inspect_dataset(root)
            else:
                dataset_inspection._inspection_runtime_metadata(
                    session, batch_rows=1, frame_observer=fail_consumer
                )
        assert observed.value is primary
        assert len(created) == 1
        assert created[0].gi_frame is None
        notes = getattr(primary, "__notes__", [])
        assert not notes
        assert not unraisable
    finally:
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        for iterator in created:
            iterator.close()
