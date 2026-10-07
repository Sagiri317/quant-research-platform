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
    ("field", "value"),
    (
        ("validation_status", "FAIL"),
        ("duplicate_primary_key_count", 1),
        ("null_count_summary", {"market_id": 1}),
        ("non_finite_count_summary", {"value": 1}),
    ),
)
def test_inspection_recomputes_persisted_summary_instead_of_trusting_manifest(
    tmp_path: Path, field: str, value: object
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        read_dataset(root)
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        fingerprint_dataset(root)
    if field == "validation_status":
        assert inspect_dataset(root).manifest[field] == value
    else:
        with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
            inspect_dataset(root)


@pytest.mark.parametrize(
    "null_count_summary",
    ({}, {"market_id": 0, "asset_id": 0}),
    ids=("empty", "composite-key-member-omitted"),
)
def test_inspection_rejects_missing_primary_key_null_facts(
    tmp_path: Path, null_count_summary: dict[str, int]
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["null_count_summary"] = null_count_summary
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        read_dataset(root)
    with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
        fingerprint_dataset(root)
    with pytest.raises(ValueError, match="dataset_manifest_summary_mismatch"):
        inspect_dataset(root)


def test_session_consuming_inspection_does_not_reopen_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None, require_consumable=False
    )
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
@pytest.mark.parametrize("identity_changed", (False, True))
def test_inspection_closes_native_iterator_and_preserves_consumer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
    identity_changed: bool,
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
        if identity_changed:
            os.utime(
                path,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000),
            )
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
        if identity_changed:
            assert "dataset verification also failed: ValueError" in notes
        else:
            assert not notes
        assert not unraisable
    finally:
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        for iterator in created:
            iterator.close()
    session.verify()


def test_inspection_surfaces_native_final_identity_failure_without_consumer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    path = session.read_paths[0]
    original_stat = path.stat()
    original_partition_label = dataset_inspection.relative_partition_label

    def change_identity(data_root, current):
        os.utime(
            path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000)
        )
        return original_partition_label(data_root, current)

    monkeypatch.setattr(dataset_inspection, "relative_partition_label", change_identity)
    try:
        with pytest.raises(ValueError, match="dataset_file_identity_changed"):
            inspect_dataset(root)
    finally:
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    session.verify()
