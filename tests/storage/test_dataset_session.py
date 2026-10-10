from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.storage.datasets import (
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_replace,
    filesystem_unlink,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from tests.storage.dataset_test_support import _persisted_table
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_dataset_session_returns_fresh_contract_views_and_keeps_frozen_truth(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "immutable_session")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("immutable_session"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    expected_primary_key = tuple(session.spec["primary_key"])
    expected_schema = session.schema
    expected_logical = session.logical_dtypes
    session.spec["primary_key"].clear()
    session.manifest["physical_schema"].clear()
    session.schema.clear()
    session.logical_dtypes["columns"].clear()
    assert tuple(session.spec["primary_key"]) == expected_primary_key
    assert session.schema == expected_schema
    assert session.logical_dtypes == expected_logical
    session.verify()


def test_dataset_session_rejects_raw_contract_and_inventory_drift(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_drift")
    frame = make_default_dataset_frame()
    write_dataset(root, frame, spec=make_processed_dataset_spec("session_drift"))
    raw_session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None
    )
    spec_path = root / "spec.json"
    filesystem_write_bytes(spec_path, raw_session.spec_bytes + b" ")
    with pytest.raises(ValueError, match="dataset_spec_identity_changed"):
        raw_session.verify()
    filesystem_write_bytes(spec_path, raw_session.spec_bytes)
    inventory_session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None
    )
    added = root / "data" / "year=2016" / "part-000.parquet"
    dataset_writing.write_parquet_file(_persisted_table(frame), added)
    with pytest.raises(ValueError, match="dataset_inventory_changed"):
        inventory_session.verify()
    filesystem_unlink(added)
    removal_session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None
    )
    filesystem_unlink(removal_session.all_paths[0])
    with pytest.raises(ValueError, match="dataset_inventory_changed"):
        removal_session.verify()


@pytest.mark.parametrize(
    ("relative_path", "error"),
    (
        (Path("spec.json"), "dataset_spec_identity_changed"),
        (Path("_manifest/manifest.json"), "dataset_manifest_identity_changed"),
    ),
)
def test_dataset_session_rejects_same_bytes_metadata_replacement(
    tmp_path: Path, relative_path: Path, error: str
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"session_{relative_path.name}_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"session_{relative_path.name}_drift"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    path = root / relative_path
    replacement = path.with_name(f"{path.name}.replacement")
    filesystem_write_bytes(replacement, filesystem_read_bytes(path))
    filesystem_replace(replacement, path)
    with pytest.raises(ValueError, match=error):
        session.verify()


def test_physical_decoder_rechecks_the_same_handle_after_decode(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "handle_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("handle_drift"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    path = session.all_paths[0]

    def mutate_after_open(_metadata) -> None:
        import os

        native = Path(native_filesystem_path(path))
        before = native.stat()
        os.utime(native, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))

    with pytest.raises(ValueError, match="dataset_file_identity_changed"):
        list(
            dataset_physical.iter_dataset_part_batches(
                path,
                expected_identity=session.metadata_for(path).identity,
                metadata_observer=mutate_after_open,
                batch_rows=2048,
            )
        )


def test_writer_observes_each_closed_part_once_and_passes_live_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_receipt")
    frame = make_default_dataset_frame()
    observed_paths: list[Path] = []
    decoder = dataset_writing.iter_declared_parquet_member_batches
    receipt_checks: list[bool] = []

    def record_decode(path, **kwargs):
        observed_paths.append(Path(path))
        yield from decoder(path, **kwargs)

    def validate_receipt(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        receipt.session.verify()
        receipt.session.verify()
        receipt_checks.append(
            receipt.exact_primary_key_index.identity_payload_count(namespace="dataset")
            == len(frame)
        )
        assert receipt.exact_primary_key_index.add_table(
            pa.Table.from_pandas(frame, preserve_index=False)
        ) == len(frame)

    monkeypatch.setattr(
        dataset_writing, "iter_declared_parquet_member_batches", record_decode
    )
    dataset_writing.write_dataset(
        root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("writer_receipt"),
        staged_validator=validate_receipt,
    )
    assert receipt_checks == [True]
    assert len(observed_paths) == len(set(observed_paths))
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
    )
    assert len(observed_paths) == manifest["file_count"]


def test_open_dataset_session_preserves_missing_manifest_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "missing_manifest_preflight")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("missing_manifest_preflight"),
    )
    filesystem_unlink(root / "_manifest" / "manifest.json")
    metadata_reads = 0

    def reject_metadata_read(*_args, **_kwargs):
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran for missing manifest")

    monkeypatch.setattr(dataset_reading, "read_qrp_json_file", reject_metadata_read)
    with pytest.raises(ValueError, match="dataset_manifest_missing"):
        dataset_reading.open_dataset_session(root, columns=None, time_window=None)
        pass
    assert metadata_reads == 0


@pytest.mark.parametrize("entrypoint", ("owned", "borrowed"))
@pytest.mark.parametrize("failure", ("missing", "not_directory", "io", "permission"))
def test_dataset_session_metadata_stat_failure_precedes_metadata_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str, failure: str
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"metadata_stat_{entrypoint}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"metadata_stat_{entrypoint}"),
    )
    metadata_reads = 0
    primary = {
        "missing": FileNotFoundError("metadata_missing"),
        "not_directory": NotADirectoryError("metadata_parent_invalid"),
        "io": OSError(5, "metadata_io"),
        "permission": PermissionError("metadata_permission"),
    }[failure]

    def fail_owner_preflight_identity(_path: Path):
        raise primary

    def reject_metadata_read(*_args, **_kwargs):
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran after stat failure")

    source = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    with monkeypatch.context() as identity_patch:
        identity_patch.setattr(
            dataset_reading, "dataset_file_identity", fail_owner_preflight_identity
        )
        identity_patch.setattr(
            dataset_reading, "read_qrp_json_file", reject_metadata_read
        )
        fatal = failure in {"io", "permission"}
        with pytest.raises(
            type(primary) if fatal else ValueError,
            match=None if fatal else "dataset_metadata_file_invalid",
        ) as caught:
            if entrypoint == "owned":
                dataset_reading.open_dataset_session(
                    root, columns=None, time_window=None
                )
                pass
            else:
                dataset_reading.dataset_session_from_inventory(
                    root, inventory=source.inventory, columns=None, time_window=None
                )
    assert metadata_reads == 0
    if fatal:
        assert caught.value is primary


def test_writer_manifest_uses_persisted_bytes_not_prewrite_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "persisted_truth")
    original = dataset_writing.write_parquet_file

    def write_infinite(table, path, **kwargs):
        position = table.schema.get_field_index("value")
        persisted = table.set_column(
            position,
            table.schema.field(position),
            pa.array(
                [float("inf")] * table.num_rows, type=table.schema.field(position).type
            ),
        )
        return original(persisted, path, **kwargs)

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_infinite)
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("persisted_truth"),
        )


def test_exact_primary_key_index_supports_nested_arrow_key_identity() -> None:
    table = pa.table(
        {
            "key": pa.array(
                [{"items": [1, 2], "label": "a"}, {"items": [2, 1], "label": "a"}],
                type=pa.struct(
                    [
                        pa.field("items", pa.list_(pa.int64())),
                        pa.field("label", pa.string()),
                    ]
                ),
            )
        }
    )
    index = dataset_writing.ExactPrimaryKeyIndex(primary_key=("key",))
    assert index.add_table(table) == 0
    assert index.add_table(table) == 2


def test_dataset_global_nullable_integer_plan_is_chunk_order_independent(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame()
    numpy_chunk = base.iloc[:1].copy()
    nullable_chunk = base.iloc[1:2].copy()
    numpy_chunk["value"] = pd.Series([1], index=numpy_chunk.index, dtype="int64")
    nullable_chunk["value"] = pd.Series([2], index=nullable_chunk.index, dtype="Int64")
    manifests = []
    for name, chunks in (
        ("numpy_first", (numpy_chunk, nullable_chunk)),
        ("nullable_first", (nullable_chunk, numpy_chunk)),
    ):
        root = make_dataset_artifact_root(tmp_path / name, name)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        assert str(read_dataset(root)["value"].dtype) == "Int64"
        manifests.append(
            json.loads(
                filesystem_read_text(
                    root / "_manifest" / "manifest.json", encoding="utf-8"
                )
            )
        )
    assert manifests[0]["logical_dtypes"] == manifests[1]["logical_dtypes"]
    assert manifests[0]["logical_dtype_hash"] == manifests[1]["logical_dtype_hash"]


def test_chunked_reader_preserves_declared_object_dtype_like_materialized(
    tmp_path: Path,
) -> None:
    from quant_research_platform.storage.datasets import DatasetReadMode

    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(range(len(frame)), dtype=object)
    root = make_dataset_artifact_root(tmp_path, "object_numeric")
    write_dataset(root, frame, spec=make_processed_dataset_spec("object_numeric"))
    materialized = read_dataset(root)
    chunked = pd.concat(
        list(read_dataset(root, mode=DatasetReadMode.CHUNKED)), ignore_index=True
    )
    assert materialized["value"].dtype == object
    pd.testing.assert_frame_equal(chunked, materialized)


def test_all_empty_logical_dtype_plan_is_chunk_order_independent(
    tmp_path: Path,
) -> None:
    base = make_default_dataset_frame().iloc[:0].copy()
    numpy_empty = base.astype({"value": "int64"})
    nullable_empty = base.astype({"value": "Int64"})
    observed = []
    for name, chunks in (
        ("empty_numpy_first", (numpy_empty, nullable_empty)),
        ("empty_nullable_first", (nullable_empty, numpy_empty)),
    ):
        root = make_dataset_artifact_root(tmp_path / name, name)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        frame = read_dataset(root)
        assert frame.empty
        assert str(frame["value"].dtype) == "Int64"
        observed.append(
            json.loads(
                filesystem_read_text(
                    root / "_manifest" / "manifest.json", encoding="utf-8"
                )
            )["logical_dtypes"]
        )
    assert observed[0] == observed[1]


@pytest.mark.parametrize("reverse", (False, True))
def test_writer_rejects_incompatible_physical_width_in_both_chunk_orders(
    tmp_path: Path, reverse: bool
) -> None:
    base = make_default_dataset_frame()
    narrow = base.iloc[:1].copy()
    wide = base.iloc[1:2].copy()
    narrow["value"] = pd.Series([1], index=narrow.index, dtype="int32")
    wide["value"] = pd.Series([2], index=wide.index, dtype="int64")
    chunks = (wide, narrow) if reverse else (narrow, wide)
    root = make_dataset_artifact_root(tmp_path, f"physical_width_{reverse}")
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        write_dataset(
            root, chunks, spec=make_processed_dataset_spec(f"physical_width_{reverse}")
        )


def test_writer_rejects_malformed_persisted_logical_dtype_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "malformed_logical_dtype")
    monkeypatch.setattr(
        dataset_writing, "logical_dtype_metadata_bytes", lambda _payload: b"{}"
    )
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("malformed_logical_dtype"),
        )


def test_writer_rejects_persisted_category_outside_declared_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_category_domain")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical(frame["asset_id"])
    original = dataset_writing.write_parquet_file

    def write_extra_domain(current, path, **kwargs):
        persisted = current.to_pandas(use_threads=False)
        persisted["asset_id"] = persisted["asset_id"].cat.add_categories(
            ["FORGED_UNUSED_CATEGORY"]
        )
        persisted.loc[persisted.index[0], "asset_id"] = "FORGED_UNUSED_CATEGORY"
        forged = pa.Table.from_pandas(persisted, preserve_index=False)
        return original(
            forged.replace_schema_metadata(current.schema.metadata), path, **kwargs
        )

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_extra_domain)
    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        write_dataset(
            root, frame, spec=make_processed_dataset_spec("forged_category_domain")
        )


def test_full_inspection_rejects_summary_outside_partition_dates(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_dates")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("forged_dates"),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["date_min"] = "1900-01-01"
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_partition_date_mismatch"):
        inspect_dataset(root)


def test_writer_receipt_reuses_observed_metadata_without_reopening_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "receipt_no_reopen")

    def forbidden_reopen(*_args, **_kwargs):
        raise AssertionError("writer_receipt_must_not_reopen_staged_parquet")

    monkeypatch.setattr(dataset_reading, "inspect_dataset_part", forbidden_reopen)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("receipt_no_reopen"),
    )


@pytest.mark.parametrize(
    "operation", (read_dataset, inspect_dataset, fingerprint_dataset)
)
def test_persisted_identity_rejects_same_statistics_member_replacement(
    tmp_path, operation
):
    root = make_dataset_artifact_root(tmp_path, "content_bound")
    replacement = make_dataset_artifact_root(tmp_path / "replacement", "content_bound")
    frame = make_default_dataset_frame()
    frame["timestamp"] = pd.date_range("2020-01-01", periods=3)
    frame["value"] = [0.0, 10.0, 100.0]
    changed = frame.copy()
    changed.loc[changed.index[1], "value"] = 40.0
    spec = make_processed_dataset_spec("content_bound")
    write_dataset(root, frame, spec=spec)
    write_dataset(replacement, changed, spec=spec)
    pd.testing.assert_frame_equal(read_dataset(root), frame, check_dtype=False)
    (member,) = filesystem_glob(root / "data", "**/*.parquet")
    (other,) = filesystem_glob(replacement / "data", "**/*.parquet")
    original = filesystem_read_bytes(member)
    substituted = filesystem_read_bytes(other)
    assert len(original) == len(substituted) and original != substituted
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = filesystem_read_bytes(manifest_path)
    filesystem_write_bytes(member, substituted)
    with pytest.raises(ValueError, match="dataset_content_identity_changed"):
        operation(root)
    assert filesystem_read_bytes(manifest_path) == manifest


@pytest.mark.parametrize("changed", (False, True))
def test_session_validates_each_distinct_consecutive_logical_schema_once(
    tmp_path, monkeypatch, changed
):
    root = make_dataset_artifact_root(tmp_path, "category_scan")
    frames = []
    for index in range(4):
        frame = make_default_dataset_frame(
            asset_ids=["a"], dates=[f"2020-01-{index + 1:02}"], values=["v0"]
        )
        frame["value"] = pd.Categorical(
            frame["value"],
            categories=[f"v{i}" for i in range(20 + (changed and index >= 2))],
        )
        frames.append(frame)
    write_dataset(root, iter(frames), spec=make_processed_dataset_spec("category_scan"))
    original = dataset_reading.validate_persisted_logical_schema
    calls = []

    def checked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(dataset_reading, "validate_persisted_logical_schema", checked)
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    assert len(session.all_paths) == 4
    assert len(calls) == (2 if changed else 1)
    session.verify()
