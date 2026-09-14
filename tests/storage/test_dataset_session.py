from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _bound_first_frames,
    _persisted_table,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_storage_datasets_public_operations_reject_resource_policy_control() -> None:
    spec = make_processed_dataset_spec("no_resource_policy_control")
    calls = (
        lambda: write_dataset(
            Path("ignored"),
            pd.DataFrame(),
            spec=spec,
            resource_policy=None,
        ),
        lambda: read_dataset(Path("ignored"), resource_policy=None),
        lambda: inspect_dataset(Path("ignored"), resource_policy=None),
        lambda: fingerprint_dataset(Path("ignored"), resource_policy=None),
        lambda: compare_datasets(
            Path("left"),
            Path("right"),
            resource_policy=None,
        ),
    )

    for call in calls:
        with pytest.raises(TypeError):
            call()


def test_dataset_session_returns_fresh_contract_views_and_keeps_frozen_truth(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "immutable_session")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("immutable_session"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
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
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("session_drift"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as raw_session:
        spec_path = root / "spec.json"
        spec_path.write_bytes(raw_session.spec_bytes + b" ")
        with pytest.raises(ValueError, match="dataset_spec_identity_changed"):
            raw_session.verify()
        spec_path.write_bytes(raw_session.spec_bytes)

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as inventory_session:
        added = root / "data" / "year=2016" / "part-000.parquet"
        dataset_writing.write_parquet_file(
            _persisted_table(frame),
            added,
        )
        with pytest.raises(ValueError, match="dataset_inventory_changed"):
            inventory_session.verify()
        added.unlink()

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as removal_session:
        removal_session.all_paths[0].unlink()
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
    tmp_path: Path,
    relative_path: Path,
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"session_{relative_path.name}_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"session_{relative_path.name}_drift"),
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        path = root / relative_path
        replacement = path.with_name(f"{path.name}.replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)

        with pytest.raises(ValueError, match=error):
            session.verify()


def test_physical_decoder_rechecks_the_same_handle_after_decode(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "handle_drift")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("handle_drift"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        path = session.all_paths[0]

        def mutate_after_open(_metadata) -> None:  # noqa: ANN001
            path.touch()

        with pytest.raises(ValueError, match="dataset_file_identity_changed"):
            list(
                dataset_physical.iter_dataset_part_batches(
                    path,
                    expected_identity=session.metadata_for(path).identity,
                    target_chunk_bytes=1024**2,
                    metadata_observer=mutate_after_open,
                )
            )


def test_writer_observes_each_closed_part_once_and_passes_live_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_receipt")
    frame = make_default_dataset_frame()
    observed_paths: list[Path] = []
    decoder = dataset_writing.iter_declared_parquet_member_batches
    receipt_checks: list[bool] = []

    def record_decode(path, **kwargs):  # noqa: ANN001
        observed_paths.append(Path(path))
        yield from decoder(path, **kwargs)

    def validate_receipt(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        receipt.session.verify()
        receipt.session.close()
        receipt.session.verify()
        receipt_checks.append(receipt.exact_primary_key_index.database_path.is_file())
        assert receipt.exact_primary_key_index.add_frame(frame) == len(frame)

    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_decode,
    )
    dataset_writing.write_dataset(
        root,
        _bound_first_frames((frame.iloc[:1].copy(), frame.iloc[1:].copy())),
        spec=make_processed_dataset_spec("writer_receipt"),
        staged_validator=validate_receipt,
        resource_policy=capture_test_policy(
            available_memory_bytes=64 * 1024**2,
            explicit_cap_bytes=32 * 1024 * 1024,
        ),
    )

    assert receipt_checks == [True]
    assert len(observed_paths) == len(set(observed_paths))
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(observed_paths) == manifest["file_count"]


def test_open_dataset_session_owns_and_closes_inventory(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_inventory_owner")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_inventory_owner"),
    )
    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
    ) as session:
        scratch = session.inventory.scratch_path
        assert scratch.exists()

    assert not scratch.exists()


def test_open_dataset_session_admits_metadata_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "session_nested_retained")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("session_nested_retained"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    metadata_reads = 0

    def reject_metadata_read(*_args: object, **_kwargs: object) -> None:
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata decode ran before session admission")

    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        reject_metadata_read,
    )

    with runtime_resource_policy_scope(policy):
        blocking_lease = policy.lease(
            "test.dataset.session.capacity",
            policy.planning_headroom_bytes,
        )
        try:
            with pytest.raises(
                MemoryError,
                match="runtime_memory_budget_exceeded",
            ):
                with dataset_reading.open_dataset_session(
                    root,
                    columns=None,
                    time_window=None,
                    resource_policy=policy,
                ):
                    pass
        finally:
            blocking_lease.close()
    assert metadata_reads == 0


def test_session_from_inventory_admits_lease_before_metadata_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "writer_session_overlap")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("writer_session_overlap"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as source:
        metadata_reads = 0

        def reject_metadata_read(*_args: object, **_kwargs: object) -> None:
            nonlocal metadata_reads
            metadata_reads += 1
            raise AssertionError("metadata decode ran before session admission")

        with monkeypatch.context() as admission_patch:
            admission_patch.setattr(
                dataset_reading,
                "read_bounded_qrp_json",
                reject_metadata_read,
            )
            blocking_lease = policy.lease(
                "test.dataset.session.capacity",
                policy.planning_headroom_bytes,
            )
            try:
                with pytest.raises(
                    MemoryError,
                    match="runtime_memory_budget_exceeded",
                ):
                    dataset_reading.dataset_session_from_inventory(
                        root,
                        inventory=source.inventory,
                        columns=None,
                        time_window=None,
                        resource_policy=policy,
                    )
            finally:
                blocking_lease.close()

        assert metadata_reads == 0


def test_open_dataset_session_preserves_missing_manifest_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "missing_manifest_preflight")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("missing_manifest_preflight"),
    )
    (root / "_manifest" / "manifest.json").unlink()
    metadata_reads = 0

    def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran for missing manifest")

    monkeypatch.setattr(
        dataset_reading,
        "read_bounded_qrp_json",
        reject_metadata_read,
    )

    with pytest.raises(ValueError, match="dataset_manifest_missing"):
        with dataset_reading.open_dataset_session(
            root,
            columns=None,
            time_window=None,
        ):
            pass

    assert metadata_reads == 0


@pytest.mark.parametrize("entrypoint", ("owned", "borrowed"))
def test_dataset_session_metadata_stat_failure_precedes_metadata_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"metadata_stat_{entrypoint}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"metadata_stat_{entrypoint}"),
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=4 * 1024 * 1024,
    )
    metadata_reads = 0

    def fail_owner_preflight_identity(_path: Path):  # noqa: ANN202
        raise OSError("metadata_stat_failed")

    def reject_metadata_read(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        nonlocal metadata_reads
        metadata_reads += 1
        raise AssertionError("metadata read ran after stat failure")

    with dataset_reading.open_dataset_session(
        root,
        columns=None,
        time_window=None,
        resource_policy=policy,
    ) as source:
        with monkeypatch.context() as admission_patch:
            admission_patch.setattr(
                dataset_reading,
                "dataset_file_identity",
                fail_owner_preflight_identity,
            )
            admission_patch.setattr(
                dataset_reading,
                "read_bounded_qrp_json",
                reject_metadata_read,
            )
            with pytest.raises(ValueError, match="dataset_metadata_file_invalid"):
                if entrypoint == "owned":
                    with dataset_reading.open_dataset_session(
                        root,
                        columns=None,
                        time_window=None,
                        resource_policy=policy,
                    ):
                        pass
                else:
                    dataset_reading.dataset_session_from_inventory(
                        root,
                        inventory=source.inventory,
                        columns=None,
                        time_window=None,
                        resource_policy=policy,
                    )

    assert metadata_reads == 0


def test_writer_bounds_raw_pandas_arrow_and_persisted_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "bounded_writer")
    row_count = 100
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * row_count,
            "asset_id": [f"asset-{index:03d}" for index in range(row_count)],
            "timestamp": pd.date_range("2026-01-01", periods=row_count),
            "value": ["x" * 20_000 for _ in range(row_count)],
        }
    )
    input_bytes = pandas_frame_payload_bytes(frame)
    working_set_cap = (
        dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
        + dataset_physical._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
        + 2 * input_bytes
    )
    written_table_bytes: list[int] = []
    converted_frame_bytes: list[int] = []
    original_write = dataset_writing.write_parquet_file
    original_decode = dataset_writing.iter_declared_parquet_member_batches

    def record_write(table: pa.Table, path: Path) -> Path:
        written_table_bytes.append(table.nbytes)
        return original_write(table, path)

    def record_decode(*args, **kwargs):
        for current in original_decode(*args, **kwargs):
            converted_frame_bytes.append(pandas_frame_payload_bytes(current))
            yield current

    monkeypatch.setattr(dataset_writing, "write_parquet_file", record_write)
    monkeypatch.setattr(
        dataset_writing,
        "iter_declared_parquet_member_batches",
        record_decode,
    )
    result = dataset_writing.write_dataset(
        root,
        _bound_first_frames(
            (frame,), bound=pandas_frame_payload_bytes(frame) + 128 * 1024
        ),
        spec=make_processed_dataset_spec("bounded_writer"),
        resource_policy=capture_test_policy(
            available_memory_bytes=16 * 1024**2,
            explicit_cap_bytes=working_set_cap,
        ),
    )

    assert result.root == root
    assert result.data_root == root / "data"
    assert result.row_count == row_count
    assert len(written_table_bytes) > 1
    assert len(converted_frame_bytes) > 1
    assert max(written_table_bytes) < input_bytes
    assert max(converted_frame_bytes) < input_bytes


def test_private_writer_returns_final_inspection_and_public_writer_returns_none(
    tmp_path: Path,
) -> None:
    private_root = make_dataset_artifact_root(tmp_path, "private_return")
    public_root = make_dataset_artifact_root(tmp_path, "public_return")
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=32 * 1024 * 1024,
    )
    private_result = dataset_writing.write_dataset(
        private_root,
        _bound_first_frames((make_default_dataset_frame(),)),
        spec=make_processed_dataset_spec("private_return"),
        resource_policy=policy,
    )

    assert private_result.root == private_root
    assert private_result.data_root == private_root / "data"
    assert private_result.manifest["row_count"] == private_result.row_count
    assert (
        write_dataset(
            public_root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("public_return"),
        )
        is None
    )


def test_writer_manifest_uses_persisted_bytes_not_prewrite_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "persisted_truth")
    original = dataset_writing.write_parquet_file

    def write_infinite(table, path):  # noqa: ANN001
        position = table.schema.get_field_index("value")
        persisted = table.set_column(
            position,
            table.schema.field(position),
            pa.array(
                [float("inf")] * table.num_rows,
                type=table.schema.field(position).type,
            ),
        )
        return original(persisted, path)

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_infinite)
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("persisted_truth"),
        )


def test_exact_primary_key_index_supports_nested_arrow_key_identity(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "key": pa.array(
                [
                    {"items": [1, 2], "label": "a"},
                    {"items": [2, 1], "label": "a"},
                ],
                type=pa.struct(
                    [
                        pa.field("items", pa.list_(pa.int64())),
                        pa.field("label", pa.string()),
                    ]
                ),
            )
        }
    )
    with dataset_writing.ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
    ) as index:
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
                (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
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
                (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
            )["logical_dtypes"]
        )
    assert observed[0] == observed[1]


@pytest.mark.parametrize("reverse", (False, True))
def test_writer_rejects_incompatible_physical_width_in_both_chunk_orders(
    tmp_path: Path,
    reverse: bool,
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
            root,
            chunks,
            spec=make_processed_dataset_spec(f"physical_width_{reverse}"),
        )


def test_writer_rejects_malformed_persisted_logical_dtype_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "malformed_logical_dtype")
    monkeypatch.setattr(
        dataset_writing,
        "logical_dtype_metadata_bytes",
        lambda _payload: b"{}",
    )
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("malformed_logical_dtype"),
        )


def test_writer_rejects_persisted_category_outside_declared_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_category_domain")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical(frame["asset_id"])
    original = dataset_writing.write_parquet_file

    def write_extra_domain(current, path):  # noqa: ANN001
        persisted = current.to_pandas(use_threads=False)
        persisted["asset_id"] = persisted["asset_id"].cat.add_categories(
            ["FORGED_UNUSED_CATEGORY"]
        )
        persisted.loc[persisted.index[0], "asset_id"] = "FORGED_UNUSED_CATEGORY"
        forged = pa.Table.from_pandas(persisted, preserve_index=False)
        return original(
            forged.replace_schema_metadata(current.schema.metadata),
            path,
        )

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_extra_domain)
    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("forged_category_domain"),
        )


def test_full_inspection_rejects_forged_persisted_date_summary(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_dates")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("forged_dates"),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["date_min"] = "1900-01-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_partition_date_mismatch"):
        inspect_dataset(root)


def test_writer_receipt_reuses_observed_metadata_without_reopening_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    (member,) = (root / "data").rglob("*.parquet")
    (other,) = (replacement / "data").rglob("*.parquet")
    original = member.read_bytes()
    substituted = other.read_bytes()
    assert len(original) == len(substituted) and original != substituted
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = manifest_path.read_bytes()
    member.write_bytes(substituted)
    with pytest.raises(ValueError, match="dataset_content_identity_changed"):
        operation(root)
    assert manifest_path.read_bytes() == manifest
