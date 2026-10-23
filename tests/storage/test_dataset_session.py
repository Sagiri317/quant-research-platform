from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_unlink,
    filesystem_write_text,
    native_filesystem_path,
)
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


def test_writer_validates_each_canonical_part_without_full_readback(
    tmp_path, monkeypatch
):
    root = make_dataset_artifact_root(tmp_path, "canonical_stream")
    frame = make_default_dataset_frame()
    observed = []
    original = dataset_writing.inspect_dataset_part

    def record_footer(path, **kwargs):
        observed.append(Path(path))
        return original(path, **kwargs)

    def reject_full_read(*_args, **_kwargs):
        raise AssertionError("writer_redecoded_canonical_rows")

    monkeypatch.setattr(dataset_writing, "inspect_dataset_part", record_footer)
    monkeypatch.setattr(
        dataset_reading, "iter_declared_parquet_member_batches", reject_full_read
    )
    result = dataset_writing.write_dataset(
        root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("canonical_stream"),
    )
    assert len(observed) == len(set(observed)) == result.file_count
    assert result.row_count == len(frame)
    assert result.manifest["duplicate_primary_key_count"] == 0


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
    assert metadata_reads == 0


def test_canonical_nonfinite_rows_rejected_before_publication(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "canonical_invalid")
    frame = make_default_dataset_frame().assign(value=float("inf"))
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            root, frame, spec=make_processed_dataset_spec("canonical_invalid")
        )
    assert not root.exists()
    assert not list(root.parent.glob(".a-*"))


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
                Path(
                    native_filesystem_path(root / "_manifest" / "manifest.json")
                ).read_text(encoding="utf-8")
            )
        )
    assert manifests[0]["logical_dtypes"] == manifests[1]["logical_dtypes"]


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
                Path(
                    native_filesystem_path(root / "_manifest" / "manifest.json")
                ).read_text(encoding="utf-8")
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


def test_writer_rejects_canonical_category_outside_declared_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "forged_category_domain")
    frame = make_default_dataset_frame()
    frame["asset_id"] = pd.Categorical(frame["asset_id"])
    original = dataset_writing._WriteStats.table_for_frame

    def write_extra_domain(owner, source, **kwargs):
        current = original(owner, source, **kwargs)
        persisted = current.to_pandas(use_threads=False)
        persisted["asset_id"] = persisted["asset_id"].cat.add_categories(
            ["FORGED_UNUSED_CATEGORY"]
        )
        persisted.loc[persisted.index[0], "asset_id"] = "FORGED_UNUSED_CATEGORY"
        forged = pa.Table.from_pandas(persisted, preserve_index=False)
        return forged.cast(current.schema)

    monkeypatch.setattr(
        dataset_writing._WriteStats, "table_for_frame", write_extra_domain
    )
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
    manifest = json.loads(
        Path(native_filesystem_path(manifest_path)).read_text(encoding="utf-8")
    )
    manifest["date_min"] = "1900-01-01"
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_partition_date_mismatch"):
        inspect_dataset(root)
