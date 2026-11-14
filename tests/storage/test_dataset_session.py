from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
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
    schema = session.schema
    expected_kind = schema[0]["type"]["kind"]
    schema[0]["type"]["kind"] = "caller_owned"
    assert session.schema[0]["type"]["kind"] == expected_kind
    session.logical_dtypes["columns"].clear()
    assert tuple(session.spec["primary_key"]) == expected_primary_key
    assert session.schema == expected_schema
    assert session.logical_dtypes == expected_logical


def test_writer_summarizes_input_without_decoding_its_output(tmp_path, monkeypatch):
    root = make_dataset_artifact_root(tmp_path, "canonical_stream")
    frame = make_default_dataset_frame()
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")

    def reject_full_read(*_args, **_kwargs):
        raise AssertionError("writer_redecoded_canonical_rows")

    with monkeypatch.context() as patch:
        patch.setattr(
            dataset_reading, "iter_declared_parquet_member_batches", reject_full_read
        )
        patch.setattr(dataset_reading, "_arrow_slice_to_pandas", reject_full_read)
        result = dataset_writing.write_dataset(
            root,
            (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
            spec=make_processed_dataset_spec("canonical_stream"),
        )
    assert result.row_count == len(frame)
    reread = read_dataset(root)
    pd.testing.assert_frame_equal(
        reread.sort_values(["timestamp", "asset_id"]).reset_index(drop=True),
        frame.sort_values(["timestamp", "asset_id"]).reset_index(drop=True),
    )
    inspected = inspect_dataset(root)
    assert inspected.row_count == result.row_count
    assert inspected.partition_row_counts == result.partition_row_counts


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
