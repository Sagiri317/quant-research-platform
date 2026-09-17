from __future__ import annotations

import importlib
import importlib.util
import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.storage.datasets import (
    read_dataset,
    write_dataset,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_writer_rejects_inconsistent_ordered_categorical_domains(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "ordered_categorical_parts")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["asset_id"] = pd.Categorical(
        first["asset_id"],
        categories=[first["asset_id"].iloc[0]],
        ordered=True,
    )
    second["asset_id"] = pd.Categorical(
        second["asset_id"],
        categories=[second["asset_id"].iloc[0]],
        ordered=True,
    )

    with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
        write_dataset(
            root,
            (first, second),
            spec=make_processed_dataset_spec("ordered_categorical_parts"),
        )

    assert not root.exists()


def test_exact_primary_key_index_is_spillable_exact_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(index_root)
    )
    module_spec = importlib.util.find_spec("quant_research_platform.storage.exact_keys")
    assert module_spec is not None, "storage exact key index module is missing"
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index_root = tmp_path / "index"
    index_root.mkdir()
    index = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id", "timestamp"),
    )
    first = pd.DataFrame(
        {
            "asset_id": ["asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:01:00"]),
        }
    )
    duplicate_same_instant = pd.DataFrame(
        {
            "asset_id": ["asset_a"],
            "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
        }
    )

    with index:
        database_path = index.database_path
        assert database_path.is_file()
        assert index.add_table(pa.Table.from_pandas(first, preserve_index=False)) == 0
        assert (
            index.add_table(
                pa.Table.from_pandas(duplicate_same_instant, preserve_index=False)
            )
            == 1
        )
        for suffix in ("-journal", "-wal", "-shm"):
            Path(f"{database_path}{suffix}").write_bytes(b"scratch")

    assert not database_path.exists()
    assert list(index_root.iterdir()) == []

    numeric = module.ExactPrimaryKeyIndex(
        primary_key=("numeric_key",),
    )
    with numeric:
        assert (
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"numeric_key": [-0.0]}), preserve_index=False
                )
            )
            == 0
        )
        assert (
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"numeric_key": [0.0]}), preserve_index=False
                )
            )
            == 1
        )
    assert list(index_root.iterdir()) == []

    exceptional = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
    )
    with pytest.raises(RuntimeError, match="stop"):
        with exceptional:
            exceptional_path = exceptional.database_path
            raise RuntimeError("stop")
    assert not exceptional_path.exists()
    assert list(index_root.iterdir()) == []


def test_exact_key_index_cleans_scratch_when_sqlite_connect_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    index = module.ExactPrimaryKeyIndex(
        primary_key=("key",),
    )

    def fail_connect(*_args, **_kwargs):
        raise module.sqlite3.OperationalError("synthetic_connect_failure")

    monkeypatch.setattr(module.sqlite3, "connect", fail_connect)

    with pytest.raises(module.sqlite3.OperationalError):
        index.__enter__()

    assert list(tmp_path.iterdir()) == []


def test_exact_key_index_inserts_unique_rows_from_duplicate_chunk(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
    ) as index:
        assert (
            index.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": ["a", "a", "b"]}), preserve_index=False
                )
            )
            == 1
        )
        assert (
            index.add_table(
                pa.Table.from_pandas(pd.DataFrame({"key": ["b"]}), preserve_index=False)
            )
            == 1
        )


def test_exact_key_index_spills_sqlite_temp_state_to_disk(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
    ) as index:
        assert index._connection is not None
        temp_store = int(index._connection.execute("PRAGMA temp_store").fetchone()[0])
        index.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": ["a", "b"]}), preserve_index=False
            ),
            namespace="left",
        )
        index.add_table(
            pa.Table.from_pandas(
                pd.DataFrame({"key": ["b", "c"]}), preserve_index=False
            ),
            namespace="right",
        )

        assert temp_store == 1
        assert index.difference_counts(
            expected_namespace="left",
            actual_namespace="right",
        ) == (1, 1)


def test_exact_primary_key_index_uses_target_arrow_scalar_semantics(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.float64())]),
    ) as numeric:
        assert (
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [Decimal("1.0")]}), preserve_index=False
                )
            )
            == 0
        )
        assert (
            numeric.add_table(
                pa.Table.from_pandas(pd.DataFrame({"key": [1.0]}), preserve_index=False)
            )
            == 1
        )
        assert (
            numeric.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [-0.0]}), preserve_index=False
                )
            )
            == 0
        )
        assert (
            numeric.add_table(
                pa.Table.from_pandas(pd.DataFrame({"key": [0.0]}), preserve_index=False)
            )
            == 1
        )
        for infinite_value in (math.inf, -math.inf):
            with pytest.raises(
                ValueError,
                match="invalid_dataset_primary_key_value",
            ):
                numeric.add_table(
                    pa.Table.from_pandas(
                        pd.DataFrame({"key": [infinite_value]}), preserve_index=False
                    )
                )
        for null_value in (None, math.nan):
            with pytest.raises(ValueError, match="dataset_primary_key_null"):
                numeric.add_table(
                    pa.Table.from_pandas(
                        pd.DataFrame({"key": [null_value]}), preserve_index=False
                    )
                )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.timestamp("ns"))]),
    ) as timestamps:
        assert (
            timestamps.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [date(2026, 1, 1)]}), preserve_index=False
                )
            )
            == 0
        )
        assert (
            timestamps.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": [datetime(2026, 1, 1)]}), preserve_index=False
                )
            )
            == 1
        )

    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", pa.string())]),
    ) as strings:
        assert (
            strings.add_table(
                pa.Table.from_pandas(pd.DataFrame({"key": ["é"]}), preserve_index=False)
            )
            == 0
        )
        assert (
            strings.add_table(
                pa.Table.from_pandas(
                    pd.DataFrame({"key": ["e\u0301"]}), preserve_index=False
                )
            )
            == 0
        )


def test_exact_primary_key_accepts_timestamp_outside_nanosecond_range(
    tmp_path: Path,
) -> None:
    module = importlib.import_module("quant_research_platform.storage.exact_keys")
    out_of_nanosecond_range = 32_503_680_000
    timestamp_type = pa.timestamp("s", tz="UTC")
    table = pa.table({"key": pa.array([out_of_nanosecond_range], type=timestamp_type)})
    with module.ExactPrimaryKeyIndex(
        primary_key=("key",),
        key_schema=pa.schema([pa.field("key", timestamp_type)]),
    ) as timestamps:
        assert timestamps.add_table(table) == 0
        assert timestamps.add_table(table) == 1


def test_dataset_writer_accepts_dictionary_encoded_primary_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_tracking = importlib.import_module("quant_research_platform.storage.exact_keys")
    tracked_schemas: list[pa.Schema] = []
    original_add_table = key_tracking.ExactPrimaryKeyIndex.add_table

    def record_schema(index, table, **kwargs):  # noqa: ANN001
        tracked_schemas.append(table.schema.remove_metadata())
        return original_add_table(index, table, **kwargs)

    monkeypatch.setattr(
        key_tracking.ExactPrimaryKeyIndex,
        "add_table",
        record_schema,
    )
    root = make_dataset_artifact_root(tmp_path, "categorical_keys")
    frame = make_default_dataset_frame()
    frame["asset_id"] = frame["asset_id"].astype("category")

    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("categorical_keys"),
    )

    assert sorted(read_dataset(root)["asset_id"].astype(str).tolist()) == sorted(
        frame["asset_id"].astype(str).tolist()
    )
    part = next((root / "data").rglob("part-*.parquet"))
    persisted_key_schema = pa.schema(
        [pq.ParquetFile(part).schema_arrow.field("asset_id")]
    )
    assert tracked_schemas
    assert all(
        pa.schema([schema.field("asset_id")]) == persisted_key_schema
        for schema in tracked_schemas
    )


def test_write_dataset_rejects_duplicate_across_chunks_and_cleans_stage(
    tmp_path: Path,
) -> None:
    frame = make_default_dataset_frame().iloc[:2].reset_index(drop=True)
    root = make_dataset_artifact_root(tmp_path, "duplicate_across_chunks")

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            root,
            (frame, frame.iloc[[0]].copy()),
            spec=make_processed_dataset_spec("duplicate_across_chunks"),
        )

    assert not root.exists()
    assert not list(root.parent.glob(".a-*"))


def test_write_dataset_rejects_existing_root_without_overwrite(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    original = read_dataset(root)

    with pytest.raises(ValueError, match="dataset_exists"):
        write_dataset(
            root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
        )

    pd.testing.assert_frame_equal(read_dataset(root), original)


def test_write_dataset_rejects_generic_tmp_artifact_root_fail_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid_processed_dataset_root"):
        write_dataset(
            tmp_path / "data" / "processed" / "a_share" / "factor" / ".tmp-user",
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec(),
        )


def test_write_dataset_rejects_writer_scratch_name_as_final_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / ".x.tmp-user"

    with pytest.raises(ValueError, match="invalid_processed_dataset_root"):
        write_dataset(
            root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
        )

    assert not root.exists()
