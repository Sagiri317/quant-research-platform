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
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_is_file,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_write_bytes,
    native_filesystem_path,
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

    assert not filesystem_exists(root)


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
    filesystem_mkdir(index_root)
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
        assert filesystem_is_file(database_path)
        assert index.add_table(pa.Table.from_pandas(first, preserve_index=False)) == 0
        assert (
            index.add_table(
                pa.Table.from_pandas(duplicate_same_instant, preserve_index=False)
            )
            == 1
        )
        for suffix in ("-journal", "-wal", "-shm"):
            filesystem_write_bytes(Path(f"{database_path}{suffix}"), b"scratch")

    assert not filesystem_exists(database_path)
    assert list(filesystem_iterdir(index_root)) == []

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
    assert list(filesystem_iterdir(index_root)) == []

    exceptional = module.ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
    )
    with pytest.raises(RuntimeError, match="stop"):
        with exceptional:
            exceptional_path = exceptional.database_path
            raise RuntimeError("stop")
    assert not filesystem_exists(exceptional_path)
    assert list(filesystem_iterdir(index_root)) == []


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

    assert list(filesystem_iterdir(tmp_path)) == []


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


def test_exact_key_tmpfs_row_stream_keeps_capacity_for_sibling_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage._scratch_ownership import _memory_filesystem
    from quant_research_platform.storage.exact_keys import (
        EXACT_KEY_SQLITE_CACHE_BYTES,
        ExactPrimaryKeyIndex,
    )
    from tests.helpers.runtime_resources import capture_test_policy

    if not _memory_filesystem(tmp_path):
        pytest.skip("requires actual tmpfs")
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=EXACT_KEY_SQLITE_CACHE_BYTES + 512 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("storage.exact_keys", EXACT_KEY_SQLITE_CACHE_BYTES):
            with ExactPrimaryKeyIndex(primary_key=()) as index:
                rows = (
                    ("stream", position.to_bytes(4, "big"), b"value")
                    for position in range(32)
                )
                assert index.add_identity_payload_rows(rows=rows) == 32
                assert index.identity_payload_count(namespace="stream") == 32
                with policy.lease("test.sibling", 64 * 1024):
                    assert policy.used_memory_bytes <= policy.explicit_memory_cap_bytes
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, ".k-*"))


def test_exact_key_tmpfs_low_disk_uses_row_credit_before_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import _scratch_ownership, exact_keys
    from tests.helpers.runtime_resources import capture_test_policy

    if not _scratch_ownership._memory_filesystem(tmp_path):
        pytest.skip("requires actual tmpfs")
    monkeypatch.setattr(exact_keys, "gettempdir", lambda: str(tmp_path))
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease(
            "storage.exact_keys", exact_keys.EXACT_KEY_SQLITE_CACHE_BYTES
        ):
            with exact_keys.ExactPrimaryKeyIndex(primary_key=("key",)) as index:
                assert index.add_table(pa.table({"key": [1]})) == 0
                connection = index._connection
                assert connection is not None
                scratch = index._scratch_reservation
                assert scratch is not None
                resident_bytes = (
                    sum(
                        int(
                            connection.execute(
                                f"PRAGMA {schema}.page_count"
                            ).fetchone()[0]
                        )
                        for schema in ("main", "temp")
                    )
                    * exact_keys.EXACT_KEY_SQLITE_PAGE_BYTES
                )
                assert scratch.reserved_bytes == resident_bytes

                class ProbeComplete(Exception):
                    pass

                batch_targets: list[int] = []
                resize = scratch.resize

                def capture_batch_target(total_bytes: int) -> None:
                    batch_targets.append(total_bytes)
                    raise ProbeComplete

                monkeypatch.setattr(scratch, "resize", capture_batch_target)
                with pytest.raises(ProbeComplete):
                    index._admit_physical_write(
                        1024**2,
                        minimum_bytes=8 * exact_keys.EXACT_KEY_SQLITE_PAGE_BYTES,
                    )
                monkeypatch.setattr(scratch, "resize", resize)
                assert len(batch_targets) == 1
                batch_growth = batch_targets[0] - scratch.reserved_bytes
                free_disk = batch_targets[0] - 2 * resident_bytes + resident_bytes // 2
                assert 0 < free_disk < batch_growth

                observed_growth: list[int] = []

                def constrained_disk(_path: Path, *, reserve_bytes: int) -> int:
                    observed_growth.append(reserve_bytes)
                    if reserve_bytes > free_disk:
                        raise RuntimeError("insufficient_free_disk")
                    return free_disk

                monkeypatch.setattr(
                    exact_keys, "require_free_disk_headroom", constrained_disk
                )
                monkeypatch.setattr(
                    _scratch_ownership, "require_free_disk_headroom", constrained_disk
                )
                assert index.add_table(pa.table({"key": [2]})) == 0
                assert index.identity_payload_count(namespace="dataset") == 2
                assert observed_growth and max(observed_growth) <= free_disk
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, ".k-*"))


def test_comparison_tmpfs_reuses_bounded_row_payload_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.runtime.resources import (
        MemoryLease,
        runtime_resource_policy_scope,
    )
    from quant_research_platform.storage._scratch_ownership import _memory_filesystem
    from quant_research_platform.storage.exact_keys import (
        EXACT_KEY_SQLITE_CACHE_BYTES,
        ExactPrimaryKeyIndex,
    )
    from tests.helpers.runtime_resources import capture_test_policy

    if not _memory_filesystem(tmp_path):
        pytest.skip("requires actual tmpfs")
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    grow = MemoryLease.grow
    payload_growths = 0

    def observed_grow(lease: MemoryLease, additional_bytes: int) -> None:
        nonlocal payload_growths
        if lease.owner == "storage.dataset.comparison.payload":
            payload_growths += 1
        grow(lease, additional_bytes)

    monkeypatch.setattr(MemoryLease, "grow", observed_grow)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    table = pa.table(
        {"key": range(256), "score": [float(position) for position in range(256)]}
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("storage.exact_keys", EXACT_KEY_SQLITE_CACHE_BYTES):
            with ExactPrimaryKeyIndex(primary_key=("key",)) as index:
                assert (
                    index.add_comparison_table(
                        table,
                        namespace="left",
                        value_columns=("score",),
                        payload_target_bytes=1024**2,
                    )
                    == 0
                )
                assert index.identity_payload_count(namespace="left") == 256
                assert payload_growths < 64
                with policy.lease("test.sibling", 64 * 1024):
                    assert policy.used_memory_bytes <= policy.explicit_memory_cap_bytes
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, ".k-*"))


def test_exact_key_tmpfs_large_btree_split_stays_within_admitted_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage._scratch_ownership import _memory_filesystem
    from quant_research_platform.storage.exact_keys import (
        EXACT_KEY_SQLITE_CACHE_BYTES,
        ExactPrimaryKeyIndex,
    )

    if not _memory_filesystem(tmp_path):
        pytest.skip("requires actual tmpfs")
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )

    def key_table(positions: range) -> pa.Table:
        frame = pd.DataFrame(
            {
                "market_id": ["a_share"] * len(positions),
                "asset_id": [f"a{position % 256:03d}" for position in positions],
                "timestamp": pd.Timestamp("2020-01-01")
                + pd.to_timedelta(list(positions), unit="s"),
            }
        )
        return pa.Table.from_pandas(frame, preserve_index=False)

    with runtime_resource_policy_scope(explicit_cap_bytes=2 * 1024**3) as policy:
        with policy.lease("storage.exact_keys", EXACT_KEY_SQLITE_CACHE_BYTES):
            with ExactPrimaryKeyIndex(
                primary_key=("market_id", "asset_id", "timestamp")
            ) as index:
                connection = index._connection
                assert connection is not None
                admit = index._admit_physical_write
                admissions: list[tuple[int, int, int | None]] = []

                def observed_admit(
                    reserve_bytes: int, *, minimum_bytes: int | None = None
                ) -> int:
                    page_count = int(
                        connection.execute("PRAGMA main.page_count").fetchone()[0]
                    )
                    credit = admit(reserve_bytes, minimum_bytes=minimum_bytes)
                    admissions.append((page_count, credit, minimum_bytes))
                    return credit

                monkeypatch.setattr(index, "_admit_physical_write", observed_admit)
                for start in range(0, 40_000, 4_096):
                    assert (
                        index.add_table(
                            key_table(range(start, min(start + 4_096, 40_000)))
                        )
                        == 0
                    )
                assert index.identity_payload_count(namespace="dataset") == 40_000
                assert len(admissions) < 10_000
                assert any(
                    before.bit_length() < after.bit_length()
                    and minimum is not None
                    and credit > minimum
                    for (before, credit, minimum), (after, _, _) in zip(
                        admissions, admissions[1:], strict=False
                    )
                )
                samples = key_table(range(36_008, 36_011))
                assert index.add_table(samples) == 3
                connection = index._connection
                assert connection is not None
                for namespace, payload, _ in index._payload_rows(
                    samples, namespace="dataset"
                ):
                    assert connection.execute(
                        "SELECT payload, value FROM exact_keys "
                        "WHERE namespace = ? AND payload = ?",
                        (namespace, bytes(payload)),
                    ).fetchone() == (bytes(payload), None)
                assert policy.used_memory_bytes <= policy.explicit_memory_cap_bytes
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, ".k-*"))


def test_exact_key_tmpfs_lookup_admits_matched_namespace_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage._scratch_ownership import _memory_filesystem
    from quant_research_platform.storage.exact_keys import (
        EXACT_KEY_SQLITE_CACHE_BYTES,
        ExactPrimaryKeyIndex,
    )
    from tests.helpers.runtime_resources import capture_test_policy

    if not _memory_filesystem(tmp_path):
        pytest.skip("requires actual tmpfs")
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=EXACT_KEY_SQLITE_CACHE_BYTES + 2 * 1024**2,
    )
    table = pa.table(
        {
            "key": range(256),
            "score": [float(position) for position in range(256)],
        }
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("artifacts.prediction.lookup", EXACT_KEY_SQLITE_CACHE_BYTES):
            with ExactPrimaryKeyIndex(primary_key=("key",)) as index:
                assert (
                    index.add_float_value_table(
                        table,
                        namespace="prediction",
                        value_column="score",
                        payload_target_bytes=1024**2,
                    )
                    == 0
                )
                for _ in range(2):
                    values, found = index.lookup_float_values(
                        table.select(("key",)),
                        source_namespace="prediction",
                        matched_namespace="prediction-matched",
                        payload_target_bytes=1024**2,
                    )
                    assert list(found) == [1] * 256
                    assert list(values) == [float(position) for position in range(256)]
                assert (
                    index.identity_payload_count(namespace="prediction-matched") == 256
                )
                assert index.difference_counts(
                    expected_namespace="prediction",
                    actual_namespace="prediction-matched",
                ) == (0, 0)
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, ".k-*"))


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
    part = next(filesystem_glob(root / "data", "**/part-*.parquet"))
    persisted_key_schema = pa.schema(
        [pq.ParquetFile(native_filesystem_path(part)).schema_arrow.field("asset_id")]
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

    assert not filesystem_exists(root)
    assert not list(filesystem_glob(root.parent, ".a-*"))


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

    assert not filesystem_exists(root)
