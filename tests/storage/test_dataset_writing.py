from __future__ import annotations

import json
import math
import weakref
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import quant_research_platform.runtime.resources as runtime_resources
import quant_research_platform.storage.datasets.inventory as dataset_inventory
import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    compare_datasets,
    fingerprint_dataset,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_is_file,
    filesystem_read_text,
    filesystem_write_text,
)
from tests.helpers.runtime_resources import (
    capture_storage_test_policy,
    capture_test_policy,
)
from tests.storage.dataset_test_support import (
    _bound_first_frames,
    _input_dataset_identity_payload,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def test_write_dataset_writes_complete_artifact_root(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path)

    result = write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec()
    )
    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
    )
    spec_payload = json.loads(
        filesystem_read_text(root / "spec.json", encoding="utf-8")
    )

    assert result is None
    assert filesystem_is_file(root / "data" / "year=2014" / "part-000.parquet")
    assert filesystem_is_file(root / "data" / "year=2015" / "part-000.parquet")
    assert manifest["columns"] == ["market_id", "asset_id", "timestamp", "value"]
    assert manifest["schema_id"] == "qrp-dataset-manifest"
    assert manifest["partition_row_counts"] == {"year=2014": 1, "year=2015": 2}
    assert manifest["file_count"] == 2
    assert manifest["row_count"] == 3
    assert spec_payload["data_id"] == "x"
    assert spec_payload["schema_id"] == "qrp-dataset-spec"
    assert spec_payload["kind"] == "factor"


def test_write_dataset_partitions_without_groupby_sort_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_groupby(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("partition_write_must_not_groupby_full_frame")

    monkeypatch.setattr(pd.DataFrame, "groupby", fail_groupby)
    root = make_dataset_artifact_root(tmp_path, "no_groupby_partition")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("no_groupby_partition"),
    )

    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
    )
    assert manifest["partition_row_counts"] == {"year=2014": 1, "year=2015": 2}


def test_write_dataset_rejects_parallel_input_fingerprint_lineage(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="unsupported_dataset_instance_metadata_key: input_fingerprints",
    ):
        write_dataset(
            make_dataset_artifact_root(tmp_path),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec(),
            instance_metadata={"input_fingerprints": {}},
        )


def test_write_dataset_merges_exact_input_dataset_identities(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "identity_metadata")
    identities = {"demo_input": _input_dataset_identity_payload()}

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("identity_metadata"),
        instance_metadata={"input_dataset_identities": identities},
    )

    inspection = inspect_dataset(root)
    assert inspection.manifest["input_dataset_identities"] == identities


def test_write_dataset_merges_exact_source_artifact_identity(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "source_artifact_identity")
    source_identity = {
        "schema_id": "qrp-binance-recorder",
        "artifact_id": "run-1",
        "manifest_hash": "a" * 64,
    }

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("source_artifact_identity"),
        instance_metadata={
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            },
            "source_artifact_identity": source_identity,
        },
    )

    inspection = inspect_dataset(root)
    assert inspection.manifest["source_artifact_identity"] == source_identity


@pytest.mark.parametrize(
    "source_identity",
    (
        {},
        {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": "bad",
        },
        {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": "a" * 64,
            "extra": "x",
        },
    ),
)
def test_write_dataset_rejects_malformed_source_artifact_identity(
    tmp_path: Path,
    source_identity: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "bad_source_artifact_identity"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("bad_source_artifact_identity"),
            instance_metadata={"source_artifact_identity": source_identity},
        )


@pytest.mark.parametrize(
    "identities",
    [
        [],
        {"demo_input": _input_dataset_identity_payload(extra={"unexpected": "x"})},
        {"demo_input": _input_dataset_identity_payload(data_id="other_input")},
        {"demo_input": _input_dataset_identity_payload(extra={"spec_hash": 1})},
        {"demo_input": _input_dataset_identity_payload(extra={"manifest_hash": "x"})},
        {1: _input_dataset_identity_payload()},
    ],
)
def test_write_dataset_rejects_malformed_input_dataset_identities(
    tmp_path: Path,
    identities: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_input_dataset_identities_metadata"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "bad_identity_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("bad_identity_metadata"),
            instance_metadata={"input_dataset_identities": identities},
        )


def test_inspect_dataset_rejects_tampered_input_dataset_identity(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "tampered_identity_metadata")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("tampered_identity_metadata"),
        instance_metadata={
            "input_dataset_identities": {
                "demo_input": _input_dataset_identity_payload()
            }
        },
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["input_dataset_identities"]["demo_input"]["unknown"] = "x"
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid_input_dataset_identities_metadata"):
        inspect_dataset(root)


def test_inspect_dataset_rejects_parallel_input_fingerprint_lineage(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "parallel_fingerprint_lineage")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("parallel_fingerprint_lineage"),
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["input_fingerprints"] = {}
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="dataset_manifest_fields_invalid",
    ):
        inspect_dataset(root)


def test_write_dataset_rejects_unsupported_reserved_and_unsafe_instance_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "unsupported_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("unsupported_metadata"),
            instance_metadata={"other_metadata": {}},
        )

    with pytest.raises(ValueError, match="reserved_dataset_instance_metadata_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "reserved_metadata"),
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("reserved_metadata"),
            instance_metadata={"row_count": 999},
        )


def test_write_dataset_accepts_dataframe_and_dataframe_chunks(tmp_path: Path) -> None:
    materialized_root = make_dataset_artifact_root(tmp_path, "materialized")
    chunked_root = make_dataset_artifact_root(tmp_path, "chunked")
    frame = make_default_dataset_frame()

    write_dataset(
        materialized_root, frame, spec=make_processed_dataset_spec("materialized")
    )
    write_dataset(
        chunked_root,
        (frame.iloc[:1].copy(), frame.iloc[1:].copy()),
        spec=make_processed_dataset_spec("chunked"),
    )

    materialized = read_dataset(materialized_root)
    chunked = read_dataset(chunked_root)
    pd.testing.assert_frame_equal(
        materialized.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        chunked.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
    )
    assert (
        fingerprint_dataset(materialized_root).content_hash
        == fingerprint_dataset(chunked_root).content_hash
    )
    pd.testing.assert_frame_equal(
        materialized.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        frame.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
            drop=True
        ),
        check_dtype=False,
    )


def test_writer_reserves_complete_fixed_owner_before_sqlite_or_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenInput:
        def __iter__(self) -> ForbiddenInput:
            return self

        def __next__(self) -> pd.DataFrame:
            raise AssertionError("input_pulled_before_storage_owner_admission")

        def close(self) -> None:
            pass

    def reject_connect(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("sqlite_opened_before_storage_owner_admission")

    monkeypatch.setattr(dataset_inventory.sqlite3, "connect", reject_connect)
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=(
            dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + dataset_inventory._DATASET_INVENTORY_SQLITE_STARTUP_RESERVE_BYTES
            - 1
        ),
    )

    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "fixed_owner_admission"),
            ForbiddenInput(),
            spec=make_processed_dataset_spec("fixed_owner_admission"),
            resource_policy=policy,
        )


def test_private_writer_admits_producer_bound_before_data_pull(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=2 * 1024 * 1024,
    )
    pulls = 0

    def producer():
        nonlocal pulls
        yield policy.memory_budget_bytes
        pulls += 1
        frame = make_default_dataset_frame().iloc[:1].copy()
        yield runtime_resources.OwnedValue(frame, None)  # type: ignore[arg-type]

    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "producer_bound_admission"),
            producer(),
            spec=make_processed_dataset_spec("producer_bound_admission"),
            resource_policy=policy,
        )

    assert pulls == 0


def test_private_writer_rejects_actual_frame_above_declared_bound(
    tmp_path: Path,
) -> None:
    policy = capture_storage_test_policy()

    def producer():
        lease = yield 1
        frame = make_default_dataset_frame()
        yield runtime_resources.OwnedValue(frame, lease)
        del frame

    with pytest.raises(ValueError, match="dataset_producer_bound_invalid"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "producer_bound_invalid"),
            producer(),
            spec=make_processed_dataset_spec("producer_bound_invalid"),
            resource_policy=policy,
        )
    assert policy.used_memory_bytes == 0


def test_private_writer_transfers_one_lease_and_clears_before_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    events: list[tuple[str, bool]] = []
    lease_holder: dict[str, object] = {}
    frame_reference: weakref.ReferenceType[pd.DataFrame] | None = None
    producer_totals: list[int] = []
    original_close = runtime_resources.MemoryLease.close

    def observed_close(lease) -> None:  # noqa: ANN001
        if lease is lease_holder.get("lease"):
            events.append(
                (
                    "close",
                    frame_reference is not None and frame_reference() is None,
                )
            )
        original_close(lease)

    monkeypatch.setattr(runtime_resources.MemoryLease, "close", observed_close)

    def producer():
        nonlocal frame_reference
        lease = yield 128 * 1024
        lease_holder["lease"] = lease
        producer_totals.append(policy.used_memory_bytes)
        frame = make_default_dataset_frame().iloc[:1].copy()
        frame_reference = weakref.ref(frame)
        owned = runtime_resources.OwnedValue(frame, lease)
        producer_totals.append(policy.used_memory_bytes)
        yield owned
        del owned, frame
        events.append(("cleared", frame_reference() is None))

    dataset_writing.write_dataset(
        make_dataset_artifact_root(tmp_path, "producer_lease_transfer"),
        producer(),
        spec=make_processed_dataset_spec("producer_lease_transfer"),
        resource_policy=policy,
    )

    assert producer_totals[0] == producer_totals[1]
    assert events == [("cleared", True), ("close", True)]


def test_private_writer_handoffs_pending_lease_without_second_reservation(
    tmp_path: Path,
) -> None:
    policy = capture_storage_test_policy()
    frame = make_default_dataset_frame()
    events: list[tuple[bool, bool]] = []

    def producer():
        first_lease = yield 256 * 1024
        yield runtime_resources.OwnedValue(frame.iloc[:1].copy(), first_lease)
        pending_lease = policy.lease("source.pending", 1)
        handoff_lease = yield (256 * 1024, pending_lease)
        events.append(
            (
                handoff_lease is pending_lease,
                handoff_lease.reserved_bytes >= 256 * 1024,
            )
        )
        yield runtime_resources.OwnedValue(frame.iloc[1:].copy(), handoff_lease)

    dataset_writing.write_dataset(
        make_dataset_artifact_root(tmp_path, "pending_lease_handoff"),
        producer(),
        spec=make_processed_dataset_spec("pending_lease_handoff"),
        resource_policy=policy,
    )

    assert events == [(True, True)]
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    ("empty_first", "empty_last", "request_after_last"),
    (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ),
)
def test_private_writer_keeps_same_producer_lease_across_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_first: bool,
    empty_last: bool,
    request_after_last: bool,
) -> None:
    policy = capture_storage_test_policy()
    expected = make_default_dataset_frame()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    expected[["market_id", "asset_id"]] = expected[["market_id", "asset_id"]].astype(
        "string"
    )
    holder = {}
    events = []
    original_close = runtime_resources.MemoryLease.close

    def observed_close(lease):
        if lease is holder.get("lease"):
            events.append(("lease_closed", holder["retained"]() is None))
        original_close(lease)

    monkeypatch.setattr(runtime_resources.MemoryLease, "close", observed_close)

    def producer():
        lease = yield 256 * 1024
        holder["lease"] = lease
        retained = expected.copy()
        holder["retained"] = weakref.ref(retained)
        try:
            split = 0 if empty_first else 1
            spans = [(0, split), (split, len(retained))]
            if empty_last:
                spans.append((0, 0))
            for index, (start, end) in enumerate(spans):
                assert lease.active
                assert lease.reserved_bytes == 256 * 1024
                yield runtime_resources.OwnedValue(
                    retained.iloc[start:end].copy(), lease
                )
                if index + 1 < len(spans) or request_after_last:
                    resumed = yield (256 * 1024, lease)
                    assert resumed is lease
        finally:
            del retained
            events.append(("producer_closed", lease.active))

    def validate(_receipt):
        assert events == [("producer_closed", True), ("lease_closed", True)]

    root = make_dataset_artifact_root(tmp_path, "continuous_producer_lease")
    dataset_writing.write_dataset(
        root,
        producer(),
        spec=make_processed_dataset_spec("continuous_producer_lease"),
        resource_policy=policy,
        staged_validator=validate,
    )

    assert events == [("producer_closed", True), ("lease_closed", True)]
    assert policy.state == "CLOSED"
    assert policy.used_memory_bytes == 0
    pd.testing.assert_frame_equal(
        read_dataset(root)
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True),
        expected.sort_values(["timestamp", "asset_id"]).reset_index(drop=True),
    )


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    (
        ("zero", ValueError, "invalid_dataset_producer_handoff"),
        ("bool", ValueError, "invalid_dataset_producer_handoff"),
        ("grow", MemoryError, "runtime_memory_budget_exceeded"),
        ("producer", RuntimeError, "continuous_producer_primary"),
        ("writer", RuntimeError, "continuous_producer_primary"),
        ("cancel", KeyboardInterrupt, "continuous_producer_primary"),
        ("close", GeneratorExit, "continuous_producer_primary"),
    ),
)
def test_private_writer_closes_continuous_producer_before_its_lease_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error_type: type[BaseException],
    message: str,
) -> None:
    policy = capture_storage_test_policy()
    holder = {}
    events = []
    primary = error_type(message)
    original_close = runtime_resources.MemoryLease.close
    original_write = dataset_writing.write_parquet_file
    writes = 0

    def observed_close(lease):
        if lease is holder.get("lease"):
            events.append(("lease_closed", holder["retained"]() is None))
        original_close(lease)

    def observed_write(*args, **kwargs):
        nonlocal writes
        writes += 1
        if writes == 2 and failure in {"writer", "cancel", "close"}:
            raise primary
        return original_write(*args, **kwargs)

    monkeypatch.setattr(runtime_resources.MemoryLease, "close", observed_close)
    monkeypatch.setattr(dataset_writing, "write_parquet_file", observed_write)

    def producer():
        lease = yield 256 * 1024
        holder["lease"] = lease
        retained = make_default_dataset_frame()
        holder["retained"] = weakref.ref(retained)
        try:
            yield runtime_resources.OwnedValue(retained.iloc[:1].copy(), lease)
            bound = {
                "zero": 0,
                "bool": True,
                "grow": policy.memory_budget_bytes,
            }.get(failure, 256 * 1024)
            resumed = yield (bound, lease)
            assert resumed is lease
            assert lease.active
            if failure == "producer":
                raise primary
            yield runtime_resources.OwnedValue(retained.iloc[1:].copy(), lease)
        finally:
            del retained
            events.append(("producer_closed", lease.active))

    root = make_dataset_artifact_root(tmp_path, "continuous_producer_failure")
    with pytest.raises(error_type, match=message) as raised:
        dataset_writing.write_dataset(
            root,
            producer(),
            spec=make_processed_dataset_spec("continuous_producer_failure"),
            resource_policy=policy,
        )
    if failure in {"producer", "writer", "cancel", "close"}:
        assert raised.value is primary
    assert events == [("producer_closed", True), ("lease_closed", True)]
    assert policy.state == "CLOSED"
    assert policy.used_memory_bytes == 0
    assert not filesystem_exists(root)


@pytest.mark.parametrize("handoff_kind", ("malformed", "inactive", "foreign"))
def test_private_writer_rejects_invalid_pending_lease_handoff(
    tmp_path: Path,
    handoff_kind: str,
) -> None:
    policy = capture_storage_test_policy()
    foreign_policy: RuntimeResourcePolicy | None = None
    foreign_lease: runtime_resources.MemoryLease | None = None
    if handoff_kind == "foreign":
        foreign_policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
        foreign_lease = runtime_resources.MemoryLease(
            "foreign.pending",
            1,
            True,
            foreign_policy,
        )
        foreign_policy._leases.add(foreign_lease)

    def producer():
        first_lease = yield 256 * 1024
        yield runtime_resources.OwnedValue(
            make_default_dataset_frame().iloc[:1].copy(),
            first_lease,
        )
        if handoff_kind == "malformed":
            yield (256 * 1024, object())
        elif handoff_kind == "inactive":
            inactive = policy.lease("source.pending", 1)
            inactive.close()
            yield (256 * 1024, inactive)
        else:
            assert foreign_lease is not None
            yield (256 * 1024, foreign_lease)

    try:
        with pytest.raises(ValueError, match="invalid_dataset_producer_handoff"):
            dataset_writing.write_dataset(
                make_dataset_artifact_root(tmp_path, f"invalid_handoff_{handoff_kind}"),
                producer(),
                spec=make_processed_dataset_spec(f"invalid_handoff_{handoff_kind}"),
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == 0
    finally:
        if foreign_policy is not None and foreign_lease is not None:
            foreign_policy._leases.discard(foreign_lease)
            foreign_lease.active = False


def test_private_writer_cleanup_restores_outer_root_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    monkeypatch.setattr(
        dataset_writing,
        "write_parquet_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("writer_primary")),
    )

    with runtime_resource_policy_scope(policy):
        baseline = policy.used_memory_bytes
        with pytest.raises(RuntimeError, match="writer_primary"):
            dataset_writing.write_dataset(
                make_dataset_artifact_root(tmp_path, "outer_root_baseline"),
                _bound_first_frames((make_default_dataset_frame().iloc[:1].copy(),)),
                spec=make_processed_dataset_spec("outer_root_baseline"),
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == baseline
    assert policy.state == "CLOSED"


def test_private_writer_close_failure_preserves_primary_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    primary = RuntimeError("writer_primary")
    monkeypatch.setattr(
        dataset_writing,
        "write_parquet_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    def producer():
        lease = yield 128 * 1024
        try:
            yield runtime_resources.OwnedValue(
                make_default_dataset_frame().iloc[:1].copy(),
                lease,
            )
        finally:
            raise OSError("producer_close_failed")

    def run() -> None:
        with runtime_resource_policy_scope(policy):
            dataset_writing.write_dataset(
                make_dataset_artifact_root(tmp_path, "producer_close_failure"),
                producer(),
                spec=make_processed_dataset_spec("producer_close_failure"),
                resource_policy=policy,
            )

    with pytest.raises(RuntimeError, match="writer_primary") as raised:
        run()
    assert raised.value is primary
    assert "dataset producer close also failed: OSError" in primary.__notes__
    assert policy.state == "CLOSED"
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("primary_type", (RuntimeError, KeyboardInterrupt, SystemExit))
def test_private_writer_closes_producer_before_releasing_lease_on_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_type: type[BaseException],
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024**2)
    events: list[tuple[str, bool]] = []
    lease_holder: dict[str, object] = {}
    frame_reference: weakref.ReferenceType[pd.DataFrame] | None = None
    original_close = runtime_resources.MemoryLease.close

    def observed_close(lease) -> None:  # noqa: ANN001
        if lease is lease_holder.get("lease"):
            events.append(
                (
                    "close",
                    frame_reference is not None and frame_reference() is None,
                )
            )
        original_close(lease)

    monkeypatch.setattr(runtime_resources.MemoryLease, "close", observed_close)
    monkeypatch.setattr(
        dataset_writing,
        "write_parquet_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            primary_type("producer_primary")
        ),
    )

    def producer():
        nonlocal frame_reference
        frame: pd.DataFrame | None = None
        owned = None
        try:
            lease = yield 128 * 1024
            lease_holder["lease"] = lease
            frame = make_default_dataset_frame().iloc[:1].copy()
            frame_reference = weakref.ref(frame)
            owned = runtime_resources.OwnedValue(frame, lease)
            yield owned
        finally:
            owned = None
            frame = None
            events.append(
                ("cleared", frame_reference is not None and frame_reference() is None)
            )

    with pytest.raises(primary_type, match="producer_primary"):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(
                tmp_path, f"producer_primary_{primary_type.__name__}"
            ),
            producer(),
            spec=make_processed_dataset_spec(
                f"producer_primary_{primary_type.__name__}"
            ),
            resource_policy=policy,
        )

    assert events[0][0] == "cleared"
    assert events[-1] == ("close", True)


@pytest.mark.parametrize(
    "exit_mode",
    (
        "success",
        "runtime_error",
        "keyboard_interrupt",
        "system_exit",
        "close",
        "target_exists",
    ),
)
def test_writer_closes_actual_input_iterator_on_every_exit_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
) -> None:
    frame = make_default_dataset_frame().iloc[:1].copy()

    class CloseableChunks:
        def __init__(self) -> None:
            self._chunks = iter((frame,))
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._chunks)

        def close(self) -> None:
            self.close_calls += 1

    source = CloseableChunks()
    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
        "close": GeneratorExit,
    }
    failure_type = failure_types.get(exit_mode)
    if failure_type is not None:

        def fail_write(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)
    root = make_dataset_artifact_root(tmp_path, f"writer_close_{exit_mode}")
    if exit_mode == "target_exists":
        target_source = _bound_first_frames(source)
        next(target_source)
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((frame,)),
            spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
            resource_policy=current_runtime_resource_policy(),
        )
        with pytest.raises(ValueError, match="dataset_exists"):
            dataset_writing.write_dataset(
                root,
                target_source,
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
                resource_policy=current_runtime_resource_policy(),
            )
    elif failure_type is None:
        dataset_writing.write_dataset(
            root,
            _bound_first_frames(source),
            spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
            resource_policy=current_runtime_resource_policy(),
        )
    else:
        with pytest.raises(failure_type, match="writer_exit"):
            dataset_writing.write_dataset(
                root,
                _bound_first_frames(source),
                spec=make_processed_dataset_spec(f"writer_close_{exit_mode}"),
                resource_policy=current_runtime_resource_policy(),
            )

    assert source.close_calls == 1


@pytest.mark.parametrize(
    ("exit_mode", "expected_error", "expected_message"),
    (
        ("success", OSError, "source_close_failure"),
        ("runtime_error", RuntimeError, "writer_exit"),
        ("keyboard_interrupt", KeyboardInterrupt, "writer_exit"),
        ("system_exit", SystemExit, "writer_exit"),
        ("target_exists", ValueError, "dataset_exists"),
    ),
)
def test_writer_close_failure_precedes_publish_but_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exit_mode: str,
    expected_error: type[BaseException],
    expected_message: str,
) -> None:
    monkeypatch.setattr(runtime_resources, "_PROCESS_POISONED", False)
    frame = make_default_dataset_frame().iloc[:1].copy()

    class ThrowingCloseIterator:
        def __init__(self) -> None:
            self._chunks = iter((frame,))
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self) -> pd.DataFrame:
            return next(self._chunks)

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("source_close_failure")

    class IterableFacade:
        def __init__(self, source: ThrowingCloseIterator) -> None:
            self.source = source
            self.close_calls = 0

        def __iter__(self):  # noqa: ANN204
            return self.source

        def close(self) -> None:
            self.close_calls += 1

    source = ThrowingCloseIterator()
    facade = IterableFacade(source)
    failure_types = {
        "runtime_error": RuntimeError,
        "keyboard_interrupt": KeyboardInterrupt,
        "system_exit": SystemExit,
    }
    failure_type = failure_types.get(exit_mode)
    if failure_type is not None:

        def fail_write(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            raise failure_type("writer_exit")

        monkeypatch.setattr(dataset_writing, "write_parquet_file", fail_write)

    root = make_dataset_artifact_root(tmp_path, f"writer_close_failure_{exit_mode}")
    cleanup_policies: list[RuntimeResourcePolicy] = []
    if exit_mode == "target_exists":
        target_source = _bound_first_frames(facade)
        next(target_source)
        first_policy = capture_storage_test_policy()
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((frame,)),
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
            resource_policy=first_policy,
        )
        cleanup_policies.append(first_policy)
        input_data = target_source
    else:
        input_data = _bound_first_frames(facade)
    issued_policy = capture_storage_test_policy()
    cleanup_policies.append(issued_policy)

    with pytest.raises(expected_error, match=expected_message):
        dataset_writing.write_dataset(
            root,
            input_data,
            spec=make_processed_dataset_spec(f"writer_close_failure_{exit_mode}"),
            resource_policy=issued_policy,
        )

    assert source.close_calls == 1
    assert facade.close_calls == 0
    assert filesystem_exists(root) is (exit_mode == "target_exists")
    for policy in cleanup_policies:
        if policy.state == "POISONED":
            policy._lock.close()  # noqa: SLF001


def test_write_dataset_preserves_schema_for_empty_chunks(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "empty_schema")
    empty_chunk = make_default_dataset_frame().iloc[:0].copy()

    write_dataset(root, [empty_chunk], spec=make_processed_dataset_spec("empty_schema"))

    manifest = json.loads(
        filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
    )
    materialized = read_dataset(root)
    inspection = inspect_dataset(root)
    assert manifest["row_count"] == 0
    assert manifest["columns"] == ["market_id", "asset_id", "timestamp", "value"]
    assert manifest["file_count"] == 1
    assert materialized.empty
    assert materialized.columns.tolist() == manifest["columns"]
    assert inspection.row_count == 0
    assert inspection.columns == tuple(manifest["columns"])


def test_write_dataset_rejects_invalid_empty_and_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="invalid_dataset_write_input"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "invalid"),
            [make_default_dataset_frame().iloc[:1], {"not": "a frame"}],
            spec=make_processed_dataset_spec("invalid"),
        )

    with pytest.raises(ValueError, match="empty_dataset_schema_missing"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "empty_iterable"),
            (),
            spec=make_processed_dataset_spec("empty_iterable"),
        )

    duplicate = pd.concat(
        [make_default_dataset_frame().iloc[:1], make_default_dataset_frame().iloc[:1]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "duplicate"),
            duplicate,
            spec=make_processed_dataset_spec("duplicate"),
        )

    null_key = make_default_dataset_frame()
    null_key.loc[0, "asset_id"] = None
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "null_key"),
            null_key,
            spec=make_processed_dataset_spec("null_key"),
        )

    infinite = make_default_dataset_frame()
    infinite.loc[0, "value"] = -math.inf
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            make_dataset_artifact_root(tmp_path, "infinite"),
            infinite,
            spec=make_processed_dataset_spec("infinite"),
        )

    missing_required = make_default_dataset_frame().drop(columns=["value"])
    missing_root = make_dataset_artifact_root(tmp_path, "missing_required")
    with pytest.raises(ValueError, match="missing_required_columns: value"):
        write_dataset(
            missing_root,
            missing_required,
            spec=make_processed_dataset_spec("missing_required"),
        )
    assert not filesystem_exists(missing_root)


@pytest.mark.parametrize(
    "equivalent_asset_ids",
    [
        pd.Series([1, 1.0], dtype="object"),
        pd.Series([np.int64(1), 1], dtype="object"),
        pd.Series([date(2026, 1, 1), datetime(2026, 1, 1)], dtype="object"),
    ],
    ids=["int-float", "numpy-python", "date-datetime"],
)
def test_write_dataset_rejects_keys_equal_after_arrow_conversion(
    tmp_path: Path,
    equivalent_asset_ids: pd.Series,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "arrow_key_identity")
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": equivalent_asset_ids,
            "timestamp": [pd.Timestamp("2026-01-02")] * 2,
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec("arrow_key_identity"),
        )


def test_writer_key_index_uses_private_temporary_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch_root = tmp_path / "sqlite-temp"
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir",
        lambda: str(scratch_root),
    )
    observed_database_paths: list[Path] = []
    original_tracker = dataset_writing.ExactPrimaryKeyIndex

    class RecordingTracker(original_tracker):
        def add_table(self, table: pa.Table, *, namespace: str = "dataset") -> int:
            observed_database_paths.append(self.database_path)
            return super().add_table(table, namespace=namespace)

    monkeypatch.setattr(dataset_writing, "ExactPrimaryKeyIndex", RecordingTracker)
    root = make_dataset_artifact_root(tmp_path, "staged_key_scratch")

    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("staged_key_scratch"),
    )

    assert observed_database_paths
    assert all(
        path.parent.parent == scratch_root and path.parent.name.startswith(".k-")
        for path in observed_database_paths
    )
    assert not any(filesystem_exists(path) for path in observed_database_paths)
    assert not list(filesystem_glob(root, "**/.owner.start"))


def test_single_short_string_key_row_uses_measured_conversion_size() -> None:
    batch = pa.record_batch(
        [
            pa.array(["a_share"], type=pa.large_string()),
            pa.array(["000001"], type=pa.large_string()),
            pa.array([pd.Timestamp("2014-01-02")], type=pa.timestamp("ns")),
        ],
        names=["market_id", "asset_id", "timestamp"],
    )

    chunks = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=4_096,
        )
    )

    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], batch.to_pandas(use_threads=False))


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64"))
def test_float_primary_key_zero_identity_publication(tmp_path, dtype):
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["asset_id"] = np.array([-0.0, 0.0], dtype=dtype)
    frame["timestamp"] = pd.Timestamp("2020-01-01")
    root = make_dataset_artifact_root(tmp_path, "float_key")
    spec = make_processed_dataset_spec("float_key")
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        write_dataset(root, frame, spec=spec)
    assert not filesystem_exists(root)
    assert not list(filesystem_glob(root.parent, ".a-*"))
    frame["asset_id"] = np.array([1.0, 2.0], dtype=dtype)
    write_dataset(root, frame, spec=spec)
    restored = read_dataset(root)
    assert restored["asset_id"].dtype == frame["asset_id"].dtype
    assert sorted(restored["asset_id"]) == [1.0, 2.0]
    with pytest.raises(ValueError, match="unsupported_dataset_fingerprint_value"):
        fingerprint_dataset(root)
    comparison = compare_datasets(root, root)
    assert comparison.status == "INSPECTION_FAILED"


@pytest.mark.parametrize(
    "mutation",
    (
        "reverse",
        "duplicate_date",
        "date_format",
        "empty",
        "missing_hash",
        "schema",
        "duplicate_identity",
        "nested",
        "hash",
    ),
)
def test_archive_source_identity_is_validated_on_write_and_read(tmp_path, mutation):
    from copy import deepcopy

    from quant_research_platform.sources.tardis.raw import SCHEMA_ID

    identity = {
        "schema_id": SCHEMA_ID,
        "archives": [
            {
                "date": day,
                "schema_id": SCHEMA_ID,
                "artifact_id": "native_archive_" + token * 24,
                "manifest_hash": token * 64,
            }
            for day, token in (("2026-07-01", "a"), ("2026-08-01", "b"))
        ],
    }
    spec = make_processed_dataset_spec("archive_identity")
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=spec,
        instance_metadata={"source_artifact_identity": identity},
    )
    assert inspect_dataset(root).manifest["source_artifact_identity"] == identity
    assert len(read_dataset(root)) == 3
    invalid = deepcopy(identity)
    archives = invalid["archives"]
    if mutation == "reverse":
        archives.reverse()
    elif mutation == "duplicate_date":
        archives[1]["date"] = archives[0]["date"]
    elif mutation == "date_format":
        archives[0]["date"] = "20260701"
    elif mutation == "empty":
        archives.clear()
    elif mutation == "missing_hash":
        del archives[0]["manifest_hash"]
    elif mutation == "schema":
        archives[0]["schema_id"] = "other"
    elif mutation == "duplicate_identity":
        archives[1]["artifact_id"] = archives[0]["artifact_id"]
        archives[1]["manifest_hash"] = archives[0]["manifest_hash"]
    elif mutation == "nested":
        archives[0] = identity
    else:
        archives[0]["manifest_hash"] = "bad"
    candidate = make_dataset_artifact_root(tmp_path / "candidate", spec.data_id)
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        write_dataset(
            candidate,
            make_default_dataset_frame(),
            spec=spec,
            instance_metadata={"source_artifact_identity": invalid},
        )
    assert not candidate.exists()
    assert inspect_dataset(root).manifest["source_artifact_identity"] == identity
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["source_artifact_identity"] = invalid
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid_source_artifact_identity"):
        read_dataset(root)


@pytest.mark.parametrize("handoff", (False, True))
@pytest.mark.parametrize("explicit_cap", (None, 8 * 1024**2))
def test_private_writer_rejects_unadmitted_physical_bound_before_send(
    tmp_path, handoff, explicit_cap
):
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2, explicit_cap_bytes=explicit_cap
    )
    entered = False

    def producer():
        nonlocal entered
        pending = policy.lease("test.pending", 128 * 1024) if handoff else None
        try:
            yield (16 * 1024**2, pending) if handoff else 16 * 1024**2
            entered = True
            raise AssertionError("producer_entered_before_admission")
        finally:
            if pending is not None:
                pending.close()

    error = ValueError if explicit_cap is None else MemoryError
    message = (
        "dataset_producer_working_set_exceeds_memory_capacity"
        if explicit_cap is None
        else "runtime_memory_budget_exceeded"
    )
    with pytest.raises(error, match=message):
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "physical_admission"),
            producer(),
            spec=make_processed_dataset_spec("physical_admission"),
            resource_policy=policy,
        )
    assert not entered
    assert policy.used_memory_bytes == 0


def test_private_writer_admits_only_new_handoff_bytes_above_soft_target(tmp_path):
    policy = capture_test_policy(available_memory_bytes=8 * 1024**2)
    entered = False
    primary = RuntimeError("producer_admitted")

    def producer():
        nonlocal entered
        with policy.lease("test.pending", 1024**2) as pending:
            with policy.lease(
                "test.retained", policy.operation_headroom_bytes - 768 * 1024
            ):
                assert policy.planning_headroom_bytes == 0
                received = yield (1536 * 1024, pending)
                assert received is pending
                assert received.reserved_bytes == 1536 * 1024
                entered = True
                raise primary

    with pytest.raises(RuntimeError, match="producer_admitted") as caught:
        dataset_writing.write_dataset(
            make_dataset_artifact_root(tmp_path, "incremental_admission"),
            producer(),
            spec=make_processed_dataset_spec("incremental_admission"),
            resource_policy=policy,
        )
    assert caught.value is primary
    assert entered
    assert policy.used_memory_bytes == 0


def _footer_boundary_frame_and_spec(data_id, *, partitioned):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole

    rows = 600
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["a_share"] * rows, dtype=object),
            "asset_id": pd.Series(
                ["asset_a", "asset_b", "asset_c"] * 200
                if partitioned
                else [f"a{index:06d}" for index in range(rows)],
                dtype=object,
            ),
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="B")
            .repeat(3)
            .astype("datetime64[ns]"),
            **{
                f"factor_{index:02d}": np.arange(rows, dtype=float)
                for index in range(34)
            },
            "note": pd.Series(
                [None] * 220 + [f"value_{index:04d}" for index in range(220, rows)],
                dtype=object,
            ),
            "category": pd.Categorical(
                ["shared", "observed"] * 300,
                categories=["shared", "observed", "unused_" + "汉" * 128],
                ordered=True,
            ),
        }
    )
    frame.loc[rows - 1, "note"] = "汉" * 128
    frame = pd.concat(
        (frame.iloc[220:320], frame.iloc[:220], frame.iloc[320:]), ignore_index=True
    )
    spec = replace(
        make_processed_dataset_spec(data_id),
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            **{f"factor_{index:02d}": ColumnRole.FACTOR for index in range(34)},
            "note": ColumnRole.CONTEXT,
            "category": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp")
        if partitioned
        else ("market_id", "asset_id"),
        partition_columns=("year",) if partitioned else (),
    )
    return frame, spec


@pytest.mark.parametrize("partitioned", (False, True))
def test_writer_plans_complete_footer_and_preserves_bounded_logical_schema(
    tmp_path, monkeypatch, partitioned
):
    frame, spec = _footer_boundary_frame_and_spec(
        "footer_boundary", partitioned=partitioned
    )
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=7 * 1024**2,
    )
    writes = []
    original_write = dataset_writing.pq.write_table

    def observe_write(table, *args, **kwargs):
        note = table.column("note")
        writes.append((table.num_rows, note.null_count == table.num_rows))
        assert pa.types.is_string(note.type) or pa.types.is_large_string(note.type)
        assert pa.types.is_dictionary(table.schema.field("category").type)
        minimum_native = dataset_writing._parquet_row_group_footer_upper_bound(
            table.schema, rows=table.num_rows
        ) + dataset_writing._table_parquet_page_working_set_bytes(table, rows=1)
        native_leases = [
            lease
            for lease in policy._leases
            if lease.owner == "storage.parquet.native" and lease.active
        ]
        assert len(native_leases) == 1
        assert native_leases[0].reserved_bytes >= minimum_native
        return original_write(table, *args, **kwargs)

    monkeypatch.setattr(dataset_writing.pq, "write_table", observe_write)
    dataset_writing.write_dataset(
        root,
        _bound_first_frames(
            (frame,), bound=runtime_resources.pandas_frame_payload_bytes(frame)
        ),
        spec=spec,
        resource_policy=policy,
    )
    assert len(writes) > 1
    assert sum(rows for rows, _ in writes) == len(frame)
    assert any(all_null for _, all_null in writes)
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"
    expected = frame.sort_values(list(spec.primary_key)).reset_index(drop=True)
    actual = (
        read_dataset(root).sort_values(list(spec.primary_key)).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        actual.drop(columns="note"), expected.drop(columns="note")
    )
    assert actual["note"].dtype == expected["note"].dtype
    missing = expected["note"].isna()
    pd.testing.assert_series_equal(actual["note"].isna(), missing)
    pd.testing.assert_series_equal(
        actual.loc[~missing, "note"], expected.loc[~missing, "note"]
    )


def test_writer_rejects_when_single_row_and_complete_footer_do_not_fit(tmp_path):
    from tests.storage.dataset_test_support import _persisted_table

    previous = make_dataset_artifact_root(tmp_path, "previous_good")
    write_dataset(
        previous,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("previous_good"),
    )
    previous_bytes = {
        path.relative_to(previous): path.read_bytes()
        for path in previous.rglob("*")
        if path.is_file()
    }
    frame, spec = _footer_boundary_frame_and_spec("footer_nonfit", partitioned=True)
    frame = frame.iloc[:1].copy()
    persisted = dataset_writing.table_with_decoded_row_bounds(_persisted_table(frame))
    footer_bytes = dataset_writing._parquet_row_group_footer_upper_bound(
        persisted.schema, rows=1
    )
    minimum_page_bytes = dataset_writing._table_parquet_page_working_set_bytes(
        persisted, rows=1
    )
    del persisted
    frame_bytes = runtime_resources.pandas_frame_payload_bytes(frame)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=(
            dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + frame_bytes
            + minimum_page_bytes
            + footer_bytes // 2
        ),
    )
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    with pytest.raises(
        ValueError, match="dataset_working_set_exceeds_task_memory_budget"
    ):
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((frame,), bound=frame_bytes),
            spec=spec,
            resource_policy=policy,
        )
    assert not root.exists()
    assert set(root.parent.iterdir()) == {previous}
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"
    assert previous_bytes == {
        path.relative_to(previous): path.read_bytes()
        for path in previous.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("failure_stage", ("prepare", "arrow", "metadata"))
def test_writer_footer_preview_failure_drops_owned_objects_before_release(
    tmp_path, monkeypatch, failure_stage
):
    policy = capture_storage_test_policy()
    root = make_dataset_artifact_root(tmp_path, "preview_failure")
    frame_refs = []
    table_refs = []
    releases = []
    primary = KeyboardInterrupt("footer_preview_primary")
    original_prepare = dataset_writing._WriteStats.prepare_frame
    original_table = dataset_writing._WriteStats.table_for_frame
    original_bounds = dataset_writing.table_with_decoded_row_bounds
    original_close = runtime_resources.MemoryLease.close

    def prepare(stats, frame):
        prepared, logical = original_prepare(stats, frame)
        frame_refs.append(weakref.ref(prepared))
        if failure_stage == "prepare":
            raise primary
        return prepared, logical

    def table_for_frame(stats, frame, **kwargs):
        table = original_table(stats, frame, **kwargs)
        table_refs.append(weakref.ref(table))
        if failure_stage == "arrow":
            raise primary
        return table

    def with_bounds(table):
        persisted = original_bounds(table)
        table_refs.append(weakref.ref(persisted))
        if failure_stage == "metadata":
            raise primary
        return persisted

    def close(lease):
        if lease.active and lease.owner == "storage.dataset.arrow":
            releases.append(
                ("arrow", all(reference() is None for reference in table_refs))
            )
        if lease.active and lease.owner == "storage.dataset.prepare":
            releases.append(
                ("prepare", all(reference() is None for reference in frame_refs))
            )
        return original_close(lease)

    monkeypatch.setattr(dataset_writing._WriteStats, "prepare_frame", prepare)
    monkeypatch.setattr(dataset_writing._WriteStats, "table_for_frame", table_for_frame)
    monkeypatch.setattr(dataset_writing, "table_with_decoded_row_bounds", with_bounds)
    monkeypatch.setattr(runtime_resources.MemoryLease, "close", close)
    with pytest.raises(KeyboardInterrupt, match="footer_preview_primary") as caught:
        dataset_writing.write_dataset(
            root,
            _bound_first_frames((make_default_dataset_frame(),)),
            spec=make_processed_dataset_spec("preview_failure"),
            resource_policy=policy,
        )
    assert caught.value is primary
    assert primary.__traceback__ is not None
    expected_owners = (
        ["prepare"] if failure_stage == "prepare" else ["arrow", "prepare"]
    )
    assert releases == [(owner, True) for owner in expected_owners]
    assert frame_refs
    assert not root.exists()
    assert list(root.parent.iterdir()) == []
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


def _schema_name_frame_and_spec(data_id, *, schema_kind, width):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole

    name = "k" * width
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["a_share"] * 4, dtype=object),
            "asset_id": pd.Series(
                [f"{index:06d}" for index in range(1, 5)], dtype=object
            ),
            "timestamp": pd.Series(
                pd.Timestamp("2024-01-02"), index=range(4), dtype="datetime64[ns]"
            ),
        }
    )
    payload_column = name if schema_kind == "top_level" else "payload"
    frame[payload_column] = (
        pd.Series([1, 2, 3, 4], dtype="int64")
        if schema_kind == "top_level"
        else pd.Series([{name: {"items": [1, 2]}} for _ in range(4)], dtype=object)
    )
    spec = replace(
        make_processed_dataset_spec(data_id),
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            payload_column: ColumnRole.CONTEXT,
        },
    )
    return frame, spec, name


@pytest.mark.parametrize("schema_kind", ("top_level", "nested"))
def test_public_writer_admits_schema_names_before_arrow_and_round_trips(
    tmp_path, monkeypatch, schema_kind
):
    from tests.helpers.runtime_resources import _TestMachineLock

    policies = []
    reservations = {}
    payload_sizes = []
    expected_roots = set()
    original_capture = runtime_resources._capture_runtime_resource_policy
    original_table = dataset_writing._WriteStats.table_for_frame
    current_width = 0

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        policy = original_capture(explicit_cap_bytes=64 * 1024**2)
        policies.append(policy)
        return policy

    def observe_table(stats, frame, **kwargs):
        policy = current_runtime_resource_policy()
        assert policy is not None
        metadata = [
            lease.reserved_bytes
            for lease in policy._leases
            if lease.active and lease.owner == "storage.dataset.schema"
        ]
        assert len(metadata) == 1
        reservations[current_width] = max(
            reservations.get(current_width, 0), metadata[0]
        )
        return original_table(stats, frame, **kwargs)

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 128 * 1024**2
    )
    monkeypatch.setattr(
        runtime_resources._MachineLock, "acquire", lambda: _TestMachineLock()
    )
    monkeypatch.setattr(
        runtime_resources, "_capture_runtime_resource_policy", capture_policy
    )
    monkeypatch.setattr(dataset_writing._WriteStats, "table_for_frame", observe_table)
    for current_width in (8, 4096):
        frame, spec, name = _schema_name_frame_and_spec(
            f"schema_{schema_kind}_{current_width}",
            schema_kind=schema_kind,
            width=current_width,
        )
        payload_sizes.append(runtime_resources.pandas_frame_payload_bytes(frame))
        root = make_dataset_artifact_root(tmp_path, spec.data_id)
        expected_roots.add(root)
        write_dataset(root, frame, spec=spec)
        actual = read_dataset(root).sort_values("asset_id").reset_index(drop=True)
        if schema_kind == "top_level":
            pd.testing.assert_frame_equal(actual, frame)
        else:
            pd.testing.assert_frame_equal(
                actual.drop(columns="payload"), frame.drop(columns="payload")
            )
            assert actual["payload"].dtype == frame["payload"].dtype
            for value in actual["payload"]:
                assert set(value) == {name}
                assert set(value[name]) == {"items"}
                assert list(value[name]["items"]) == [1, 2]
            # Arrow returns nested lists as arrays; exercise their real rewrite path.
            rewritten = make_dataset_artifact_root(
                tmp_path / f"rewritten_{current_width}", spec.data_id
            )
            write_dataset(rewritten, actual, spec=spec)
            restored = (
                read_dataset(rewritten).sort_values("asset_id").reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(restored, actual)
            assert set(rewritten.parent.iterdir()) == {rewritten}
        assert set(root.parent.iterdir()) == expected_roots
    assert payload_sizes[0] == payload_sizes[1]
    assert reservations[4096] > reservations[8]
    assert policies
    assert all(
        policy.used_memory_bytes == 0 and policy.state == "CLOSED"
        for policy in policies
    )


@pytest.mark.parametrize("schema_kind", ("top_level", "nested"))
def test_public_writer_rejects_schema_nonfit_before_arrow_preserving_old_output(
    tmp_path, monkeypatch, schema_kind
):
    from tests.helpers.runtime_resources import _TestMachineLock

    previous = make_dataset_artifact_root(tmp_path, "schema_previous_good")
    write_dataset(
        previous,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("schema_previous_good"),
    )
    previous_bytes = {
        path.relative_to(previous): path.read_bytes()
        for path in previous.rglob("*")
        if path.is_file()
    }
    policies = []
    original_capture = runtime_resources._capture_runtime_resource_policy

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        policy = original_capture(
            explicit_cap_bytes=(
                dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
                + dataset_writing._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
                + 256 * 1024
            )
        )
        policies.append(policy)
        return policy

    def unexpected_arrow(*_args, **_kwargs):
        raise AssertionError("schema_nonfit_entered_arrow")

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 128 * 1024**2
    )
    monkeypatch.setattr(
        runtime_resources._MachineLock, "acquire", lambda: _TestMachineLock()
    )
    monkeypatch.setattr(
        runtime_resources, "_capture_runtime_resource_policy", capture_policy
    )
    monkeypatch.setattr(
        dataset_writing._WriteStats, "table_for_frame", unexpected_arrow
    )
    frame, spec, _name = _schema_name_frame_and_spec(
        f"schema_nonfit_{schema_kind}", schema_kind=schema_kind, width=4096
    )
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    with pytest.raises(
        ValueError, match="dataset_working_set_exceeds_task_memory_budget"
    ):
        write_dataset(root, frame, spec=spec)
    assert not root.exists()
    assert set(root.parent.iterdir()) == {previous}
    assert previous_bytes == {
        path.relative_to(previous): path.read_bytes()
        for path in previous.rglob("*")
        if path.is_file()
    }
    assert len(policies) == 1
    assert policies[0].used_memory_bytes == 0
    assert policies[0].state == "CLOSED"


def test_public_writer_omits_caller_attrs_before_copy_and_arrow_serialization(
    tmp_path, monkeypatch
):
    class CallerOnly:
        def __deepcopy__(self, _memo):
            raise AssertionError("caller_attrs_were_copied")

    expected = make_default_dataset_frame()
    expected["timestamp"] = expected["timestamp"].astype("datetime64[ns]")
    frames = (expected.iloc[:1].copy(), expected.iloc[1:].copy())
    caller_only = CallerOnly()
    payload = ["x" * 200_000, caller_only]
    for frame in frames:
        frame.attrs["caller_only"] = payload
    original_attrs = [frame.attrs for frame in frames]
    view_refs = []
    view_releases = []
    original_declared = dataset_writing._WriteStats.declared_frame
    original_close = runtime_resources.MemoryLease.close

    def declared(stats, frame, **kwargs):
        assert all(reference() is None for reference in view_refs)
        assert frame.attrs == {}
        policy = current_runtime_resource_policy()
        assert policy is not None
        assert (
            sum(
                lease.active and lease.owner == "storage.dataset.input.view"
                for lease in policy._leases
            )
            == 1
        )
        view_refs.append(weakref.ref(frame))
        return original_declared(stats, frame, **kwargs)

    def close(lease):
        if lease.active and lease.owner == "storage.dataset.input.view":
            view_releases.append(all(reference() is None for reference in view_refs))
        return original_close(lease)

    monkeypatch.setattr(dataset_writing._WriteStats, "declared_frame", declared)
    monkeypatch.setattr(runtime_resources.MemoryLease, "close", close)
    root = make_dataset_artifact_root(tmp_path, "caller_attrs")
    write_dataset(
        root,
        frames,
        spec=make_processed_dataset_spec("caller_attrs"),
        instance_metadata={"producer_diagnostics": {"label": "preserved"}},
    )
    primary_key = list(make_processed_dataset_spec("caller_attrs").primary_key)
    pd.testing.assert_frame_equal(
        read_dataset(root).sort_values(primary_key).reset_index(drop=True),
        expected.sort_values(primary_key).reset_index(drop=True),
    )
    assert len(view_refs) == len(frames)
    assert view_releases == [True] * len(frames)
    for frame, attrs in zip(frames, original_attrs, strict=True):
        assert frame.attrs is attrs
        assert frame.attrs["caller_only"] is payload
    assert payload[0] == "x" * 200_000
    assert payload[1] is caller_only
    manifest = json.loads((root / "_manifest" / "manifest.json").read_text())
    assert manifest["producer_diagnostics"] == {"label": "preserved"}


@pytest.mark.parametrize("cancel_write", (False, True))
def test_empty_publication_tables_die_before_arrow_release(
    tmp_path, monkeypatch, cancel_write
):
    references = []
    releases = []
    primary = KeyboardInterrupt("empty_publication_primary")
    original_tables = dataset_writing._bounded_persisted_tables
    original_write = dataset_writing.pq.write_table
    original_close = runtime_resources.MemoryLease.close

    def observe_tables(table, **kwargs):
        if table.num_rows == 0:
            references.append(weakref.ref(table))
        return original_tables(table, **kwargs)

    def write(table, *args, **kwargs):
        if table.num_rows == 0 and cancel_write:
            raise primary
        return original_write(table, *args, **kwargs)

    def close(lease):
        if lease.active and lease.owner == "storage.dataset.arrow" and references:
            releases.append(all(reference() is None for reference in references))
        return original_close(lease)

    monkeypatch.setattr(dataset_writing, "_bounded_persisted_tables", observe_tables)
    monkeypatch.setattr(dataset_writing.pq, "write_table", write)
    monkeypatch.setattr(runtime_resources.MemoryLease, "close", close)
    root = make_dataset_artifact_root(tmp_path, "empty_publication_lifetime")
    frame = make_default_dataset_frame().iloc[:0].copy()
    frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
    spec = make_processed_dataset_spec("empty_publication_lifetime")
    if cancel_write:
        with pytest.raises(
            KeyboardInterrupt, match="empty_publication_primary"
        ) as caught:
            write_dataset(root, frame, spec=spec)
        assert caught.value is primary
        assert primary.__traceback__ is not None
        assert not root.exists()
        assert list(root.parent.iterdir()) == []
    else:
        write_dataset(root, frame, spec=spec)
        pd.testing.assert_frame_equal(read_dataset(root), frame)
    assert references
    assert releases and all(releases)


@pytest.mark.parametrize(
    "leaf_kind",
    (
        "text",
        "bytes",
        "list",
        "unicode_array",
        "bytes_array",
        "numeric_array",
        "object_array",
        "memoryview",
        "noncontiguous_memoryview",
    ),
)
def test_public_writer_admits_nested_leaf_buffers_and_preserves_values(
    tmp_path, monkeypatch, leaf_kind
):
    from tests.helpers.runtime_resources import _TestMachineLock

    policies = []
    original_capture = runtime_resources._capture_runtime_resource_policy

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        policy = original_capture(explicit_cap_bytes=64 * 1024**2)
        policies.append(policy)
        return policy

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 128 * 1024**2
    )
    monkeypatch.setattr(
        runtime_resources._MachineLock, "acquire", lambda: _TestMachineLock()
    )
    monkeypatch.setattr(
        runtime_resources, "_capture_runtime_resource_policy", capture_policy
    )
    for width in (8, 65_536):
        frame, spec, name = _schema_name_frame_and_spec(
            f"nested_leaf_{leaf_kind}_{width}", schema_kind="nested", width=1
        )
        frame = frame.iloc[:1].copy()
        if leaf_kind == "text":
            leaf = "a" * width
        elif leaf_kind == "bytes":
            leaf = b"a" * width
        elif leaf_kind == "list":
            leaf = ["a" * width]
        elif leaf_kind == "unicode_array":
            leaf = np.array(["a" * width], dtype=f"U{width}")
        elif leaf_kind == "bytes_array":
            leaf = np.array([b"a" * width], dtype=f"S{width}")
        elif leaf_kind == "numeric_array":
            leaf = np.arange(width, dtype=np.int64)
        elif leaf_kind == "memoryview":
            leaf = memoryview(b"a" * width)
        elif leaf_kind == "noncontiguous_memoryview":
            leaf = memoryview(b"ab" * width)[::2]
        else:
            leaf = np.array(["a" * width], dtype=object)
        frame["payload"] = pd.Series([{name: leaf}], index=frame.index, dtype=object)
        root = make_dataset_artifact_root(tmp_path, spec.data_id)
        write_dataset(root, frame, spec=spec)
        restored = read_dataset(root)
        pd.testing.assert_frame_equal(
            restored.drop(columns="payload"), frame.drop(columns="payload")
        )
        assert restored["payload"].dtype == frame["payload"].dtype
        actual = restored["payload"].iloc[0]
        assert set(actual) == {name}
        if leaf_kind in ("memoryview", "noncontiguous_memoryview"):
            assert actual[name] == b"a" * width
        elif leaf_kind in ("text", "bytes"):
            assert actual[name] == leaf
        else:
            np.testing.assert_array_equal(actual[name], leaf)
    assert policies
    assert all(
        policy.used_memory_bytes == 0 and policy.state == "CLOSED"
        for policy in policies
    )


@pytest.mark.parametrize(
    "leaf_kind",
    ("text", "unicode_array", "object_array", "memoryview", "noncontiguous_memoryview"),
)
def test_public_writer_rejects_nested_leaf_nonfit_before_arrow(
    tmp_path, monkeypatch, leaf_kind
):
    from tests.helpers.runtime_resources import _TestMachineLock

    frame, spec, name = _schema_name_frame_and_spec(
        f"nested_leaf_nonfit_{leaf_kind}", schema_kind="nested", width=1
    )
    frame = frame.iloc[:1].copy()
    value = "a" * 200_000
    if leaf_kind == "text":
        leaf = value
    elif leaf_kind == "memoryview":
        leaf = memoryview(b"a" * 200_000)
    elif leaf_kind == "noncontiguous_memoryview":
        leaf = memoryview(b"ab" * 200_000)[::2]
    else:
        leaf = np.array(
            [value], dtype="U200000" if leaf_kind == "unicode_array" else object
        )
    frame["payload"] = pd.Series([{name: leaf}], index=frame.index, dtype=object)
    original_capture = runtime_resources._capture_runtime_resource_policy
    policies = []

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        policy = original_capture(
            explicit_cap_bytes=(
                dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
                + dataset_writing._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
                + 256 * 1024
            )
        )
        policies.append(policy)
        return policy

    def unexpected_arrow(*_args, **_kwargs):
        raise AssertionError("nested_leaf_nonfit_entered_arrow")

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 128 * 1024**2
    )
    monkeypatch.setattr(
        runtime_resources._MachineLock, "acquire", lambda: _TestMachineLock()
    )
    monkeypatch.setattr(
        runtime_resources, "_capture_runtime_resource_policy", capture_policy
    )
    monkeypatch.setattr(
        dataset_writing._WriteStats, "table_for_frame", unexpected_arrow
    )
    root = make_dataset_artifact_root(tmp_path, spec.data_id)
    with pytest.raises(
        ValueError, match="dataset_working_set_exceeds_task_memory_budget"
    ):
        write_dataset(root, frame, spec=spec)
    assert not root.exists()
    assert list(root.parent.iterdir()) == []
    assert len(policies) == 1
    assert policies[0].used_memory_bytes == 0
    assert policies[0].state == "CLOSED"


@pytest.mark.parametrize("drift_kind", ("numeric_type", "nested_field"))
def test_public_writer_rejects_nonnull_object_schema_drift(tmp_path, drift_kind):
    from quant_research_platform.contracts.columns import ColumnRole

    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    initial = 1 if drift_kind == "numeric_type" else {"initial": 1}
    changed = 2.0 if drift_kind == "numeric_type" else {"initial": 2, "added": 3}
    first["value"] = pd.Series([initial], index=first.index, dtype=object)
    second["value"] = pd.Series([changed], index=second.index, dtype=object)
    root = make_dataset_artifact_root(tmp_path, "object_numeric_drift")
    spec = make_processed_dataset_spec(
        "object_numeric_drift", value_role=ColumnRole.CONTEXT
    )
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        write_dataset(root, (first, second), spec=spec)
    assert not root.exists()
    assert list(root.parent.iterdir()) == []


@pytest.mark.parametrize("cap_mib,nonnull_first", ((64, False), (7, False), (7, True)))
def test_public_writer_uses_complete_object_string_facts_before_bounded_schema(
    tmp_path, monkeypatch, cap_mib, nonnull_first
):
    from dataclasses import replace

    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.data import ProcessedDataKind

    rows = 6000
    frame = pd.DataFrame(
        {
            "asset_id": pd.Series(
                [f"synthetic_{index:04d}" for index in range(rows)], dtype=object
            ),
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="D").astype(
                "datetime64[ns]"
            ),
            **{
                f"factor_{index:02d}": np.arange(rows, dtype=np.float64)
                for index in range(34)
            },
            "note": pd.Series(
                [None] * 5500 + [f"note_{index:04d}" for index in range(5500, rows)],
                dtype=object,
            ),
        }
    )
    expected = frame.copy(deep=True)
    if nonnull_first:
        frame = pd.concat((frame.iloc[5500:], frame.iloc[:5500]), ignore_index=True)
    before = frame.copy(deep=True)
    spec = replace(
        make_processed_dataset_spec(
            "null_prefix_string",
            kind=ProcessedDataKind.PANEL,
            primary_key=("asset_id", "timestamp"),
        ),
        column_roles={
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            **{f"factor_{index:02d}": ColumnRole.FACTOR for index in range(34)},
            "note": ColumnRole.CONTEXT,
        },
    )
    root = make_dataset_artifact_root(
        tmp_path, spec.data_id, kind=ProcessedDataKind.PANEL
    )
    original_capture = runtime_resources._capture_runtime_resource_policy
    policies = []

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        policy = original_capture(explicit_cap_bytes=cap_mib * 1024**2)
        policies.append(policy)
        return policy

    with monkeypatch.context() as bounded:
        bounded.setattr(
            runtime_resources, "_capture_runtime_resource_policy", capture_policy
        )
        write_dataset(root, frame, spec=spec)
    assert len(policies) == 1
    assert policies[0].used_memory_bytes == 0
    assert policies[0].state == "CLOSED"

    # Readback uses its own default actual-capacity root, not the write cap.
    actual = (
        read_dataset(root).sort_values(list(spec.primary_key)).reset_index(drop=True)
    )
    expected = expected.sort_values(list(spec.primary_key)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual.drop(columns="note"), expected.drop(columns="note")
    )
    assert actual["note"].dtype == expected["note"].dtype
    pd.testing.assert_series_equal(actual["note"].isna(), expected["note"].isna())
    pd.testing.assert_series_equal(actual["note"].dropna(), expected["note"].dropna())
    assert len(actual) == 6000
    assert actual["note"].isna().sum() == 5500
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("failure_stage", ("plan", "view"))
def test_writer_admission_failure_clears_owned_item_before_input_release(
    tmp_path, monkeypatch, failure_stage
):
    from quant_research_platform.contracts.columns import ColumnRole

    class WeakMapping(dict):
        pass

    policy = capture_storage_test_policy()
    primary = MemoryError("input_view_admission_primary")
    holder = {}
    references = []
    closed = []
    original_lease = RuntimeResourcePolicy.lease
    original_close = runtime_resources.MemoryLease.close

    def lease(root_policy, owner, reserved_bytes):
        if owner == "storage.dataset.input.view" and failure_stage == "view":
            raise primary
        return original_lease(root_policy, owner, reserved_bytes)

    def close(acquired):
        if acquired is holder.get("lease"):
            closed.append(all(reference() is None for reference in references))
        return original_close(acquired)

    def producer():
        frame = payload = owned = None
        try:
            acquired = yield 128 * 1024
            holder["lease"] = acquired
            frame = make_default_dataset_frame(value_column="payload").iloc[:1].copy()
            payload = WeakMapping(
                {"k" * (100_000 if failure_stage == "plan" else 1): 1}
            )
            frame["payload"] = pd.Series([payload], index=frame.index, dtype=object)
            references.extend((weakref.ref(frame), weakref.ref(payload)))
            owned = runtime_resources.OwnedValue(frame, acquired)
            yield owned
        finally:
            frame = payload = owned = None

    monkeypatch.setattr(RuntimeResourcePolicy, "lease", lease)
    monkeypatch.setattr(runtime_resources.MemoryLease, "close", close)
    root = make_dataset_artifact_root(tmp_path, "input_admission_failure")
    spec = make_processed_dataset_spec(
        "input_admission_failure", value_column="payload", value_role=ColumnRole.CONTEXT
    )
    error_type = ValueError if failure_stage == "plan" else MemoryError
    message = (
        "dataset_working_set_exceeds_task_memory_budget"
        if failure_stage == "plan"
        else "input_view_admission_primary"
    )
    with pytest.raises(error_type, match=message) as caught:
        dataset_writing.write_dataset(
            root, producer(), spec=spec, resource_policy=policy
        )
    if failure_stage == "view":
        assert caught.value is primary
    assert caught.value.__traceback__ is not None
    assert references
    assert closed == [True]
    assert not root.exists()
    assert list(root.parent.iterdir()) == []
    assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"


@pytest.mark.parametrize("contiguous", (False, True))
@pytest.mark.parametrize("reject", (False, True))
def test_public_writer_root_memoryview_admission_preserves_binary_values(
    tmp_path, monkeypatch, contiguous, reject
):
    from tests.helpers.runtime_resources import _TestMachineLock

    policies = []
    original_capture = runtime_resources._capture_runtime_resource_policy

    def capture_policy(*, explicit_cap_bytes=None):
        assert explicit_cap_bytes is None
        cap = (
            dataset_writing._DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
            + dataset_writing._PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
            + 256 * 1024
            if reject
            else 64 * 1024**2
        )
        policy = original_capture(explicit_cap_bytes=cap)
        policies.append(policy)
        return policy

    def unexpected_arrow(*_args, **_kwargs):
        raise AssertionError("root_memoryview_nonfit_entered_arrow")

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 128 * 1024**2
    )
    monkeypatch.setattr(
        runtime_resources._MachineLock, "acquire", lambda: _TestMachineLock()
    )
    monkeypatch.setattr(
        runtime_resources, "_capture_runtime_resource_policy", capture_policy
    )
    if reject:
        monkeypatch.setattr(
            dataset_writing._WriteStats, "table_for_frame", unexpected_arrow
        )
    for width in (200_000,) if reject else (8, 65_536):
        frame, spec, _name = _schema_name_frame_and_spec(
            f"root_memoryview_{contiguous}_{width}", schema_kind="nested", width=1
        )
        frame = frame.iloc[:1].copy()
        view = (
            memoryview(b"a" * width) if contiguous else memoryview(b"ab" * width)[::2]
        )
        frame.at[frame.index[0], "payload"] = view
        assert frame["payload"].iloc[0] is view
        root = make_dataset_artifact_root(tmp_path, spec.data_id)
        if reject:
            with pytest.raises(
                ValueError, match="dataset_working_set_exceeds_task_memory_budget"
            ):
                write_dataset(root, frame, spec=spec)
            assert not root.exists()
            assert list(root.parent.iterdir()) == []
        else:
            write_dataset(root, frame, spec=spec)
            actual = read_dataset(root)
            pd.testing.assert_frame_equal(
                actual.drop(columns="payload"), frame.drop(columns="payload")
            )
            assert actual["payload"].dtype == frame["payload"].dtype
            assert type(actual["payload"].iloc[0]) is bytes
            assert actual["payload"].iloc[0] == b"a" * width
        assert frame["payload"].iloc[0] is view
    assert policies
    assert all(
        policy.used_memory_bytes == 0 and policy.state == "CLOSED"
        for policy in policies
    )


@pytest.mark.parametrize("rows", [2, 1024])
def test_incremental_native_footer_is_admitted_after_final_batch_selection(
    tmp_path, monkeypatch, rows
):
    import io

    from quant_research_platform.storage import parquet as owner

    monkeypatch.setattr(
        runtime_resources, "_available_memory_bytes", lambda: 32 * 1024**2
    )
    table = pa.table({f"column_{index}": range(rows) for index in range(4)})
    reached = False
    original = owner.pq.ParquetWriter

    def construct(*args, **kwargs):
        nonlocal reached
        reached = True
        return original(*args, **kwargs)

    monkeypatch.setattr(owner.pq, "ParquetWriter", construct)
    with runtime_resources.runtime_resource_policy_scope() as policy:
        with policy.lease("tests.existing", policy.memory_budget_bytes):
            assert policy.planning_headroom_bytes == 0
            handle = io.BytesIO()
            if rows == 1024:
                with pytest.raises(ValueError, match="runtime_memory_budget_exceeded"):
                    with owner.parquet_table_writer(
                        handle, parent=tmp_path, compression="NONE"
                    ) as write:
                        write(table)
                assert not reached
            else:
                with owner.parquet_table_writer(
                    handle, parent=tmp_path, compression="NONE"
                ) as write:
                    write(table)
                assert reached
                assert owner.pq.read_table(io.BytesIO(handle.getvalue())).equals(table)
        assert policy.used_memory_bytes == 0
    assert policy.state == "CLOSED"
