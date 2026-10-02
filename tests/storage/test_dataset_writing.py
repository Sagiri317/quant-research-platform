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
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.dataset_test_support import (
    _bound_first_frames,
    _input_dataset_identity_payload,
    _writer_test_policy,
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
    policy = _writer_test_policy()

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
    policy = _writer_test_policy()
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
    policy = _writer_test_policy()
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
    policy = _writer_test_policy()
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
    policy = _writer_test_policy()
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
        first_policy = _writer_test_policy()
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
    issued_policy = _writer_test_policy()
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
