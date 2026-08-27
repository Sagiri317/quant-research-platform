from __future__ import annotations

import json
import os
import subprocess
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_research_platform.runtime.resources as runtime_resources
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.workflows import (
    validate_schema_inventory_summary,
)
from quant_research_platform.interfaces import private_cli
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    _OwnedValue,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.workflows.diagnostics import schema_inventory
from tests.helpers.runtime_resources import capture_test_policy


def _collect_inventory(
    root: Path,
    *,
    important_null_count_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    with schema_inventory.iter_schema_inventory_items(
        root,
        important_null_count_columns=important_null_count_columns,
    ) as items:
        return {"schema_id": "qrp-schema-inventory", "datasets": list(items)}


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_schema_inventory_does_not_descend_nested_junction(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    layer_root.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "spec.json").write_text("{}", encoding="utf-8")
    junction = layer_root / "nested"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert tuple(schema_inventory._iter_dataset_artifact_roots(tmp_path)) == ()


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_schema_inventory_rejects_junction_project_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    root = tmp_path / "project"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(external)],
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(root))


def test_schema_inventory_rejects_aliased_data_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external"
    (external / "processed").mkdir(parents=True)
    try:
        (root / "data").symlink_to(external, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(root))


def test_schema_inventory_rejects_alias_above_supplied_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    (target / "project" / "data" / "processed").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(alias / "project"))


def test_schema_inventory_rejects_spec_symlink(tmp_path: Path) -> None:
    layer_root = tmp_path / "data" / "processed"
    artifact_root = layer_root / "artifact"
    artifact_root.mkdir(parents=True)
    external = tmp_path / "spec.json"
    external.write_text("{}", encoding="utf-8")
    try:
        (artifact_root / "spec.json").symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        tuple(schema_inventory._iter_dataset_artifact_roots(tmp_path))


def test_schema_inventory_uses_storage_exact_duplicate_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_factor",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share"] * 3,
                "asset_id": ["asset_a", "asset_b", "asset_c"],
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="D"),
                "value": [1.0, 2.0, 3.0],
            }
        ),
        spec=spec,
    )

    monkeypatch.setattr(
        schema_inventory.pd.util,
        "hash_pandas_object",
        lambda values, **_kwargs: pd.Series(
            [0] * len(values),
            dtype="uint64",
        ),
    )

    payload = _collect_inventory(tmp_path)

    assert payload["datasets"][0]["duplicate_primary_key_count"] == 0


def test_schema_inventory_captures_one_policy_for_all_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
    )
    captures = {"count": 0}
    observed_policies: list[RuntimeResourcePolicy] = []
    observed_reserved_bytes: list[int] = []

    def capture_policy() -> RuntimeResourcePolicy:
        captures["count"] += 1
        return policy

    def inventory_item(
        _project_root: Path,
        artifact_root: Path,
        *,
        important_null_count_columns: frozenset[str],
        resource_policy: RuntimeResourcePolicy,
        retained_items,
    ) -> tuple[dict[str, object], int]:
        assert important_null_count_columns == frozenset()
        assert current_runtime_resource_policy() is resource_policy
        observed_policies.append(resource_policy)
        observed_reserved_bytes.append(resource_policy.used_memory_bytes)
        retained_items.enter_context(
            resource_policy.lease("schema.inventory.items", 64)
        )
        return {"logical_relative_root": artifact_root.name}, 64

    monkeypatch.setattr(
        runtime_resources,
        "_capture_runtime_resource_policy",
        lambda **_kwargs: capture_policy(),
    )
    monkeypatch.setattr(
        schema_inventory,
        "_iter_dataset_artifact_roots",
        lambda _root: (tmp_path / "dataset-a", tmp_path / "dataset-b"),
    )
    monkeypatch.setattr(schema_inventory, "_inventory_item", inventory_item)

    payload = _collect_inventory(tmp_path)

    assert captures["count"] == 1
    assert observed_policies == [policy, policy]
    assert observed_reserved_bytes[0] == 0
    assert observed_reserved_bytes[1] == 0
    assert policy.used_memory_bytes == 0
    assert [item["logical_relative_root"] for item in payload["datasets"]] == [
        "dataset-a",
        "dataset-b",
    ]


def test_schema_inventory_cli_keeps_one_policy_through_stream_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    observed: list[RuntimeResourcePolicy] = []
    observed_reserved: list[int] = []
    cleanup_states: list[str] = []
    monkeypatch.setattr(
        runtime_resources,
        "_capture_runtime_resource_policy",
        lambda **_kwargs: policy,
    )

    def write(_root: Path, _output: Path, *, resource_policy=None) -> Path:
        assert resource_policy is policy
        with policy.lease("test.schema.inventory", 64):
            observed.append(resource_policy)
            observed_reserved.append(policy.used_memory_bytes)
            cleanup_states.append(policy.state)
            _output.write_text("{}", encoding="utf-8")
        return _output

    monkeypatch.setattr(private_cli, "write_schema_inventory", write)
    private_cli.schema_inventory(tmp_path, tmp_path / "inventory.json")

    assert observed == [policy]
    assert observed_reserved == [64]
    assert cleanup_states == ["ACTIVE"]
    assert policy.used_memory_bytes == 0


def test_schema_inventory_streams_accumulated_items_over_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 750,
    )
    monkeypatch.setattr(
        runtime_resources,
        "_capture_runtime_resource_policy",
        lambda **_kwargs: policy,
    )
    monkeypatch.setattr(
        schema_inventory,
        "_iter_dataset_artifact_roots",
        lambda _root: (tmp_path / "dataset-a", tmp_path / "dataset-b"),
    )
    base_item = _valid_inventory_payload()["datasets"][0]
    artifact_roots = tuple(tmp_path / f"dataset-{index:03d}" for index in range(100))
    monkeypatch.setattr(
        schema_inventory,
        "_iter_dataset_artifact_roots",
        lambda _root: artifact_roots,
    )

    def inventory_item(_project_root, artifact_root, **kwargs):  # noqa: ANN001, ANN202
        item = deepcopy(base_item)
        artifact_id = artifact_root.name
        item["artifact_id"] = artifact_id
        item["logical_relative_root"] = (
            f"data/processed/a_share/factor/{artifact_id}"
        )
        item["warnings"] = ["w" * 1000]
        kwargs["retained_items"].enter_context(
            kwargs["resource_policy"].lease("schema.inventory.items", 500)
        )
        return item, 500

    monkeypatch.setattr(schema_inventory, "_inventory_item", inventory_item)

    output = tmp_path / "inventory.json"
    schema_inventory.write_schema_inventory(
        tmp_path,
        output,
        resource_policy=policy,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["datasets"]) == len(artifact_roots)
    assert output.stat().st_size > policy.memory_budget_bytes
    assert policy.used_memory_bytes == 0


def test_schema_inventory_item_state_is_admitted_before_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=16 * 1024,
    )
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "market_bars" / "demo"
    )
    inspection = SimpleNamespace(
        spec={"primary_key": ["timestamp"], "partition_columns": []},
        columns=("timestamp", "value"),
        manifest={"duplicate_primary_key_count": 0},
        file_count=1,
        partitions=(),
        row_count=1,
        date_min="2026-01-01",
        date_max="2026-01-01",
    )
    def owned_inspection(*_args, resource_policy, **_kwargs):  # noqa: ANN001, ANN003
        return _OwnedValue(
            inspection,
            resource_policy.lease("test.schema.inspection", 0),
        )
    monkeypatch.setattr(
        schema_inventory,
        "_inspect_dataset_session_owned",
        owned_inspection,
    )
    monkeypatch.setattr(
        schema_inventory,
        "_aggregate_dataset_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inventory aggregation entered before state admission")
        ),
    )
    session = SimpleNamespace(verify=lambda: None)

    with runtime_resource_policy_scope(policy), ExitStack() as retained:
        with pytest.raises(
            ValueError,
            match="schema_inventory_payload_exceeds_task_memory_budget",
        ):
            schema_inventory._inventory_item_from_session(
                tmp_path,
                artifact_root,
                session=session,
                important_null_count_columns=frozenset(),
                resource_policy=policy,
                retained_items=retained,
            )


def test_schema_inventory_rejects_replacement_between_open_and_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_replacement",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    current_root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    replacement_root = (
        tmp_path
        / "replacement"
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / spec.data_id
    )
    backup_root = current_root.with_name(current_root.name + "-before-replacement")

    def frame(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "market_id": ["a_share"] * rows,
                "asset_id": [f"asset_{index}" for index in range(rows)],
                "timestamp": pd.date_range("2026-01-01", periods=rows, freq="D"),
                "value": [float(index) for index in range(rows)],
            }
        )

    write_dataset(current_root, frame(2), spec=spec)
    write_dataset(replacement_root, frame(3), spec=spec)
    real_open = schema_inventory.open_dataset_session
    real_aggregate = schema_inventory._aggregate_dataset_stats
    open_count = 0

    def record_open(root: Path, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(root, **kwargs)

    def replace_before_stats(session, *args, **kwargs):  # noqa: ANN001
        current_root.rename(backup_root)
        replacement_root.rename(current_root)
        return real_aggregate(session, *args, **kwargs)

    monkeypatch.setattr(schema_inventory, "open_dataset_session", record_open)
    monkeypatch.setattr(
        schema_inventory,
        "_aggregate_dataset_stats",
        replace_before_stats,
    )

    with pytest.raises(
        ValueError,
        match="dataset_(spec|manifest)_identity_changed",
    ):
        with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
            tuple(items)

    assert open_count == 1


@pytest.mark.parametrize(
    ("forged_summary", "error"),
    (
        (
            {"row_count": 3, "partition_row_counts": {"year=2026": 3}},
            "manifest_row_count_mismatch",
        ),
        (
            {
                "partitions": ["year=2099"],
                "partition_row_counts": {"year=2099": 2},
            },
            "manifest_partition_date_mismatch",
        ),
        (
            {
                "null_count_summary": {
                    "market_id": 1,
                    "asset_id": 0,
                    "timestamp": 0,
                }
            },
            "dataset_manifest_summary_mismatch",
        ),
    ),
    ids=("row-count", "partition-summary", "observer-summary"),
)
def test_schema_inventory_rejects_forged_storage_summary(
    tmp_path: Path,
    forged_summary: dict[str, object],
    error: str,
) -> None:
    spec = ProcessedDataSpec(
        data_id="inventory_forged_summary",
        market_id=MarketId("a_share"),
        kind=ProcessedDataKind.FACTOR,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    root = tmp_path / "data" / "processed" / "a_share" / "factor" / spec.data_id
    write_dataset(
        root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["asset_a", "asset_b"],
                "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                "value": [1.0, 2.0],
            }
        ),
        spec=spec,
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(forged_summary)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        with schema_inventory.iter_schema_inventory_items(tmp_path) as items:
            tuple(items)


@pytest.mark.parametrize("value", (True, 1, ""))
def test_schema_inventory_rejects_important_column_coercion(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_inventory_important_columns_invalid",
    ):
        with schema_inventory.iter_schema_inventory_items(
            tmp_path,
            important_null_count_columns=(value,),  # type: ignore[arg-type]
        ):
            pass


def _valid_inventory_payload() -> dict[str, object]:
    columns = ("market_id", "asset_id", "timestamp", "value")
    dtypes = {
        "market_id": "object",
        "asset_id": "object",
        "timestamp": "datetime64[ns]",
        "value": "float64",
    }
    primary_key = ("market_id", "asset_id", "timestamp")
    partition_columns = ("year",)
    return {
        "schema_id": "qrp-schema-inventory",
        "datasets": [
            {
                "artifact_id": "inventory_factor",
                "layer": "processed",
                "kind": "factor",
                "logical_relative_root": (
                    "data/processed/a_share/factor/inventory_factor"
                ),
                "columns": list(columns),
                "dtypes": dtypes,
                "primary_key": list(primary_key),
                "partition_columns": list(partition_columns),
                "file_count": 1,
                "row_count": 3,
                "row_count_by_year": {"2026": 3},
                "min_timestamp": "2026-01-01",
                "max_timestamp": "2026-01-03",
                "duplicate_primary_key_count": 0,
                "null_key_counts": {column: 0 for column in primary_key},
                "selected_important_column_null_counts": {"value": 0},
                "schema_hash": schema_inventory._schema_hash(
                    columns=columns,
                    dtypes=dtypes,
                    primary_key=primary_key,
                    partition_columns=partition_columns,
                ),
                "warnings": [],
            }
        ],
    }


@pytest.mark.parametrize("failure_codes", ({"failure": "code"}, ("failure",)))
def test_schema_inventory_summary_rejects_non_list_validation_failure_codes(
    failure_codes: object,
) -> None:
    value = {
        "schema_id": "qrp-schema-inventory-summary",
        "status": "FAIL",
        "failure_codes": failure_codes,
        "dataset_count": 1,
        "total_row_count": 2,
    }

    with pytest.raises(ValueError, match="schema_inventory_summary_invalid"):
        validate_schema_inventory_summary(
            value,
            validation_status="FAIL",
            validation_failure_codes=failure_codes,  # type: ignore[arg-type]
            dataset_count=1,
            total_row_count=2,
        )


def test_schema_inventory_summary_uses_streaming_production_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=16 * 1024 * 1024)
    artifact_root = (
        tmp_path / "data" / "processed" / "a_share" / "factor" / "inventory_factor"
    )
    item = deepcopy(_valid_inventory_payload())["datasets"][0]

    monkeypatch.setattr(
        schema_inventory,
        "_iter_dataset_artifact_roots",
        lambda _root: (artifact_root,),
    )
    monkeypatch.setattr(
        schema_inventory,
        "_inventory_item",
        lambda *_args, **_kwargs: (item, 0),
    )

    summary = schema_inventory.summarize_schema_inventory(
        tmp_path,
        resource_policy=policy,
    )

    assert summary["schema_id"] == "qrp-schema-inventory-summary"
    assert summary["dataset_count"] == 1
    assert summary["total_row_count"] == 3
    assert len(summary["schema_hash"]) == 64
    assert policy.used_memory_bytes == 0
