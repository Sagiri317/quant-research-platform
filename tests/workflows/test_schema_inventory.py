from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

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
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.workflows.diagnostics import schema_inventory


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

    payload = schema_inventory.build_schema_inventory_payload(tmp_path)

    assert payload["datasets"][0]["duplicate_primary_key_count"] == 0


def test_schema_inventory_uses_policy_bound_physical_batches() -> None:
    source = inspect.getsource(schema_inventory._aggregate_dataset_stats)

    assert "DatasetReadMode.CHUNKED" not in source
    assert "target_chunk_bytes=resource_policy.payload_target_bytes" in source
    assert "ExactPrimaryKeyIndex" not in source
    assert "sqlite_cache_bytes" not in source
    assert "duplicate_primary_key_count = exact_duplicate_count" in source


def test_schema_inventory_captures_one_policy_for_all_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=16 * 1024 * 1024,
        environ={},
    )
    captures = {"count": 0}
    observed_policies: list[RuntimeResourcePolicy] = []

    def capture_policy() -> RuntimeResourcePolicy:
        captures["count"] += 1
        return policy

    def inventory_item(
        _project_root: Path,
        artifact_root: Path,
        *,
        important_null_count_columns: frozenset[str],
        resource_policy: RuntimeResourcePolicy,
    ) -> dict[str, object]:
        assert important_null_count_columns == frozenset()
        assert active_runtime_resource_policy() is resource_policy
        observed_policies.append(resource_policy)
        return {"logical_relative_root": artifact_root.name}

    monkeypatch.setattr(
        schema_inventory,
        "active_runtime_resource_policy",
        capture_policy,
    )
    monkeypatch.setattr(
        schema_inventory,
        "_iter_dataset_artifact_roots",
        lambda _root: (tmp_path / "dataset-a", tmp_path / "dataset-b"),
    )
    monkeypatch.setattr(schema_inventory, "_inventory_item", inventory_item)

    payload = schema_inventory.build_schema_inventory_payload(tmp_path)

    assert captures["count"] == 1
    assert observed_policies == [policy, policy]
    assert [item["logical_relative_root"] for item in payload["datasets"]] == [
        "dataset-a",
        "dataset-b",
    ]


def test_schema_inventory_reuses_session_duplicate_count() -> None:
    source = inspect.getsource(schema_inventory._inventory_item) + inspect.getsource(
        schema_inventory._inventory_item_from_session
    )

    assert "inspect_dataset_session(" in source
    assert 'inspection.manifest["duplicate_primary_key_count"]' in source
    assert "ExactPrimaryKeyIndex" not in source


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
        match="dataset_manifest_identity_changed",
    ):
        schema_inventory.build_schema_inventory_payload(tmp_path)

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
        schema_inventory.build_schema_inventory_payload(tmp_path)


@pytest.mark.parametrize("value", (True, 1, ""))
def test_schema_inventory_rejects_important_column_coercion(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="schema_inventory_important_columns_invalid",
    ):
        schema_inventory.build_schema_inventory_payload(
            tmp_path,
            important_null_count_columns=(value,),  # type: ignore[arg-type]
        )


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


def test_schema_inventory_summary_validates_complete_exact_payload() -> None:
    summary = schema_inventory.summarize_schema_inventory_payload(
        _valid_inventory_payload()
    )

    assert summary["schema_id"] == "qrp-schema-inventory-summary"
    assert summary["dataset_count"] == 1
    assert summary["total_row_count"] == 3
    assert len(summary["schema_hash"]) == 64


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


@pytest.mark.parametrize(
    ("layer", "kind", "logical_roots"),
    (
        (
            "interim",
            "interim",
            (
                "data/interim/source_a/inventory_factor",
                "data/interim/source_b/inventory_factor",
            ),
        ),
        (
            "processed",
            "factor",
            (
                "data/processed/a_share/factor/inventory_factor",
                "data/processed/crypto/factor/inventory_factor",
            ),
        ),
        (
            "research",
            "research_dataset",
            (
                "data/research/a_share/inventory_factor",
                "data/research/crypto/inventory_factor",
            ),
        ),
    ),
)
def test_schema_inventory_namespace_is_part_of_dataset_identity_and_summary_hash(
    layer: str,
    kind: str,
    logical_roots: tuple[str, str],
) -> None:
    payload = deepcopy(_valid_inventory_payload())
    dataset = payload["datasets"][0]
    dataset["layer"] = layer
    dataset["kind"] = kind
    dataset["logical_relative_root"] = logical_roots[0]
    alternate = deepcopy(payload)
    alternate["datasets"][0]["logical_relative_root"] = logical_roots[1]

    combined = deepcopy(payload)
    combined["datasets"].extend(alternate["datasets"])
    combined["datasets"].sort(key=lambda item: item["logical_relative_root"])

    validated = schema_inventory.validate_schema_inventory_payload(combined)
    assert len(validated["datasets"]) == 2
    assert (
        schema_inventory.summarize_schema_inventory_payload(payload)["schema_hash"]
        != schema_inventory.summarize_schema_inventory_payload(alternate)["schema_hash"]
    )


@pytest.mark.parametrize(
    ("layer", "kind", "logical_root"),
    (
        ("processed", "factor", "data"),
        ("interim", "interim", "data/interim/source/extra/inventory_factor"),
        ("processed", "factor", "data/processed/factor/inventory_factor"),
        (
            "processed",
            "factor/nested",
            "data/processed/a_share/factor/nested/inventory_factor",
        ),
        (
            "research",
            "research_dataset",
            "data/research/a_share/extra/inventory_factor",
        ),
    ),
)
def test_schema_inventory_rejects_non_storage_layout_cardinality(
    layer: str,
    kind: str,
    logical_root: str,
) -> None:
    payload = deepcopy(_valid_inventory_payload())
    dataset = payload["datasets"][0]
    dataset["layer"] = layer
    dataset["kind"] = kind
    dataset["logical_relative_root"] = logical_root

    with pytest.raises(ValueError, match="schema_inventory_dataset_invalid"):
        schema_inventory.validate_schema_inventory_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_field",
        "raw_layer",
        "unknown_layer",
        "empty_datasets",
        "bool_row_count",
        "duplicate_columns",
        "missing_dtype",
        "wrong_schema_hash",
        "half_date_range",
        "bool_year_count",
        "year_total_exceeds_rows",
        "non_null_timestamp_year_total_undercounts_rows",
        "path_identity_mismatch",
        "noncanonical_date",
        "reversed_date_range",
        "primary_key_not_a_column",
        "null_count_exceeds_rows",
        "unsorted_datasets",
    ),
)
def test_schema_inventory_exact_grammar_rejects_mutations(mutation: str) -> None:
    payload = deepcopy(_valid_inventory_payload())
    dataset = payload["datasets"][0]
    if mutation == "unknown_top_field":
        payload["unknown"] = True
    elif mutation == "raw_layer":
        dataset["layer"] = "raw"
        dataset["logical_relative_root"] = "data/raw/source/inventory_factor"
    elif mutation == "unknown_layer":
        dataset["layer"] = "archive"
        dataset["logical_relative_root"] = "data/archive/source/inventory_factor"
    elif mutation == "empty_datasets":
        payload["datasets"] = []
    elif mutation == "bool_row_count":
        dataset["row_count"] = True
    elif mutation == "duplicate_columns":
        dataset["columns"].append("value")
    elif mutation == "missing_dtype":
        dataset["dtypes"].pop("value")
    elif mutation == "wrong_schema_hash":
        dataset["schema_hash"] = "0" * 64
    elif mutation == "half_date_range":
        dataset["max_timestamp"] = None
    elif mutation == "bool_year_count":
        dataset["row_count_by_year"]["2026"] = True
    elif mutation == "year_total_exceeds_rows":
        dataset["row_count_by_year"]["2026"] = 4
    elif mutation == "non_null_timestamp_year_total_undercounts_rows":
        dataset["row_count_by_year"]["2026"] = 2
    elif mutation == "path_identity_mismatch":
        dataset["artifact_id"] = "other_factor"
    elif mutation == "noncanonical_date":
        dataset["min_timestamp"] = "20260101"
    elif mutation == "reversed_date_range":
        dataset["min_timestamp"] = "2026-01-04"
    elif mutation == "primary_key_not_a_column":
        dataset["primary_key"].append("missing_key")
        dataset["null_key_counts"]["missing_key"] = 0
        dataset["schema_hash"] = schema_inventory._schema_hash(
            columns=tuple(dataset["columns"]),
            dtypes=dataset["dtypes"],
            primary_key=tuple(dataset["primary_key"]),
            partition_columns=tuple(dataset["partition_columns"]),
        )
    elif mutation == "null_count_exceeds_rows":
        dataset["null_key_counts"]["asset_id"] = 4
    elif mutation == "unsorted_datasets":
        second = deepcopy(dataset)
        second["artifact_id"] = "another_factor"
        second["logical_relative_root"] = "data/processed/a_share/factor/another_factor"
        payload["datasets"] = [dataset, second]

    with pytest.raises(ValueError, match="schema_inventory_"):
        schema_inventory.summarize_schema_inventory_payload(payload)


@pytest.mark.parametrize(
    "timestamp_contract",
    ("not_primary_key", "nullable_primary_key"),
)
def test_schema_inventory_year_counts_remain_bounded_diagnostic_without_exact_pk(
    timestamp_contract: str,
) -> None:
    payload = deepcopy(_valid_inventory_payload())
    dataset = payload["datasets"][0]
    dataset["row_count_by_year"]["2026"] = 2
    if timestamp_contract == "not_primary_key":
        dataset["primary_key"].remove("timestamp")
        dataset["null_key_counts"].pop("timestamp")
        dataset["schema_hash"] = schema_inventory._schema_hash(
            columns=tuple(dataset["columns"]),
            dtypes=dataset["dtypes"],
            primary_key=tuple(dataset["primary_key"]),
            partition_columns=tuple(dataset["partition_columns"]),
        )
    else:
        dataset["null_key_counts"]["timestamp"] = 1

    assert schema_inventory.validate_schema_inventory_payload(payload) is not None
