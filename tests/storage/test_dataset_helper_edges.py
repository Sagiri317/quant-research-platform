from __future__ import annotations

import json
import os
import subprocess
import tracemalloc
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.comparison as dataset_comparison
import quant_research_platform.storage.datasets.fingerprinting as dataset_fingerprinting
import quant_research_platform.storage.datasets.inventory as inventory_owner
import quant_research_platform.storage.exact_keys as key_tracking
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    InterimDataSpec,
    data_frequency_slug,
)
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.storage import datasets
from quant_research_platform.storage.datasets.inventory import DatasetSessionInventory
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage.datasets.reading import (
    read_columns_for_time_window,
)
from quant_research_platform.storage.datasets.schema import (
    data_spec_payload,
    logical_dtype_payload_from_frame,
    physical_schema_from_schema,
    validate_dataset_manifest_payload,
    validate_dataset_spec,
    validate_dataset_spec_payload,
    validate_physical_schema_payload,
)
from quant_research_platform.storage.datasets.writing import _WriteStats
from quant_research_platform.storage.exact_keys import (
    ExactPrimaryKeyIndex,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.runtime_resources import capture_test_policy
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_helper_edges_dataset_frame,
    make_processed_dataset_spec,
    write_low_level_parquet_artifact,
)


def _discovered_paths(data_root: Path, *, spec: dict[str, object]) -> list[Path]:
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with (
        runtime_resource_policy_scope(policy),
        DatasetSessionInventory(
            data_root=data_root,
            resource_policy=policy,
        ) as inventory,
    ):
        inventory.discover(spec)
        return list(inventory.iter_candidate_paths())


def test_dataset_inventory_lease_admits_shrinks_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.datasets.inventory.gettempdir",
        lambda: str(tmp_path),
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    low_policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=1024,
    )

    with runtime_resource_policy_scope(low_policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            DatasetSessionInventory(
                data_root=data_root,
                resource_policy=low_policy,
            )
    assert not list(tmp_path.glob(".i-*"))

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024,
        explicit_cap_bytes=64 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        inventory = DatasetSessionInventory(
            data_root=data_root,
            resource_policy=policy,
        )
        assert policy.used_memory_bytes == 1024
        inventory.close()
        assert policy.used_memory_bytes == 0
    assert not list(tmp_path.glob(".i-*"))


def test_dataset_inventory_rejection_releases_same_root_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    policy = capture_test_policy(available_memory_bytes=64 * 1024)

    def reject_scratch_path(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid_storage_scratch_path")

    monkeypatch.setattr(
        inventory_owner,
        "require_unaliased_path",
        reject_scratch_path,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="invalid_storage_scratch_path"):
            DatasetSessionInventory(
                data_root=data_root,
                resource_policy=policy,
            )
        assert policy.used_memory_bytes == 0


def test_dataset_inventory_failure_and_primary_cleanup_release_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.datasets.inventory.gettempdir",
        lambda: str(tmp_path),
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    policy = capture_test_policy(available_memory_bytes=64 * 1024)

    def fail_connect(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("sqlite_start_failed")

    with runtime_resource_policy_scope(policy):
        with monkeypatch.context() as sqlite_failure:
            sqlite_failure.setattr(
                "quant_research_platform.storage.datasets.inventory.sqlite3.connect",
                fail_connect,
            )
            with pytest.raises(RuntimeError, match="sqlite_start_failed"):
                DatasetSessionInventory(
                    data_root=data_root,
                    resource_policy=policy,
                )
        assert policy.used_memory_bytes == 0

        with pytest.raises(KeyboardInterrupt):
            with DatasetSessionInventory(
                data_root=data_root,
                resource_policy=policy,
            ):
                raise KeyboardInterrupt
        assert policy.used_memory_bytes == 0
    assert not list(tmp_path.glob(".i-*"))


def test_exact_key_index_classifies_typed_exact_duplicates_and_conflicts(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "asset_id": pa.array(["a", "a", "b"], type=pa.string()),
            "value": pa.array([float("inf"), float("inf"), None], type=pa.float64()),
        }
    )
    conflict = pa.table(
        {
            "asset_id": pa.array(["a"], type=pa.string()),
            "value": pa.array([2.0], type=pa.float64()),
        }
    )
    exact_again = pa.table(
        {
            "value": pa.array([float("inf")], type=pa.float64()),
            "asset_id": pa.array(["a"], type=pa.string()),
        }
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        keep_positions, duplicate_count = index.classify_comparison_table(
            table,
            namespace="reference",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert list(keep_positions) == [0, 2]
        assert keep_positions.view.nbytes < keep_positions.retained_memory_bytes
        assert duplicate_count == 1
        keep_positions, duplicate_count = index.classify_comparison_table(
            exact_again,
            namespace="reference",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert list(keep_positions) == []
        assert duplicate_count == 1
        keep_positions, duplicate_count = index.classify_comparison_table(
            exact_again,
            namespace="independent",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert list(keep_positions) == [0]
        assert duplicate_count == 0
        with pytest.raises(ValueError, match="conflicting_exact_primary_key_payload"):
            index.classify_comparison_table(
                conflict,
                namespace="reference",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
        keep_positions, duplicate_count = index.classify_comparison_table(
            pa.table(
                {
                    "asset_id": pa.array(["c"], type=pa.string()),
                    "value": pa.array([None], type=pa.float64()),
                }
            ),
            namespace="reference",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert list(keep_positions) == [0]
        assert duplicate_count == 0


def test_exact_key_classifier_batches_sql_and_returns_compact_positions(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "asset_id": pa.array(
                [f"asset-{position:04d}" for position in range(256)],
                type=pa.string(),
            )
        }
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        statements: list[str] = []
        assert index._connection is not None
        index._connection.set_trace_callback(statements.append)
        keep_positions, duplicate_count = index.classify_comparison_table(
            table,
            namespace="batch",
            value_columns=(),
            payload_target_bytes=4096,
        )

    lookup_statements = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
    ]
    assert len(lookup_statements) <= 4
    assert isinstance(keep_positions.view, memoryview)
    assert keep_positions.view.itemsize <= 8
    assert keep_positions.view.nbytes == 256 * keep_positions.view.itemsize
    assert keep_positions.retained_memory_bytes == 256 * keep_positions.view.itemsize
    assert keep_positions.view.tolist() == list(range(256))
    with pytest.raises(TypeError):
        keep_positions.view[0] = 1
    assert duplicate_count == 0


def test_exact_key_classifier_uses_one_sorter_free_staging_transaction(
    tmp_path: Path,
) -> None:
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        index.classify_comparison_table(
            pa.table({"asset_id": ["existing"], "value": [1]}),
            namespace="ordered",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert index._connection is not None
        connection = index._connection
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        try:
            keep, duplicate_count = index.classify_comparison_table(
                pa.table(
                    {
                        "asset_id": ["new-b", "existing", "new-a", "new-b"],
                        "value": [2, 1, 3, 2],
                    }
                ),
                namespace="ordered",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
        finally:
            connection.set_trace_callback(None)

        assert list(keep) == [0, 2]
        assert duplicate_count == 2
        assert connection.execute("PRAGMA temp_store").fetchone() == (1,)
        staged_statements = [
            statement
            for statement in statements
            if "comparison_rows" in statement.casefold()
        ]
        plan = [
            detail
            for statement in staged_statements
            for *_prefix, detail in connection.execute(
                "EXPLAIN QUERY PLAN " + statement
            )
        ]
        assert not any("TEMP B-TREE" in detail for detail in plan)
        assert [
            statement.lstrip().split(maxsplit=1)[0].upper()
            for statement in statements
            if statement.lstrip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK"))
        ] == ["BEGIN", "COMMIT"]
        remaining = connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone()
        assert remaining == (0,)


def test_exact_key_classifier_preflights_compact_positions_before_mutation(
    tmp_path: Path,
) -> None:
    table = pa.table({"asset_id": [f"asset-{position:02d}" for position in range(64)]})

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(
            ValueError,
            match="exact_key_position_state_exceeds_payload_target",
        ):
            index.classify_comparison_table(
                table,
                namespace="bounded",
                value_columns=(),
                payload_target_bytes=256,
            )
        assert index.add_table(table, namespace="bounded") == 0


@pytest.mark.parametrize(
    ("namespace", "value_columns", "error"),
    (
        ("", (), "invalid_exact_key_namespace"),
        ("grammar", ("value", "value"), "invalid_dataset_comparison_columns"),
        ("grammar", ("asset_id",), "invalid_dataset_comparison_columns"),
        ("grammar", ("missing",), "missing_dataset_comparison_column"),
    ),
)
def test_exact_key_classifier_validates_grammar_before_position_admission(
    tmp_path: Path,
    namespace: str,
    value_columns: tuple[str, ...],
    error: str,
) -> None:
    table = pa.table(
        {
            "asset_id": [f"asset-{position:02d}" for position in range(64)],
            "value": range(64),
        }
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(ValueError, match=error):
            index.classify_comparison_table(
                table,
                namespace=namespace,
                value_columns=value_columns,
                payload_target_bytes=256,
            )


def test_exact_key_classifier_gates_disk_overlap_before_exact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def gate(_path: Path, *, reserve_bytes: int) -> int:
        nonlocal calls
        assert reserve_bytes > 0
        calls += 1
        if calls == 3:
            raise OSError("disk_headroom_exhausted")
        return 1024**3

    monkeypatch.setattr(key_tracking, "require_free_disk_headroom", gate)
    table = pa.table({"asset_id": ["a", "b"]})
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(OSError, match="disk_headroom_exhausted"):
            index.classify_comparison_table(
                table,
                namespace="overlap",
                value_columns=(),
                payload_target_bytes=1024,
            )
        assert index._connection is not None
        assert index._connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone() == (0,)
        assert index.add_table(table, namespace="overlap") == 0


def test_exact_key_classifier_renews_disk_gate_for_secondary_index_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {"asset_id": ["a" * 300_000, "b" * 300_000]},
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        reserves: list[int] = []

        def gate(_path: Path, *, reserve_bytes: int) -> None:
            reserves.append(reserve_bytes)
            if len(reserves) == 2 and reserve_bytes == reserves[0]:
                raise OSError("disk_headroom_exhausted")

        monkeypatch.setattr(key_tracking, "require_free_disk_headroom", gate)
        with pytest.raises(OSError, match="disk_headroom_exhausted"):
            index.classify_comparison_table(
                table,
                namespace="indexed-overlap",
                value_columns=(),
                payload_target_bytes=1024**2,
            )
        assert index._connection is not None
        assert index._connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone() == (0,)
        assert index._connection.execute(
            "SELECT COUNT(*) FROM exact_keys WHERE namespace = ?",
            ("indexed-overlap",),
        ).fetchone() == (0,)


def test_exact_key_classifier_gates_exact_insert_physical_growth_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {"asset_id": ["a" * 300_000, "b" * 300_000]},
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        reserves: list[int] = []

        def gate(_path: Path, *, reserve_bytes: int) -> None:
            reserves.append(reserve_bytes)
            if len(reserves) == 3 and reserve_bytes >= reserves[0]:
                raise OSError("disk_headroom_exhausted")

        monkeypatch.setattr(key_tracking, "require_free_disk_headroom", gate)
        with pytest.raises(OSError, match="disk_headroom_exhausted"):
            index.classify_comparison_table(
                table,
                namespace="exact-overlap",
                value_columns=(),
                payload_target_bytes=1024**2,
            )
        assert len(reserves) == 3
        assert index._connection is not None
        assert index._connection.execute(
            "SELECT COUNT(*) FROM comparison_rows"
        ).fetchone() == (0,)
        assert index._connection.execute(
            "SELECT COUNT(*) FROM exact_keys WHERE namespace = ?",
            ("exact-overlap",),
        ).fetchone() == (0,)


def test_exact_key_classifier_batches_large_duplicate_group(tmp_path: Path) -> None:
    table = pa.table({"asset_id": ["same"] * 4096})

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        statements: list[str] = []
        assert index._connection is not None
        index._connection.set_trace_callback(statements.append)
        keep, duplicate_count = index.classify_comparison_table(
            table,
            namespace="duplicates",
            value_columns=(),
            payload_target_bytes=64 * 1024,
        )

    assert list(keep) == [0]
    assert duplicate_count == 4095
    assert (
        sum(statement.lstrip().upper().startswith("SELECT") for statement in statements)
        <= 4
    )


@pytest.mark.parametrize("payload_target_bytes", (True, False, 0, -1, 1.5, "1"))
def test_exact_key_classifier_requires_positive_non_bool_payload_target(
    tmp_path: Path,
    payload_target_bytes: object,
) -> None:
    table = pa.table({"asset_id": ["a"], "value": [1]})

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(
            ValueError,
            match="^invalid_dataset_comparison_payload_target$",
        ):
            index.classify_comparison_table(
                table,
                namespace="grammar",
                value_columns=("value",),
                payload_target_bytes=payload_target_bytes,  # type: ignore[arg-type]
            )


def test_data_spec_payload_rejects_wrong_kind_source_and_market_ids() -> None:
    processed = make_processed_dataset_spec("storage_demo", market_id="dummy")
    interim = InterimDataSpec(
        data_id=processed.data_id,
        frequency=processed.frequency,
        grain=processed.grain,
        column_roles=processed.column_roles,
        primary_key=processed.primary_key,
        partition_columns=processed.partition_columns,
        source_id=SourceId("csmar"),
    )
    research = make_demo_research_dataset_spec()

    with pytest.raises(ValueError, match="invalid_source_id"):
        replace(interim, source_id=7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid_market_id"):
        replace(processed, market_id=7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid_market_id"):
        replace(research, market_id=7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("data_id", 7, "data_id_required"),
        ("frequency", "invalid", "invalid_data_frequency"),
        ("grain", "invalid", "invalid_data_grain"),
        (
            "primary_key",
            ["market_id", "asset_id", "timestamp"],
            "primary_key_must_be_tuple",
        ),
        ("partition_columns", ["year"], "partition_columns_must_be_tuple"),
        ("kind", "invalid", "invalid_processed_data_kind"),
        (
            "source_input_data_ids",
            ["raw"],
            "source_input_data_ids_must_be_non_empty_strings",
        ),
        (
            "processed_input_data_ids",
            ["processed"],
            "processed_input_data_ids_must_be_non_empty_strings",
        ),
    ),
)
def test_processed_spec_rejects_invalid_fields_at_construction(
    field: str,
    value: object,
    error: str,
) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with pytest.raises(ValueError, match=error):
        replace(spec, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("market_id", 7, "dataset_market_id_invalid"),
        (
            "input_data_ids",
            ["panel", "factor", "label"],
            "dataset_input_data_ids_invalid",
        ),
        ("base_input_data_id", 7, "dataset_base_input_data_id_invalid"),
        ("factor_columns", ["factor"], "dataset_factor_columns_invalid"),
        ("feature_columns", [], "dataset_feature_columns_invalid"),
        ("feature_transforms", [], "dataset_feature_transforms_invalid"),
        ("label_columns", ["label"], "dataset_label_columns_invalid"),
        ("filter_columns", [], "dataset_filter_columns_invalid"),
        ("context_columns", [], "dataset_context_columns_invalid"),
        ("key_coverage", [], "dataset_key_coverage_invalid"),
    ),
)
def test_validate_dataset_spec_rejects_wrong_kind_research_fields(
    field: str,
    value: object,
    error: str,
) -> None:
    spec = make_demo_research_dataset_spec()
    object.__setattr__(spec, field, value)

    with pytest.raises(ValueError, match=error):
        validate_dataset_spec(spec)


def test_validate_dataset_spec_accepts_only_research_immutable_semantic_mappings() -> (
    None
):
    research = make_demo_research_dataset_spec()

    assert type(research.column_roles) is MappingProxyType
    assert type(research.key_coverage) is MappingProxyType
    assert validate_dataset_spec(research) == "timestamp"

    processed = make_processed_dataset_spec("storage_demo", market_id="dummy")
    object.__setattr__(
        processed,
        "column_roles",
        MappingProxyType(dict(processed.column_roles)),
    )
    assert validate_dataset_spec(processed) == "timestamp"


@pytest.mark.parametrize(
    ("field", "error"),
    (
        ("column_roles", "dataset_column_roles_invalid"),
        ("key_coverage", "dataset_key_coverage_invalid"),
    ),
)
def test_validate_dataset_spec_rejects_forged_mutable_research_semantic_mapping(
    field: str,
    error: str,
) -> None:
    research = make_demo_research_dataset_spec()
    object.__setattr__(research, field, dict(getattr(research, field)))

    with pytest.raises(ValueError, match=error):
        validate_dataset_spec(research)


@pytest.mark.parametrize(
    "replacement",
    ([0.0, 1.0], [False, True]),
    ids=("int-float", "int-bool"),
)
def test_manifest_spec_binding_is_typed_exact_recursively(
    replacement: list[object],
) -> None:
    base = make_demo_research_dataset_spec()
    factor_column = base.factor_columns[0]
    feature_column = f"{factor_column}_winsorized"
    spec = data_spec_payload(
        make_demo_research_dataset_spec(
            feature_columns=(feature_column,),
            feature_transforms=(
                FeatureTransformSpec(
                    transform_id=feature_column,
                    source_columns=(factor_column,),
                    transform="winsorize",
                    output_columns=(feature_column,),
                    winsorize_quantiles=(0, 1),
                ),
            ),
        )
    )
    physical_schema = physical_schema_from_schema(
        pa.schema(
            [
                pa.field(column, pa.string(), nullable=True)
                for column in spec["required_columns"]
            ]
        )
    )
    manifest = {
        **{key: deepcopy(value) for key, value in spec.items() if key != "schema_id"},
        "schema_id": "qrp-dataset-manifest",
        "row_count": 0,
        "file_count": 1,
        "file_size_bytes": 0,
        "columns": list(spec["required_columns"]),
        "physical_schema": physical_schema,
        "schema_hash": canonical_json_hash(physical_schema),
        "logical_dtypes": logical_dtype_payload_from_frame(
            pd.DataFrame(
                {
                    column: pd.Series(dtype="object")
                    for column in spec["required_columns"]
                }
            )
        ),
        "partition_row_counts": {"year=2024": 0},
        "partition_count": 1,
        "partitions": ["year=2024"],
        "date_min": None,
        "date_max": None,
        "null_count_summary": {},
        "non_finite_count_summary": {},
        "duplicate_primary_key_count": 0,
        "factor_nan_profile": {},
        "label_nan_profile": {},
        "validation_status": "PASS",
    }
    manifest["logical_dtype_hash"] = canonical_json_hash(manifest["logical_dtypes"])
    manifest["feature_transforms"][0]["winsorize_quantiles"] = replacement

    if all(type(item) is bool for item in replacement):
        with pytest.raises(ValueError, match="dataset_manifest_spec_mismatch"):
            validate_dataset_manifest_payload(manifest, spec=spec)
    else:
        validate_dataset_manifest_payload(manifest, spec=spec)


def test_storage_maps_huge_winsorize_quantiles_to_stable_owner_error() -> None:
    factor_column = make_demo_research_dataset_spec().factor_columns[0]
    feature_column = f"{factor_column}_winsorized"
    transform = FeatureTransformSpec(
        transform_id=feature_column,
        source_columns=(factor_column,),
        transform="winsorize",
        output_columns=(feature_column,),
        winsorize_quantiles=(0, 1),
    )
    runtime_spec = make_demo_research_dataset_spec(
        feature_columns=(feature_column,),
        feature_transforms=(transform,),
    )
    object.__setattr__(transform, "winsorize_quantiles", (0, 10**400))

    with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
        validate_dataset_spec(runtime_spec)

    persisted_spec = data_spec_payload(
        make_demo_research_dataset_spec(
            feature_columns=(feature_column,),
            feature_transforms=(
                FeatureTransformSpec(
                    transform_id=feature_column,
                    source_columns=(factor_column,),
                    transform="winsorize",
                    output_columns=(feature_column,),
                    winsorize_quantiles=(0, 1),
                ),
            ),
        )
    )
    persisted_spec["feature_transforms"][0]["winsorize_quantiles"] = [
        0,
        10**400,
    ]

    with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
        validate_dataset_spec_payload(persisted_spec)


@pytest.mark.parametrize("wrong_kind", ([], {}), ids=("list", "dict"))
def test_persisted_research_key_coverage_rejects_unhashable_policy_with_owner_error(
    wrong_kind: object,
) -> None:
    payload = data_spec_payload(make_demo_research_dataset_spec())
    data_id = payload["input_data_ids"][0]
    payload["key_coverage"][data_id] = wrong_kind

    with pytest.raises(ValueError, match="dataset_key_coverage_invalid"):
        validate_dataset_spec_payload(payload)


@pytest.mark.parametrize(
    ("field", "wrong_kind"),
    (
        ("transform", []),
        ("transform", {}),
        ("fillna_policy", []),
        ("fillna_policy", {}),
    ),
    ids=("transform-list", "transform-dict", "fillna-list", "fillna-dict"),
)
def test_persisted_feature_transform_rejects_unhashable_enum_with_owner_error(
    field: str,
    wrong_kind: object,
) -> None:
    factor_column = make_demo_research_dataset_spec().factor_columns[0]
    feature_column = f"{factor_column}_rank"
    payload = data_spec_payload(
        make_demo_research_dataset_spec(
            feature_columns=(feature_column,),
            feature_transforms=(
                FeatureTransformSpec(
                    transform_id=feature_column,
                    source_columns=(factor_column,),
                    transform="cross_sectional_rank",
                    output_columns=(feature_column,),
                ),
            ),
        )
    )
    payload["feature_transforms"][0][field] = wrong_kind

    with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
        validate_dataset_spec_payload(payload)


@pytest.mark.parametrize(
    "type_payload",
    (
        {"kind": "decimal", "bit_width": 128, "precision": 39, "scale": 0},
        {"kind": "decimal", "bit_width": 256, "precision": 77, "scale": 0},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": -1},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 19},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": -1},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": 41},
        {
            "kind": "fixed_size_binary",
            "byte_width": 0,
        },
        {"kind": "struct", "fields": []},
        {
            "kind": "list",
            "value_field": {
                "name": "item",
                "nullable": True,
                "type": {"kind": "struct", "fields": []},
            },
        },
        {"kind": "date", "unit": "millisecond"},
        {"kind": "time", "unit": "s"},
        {"kind": "timestamp", "unit": "s", "timezone": None},
        {
            "kind": "dictionary",
            "index_type": {"kind": "integer", "signed": False, "bit_width": 8},
            "value_type": {"kind": "string"},
            "ordered": False,
        },
        {
            "kind": "map",
            "keys_sorted": False,
            "key_field": {
                "name": "key",
                "nullable": True,
                "type": {"kind": "string"},
            },
            "item_field": {
                "name": "value",
                "nullable": True,
                "type": {"kind": "integer", "signed": True, "bit_width": 64},
            },
        },
    ),
    ids=(
        "decimal128-precision",
        "decimal256-precision",
        "decimal128-negative-scale",
        "decimal128-scale-over-precision",
        "decimal256-negative-scale",
        "decimal256-scale-over-precision",
        "fixed-size-binary-zero",
        "empty-struct",
        "nested-empty-struct",
        "date-millisecond",
        "time-second",
        "timestamp-second",
        "dictionary-unsigned-index",
        "map-key-nullable",
    ),
)
def test_physical_schema_rejects_unpersistable_descriptors(
    type_payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
        validate_physical_schema_payload(
            [{"name": "value", "nullable": True, "type": type_payload}]
        )


@pytest.mark.parametrize(
    "data_type",
    (
        pa.decimal128(18, 0),
        pa.decimal128(38, 38),
        pa.decimal256(40, 0),
        pa.decimal256(76, 76),
        pa.float16(),
        pa.date32(),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
        pa.timestamp("ms"),
        pa.timestamp("us"),
        pa.timestamp("ns", tz=""),
        pa.duration("ns"),
        pa.large_string(),
        pa.large_binary(),
        pa.binary(16),
        pa.list_(pa.field("element", pa.int32(), nullable=False), 0),
        pa.struct([pa.field("child", pa.int64(), nullable=False)]),
        pa.map_(pa.string(), pa.int64()),
        pa.dictionary(pa.int8(), pa.string()),
        pa.list_(
            pa.field(
                "element",
                pa.dictionary(pa.int16(), pa.string()),
                nullable=True,
            )
        ),
    ),
    ids=(
        "decimal128-scale-zero",
        "decimal128-scale-precision",
        "decimal256-scale-zero",
        "decimal256-scale-precision",
        "float16",
        "date-day",
        "time-ms",
        "time-us",
        "time-ns",
        "timestamp-ms",
        "timestamp-us",
        "timestamp-ns-empty-timezone",
        "duration-ns",
        "large-string",
        "large-binary",
        "fixed-size-binary-positive",
        "fixed-size-list-zero",
        "nonempty-struct",
        "map",
        "signed-dictionary",
        "nested-dictionary",
    ),
)
def test_physical_schema_accepts_persisted_parquet_identity_round_trip(
    tmp_path: Path,
    data_type: pa.DataType,
) -> None:
    schema = pa.schema([pa.field("value", data_type, nullable=True)])
    expected = physical_schema_from_schema(schema)
    table = pa.Table.from_arrays(
        [pa.array([], type=data_type)],
        schema=schema,
    )
    path = tmp_path / "part-00000.parquet"
    pq.write_table(table, path)
    parquet_file = pq.ParquetFile(path)
    try:
        persisted = physical_schema_from_schema(parquet_file.schema_arrow)
    finally:
        parquet_file.close()

    assert persisted == expected


@pytest.mark.parametrize("invalid_count", (Decimal("1"), Fraction(1, 1)))
def test_manifest_counts_reject_non_int_numeric_types(
    tmp_path: Path,
    invalid_count: object,
) -> None:
    root = make_dataset_artifact_root(
        tmp_path,
        data_id="storage_demo",
        market_id="dummy",
    )
    frame = make_helper_edges_dataset_frame(market_id="dummy")
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    datasets.write_dataset(root, frame, spec=spec)
    spec_payload = json.loads((root / "spec.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (root / "_manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["row_count"] = invalid_count

    with pytest.raises(ValueError, match="manifest_row_count_invalid"):
        validate_dataset_manifest_payload(manifest, spec=spec_payload)


def test_low_level_fixture_requires_explicit_primary_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="low_level_fixture_primary_key_required"):
        write_low_level_parquet_artifact(
            tmp_path / "fixture",
            pd.DataFrame({"value": [1]}),
            primary_key=(),
            partition_columns=(),
        )


def test_exact_key_index_stores_primary_keys_once_and_category_values_only(
    tmp_path: Path,
) -> None:
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        index.add_table(pa.table({"asset_id": ["a", "b"]}))
        assert index._connection is not None
        dataset_rows = index._connection.execute(
            "SELECT payload, value FROM exact_keys "
            "WHERE namespace = 'dataset' ORDER BY payload"
        ).fetchall()
        assert len(dataset_rows) == 2
        assert all(payload and value is None for payload, value in dataset_rows)

        index.add_identity_payloads(
            namespace="logical-category:asset_id",
            payloads=((b"b", b"value-b"), (b"a", b"value-a")),
        )
        assert list(
            index.identity_payload_values(namespace="logical-category:asset_id")
        ) == [b"value-a", b"value-b"]


def test_exact_key_index_cleans_scratch_when_open_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        key_tracking.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
        ).__enter__()

    assert tuple(tmp_path.iterdir()) == ()


def test_exact_key_index_preserves_open_failure_when_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )

    class FailingConnection:
        def execute(self, _statement: str) -> None:
            raise RuntimeError("index_initialization_failed")

        def close(self) -> None:
            raise RuntimeError("index_close_failed")

    def connect(path: Path) -> FailingConnection:
        Path(path).touch()
        return FailingConnection()

    monkeypatch.setattr(key_tracking.sqlite3, "connect", connect)

    with pytest.raises(RuntimeError, match="index_initialization_failed"):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
        ).__enter__()

    assert tuple(tmp_path.iterdir()) == ()


def test_exact_key_index_compares_persisted_row_values(
    tmp_path: Path,
) -> None:
    left = pa.table(
        {
            "asset_id": ["a", "b", "d"],
            "value": [1.0, None, None],
            "nested": [[1, 2], [3], [4]],
        }
    )
    right = pa.table(
        {
            "asset_id": ["b", "c", "d"],
            "value": [2.0, float("inf"), None],
            "nested": [[3], [4], [4]],
        }
    )

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        assert (
            index.add_comparison_table(
                left,
                namespace="left",
                value_columns=("value", "nested"),
                payload_target_bytes=1024**2,
            )
            == 0
        )
        assert (
            index.add_comparison_table(
                right,
                namespace="right",
                value_columns=("value", "nested"),
                payload_target_bytes=1024**2,
            )
            == 0
        )
        assert index.comparison_counts(
            expected_namespace="left",
            actual_namespace="right",
        ) == (1, 1, 1)
        map_type = pa.map_(pa.string(), pa.int64())
        index.add_comparison_table(
            pa.table(
                {
                    "asset_id": ["map"],
                    "value": pa.array([[("b", 2), ("a", 1)]], type=map_type),
                }
            ),
            namespace="map-left",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        index.add_comparison_table(
            pa.table(
                {
                    "asset_id": ["map"],
                    "value": pa.array([[("a", 1), ("b", 2)]], type=map_type),
                }
            ),
            namespace="map-right",
            value_columns=("value",),
            payload_target_bytes=1024**2,
        )
        assert index.comparison_counts(
            expected_namespace="map-left",
            actual_namespace="map-right",
        ) == (0, 0, 0)


def _nested_float_zero_array(case: str) -> pa.Array:
    if case == "list":
        return pa.array([[-0.0], [0.0]], type=pa.list_(pa.float32()))
    if case == "large-list":
        return pa.array([[-0.0], [0.0]], type=pa.large_list(pa.float64()))
    if case == "fixed-list":
        return pa.array([[-0.0], [0.0]], type=pa.list_(pa.float32(), 1))
    if case == "struct":
        return pa.array(
            [{"zero": -0.0}, {"zero": 0.0}],
            type=pa.struct([pa.field("zero", pa.float64())]),
        )
    if case == "map-key":
        return pa.array(
            [[(-0.0, 1)], [(0.0, 1)]],
            type=pa.map_(pa.float32(), pa.int64()),
        )
    if case == "map-item":
        return pa.array(
            [[("zero", -0.0)], [("zero", 0.0)]],
            type=pa.map_(pa.string(), pa.float64()),
        )
    if case == "dictionary":
        return pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int8()),
            pa.array([-0.0, 0.0], type=pa.float32()),
        )
    return pa.array(
        [
            [{"zero": [-0.0]}],
            [{"zero": [0.0]}],
        ],
        type=pa.list_(
            pa.struct([pa.field("zero", pa.list_(pa.float64()))]),
        ),
    )


@pytest.mark.parametrize(
    "case",
    (
        "list",
        "large-list",
        "fixed-list",
        "struct",
        "map-key",
        "map-item",
        "dictionary",
        "recursive",
    ),
)
def test_nested_float_primary_key_zero_keeps_bit_identity(
    tmp_path: Path,
    case: str,
) -> None:
    table = pa.Table.from_arrays([_nested_float_zero_array(case)], names=("key",))

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
    ) as index:
        assert index.add_table(table, namespace="writer") == 0
        index.add_comparison_table(
            table.slice(0, 1),
            namespace="left",
            value_columns=(),
            payload_target_bytes=1024**2,
        )
        index.add_comparison_table(
            table.slice(1, 1),
            namespace="right",
            value_columns=(),
            payload_target_bytes=1024**2,
        )
        assert index.comparison_counts(
            expected_namespace="left",
            actual_namespace="right",
        ) == (1, 1, 0)


@pytest.mark.parametrize("data_type", (pa.float32(), pa.float64()))
def test_top_level_float_primary_key_fingerprint_rejects_unsupported_width(
    data_type: pa.DataType,
) -> None:
    table = pa.table({"key": pa.array([-0.0, 0.0], type=data_type)})
    with pytest.raises(ValueError, match="unsupported_dataset_fingerprint_value"):
        tuple(
            dataset_fingerprinting._iter_fingerprint_hashes(
                table.to_pandas(),
                target_bytes=1024**2,
                arrow_schema=table.schema,
                primary_key=("key",),
            )
        )


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_nested_primary_key_nonfinite_rejects_but_nested_value_allows(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    nested = pa.array([[nonfinite]], type=pa.list_(pa.float64()))
    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(ValueError, match="dataset_primary_key|null|invalid"):
            index.add_table(pa.Table.from_arrays([nested], names=("key",)))

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
    ) as index:
        assert (
            index.add_comparison_table(
                pa.table({"key": ["row"], "value": nested}),
                namespace="value",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
            == 0
        )


def test_exact_comparison_payload_target_has_an_exact_encoded_boundary(
    tmp_path: Path,
) -> None:
    table = pa.table({"asset_id": ["a"], "value": [1]})

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        _, key_payload, value_payload = next(
            index._prepared_comparison_rows(
                table,
                namespace="measure",
                value_columns=("value",),
                payload_target_bytes=1024,
            )
        )
        exact_target = len(key_payload) + len(value_payload)
        assert (
            index.add_comparison_table(
                table,
                namespace="exact",
                value_columns=("value",),
                payload_target_bytes=exact_target,
            )
            == 0
        )
        with pytest.raises(
            ValueError,
            match="dataset_comparison_payload_exceeds_target_bytes",
        ):
            index.add_comparison_table(
                table,
                namespace="too-small",
                value_columns=("value",),
                payload_target_bytes=exact_target - 1,
            )


def test_exact_comparison_primary_key_only_projection_has_empty_value(
    tmp_path: Path,
) -> None:
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        assert (
            index.add_comparison_table(
                pa.table({"asset_id": ["a"]}),
                namespace="primary-key-only",
                value_columns=(),
                payload_target_bytes=1024,
            )
            == 0
        )
        assert tuple(index.identity_payload_values(namespace="primary-key-only")) == (
            b"",
        )


def test_exact_comparison_nested_payload_stops_at_remaining_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = pa.table(
        {
            "asset_id": ["a"],
            "value": pa.array(
                [[None] * 5_000],
                type=pa.list_(pa.int64()),
            ),
        }
    )
    encode = key_tracking._append_arrow_scalar_payload
    calls = 0

    def counted_encode(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 64:
            raise AssertionError("exact_nested_payload_traversed_past_target")
        encode(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        key_tracking,
        "_append_arrow_scalar_payload",
        counted_encode,
    )
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        with pytest.raises(
            ValueError,
            match="dataset_comparison_payload_exceeds_target_bytes",
        ):
            index.add_comparison_table(
                table,
                namespace="bounded",
                value_columns=("value",),
                payload_target_bytes=1024,
            )
    assert calls <= 64


def test_exact_comparison_python_allocation_diagnostic_uses_one_bounded_row_buffer(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "asset_id": ["a"],
            "value": pa.array(
                [[None] * 24_000],
                type=pa.list_(pa.int64()),
            ),
        }
    )
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        _, key_payload, value_payload = next(
            index._prepared_comparison_rows(
                table,
                namespace="measure",
                value_columns=("value",),
                payload_target_bytes=1024**2,
            )
        )
        exact_target = len(key_payload) + len(value_payload)
        tracemalloc.start()
        try:
            assert (
                index.add_comparison_table(
                    table,
                    namespace="near-boundary",
                    value_columns=("value",),
                    payload_target_bytes=exact_target,
                )
                == 0
            )
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

    assert peak_bytes < exact_target * 4


@pytest.mark.parametrize(
    "data_type",
    (
        pa.string(),
        pa.large_string(),
        pa.binary(),
        pa.large_binary(),
        pa.binary(8),
    ),
)
def test_exact_variable_width_rejects_before_scalar_materialization(
    data_type: pa.DataType,
) -> None:
    class OversizedScalar:
        is_valid = True

        def __init__(self) -> None:
            self.as_buffer_calls = 0
            self.as_py_calls = 0

        def as_buffer(self) -> object:
            self.as_buffer_calls += 1
            return type("BufferView", (), {"size": 8 * 1024**2})()

        def as_py(self) -> object:
            self.as_py_calls += 1
            raise AssertionError("oversized_scalar_was_materialized")

    scalar = OversizedScalar()
    with pytest.raises(
        ValueError,
        match="dataset_comparison_payload_exceeds_target_bytes",
    ):
        key_tracking._append_arrow_scalar_payload(
            key_tracking._BoundedPayloadSink(1024),
            scalar,  # type: ignore[arg-type]
            data_type,
            allow_null=True,
        )
    assert scalar.as_buffer_calls == 1
    assert scalar.as_py_calls == 0


def test_exact_variable_width_columns_share_the_row_payload_target(
    tmp_path: Path,
) -> None:
    table = pa.table(
        {
            "asset_id": ["a"],
            "text": ["指标" * 16],
            "binary": [b"payload" * 16],
        }
    )
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    ) as index:
        _, key_payload, value_payload = next(
            index._prepared_comparison_rows(
                table,
                namespace="measure",
                value_columns=("text", "binary"),
                payload_target_bytes=1024**2,
            )
        )
        exact_target = len(key_payload) + len(value_payload)
        assert (
            index.add_comparison_table(
                table,
                namespace="exact-variable-width",
                value_columns=("text", "binary"),
                payload_target_bytes=exact_target,
            )
            == 0
        )
        with pytest.raises(
            ValueError,
            match="dataset_comparison_payload_exceeds_target_bytes",
        ):
            index.add_comparison_table(
                table,
                namespace="small-variable-width",
                value_columns=("text", "binary"),
                payload_target_bytes=exact_target - 1,
            )


def test_exact_map_accounts_compact_sort_state_before_allocation() -> None:
    map_type = pa.map_(pa.string(), pa.int64())
    scalar = pa.array(
        [[(f"key-{index:04d}", index) for index in reversed(range(256))]],
        type=map_type,
    )[0]
    unbounded = key_tracking._BoundedPayloadSink(None)
    key_tracking._append_arrow_scalar_payload(unbounded, scalar, map_type)

    with pytest.raises(
        ValueError,
        match="dataset_comparison_payload_exceeds_target_bytes",
    ):
        key_tracking._append_arrow_scalar_payload(
            key_tracking._BoundedPayloadSink(len(unbounded.buffer)),
            scalar,
            map_type,
        )


def test_exact_nested_map_encoding_has_no_recursive_reencode_amplification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = key_tracking._append_arrow_scalar_payload

    for depth in range(1, 9):
        data_type: pa.DataType = pa.int64()
        value: object = 1
        for _ in range(depth):
            data_type = pa.map_(pa.string(), data_type)
            value = [("key", value)]
        scalar = pa.array([value], type=data_type)[0]
        calls = 0

        def counted(*args: object, **kwargs: object) -> None:
            nonlocal calls
            calls += 1
            original(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            key_tracking,
            "_append_arrow_scalar_payload",
            counted,
        )
        key_tracking._append_arrow_scalar_payload(
            key_tracking._BoundedPayloadSink(None),
            scalar,
            data_type,
        )
        monkeypatch.setattr(
            key_tracking,
            "_append_arrow_scalar_payload",
            original,
        )

        assert calls == depth * 2 + 1


def test_exact_map_high_cardinality_and_multiplicity_are_order_independent() -> None:
    map_type = pa.map_(pa.string(), pa.int64())
    entries = [(f"key-{index:04d}", index % 7) for index in range(512)]
    entries.extend((("duplicate", 2), ("duplicate", 1)))
    forward = pa.array([entries], type=map_type)[0]
    reverse = pa.array([list(reversed(entries))], type=map_type)[0]

    forward_payload = key_tracking._BoundedPayloadSink(None)
    reverse_payload = key_tracking._BoundedPayloadSink(None)
    key_tracking._append_arrow_scalar_payload(forward_payload, forward, map_type)
    key_tracking._append_arrow_scalar_payload(reverse_payload, reverse, map_type)

    assert forward_payload.buffer == reverse_payload.buffer


def test_exact_key_index_close_failure_does_not_replace_primary(
    tmp_path: Path,
) -> None:
    class FailingCloseConnection:
        def __init__(self, connection: object) -> None:
            self.connection = connection

        def close(self) -> None:
            self.connection.close()  # type: ignore[attr-defined]
            raise RuntimeError("close_failure")

    index = ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
    )
    with pytest.raises(RuntimeError, match="primary_failure"):
        with index:
            scratch = index.database_path.parent
            assert index._connection is not None
            index._connection = FailingCloseConnection(index._connection)  # type: ignore[assignment]
            raise RuntimeError("primary_failure")

    assert not scratch.exists()


def test_dataset_comparison_failure_summary_does_not_expose_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    monkeypatch.setattr(
        dataset_comparison,
        "open_dataset_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(str(tmp_path))),
    )

    comparison = datasets.compare_datasets(left, right)

    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


def test_write_stats_fail_closed_paths(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with (
        runtime_resource_policy_scope(policy),
        ExactPrimaryKeyIndex(
            primary_key=spec.primary_key,
            timestamp_columns=("timestamp",),
        ) as tracker,
        DatasetSessionInventory(
            data_root=Path("data"),
            resource_policy=policy,
        ) as inventory,
    ):
        stats = _WriteStats(
            spec=spec,
            data_root=Path("data"),
            key_tracker=tracker,
            inventory=inventory,
        )

        with pytest.raises(ValueError, match="unregistered_output_columns"):
            stats.declared_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(extra=1),
                target_chunk_bytes=1024**2,
            )
        with pytest.raises(ValueError, match="missing_required_columns"):
            stats.declared_frame(
                make_helper_edges_dataset_frame(market_id="dummy").drop(
                    columns=["value"]
                ),
                target_chunk_bytes=1024**2,
            )
        with pytest.raises(ValueError, match="dataset_schema_missing"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=tracker,
                inventory=inventory,
            ).payload(target_chunk_bytes=1024**2)


def test_exact_index_preserves_decimal_values_beyond_context_precision(
    tmp_path: Path,
) -> None:
    schema = pa.schema([pa.field("key", pa.decimal128(38, 0))])
    first = Decimal("12345678901234567890123456789012345678")
    second = Decimal("12345678901234567890123456789012345679")

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        key_schema=schema,
    ) as index:
        assert index.add_table(pa.table({"key": [first]})) == 0
        assert index.add_table(pa.table({"key": [second]})) == 0


def test_exact_index_preserves_unclaimed_key_scratch_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    unclaimed = tmp_path / ".k-user-cache"
    unclaimed.mkdir()
    sentinel = unclaimed / "sentinel.txt"
    sentinel.write_text("unclaimed\n", encoding="utf-8")

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
    ):
        pass

    assert sentinel.read_text(encoding="utf-8") == "unclaimed\n"


def test_write_stats_rejects_invalid_non_null_timestamp(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    policy = capture_test_policy(available_memory_bytes=64 * 1024)
    with (
        runtime_resource_policy_scope(policy),
        ExactPrimaryKeyIndex(
            primary_key=spec.primary_key,
            timestamp_columns=("timestamp",),
        ) as tracker,
        DatasetSessionInventory(
            data_root=Path("data"),
            resource_policy=policy,
        ) as inventory,
    ):
        with pytest.raises(ValueError, match="invalid_dataset_timestamp_value"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=tracker,
                inventory=inventory,
            ).prepare_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    timestamp=["2024-01-02", "bad-date"]
                )
            )


def test_dataset_compare_reports_missing_sides(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"

    assert datasets.compare_datasets(left, right).status == "MISSING_LEFT"
    left.mkdir()
    assert datasets.compare_datasets(left, right).status == "MISSING_RIGHT"


def test_dataset_partition_and_manifest_shape_helpers() -> None:
    frame = make_helper_edges_dataset_frame(market_id="dummy")

    assert (
        list(
            partition_frames(
                frame.iloc[0:0],
                timestamp_column="timestamp",
                partition_columns=("year",),
            )
        )
        == []
    )
    assert (
        list(
            partition_frames(
                frame,
                timestamp_column="timestamp",
                partition_columns=(),
            )
        )[0][0]
        == ""
    )
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        list(
            partition_frames(
                frame,
                timestamp_column="timestamp",
                partition_columns=("month",),
            )
        )
    with pytest.raises(ValueError, match="missing_partition_timestamp_column"):
        list(
            partition_frames(
                frame.drop(columns=["timestamp"]),
                timestamp_column="timestamp",
                partition_columns=("year",),
            )
        )


def test_dataset_spec_payload_accepts_exact_base_variant_and_rejects_extras() -> None:
    base_spec = {
        "schema_id": "qrp-dataset-spec",
        "data_id": "asset_reference",
        "frequency": data_frequency_slug(DataFrequency.DAILY),
        "grain": "asset_reference",
        "primary_key": ["asset_id"],
        "partition_columns": [],
        "required_columns": ["asset_id", "name"],
        "column_roles": {"asset_id": "key", "name": "context"},
    }

    validate_dataset_spec_payload(base_spec)
    with pytest.raises(ValueError, match="dataset_spec_fields_invalid"):
        validate_dataset_spec_payload({**base_spec, "unknown": "value"})
    with pytest.raises(ValueError, match="invalid_primary_key_roles"):
        validate_dataset_spec_payload(
            {
                **base_spec,
                "column_roles": {"asset_id": "context", "name": "context"},
            }
        )


def test_physical_schema_identity_is_recursive_and_parameter_complete() -> None:
    nested_field = pa.field(
        "items",
        pa.list_(pa.field("item", pa.int64(), nullable=False)),
        nullable=True,
    )
    schema = pa.schema(
        [
            pa.field(
                "payload",
                pa.struct([nested_field]),
                nullable=False,
            ),
            pa.field(
                "category",
                pa.dictionary(pa.int8(), pa.string(), ordered=True),
                nullable=True,
            ),
        ]
    )

    identity = physical_schema_from_schema(schema)
    validate_physical_schema_payload(identity)

    payload_type = identity[0]["type"]
    assert payload_type["kind"] == "struct"
    list_field = payload_type["fields"][0]
    assert list_field["nullable"] is True
    assert list_field["type"]["value_field"]["nullable"] is False
    dictionary_type = identity[1]["type"]
    assert dictionary_type == {
        "kind": "dictionary",
        "index_type": {"kind": "integer", "signed": True, "bit_width": 8},
        "value_type": {"kind": "string"},
        "ordered": True,
    }

    top_nullable = physical_schema_from_schema(
        pa.schema(
            [schema.field("payload").with_nullable(True), schema.field("category")]
        )
    )
    child_nullable = physical_schema_from_schema(
        pa.schema(
            [
                pa.field(
                    "payload",
                    pa.struct(
                        [
                            pa.field(
                                "items",
                                pa.list_(pa.field("item", pa.int64(), nullable=True)),
                                nullable=True,
                            )
                        ]
                    ),
                    nullable=False,
                ),
                schema.field("category"),
            ]
        )
    )
    dictionary_changed = physical_schema_from_schema(
        pa.schema(
            [
                schema.field("payload"),
                pa.field(
                    "category",
                    pa.dictionary(pa.int16(), pa.string(), ordered=False),
                    nullable=True,
                ),
            ]
        )
    )
    assert (
        len(
            {
                canonical_json_hash(identity),
                canonical_json_hash(top_nullable),
                canonical_json_hash(child_nullable),
                canonical_json_hash(dictionary_changed),
            }
        )
        == 4
    )


def test_physical_schema_identity_rejects_unsupported_or_mutated_grammar() -> None:
    with pytest.raises(ValueError, match="unsupported_dataset_arrow_type"):
        physical_schema_from_schema(pa.schema([pa.field("unsupported", pa.null())]))

    identity = physical_schema_from_schema(
        pa.schema([pa.field("value", pa.int64(), nullable=False)])
    )
    identity[0]["type"]["unknown"] = True
    with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
        validate_physical_schema_payload(identity)


@pytest.mark.parametrize(
    "data_type",
    (pa.date64(), pa.list_(pa.date64())),
    ids=("top-level", "nested"),
)
def test_physical_schema_encoder_rejects_unsupported_date64(
    data_type: pa.DataType,
) -> None:
    with pytest.raises(ValueError, match="unsupported_dataset_arrow_type"):
        physical_schema_from_schema(
            pa.schema([pa.field("unsupported_date64", data_type)])
        )


@pytest.mark.parametrize("wrong_kind", ([], {}), ids=("list", "dict"))
@pytest.mark.parametrize(
    ("type_payload", "nested"),
    (
        ({"kind": "time", "unit": "ms"}, False),
        ({"kind": "duration", "unit": "ms"}, False),
        ({"kind": "timestamp", "unit": "ms", "timezone": None}, False),
        (
            {
                "kind": "struct",
                "fields": [
                    {
                        "name": "child",
                        "nullable": True,
                        "type": {
                            "kind": "timestamp",
                            "unit": "ms",
                            "timezone": None,
                        },
                    }
                ],
            },
            True,
        ),
    ),
    ids=("time", "duration", "timestamp", "nested-timestamp"),
)
def test_physical_schema_units_reject_unhashable_values_at_any_depth(
    type_payload: dict[str, object],
    nested: bool,
    wrong_kind: object,
) -> None:
    schema_payload = [
        {"name": "value", "nullable": True, "type": deepcopy(type_payload)}
    ]
    validate_physical_schema_payload(schema_payload)
    target = schema_payload[0]["type"]
    if nested:
        target = target["fields"][0]["type"]
    target["unit"] = wrong_kind

    with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
        validate_physical_schema_payload(schema_payload)


@pytest.mark.parametrize("wrong_kind", ("float", "bool", "string"))
@pytest.mark.parametrize(
    ("type_payload", "parameter_path"),
    (
        (
            {"kind": "integer", "signed": True, "bit_width": 32},
            ("bit_width",),
        ),
        ({"kind": "float", "bit_width": 64}, ("bit_width",)),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("bit_width",),
        ),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("precision",),
        ),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("scale",),
        ),
        (
            {"kind": "fixed_size_binary", "byte_width": 16},
            ("byte_width",),
        ),
        (
            {
                "kind": "fixed_size_list",
                "list_size": 3,
                "value_field": {
                    "name": "item",
                    "nullable": False,
                    "type": {"kind": "integer", "signed": True, "bit_width": 32},
                },
            },
            ("list_size",),
        ),
        (
            {
                "kind": "dictionary",
                "index_type": {
                    "kind": "integer",
                    "signed": True,
                    "bit_width": 8,
                },
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("index_type", "bit_width"),
        ),
        (
            {
                "kind": "dictionary",
                "index_type": {
                    "kind": "integer",
                    "signed": True,
                    "bit_width": 8,
                },
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("value_type", "bit_width"),
        ),
    ),
)
def test_physical_schema_numeric_parameters_require_exact_int_kind_recursively(
    type_payload: dict[str, object],
    parameter_path: tuple[str, ...],
    wrong_kind: str,
) -> None:
    schema_payload = [
        {"name": "value", "nullable": True, "type": deepcopy(type_payload)}
    ]
    validate_physical_schema_payload(schema_payload)
    target = schema_payload[0]["type"]
    for key in parameter_path[:-1]:
        target = target[key]
    parameter = parameter_path[-1]
    exact_value = target[parameter]
    target[parameter] = {
        "float": float(exact_value),
        "bool": True,
        "string": str(exact_value),
    }[wrong_kind]

    with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
        validate_physical_schema_payload(schema_payload)


def test_dataset_timestamp_and_candidate_file_helpers(tmp_path: Path) -> None:
    spec = data_spec_payload(
        make_processed_dataset_spec("storage_demo", market_id="dummy")
    )
    assert read_columns_for_time_window(
        ("market_id",),
        spec=spec,
        time_window=TimeWindow("2024-01-01", "2024-01-31"),
    ) == ("market_id", "timestamp")
    with pytest.raises(ValueError, match="invalid_column_roles"):
        timestamp_column_from_spec_payload({"column_roles": []})
    with pytest.raises(ValueError, match="invalid_timestamp_primary_key"):
        timestamp_column_from_spec_payload(
            {
                "primary_key": ["timestamp", "as_of"],
                "column_roles": {
                    "timestamp": ColumnRole.TIMESTAMP.value,
                    "as_of": ColumnRole.TIMESTAMP.value,
                },
            }
        )

    data_root = tmp_path / "data"
    data_root.mkdir()
    assert (
        _discovered_paths(
            data_root,
            spec={**spec, "partition_columns": []},
        )
        == []
    )
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        _discovered_paths(
            data_root,
            spec={**spec, "partition_columns": ["month"]},
        )


def test_inventory_discovery_orders_numeric_part_indices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for index in (1001, 999, 1000, 998):
        pq.write_table(
            pa.table({"value": [index]}),
            data_root / f"part-{index}.parquet",
        )

    paths = _discovered_paths(
        data_root,
        spec={"partition_columns": []},
    )

    assert [path.name for path in paths] == [
        "part-998.parquet",
        "part-999.parquet",
        "part-1000.parquet",
        "part-1001.parquet",
    ]


@pytest.mark.parametrize(
    "names",
    [
        ("part-invalid.parquet",),
        ("part-1.parquet", "part-001.parquet"),
        ("part-000.parquet", "stray.parquet"),
    ],
    ids=["invalid", "duplicate-index", "stray"],
)
def test_inventory_discovery_rejects_invalid_or_ambiguous_parts(
    tmp_path: Path,
    names: tuple[str, ...],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in names:
        pq.write_table(pa.table({"value": [1]}), data_root / name)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(
            data_root,
            spec={"partition_columns": []},
        )


def test_inventory_discovery_rejects_duplicate_numeric_years(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    for year_name in ("year=1", "year=01"):
        year_root = data_root / year_name
        year_root.mkdir(parents=True)
        pq.write_table(pa.table({"value": [1]}), year_root / "part-000.parquet")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(
            data_root,
            spec={"partition_columns": ["year"]},
        )


def test_inventory_discovery_rejects_resolved_path_outside_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.parquet"
    pq.write_table(pa.table({"value": [1]}), external)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        alias = data_root / "part-000.parquet"
        try:
            alias.symlink_to(external)
        except OSError as exc:
            pytest.skip(f"symlink_not_available: {exc}")
        _discovered_paths(data_root, spec={"partition_columns": []})


def test_inventory_discovery_rejects_symlinked_part_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    part = data_root / "part-000.parquet"
    pq.write_table(pa.table({"value": [1]}), part)
    alias = data_root / "part-001.parquet"
    try:
        alias.symlink_to(part)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})


def test_inventory_discovery_rejects_symlinked_partition_alias(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    year = data_root / "year=2024"
    year.mkdir(parents=True)
    pq.write_table(pa.table({"value": [1]}), year / "part-000.parquet")
    alias = data_root / "year=2025"
    try:
        alias.symlink_to(year, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": ["year"]})


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_inventory_discovery_rejects_junction_data_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pq.write_table(pa.table({"value": [1]}), external / "part-000.parquet")
    data_root = tmp_path / "data"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(data_root), str(external)],
        check=False,
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert data_root.is_junction()
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})


@pytest.mark.skipif(
    os.name != "nt" or not hasattr(Path, "is_junction"),
    reason="Windows junction capability required",
)
def test_inventory_discovery_does_not_descend_nested_junction(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pq.write_table(pa.table({"value": [1]}), external / "part-000.parquet")
    data_root = tmp_path / "data"
    data_root.mkdir()
    junction = data_root / "nested"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert junction.is_junction()
    assert _discovered_paths(data_root, spec={"partition_columns": []}) == []


def test_inventory_discovery_maps_scan_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()

    def fail_scan(_path):
        raise OSError("scan_failed")

    monkeypatch.setattr(os, "scandir", fail_scan)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})


def test_directory_scan_processes_first_entry_before_enumerating_siblings(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    root.mkdir()
    for index in range(64):
        pq.write_table(pa.table({"value": [index]}), root / f"part-{index:03d}.parquet")
    scanned, processed = [], []
    scan = os.scandir

    class Entries:
        def __init__(self, path):
            self.inner = scan(path)

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.inner)
            scanned.append(entry.name)
            return entry

        def close(self):
            self.inner.close()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        os,
        "scandir",
        lambda path: (
            Entries(path)
            if not isinstance(path, int) and Path(path).name == root.name
            else scan(path)
        ),
    )
    validate = DatasetSessionInventory._validated_coordinates

    def observed(self, *args, **kwargs):
        processed.append(len(scanned))
        return validate(self, *args, **kwargs)

    monkeypatch.setattr(DatasetSessionInventory, "_validated_coordinates", observed)
    paths = _discovered_paths(root, spec={"partition_columns": []})
    assert [path.name for path in paths] == [f"part-{i:03d}.parquet" for i in range(64)]
    assert processed[0] == 2


def test_inventory_discovery_rejects_hardlinked_part_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    first = data_root / "part-000.parquet"
    alias = data_root / "part-001.parquet"
    pq.write_table(pa.table({"value": [1]}), first)
    try:
        os.link(first, alias)
    except OSError:
        pytest.skip("hardlink_not_available")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})


def test_inventory_discovery_accepts_single_external_hardlink(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.parquet"
    part = data_root / "part-000.parquet"
    pq.write_table(pa.table({"value": [1]}), external)
    try:
        os.link(external, part)
    except OSError:
        pytest.skip("hardlink_not_available")

    assert _discovered_paths(data_root, spec={"partition_columns": []}) == [part]


def test_inventory_discovery_rejects_case_variant_parquet_suffix(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    pq.write_table(pa.table({"value": [1]}), data_root / "part-000.parquet")
    pq.write_table(pa.table({"value": [2]}), data_root / "part-001.PARQUET")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})
