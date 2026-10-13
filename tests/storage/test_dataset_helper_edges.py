from __future__ import annotations

import json
import os
import subprocess
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
from quant_research_platform.storage import datasets
from quant_research_platform.storage.datasets.inventory import DatasetSessionInventory
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage.datasets.reading import (
    projection_columns_for_window,
)
from quant_research_platform.storage.datasets.schema import (
    data_spec_payload,
    physical_schema_from_schema,
    validate_dataset_manifest_payload,
    validate_dataset_spec,
    validate_dataset_spec_payload,
    validate_physical_schema_payload,
)
from quant_research_platform.storage.datasets.writing import _WriteStats
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import native_filesystem_path
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_helper_edges_dataset_frame,
    make_processed_dataset_spec,
    write_low_level_parquet_artifact,
)


def _discovered_paths(data_root: Path, *, spec: dict[str, object]) -> list[Path]:
    inventory = DatasetSessionInventory(data_root=data_root)
    inventory.discover(spec)
    return list(inventory.paths)


def test_exact_key_index_classifies_typed_exact_duplicates_and_conflicts() -> None:
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
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    keep_positions, duplicate_count = index.classify_comparison_table(
        table, namespace="reference", value_columns=("value",)
    )
    assert list(keep_positions) == [0, 2]
    assert duplicate_count == 1
    keep_positions, duplicate_count = index.classify_comparison_table(
        exact_again, namespace="reference", value_columns=("value",)
    )
    assert list(keep_positions) == []
    assert duplicate_count == 1
    keep_positions, duplicate_count = index.classify_comparison_table(
        exact_again, namespace="independent", value_columns=("value",)
    )
    assert list(keep_positions) == [0]
    assert duplicate_count == 0
    with pytest.raises(ValueError, match="conflicting_exact_primary_key_payload"):
        index.classify_comparison_table(
            conflict, namespace="reference", value_columns=("value",)
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
    )
    assert list(keep_positions) == [0]
    assert duplicate_count == 0


@pytest.mark.parametrize(
    ("namespace", "value_columns", "error"),
    (
        ("", (), "invalid_exact_key_namespace"),
        ("grammar", ("value", "value"), "invalid_dataset_comparison_columns"),
        ("grammar", ("asset_id",), "invalid_dataset_comparison_columns"),
        ("grammar", ("missing",), "missing_dataset_comparison_column"),
    ),
)
def test_exact_key_classifier_validates_comparison_columns(
    namespace: str, value_columns: tuple[str, ...], error: str
) -> None:
    table = pa.table(
        {
            "asset_id": [f"asset-{position:02d}" for position in range(64)],
            "value": range(64),
        }
    )
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    with pytest.raises(ValueError, match=error):
        index.classify_comparison_table(
            table, namespace=namespace, value_columns=value_columns
        )


def test_exact_key_classifier_batches_large_duplicate_group() -> None:
    table = pa.table({"asset_id": ["same"] * 4096})
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    keep, duplicate_count = index.classify_comparison_table(
        table, namespace="duplicates", value_columns=()
    )
    assert list(keep) == [0]
    assert duplicate_count == 4095


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
        replace(interim, source_id=7)
    with pytest.raises(ValueError, match="invalid_market_id"):
        replace(processed, market_id=7)
    with pytest.raises(ValueError, match="invalid_market_id"):
        replace(research, market_id=7)


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
    field: str, value: object, error: str
) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with pytest.raises(ValueError, match=error):
        replace(spec, **{field: value})


@pytest.mark.parametrize("field", ("source_input_data_ids", "processed_input_data_ids"))
def test_processed_dependencies_are_unique_at_constructor_and_persisted_boundary(field):
    spec = make_processed_dataset_spec("unique_dependencies")
    with pytest.raises(ValueError, match="unique"):
        replace(spec, **{field: ("source", "source")})
    payload = data_spec_payload(spec)
    payload[field] = ["source", "source"]
    with pytest.raises(ValueError):
        validate_dataset_spec_payload(payload)


def test_validate_dataset_spec_accepts_only_research_immutable_semantic_mappings() -> (
    None
):
    research = make_demo_research_dataset_spec()
    assert type(research.column_roles) is MappingProxyType
    assert type(research.key_coverage) is MappingProxyType
    validate_dataset_spec(research)
    processed = make_processed_dataset_spec("storage_demo", market_id="dummy")
    validate_dataset_spec(processed)


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
    "type_payload",
    (
        {"kind": "decimal", "bit_width": 128, "precision": 39, "scale": 0},
        {"kind": "decimal", "bit_width": 256, "precision": 77, "scale": 0},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": -1},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 19},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": -1},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": 41},
        {"kind": "fixed_size_binary", "byte_width": 0},
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
            "key_field": {"name": "key", "nullable": True, "type": {"kind": "string"}},
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
            pa.field("element", pa.dictionary(pa.int16(), pa.string()), nullable=True)
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
    tmp_path: Path, data_type: pa.DataType
) -> None:
    schema = pa.schema([pa.field("value", data_type, nullable=True)])
    expected = physical_schema_from_schema(schema)
    table = pa.Table.from_arrays([pa.array([], type=data_type)], schema=schema)
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
    tmp_path: Path, invalid_count: object
) -> None:
    root = make_dataset_artifact_root(
        tmp_path, data_id="storage_demo", market_id="dummy"
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


def test_exact_key_index_stores_primary_keys_once_and_category_values_only() -> None:
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    index.add_table(pa.table({"asset_id": ["a", "b"]}))
    dataset_rows = sorted(index._namespaces["dataset"].items())
    assert len(dataset_rows) == 2
    assert all((payload and value is None for payload, value in dataset_rows))
    index.add_identity_payloads(
        namespace="logical-category:asset_id",
        payloads=((b"b", b"value-b"), (b"a", b"value-a")),
    )
    assert list(
        index.identity_payload_values(namespace="logical-category:asset_id")
    ) == [b"value-a", b"value-b"]


def test_exact_key_index_compares_persisted_row_values() -> None:
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
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    assert (
        index.add_comparison_table(
            left, namespace="left", value_columns=("value", "nested")
        )
        == 0
    )
    assert index._compare_table(
        right,
        expected_namespace="left",
        actual_namespace="right",
        value_columns=("value", "nested"),
    ) == (0, 1)
    assert index.difference_counts(
        expected_namespace="left", actual_namespace="right"
    ) == (1, 1)
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
    )
    assert index._compare_table(
        pa.table(
            {
                "asset_id": ["map"],
                "value": pa.array([[("a", 1), ("b", 2)]], type=map_type),
            }
        ),
        expected_namespace="map-left",
        actual_namespace="map-right",
        value_columns=("value",),
    ) == (0, 0)
    assert index.difference_counts(
        expected_namespace="map-left", actual_namespace="map-right"
    ) == (0, 0)


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
            [[(-0.0, 1)], [(0.0, 1)]], type=pa.map_(pa.float32(), pa.int64())
        )
    if case == "map-item":
        return pa.array(
            [[("zero", -0.0)], [("zero", 0.0)]], type=pa.map_(pa.string(), pa.float64())
        )
    if case == "dictionary":
        return pa.DictionaryArray.from_arrays(
            pa.array([0, 1], type=pa.int8()), pa.array([-0.0, 0.0], type=pa.float32())
        )
    return pa.array(
        [[{"zero": [-0.0]}], [{"zero": [0.0]}]],
        type=pa.list_(pa.struct([pa.field("zero", pa.list_(pa.float64()))])),
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
def test_nested_float_primary_key_zero_keeps_bit_identity(case: str) -> None:
    table = pa.Table.from_arrays([_nested_float_zero_array(case)], names=("key",))
    index = ExactPrimaryKeyIndex(primary_key=("key",))
    assert index.add_table(table, namespace="writer") == 0
    index.add_comparison_table(table.slice(0, 1), namespace="left", value_columns=())
    assert index._compare_table(
        table.slice(1, 1),
        expected_namespace="left",
        actual_namespace="right",
        value_columns=(),
    ) == (0, 0)
    assert index.difference_counts(
        expected_namespace="left", actual_namespace="right"
    ) == (1, 1)


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_nested_primary_key_nonfinite_rejects_but_nested_value_allows(
    nonfinite: float,
) -> None:
    nested = pa.array([[nonfinite]], type=pa.list_(pa.float64()))
    index = ExactPrimaryKeyIndex(primary_key=("key",))
    with pytest.raises(ValueError, match="dataset_primary_key|null|invalid"):
        index.add_table(pa.Table.from_arrays([nested], names=("key",)))
    index = ExactPrimaryKeyIndex(primary_key=("key",))
    assert (
        index.add_comparison_table(
            pa.table({"key": ["row"], "value": nested}),
            namespace="value",
            value_columns=("value",),
        )
        == 0
    )


def test_exact_comparison_primary_key_only_projection_has_empty_value() -> None:
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    assert (
        index.add_comparison_table(
            pa.table({"asset_id": ["a"]}),
            namespace="primary-key-only",
            value_columns=(),
        )
        == 0
    )
    assert tuple(index.identity_payload_values(namespace="primary-key-only")) == (b"",)


def test_exact_map_high_cardinality_and_multiplicity_are_order_independent() -> None:
    map_type = pa.map_(pa.string(), pa.int64())
    entries = [(f"key-{index:04d}", index % 7) for index in range(512)]
    entries.extend((("duplicate", 2), ("duplicate", 1), ("duplicate", 1)))
    forward = pa.array([entries], type=map_type)[0]
    reverse = pa.array([list(reversed(entries))], type=map_type)[0]
    forward_payload = bytearray()
    reverse_payload = bytearray()
    key_tracking._append_arrow_scalar_payload(forward_payload, forward, map_type)
    key_tracking._append_arrow_scalar_payload(reverse_payload, reverse, map_type)
    assert forward_payload == reverse_payload
    reduced_payload = bytearray()
    key_tracking._append_arrow_scalar_payload(
        reduced_payload, pa.array([entries[:-1]], type=map_type)[0], map_type
    )
    assert reduced_payload != forward_payload


def test_dataset_comparison_failure_summary_does_not_expose_local_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_exact_index_preserves_decimal_values_beyond_context_precision() -> None:
    schema = pa.schema([pa.field("key", pa.decimal128(38, 0))])
    first = Decimal("12345678901234567890123456789012345678")
    second = Decimal("12345678901234567890123456789012345679")
    index = ExactPrimaryKeyIndex(primary_key=("key",), key_schema=schema)
    assert index.add_table(pa.table({"key": [first]})) == 0
    assert index.add_table(pa.table({"key": [second]})) == 0


def test_write_stats_rejects_invalid_non_null_timestamp() -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    tracker = ExactPrimaryKeyIndex(primary_key=spec.primary_key)
    inventory = DatasetSessionInventory(data_root=Path("data"))
    with pytest.raises(ValueError, match="invalid_dataset_timestamp_value"):
        _WriteStats(
            spec=spec, data_root=Path("data"), key_tracker=tracker, inventory=inventory
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
            partition_frames(frame, timestamp_column="timestamp", partition_columns=())
        )[0][0]
        == ""
    )
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        list(
            partition_frames(
                frame, timestamp_column="timestamp", partition_columns=("month",)
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
            {**base_spec, "column_roles": {"asset_id": "context", "name": "context"}}
        )


def test_physical_schema_identity_is_recursive_and_parameter_complete() -> None:
    nested_field = pa.field(
        "items", pa.list_(pa.field("item", pa.int64(), nullable=False)), nullable=True
    )
    schema = pa.schema(
        [
            pa.field("payload", pa.struct([nested_field]), nullable=False),
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
    "data_type", (pa.date64(), pa.list_(pa.date64())), ids=("top-level", "nested")
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
                        "type": {"kind": "timestamp", "unit": "ms", "timezone": None},
                    }
                ],
            },
            True,
        ),
    ),
    ids=("time", "duration", "timestamp", "nested-timestamp"),
)
def test_physical_schema_units_reject_unhashable_values_at_any_depth(
    type_payload: dict[str, object], nested: bool, wrong_kind: object
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
        ({"kind": "integer", "signed": True, "bit_width": 32}, ("bit_width",)),
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
        ({"kind": "fixed_size_binary", "byte_width": 16}, ("byte_width",)),
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
                "index_type": {"kind": "integer", "signed": True, "bit_width": 8},
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("index_type", "bit_width"),
        ),
        (
            {
                "kind": "dictionary",
                "index_type": {"kind": "integer", "signed": True, "bit_width": 8},
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("value_type", "bit_width"),
        ),
    ),
)
def test_physical_schema_numeric_parameters_require_exact_int_kind_recursively(
    type_payload: dict[str, object], parameter_path: tuple[str, ...], wrong_kind: str
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
    assert projection_columns_for_window(
        ("market_id",), spec=spec, time_window=TimeWindow("2024-01-01", "2024-01-31")
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
    assert _discovered_paths(data_root, spec={**spec, "partition_columns": []}) == []
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        _discovered_paths(data_root, spec={**spec, "partition_columns": ["month"]})


def test_inventory_discovery_orders_numeric_part_indices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for index in (1001, 999, 1000, 998):
        pq.write_table(
            pa.table({"value": [index]}), data_root / f"part-{index}.parquet"
        )
    paths = _discovered_paths(data_root, spec={"partition_columns": []})
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
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in names:
        pq.write_table(pa.table({"value": [1]}), data_root / name)
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": []})


def test_inventory_discovery_rejects_duplicate_numeric_years(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for year_name in ("year=1", "year=01"):
        year_root = data_root / year_name
        year_root.mkdir(parents=True)
        pq.write_table(pa.table({"value": [1]}), year_root / "part-000.parquet")
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _discovered_paths(data_root, spec={"partition_columns": ["year"]})


def test_inventory_discovery_rejects_resolved_path_outside_root(tmp_path: Path) -> None:
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


def test_inventory_discovery_rejects_symlinked_partition_alias(tmp_path: Path) -> None:
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
def test_inventory_discovery_does_not_descend_nested_junction(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pq.write_table(pa.table({"value": [1]}), external / "part-000.parquet")
    data_root = tmp_path / "data"
    data_root.mkdir()
    junction = data_root / "nested"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)], capture_output=True
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")
    assert junction.is_junction()
    assert _discovered_paths(data_root, spec={"partition_columns": []}) == []


@pytest.mark.parametrize(
    "failure_type", (FileNotFoundError, NotADirectoryError, OSError, PermissionError)
)
def test_inventory_discovery_classifies_scan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[OSError]
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    primary = failure_type("scan_failed")
    native_scan = os.scandir

    def fail_scan(path):
        if path == native_filesystem_path(data_root):
            raise primary
        return native_scan(path)

    monkeypatch.setattr(os, "scandir", fail_scan)
    missing = failure_type in (FileNotFoundError, NotADirectoryError)
    with pytest.raises(ValueError if missing else failure_type) as caught:
        _discovered_paths(data_root, spec={"partition_columns": []})
    if missing:
        assert str(caught.value) == "invalid_dataset_part_files"
        assert caught.value.__cause__ is primary
    else:
        assert caught.value is primary
    assert not getattr(caught.value, "__notes__", ())


def test_directory_scan_processes_first_entry_before_enumerating_siblings(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    root.mkdir()
    for index in range(64):
        pq.write_table(pa.table({"value": [index]}), root / f"part-{index:03d}.parquet")
    scanned, processed = ([], [])
    scan = os.scandir

    class Entries:
        def __init__(self, path):
            self.inner = scan(path)

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.inner)
            scanned.append(entry.name)
            assert processed or len(scanned) <= 2
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
    assert 1 <= processed[0] <= 2


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


@pytest.mark.parametrize("data_type", (pa.float16(), pa.float32(), pa.float64()))
def test_top_level_float_zero_key_encodings_agree(data_type):
    table = pa.table({"key": pa.array([-0.0, 0.0], type=data_type)})
    index = ExactPrimaryKeyIndex(primary_key=("key",))
    assert index.add_table(table, namespace="writer") == 1
    assert (
        index.add_comparison_table(table, namespace="comparison", value_columns=()) == 1
    )


def test_exact_comparison_bad_late_key_keeps_right_batch_atomic():
    index = ExactPrimaryKeyIndex(primary_key=("key",))
    index.add_comparison_table(
        pa.table({"key": [[0.0]], "value": [1.0]}),
        namespace="left",
        value_columns=("value",),
    )
    right = pa.table({"key": [[1.0], [float("inf")]], "value": [2.0, 3.0]})
    with pytest.raises(ValueError, match="^invalid_dataset_primary_key_value$"):
        index._compare_table(
            right,
            expected_namespace="left",
            actual_namespace="right",
            value_columns=("value",),
        )
    assert index.identity_payload_count(namespace="right") == 0
    assert index.identity_payload_count(namespace="left") == 1


@pytest.mark.parametrize("invalid", ([], {}, True, 10**400, "winsorize"))
def test_feature_recipe_rejects_invalid_persisted_recipe(invalid):
    factor = make_demo_research_dataset_spec().factor_columns[0]
    transform = FeatureTransformSpec("a_share_canonical", (factor,))
    spec = make_demo_research_dataset_spec(
        feature_columns=transform.output_columns,
        feature_transforms=(transform,),
    )
    payload = data_spec_payload(spec)
    payload["feature_transforms"][0]["recipe_id"] = invalid
    with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
        validate_dataset_spec_payload(payload)


def test_feature_recipe_is_the_only_persisted_transform_configuration():
    factor = make_demo_research_dataset_spec().factor_columns[0]
    identities = []
    for recipe in ("a_share_canonical", "a_share_quantile_1_99"):
        transform = FeatureTransformSpec(recipe, (factor,))
        payload = data_spec_payload(
            make_demo_research_dataset_spec(
                feature_columns=transform.output_columns,
                feature_transforms=(transform,),
            )
        )
        validate_dataset_spec_payload(payload)
        assert payload["feature_transforms"] == [
            {"recipe_id": recipe, "source_columns": [factor]}
        ]
        identities.append(canonical_json_hash(payload))
        payload["feature_transforms"][0]["fillna_policy"] = "zero"
        with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
            validate_dataset_spec_payload(payload)
    assert identities[0] != identities[1]
