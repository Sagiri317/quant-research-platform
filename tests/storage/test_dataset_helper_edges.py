from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.comparison as dataset_comparison
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
            "asset_id": ["asset-00"],
            "value": [0],
        }
    )
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    with pytest.raises(ValueError, match=error):
        index.classify_comparison_table(
            table, namespace=namespace, value_columns=value_columns
        )


def test_exact_key_classifier_classifies_pending_duplicates() -> None:
    table = pa.table({"asset_id": ["same"] * 3})
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    keep, duplicate_count = index.classify_comparison_table(
        table, namespace="duplicates", value_columns=()
    )
    assert list(keep) == [0]
    assert duplicate_count == 2


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


def test_persisted_research_key_coverage_rejects_unhashable_policy_with_owner_error():
    wrong_kind = []
    payload = data_spec_payload(make_demo_research_dataset_spec())
    data_id = payload["input_data_ids"][0]
    payload["key_coverage"][data_id] = wrong_kind
    with pytest.raises(ValueError, match="dataset_key_coverage_invalid"):
        validate_dataset_spec_payload(payload)


@pytest.mark.parametrize(
    "type_payload",
    [
        {"kind": "decimal", "bit_width": 128, "precision": 39, "scale": 0},
        {"kind": "decimal", "bit_width": 256, "precision": 77, "scale": 0},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": -1},
        {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 19},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": -1},
        {"kind": "decimal", "bit_width": 256, "precision": 40, "scale": 41},
        {"kind": "fixed_size_binary", "byte_width": 0},
        {"kind": "date", "unit": "millisecond"},
        {"kind": "time", "unit": "s"},
        {"kind": "timestamp", "unit": "s", "timezone": None},
        {
            "kind": "dictionary",
            "index_type": {"kind": "integer", "signed": False, "bit_width": 8},
            "value_type": {"kind": "string"},
            "ordered": False,
        },
    ],
    ids=[
        "decimal128-precision",
        "decimal256-precision",
        "decimal128-negative-scale",
        "decimal128-scale-over-precision",
        "decimal256-negative-scale",
        "decimal256-scale-over-precision",
        "fixed-size-binary-zero",
        "date-millisecond",
        "time-second",
        "timestamp-second",
        "dictionary-unsigned-index",
    ],
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
    [
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
        pa.dictionary(pa.int8(), pa.string()),
    ],
    ids=[
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
        "signed-dictionary",
    ],
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


def test_manifest_counts_reject_non_int_numeric_types(tmp_path: Path) -> None:
    invalid_count = Decimal("1")
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
            "label": ["a", "b", "c"],
        }
    )
    right = pa.table(
        {
            "asset_id": ["b", "c", "d"],
            "value": [2.0, float("inf"), None],
            "label": ["b", "c", "c"],
        }
    )
    index = ExactPrimaryKeyIndex(primary_key=("asset_id",))
    assert (
        index.add_comparison_table(
            left, namespace="left", value_columns=("value", "label")
        )
        == 0
    )
    assert index._compare_table(
        right,
        expected_namespace="left",
        actual_namespace="right",
        value_columns=("value", "label"),
    ) == (0, 1)
    assert index.difference_counts(
        expected_namespace="left", actual_namespace="right"
    ) == (1, 1)


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_scalar_primary_key_nonfinite_rejects_but_scalar_value_allows(
    nonfinite: float,
) -> None:
    nested = pa.array([nonfinite], type=pa.float64())
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
    with pytest.raises(ValueError, match="invalid_dataset_timestamp_value"):
        _WriteStats(
            spec=spec, data_root=Path("data"), key_tracker=tracker
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
        "column_units": {},
        "label_availability_columns": {},
    }
    validate_dataset_spec_payload(base_spec)
    with pytest.raises(ValueError, match="dataset_spec_fields_invalid"):
        validate_dataset_spec_payload({**base_spec, "unknown": "value"})
    with pytest.raises(ValueError, match="invalid_primary_key_roles"):
        validate_dataset_spec_payload(
            {**base_spec, "column_roles": {"asset_id": "context", "name": "context"}}
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


@pytest.mark.parametrize(
    "type_payload",
    [
        {"kind": "time", "unit": "ms"},
        {"kind": "duration", "unit": "ms"},
        {"kind": "timestamp", "unit": "ms", "timezone": None},
    ],
    ids=["time", "duration", "timestamp"],
)
def test_physical_schema_units_reject_unhashable_values(
    type_payload: dict[str, object],
) -> None:
    wrong_kind = []
    schema_payload = [
        {"name": "value", "nullable": True, "type": deepcopy(type_payload)}
    ]
    validate_physical_schema_payload(schema_payload)
    target = schema_payload[0]["type"]
    target["unit"] = wrong_kind
    with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
        validate_physical_schema_payload(schema_payload)


@pytest.mark.parametrize(
    ("type_payload", "parameter_path", "wrong_kind"),
    [
        ({"kind": "integer", "signed": True, "bit_width": 32}, ("bit_width",), "float"),
        ({"kind": "float", "bit_width": 64}, ("bit_width",), "float"),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("bit_width",),
            "float",
        ),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("precision",),
            "float",
        ),
        (
            {"kind": "decimal", "bit_width": 128, "precision": 18, "scale": 4},
            ("scale",),
            "float",
        ),
        ({"kind": "fixed_size_binary", "byte_width": 16}, ("byte_width",), "float"),
        (
            {
                "kind": "dictionary",
                "index_type": {"kind": "integer", "signed": True, "bit_width": 8},
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("index_type", "bit_width"),
            "float",
        ),
        (
            {
                "kind": "dictionary",
                "index_type": {"kind": "integer", "signed": True, "bit_width": 8},
                "value_type": {"kind": "float", "bit_width": 32},
                "ordered": False,
            },
            ("value_type", "bit_width"),
            "float",
        ),
        ({"kind": "integer", "signed": True, "bit_width": 32}, ("bit_width",), "bool"),
        (
            {"kind": "integer", "signed": True, "bit_width": 32},
            ("bit_width",),
            "string",
        ),
    ],
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


@pytest.mark.parametrize(
    "names",
    [
        ("part-invalid.parquet",),
        ("part-000.parquet", "stray.parquet"),
    ],
    ids=["invalid", "stray"],
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
        pa.table({"key": [0.0], "value": [1.0]}),
        namespace="left",
        value_columns=("value",),
    )
    right = pa.table({"key": [1.0, float("inf")], "value": [2.0, 3.0]})
    with pytest.raises(ValueError, match="^invalid_dataset_primary_key_value$"):
        index._compare_table(
            right,
            expected_namespace="left",
            actual_namespace="right",
            value_columns=("value",),
        )
    assert index.identity_payload_count(namespace="right") == 0
    assert index.identity_payload_count(namespace="left") == 1


@pytest.mark.parametrize("invalid", ([], {}, True, "winsorize"))
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


def test_resolved_feature_options_are_the_persisted_transform_configuration():
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
            {
                "recipe_id": recipe,
                "source_columns": [factor],
                "output_groups": ["processed"],
                "clip": transform.clip,
                "industry_neutralize": True,
                "size_neutralize": True,
                "size_column": "log_total_market_cap",
                "center": True,
                "scale": True,
            }
        ]
        identities.append(canonical_json_hash(payload))
        payload["feature_transforms"][0]["fillna_policy"] = "zero"
        with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
            validate_dataset_spec_payload(payload)
    assert identities[0] != identities[1]


def test_persisted_feature_size_cannot_be_inferred_from_new_defaults():
    factor = make_demo_research_dataset_spec().factor_columns[0]
    transform = FeatureTransformSpec("a_share_canonical", (factor,))
    payload = data_spec_payload(
        make_demo_research_dataset_spec(
            feature_columns=transform.output_columns, feature_transforms=(transform,)
        )
    )
    del payload["feature_transforms"][0]["size_column"]
    with pytest.raises(ValueError, match="dataset_feature_transforms_invalid"):
        validate_dataset_spec_payload(payload)
