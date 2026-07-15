from __future__ import annotations

import inspect
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

import quant_research_platform.storage._dataset_comparison as dataset_comparison
import quant_research_platform.storage._dataset_key_tracking as key_tracking
from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    InterimDataSpec,
    data_frequency_slug,
)
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from quant_research_platform.storage._dataset_key_tracking import (
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_manifest_validation import (
    validate_manifest_runtime_metadata_shape,
)
from quant_research_platform.storage._dataset_paths import (
    _require_only_expected_parquet_paths,
    candidate_data_files,
    partition_frames,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_reading import (
    filter_frame_by_time_window,
    read_columns_for_time_window,
)
from quant_research_platform.storage._dataset_schema import (
    data_spec_payload,
    logical_dtype_payload_from_frame,
    physical_schema_from_schema,
    validate_dataset_manifest_payload,
    validate_dataset_spec,
    validate_dataset_spec_payload,
    validate_physical_schema_payload,
)
from quant_research_platform.storage._dataset_writing import _WriteStats
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_helper_edges_dataset_frame,
    make_processed_dataset_spec,
    write_low_level_parquet_artifact,
)


def test_exact_key_index_does_not_bypass_bounded_pandas_conversion() -> None:
    source = inspect.getsource(ExactPrimaryKeyIndex.add_table)

    assert ".to_pandas(" not in source


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

    with pytest.raises(ValueError, match="dataset_source_id_invalid"):
        data_spec_payload(replace(interim, source_id=7))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dataset_market_id_invalid"):
        data_spec_payload(replace(processed, market_id=7))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dataset_market_id_invalid"):
        data_spec_payload(replace(research, market_id=7))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("data_id", 7, "dataset_data_id_invalid"),
        (
            "frequency",
            data_frequency_slug(DataFrequency.DAILY),
            "dataset_frequency_invalid",
        ),
        ("grain", "asset_timestamp", "dataset_grain_invalid"),
        (
            "primary_key",
            ["market_id", "asset_id", "timestamp"],
            "dataset_primary_key_invalid",
        ),
        ("partition_columns", ["year"], "dataset_partition_columns_invalid"),
        ("kind", "factor", "dataset_kind_invalid"),
        ("source_input_data_ids", ["raw"], "dataset_source_input_data_ids_invalid"),
        (
            "processed_input_data_ids",
            ["processed"],
            "dataset_processed_input_data_ids_invalid",
        ),
    ),
)
def test_write_dataset_rejects_wrong_kind_processed_spec_before_io(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    root = make_dataset_artifact_root(tmp_path, data_id="storage_demo")
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    invalid_spec = replace(spec, **{field: value})

    with pytest.raises(ValueError, match=error):
        datasets.write_dataset(
            root,
            make_helper_edges_dataset_frame(market_id="dummy"),
            spec=invalid_spec,
        )

    assert not root.exists()


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


def test_validate_dataset_spec_accepts_only_research_immutable_semantic_mappings(
) -> None:
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
    with pytest.raises(ValueError, match="dataset_column_roles_invalid"):
        validate_dataset_spec(processed)


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
    manifest["logical_dtype_hash"] = canonical_json_hash(
        manifest["logical_dtypes"]
    )
    manifest["feature_transforms"][0]["winsorize_quantiles"] = replacement

    with pytest.raises(ValueError, match="dataset_manifest_spec_mismatch"):
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


def test_exact_key_index_rejects_sub_page_cache_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid_exact_key_sqlite_cache_budget"):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
            scratch_parent=tmp_path,
            sqlite_cache_bytes=512,
        )


def test_exact_key_index_sqlite_page_fits_declared_cache_budget(
    tmp_path: Path,
) -> None:
    cache_bytes = 1024

    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        sqlite_cache_bytes=cache_bytes,
    ) as index:
        assert index._connection is not None
        page_size = int(
            index._connection.execute("PRAGMA page_size").fetchone()[0]
        )

    assert page_size <= cache_bytes


@pytest.mark.parametrize(
    "cache_bytes",
    (True, 1024.0, 8 * 1024 * 1024 + 1),
)
def test_exact_key_index_rejects_nonexact_or_oversized_cache_budget(
    tmp_path: Path,
    cache_bytes: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_exact_key_sqlite_cache_budget"):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
            scratch_parent=tmp_path,
            sqlite_cache_bytes=cache_bytes,  # type: ignore[arg-type]
        )


def test_exact_key_index_stores_primary_keys_once_and_category_values_only(
    tmp_path: Path,
) -> None:
    with ExactPrimaryKeyIndex(
        primary_key=("asset_id",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
        sqlite_cache_bytes=8 * 1024 * 1024,
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
            index.identity_payload_values(
                namespace="logical-category:asset_id"
            )
        ) == [b"value-a", b"value-b"]


def test_exact_key_index_cleans_scratch_when_open_is_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        key_tracking.sqlite3,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        ExactPrimaryKeyIndex(
            primary_key=("asset_id",),
            timestamp_columns=(),
            scratch_parent=tmp_path,
        ).__enter__()

    assert tuple(tmp_path.iterdir()) == ()


def test_exact_key_index_preserves_open_failure_when_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            scratch_parent=tmp_path,
        ).__enter__()

    assert tuple(tmp_path.iterdir()) == ()


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
        "fingerprint_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(str(tmp_path))),
    )

    comparison = datasets.compare_datasets(left, right)

    assert comparison.status == "INSPECTION_FAILED"
    assert comparison.summary == {"error": "dataset_inspection_failed"}


def test_write_stats_fail_closed_paths(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as tracker:
        stats = _WriteStats(
            spec=spec,
            data_root=Path("data"),
            key_tracker=tracker,
        )

        with pytest.raises(ValueError, match="unregistered_output_columns"):
            stats.declared_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    extra=1
                ),
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
        scratch_parent=tmp_path,
        key_schema=schema,
    ) as index:
        assert index.add_table(pa.table({"key": [first]})) == 0
        assert index.add_table(pa.table({"key": [second]})) == 0


def test_exact_index_preserves_unclaimed_key_scratch_directory(
    tmp_path: Path,
) -> None:
    unclaimed = tmp_path / ".k-user-cache"
    unclaimed.mkdir()
    sentinel = unclaimed / "sentinel.txt"
    sentinel.write_text("unclaimed\n", encoding="utf-8")

    with ExactPrimaryKeyIndex(
        primary_key=("key",),
        timestamp_columns=(),
        scratch_parent=tmp_path,
    ):
        pass

    assert sentinel.read_text(encoding="utf-8") == "unclaimed\n"


def test_write_stats_rejects_invalid_non_null_timestamp(tmp_path: Path) -> None:
    spec = make_processed_dataset_spec("storage_demo", market_id="dummy")
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=("timestamp",),
        scratch_parent=tmp_path,
    ) as tracker:
        with pytest.raises(ValueError, match="invalid_dataset_timestamp_value"):
            _WriteStats(
                spec=spec,
                data_root=Path("data"),
                key_tracker=tracker,
            ).prepare_frame(
                make_helper_edges_dataset_frame(market_id="dummy").assign(
                    timestamp=["2024-01-02", "bad-date"]
                )
            )


def test_dataset_compare_and_filter_fail_closed_paths(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"

    assert datasets.compare_datasets(left, right).status == "MISSING_LEFT"
    left.mkdir()
    assert datasets.compare_datasets(left, right).status == "MISSING_RIGHT"

    spec_payload = {
        "primary_key": ["market_id", "asset_id"],
        "column_roles": {
            "market_id": ColumnRole.KEY.value,
            "asset_id": ColumnRole.KEY.value,
        },
    }
    with pytest.raises(ValueError, match="time_window_not_supported"):
        filter_frame_by_time_window(
            make_helper_edges_dataset_frame(market_id="dummy"),
            spec=spec_payload,
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    with pytest.raises(ValueError, match="time_window_timestamp_column_not_selected"):
        filter_frame_by_time_window(
            make_helper_edges_dataset_frame(market_id="dummy").drop(
                columns=["timestamp"]
            ),
            spec=data_spec_payload(
                make_processed_dataset_spec("storage_demo", market_id="dummy")
            ),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )
    assert filter_frame_by_time_window(
        make_helper_edges_dataset_frame(market_id="dummy"),
        spec=data_spec_payload(
            make_processed_dataset_spec("storage_demo", market_id="dummy")
        ),
        time_window=TimeWindow("2025-01-03", "2025-01-31"),
    ).empty

    with pytest.raises(ValueError, match="invalid_time_window_timestamp_value"):
        filter_frame_by_time_window(
            make_helper_edges_dataset_frame(market_id="dummy").assign(
                timestamp=["2024-01-02", "bad-date"]
            ),
            spec=data_spec_payload(
                make_processed_dataset_spec("storage_demo", market_id="dummy")
            ),
            time_window=TimeWindow("2024-01-01", "2024-01-31"),
        )


def test_dataset_partition_and_manifest_shape_helpers(tmp_path: Path) -> None:
    frame = make_helper_edges_dataset_frame(market_id="dummy")

    assert list(
        partition_frames(
            frame.iloc[0:0],
            timestamp_column="timestamp",
            partition_columns=("year",),
        )
    ) == []
    assert list(
        partition_frames(
            frame,
            timestamp_column="timestamp",
            partition_columns=(),
        )
    )[0][0] == ""
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

    physical_schema = physical_schema_from_schema(
        pa.schema([pa.field("market_id", pa.string(), nullable=False)])
    )
    manifest = {
        "columns": ["market_id"],
        "physical_schema": physical_schema,
        "schema_hash": canonical_json_hash(physical_schema),
        "file_count": 1,
        "row_count": 2,
        "partition_row_counts": {"year=2024": 2},
    }
    assert validate_manifest_runtime_metadata_shape(manifest)[0] == 1
    for bad_manifest, error in (
        ({**manifest, "columns": []}, "manifest_columns_invalid"),
        ({**manifest, "schema_hash": ""}, "manifest_schema_hash_invalid"),
        ({**manifest, "schema_hash": "A" * 64}, "manifest_schema_hash_invalid"),
        ({**manifest, "file_count": "bad"}, "manifest_counts_invalid"),
        ({**manifest, "file_count": True}, "manifest_counts_invalid"),
        ({**manifest, "row_count": 2.0}, "manifest_counts_invalid"),
        ({**manifest, "row_count": -1}, "manifest_counts_invalid"),
        (
            {**manifest, "partition_row_counts": {}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"": 1}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"year=2024": "bad"}},
            "manifest_partition_row_counts_invalid",
        ),
        (
            {**manifest, "partition_row_counts": {"year=2024": -1}},
            "manifest_partition_row_counts_invalid",
        ),
        ({**manifest, "row_count": 3}, "manifest_row_count_mismatch"),
    ):
        with pytest.raises(ValueError, match=error):
            validate_manifest_runtime_metadata_shape(bad_manifest)


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
                                pa.list_(
                                    pa.field("item", pa.int64(), nullable=True)
                                ),
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
    assert len(
        {
            canonical_json_hash(identity),
            canonical_json_hash(top_nullable),
            canonical_json_hash(child_nullable),
            canonical_json_hash(dictionary_changed),
        }
    ) == 4


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
    assert candidate_data_files(
        data_root,
        spec={**spec, "partition_columns": []},
    ) == []
    with pytest.raises(ValueError, match="unsupported_partitioning"):
        candidate_data_files(
            data_root,
            spec={**spec, "partition_columns": ["month"]},
        )
def test_candidate_data_files_orders_numeric_part_indices(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for index in (1001, 999, 1000, 998):
        pq.write_table(
            pa.table({"value": [index]}),
            data_root / f"part-{index}.parquet",
        )

    paths = candidate_data_files(
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
def test_candidate_data_files_rejects_invalid_or_ambiguous_parts(
    tmp_path: Path,
    names: tuple[str, ...],
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    for name in names:
        pq.write_table(pa.table({"value": [1]}), data_root / name)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(
            data_root,
            spec={"partition_columns": []},
        )


def test_candidate_data_files_rejects_duplicate_numeric_years(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    for year_name in ("year=1", "year=01"):
        year_root = data_root / year_name
        year_root.mkdir(parents=True)
        pq.write_table(pa.table({"value": [1]}), year_root / "part-000.parquet")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(
            data_root,
            spec={"partition_columns": ["year"]},
        )


def test_candidate_data_files_rejects_resolved_path_outside_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.parquet"
    pq.write_table(pa.table({"value": [1]}), external)

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        _require_only_expected_parquet_paths(data_root, [external])


def test_candidate_data_files_rejects_symlinked_part_alias(tmp_path: Path) -> None:
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
        _require_only_expected_parquet_paths(data_root, [part, alias])


def test_candidate_data_files_rejects_symlinked_partition_alias(
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
        candidate_data_files(data_root, spec={"partition_columns": ["year"]})


@pytest.mark.skipif(not hasattr(Path, "is_junction"), reason="junction API required")
def test_candidate_data_files_rejects_junction_data_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    pq.write_table(pa.table({"value": [1]}), external / "part-000.parquet")
    data_root = tmp_path / "data"
    created = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(data_root), str(external)],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        pytest.skip("junction_not_available")

    assert data_root.is_junction()
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})


def test_candidate_data_files_rejects_hardlinked_part_alias(tmp_path: Path) -> None:
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
        candidate_data_files(data_root, spec={"partition_columns": []})


def test_candidate_data_files_rejects_single_external_hardlink(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.parquet"
    part = data_root / "part-000.parquet"
    pq.write_table(pa.table({"value": [1]}), external)
    try:
        os.link(external, part)
    except OSError:
        pytest.skip("hardlink_not_available")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})


def test_candidate_data_files_rejects_case_variant_parquet_suffix(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    pq.write_table(pa.table({"value": [1]}), data_root / "part-000.parquet")
    pq.write_table(pa.table({"value": [2]}), data_root / "part-001.PARQUET")

    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        candidate_data_files(data_root, spec={"partition_columns": []})
