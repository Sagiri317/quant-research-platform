from __future__ import annotations

import os
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.comparison as dataset_comparison
from quant_research_platform.contracts.data import (
    InterimDataSpec,
)
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import datasets
from quant_research_platform.storage.datasets.inventory import DatasetSessionInventory
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
)
from quant_research_platform.storage.datasets.reading import (
    projection_columns_for_window,
)
from quant_research_platform.storage.datasets.schema import (
    data_spec_payload,
    physical_schema_from_schema,
)
from quant_research_platform.storage.datasets.writing import _WriteStats
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import native_filesystem_path
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.storage.low_level_fixtures import (
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


def test_dataset_timestamp_and_candidate_file_helpers(tmp_path: Path) -> None:
    spec = data_spec_payload(
        make_processed_dataset_spec("storage_demo", market_id="dummy")
    )
    assert projection_columns_for_window(
        ("market_id",), spec=spec, time_window=TimeWindow("2024-01-01", "2024-01-31")
    ) == ("market_id", "timestamp")
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
        identities.append(payload)
    assert identities[0] != identities[1]
