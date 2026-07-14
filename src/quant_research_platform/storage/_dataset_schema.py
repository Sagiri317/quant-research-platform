"""Dataset schema, metadata, and payload validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts._json import (
    canonical_identity_bytes,
    canonical_json_hash,
    render_qrp_json_object,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import (
    FEATURE_TRANSFORMS,
    FILLNA_POLICIES,
    FeatureTransformSpec,
    validated_winsorize_quantiles,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.policy.check_status import VALID_STATUS_TOKENS

_DATASET_INSTANCE_METADATA_ALLOWED_KEYS = frozenset({"input_dataset_identities"})
_DATASET_MANIFEST_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "data_id",
        "frequency",
        "grain",
        "primary_key",
        "partition_columns",
        "required_columns",
        "column_roles",
        "source_id",
        "market_id",
        "kind",
        "source_input_data_ids",
        "processed_input_data_ids",
        "input_data_ids",
        "base_input_data_id",
        "factor_columns",
        "feature_columns",
        "feature_transforms",
        "label_columns",
        "filter_columns",
        "context_columns",
        "key_coverage",
        "row_count",
        "file_count",
        "file_size_bytes",
        "columns",
        "physical_schema",
        "schema_hash",
        "partition_row_counts",
        "partition_count",
        "partitions",
        "date_min",
        "date_max",
        "null_count_summary",
        "non_finite_count_summary",
        "duplicate_primary_key_count",
        "factor_nan_profile",
        "label_nan_profile",
        "validation_status",
    }
)
_DATASET_INPUT_IDENTITY_KEYS = frozenset(
    {"data_id", "manifest_hash", "spec_hash"}
)
_DATASET_SPEC_COMMON_KEYS = frozenset(
    {
        "schema_id",
        "data_id",
        "frequency",
        "grain",
        "primary_key",
        "partition_columns",
        "required_columns",
        "column_roles",
    }
)
_DATASET_SPEC_BASE_KEYS = _DATASET_SPEC_COMMON_KEYS
_DATASET_SPEC_INTERIM_KEYS = _DATASET_SPEC_COMMON_KEYS | {"source_id"}
_DATASET_SPEC_PROCESSED_KEYS = _DATASET_SPEC_COMMON_KEYS | {
    "market_id",
    "kind",
    "source_input_data_ids",
    "processed_input_data_ids",
}
_DATASET_SPEC_RESEARCH_KEYS = _DATASET_SPEC_COMMON_KEYS | {
    "market_id",
    "input_data_ids",
    "base_input_data_id",
    "factor_columns",
    "feature_columns",
    "feature_transforms",
    "label_columns",
    "filter_columns",
    "context_columns",
    "key_coverage",
}
_DATASET_SPEC_VARIANT_KEYS = (
    _DATASET_SPEC_BASE_KEYS,
    _DATASET_SPEC_INTERIM_KEYS,
    _DATASET_SPEC_PROCESSED_KEYS,
    _DATASET_SPEC_RESEARCH_KEYS,
)
_DATASET_MANIFEST_RUNTIME_KEYS = frozenset(
    {
        "row_count",
        "file_count",
        "file_size_bytes",
        "columns",
        "physical_schema",
        "schema_hash",
        "partition_row_counts",
        "partition_count",
        "partitions",
        "date_min",
        "date_max",
        "null_count_summary",
        "non_finite_count_summary",
        "duplicate_primary_key_count",
        "factor_nan_profile",
        "label_nan_profile",
        "validation_status",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


def merge_dataset_instance_metadata(
    payload: Mapping[str, Any],
    instance_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(payload)
    if instance_metadata is None:
        return result
    if not isinstance(instance_metadata, Mapping):
        raise TypeError("invalid_dataset_instance_metadata")
    metadata: dict[str, Any] = {}
    for key, value in instance_metadata.items():
        if not isinstance(key, str):
            raise ValueError("unsupported_dataset_instance_metadata_key")
        if key in _DATASET_MANIFEST_RESERVED_KEYS or key in result:
            raise ValueError(f"reserved_dataset_instance_metadata_key: {key}")
        if key not in _DATASET_INSTANCE_METADATA_ALLOWED_KEYS:
            raise ValueError(f"unsupported_dataset_instance_metadata_key: {key}")
        if key == "input_dataset_identities":
            metadata[key] = _validate_input_dataset_identities_metadata(value)
    _require_json_serializable(metadata)
    result.update(metadata)
    return result


def _require_json_serializable(value: Mapping[str, Any]) -> None:
    try:
        render_qrp_json_object(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_instance_metadata_not_json_serializable") from exc


def _validate_input_dataset_identities_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("invalid_input_dataset_identities_metadata")
    identities: dict[str, Any] = {}
    for data_id, identity in value.items():
        if not isinstance(data_id, str) or not data_id:
            raise ValueError("invalid_input_dataset_identities_metadata")
        identities[data_id] = _validate_input_dataset_identity_payload(
            identity,
            expected_data_id=data_id,
        )
    return identities


def _validate_input_dataset_identity_payload(
    value: Any,
    *,
    expected_data_id: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _DATASET_INPUT_IDENTITY_KEYS:
        raise ValueError("invalid_input_dataset_identities_metadata")
    data_id = value["data_id"]
    if data_id != expected_data_id:
        raise ValueError("invalid_input_dataset_identities_metadata")
    return {
        "data_id": expected_data_id,
        "spec_hash": _require_identity_hash(value["spec_hash"]),
        "manifest_hash": _require_identity_hash(value["manifest_hash"]),
    }


def _require_identity_hash(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError("invalid_input_dataset_identities_metadata")
    return value


def dataset_identity_payload(
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    data_id = spec.get("data_id")
    if not isinstance(data_id, str) or not data_id:
        raise ValueError("invalid_dataset_identity")
    return {
        "data_id": data_id,
        "spec_hash": canonical_json_hash(spec),
        "manifest_hash": canonical_json_hash(manifest),
    }


def validate_dataset_manifest_instance_metadata(
    manifest: Mapping[str, Any],
) -> None:
    unknown = set(manifest).difference(
        _DATASET_MANIFEST_RESERVED_KEYS,
        _DATASET_INSTANCE_METADATA_ALLOWED_KEYS,
        {"schema_id"},
    )
    if unknown:
        raise ValueError(
            "unsupported_dataset_instance_metadata_key: " + ", ".join(sorted(unknown))
        )
    if "input_dataset_identities" in manifest:
        _validate_input_dataset_identities_metadata(
            manifest["input_dataset_identities"]
        )


def validate_dataset_spec(spec: DataSpec) -> str | None:
    if type(spec) not in {
        DataSpec,
        InterimDataSpec,
        ProcessedDataSpec,
        ResearchDatasetSpec,
    }:
        raise ValueError("dataset_spec_type_invalid")
    if type(spec.schema_version) is not str or spec.schema_version != "0.1":
        raise ValueError("unsupported_schema_version")
    _require_runtime_string(spec.data_id, "dataset_data_id_invalid")
    if type(spec.frequency) is not DataFrequency:
        raise ValueError("dataset_frequency_invalid")
    if type(spec.grain) is not DataGrain:
        raise ValueError("dataset_grain_invalid")
    primary_key = _require_runtime_string_tuple(
        spec.primary_key,
        "dataset_primary_key_invalid",
        allow_empty=False,
    )
    _require_runtime_string_tuple(
        spec.partition_columns,
        "dataset_partition_columns_invalid",
        allow_empty=True,
    )
    column_roles_are_valid_mapping = (
        type(spec.column_roles) is MappingProxyType
        if type(spec) is ResearchDatasetSpec
        else type(spec.column_roles) is dict
    )
    if not column_roles_are_valid_mapping or not spec.column_roles:
        raise ValueError("dataset_column_roles_invalid")
    if any(
        type(column) is not str
        or not column
        or type(role) is not ColumnRole
        for column, role in spec.column_roles.items()
    ):
        raise ValueError("dataset_column_roles_invalid")
    if (
        type(spec.required_columns) is not tuple
        or spec.required_columns != tuple(spec.column_roles)
    ):
        raise ValueError("dataset_required_columns_invalid")
    if not set(primary_key).issubset(spec.column_roles):
        raise ValueError("dataset_primary_key_invalid")
    if spec.partition_columns not in {("year",), ()}:
        raise ValueError("unsupported_partitioning")
    _validate_dataset_spec_variant(spec)
    invalid_key_roles = [
        column
        for column in primary_key
        if spec.column_roles.get(column) not in {ColumnRole.KEY, ColumnRole.TIMESTAMP}
    ]
    if invalid_key_roles:
        raise ValueError(f"invalid_primary_key_roles: {', '.join(invalid_key_roles)}")
    timestamp_columns = [
        column
        for column in primary_key
        if spec.column_roles[column] is ColumnRole.TIMESTAMP
    ]
    if spec.partition_columns == ():
        if timestamp_columns:
            raise ValueError("unpartitioned_data_must_not_use_timestamp_key")
        return None
    if len(timestamp_columns) != 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0]


def _validate_dataset_spec_variant(spec: DataSpec) -> None:
    if type(spec) is DataSpec:
        return
    if type(spec) is InterimDataSpec:
        _require_runtime_string(spec.source_id, "dataset_source_id_invalid")
        return
    if type(spec) is ProcessedDataSpec:
        _require_runtime_string(spec.market_id, "dataset_market_id_invalid")
        if type(spec.kind) is not ProcessedDataKind:
            raise ValueError("dataset_kind_invalid")
        _require_runtime_string_tuple(
            spec.source_input_data_ids,
            "dataset_source_input_data_ids_invalid",
            allow_empty=True,
        )
        _require_runtime_string_tuple(
            spec.processed_input_data_ids,
            "dataset_processed_input_data_ids_invalid",
            allow_empty=True,
        )
        return
    if type(spec) is not ResearchDatasetSpec:
        raise ValueError("dataset_spec_type_invalid")
    _require_runtime_string(spec.market_id, "dataset_market_id_invalid")
    input_data_ids = _require_runtime_string_tuple(
        spec.input_data_ids,
        "dataset_input_data_ids_invalid",
        allow_empty=False,
    )
    _require_runtime_string(
        spec.base_input_data_id,
        "dataset_base_input_data_id_invalid",
    )
    for value, error, allow_empty in (
        (spec.factor_columns, "dataset_factor_columns_invalid", True),
        (spec.feature_columns, "dataset_feature_columns_invalid", True),
        (spec.label_columns, "dataset_label_columns_invalid", False),
        (spec.filter_columns, "dataset_filter_columns_invalid", True),
        (spec.context_columns, "dataset_context_columns_invalid", True),
    ):
        _require_runtime_string_tuple(value, error, allow_empty=allow_empty)
    if type(spec.feature_transforms) is not tuple:
        raise ValueError("dataset_feature_transforms_invalid")
    for transform in spec.feature_transforms:
        _validate_runtime_feature_transform(transform)
    if (
        type(spec.key_coverage) is not MappingProxyType
        or set(spec.key_coverage) != set(input_data_ids)
        or any(
            type(data_id) is not str
            or not data_id
            or type(policy) is not str
            or policy not in {"strict", "sparse"}
            for data_id, policy in spec.key_coverage.items()
        )
    ):
        raise ValueError("dataset_key_coverage_invalid")
    try:
        ResearchDatasetSpec(
            data_id=spec.data_id,
            frequency=spec.frequency,
            grain=spec.grain,
            column_roles=dict(spec.column_roles),
            primary_key=tuple(spec.primary_key),
            partition_columns=tuple(spec.partition_columns),
            schema_version=spec.schema_version,
            market_id=spec.market_id,
            input_data_ids=tuple(spec.input_data_ids),
            base_input_data_id=spec.base_input_data_id,
            factor_columns=tuple(spec.factor_columns),
            feature_columns=tuple(spec.feature_columns),
            feature_transforms=tuple(spec.feature_transforms),
            label_columns=tuple(spec.label_columns),
            filter_columns=tuple(spec.filter_columns),
            context_columns=tuple(spec.context_columns),
            key_coverage=dict(spec.key_coverage),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_research_spec_invalid") from exc


def _validate_runtime_feature_transform(value: object) -> None:
    if type(value) is not FeatureTransformSpec:
        raise ValueError("dataset_feature_transforms_invalid")
    if type(value.contract_version) is not str or value.contract_version != "0.1":
        raise ValueError("dataset_feature_transforms_invalid")
    _require_runtime_string(value.transform_id, "dataset_feature_transforms_invalid")
    _require_runtime_string_tuple(
        value.source_columns,
        "dataset_feature_transforms_invalid",
        allow_empty=False,
    )
    _require_runtime_string_tuple(
        value.output_columns,
        "dataset_feature_transforms_invalid",
        allow_empty=False,
    )
    if type(value.transform) is not str or value.transform not in FEATURE_TRANSFORMS:
        raise ValueError("dataset_feature_transforms_invalid")
    if (
        type(value.fillna_policy) is not str
        or value.fillna_policy not in FILLNA_POLICIES
    ):
        raise ValueError("dataset_feature_transforms_invalid")
    try:
        quantiles = (
            None
            if value.winsorize_quantiles is None
            else validated_winsorize_quantiles(value.winsorize_quantiles)
        )
        FeatureTransformSpec(
            transform_id=value.transform_id,
            source_columns=tuple(value.source_columns),
            transform=value.transform,
            output_columns=tuple(value.output_columns),
            winsorize_quantiles=quantiles,
            fillna_policy=value.fillna_policy,
            contract_version=value.contract_version,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_feature_transforms_invalid") from exc


def _require_runtime_string(value: object, error: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(error)
    return value


def _require_runtime_string_tuple(
    value: object,
    error: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if (
        type(value) is not tuple
        or (not value and not allow_empty)
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(error)
    return value


def validated_physical_schema(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("dataset_data_dir_missing")
    first: list[dict[str, Any]] | None = None
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        try:
            current = physical_schema_from_file(parquet_file)
            if first is None:
                first = current
            elif current != first:
                raise ValueError("dataset_schema_mismatch")
        finally:
            parquet_file.close()
    if first is None:
        raise AssertionError("validated_physical_schema_requires_paths")
    return first


def physical_schema_from_file(
    parquet_file: pq.ParquetFile,
) -> list[dict[str, Any]]:
    return physical_schema_from_schema(parquet_file.schema_arrow)


def physical_schema_from_schema(schema: pa.Schema) -> list[dict[str, Any]]:
    if not isinstance(schema, pa.Schema):
        raise TypeError("dataset_arrow_schema_required")
    payload = [_physical_field_payload(field) for field in schema]
    validate_physical_schema_payload(payload)
    return payload


def physical_schema_columns(schema: object) -> tuple[str, ...]:
    validate_physical_schema_payload(schema)
    return tuple(field["name"] for field in schema)


def _physical_field_payload(field: pa.Field) -> dict[str, Any]:
    return {
        "name": field.name,
        "nullable": field.nullable,
        "type": _physical_type_payload(field.type),
    }


def _physical_type_payload(data_type: pa.DataType) -> dict[str, Any]:
    if pa.types.is_boolean(data_type):
        return {"kind": "bool"}
    if pa.types.is_integer(data_type):
        return {
            "kind": "integer",
            "signed": pa.types.is_signed_integer(data_type),
            "bit_width": data_type.bit_width,
        }
    if pa.types.is_floating(data_type):
        return {"kind": "float", "bit_width": data_type.bit_width}
    if pa.types.is_decimal(data_type):
        return {
            "kind": "decimal",
            "bit_width": data_type.bit_width,
            "precision": data_type.precision,
            "scale": data_type.scale,
        }
    if pa.types.is_date32(data_type):
        return {"kind": "date", "unit": "day"}
    if pa.types.is_date64(data_type):
        return {"kind": "date", "unit": "millisecond"}
    if pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        return {"kind": "time", "unit": data_type.unit}
    if pa.types.is_timestamp(data_type):
        return {
            "kind": "timestamp",
            "unit": data_type.unit,
            "timezone": data_type.tz,
        }
    if pa.types.is_duration(data_type):
        return {"kind": "duration", "unit": data_type.unit}
    if pa.types.is_string(data_type):
        return {"kind": "string"}
    if pa.types.is_large_string(data_type):
        return {"kind": "large_string"}
    if pa.types.is_binary(data_type):
        return {"kind": "binary"}
    if pa.types.is_large_binary(data_type):
        return {"kind": "large_binary"}
    if pa.types.is_fixed_size_binary(data_type):
        return {"kind": "fixed_size_binary", "byte_width": data_type.byte_width}
    if pa.types.is_struct(data_type):
        return {
            "kind": "struct",
            "fields": [_physical_field_payload(field) for field in data_type],
        }
    if pa.types.is_list(data_type):
        return {
            "kind": "list",
            "value_field": _physical_field_payload(data_type.value_field),
        }
    if pa.types.is_large_list(data_type):
        return {
            "kind": "large_list",
            "value_field": _physical_field_payload(data_type.value_field),
        }
    if pa.types.is_fixed_size_list(data_type):
        return {
            "kind": "fixed_size_list",
            "list_size": data_type.list_size,
            "value_field": _physical_field_payload(data_type.value_field),
        }
    if pa.types.is_map(data_type):
        return {
            "kind": "map",
            "keys_sorted": data_type.keys_sorted,
            "key_field": _physical_field_payload(data_type.key_field),
            "item_field": _physical_field_payload(data_type.item_field),
        }
    if pa.types.is_dictionary(data_type):
        return {
            "kind": "dictionary",
            "index_type": _physical_type_payload(data_type.index_type),
            "value_type": _physical_type_payload(data_type.value_type),
            "ordered": data_type.ordered,
        }
    raise ValueError("unsupported_dataset_arrow_type")


def data_spec_payload(spec: DataSpec) -> dict[str, Any]:
    validate_dataset_spec(spec)
    payload: dict[str, Any] = {
        "schema_id": "qrp-dataset-spec",
        "data_id": spec.data_id,
        "frequency": spec.frequency.value,
        "grain": spec.grain.value,
        "primary_key": list(spec.primary_key),
        "partition_columns": list(spec.partition_columns),
        "required_columns": list(spec.required_columns),
        "column_roles": {
            column: role.value for column, role in sorted(spec.column_roles.items())
        },
    }
    if isinstance(spec, InterimDataSpec):
        payload.update({"source_id": spec.source_id})
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {
                "market_id": spec.market_id,
                "kind": spec.kind.value,
                "source_input_data_ids": list(spec.source_input_data_ids),
                "processed_input_data_ids": list(spec.processed_input_data_ids),
            }
        )
    if isinstance(spec, ResearchDatasetSpec):
        payload.update(
            {
                "market_id": spec.market_id,
                "input_data_ids": list(spec.input_data_ids),
                "base_input_data_id": spec.base_input_data_id,
                "factor_columns": list(spec.factor_columns),
                "feature_columns": list(spec.feature_columns),
                "feature_transforms": [
                    _feature_transform_payload(transform)
                    for transform in spec.feature_transforms
                ],
                "label_columns": list(spec.label_columns),
                "filter_columns": list(spec.filter_columns),
                "context_columns": list(spec.context_columns),
                "key_coverage": dict(spec.key_coverage),
            }
        )
    return payload


def _feature_transform_payload(transform: FeatureTransformSpec) -> dict[str, Any]:
    return {
        "transform_id": transform.transform_id,
        "source_columns": list(transform.source_columns),
        "transform": transform.transform,
        "output_columns": list(transform.output_columns),
        "winsorize_quantiles": (
            None
            if transform.winsorize_quantiles is None
            else list(transform.winsorize_quantiles)
        ),
        "fillna_policy": transform.fillna_policy,
    }


def date_string(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def validate_dataset_spec_payload(spec: Mapping[str, Any]) -> None:
    if not isinstance(spec, Mapping) or spec.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    keys = frozenset(spec)
    if keys not in _DATASET_SPEC_VARIANT_KEYS:
        raise ValueError("dataset_spec_fields_invalid")
    data_id = _require_nonempty_string(spec.get("data_id"), "dataset_data_id_invalid")
    frequency = _require_enum_string(
        spec.get("frequency"),
        {item.value for item in DataFrequency},
        "dataset_frequency_invalid",
    )
    grain = _require_enum_string(
        spec.get("grain"),
        {item.value for item in DataGrain},
        "dataset_grain_invalid",
    )
    primary_key = _require_string_list(
        spec.get("primary_key"),
        "dataset_primary_key_invalid",
        allow_empty=False,
    )
    partition_columns = _require_string_list(
        spec.get("partition_columns"),
        "dataset_partition_columns_invalid",
        allow_empty=True,
    )
    if partition_columns not in ([], ["year"]):
        raise ValueError("dataset_partition_columns_invalid")
    required_columns = _require_string_list(
        spec.get("required_columns"),
        "dataset_required_columns_invalid",
        allow_empty=False,
    )
    column_roles = spec.get("column_roles")
    if not isinstance(column_roles, dict) or set(column_roles) != set(required_columns):
        raise ValueError("dataset_column_roles_invalid")
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or value not in {role.value for role in ColumnRole}
        for key, value in column_roles.items()
    ):
        raise ValueError("dataset_column_roles_invalid")
    if not set(primary_key).issubset(required_columns):
        raise ValueError("dataset_primary_key_invalid")
    common_kwargs: dict[str, Any] = {
        "data_id": data_id,
        "frequency": DataFrequency(frequency),
        "grain": DataGrain(grain),
        "column_roles": {
            column: ColumnRole(column_roles[column]) for column in required_columns
        },
        "primary_key": tuple(primary_key),
        "partition_columns": tuple(partition_columns),
    }

    if keys == _DATASET_SPEC_BASE_KEYS:
        validate_dataset_spec(DataSpec(**common_kwargs))
        return
    if keys == _DATASET_SPEC_INTERIM_KEYS:
        source_id = _require_nonempty_string(
            spec.get("source_id"), "dataset_source_id_invalid"
        )
        validate_dataset_spec(
            InterimDataSpec(**common_kwargs, source_id=SourceId(source_id))
        )
        return
    if keys == _DATASET_SPEC_PROCESSED_KEYS:
        market_id = _require_nonempty_string(
            spec.get("market_id"), "dataset_market_id_invalid"
        )
        kind = _require_enum_string(
            spec.get("kind"),
            {item.value for item in ProcessedDataKind},
            "dataset_kind_invalid",
        )
        source_input_data_ids = _require_string_list(
            spec.get("source_input_data_ids"),
            "dataset_source_input_data_ids_invalid",
            allow_empty=True,
        )
        processed_input_data_ids = _require_string_list(
            spec.get("processed_input_data_ids"),
            "dataset_processed_input_data_ids_invalid",
            allow_empty=True,
        )
        validate_dataset_spec(
            ProcessedDataSpec(
                **common_kwargs,
                market_id=MarketId(market_id),
                kind=ProcessedDataKind(kind),
                source_input_data_ids=tuple(source_input_data_ids),
                processed_input_data_ids=tuple(processed_input_data_ids),
            )
        )
        return

    market_id = _require_nonempty_string(
        spec.get("market_id"), "dataset_market_id_invalid"
    )
    input_data_ids = _require_string_list(
        spec.get("input_data_ids"),
        "dataset_input_data_ids_invalid",
        allow_empty=False,
    )
    base_input_data_id = _require_nonempty_string(
        spec.get("base_input_data_id"), "dataset_base_input_data_id_invalid"
    )
    factor_columns = _require_string_list(
        spec.get("factor_columns"),
        "dataset_factor_columns_invalid",
        allow_empty=True,
    )
    feature_columns = _require_string_list(
        spec.get("feature_columns"),
        "dataset_feature_columns_invalid",
        allow_empty=True,
    )
    label_columns = _require_string_list(
        spec.get("label_columns"),
        "dataset_label_columns_invalid",
        allow_empty=False,
    )
    filter_columns = _require_string_list(
        spec.get("filter_columns"),
        "dataset_filter_columns_invalid",
        allow_empty=True,
    )
    context_columns = _require_string_list(
        spec.get("context_columns"),
        "dataset_context_columns_invalid",
        allow_empty=True,
    )
    feature_transforms = _validate_feature_transform_payloads(
        spec.get("feature_transforms")
    )
    key_coverage = spec.get("key_coverage")
    if not isinstance(key_coverage, dict) or set(key_coverage) != set(input_data_ids):
        raise ValueError("dataset_key_coverage_invalid")
    for value in key_coverage.values():
        _require_enum_string(
            value,
            {"strict", "sparse"},
            "dataset_key_coverage_invalid",
        )
    try:
        research_spec = ResearchDatasetSpec(
            **common_kwargs,
            market_id=market_id,
            input_data_ids=tuple(input_data_ids),
            base_input_data_id=base_input_data_id,
            factor_columns=tuple(factor_columns),
            feature_columns=tuple(feature_columns),
            feature_transforms=tuple(feature_transforms),
            label_columns=tuple(label_columns),
            filter_columns=tuple(filter_columns),
            context_columns=tuple(context_columns),
            key_coverage=dict(key_coverage),
        )
        validate_dataset_spec(research_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_research_spec_invalid") from exc


def validate_dataset_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
) -> None:
    validate_dataset_spec_payload(spec)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_id") != "qrp-dataset-manifest"
    ):
        raise ValueError("dataset_manifest_schema_id_invalid")
    expected_keys = (
        (set(spec) - {"schema_id"})
        | _DATASET_MANIFEST_RUNTIME_KEYS
        | {"schema_id"}
    )
    actual_keys = set(manifest)
    if actual_keys not in (
        expected_keys,
        expected_keys | _DATASET_INSTANCE_METADATA_ALLOWED_KEYS,
    ):
        raise ValueError("dataset_manifest_fields_invalid")
    for key in set(spec) - {"schema_id"}:
        try:
            matches_spec = canonical_identity_bytes(
                manifest.get(key)
            ) == canonical_identity_bytes(spec[key])
        except (TypeError, ValueError):
            matches_spec = False
        if not matches_spec:
            raise ValueError(f"dataset_manifest_spec_mismatch: {key}")

    for key in (
        "row_count",
        "file_count",
        "file_size_bytes",
        "partition_count",
        "duplicate_primary_key_count",
    ):
        _require_nonnegative_int(manifest.get(key), f"manifest_{key}_invalid")
    columns = _require_string_list(
        manifest.get("columns"), "manifest_columns_invalid", allow_empty=False
    )
    if columns != spec["required_columns"]:
        raise ValueError("manifest_columns_mismatch")
    physical_schema = manifest.get("physical_schema")
    validate_physical_schema_payload(physical_schema)
    if list(physical_schema_columns(physical_schema)) != columns:
        raise ValueError("manifest_columns_mismatch")
    schema_hash = _require_hash(
        manifest.get("schema_hash"), "manifest_schema_hash_invalid"
    )
    if canonical_json_hash(physical_schema) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")

    partitions = _require_string_list(
        manifest.get("partitions"), "manifest_partitions_invalid", allow_empty=False
    )
    partition_counts = _require_count_mapping(
        manifest.get("partition_row_counts"),
        allowed_keys=set(partitions),
        error="manifest_partition_row_counts_invalid",
    )
    if set(partition_counts) != set(partitions):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if manifest["partition_count"] != len(partitions):
        raise ValueError("manifest_partition_count_mismatch")
    if sum(partition_counts.values()) != manifest["row_count"]:
        raise ValueError("manifest_row_count_mismatch")

    _validate_date_range(manifest.get("date_min"), manifest.get("date_max"))
    _require_count_mapping(
        manifest.get("null_count_summary"),
        allowed_keys=set(columns),
        error="manifest_null_count_summary_invalid",
    )
    _require_count_mapping(
        manifest.get("non_finite_count_summary"),
        allowed_keys=set(columns),
        error="manifest_non_finite_count_summary_invalid",
    )
    factor_columns = {
        column
        for column, role in spec["column_roles"].items()
        if role == ColumnRole.FACTOR.value
    }
    label_columns = {
        column
        for column, role in spec["column_roles"].items()
        if role == ColumnRole.LABEL.value
    }
    _require_count_mapping(
        manifest.get("factor_nan_profile"),
        allowed_keys=factor_columns,
        error="manifest_factor_nan_profile_invalid",
    )
    _require_count_mapping(
        manifest.get("label_nan_profile"),
        allowed_keys=label_columns,
        error="manifest_label_nan_profile_invalid",
    )
    status = manifest.get("validation_status")
    if not isinstance(status, str) or status not in VALID_STATUS_TOKENS:
        raise ValueError("manifest_validation_status_invalid")
    validate_dataset_manifest_instance_metadata(manifest)


def validate_dataset_primary_key_null_summary(manifest: Mapping[str, Any]) -> None:
    """Require an observed zero null count for every declared primary-key column."""

    primary_key = manifest.get("primary_key")
    null_counts = manifest.get("null_count_summary")
    if (
        not isinstance(primary_key, list)
        or not primary_key
        or any(not isinstance(column, str) or not column for column in primary_key)
        or not isinstance(null_counts, Mapping)
    ):
        raise ValueError("dataset_manifest_not_consumable")
    try:
        primary_key_null_counts = tuple(null_counts[column] for column in primary_key)
    except KeyError:
        raise ValueError("dataset_manifest_not_consumable") from None
    if any(type(count) is not int or count != 0 for count in primary_key_null_counts):
        raise ValueError("dataset_manifest_not_consumable")


def validate_dataset_manifest_consumable(manifest: Mapping[str, Any]) -> None:
    """Reject structurally readable manifests that normal consumers cannot use."""

    non_finite_counts = manifest.get("non_finite_count_summary")
    if (
        manifest.get("validation_status") != "PASS"
        or type(manifest.get("duplicate_primary_key_count")) is not int
        or manifest.get("duplicate_primary_key_count") != 0
        or not isinstance(non_finite_counts, Mapping)
        or any(
            type(count) is not int or count != 0
            for count in non_finite_counts.values()
        )
    ):
        raise ValueError("dataset_manifest_not_consumable")
    validate_dataset_primary_key_null_summary(manifest)


def validate_physical_schema_payload(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("manifest_physical_schema_invalid")
    names: set[str] = set()
    for field in value:
        name = _validate_physical_field_payload(field)
        if name in names:
            raise ValueError("manifest_physical_schema_invalid")
        names.add(name)


def _validate_physical_field_payload(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {"name", "nullable", "type"}:
        raise ValueError("manifest_physical_schema_invalid")
    name = value["name"]
    if not isinstance(name, str) or not name or type(value["nullable"]) is not bool:
        raise ValueError("manifest_physical_schema_invalid")
    _validate_physical_type_payload(value["type"])
    return name


def _validate_physical_type_payload(value: object) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise ValueError("manifest_physical_schema_invalid")
    kind = value["kind"]
    if kind in {"bool", "string", "large_string", "binary", "large_binary"}:
        _require_exact_keys(value, {"kind"})
        return
    if kind == "integer":
        _require_exact_keys(value, {"kind", "signed", "bit_width"})
        if (
            type(value["signed"]) is not bool
            or not _is_exact_int(value["bit_width"])
            or value["bit_width"] not in {8, 16, 32, 64}
        ):
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "float":
        _require_exact_keys(value, {"kind", "bit_width"})
        if (
            not _is_exact_int(value["bit_width"])
            or value["bit_width"] not in {16, 32, 64}
        ):
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "decimal":
        _require_exact_keys(value, {"kind", "bit_width", "precision", "scale"})
        if (
            not _is_exact_int(value["bit_width"])
            or value["bit_width"] not in {128, 256}
            or not _is_exact_int(value["precision"])
            or value["precision"] <= 0
            or value["precision"] > (38 if value["bit_width"] == 128 else 76)
            or not _is_exact_int(value["scale"])
            or value["scale"] < 0
            or value["scale"] > value["precision"]
        ):
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "fixed_size_binary":
        _require_exact_keys(value, {"kind", "byte_width"})
        if not _is_exact_int(value["byte_width"]) or value["byte_width"] <= 0:
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "date":
        _require_exact_keys(value, {"kind", "unit"})
        if value["unit"] != "day":
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "time":
        _require_exact_keys(value, {"kind", "unit"})
        _require_enum_string(
            value["unit"], {"ms", "us", "ns"}, "manifest_physical_schema_invalid"
        )
        return
    if kind == "duration":
        _require_exact_keys(value, {"kind", "unit"})
        _require_enum_string(
            value["unit"],
            {"s", "ms", "us", "ns"},
            "manifest_physical_schema_invalid",
        )
        return
    if kind == "timestamp":
        _require_exact_keys(value, {"kind", "unit", "timezone"})
        _require_enum_string(
            value["unit"], {"ms", "us", "ns"}, "manifest_physical_schema_invalid"
        )
        if not (value["timezone"] is None or isinstance(value["timezone"], str)):
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "struct":
        _require_exact_keys(value, {"kind", "fields"})
        fields = value["fields"]
        if not isinstance(fields, list) or not fields:
            raise ValueError("manifest_physical_schema_invalid")
        names = [_validate_physical_field_payload(field) for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind in {"list", "large_list"}:
        _require_exact_keys(value, {"kind", "value_field"})
        _validate_physical_field_payload(value["value_field"])
        return
    if kind == "fixed_size_list":
        _require_exact_keys(value, {"kind", "list_size", "value_field"})
        if not _is_exact_int(value["list_size"]) or value["list_size"] < 0:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_field_payload(value["value_field"])
        return
    if kind == "map":
        _require_exact_keys(
            value, {"kind", "keys_sorted", "key_field", "item_field"}
        )
        if type(value["keys_sorted"]) is not bool:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_field_payload(value["key_field"])
        if value["key_field"]["nullable"] is not False:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_field_payload(value["item_field"])
        return
    if kind == "dictionary":
        _require_exact_keys(
            value, {"kind", "index_type", "value_type", "ordered"}
        )
        if type(value["ordered"]) is not bool:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_type_payload(value["index_type"])
        _validate_physical_type_payload(value["value_type"])
        if (
            value["index_type"].get("kind") != "integer"
            or value["index_type"].get("signed") is not True
        ):
            raise ValueError("manifest_physical_schema_invalid")
        return
    raise ValueError("manifest_physical_schema_invalid")


def _validate_feature_transform_payloads(value: object) -> list[FeatureTransformSpec]:
    if not isinstance(value, list):
        raise ValueError("dataset_feature_transforms_invalid")
    transforms: list[FeatureTransformSpec] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "transform_id",
            "source_columns",
            "transform",
            "output_columns",
            "winsorize_quantiles",
            "fillna_policy",
        }:
            raise ValueError("dataset_feature_transforms_invalid")
        transform = _require_enum_string(
            item["transform"],
            FEATURE_TRANSFORMS,
            "dataset_feature_transforms_invalid",
        )
        fillna_policy = _require_enum_string(
            item["fillna_policy"],
            FILLNA_POLICIES,
            "dataset_feature_transforms_invalid",
        )
        source_columns = _require_string_list(
            item["source_columns"],
            "dataset_feature_transforms_invalid",
            allow_empty=False,
        )
        output_columns = _require_string_list(
            item["output_columns"],
            "dataset_feature_transforms_invalid",
            allow_empty=False,
        )
        transform_id = _require_nonempty_string(
            item["transform_id"], "dataset_feature_transforms_invalid"
        )
        quantiles = item["winsorize_quantiles"]
        if quantiles is not None:
            if type(quantiles) is not list:
                raise ValueError("dataset_feature_transforms_invalid")
            try:
                quantile_tuple: tuple[float, float] | None = (
                    validated_winsorize_quantiles(tuple(quantiles))
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("dataset_feature_transforms_invalid") from exc
        else:
            quantile_tuple = None
        try:
            transforms.append(
                FeatureTransformSpec(
                    transform_id=transform_id,
                    source_columns=tuple(source_columns),
                    transform=transform,
                    output_columns=tuple(output_columns),
                    winsorize_quantiles=quantile_tuple,
                    fillna_policy=fillna_policy,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset_feature_transforms_invalid") from exc
    return transforms


def _require_nonempty_string(value: object, error: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(error)
    return value


def _require_enum_string(value: object, allowed: set[str], error: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(error)
    return value


def _require_string_list(
    value: object,
    error: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(error)
    return value


def _require_nonnegative_int(value: object, error: str) -> int:
    if not _is_exact_int(value) or value < 0:
        raise ValueError(error)
    return value


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _require_hash(value: object, error: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(error)
    return value


def _require_count_mapping(
    value: object,
    *,
    allowed_keys: set[str],
    error: str,
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(error)
    result: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key not in allowed_keys
            or not _is_exact_int(item)
            or item < 0
        ):
            raise ValueError(error)
        result[key] = item
    return result


def _validate_date_range(date_min: object, date_max: object) -> None:
    if date_min is None and date_max is None:
        return
    if not isinstance(date_min, str) or not isinstance(date_max, str):
        raise ValueError("manifest_date_range_invalid")
    try:
        minimum = date.fromisoformat(date_min)
        maximum = date.fromisoformat(date_max)
    except ValueError as exc:
        raise ValueError("manifest_date_range_invalid") from exc
    if minimum.isoformat() != date_min or maximum.isoformat() != date_max:
        raise ValueError("manifest_date_range_invalid")
    if minimum > maximum:
        raise ValueError("manifest_date_range_invalid")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("manifest_physical_schema_invalid")
