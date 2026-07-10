"""Dataset schema, metadata, and payload validation helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.storage.manifests import require_qrp_json_identity

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
        "label_columns",
        "filter_columns",
        "context_columns",
        "key_coverage",
        "row_count",
        "file_count",
        "file_size_bytes",
        "columns",
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
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except TypeError as exc:
        raise ValueError("dataset_instance_metadata_not_json_serializable") from exc
    except ValueError as exc:
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
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise ValueError("invalid_input_dataset_identities_metadata")
    return value.lower()


def dataset_identity_payload(
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    data_id = spec.get("data_id")
    if not isinstance(data_id, str) or not data_id:
        raise ValueError("invalid_dataset_identity")
    return {
        "data_id": data_id,
        "spec_hash": sha256_json(spec),
        "manifest_hash": sha256_json(manifest),
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
    if spec.schema_version != "0.1":
        raise ValueError("unsupported_schema_version")
    if not isinstance(spec.grain, DataGrain):
        raise ValueError("invalid_grain")
    if not spec.primary_key:
        raise ValueError("missing_primary_key")
    if spec.partition_columns not in {("year",), ()}:
        raise ValueError("unsupported_partitioning")
    for column, role in spec.column_roles.items():
        if not isinstance(role, ColumnRole):
            raise ValueError(f"invalid_column_role: {column}")
    invalid_key_roles = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) not in {ColumnRole.KEY, ColumnRole.TIMESTAMP}
    ]
    if invalid_key_roles:
        raise ValueError(f"invalid_primary_key_roles: {', '.join(invalid_key_roles)}")
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles[column] is ColumnRole.TIMESTAMP
    ]
    if spec.partition_columns == ():
        if timestamp_columns:
            raise ValueError("unpartitioned_data_must_not_use_timestamp_key")
        return None
    if len(timestamp_columns) != 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0]


def validated_schema(paths: list[Path]) -> tuple[tuple[str, str], ...]:
    if not paths:
        raise ValueError("dataset_data_dir_missing")
    schemas = [canonical_schema(path) for path in paths]
    first = schemas[0]
    if any(schema != first for schema in schemas[1:]):
        raise ValueError("dataset_schema_mismatch")
    return first


def canonical_schema(path: Path) -> tuple[tuple[str, str], ...]:
    schema = pq.ParquetFile(path).schema_arrow
    return canonical_schema_from_schema(schema)


def canonical_schema_from_file(
    parquet_file: pq.ParquetFile,
) -> tuple[tuple[str, str], ...]:
    return canonical_schema_from_schema(parquet_file.schema_arrow)


def canonical_schema_from_schema(schema: Any) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.type)) for field in schema)


def data_spec_payload(spec: DataSpec) -> dict[str, Any]:
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
        payload.update({"source_id": str(spec.source_id)})
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {
                "market_id": str(spec.market_id),
                "kind": spec.kind.value,
                "source_input_data_ids": list(spec.source_input_data_ids),
                "processed_input_data_ids": list(spec.processed_input_data_ids),
            }
        )
    if isinstance(spec, ResearchDatasetSpec):
        payload.update(
            {
                "market_id": str(spec.market_id),
                "input_data_ids": list(spec.input_data_ids),
                "base_input_data_id": spec.base_input_data_id,
                "factor_columns": list(spec.factor_columns),
                "feature_columns": list(spec.feature_columns),
                "label_columns": list(spec.label_columns),
                "filter_columns": list(spec.filter_columns),
                "context_columns": list(spec.context_columns),
                "key_coverage": dict(spec.key_coverage),
            }
        )
    return payload


def require_qrp_json_schema_id(
    payload: object,
    *,
    expected: str,
    error: str,
) -> None:
    require_qrp_json_identity(
        payload,
        expected_schema_id=expected,
        error=error,
    )


def date_string(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
