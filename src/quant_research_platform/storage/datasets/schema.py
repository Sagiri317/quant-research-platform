"""Dataset schema, metadata, and payload validation helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

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
    FeatureTransformSpec,
)
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    parse_qrp_json_object,
    qrp_json_object_snapshot,
    render_qrp_json_object,
)
from quant_research_platform.storage.datasets.paths import (
    parse_year_partition_label,
    timestamp_column_from_spec_payload,
)

_DATASET_METADATA_ALLOWED_KEYS = frozenset(
    {"producer_diagnostics", "inputs", "parameters", "source_evidence"}
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
        "column_units",
        "label_availability_columns",
    }
)
_DATASET_SPEC_BASE_KEYS = _DATASET_SPEC_COMMON_KEYS
_DATASET_SPEC_INTERIM_KEYS = _DATASET_SPEC_COMMON_KEYS | {"source_id"}
_DATASET_SPEC_PROCESSED_KEYS = _DATASET_SPEC_COMMON_KEYS | {
    "market_id",
    "kind",
    "source_input_data_ids",
    "optional_source_input_data_ids",
    "processed_input_data_ids",
    "parameters",
}
_DATASET_SPEC_RESEARCH_KEYS = _DATASET_SPEC_COMMON_KEYS | {
    "market_id",
    "input_data_ids",
    "base_input_data_id",
    "factor_columns",
    "feature_columns",
    "feature_transforms",
    "eligibility_recipe",
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
        "logical_dtypes",
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
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


def merge_dataset_metadata(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(payload)
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise TypeError("invalid_dataset_metadata")
        unknown = set(metadata) - _DATASET_METADATA_ALLOWED_KEYS
        if unknown:
            raise ValueError("unsupported_dataset_metadata_key")
        for key, value in metadata.items():
            result[key] = qrp_json_object_snapshot(value)
    return result


def _require_identity_hash(value: Any, error: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in _LOWER_HEX for c in value)
    ):
        raise ValueError(error)
    return value


def validate_dataset_spec(spec: DataSpec) -> None:
    if type(spec) not in {
        DataSpec,
        InterimDataSpec,
        ProcessedDataSpec,
        ResearchDatasetSpec,
    }:
        raise ValueError("dataset_spec_type_invalid")


def arrow_type_has_null(data_type: pa.DataType) -> bool:
    return pa.types.is_null(data_type) or any(
        arrow_type_has_null(data_type.field(index).type)
        for index in range(data_type.num_fields)
    )


def arrow_type_resolves_null(old: pa.DataType, new: pa.DataType) -> bool:
    if pa.types.is_null(old):
        return True
    if old.id != new.id or old.num_fields != new.num_fields:
        return False
    if not old.num_fields:
        return old == new
    if pa.types.is_fixed_size_list(old) and old.list_size != new.list_size:
        return False
    if pa.types.is_map(old) and old.keys_sorted != new.keys_sorted:
        return False
    return all(
        old.field(index).with_type(new.field(index).type) == new.field(index)
        and arrow_type_resolves_null(old.field(index).type, new.field(index).type)
        for index in range(old.num_fields)
    )


def physical_schema_from_schema(schema: pa.Schema) -> list[dict[str, Any]]:
    if not isinstance(schema, pa.Schema):
        raise TypeError("dataset_arrow_schema_required")
    payload = [_physical_field_payload(field) for field in schema]
    validate_physical_schema_payload(payload)
    return payload


def logical_dtype_payload_from_frame(
    frame: pd.DataFrame, *, columns: tuple[str, ...] | None = None
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("dataset_frame_required")
    selected = (
        tuple(str(column) for column in frame.columns) if columns is None else columns
    )
    if any(column not in frame for column in selected):
        raise ValueError("dataset_column_not_found")
    payload = {
        "schema_id": "qrp-dataset-part-logical-dtypes",
        "columns": [
            _logical_dtype_descriptor(str(column), frame[column]) for column in selected
        ],
    }
    for descriptor in payload["columns"]:
        if descriptor["kind"] != "categorical" or descriptor["ordered"]:
            continue
        for category in descriptor["categories"]:
            if category["type"] == "decimal" and "." in category["value"]:
                category["value"] = category["value"].rstrip("0").rstrip(".")
    return payload


def logical_dtype_metadata_bytes(payload: Mapping[str, Any]) -> bytes:
    validated = validate_logical_dtype_payload(payload)
    return (render_qrp_json_object(validated) + "\n").encode("utf-8")


def encode_logical_category_value(value: Mapping[str, Any]) -> tuple[bytes, bytes]:
    return (
        canonical_identity_bytes(value),
        (
            render_qrp_json_object({"schema_id": "qrp-category-value", "value": value})
            + "\n"
        ).encode("utf-8"),
    )


def parse_logical_dtype_metadata(
    raw: bytes, *, selected_columns: tuple[str, ...] | None = None
) -> dict[str, Any]:
    try:
        payload = parse_qrp_json_object(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_logical_dtypes_invalid") from exc
    return validate_logical_dtype_payload(payload, selected_columns=selected_columns)


def validate_logical_dtype_payload(
    value: object, *, selected_columns: tuple[str, ...] | None = None
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"schema_id", "columns"}:
        raise ValueError("dataset_logical_dtypes_invalid")
    if value.get("schema_id") != "qrp-dataset-part-logical-dtypes":
        raise ValueError("dataset_logical_dtypes_invalid")
    columns = value.get("columns")
    if type(columns) is not list or not columns:
        raise ValueError("dataset_logical_dtypes_invalid")
    names: list[str] = []
    for descriptor in columns:
        if type(descriptor) is not dict:
            raise ValueError("dataset_logical_dtypes_invalid")
        name = descriptor.get("name")
        kind = descriptor.get("kind")
        if type(name) is not str or not name or type(kind) is not str or (not kind):
            raise ValueError("dataset_logical_dtypes_invalid")
        names.append(name)
        if selected_columns is not None and name not in selected_columns:
            continue
        if kind == "categorical":
            if set(descriptor) != {"name", "kind", "ordered", "categories"}:
                raise ValueError("dataset_logical_dtypes_invalid")
            if type(descriptor["ordered"]) is not bool:
                raise ValueError("dataset_logical_dtypes_invalid")
            categories = descriptor["categories"]
            if type(categories) is not list:
                raise ValueError("dataset_logical_dtypes_invalid")
            identities = [_typed_category_identity(item) for item in categories]
            if len(identities) != len(set(identities)):
                raise ValueError("dataset_logical_dtypes_invalid")
        elif set(descriptor) != {"name", "kind", "dtype"}:
            raise ValueError("dataset_logical_dtypes_invalid")
        elif type(descriptor["dtype"]) is not str or not descriptor["dtype"]:
            raise ValueError("dataset_logical_dtypes_invalid")
        try:
            canonical = _canonical_logical_dtype_descriptor(descriptor)
            is_canonical = canonical_identity_bytes(
                canonical
            ) == canonical_identity_bytes(descriptor)
        except (TypeError, ValueError, OverflowError):
            is_canonical = False
        if not is_canonical:
            raise ValueError("dataset_logical_dtypes_invalid")
    if len(names) != len(set(names)):
        raise ValueError("dataset_logical_dtypes_invalid")
    return {
        "schema_id": value["schema_id"],
        "columns": [dict(item) for item in columns],
    }


def validate_persisted_logical_schema(
    schema: pa.Schema, logical_dtypes: Mapping[str, Any]
) -> None:
    """Require logical dtypes to preserve the persisted physical values."""
    if {item["name"] for item in logical_dtypes["columns"]} != set(schema.names):
        raise ValueError("dataset_logical_dtype_mismatch")
    for descriptor in logical_dtypes["columns"]:
        data_type = schema.field(descriptor["name"]).type
        if descriptor["kind"] == "categorical":
            if pa.types.is_dictionary(data_type):
                if bool(data_type.ordered) != bool(descriptor["ordered"]):
                    raise ValueError("dataset_categorical_domain_mismatch")
                data_type = data_type.value_type
            categories = pa.Array.from_pandas(
                pd.Index(
                    [_typed_category_value(item) for item in descriptor["categories"]]
                )
            )
            try:
                restored = categories.cast(data_type, safe=True).cast(
                    categories.type, safe=True
                )
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
            ) as error:
                raise ValueError("dataset_categorical_domain_mismatch") from error
            if not restored.equals(categories):
                raise ValueError("dataset_categorical_domain_mismatch")
            continue
        if pa.types.is_dictionary(data_type):
            raise ValueError("dataset_logical_dtype_mismatch")
        if descriptor["kind"] == "object":
            continue
        expected = pa.Array.from_pandas(pd.Series([], dtype=descriptor["dtype"])).type
        if (
            pa.types.is_timestamp(data_type)
            and pa.types.is_timestamp(expected)
            and (data_type.tz == expected.tz)
        ):
            continue
        if (pa.types.is_string(data_type) or pa.types.is_large_string(data_type)) and (
            pa.types.is_string(expected) or pa.types.is_large_string(expected)
        ):
            continue
        if data_type != expected:
            raise ValueError("dataset_logical_dtype_mismatch")


def _canonical_logical_dtype_descriptor(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    name = descriptor["name"]
    if descriptor["kind"] == "categorical":
        dtype: object = pd.CategoricalDtype(
            categories=[
                _typed_category_value(item) for item in descriptor["categories"]
            ],
            ordered=descriptor["ordered"],
        )
    else:
        dtype = descriptor["dtype"]
    return _logical_dtype_descriptor(name, pd.Series([], dtype=dtype))


def merge_logical_dtype_payloads(payloads: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge declarations already constructed or validated by the dtype owner."""
    if not payloads:
        raise ValueError("dataset_logical_dtypes_missing")
    first = payloads[0]
    names = tuple(item["name"] for item in first["columns"])
    merged = [dict(item) for item in first["columns"]]
    for payload in payloads[1:]:
        if tuple(item["name"] for item in payload["columns"]) != names:
            raise ValueError("dataset_logical_dtype_mismatch")
        for index, current in enumerate(payload["columns"]):
            expected = merged[index]
            if current["kind"] != expected["kind"]:
                promoted = _promote_compatible_logical_dtype(expected, current)
                if promoted is None:
                    raise ValueError("dataset_logical_dtype_mismatch")
                merged[index] = promoted
                continue
            if current["kind"] != "categorical":
                if current != expected:
                    raise ValueError("dataset_logical_dtype_mismatch")
                continue
            if current["ordered"] != expected["ordered"]:
                raise ValueError("dataset_logical_dtype_mismatch")
            if current["ordered"]:
                if current["categories"] != expected["categories"]:
                    raise ValueError("dataset_ordered_categorical_mismatch")
                continue
            domains = {
                _typed_category_identity(item): item
                for item in [*expected["categories"], *current["categories"]]
            }
            expected["categories"] = list(domains.values())
    for descriptor in merged:
        if descriptor["kind"] == "categorical" and (not descriptor["ordered"]):
            descriptor["categories"] = sorted(
                descriptor["categories"], key=_typed_category_identity
            )
    return {"schema_id": first["schema_id"], "columns": merged}


def _promote_compatible_logical_dtype(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any] | None:
    kinds = {left["kind"], right["kind"]}
    if kinds == {"numpy_integer", "nullable_integer"}:
        numpy_descriptor = left if left["kind"] == "numpy_integer" else right
        nullable_descriptor = left if left["kind"] == "nullable_integer" else right
        if nullable_descriptor["dtype"].lower() == numpy_descriptor["dtype"]:
            return dict(nullable_descriptor)
    if kinds == {"numpy_boolean", "nullable_boolean"}:
        nullable_descriptor = left if left["kind"] == "nullable_boolean" else right
        return dict(nullable_descriptor)
    if kinds == {"numpy_float", "nullable_float"}:
        numpy_descriptor = left if left["kind"] == "numpy_float" else right
        nullable_descriptor = left if left["kind"] == "nullable_float" else right
        if nullable_descriptor["dtype"].lower() == numpy_descriptor["dtype"]:
            return dict(nullable_descriptor)
    return None


def pandas_dtype_overrides_from_logical_payload(
    payload: Mapping[str, Any], *, columns: tuple[str, ...]
) -> dict[str, object]:
    """Construct pandas dtypes from an already admitted logical declaration."""
    selected = set(columns)
    result: dict[str, object] = {}
    for descriptor in payload["columns"]:
        name = descriptor["name"]
        if name not in selected:
            continue
        kind = descriptor["kind"]
        if kind == "categorical":
            result[name] = pd.CategoricalDtype(
                categories=[
                    _typed_category_value(item) for item in descriptor["categories"]
                ],
                ordered=bool(descriptor["ordered"]),
            )
        elif kind in {
            "numpy_integer",
            "numpy_boolean",
            "numpy_float",
            "nullable_integer",
            "nullable_boolean",
            "nullable_float",
            "string",
            "datetime",
            "timedelta",
            "object",
        }:
            result[name] = pd.api.types.pandas_dtype(descriptor["dtype"])
    return result


def _logical_dtype_descriptor(name: str, series: pd.Series) -> dict[str, Any]:
    dtype = _canonical_pandas_logical_dtype(series.dtype)
    if isinstance(dtype, pd.CategoricalDtype):
        return {
            "name": name,
            "kind": "categorical",
            "ordered": bool(dtype.ordered),
            "categories": [
                _typed_category_value_payload(item) for item in dtype.categories
            ],
        }
    dtype_name = str(dtype)
    if isinstance(dtype, pd.StringDtype):
        kind = "string"
    elif isinstance(dtype, pd.BooleanDtype):
        kind = "nullable_boolean"
    elif isinstance(dtype, pd.api.extensions.ExtensionDtype) and dtype_name.startswith(
        ("Int", "UInt")
    ):
        kind = "nullable_integer"
    elif isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
        kind = "nullable_float"
    elif pd.api.types.is_integer_dtype(dtype):
        kind = "numpy_integer"
    elif pd.api.types.is_bool_dtype(dtype):
        kind = "numpy_boolean"
    elif pd.api.types.is_float_dtype(dtype):
        kind = "numpy_float"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        kind = "datetime"
    elif pd.api.types.is_timedelta64_dtype(dtype):
        kind = "timedelta"
    elif pd.api.types.is_object_dtype(dtype):
        kind = "object"
    else:
        raise ValueError("unsupported_dataset_logical_dtype")
    return {"name": name, "kind": kind, "dtype": dtype_name}


def _canonical_pandas_logical_dtype(dtype: object) -> object:
    if not isinstance(dtype, pd.ArrowDtype):
        return dtype
    arrow_type = dtype.pyarrow_dtype
    if pa.types.is_boolean(arrow_type):
        return pd.BooleanDtype()
    if pa.types.is_signed_integer(arrow_type):
        return pd.api.types.pandas_dtype(f"Int{arrow_type.bit_width}")
    if pa.types.is_unsigned_integer(arrow_type):
        return pd.api.types.pandas_dtype(f"UInt{arrow_type.bit_width}")
    if pa.types.is_floating(arrow_type) and arrow_type.bit_width in {32, 64}:
        return pd.api.types.pandas_dtype(f"Float{arrow_type.bit_width}")
    if pa.types.is_timestamp(arrow_type):
        if arrow_type.tz is not None:
            return pd.DatetimeTZDtype(unit=arrow_type.unit, tz=arrow_type.tz)
        return np.dtype(f"datetime64[{arrow_type.unit}]")
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return pd.StringDtype()
    raise ValueError("unsupported_dataset_logical_dtype")


def _typed_category_value_payload(value: object) -> dict[str, Any]:
    if isinstance(value, np.generic):
        return _typed_category_value_payload(value.item())
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("dataset_categorical_value_invalid")
        return {"type": "float", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    if type(value) is bytes:
        return {"type": "bytes", "value": value.hex()}
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("dataset_categorical_value_invalid")
        if value.is_zero():
            value = Decimal(0)
        return {"type": "decimal", "value": format(value, "f")}
    if type(value) is date and type(value) is not datetime:
        return {"type": "date", "value": value.isoformat()}
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            raise ValueError("dataset_categorical_value_invalid")
        zone = getattr(timestamp.tz, "key", getattr(timestamp.tz, "zone", None))
        return {
            "type": "timestamp",
            "value": f"{timestamp.isoformat()}@{timestamp.unit}"
            + (f"[{zone}]" if zone is not None else ""),
        }
    raise ValueError("dataset_categorical_value_invalid")


def _typed_category_identity(value: object) -> bytes:
    if type(value) is not dict or set(value) != {"type", "value"}:
        raise ValueError("dataset_logical_dtypes_invalid")
    _typed_category_value(value)
    return canonical_identity_bytes(value)


def _typed_category_value(value: Mapping[str, Any]) -> object:
    kind = value.get("type")
    payload = value.get("value")
    if kind == "bool" and type(payload) is bool:
        return payload
    if kind == "int" and type(payload) is int:
        return payload
    if kind == "float" and type(payload) is float and pd.notna(payload):
        return payload
    if kind == "str" and type(payload) is str:
        return payload
    if kind == "bytes" and type(payload) is str:
        try:
            return bytes.fromhex(payload)
        except ValueError as exc:
            raise ValueError("dataset_logical_dtypes_invalid") from exc
    if kind == "decimal" and type(payload) is str:
        if "e" in payload.lower():
            raise ValueError("dataset_logical_dtypes_invalid")
        try:
            result = Decimal(payload)
        except (ValueError, InvalidOperation) as exc:
            raise ValueError("dataset_logical_dtypes_invalid") from exc
        if result.is_finite():
            return result
    if kind == "date" and type(payload) is str:
        try:
            return date.fromisoformat(payload)
        except ValueError as exc:
            raise ValueError("dataset_logical_dtypes_invalid") from exc
    if kind == "timestamp" and type(payload) is str:
        try:
            zone = None
            if payload.endswith("]"):
                payload, _, zone = payload[:-1].rpartition("[")
            instant, _, unit = payload.rpartition("@")
            result = pd.Timestamp(instant).as_unit(unit)
            if zone is not None:
                result = result.tz_convert(zone)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("dataset_logical_dtypes_invalid") from exc
        if not pd.isna(result):
            return result
    raise ValueError("dataset_logical_dtypes_invalid")


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
    if pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        return {"kind": "time", "unit": data_type.unit}
    if pa.types.is_timestamp(data_type):
        return {"kind": "timestamp", "unit": data_type.unit, "timezone": data_type.tz}
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


def arrow_schema_from_payload(value: list[dict[str, Any]]) -> pa.Schema:
    """Restore the writer-declared Arrow schema without scanning unrelated files."""

    def field(item):
        return pa.field(item["name"], dtype(item["type"]), nullable=item["nullable"])

    def dtype(item):
        kind = item["kind"]
        if kind == "integer":
            return getattr(
                pa, ("int" if item["signed"] else "uint") + str(item["bit_width"])
            )()
        if kind == "float":
            return getattr(pa, "float" + str(item["bit_width"]))()
        if kind == "decimal":
            return getattr(pa, "decimal" + str(item["bit_width"]))(
                item["precision"], item["scale"]
            )
        if kind == "date":
            return pa.date32()
        if kind == "time":
            return (pa.time32 if item["unit"] in {"s", "ms"} else pa.time64)(
                item["unit"]
            )
        if kind == "timestamp":
            return pa.timestamp(item["unit"], tz=item["timezone"])
        if kind == "duration":
            return pa.duration(item["unit"])
        if kind == "fixed_size_binary":
            return pa.binary(item["byte_width"])
        if kind == "struct":
            return pa.struct([field(child) for child in item["fields"]])
        if kind in {"list", "large_list", "fixed_size_list"}:
            return (
                pa.list_(field(item["value_field"]), item["list_size"])
                if kind == "fixed_size_list"
                else (pa.large_list if kind == "large_list" else pa.list_)(
                    field(item["value_field"])
                )
            )
        if kind == "map":
            return pa.map_(
                field(item["key_field"]),
                field(item["item_field"]),
                keys_sorted=item["keys_sorted"],
            )
        if kind == "dictionary":
            return pa.dictionary(
                dtype(item["index_type"]),
                dtype(item["value_type"]),
                ordered=item["ordered"],
            )
        return {
            "bool": pa.bool_,
            "string": pa.string,
            "large_string": pa.large_string,
            "binary": pa.binary,
            "large_binary": pa.large_binary,
        }[kind]()

    return pa.schema([field(item) for item in value])


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
        "column_units": dict(spec.column_units),
        "label_availability_columns": dict(spec.label_availability_columns),
    }
    if isinstance(spec, InterimDataSpec):
        payload.update({"source_id": spec.source_id})
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {
                "market_id": spec.market_id,
                "kind": spec.kind.value,
                "source_input_data_ids": list(spec.source_input_data_ids),
                "optional_source_input_data_ids": list(
                    spec.optional_source_input_data_ids
                ),
                "processed_input_data_ids": list(spec.processed_input_data_ids),
                "parameters": parse_qrp_json_object(
                    spec.parameters_json.encode("utf-8")
                ),
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
                "eligibility_recipe": spec.eligibility_recipe,
                "context_columns": list(spec.context_columns),
                "key_coverage": dict(spec.key_coverage),
            }
        )
    return payload


def _feature_transform_payload(transform: FeatureTransformSpec) -> dict[str, Any]:
    return {
        "recipe_id": transform.recipe_id,
        "source_columns": list(transform.source_columns),
        "clip": transform.clip,
        "industry_neutralize": transform.industry_neutralize,
        "size_neutralize": transform.size_neutralize,
        "size_column": transform.size_column,
        "center": transform.center,
        "scale": transform.scale,
        "impute_missing": transform.impute_missing,
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
        spec.get("grain"), {item.value for item in DataGrain}, "dataset_grain_invalid"
    )
    primary_key = _require_string_list(
        spec.get("primary_key"), "dataset_primary_key_invalid", allow_empty=False
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
        (
            not isinstance(key, str)
            or not isinstance(value, str)
            or value not in {role.value for role in ColumnRole}
            for key, value in column_roles.items()
        )
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
        "column_units": spec.get("column_units"),
        "label_availability_columns": spec.get("label_availability_columns"),
    }
    if keys == _DATASET_SPEC_BASE_KEYS:
        DataSpec(**common_kwargs)
        return
    if keys == _DATASET_SPEC_INTERIM_KEYS:
        source_id = _require_nonempty_string(
            spec.get("source_id"), "dataset_source_id_invalid"
        )
        InterimDataSpec(**common_kwargs, source_id=SourceId(source_id))
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
        ProcessedDataSpec(
            **common_kwargs,
            market_id=MarketId(market_id),
            kind=ProcessedDataKind(kind),
            source_input_data_ids=tuple(source_input_data_ids),
            optional_source_input_data_ids=tuple(
                _require_string_list(
                    spec.get("optional_source_input_data_ids"),
                    "dataset_optional_source_input_data_ids_invalid",
                    allow_empty=True,
                )
            ),
            processed_input_data_ids=tuple(processed_input_data_ids),
            parameters_json=render_qrp_json_object(spec["parameters"]),
        )
        return
    market_id = _require_nonempty_string(
        spec.get("market_id"), "dataset_market_id_invalid"
    )
    input_data_ids = _require_string_list(
        spec.get("input_data_ids"), "dataset_input_data_ids_invalid", allow_empty=False
    )
    base_input_data_id = _require_nonempty_string(
        spec.get("base_input_data_id"), "dataset_base_input_data_id_invalid"
    )
    factor_columns = _require_string_list(
        spec.get("factor_columns"), "dataset_factor_columns_invalid", allow_empty=True
    )
    feature_columns = _require_string_list(
        spec.get("feature_columns"), "dataset_feature_columns_invalid", allow_empty=True
    )
    label_columns = _require_string_list(
        spec.get("label_columns"), "dataset_label_columns_invalid", allow_empty=True
    )
    filter_columns = _require_string_list(
        spec.get("filter_columns"), "dataset_filter_columns_invalid", allow_empty=True
    )
    context_columns = _require_string_list(
        spec.get("context_columns"), "dataset_context_columns_invalid", allow_empty=True
    )
    feature_transforms = _validate_feature_transform_payloads(
        spec.get("feature_transforms")
    )
    key_coverage = spec.get("key_coverage")
    if not isinstance(key_coverage, dict) or set(key_coverage) != set(input_data_ids):
        raise ValueError("dataset_key_coverage_invalid")
    for value in key_coverage.values():
        _require_enum_string(
            value, {"strict", "sparse"}, "dataset_key_coverage_invalid"
        )
    try:
        ResearchDatasetSpec(
            **common_kwargs,
            market_id=market_id,
            input_data_ids=tuple(input_data_ids),
            base_input_data_id=base_input_data_id,
            factor_columns=tuple(factor_columns),
            feature_columns=tuple(feature_columns),
            feature_transforms=tuple(feature_transforms),
            label_columns=tuple(label_columns),
            filter_columns=tuple(filter_columns),
            eligibility_recipe=spec.get("eligibility_recipe"),
            context_columns=tuple(context_columns),
            key_coverage=dict(key_coverage),
        )

    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_research_spec_invalid") from exc


def validate_dataset_manifest_payload(
    manifest: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    logical_columns: tuple[str, ...] | None = None,
) -> None:
    validate_dataset_spec_payload(spec)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_id") != "qrp-dataset-manifest"
    ):
        raise ValueError("dataset_manifest_schema_id_invalid")
    expected_keys = _DATASET_MANIFEST_RUNTIME_KEYS | {"schema_id"}
    actual_keys = set(manifest)
    if not expected_keys.issubset(actual_keys) or not (
        actual_keys - expected_keys
    ).issubset(_DATASET_METADATA_ALLOWED_KEYS):
        raise ValueError("dataset_manifest_fields_invalid")
    for key in (
        "row_count",
        "file_count",
        "file_size_bytes",
        "partition_count",
        "duplicate_primary_key_count",
    ):
        value = manifest.get(key)
        if not _is_exact_int(value) or value < 0:
            raise ValueError(f"manifest_{key}_invalid")
    columns = _require_string_list(
        manifest.get("columns"), "manifest_columns_invalid", allow_empty=False
    )
    if columns != spec["required_columns"]:
        raise ValueError("manifest_columns_mismatch")
    physical_schema = manifest.get("physical_schema")
    validate_physical_schema_payload(physical_schema)
    if list(physical_schema_columns(physical_schema)) != columns:
        raise ValueError("manifest_columns_mismatch")
    _require_identity_hash(manifest.get("schema_hash"), "manifest_schema_hash_invalid")
    logical_dtypes = validate_logical_dtype_payload(
        manifest.get("logical_dtypes"), selected_columns=logical_columns
    )
    if [item["name"] for item in logical_dtypes["columns"]] != columns:
        raise ValueError("manifest_logical_dtypes_mismatch")
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
    date_range = _validate_date_range(
        manifest.get("date_min"), manifest.get("date_max")
    )
    null_counts = _require_count_mapping(
        manifest.get("null_count_summary"),
        allowed_keys=set(columns),
        error="manifest_null_count_summary_invalid",
    )
    non_finite_counts = _require_count_mapping(
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
    factor_counts = _require_count_mapping(
        manifest.get("factor_nan_profile"),
        allowed_keys=factor_columns,
        error="manifest_factor_nan_profile_invalid",
    )
    label_counts = _require_count_mapping(
        manifest.get("label_nan_profile"),
        allowed_keys=label_columns,
        error="manifest_label_nan_profile_invalid",
    )
    row_count = manifest["row_count"]
    if (
        any(
            count > row_count
            for counts in (
                null_counts,
                non_finite_counts,
                factor_counts,
                label_counts,
            )
            for count in counts.values()
        )
        or manifest["duplicate_primary_key_count"] > row_count
    ):
        raise ValueError("manifest_summary_count_invalid")
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is not None:
        if (row_count == 0) != (date_range is None):
            raise ValueError("manifest_date_range_invalid")
        if tuple(spec["partition_columns"]) == ("year",):
            partition_year_by_label: dict[str, int] = {}
            for partition in partitions:
                if (
                    partition == "unpartitioned"
                    and row_count == 0
                    and (partition_counts[partition] == 0)
                ):
                    continue
                partition_year_by_label[partition] = parse_year_partition_label(
                    partition
                )
            partition_years = {
                partition_year_by_label[partition]
                for partition in partitions
                if partition_counts[partition] > 0
            }
            if date_range is not None:
                minimum, maximum = date_range
                if (
                    not partition_years
                    or minimum.year not in partition_years
                    or maximum.year not in partition_years
                    or any(
                        year < minimum.year or year > maximum.year
                        for year in partition_years
                    )
                ):
                    raise ValueError("manifest_partition_date_mismatch")
    elif date_range is not None:
        raise ValueError("manifest_date_range_invalid")
    merge_dataset_metadata(
        {},
        {
            key: manifest[key]
            for key in _DATASET_METADATA_ALLOWED_KEYS
            if key in manifest
        },
    )


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
        if not _is_exact_int(value["bit_width"]) or value["bit_width"] not in {
            16,
            32,
            64,
        }:
            raise ValueError("manifest_physical_schema_invalid")
        return
    if kind == "decimal":
        _require_exact_keys(value, {"kind", "bit_width", "precision", "scale"})
        if (
            not _is_exact_int(value["bit_width"])
            or value["bit_width"] not in {128, 256}
            or (not _is_exact_int(value["precision"]))
            or (value["precision"] <= 0)
            or (value["precision"] > (38 if value["bit_width"] == 128 else 76))
            or (not _is_exact_int(value["scale"]))
            or (value["scale"] < 0)
            or (value["scale"] > value["precision"])
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
            value["unit"], {"s", "ms", "us", "ns"}, "manifest_physical_schema_invalid"
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
        _require_exact_keys(value, {"kind", "keys_sorted", "key_field", "item_field"})
        if type(value["keys_sorted"]) is not bool:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_field_payload(value["key_field"])
        if value["key_field"]["nullable"] is not False:
            raise ValueError("manifest_physical_schema_invalid")
        _validate_physical_field_payload(value["item_field"])
        return
    if kind == "dictionary":
        _require_exact_keys(value, {"kind", "index_type", "value_type", "ordered"})
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
    if type(value) is not list:
        raise ValueError("dataset_feature_transforms_invalid")
    transforms = []
    for item in value:
        if type(item) is not dict or set(item) != {
            "recipe_id",
            "source_columns",
            "clip",
            "industry_neutralize",
            "size_neutralize",
            "size_column",
            "center",
            "scale",
            "impute_missing",
        }:
            raise ValueError("dataset_feature_transforms_invalid")
        sources = _require_string_list(
            item["source_columns"],
            "dataset_feature_transforms_invalid",
            allow_empty=False,
        )
        try:
            transforms.append(
                FeatureTransformSpec(**{**item, "source_columns": tuple(sources)})
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


def _require_string_list(value: object, error: str, *, allow_empty: bool) -> list[str]:
    if (
        not isinstance(value, list)
        or (not value and (not allow_empty))
        or any(not isinstance(item, str) or not item for item in value)
        or (len(value) != len(set(value)))
    ):
        raise ValueError(error)
    return value


def _is_exact_int(value: object) -> bool:
    return type(value) is int


def _require_count_mapping(
    value: object, *, allowed_keys: set[str], error: str
) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(error)
    result: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key not in allowed_keys
            or (not _is_exact_int(item))
            or (item < 0)
        ):
            raise ValueError(error)
        result[key] = item
    return result


def _validate_date_range(
    date_min: object, date_max: object
) -> tuple[date, date] | None:
    if date_min is None and date_max is None:
        return None
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
    return (minimum, maximum)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("manifest_physical_schema_invalid")


def validate_frame_objects(frame: pd.DataFrame) -> tuple[str, ...]:
    """Reject recursive object graphs and identify strings before batch slicing."""

    def visit(value: object, active: list[object]) -> None:
        if not isinstance(value, (dict, list, tuple, np.ndarray)):
            return
        if any(value is parent for parent in active):
            raise ValueError("qrp_json_cycle")
        if isinstance(value, np.ndarray) and value.dtype.kind != "O":
            return
        active.append(value)
        try:
            children = (
                value.values()
                if isinstance(value, dict)
                else value.flat
                if isinstance(value, np.ndarray)
                else value
            )
            for child in children:
                visit(child, active)
        finally:
            active.pop()

    strings = []
    for name, values in zip(frame.columns, frame._iter_column_arrays(), strict=True):
        if not pd.api.types.is_object_dtype(values.dtype):
            continue
        has_string, strings_or_nulls = (False, True)
        for value in values:
            if isinstance(value, str):
                has_string = True
            elif not pd.api.types.is_scalar(value) or not pd.isna(value):
                strings_or_nulls = False
            visit(value, [])
        if has_string and strings_or_nulls:
            strings.append(name)
    return tuple(strings)
