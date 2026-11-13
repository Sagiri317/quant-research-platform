"""Validate dataset declarations and convert between logical pandas and Arrow
schemas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.data import (
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.features import (
    FeatureTransformSpec,
)
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    parse_qrp_json_object,
    render_qrp_json_object,
)

_DATASET_RESEARCH_METADATA_KEYS = frozenset(
    {"producer_diagnostics", "inputs", "parameters", "source_evidence"}
)


def merge_dataset_metadata(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep supplied research facts; computed table facts retain their own owner."""
    return dict(payload) if metadata is None else {**metadata, **payload}


def physical_schema_from_schema(schema: pa.Schema) -> list[dict[str, Any]]:
    payload = [_physical_field_payload(field) for field in schema]
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
    return payload


def logical_dtype_metadata_bytes(payload: Mapping[str, Any]) -> bytes:
    return (render_qrp_json_object(payload) + "\n").encode("utf-8")


def encode_logical_category_value(value: Mapping[str, Any]) -> tuple[bytes, bytes]:
    return (
        canonical_identity_bytes(value),
        (
            render_qrp_json_object({"schema_id": "qrp-category-value", "value": value})
            + "\n"
        ).encode("utf-8"),
    )


def parse_logical_dtype_metadata(raw: bytes) -> dict[str, Any]:
    try:
        payload = parse_qrp_json_object(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("dataset_logical_dtypes_invalid") from exc
    return payload


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
    """Construct pandas dtypes from a validated logical declaration."""
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
    raise ValueError("dataset_logical_dtypes_invalid")


def physical_schema_columns(schema: object) -> tuple[str, ...]:
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
                "target_transform": spec.target_transform,
                "target_label_column": spec.target_label_column,
                "target_clip": spec.target_clip,
                "label_columns": list(spec.label_columns),
                "filter_columns": list(spec.filter_columns),
                "eligibility_recipe": spec.eligibility_recipe,
                "minimum_listing_sessions": spec.minimum_listing_sessions,
                "exclude_st": spec.exclude_st,
                "exclude_pt": spec.exclude_pt,
                "context_columns": list(spec.context_columns),
                "key_coverage": dict(spec.key_coverage),
            }
        )
        if spec.risk_model is not None:
            payload["risk_model"] = spec.risk_model.to_dict()
    return payload


def _feature_transform_payload(transform: FeatureTransformSpec) -> dict[str, Any]:
    return {
        "recipe_id": transform.recipe_id,
        "source_columns": list(transform.source_columns),
        "output_groups": list(transform.output_groups),
        "clip": transform.clip,
        "industry_neutralize": transform.industry_neutralize,
        "size_neutralize": transform.size_neutralize,
        "size_column": transform.size_column,
        "center": transform.center,
        "scale": transform.scale,
        **(
            {"risk_model": transform.risk_model.to_dict()}
            if transform.risk_model is not None
            else {}
        ),
    }


def date_string(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def object_string_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Identify scalar string columns before batch slicing can hide their values."""
    return tuple(
        name
        for name, values in zip(frame.columns, frame._iter_column_arrays(), strict=True)
        if pd.api.types.is_object_dtype(values.dtype)
        and pd.api.types.infer_dtype(values, skipna=True) == "string"
    )
