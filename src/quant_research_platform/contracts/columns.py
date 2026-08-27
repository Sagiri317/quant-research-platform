"""Column roles used by portable 0.1 data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ColumnRole(StrEnum):
    KEY = "key"
    TIMESTAMP = "timestamp"
    SOURCE = "source"
    FACTOR = "factor"
    LABEL = "label"
    MASK = "mask"
    UNIVERSE = "universe"
    CONTEXT = "context"
    FEATURE = "feature"
    PREDICTION = "prediction"
    PORTFOLIO = "portfolio"


class LogicalType(StrEnum):
    STRING_ID = "string_id"
    TIMESTAMP = "timestamp"
    DATE = "date"
    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORY = "category"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: ColumnRole
    dtype: str
    logical_type: LogicalType | None = None
    nullable: bool = False
    schema_version: str = "0.1"
    description: str = ""

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "0.1":
            raise ValueError("unsupported_field_schema_version")
        if type(self.name) is not str or not self.name:
            raise ValueError("field_name_required")
        if type(self.dtype) is not str or not self.dtype:
            raise ValueError("field_dtype_required")
        try:
            role = ColumnRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported_field_role") from exc
        if type(self.nullable) is not bool:
            raise ValueError("field_nullable_must_be_boolean")
        if type(self.description) is not str:
            raise ValueError("field_description_must_be_string")
        if self.logical_type is not None:
            try:
                logical_type = LogicalType(self.logical_type)
            except (TypeError, ValueError) as exc:
                raise ValueError("unsupported_field_logical_type") from exc
        else:
            logical_type = infer_logical_type(dtype=self.dtype, role=role)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "logical_type", logical_type)


def infer_logical_type(*, dtype: str, role: ColumnRole) -> LogicalType:
    if role == ColumnRole.KEY or role == ColumnRole.SOURCE:
        return LogicalType.STRING_ID
    normalized = dtype.lower()
    if "timestamp" in normalized:
        return LogicalType.TIMESTAMP
    if normalized == "date":
        return LogicalType.DATE
    if normalized in {"bool", "boolean"}:
        return LogicalType.BOOLEAN
    if normalized.startswith("int"):
        return LogicalType.INTEGER
    if normalized.startswith("float"):
        return LogicalType.FLOAT
    return LogicalType.CATEGORY
