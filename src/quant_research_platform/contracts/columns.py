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
        if self.schema_version != "0.1":
            raise ValueError("unsupported_field_schema_version")
        if self.logical_type is None:
            object.__setattr__(
                self,
                "logical_type",
                infer_logical_type(dtype=self.dtype, role=self.role),
            )


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
