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


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: ColumnRole
    dtype: str
    nullable: bool = False
    schema_version: str = "0.1"
    description: str = ""


__all__ = ["ColumnRole", "FieldSpec"]
