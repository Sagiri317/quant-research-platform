"""Processed data frequency and grain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.ids import MarketId, SourceId


class DataFrequency(StrEnum):
    DAILY = "1d"
    MINUTE = "1m"
    TICK = "tick"


class DataGrain(StrEnum):
    ASSET_TIMESTAMP = "asset_timestamp"
    MARKET_TIMESTAMP = "market_timestamp"
    TRADE_EVENT = "trade_event"
    ORDER_BOOK_LEVEL = "order_book_level"
    RESEARCH_ROW = "research_row"


@dataclass(frozen=True, kw_only=True)
class DataSpec:
    market_id: MarketId
    frequency: DataFrequency
    grain: DataGrain
    column_roles: dict[str, ColumnRole]
    primary_key: tuple[str, ...]
    partition_columns: tuple[str, ...]
    schema_version: str = "0.1"
    required_columns: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != "0.1":
            raise ValueError("unsupported_schema_version")
        missing = [
            column for column in self.primary_key if column not in self.column_roles
        ]
        if missing:
            raise ValueError(f"missing_primary_key_columns: {', '.join(missing)}")
        object.__setattr__(self, "required_columns", tuple(self.column_roles))


@dataclass(frozen=True, kw_only=True)
class InterimDataSpec(DataSpec):
    source_id: SourceId
    interim_data_id: str


@dataclass(frozen=True, kw_only=True)
class ProcessedDataSpec(DataSpec):
    category: str
    processed_data_id: str


__all__ = [
    "DataFrequency",
    "DataGrain",
    "DataSpec",
    "InterimDataSpec",
    "ProcessedDataSpec",
]
