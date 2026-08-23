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
    ASSET_REFERENCE = "asset_reference"
    ASSET_REFERENCE_TIMESTAMP = "asset_reference_timestamp"
    MARKET_TIMESTAMP = "market_timestamp"
    SOURCE_MARKET_CALENDAR_DATE = "source_market_calendar_date"
    TRADE_EVENT = "trade_event"
    ORDER_BOOK_LEVEL = "order_book_level"
    RESEARCH_ROW = "research_row"


class ProcessedDataKind(StrEnum):
    PANEL = "panel"
    ANNOTATION = "annotation"
    FACTOR = "factor"
    LABEL = "label"


def normalize_data_frequency(value: DataFrequency | str) -> DataFrequency:
    """Return the platform frequency vocabulary item for a raw value."""

    if isinstance(value, DataFrequency):
        return value
    try:
        return DataFrequency(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid_data_frequency: {value}") from exc


def data_frequency_slug(value: DataFrequency | str) -> str:
    return normalize_data_frequency(value).value


def horizon_unit_suffix(value: DataFrequency | str) -> str:
    frequency = normalize_data_frequency(value)
    if frequency is DataFrequency.DAILY:
        return "d"
    if frequency is DataFrequency.MINUTE:
        return "m"
    if frequency is DataFrequency.TICK:
        return "tick"
    raise ValueError(f"unsupported_data_frequency: {frequency}")


def require_supported_data_frequency(
    value: DataFrequency | str,
    *,
    supported: tuple[DataFrequency, ...],
    context: str,
) -> DataFrequency:
    frequency = normalize_data_frequency(value)
    if frequency not in supported:
        raise ValueError(
            f"unsupported_data_frequency: {context}: {frequency.value}"
        )
    return frequency


@dataclass(frozen=True, kw_only=True)
class DataSpec:
    data_id: str
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
        column_roles = dict(self.column_roles)
        object.__setattr__(self, "column_roles", column_roles)
        missing = [
            column for column in self.primary_key if column not in column_roles
        ]
        if missing:
            raise ValueError(f"missing_primary_key_columns: {', '.join(missing)}")
        object.__setattr__(self, "required_columns", tuple(column_roles))


@dataclass(frozen=True, kw_only=True)
class InterimDataSpec(DataSpec):
    source_id: SourceId


@dataclass(frozen=True, kw_only=True)
class ProcessedDataSpec(DataSpec):
    market_id: MarketId
    kind: ProcessedDataKind
    source_input_data_ids: tuple[str, ...] = ()
    processed_input_data_ids: tuple[str, ...] = ()
