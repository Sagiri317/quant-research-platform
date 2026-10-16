"""Processed data frequency and grain contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.ids import (
    MarketId,
    SourceId,
    validate_market_id,
    validate_source_id,
)
from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
    render_qrp_json_object,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
)


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
    if type(value) is not str:
        raise ValueError(f"invalid_data_frequency: {value}")
    try:
        return DataFrequency(value)
    except ValueError as exc:
        raise ValueError(f"invalid_data_frequency: {value}") from exc


def normalize_data_grain(value: DataGrain | str) -> DataGrain:
    if isinstance(value, DataGrain):
        return value
    if type(value) is not str:
        raise ValueError(f"invalid_data_grain: {value}")
    try:
        return DataGrain(value)
    except ValueError as exc:
        raise ValueError(f"invalid_data_grain: {value}") from exc


def normalize_processed_data_kind(value: ProcessedDataKind | str) -> ProcessedDataKind:
    if isinstance(value, ProcessedDataKind):
        return value
    if type(value) is not str:
        raise ValueError(f"invalid_processed_data_kind: {value}")
    try:
        return ProcessedDataKind(value)
    except ValueError as exc:
        raise ValueError(f"invalid_processed_data_kind: {value}") from exc


def data_frequency_slug(value: DataFrequency | str) -> str:
    return normalize_data_frequency(value).value


def require_supported_data_frequency(
    value: DataFrequency | str,
    *,
    supported: tuple[DataFrequency, ...],
    context: str,
) -> DataFrequency:
    frequency = normalize_data_frequency(value)
    if frequency not in supported:
        raise ValueError(f"unsupported_data_frequency: {context}: {frequency.value}")
    return frequency


@dataclass(frozen=True, kw_only=True)
class DataSpec:
    data_id: str
    frequency: DataFrequency | str
    grain: DataGrain | str
    column_roles: Mapping[str, ColumnRole | str]
    primary_key: tuple[str, ...]
    partition_columns: tuple[str, ...]
    column_units: Mapping[str, str] = field(default_factory=dict)
    label_availability_columns: Mapping[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.data_id, "data_id_required")
        frequency = normalize_data_frequency(self.frequency)
        grain = normalize_data_grain(self.grain)
        if not isinstance(self.column_roles, Mapping):
            raise ValueError("column_roles_required")
        column_roles: dict[str, ColumnRole] = {}
        try:
            for column, role in self.column_roles.items():
                if type(column) is not str or not column:
                    raise ValueError("column_roles_must_use_non_empty_string_keys")
                try:
                    column_roles[column] = (
                        role if type(role) is ColumnRole else ColumnRole(role)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("column_roles_must_use_supported_roles") from exc
        except AttributeError as exc:
            raise ValueError("column_roles_required") from exc
        if not column_roles:
            raise ValueError("column_roles_required")
        _validate_column_tuple(self.primary_key, "primary_key")
        _validate_column_tuple(
            self.partition_columns,
            "partition_columns",
            required=False,
        )
        if self.partition_columns not in ((), ("year",)):
            raise ValueError("unsupported_partitioning")
        missing = [column for column in self.primary_key if column not in column_roles]
        if missing:
            raise ValueError(f"missing_primary_key_columns: {', '.join(missing)}")
        invalid_key_roles = [
            column
            for column in self.primary_key
            if column_roles[column] not in {ColumnRole.KEY, ColumnRole.TIMESTAMP}
        ]
        if invalid_key_roles:
            raise ValueError(
                f"invalid_primary_key_roles: {', '.join(invalid_key_roles)}"
            )
        timestamps = [
            column
            for column in self.primary_key
            if column_roles[column] is ColumnRole.TIMESTAMP
        ]
        if len(timestamps) > 1 or (self.partition_columns and len(timestamps) != 1):
            raise ValueError("invalid_timestamp_primary_key")
        for name in ("column_units", "label_availability_columns"):
            values = getattr(self, name)
            if not isinstance(values, Mapping) or any(
                type(column) is not str
                or column not in column_roles
                or type(value) is not str
                or not value.strip()
                for column, value in values.items()
            ):
                raise ValueError(f"invalid_{name}")
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        for label, availability in self.label_availability_columns.items():
            if (
                column_roles[label] is not ColumnRole.LABEL
                or availability not in column_roles
                or column_roles[availability]
                not in {ColumnRole.TIMESTAMP, ColumnRole.CONTEXT, ColumnRole.LABEL}
            ):
                raise ValueError("invalid_label_availability_columns")
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "grain", grain)
        object.__setattr__(self, "column_roles", MappingProxyType(column_roles))
        object.__setattr__(self, "required_columns", tuple(column_roles))


@dataclass(frozen=True, kw_only=True)
class InterimDataSpec(DataSpec):
    source_id: SourceId

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "source_id", validate_source_id(self.source_id))


@dataclass(frozen=True, kw_only=True)
class ProcessedDataSpec(DataSpec):
    market_id: MarketId
    kind: ProcessedDataKind | str
    source_input_data_ids: tuple[str, ...] = ()
    optional_source_input_data_ids: tuple[str, ...] = ()
    processed_input_data_ids: tuple[str, ...] = ()
    parameters_json: str = "{}"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        object.__setattr__(self, "kind", normalize_processed_data_kind(self.kind))
        if type(self.parameters_json) is not str:
            raise ValueError("processed_parameters_invalid")
        object.__setattr__(
            self,
            "parameters_json",
            render_qrp_json_object(
                parse_qrp_json_object(self.parameters_json.encode("utf-8"))
            ),
        )
        for field_name, values in (
            ("source_input_data_ids", self.source_input_data_ids),
            ("optional_source_input_data_ids", self.optional_source_input_data_ids),
            ("processed_input_data_ids", self.processed_input_data_ids),
        ):
            _validate_string_tuple(values, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name}_must_be_unique")
        if not set(self.optional_source_input_data_ids) <= set(
            self.source_input_data_ids
        ):
            raise ValueError("optional_source_input_not_declared")


def _validate_string_tuple(values: object, field_name: str) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or not value for value in values
    ):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
