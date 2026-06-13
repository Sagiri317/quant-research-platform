"""A-share processed data specs."""

from __future__ import annotations

from quant_research_platform.catalog.fields import platform_entity_time_key
from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.fields import (
    a_share_field_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)

PANEL_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "session_date",
    "available_at",
    "stock_name",
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "prev_close",
    "volume",
    "amount",
    "turnover",
    "return_1d",
    "limit_status",
    "is_limit_up",
    "is_limit_down",
    "is_trading",
    "is_st",
    "is_pt",
    "is_a_share_scope",
    "listing_date",
    "delisting_date",
    "company_status",
    "status_change_date",
    "listing_age_days",
    "listing_age_trading_days",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "share_change_date",
    "total_shares",
    "a_float_shares",
    "total_market_cap",
    "float_market_cap",
)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "market_type",
    "is_st",
    "is_pt",
    "limit_status",
    "listing_age_trading_days",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "total_market_cap",
    "float_market_cap",
    "amount",
    "turnover",
)

DEFAULT_UNIVERSE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible_base",
    "eligible_after_listing_age",
    "eligible_liquidity",
    "eligible",
)

AS_OF_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
)

EXECUTION_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "next_day_is_trading",
    "next_day_limit_status",
    "can_enter",
    "can_exit",
)

EVALUATION_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
    "can_exit",
)

def a_share_processed_catalog() -> ProcessedDataCatalog:
    """Register the first-stage A-share 0.1 processed data specs."""

    catalog = ProcessedDataCatalog()
    for spec in _processed_specs():
        catalog.register(spec)
    return catalog


def _processed_specs() -> tuple[ProcessedDataSpec, ...]:
    factor_columns = _registered_factor_columns()
    label_columns = _registered_label_columns()
    return (
        _processed_spec(
            kind=ProcessedDataKind.PANEL,
            data_id=DAILY_PANEL_PROCESSED_DATA_ID,
            columns=PANEL_COLUMNS,
            source_input_data_ids=(
                "daily_market",
                "stock_master",
                "market_cap",
                "trading_calendar",
            ),
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=DAILY_CONTEXT_PROCESSED_DATA_ID,
            columns=CONTEXT_COLUMNS,
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_DEFAULT_UNIVERSE_ID,
            columns=DEFAULT_UNIVERSE_COLUMNS,
            role_overrides={
                "eligible_base": ColumnRole.UNIVERSE,
                "eligible_after_listing_age": ColumnRole.UNIVERSE,
                "eligible_liquidity": ColumnRole.UNIVERSE,
                "eligible": ColumnRole.UNIVERSE,
            },
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_AS_OF_MASK_ID,
            columns=AS_OF_COLUMNS,
            role_overrides={"eligible": ColumnRole.MASK},
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_EXECUTION_MASK_ID,
            columns=EXECUTION_COLUMNS,
            role_overrides={
                "next_day_is_trading": ColumnRole.MASK,
                "can_enter": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.ANNOTATION,
            data_id=A_SHARE_EVALUATION_MASK_ID,
            columns=EVALUATION_COLUMNS,
            role_overrides={
                "eligible": ColumnRole.MASK,
                "can_exit": ColumnRole.MASK,
            },
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.FACTOR,
            data_id=PRICE_VOLUME_PROCESSED_DATA_ID,
            columns=(*platform_entity_time_key(), *factor_columns),
            role_overrides={column: ColumnRole.FACTOR for column in factor_columns},
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
        _processed_spec(
            kind=ProcessedDataKind.LABEL,
            data_id=FORWARD_RETURNS_PROCESSED_DATA_ID,
            columns=(*platform_entity_time_key(), *label_columns),
            role_overrides={column: ColumnRole.LABEL for column in label_columns},
            processed_input_data_ids=(DAILY_PANEL_PROCESSED_DATA_ID,),
        ),
    )


def _processed_spec(
    *,
    kind: ProcessedDataKind,
    data_id: str,
    columns: tuple[str, ...],
    role_overrides: dict[str, ColumnRole] | None = None,
    source_input_data_ids: tuple[str, ...] = (),
    processed_input_data_ids: tuple[str, ...] = (),
) -> ProcessedDataSpec:
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=A_SHARE_MARKET_ID,
        kind=kind,
        source_input_data_ids=source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=_column_roles(columns, role_overrides=role_overrides),
        primary_key=platform_entity_time_key(),
        partition_columns=("year",),
    )


def _column_roles(
    columns: tuple[str, ...],
    *,
    role_overrides: dict[str, ColumnRole] | None = None,
) -> dict[str, ColumnRole]:
    fields = a_share_field_catalog()
    overrides = role_overrides or {}
    roles: dict[str, ColumnRole] = {}
    for column in columns:
        if column in overrides:
            roles[column] = overrides[column]
        else:
            roles[column] = fields.get(column).role
    return roles


def _registered_factor_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in factor_definitions.a_share_factor_catalog().list()
    )


def _registered_label_columns() -> tuple[str, ...]:
    return tuple(
        definition.output_column
        for definition in sorted(
            label_definitions.a_share_label_catalog().list(),
            key=lambda item: item.horizon,
        )
    )


__all__ = [
    "AS_OF_COLUMNS",
    "CONTEXT_COLUMNS",
    "DEFAULT_UNIVERSE_COLUMNS",
    "EVALUATION_COLUMNS",
    "EXECUTION_COLUMNS",
    "PANEL_COLUMNS",
    "a_share_processed_catalog",
]
