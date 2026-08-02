"""Crypto processed data specs."""

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
from quant_research_platform.markets.implementations.crypto import fields
from quant_research_platform.markets.implementations.crypto.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
    MICROSTRUCTURE_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
)
from quant_research_platform.markets.implementations.crypto.processed import source_ids
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)

PANEL_COLUMNS = (
    *platform_entity_time_key(),
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "trade_flow_observed",
    "taker_buy_volume",
    "taker_sell_volume",
    "signed_volume",
    "best_bid",
    "best_ask",
    "quote_observed",
    "quote_age_seconds",
    "bid_quantity",
    "ask_quantity",
    "bid_depth",
    "ask_depth",
    "mid_price",
    "eligible",
)


def processed_catalog() -> ProcessedDataCatalog:
    catalog = ProcessedDataCatalog()
    for spec in (
        _processed_spec(
            kind=ProcessedDataKind.PANEL,
            data_id=MARKET_PANEL_DATA_ID,
            columns=PANEL_COLUMNS,
            source_input_data_ids=(
                source_ids.MARKET_BARS_SOURCE_DATA_ID,
                source_ids.AGGREGATE_TRADES_SOURCE_DATA_ID,
                source_ids.DEPTH_LEVELS_SOURCE_DATA_ID,
            ),
            role_overrides={"eligible": ColumnRole.UNIVERSE},
        ),
        _processed_spec(
            kind=ProcessedDataKind.FACTOR,
            data_id=MICROSTRUCTURE_PROCESSED_DATA_ID,
            columns=(*platform_entity_time_key(), *MICROSTRUCTURE_FACTOR_COLUMNS),
            processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
            role_overrides={
                column: ColumnRole.FACTOR for column in MICROSTRUCTURE_FACTOR_COLUMNS
            },
        ),
        _processed_spec(
            kind=ProcessedDataKind.LABEL,
            data_id=FORWARD_RETURNS_PROCESSED_DATA_ID,
            columns=(*platform_entity_time_key(), *CRYPTO_FORWARD_RETURN_COLUMNS),
            processed_input_data_ids=(MARKET_PANEL_DATA_ID,),
            role_overrides={
                column: ColumnRole.LABEL for column in CRYPTO_FORWARD_RETURN_COLUMNS
            },
        ),
    ):
        catalog.register(spec)
    return catalog


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
        market_id=MARKET_ID,
        kind=kind,
        source_input_data_ids=source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        frequency=DataFrequency.MINUTE,
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
    catalog = fields.field_catalog()
    overrides = role_overrides or {}
    return {
        column: overrides[column] if column in overrides else catalog.get(column).role
        for column in columns
    }


__all__ = ["PANEL_COLUMNS", "processed_catalog"]
