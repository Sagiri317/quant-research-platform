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
    HFT_FACTOR_COLUMNS,
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_COLUMNS,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
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

HFT_PRIMARY_KEY = platform_entity_time_key()
HFT_PANEL_CONTEXT_COLUMNS = (
    "segment_id",
    "book_segment_id",
    "book_connection_id",
    "trade_connection_id",
    "book_capture_seq",
    "trade_capture_seq_min",
    "trade_capture_seq_max",
    "book_observed_time_ms",
    "exchange_event_time_ms",
    "transaction_time_ms",
    "local_receive_time",
    "book_receive_monotonic_ns",
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid",
    "bid_depth_1",
    "ask_depth_1",
    "bid_depth_5",
    "ask_depth_5",
    "bid_depth_10",
    "ask_depth_10",
    "book_ofi_100ms",
    "trade_count_100ms",
    "trade_volume_100ms",
    "observed_signed_trade_volume_100ms",
    "observed_taker_buy_volume_100ms",
    "observed_taker_sell_volume_100ms",
)
HFT_PANEL_COLUMNS = (*HFT_PRIMARY_KEY, *HFT_PANEL_CONTEXT_COLUMNS, "eligible")


def processed_catalog() -> ProcessedDataCatalog:
    return ProcessedDataCatalog(
        (
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
                    column: ColumnRole.FACTOR
                    for column in MICROSTRUCTURE_FACTOR_COLUMNS
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
        )
    )


def hft_processed_catalog() -> ProcessedDataCatalog:
    return ProcessedDataCatalog(
        (
            _hft_spec(
                data_id=HFT_PANEL_DATA_ID,
                kind=ProcessedDataKind.PANEL,
                columns=HFT_PANEL_COLUMNS,
                source_input_data_ids=(source_ids.RAW_RECORDER_RUN_DATA_ID,),
            ),
            _hft_spec(
                data_id=HFT_FACTOR_DATA_ID,
                kind=ProcessedDataKind.FACTOR,
                columns=(*HFT_PRIMARY_KEY, *HFT_FACTOR_COLUMNS),
                processed_input_data_ids=(HFT_PANEL_DATA_ID,),
                semantic_role=ColumnRole.FACTOR,
            ),
            _hft_spec(
                data_id=HFT_LABEL_DATA_ID,
                kind=ProcessedDataKind.LABEL,
                columns=(*HFT_PRIMARY_KEY, *HFT_LABEL_COLUMNS),
                processed_input_data_ids=(HFT_PANEL_DATA_ID,),
                semantic_role=ColumnRole.LABEL,
            ),
        )
    )


def _hft_spec(
    *,
    data_id: str,
    kind: ProcessedDataKind,
    columns: tuple[str, ...],
    source_input_data_ids: tuple[str, ...] = (),
    processed_input_data_ids: tuple[str, ...] = (),
    semantic_role: ColumnRole | None = None,
) -> ProcessedDataSpec:
    roles = {
        column: (
            ColumnRole.TIMESTAMP
            if column == "timestamp"
            else ColumnRole.KEY
            if column in HFT_PRIMARY_KEY
            else ColumnRole.UNIVERSE
            if column == "eligible"
            else semantic_role or ColumnRole.CONTEXT
        )
        for column in columns
    }
    return ProcessedDataSpec(
        data_id=data_id,
        market_id=MARKET_ID,
        kind=kind,
        source_input_data_ids=source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        frequency=DataFrequency.TICK,
        grain=DataGrain.RESEARCH_ROW,
        column_roles=roles,
        primary_key=HFT_PRIMARY_KEY,
        partition_columns=("year",),
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
