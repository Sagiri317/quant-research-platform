"""Crypto market-fact catalog for the approved Binance HFT recording."""

from __future__ import annotations

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ENTITY_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_FACTOR_COLUMNS,
    HFT_LABEL_AVAILABILITY_COLUMNS,
    HFT_LABEL_COLUMNS,
)

HFT_PANEL_CONTEXT_COLUMNS = (
    "segment_id",
    "book_segment_id",
    "book_observation_id",
    "trade_observation_id",
    "book_source_sequence",
    "trade_source_sequence_min",
    "trade_source_sequence_max",
    "book_observed_time_ms",
    "exchange_event_time_ms",
    "transaction_time_ms",
    "observed_utc",
    "book_receive_monotonic_ns",
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "mid",
    "book_spread",
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
HFT_PANEL_COLUMNS = (*PLATFORM_ENTITY_TIME_KEY, *HFT_PANEL_CONTEXT_COLUMNS, "eligible")


def hft_processed_catalog() -> ProcessedDataCatalog:
    return ProcessedDataCatalog(
        (
            _spec(
                "binance_futures_l2_panel_100ms",
                ProcessedDataKind.PANEL,
                HFT_PANEL_COLUMNS,
            ),
            _spec(
                "binance_futures_microstructure_factors_100ms",
                ProcessedDataKind.FACTOR,
                (*PLATFORM_ENTITY_TIME_KEY, *HFT_FACTOR_COLUMNS),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                semantic_role=ColumnRole.FACTOR,
            ),
            _spec(
                "binance_futures_mid_labels_100ms",
                ProcessedDataKind.LABEL,
                (
                    *PLATFORM_ENTITY_TIME_KEY,
                    *HFT_LABEL_COLUMNS,
                    *HFT_LABEL_AVAILABILITY_COLUMNS,
                ),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                semantic_role=ColumnRole.LABEL,
            ),
        )
    )


def _spec(
    data_id: str,
    kind: ProcessedDataKind,
    columns: tuple[str, ...],
    *,
    source_input_data_ids: tuple[str, ...] = (),
    processed_input_data_ids: tuple[str, ...] = (),
    semantic_role: ColumnRole | None = None,
) -> ProcessedDataSpec:
    roles = {
        column: (
            ColumnRole.TIMESTAMP
            if column == "timestamp"
            else ColumnRole.KEY
            if column in PLATFORM_ENTITY_TIME_KEY
            else ColumnRole.UNIVERSE
            if column == "eligible"
            else ColumnRole.CONTEXT
            if column in HFT_LABEL_AVAILABILITY_COLUMNS
            else semantic_role
            if semantic_role is not None
            else ColumnRole.CONTEXT
        )
        for column in columns
    }
    return ProcessedDataSpec(
        data_id=data_id,
        market_id="crypto",
        kind=kind,
        source_input_data_ids=source_input_data_ids,
        processed_input_data_ids=processed_input_data_ids,
        frequency=DataFrequency.TICK,
        grain=DataGrain.RESEARCH_ROW,
        column_roles=roles,
        primary_key=PLATFORM_ENTITY_TIME_KEY,
        partition_columns=("year",),
    )
