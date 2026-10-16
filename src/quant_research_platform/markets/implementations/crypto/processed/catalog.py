"""Crypto market-fact catalog for the approved Binance HFT recording."""

from __future__ import annotations

import json

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.columns import (
    PLATFORM_ASSET_TIME_KEY,
    ColumnRole,
)
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_CROSS_ASSET_DATA_ID,
    HFT_DERIVATIVE_DATA_ID,
    HFT_EXTERNAL_DATA_IDS,
    HFT_INFORMATION_PANEL_DATA_ID,
    HFT_INFORMATION_STATE_DATA_ID,
    HFT_SOL_PANEL_DATA_ID,
    HFT_SPOT_CONTEXT_DATA_ID,
    HFT_SPOT_PANEL_DATA_ID,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_CROSS_ASSET_CONTEXT_COLUMNS,
    HFT_CROSS_ASSET_FACTOR_COLUMNS,
    HFT_DERIVATIVE_CONTEXT_COLUMNS,
    HFT_DERIVATIVE_FACTOR_COLUMNS,
    HFT_EXTERNAL_CLOCK_FIELDS,
    HFT_EXTERNAL_FIELDS,
    HFT_FACTOR_COLUMNS,
    HFT_INFORMATION_COLUMNS,
    HFT_SPOT_CONTEXT_COLUMNS,
    HFT_SPOT_FACTOR_COLUMNS,
    label_catalog,
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
HFT_PANEL_COLUMNS = (*PLATFORM_ASSET_TIME_KEY, *HFT_PANEL_CONTEXT_COLUMNS, "eligible")


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
                (*PLATFORM_ASSET_TIME_KEY, *HFT_FACTOR_COLUMNS),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                semantic_role=ColumnRole.FACTOR,
                parameters={"asset_ids": ["btc_usdt", "eth_usdt"]},
            ),
            _spec(HFT_SOL_PANEL_DATA_ID, ProcessedDataKind.PANEL, HFT_PANEL_COLUMNS),
            _spec(
                HFT_INFORMATION_PANEL_DATA_ID,
                ProcessedDataKind.PANEL,
                HFT_PANEL_COLUMNS,
                processed_input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    HFT_SOL_PANEL_DATA_ID,
                ),
            ),
            _spec(
                HFT_INFORMATION_STATE_DATA_ID,
                ProcessedDataKind.PANEL,
                (*PLATFORM_ASSET_TIME_KEY, *HFT_INFORMATION_COLUMNS),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                parameters={"asset_ids": ["btc_usdt", "eth_usdt"]},
            ),
            _spec(
                HFT_CROSS_ASSET_DATA_ID,
                ProcessedDataKind.FACTOR,
                (
                    *PLATFORM_ASSET_TIME_KEY,
                    *HFT_CROSS_ASSET_FACTOR_COLUMNS,
                    *HFT_CROSS_ASSET_CONTEXT_COLUMNS,
                ),
                processed_input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    HFT_INFORMATION_STATE_DATA_ID,
                ),
                semantic_role=ColumnRole.FACTOR,
            ),
            _spec(
                HFT_SPOT_PANEL_DATA_ID,
                ProcessedDataKind.PANEL,
                HFT_PANEL_COLUMNS,
            ),
            _spec(
                HFT_SPOT_CONTEXT_DATA_ID,
                ProcessedDataKind.FACTOR,
                (
                    *PLATFORM_ASSET_TIME_KEY,
                    *HFT_SPOT_FACTOR_COLUMNS,
                    *HFT_SPOT_CONTEXT_COLUMNS,
                ),
                processed_input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    HFT_SPOT_PANEL_DATA_ID,
                ),
                semantic_role=ColumnRole.FACTOR,
            ),
            _spec(
                HFT_DERIVATIVE_DATA_ID,
                ProcessedDataKind.FACTOR,
                (
                    *PLATFORM_ASSET_TIME_KEY,
                    *HFT_DERIVATIVE_FACTOR_COLUMNS,
                    *HFT_DERIVATIVE_CONTEXT_COLUMNS,
                ),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                semantic_role=ColumnRole.FACTOR,
            ),
            *(
                _spec(
                    data_id,
                    ProcessedDataKind.FACTOR,
                    (
                        *PLATFORM_ASSET_TIME_KEY,
                        *(venue + "_" + field for field in HFT_EXTERNAL_FIELDS),
                        *(venue + "_" + field for field in HFT_EXTERNAL_CLOCK_FIELDS),
                    ),
                    processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                    semantic_role=ColumnRole.FACTOR,
                )
                for venue, data_id in HFT_EXTERNAL_DATA_IDS.items()
            ),
            _spec(
                "binance_futures_mid_labels_100ms",
                ProcessedDataKind.LABEL,
                (
                    *PLATFORM_ASSET_TIME_KEY,
                    *(item.output_column for item in label_catalog().list()),
                    *(item.available_at_column for item in label_catalog().list()),
                ),
                processed_input_data_ids=("binance_futures_l2_panel_100ms",),
                semantic_role=ColumnRole.LABEL,
                parameters={
                    "asset_ids": ["btc_usdt", "eth_usdt"],
                    "labels": [item.to_manifest() for item in label_catalog().list()],
                },
            ),
        )
    )


def _spec(
    data_id: str,
    kind: ProcessedDataKind,
    columns: tuple[str, ...],
    *,
    processed_input_data_ids: tuple[str, ...] = (),
    semantic_role: ColumnRole | None = None,
    parameters=None,
) -> ProcessedDataSpec:
    roles = {
        column: (
            ColumnRole.TIMESTAMP
            if column == "timestamp"
            else ColumnRole.KEY
            if column in PLATFORM_ASSET_TIME_KEY
            else ColumnRole.UNIVERSE
            if column == "eligible"
            else ColumnRole.CONTEXT
            if column
            in (
                *(item.available_at_column for item in label_catalog().list()),
                *HFT_DERIVATIVE_CONTEXT_COLUMNS,
                *HFT_SPOT_CONTEXT_COLUMNS,
                *HFT_CROSS_ASSET_CONTEXT_COLUMNS,
                *(
                    venue + "_" + field
                    for venue in HFT_EXTERNAL_DATA_IDS
                    for field in HFT_EXTERNAL_CLOCK_FIELDS
                ),
            )
            else semantic_role
            if semantic_role is not None
            else ColumnRole.CONTEXT
        )
        for column in columns
    }
    labels = [item for item in label_catalog().list() if item.output_column in columns]
    units = {
        "future_mid_return": "decimal_return",
        "future_mid_direction": "dimensionless",
        "maker_sell_pressure_proxy": "quantity_ratio",
    }
    return ProcessedDataSpec(
        column_units={item.output_column: units[item.family] for item in labels},
        label_availability_columns={
            item.output_column: item.available_at_column for item in labels
        },
        data_id=data_id,
        market_id="crypto",
        kind=kind,
        processed_input_data_ids=processed_input_data_ids,
        parameters_json=json.dumps({} if parameters is None else parameters),
        frequency=DataFrequency.TICK,
        grain=DataGrain.RESEARCH_ROW,
        column_roles=roles,
        primary_key=PLATFORM_ASSET_TIME_KEY,
        partition_columns=("year",),
    )
