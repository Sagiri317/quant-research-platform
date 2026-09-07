"""Crypto market field catalog."""

from __future__ import annotations

from quant_research_platform.catalog.fields import FieldCatalog, platform_field_catalog
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
)
from quant_research_platform.research.protocols.crypto_hft import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
    HFT_FACTOR_COLUMNS,
    HFT_LABEL_AVAILABILITY_COLUMNS,
    HFT_LABEL_COLUMNS,
)

PANEL_FIELD_SPECS = (
    FieldSpec(name="open", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="high", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="low", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="quote_volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="trade_count", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(name="trade_flow_observed", role=ColumnRole.CONTEXT, dtype="bool"),
    FieldSpec(name="taker_buy_volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="taker_sell_volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="signed_volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="best_bid", role=ColumnRole.CONTEXT, dtype="float64", nullable=True),
    FieldSpec(name="best_ask", role=ColumnRole.CONTEXT, dtype="float64", nullable=True),
    FieldSpec(name="quote_observed", role=ColumnRole.CONTEXT, dtype="bool"),
    FieldSpec(
        name="quote_age_seconds",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(
        name="bid_quantity",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(
        name="ask_quantity",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(
        name="bid_depth",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(
        name="ask_depth",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(
        name="mid_price",
        role=ColumnRole.CONTEXT,
        dtype="float64",
        nullable=True,
    ),
    FieldSpec(name="eligible", role=ColumnRole.UNIVERSE, dtype="bool"),
)

HFT_PANEL_FIELD_SPECS = (
    FieldSpec(name="segment_id", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="book_segment_id", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="book_connection_id", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(
        name="trade_connection_id",
        role=ColumnRole.CONTEXT,
        dtype="string",
    ),
    FieldSpec(name="book_capture_seq", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(
        name="trade_capture_seq_min",
        role=ColumnRole.CONTEXT,
        dtype="int64",
        nullable=True,
    ),
    FieldSpec(
        name="trade_capture_seq_max",
        role=ColumnRole.CONTEXT,
        dtype="int64",
        nullable=True,
    ),
    FieldSpec(name="book_observed_time_ms", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(name="exchange_event_time_ms", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(name="transaction_time_ms", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(
        name="local_receive_time",
        role=ColumnRole.CONTEXT,
        dtype="datetime64[ns]",
    ),
    FieldSpec(
        name="book_receive_monotonic_ns",
        role=ColumnRole.CONTEXT,
        dtype="int64",
    ),
    FieldSpec(name="update_id", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(name="best_bid_price", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="best_bid_qty", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="best_ask_price", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="best_ask_qty", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="mid", role=ColumnRole.CONTEXT, dtype="float64"),
    *(
        FieldSpec(
            name=f"{side}_depth_{levels}",
            role=ColumnRole.CONTEXT,
            dtype="float64",
        )
        for levels in (1, 5, 10)
        for side in ("bid", "ask")
    ),
    FieldSpec(name="book_ofi_100ms", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="trade_count_100ms", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(name="trade_volume_100ms", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(
        name="observed_signed_trade_volume_100ms",
        role=ColumnRole.CONTEXT,
        dtype="float64",
    ),
    FieldSpec(
        name="observed_taker_buy_volume_100ms",
        role=ColumnRole.CONTEXT,
        dtype="float64",
    ),
    FieldSpec(
        name="observed_taker_sell_volume_100ms",
        role=ColumnRole.CONTEXT,
        dtype="float64",
    ),
)


def field_catalog() -> FieldCatalog:
    return FieldCatalog(
        (
            *platform_field_catalog().list(),
            *PANEL_FIELD_SPECS,
            *HFT_PANEL_FIELD_SPECS,
            *(
                FieldSpec(name=column, role=ColumnRole.FACTOR, dtype="float64")
                for column in MICROSTRUCTURE_FACTOR_COLUMNS
            ),
            *(
                FieldSpec(name=column, role=ColumnRole.FACTOR, dtype="float64")
                for column in HFT_FACTOR_COLUMNS
                if column not in MICROSTRUCTURE_FACTOR_COLUMNS
            ),
            *(
                FieldSpec(name=column, role=ColumnRole.LABEL, dtype="float64")
                for column in CRYPTO_FORWARD_RETURN_COLUMNS
            ),
            *(
                FieldSpec(name=column, role=ColumnRole.CONTEXT, dtype="datetime64[ns]")
                for column in HFT_LABEL_AVAILABILITY_COLUMNS
            ),
            *(
                FieldSpec(name=column, role=ColumnRole.LABEL, dtype="float64")
                for column in HFT_LABEL_COLUMNS
            ),
        )
    )
