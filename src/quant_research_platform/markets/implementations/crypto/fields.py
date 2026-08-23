"""Crypto market field catalog."""

from __future__ import annotations

from quant_research_platform.catalog.fields import FieldCatalog, platform_field_catalog
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec
from quant_research_platform.markets.implementations.crypto.labels.definitions import (
    CRYPTO_FORWARD_RETURN_COLUMNS,
)
from quant_research_platform.research.factors.microstructure.columns import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
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


def field_catalog() -> FieldCatalog:
    return FieldCatalog(
        (
            *platform_field_catalog().list(),
            *PANEL_FIELD_SPECS,
            *(
                FieldSpec(name=column, role=ColumnRole.FACTOR, dtype="float64")
                for column in MICROSTRUCTURE_FACTOR_COLUMNS
            ),
            *(
                FieldSpec(name=column, role=ColumnRole.LABEL, dtype="float64")
                for column in CRYPTO_FORWARD_RETURN_COLUMNS
            ),
        )
    )
