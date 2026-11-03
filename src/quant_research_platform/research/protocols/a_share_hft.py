"""A-share snapshot research definitions; timestamps are vendor event times."""

import json

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.labels import LabelDefinition

KEY = ("market_id", "asset_id", "timestamp")
DATA_IDS = ("hft_panel", "hft_factors", "hft_labels")
BOOK_COLUMNS = tuple(
    f"{side}_{kind}{level}"
    for side in ("bid", "ask")
    for kind in ("px", "vol")
    for level in range(1, 11)
)
FLOW_COLUMNS = (
    "observed_trade_volume",
    "observed_signed_trade_volume",
    "observed_order_volume",
    "observed_signed_order_volume",
    "observed_cancel_volume",
    "observed_signed_cancel_volume",
)
PANEL_COLUMNS = (
    *KEY,
    "session_date",
    "exchange",
    "segment_id",
    "eligible",
    "last_price",
    "mid",
    *BOOK_COLUMNS,
    *FLOW_COLUMNS,
)
FACTOR_INPUTS = {
    "spread_cny": ("bid_px1", "ask_px1"),
    "relative_spread": ("bid_px1", "ask_px1", "mid"),
    **{
        f"imbalance_{level}": tuple(
            f"{side}_{kind}{i}"
            for side in ("bid", "ask")
            for kind in ("px", "vol")
            for i in range(1, level + 1)
        )
        for level in (1, 5, 10)
    },
    "microprice_deviation": ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1", "mid"),
    "book_ofi": ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1"),
    "depth_normalized_ofi": ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1"),
    **{f"return_{seconds}s": ("mid",) for seconds in (3, 30, 60)},
    "realized_volatility_60s": ("mid",),
    **{
        f"observed_{kind}_flow_imbalance": (
            f"observed_{kind}_volume",
            f"observed_signed_{kind}_volume",
        )
        for kind in ("trade", "order", "cancel")
    },
    "observed_trade_volume_30s": ("observed_trade_volume",),
    "observed_signed_trade_volume_30s": ("observed_signed_trade_volume",),
    "book_ofi_30s": ("bid_px1", "ask_px1", "bid_vol1", "ask_vol1"),
}


def factor_catalog():
    return FactorCatalog(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            family="level2_microstructure",
            input_columns=inputs,
            output_column=name,
            required_input_data_id="hft_panel",
            timing_semantics="observed_at_or_before_vendor_quote_time_same_segment",
        )
        for name, inputs in FACTOR_INPUTS.items()
    )


def label_catalog():
    return LabelCatalog(
        LabelDefinition(
            label_id=f"future_mid_{kind}_{seconds}s",
            market_id="a_share",
            family=f"future_mid_{kind}",
            input_columns=("mid", "segment_id"),
            output_column=f"future_mid_{kind}_{seconds}s",
            label_timing="actual_future_quote_time",
            horizon=seconds * 1000,
            horizon_unit="milliseconds",
            lookahead=seconds * 1000,
            entry_rule="observed_decision_mid",
            exit_rule="first_quote_at_or_after_horizon_in_same_continuous_segment",
        )
        for seconds in (3, 30, 60)
        for kind in ("return", "direction")
    )


def _roles(columns, *, factors=(), labels=()):
    return {
        name: "timestamp"
        if name == "timestamp"
        else "key"
        if name in KEY
        else "factor"
        if name in factors
        else "label"
        if name in labels
        else "universe"
        if name == "eligible"
        else "context"
        for name in columns
    }


def processed_specs():
    labels = label_catalog().list()
    label_columns = tuple(item.output_column for item in labels)
    maturity = {item.output_column: item.available_at_column for item in labels}
    definitions = (
        ("hft_panel", "panel", PANEL_COLUMNS, (), ()),
        ("hft_factors", "factor", (*KEY, *FACTOR_INPUTS), tuple(FACTOR_INPUTS), ()),
        (
            "hft_labels",
            "label",
            (*KEY, *label_columns, *maturity.values()),
            (),
            label_columns,
        ),
    )
    return tuple(
        ProcessedDataSpec(
            data_id=name,
            market_id="a_share",
            kind=kind,
            frequency="tick",
            grain="asset_timestamp",
            primary_key=KEY,
            partition_columns=("year",),
            column_roles=_roles(columns, factors=factors, labels=selected_labels),
            column_units={
                column: "decimal_return"
                if "return" in column
                else "direction"
                if "direction" in column
                else "CNY"
                if column in {"mid", "last_price", "spread_cny"} or "_px" in column
                else "shares"
                if "volume" in column
                or column.startswith(("bid_vol", "ask_vol"))
                or column in {"book_ofi", "book_ofi_30s"}
                else "dimensionless"
                for column in columns
                if column in factors
                or column in selected_labels
                or column in (*BOOK_COLUMNS, *FLOW_COLUMNS, "mid", "last_price")
            },
            label_availability_columns=maturity if kind == "label" else {},
            source_input_data_ids=("hft_quotes", "hft_orders", "hft_trades")
            if kind == "panel"
            else (),
            processed_input_data_ids=() if kind == "panel" else ("hft_panel",),
            parameters_json=json.dumps({"max_gap_seconds": 6})
            if kind == "panel"
            else "{}",
        )
        for name, kind, columns, factors, selected_labels in definitions
    )


def research_dataset_catalog():
    labels = label_catalog().list()
    label_columns = tuple(item.output_column for item in labels)
    maturity = {item.output_column: item.available_at_column for item in labels}
    context = ("session_date", "exchange", "segment_id", "mid", *maturity.values())
    columns = (*KEY, *FACTOR_INPUTS, *label_columns, "eligible", *context)
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id="a_share_hft",
                market_id="a_share",
                input_data_ids=DATA_IDS,
                base_input_data_id="hft_panel",
                factor_columns=tuple(FACTOR_INPUTS),
                label_columns=label_columns,
                label_availability_columns=maturity,
                filter_columns=("eligible",),
                context_columns=context,
                frequency="tick",
                grain="research_row",
                column_roles=_roles(
                    columns, factors=FACTOR_INPUTS, labels=label_columns
                ),
                primary_key=KEY,
                partition_columns=("year",),
            ),
        )
    )
