"""A-share snapshot research definitions; timestamps are vendor event times."""

import json

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.research.factors.intraday_path import (
    DESCRIPTION as PATH_DESCRIPTION,
)
from quant_research_platform.research.factors.intraday_path import (
    INPUTS as PATH_INPUTS,
)
from quant_research_platform.research.factors.intraday_path import (
    UNITS as PATH_UNITS,
)

KEY = ("market_id", "asset_id", "timestamp")
DATA_IDS = (
    "hft_panel",
    "hft_factors",
    "hft_labels",
    "hft_event_factors",
    "hft_hawkes_factors",
    "hft_research_factors",
)
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
    "instrument_type",
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
DEFAULT_FACTORS = tuple(FACTOR_INPUTS)
FACTOR_INPUTS.update(
    {
        **{
            f"{prefix}_{level}": tuple(
                f"{side}_{kind}{i}"
                for side in ("bid", "ask")
                for kind in ("px", "vol")
                for i in range(1, level + 1)
            )
            for prefix in ("book_ofi", "depth_normalized_ofi")
            for level in (5, 10)
        },
        "book_ofi_near_far": BOOK_COLUMNS,
        "book_ofi_10_30s": BOOK_COLUMNS,
        "book_ofi_10_ewma": BOOK_COLUMNS,
        **{
            f"{side}_{metric}": (
                *(name for name in BOOK_COLUMNS if name.startswith(side + "_")),
                *(("mid",) if "slope" in metric else ()),
            )
            for side in ("bid", "ask")
            for metric in (
                "depth_concentration",
                "near_depth_fraction",
                "depth_slope",
                "depth_slope_change",
                "replenishment_30s",
            )
        },
        "depth_slope_difference": (*BOOK_COLUMNS, "mid"),
    }
)
INTRADAY_THEME_FACTORS = {
    **dict.fromkeys(
        (
            "price_position_300s",
            "peak_drawdown_300s",
            "support_distance_300s",
            "resistance_distance_300s",
            "volatility_compression_30_300s",
            "signed_trend_efficiency_300s",
            "failed_high_break_300s",
            "failed_low_break_300s",
        ),
        ("mid",),
    ),
    "trade_activity_ratio_30_300s": ("observed_trade_volume",),
    "flow_absorption_30s": (
        "mid",
        "observed_trade_volume",
        "observed_signed_trade_volume",
    ),
    "drawdown_sell_pressure_300s": (
        "mid",
        "observed_trade_volume",
        "observed_signed_trade_volume",
    ),
    "compression_order_flow_300s": (
        "mid",
        "bid_px1",
        "ask_px1",
        "bid_vol1",
        "ask_vol1",
    ),
}
FACTOR_INPUTS.update(INTRADAY_THEME_FACTORS)
FACTOR_INPUTS.update(PATH_INPUTS)

INTRADAY_THEME_DESCRIPTIONS = {
    "price_position_300s": (
        "(mid-prior300s low)/(prior300s high-low); flat range undefined"
    ),
    "peak_drawdown_300s": "max(1-mid/prior300s high,0)",
    "support_distance_300s": "mid/prior300s low-1",
    "resistance_distance_300s": "mid/prior300s high-1",
    "volatility_compression_30_300s": (
        "sqrt(10*sum(log-return squared,30s)/sum(log-return squared,300s))"
    ),
    "signed_trend_efficiency_300s": "sum(log-return,300s)/sum(abs(log-return),300s)",
    "trade_activity_ratio_30_300s": (
        "10*recorded trade volume30s/recorded trade volume300s"
    ),
    "flow_absorption_30s": (
        "signed trade-volume imbalance30s*"
        "(1-abs(return30s)/sqrt(realized variance300s/10))"
    ),
    "drawdown_sell_pressure_300s": (
        "peak_drawdown300s*max(-signed trade-volume imbalance30s,0)"
    ),
    "compression_order_flow_300s": (
        "max(1-volatility compression,0)*OFI30s/current best-level depth"
    ),
    "failed_high_break_300s": (
        "positive recent30s maximum excess over the prior300s high known at t-30s, "
        "only if current mid returned below that level"
    ),
    "failed_low_break_300s": (
        "positive recent30s minimum shortfall below the prior300s low known at t-30s, "
        "only if current mid returned above that level"
    ),
}
EVENT_FACTORS = {
    "trade_interval_seconds": "seconds",
    "trade_interval_mean_seconds": "seconds",
    "trade_interval_cv": "dimensionless",
    "trade_intensity": "events_per_second",
    "trade_activity_ratio": "dimensionless",
    "trade_signed_run": "events",
    "trade_signed_run_volume": "shares",
    "trade_event_flow_imbalance": "dimensionless",
    "trade_volume_flow_imbalance": "dimensionless",
    "large_trade_pressure": "dimensionless",
    "small_trade_pressure": "dimensionless",
    "trade_interval_return_interaction": "decimal_return_times_seconds",
    "observed_cancel_order_ratio": "dimensionless",
    "observed_cancel_count_fraction": "dimensionless",
    "observed_cancel_pressure_change": "dimensionless",
    "observed_order_age_seconds": "seconds",
    "observed_order_age_side_difference": "seconds",
    "observed_cancel_age_seconds": "seconds",
    "observed_age_weighted_cancel_pressure": "shares_times_seconds",
    "observed_completed_lifetime_seconds": "seconds",
    "observed_lifecycle_match_fraction": "dimensionless",
}
CANCEL_EVENT_FACTORS = frozenset(
    {
        "observed_cancel_order_ratio",
        "observed_cancel_count_fraction",
        "observed_cancel_pressure_change",
    }
)
TRADE_EVENT_FACTORS = frozenset(
    name
    for name in EVENT_FACTORS
    if name.startswith("trade_")
    or name in {"large_trade_pressure", "small_trade_pressure"}
)
LIFECYCLE_FACTORS = EVENT_FACTORS.keys() - TRADE_EVENT_FACTORS - CANCEL_EVENT_FACTORS


TRADE_INTERVAL_FACTORS = frozenset(
    {
        "trade_interval_seconds",
        "trade_interval_mean_seconds",
        "trade_interval_cv",
        "trade_activity_ratio",
        "trade_interval_return_interaction",
    }
)
EVENT_KEY_FIELDS = ("asset_id", "timestamp", "source_row", "event_kind")
EVENT_INPUTS = {
    **{
        name: {
            "hft_trades": ("price",)
            if name == "trade_interval_return_interaction"
            else ()
            if name in TRADE_INTERVAL_FACTORS or name == "trade_intensity"
            else ("side", "volume")
        }
        for name in TRADE_EVENT_FACTORS
    },
    **{
        name: dict.fromkeys(
            ("hft_orders", "hft_trades"),
            () if name == "observed_cancel_count_fraction" else ("volume",),
        )
        for name in CANCEL_EVENT_FACTORS
    },
    **dict.fromkeys(
        LIFECYCLE_FACTORS,
        {
            "hft_orders": (
                "exchange",
                "order_type",
                "side",
                "volume",
                "order_id",
                "ex_order_id",
            ),
            "hft_trades": ("volume", "bid_order_id", "ask_order_id"),
        },
    ),
}


def event_factor_streams(column, exchange):
    if column in CANCEL_EVENT_FACTORS and exchange == "SSE":
        return {"hft_orders"}
    return set(EVENT_INPUTS[column])


def event_input_columns(columns, *, order_link_field):
    """Project recorded event streams for the selected factors and link contract."""
    required = {}
    for column in columns:
        if column in LIFECYCLE_FACTORS and order_link_field is None:
            continue
        for stream, fields in EVENT_INPUTS[column].items():
            required.setdefault(stream, set(EVENT_KEY_FIELDS)).update(fields)
    if "hft_orders" in required:
        required["hft_orders"].difference_update(("order_id", "ex_order_id"))
        if order_link_field is not None and set(columns) & LIFECYCLE_FACTORS:
            required["hft_orders"].add(order_link_field)
    return {stream: tuple(sorted(fields)) for stream, fields in required.items()}


HAWKES_FACTORS = {
    "hawkes_buy_intensity": "events_per_second",
    "hawkes_sell_intensity": "events_per_second",
    "hawkes_intensity_imbalance": "dimensionless",
    "hawkes_excitation_fraction": "dimensionless",
}


def factor_catalog():
    from quant_research_platform.research.factors.intraday_context import definitions

    return FactorCatalog(
        (
            *definitions(),
            *(
                FactorDefinition(
                    factor_id=name,
                    market_id="a_share",
                    factor_set="standard",
                    family="level2_microstructure",
                    input_columns=inputs,
                    output_column=name,
                    required_input_data_id="hft_hawkes_factors"
                    if name in HAWKES_FACTORS
                    else "hft_event_factors"
                    if name in EVENT_FACTORS
                    else "hft_panel",
                    timing_semantics="observed_at_or_before_vendor_quote_time_same_segment",
                    description=PATH_DESCRIPTION
                    if name in PATH_INPUTS
                    else INTRADAY_THEME_DESCRIPTIONS.get(name, ""),
                )
                for name, inputs in {
                    **FACTOR_INPUTS,
                    **{
                        name: tuple(
                            dict.fromkeys(
                                (
                                    *EVENT_KEY_FIELDS,
                                    *(
                                        field
                                        for fields in EVENT_INPUTS[name].values()
                                        for field in fields
                                    ),
                                )
                            )
                        )
                        for name in EVENT_FACTORS
                    },
                    **dict.fromkeys(HAWKES_FACTORS, ("timestamp", "side")),
                }.items()
            ),
        )
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
        else "feature"
        if name in BOOK_COLUMNS
        else "label"
        if name in labels
        else "context"
        for name in columns
    }


def processed_specs():
    from ...markets.implementations.a_share.processed.intraday import (
        processed_spec,
    )

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
    specs = tuple(
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
                column: PATH_UNITS[column]
                if column in PATH_UNITS
                else "log_return"
                if column == "realized_volatility_60s"
                else "decimal_return"
                if "return" in column or column == "peak_drawdown_300s"
                else "direction"
                if "direction" in column
                else "CNY"
                if column in {"mid", "last_price", "spread_cny"} or "_px" in column
                else "shares"
                if "volume" in column
                or column.startswith(("bid_vol", "ask_vol"))
                or column
                in {
                    "book_ofi",
                    "book_ofi_30s",
                    "book_ofi_5",
                    "book_ofi_10",
                    "book_ofi_near_far",
                    "book_ofi_10_30s",
                    "book_ofi_10_ewma",
                }
                or "replenishment" in column
                else "per_bp"
                if "depth_slope" in column
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
            optional_source_input_data_ids=("hft_orders", "hft_trades")
            if kind == "panel"
            else (),
            processed_input_data_ids=() if kind == "panel" else ("hft_panel",),
            parameters_json=json.dumps({"max_gap_seconds": 6})
            if kind == "panel"
            else "{}",
        )
        for name, kind, columns, factors, selected_labels in definitions
    )
    return (
        processed_spec(),
        *specs,
        ProcessedDataSpec(
            data_id="hft_event_factors",
            market_id="a_share",
            kind="factor",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=KEY,
            partition_columns=("year",),
            column_roles=_roles((*KEY, *EVENT_FACTORS), factors=EVENT_FACTORS),
            column_units=EVENT_FACTORS,
            source_input_data_ids=("hft_orders", "hft_trades"),
            optional_source_input_data_ids=("hft_orders", "hft_trades"),
            processed_input_data_ids=("hft_panel",),
            parameters_json=json.dumps(
                {
                    "event_window": 50,
                    "volume_window": 10000,
                    "history_window": 200,
                    "large_quantile": 0.9,
                    "order_link_field": None,
                }
            ),
        ),
        ProcessedDataSpec(
            data_id="hft_hawkes_factors",
            market_id="a_share",
            kind="factor",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=KEY,
            partition_columns=("year",),
            column_roles=_roles((*KEY, *HAWKES_FACTORS), factors=HAWKES_FACTORS),
            column_units=HAWKES_FACTORS,
            source_input_data_ids=("hft_trades",),
            processed_input_data_ids=("hft_panel",),
        ),
    )


def research_dataset_catalog():
    labels = label_catalog().list()
    label_columns = tuple(item.output_column for item in labels)
    maturity = {item.output_column: item.available_at_column for item in labels}
    context = (
        "session_date",
        "exchange",
        "instrument_type",
        "segment_id",
        "mid",
        *maturity.values(),
    )
    columns = (*KEY, *DEFAULT_FACTORS, *label_columns, "eligible", *context)
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id="a_share_hft",
                market_id="a_share",
                input_data_ids=("hft_panel", "hft_factors", "hft_labels"),
                base_input_data_id="hft_panel",
                factor_columns=DEFAULT_FACTORS,
                label_columns=label_columns,
                label_availability_columns=maturity,
                filter_columns=("eligible",),
                context_columns=context,
                frequency="tick",
                grain="research_row",
                column_roles={
                    **_roles(columns, factors=FACTOR_INPUTS, labels=label_columns),
                    "eligible": "universe",
                },
                primary_key=KEY,
                partition_columns=("year",),
            ),
        )
    )
