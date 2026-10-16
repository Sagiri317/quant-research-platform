"""Crypto HFT protocol constants and catalogs; market facts stay in crypto."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.json_codec import (
    qrp_json_object_snapshot,
)
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import (
    TimeWindow,
    is_datetime_window,
)
from quant_research_platform.markets.implementations.crypto import ids as market_ids

PROTOCOL_ID = "crypto_hft"
HFT_FACTOR_COLUMNS = (
    "spread",
    "relative_spread",
    "imbalance_1",
    "imbalance_5",
    "imbalance_10",
    "microprice",
    "microprice_deviation",
    "ofi_100ms",
    "signed_trade_flow_100ms",
    "taker_buy_volume_100ms",
    "taker_sell_volume_100ms",
    "return_100ms",
    "return_500ms",
    "realized_volatility_1s",
    "top10_depth_per_spread",
    "trade_flow_imbalance_100ms",
    "trade_flow_imbalance_500ms",
    "trade_flow_imbalance_1s",
    "trade_flow_imbalance_5s",
    "ofi_500ms",
    "ofi_1s",
    "ofi_5s",
    "depth_normalized_ofi_500ms",
    "depth_normalized_ofi_1s",
    "depth_normalized_ofi_5s",
    "imbalance_1_minus_10",
    "imbalance_5_minus_10",
    "near_depth_share_1_10",
    "taker_sell_pressure_500ms",
    "taker_buy_pressure_500ms",
)
HFT_DERIVATIVE_FACTOR_COLUMNS = (
    "mark_mid_basis",
    "mark_index_basis",
    "perp_index_basis",
    "funding_rate",
    "time_to_next_funding",
    "log_open_interest",
    "open_interest_change_30s",
    "open_interest_change_1m",
    "open_interest_change_5m",
    "observed_force_order_imbalance_5s",
    "observed_force_order_imbalance_30s",
    "time_since_observed_force_order",
    "top_position_long_short_ratio",
    "top_account_long_short_ratio",
    "global_account_long_short_ratio",
    "taker_long_short_ratio",
)
HFT_DERIVATIVE_CHANNELS = (
    "premiumIndex",
    "openInterest",
    "forceOrder",
    "topLongShortPositionRatio",
    "topLongShortAccountRatio",
    "globalLongShortAccountRatio",
    "takerlongshortRatio",
)
HFT_DERIVATIVE_CONTEXT_COLUMNS = tuple(
    f"{channel}_{field}"
    for channel in HFT_DERIVATIVE_CHANNELS
    for field in ("age_ms", "observed_at", "event_at")
)
HFT_SPOT_FACTOR_COLUMNS = (
    "spot_perp_basis",
    "spot_return_100ms",
    "spot_return_500ms",
    "spot_ofi_500ms",
    "spot_tfi_500ms",
    "spot_perp_return_divergence_100ms",
    "spot_perp_return_divergence_500ms",
    "spot_perp_normalized_ofi_divergence_500ms",
    "spot_perp_tfi_divergence_500ms",
    "lagged_spot_return_100ms",
    "lagged_spot_ofi_500ms",
    "lagged_spot_tfi_500ms",
)
HFT_SPOT_CONTEXT_COLUMNS = (
    "spot_grid_age_ms",
    "spot_book_age_ms",
    "spot_grid_at",
    "spot_book_observed_at",
)
HFT_INFORMATION_COLUMNS = (
    "return_100ms",
    "return_500ms",
    "depth_normalized_ofi_500ms",
    "trade_flow_imbalance_500ms",
)
HFT_CROSS_ASSET_FACTOR_COLUMNS = tuple(
    f"{asset}_{field}"
    for asset in ("btc", "eth", "sol")
    for field in (
        "return_100ms",
        "return_500ms",
        "normalized_ofi_500ms",
        "tfi_500ms",
        "lagged_return_100ms",
        "lagged_tfi_500ms",
    )
)
HFT_CROSS_ASSET_CONTEXT_COLUMNS = tuple(
    f"{asset}_{field}"
    for asset in ("btc", "eth", "sol")
    for field in ("grid_age_ms", "grid_at")
)

HFT_EXTERNAL_FIELDS = (
    "mid_usdt",
    "spread_usdt",
    "return_100ms",
    "return_500ms",
    "observed_signed_base_flow_500ms",
    "observed_signed_notional_flow_500ms",
    "observed_tfi_500ms",
    "price_difference",
    "relative_mid_dispersion",
)
HFT_EXTERNAL_CLOCK_FIELDS = tuple(
    f"{source}_{field}"
    for source in ("quote", "trade")
    for field in ("age_ms", "observed_at", "event_at")
)


def label_catalog():
    """Executable continuous-segment labels, including their maturity clock."""
    definitions = []
    for horizon in (100, 500, 1_000, 5_000):
        for kind in ("return", "direction"):
            column = f"future_mid_{kind}_{horizon}ms"
            definitions.append(
                LabelDefinition(
                    label_id=column,
                    market_id=market_ids.MARKET_ID,
                    family=f"future_mid_{kind}",
                    input_columns=("mid", "segment_id", "timestamp_ms"),
                    output_column=column,
                    label_timing="decision_time_plus_horizon_lte_role_end",
                    horizon=horizon,
                    horizon_unit="milliseconds",
                    lookahead=horizon,
                    entry_rule="decision_mid",
                    exit_rule="same_continuous_segment_fixed_horizon_mid",
                )
            )
    definitions.append(
        LabelDefinition(
            label_id="maker_bid_sell_pressure_proxy_500ms",
            market_id=market_ids.MARKET_ID,
            family="maker_sell_pressure_proxy",
            input_columns=(
                "best_bid_qty",
                "observed_taker_sell_volume_100ms",
                "segment_id",
                "timestamp_ms",
            ),
            output_column="maker_bid_sell_pressure_proxy_500ms",
            label_timing="decision_time_plus_horizon_lte_role_end",
            horizon=500,
            horizon_unit="milliseconds",
            lookahead=500,
            entry_rule="decision_best_bid_quantity",
            exit_rule="same_continuous_segment_future_observed_sell_volume",
        )
    )
    return LabelCatalog(definitions)


HFT_MAKER_BID_SELL_PRESSURE_PROXY_COLUMN = "maker_bid_sell_pressure_proxy_500ms"
HFT_EXECUTION_ASSET_IDS = ("btc_usdt", "eth_usdt")

HFT_EXECUTION_CONTEXT_COLUMNS = (
    "book_segment_id",
    "trade_observation_id",
    "segment_id",
)


def label_horizon_ms(label_column):
    """Read the horizon from the executable label definition."""
    return label_catalog().get(label_column).horizon


def research_chronology(value):
    """Decode the chronology shared by model artifacts and execution operations."""
    snapshot = qrp_json_object_snapshot(value)
    if set(snapshot) != {"folds", "test"}:
        raise ValueError("crypto_research_chronology_fields_invalid")

    def window(value):
        if type(value) is not dict or set(value) != {"start", "end"}:
            raise ValueError("crypto_research_plan_window_invalid")
        result = TimeWindow(value["start"], value["end"])
        if not is_datetime_window(result):
            raise ValueError("crypto_research_plan_requires_intraday_window")
        return result

    raw_folds = snapshot["folds"]
    if type(raw_folds) is not list or not raw_folds:
        raise ValueError("crypto_research_plan_folds_required")
    folds = []
    for item in raw_folds:
        if type(item) is not dict or set(item) != {
            "fold_id",
            "train",
            "validation",
        }:
            raise ValueError("crypto_research_plan_fold_invalid")
        fold = ExperimentFold(
            fold_id=item["fold_id"],
            train=window(item["train"]),
            validation=window(item["validation"]),
        )
        if folds and folds[-1].validation.end >= fold.validation.start:
            raise ValueError("crypto_research_plan_validation_windows_overlap")
        folds.append(fold)
    if len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError("crypto_research_plan_duplicate_fold")
    test_window = window(snapshot["test"])
    if any(fold.validation.end >= test_window.start for fold in folds):
        raise ValueError("crypto_research_plan_test_window_overlap")
    return tuple(folds), test_window


def build_hft_research_dataset_catalog() -> ResearchDatasetCatalog:
    context = (
        *tuple(item.available_at_column for item in label_catalog().list()),
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
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        *HFT_FACTOR_COLUMNS,
        *tuple(item.output_column for item in label_catalog().list()),
        "eligible",
        *context,
    )
    return ResearchDatasetCatalog(
        (
            ResearchDatasetSpec(
                data_id=market_ids.HFT_RESEARCH_DATASET_ID,
                market_id=market_ids.MARKET_ID,
                input_data_ids=(
                    "binance_futures_l2_panel_100ms",
                    "binance_futures_microstructure_factors_100ms",
                    "binance_futures_mid_labels_100ms",
                ),
                base_input_data_id="binance_futures_l2_panel_100ms",
                factor_columns=HFT_FACTOR_COLUMNS,
                label_columns=tuple(
                    item.output_column for item in label_catalog().list()
                ),
                filter_columns=("eligible",),
                context_columns=context,
                frequency=DataFrequency.TICK,
                grain=DataGrain.RESEARCH_ROW,
                column_roles=_roles(
                    columns,
                    HFT_FACTOR_COLUMNS,
                    tuple(item.output_column for item in label_catalog().list()),
                    ("eligible",),
                ),
                primary_key=("market_id", "asset_id", "timestamp"),
                partition_columns=("year",),
            ),
        )
    )


def _roles(
    columns: tuple[str, ...],
    factors: tuple[str, ...],
    labels: tuple[str, ...],
    filters: tuple[str, ...],
) -> dict[str, ColumnRole]:
    return {
        column: ColumnRole.FACTOR
        if column in factors
        else ColumnRole.LABEL
        if column in labels
        else ColumnRole.UNIVERSE
        if column in filters
        else ColumnRole.TIMESTAMP
        if column == "timestamp"
        else ColumnRole.KEY
        if column in {"market_id", "asset_id"}
        else ColumnRole.CONTEXT
        for column in columns
    }


def _microstructure_inputs(column):
    """Declare only the values used by each mathematical definition."""

    def depth(level):
        return (f"bid_depth_{level}", f"ask_depth_{level}")

    if column == "spread":
        return ("book_spread",)
    if column == "relative_spread":
        return ("book_spread", "mid")
    if column in {"microprice", "microprice_deviation"}:
        return (*depth(1), "book_spread", *(("mid",) if column == "microprice" else ()))
    if column.startswith("imbalance_"):
        return tuple(
            name
            for level in column.split("_")[1:]
            if level.isdigit()
            for name in depth(int(level))
        )
    if column in {"return_100ms", "return_500ms", "realized_volatility_1s"}:
        return ("mid",)
    if column == "top10_depth_per_spread":
        return (*depth(10), "book_spread")
    if column == "near_depth_share_1_10":
        return (*depth(1), *depth(10))
    if column.startswith("depth_normalized_ofi_"):
        return ("book_ofi_100ms", *depth(1))
    if column.startswith("ofi_"):
        return ("book_ofi_100ms",)
    if column.startswith("trade_flow_imbalance_"):
        return (
            "observed_signed_trade_volume_100ms",
            "observed_taker_buy_volume_100ms",
            "observed_taker_sell_volume_100ms",
        )
    if column == "signed_trade_flow_100ms":
        return ("observed_signed_trade_volume_100ms",)
    side = "buy" if "buy" in column else "sell"
    return (
        f"observed_taker_{side}_volume_100ms",
        *(
            (f"{'ask' if side == 'buy' else 'bid'}_depth_1",)
            if "pressure" in column
            else ()
        ),
    )


def derivative_channels(columns):
    """Resolve selected factor and context columns to native state channels."""
    channels = set()
    for column in columns:
        if column in {
            "mark_mid_basis",
            "mark_index_basis",
            "perp_index_basis",
            "funding_rate",
            "time_to_next_funding",
        }:
            channels.add("premiumIndex")
        elif "open_interest" in column:
            channels.add("openInterest")
        elif "force_order" in column:
            channels.add("forceOrder")
        elif column in {
            "top_position_long_short_ratio",
            "top_account_long_short_ratio",
            "global_account_long_short_ratio",
            "taker_long_short_ratio",
        }:
            channels.add(
                {
                    "top_position_long_short_ratio": "topLongShortPositionRatio",
                    "top_account_long_short_ratio": "topLongShortAccountRatio",
                    "global_account_long_short_ratio": "globalLongShortAccountRatio",
                    "taker_long_short_ratio": "takerlongshortRatio",
                }[column]
            )
        else:
            channels.update(
                channel
                for channel in HFT_DERIVATIVE_CHANNELS
                if column.startswith(channel + "_")
            )
    return tuple(channel for channel in HFT_DERIVATIVE_CHANNELS if channel in channels)


def spot_dependencies(columns, *, perpetual=False):
    """Resolve each side of a spot/perpetual definition independently."""
    factors = []
    inputs = []
    for name in columns:
        if name == "spot_perp_basis":
            inputs.append("mid")
            continue
        if name not in HFT_SPOT_FACTOR_COLUMNS:
            continue
        divergence = name.startswith("spot_perp_")
        if perpetual and not divergence:
            continue
        name = (
            name.removeprefix("lagged_")
            .removeprefix("spot_perp_")
            .removeprefix("spot_")
            .replace("_divergence", "")
        )
        name = name.replace("normalized_ofi_", "depth_normalized_ofi_").replace(
            "tfi_", "trade_flow_imbalance_"
        )
        factors.append(name)
        inputs.extend(_microstructure_inputs(name))
    if not perpetual:
        inputs.append("observed_utc")
    return tuple(dict.fromkeys(inputs)), tuple(dict.fromkeys(factors))


def cross_asset_inputs(columns):
    fields = []
    for column in columns:
        if column not in HFT_CROSS_ASSET_FACTOR_COLUMNS:
            continue
        name = column.split("_", 1)[1].removeprefix("lagged_")
        fields.append(
            {
                "normalized_ofi_500ms": "depth_normalized_ofi_500ms",
                "tfi_500ms": "trade_flow_imbalance_500ms",
            }.get(name, name)
        )
    return tuple(dict.fromkeys(fields))


def external_channels(columns):
    channels = set()
    for column in columns:
        name = column.partition("_")[2]
        if name.startswith(("observed_", "trade_")):
            channels.add("trades")
        elif name in HFT_EXTERNAL_FIELDS or name.startswith("quote_"):
            channels.add("quotes")
    return tuple(kind for kind in ("quotes", "trades") if kind in channels)


def factor_catalog():
    from quant_research_platform.catalog.factors import FactorCatalog
    from quant_research_platform.contracts.factors import FactorDefinition, FactorSet

    definitions = [
        FactorDefinition(
            factor_id=column,
            market_id=market_ids.MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=_microstructure_inputs(column),
            output_column=column,
            family="microstructure",
            required_input_data_id="binance_futures_microstructure_factors_100ms",
            timing_semantics="observed_by_decision_on_contiguous_100ms_segment",
            null_policy="full_window_positive_denominator",
        )
        for column in HFT_FACTOR_COLUMNS
    ]
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=market_ids.MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=("mid",)
            if column in {"mark_mid_basis", "perp_index_basis"}
            else ("timestamp",),
            output_column=column,
            family="derivatives_state",
            required_input_data_id=market_ids.HFT_DERIVATIVE_DATA_ID,
            timing_semantics="native_observed_at_lte_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_DERIVATIVE_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=market_ids.MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=spot_dependencies((column,))[0],
            output_column=column,
            family="spot_perp",
            required_input_data_id=market_ids.HFT_SPOT_CONTEXT_DATA_ID,
            timing_semantics="spot_grid_available_by_perpetual_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_SPOT_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=column,
            market_id=market_ids.MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=cross_asset_inputs((column,)),
            output_column=column,
            family="cross_asset",
            required_input_data_id=market_ids.HFT_CROSS_ASSET_DATA_ID,
            timing_semantics="mature_information_grid_lte_decision_with_explicit_age_limit",
            null_policy="missing_stale_or_incomplete_window_is_null",
        )
        for column in HFT_CROSS_ASSET_FACTOR_COLUMNS
    )
    definitions.extend(
        FactorDefinition(
            factor_id=f"{venue}_{column}",
            market_id=market_ids.MARKET_ID,
            factor_set=FactorSet.STANDARD,
            input_columns=("mid",)
            if column in {"price_difference", "relative_mid_dispersion"}
            else ("timestamp",),
            output_column=f"{venue}_{column}",
            family="external_price_discovery",
            required_input_data_id=data_id,
            timing_semantics="normalized_local_timestamp_lte_binance_decision",
            null_policy="missing_stale_incomplete_or_unrepresentable_is_null",
        )
        for venue, data_id in market_ids.HFT_EXTERNAL_DATA_IDS.items()
        for column in HFT_EXTERNAL_FIELDS
    )
    return FactorCatalog(definitions)


def crypto_execution_windows(plan, output_window, *, sample_role):
    """Validate the requested final-evaluation extent for this frozen protocol."""
    if sample_role != "test":
        raise PermissionError("crypto_execution_requires_frozen_test")
    boundary = TimeWindow(**plan["test"])
    if pd.Timestamp(output_window.start) < pd.Timestamp(boundary.start) or pd.Timestamp(
        output_window.end
    ) > pd.Timestamp(boundary.end):
        raise PermissionError("crypto_execution_test_window_invalid")
    return ((plan["folds"][-1]["fold_id"], output_window),)
