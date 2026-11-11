"""Explicit preopen decisions and intraday labels for ETFs and cash indices."""

import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import time_window_timestamp_interval
from quant_research_platform.markets.implementations.a_share.calendar import (
    next_trading_sessions,
    unified_trading_sessions,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)

KEY = ("market_id", "asset_id", "timestamp")
LABEL = "next_session_intraday_return"
CONTEXT = (
    "signal_session",
    "execution_session",
    "order_submit_at",
    "return_available_at",
    LABEL + "_available_at",
    "eligible",
    "open",
    "close",
    "period_return",
)


def processed_specs():
    result = []
    for name, source in (
        ("etf_daily_panel", "public_domestic_etf_prices"),
        ("benchmark_daily_panel", "index_market"),
    ):
        etf = name == "etf_daily_panel"
        columns = (
            *KEY,
            *CONTEXT,
            *(
                (
                    "high",
                    "low",
                    "volume",
                    "amount",
                    "avg_volume_20d",
                    "avg_amount_20d",
                    "adj_close",
                    "secondary_flow_imbalance",
                    "volatility_20d",
                    "volatility_60d",
                    "implied_variance_20d",
                    "implied_variance_60d",
                    "market_return",
                )
                if etf
                else ()
            ),
            LABEL,
        )
        roles = {
            **dict.fromkeys(columns, "context"),
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            "eligible": "mask",
            LABEL: "label",
        }
        unit = "CNY_per_fund_unit" if etf else "index_points"
        units = {
            "open": unit,
            "close": unit,
            "period_return": "decimal_return",
            LABEL: "decimal_return",
        }
        if etf:
            units.update(
                high=unit,
                low=unit,
                volume="fund_units",
                amount="CNY",
                avg_volume_20d="fund_units",
                avg_amount_20d="CNY",
                adj_close="total_return_index",
                secondary_flow_imbalance="dimensionless",
                volatility_20d="decimal_return",
                volatility_60d="decimal_return",
                implied_variance_20d="annualized_variance",
                implied_variance_60d="annualized_variance",
                market_return="decimal_return",
            )
        result.append(
            ProcessedDataSpec(
                data_id=name,
                market_id="a_share",
                kind="panel",
                frequency="1d",
                grain="asset_timestamp",
                primary_key=KEY,
                partition_columns=("year",),
                column_roles=roles,
                column_units=units,
                label_availability_columns={LABEL: LABEL + "_available_at"},
                source_input_data_ids=(
                    source,
                    "trading_calendar",
                    *(
                        (
                            "public_domestic_etf_actions",
                            "public_option_risk",
                            "public_option_contracts",
                            "index_market",
                        )
                        if etf
                        else ()
                    ),
                ),
                optional_source_input_data_ids=(
                    "public_domestic_etf_actions",
                    "public_option_risk",
                    "public_option_contracts",
                    "index_market",
                )
                if etf
                else (),
            )
        )
    return tuple(result)


def configured_spec(spec, config):
    params = config.get("parameters", {})
    if (
        type(params) is not dict
        or set(params) - {"asset_ids"}
        or {"factor_columns", "label_columns", "base_panel"} & config.keys()
    ):
        raise ValueError("daily_quote_parameters_invalid")
    assets = params.get("asset_ids")
    if assets is not None and (
        type(assets) is not list
        or not assets
        or len(set(assets)) != len(assets)
        or any(type(x) is not str or not x for x in assets)
    ):
        raise ValueError("daily_quote_assets_invalid")
    return replace(
        spec,
        parameters_json=json.dumps(params),
        processed_input_data_ids=("hft_panel",)
        if "hft_panel" in config["inputs"]
        else (),
    )


def daily_decisions(prices, calendar, *, actions=None):
    """A return uses the exact next declared session; no weekend or gap compression."""
    sessions = unified_trading_sessions(calendar)
    blocks = []
    for asset, group in prices.groupby("asset_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        axis = sessions[
            (sessions >= group.index.min()) & (sessions <= group.index.max())
        ]
        data = group.reindex(axis)
        following = next_trading_sessions(calendar, axis)
        current = data[["open", "close"]]
        next_prices = current.reindex(following).set_axis(axis)
        valid = next_prices.open.gt(0) & next_prices.close.gt(0)
        out = data.copy()
        out["asset_id"] = asset
        out["market_id"] = "a_share"
        out["signal_session"] = axis
        out["execution_session"] = following
        out["timestamp"] = following + pd.Timedelta(hours=1)
        out["order_submit_at"] = out.timestamp + pd.Timedelta(seconds=1)
        out["return_available_at"] = axis + pd.Timedelta(hours=7, seconds=1)
        out[LABEL + "_available_at"] = following + pd.Timedelta(hours=7, seconds=1)
        out[LABEL] = (next_prices.close / next_prices.open - 1).where(valid)
        out["period_return"] = (
            data.close / data.close.shift().where(data.close.shift() > 0) - 1
        )
        out["adj_close"] = np.nan
        out["secondary_flow_imbalance"] = np.nan
        out["market_return"] = np.nan
        if "volume" in data and actions is None:
            out["period_return"] = np.nan
        if actions is not None:
            multiplier = pd.Series(1.0, index=axis)
            selected = actions.loc[actions.asset_id.str.split(":").str[0].eq(asset)]
            for date, events in selected.groupby("timestamp", sort=False):
                # Pre-listing conversion belongs to a different price lifecycle.
                if date <= axis.min() or date not in axis:
                    continue
                decision = out.loc[date, "timestamp"]
                visible = events.available_at.notna() & events.available_at.le(decision)
                splits = events.loc[events.event_type.eq("splits")]
                dividends = events.loc[events.event_type.eq("dividends")]
                q = (
                    (splits.numerator / splits.denominator).prod(min_count=len(splits))
                    if len(splits)
                    else 1.0
                )
                cash = (
                    dividends.amount.sum(min_count=len(dividends))
                    if len(dividends)
                    else 0.0
                )
                multiplier.loc[date] = (
                    q + cash / data.close.loc[date] if visible.all() else np.nan
                )
            out["adj_close"] = data.close * multiplier.cumprod(skipna=False)
            out["period_return"] = (
                out.adj_close / out.adj_close.shift().where(out.adj_close.shift() > 0)
                - 1
            )
        out["eligible"] = data.close.gt(0)
        for width in (20, 60):
            out[f"volatility_{width}d"] = out.period_return.rolling(
                width, min_periods=width
            ).std()
            out[f"implied_variance_{width}d"] = np.nan
        for field in ("volume", "amount"):
            if field in data:
                out[f"avg_{field}_20d"] = data[field].rolling(20, min_periods=20).mean()
        blocks.append(out.loc[out.timestamp.notna()].reset_index(drop=True))
    return (
        pd.concat(blocks, ignore_index=True)
        if blocks
        else pd.DataFrame(columns=[*KEY, *CONTEXT, LABEL])
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    source = (
        "public_domestic_etf_prices"
        if spec.data_id == "etf_daily_panel"
        else "index_market"
    )
    options = _dataset_input_options(inputs[source])
    params = json.loads(spec.parameters_json)
    asset_column = (
        "asset_id" if source == "public_domestic_etf_prices" else "source_asset_id"
    )
    if "asset_ids" in params:
        options = _dataset_input_options(
            inputs[source],
            row_filters=((asset_column, "in", tuple(params["asset_ids"])),),
        )
    prices = materialize_dataset_session(open_dataset_session(**options))
    if asset_column != "asset_id":
        prices = prices.rename(columns={asset_column: "asset_id"})
    fields = [
        "asset_id",
        "timestamp",
        "open",
        "close",
        *(name for name in ("high", "low", "volume", "amount") if name in prices),
    ]
    prices = prices.loc[:, fields]
    calendar = materialize_dataset_session(
        open_dataset_session(**_dataset_input_options(inputs["trading_calendar"]))
    )
    actions = None
    if "public_domestic_etf_actions" in inputs:
        actions = materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(inputs["public_domestic_etf_actions"])
            )
        )
        actions["available_at"] = pd.to_datetime(
            actions.available_at, utc=True
        ).dt.tz_localize(None)
    frame = daily_decisions(prices, calendar, actions=actions)
    if spec.data_id == "etf_daily_panel" and "index_market" in inputs:
        benchmark = materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(inputs["index_market"])
            ).projected_view(("source_asset_id", "timestamp", "open", "close"))
        )
        if benchmark.source_asset_id.nunique() != 1:
            raise ValueError("etf_benchmark_requires_one_explicit_index")
        benchmark = daily_decisions(
            benchmark.rename(columns={"source_asset_id": "asset_id"}), calendar
        )
        returns = benchmark.set_index("signal_session").period_return
        frame["market_return"] = frame.signal_session.map(returns)
    if {"public_option_risk", "public_option_contracts"} <= set(inputs):
        from quant_research_platform.research.factors.option_variance import (
            implied_variances,
        )

        risk, contracts = (
            materialize_dataset_session(
                open_dataset_session(**_dataset_input_options(inputs[name]))
            )
            for name in ("public_option_risk", "public_option_contracts")
        )
        implied = implied_variances(frame, risk, contracts)
        frame[list(implied)] = implied
    if "hft_panel" in inputs:
        panel = materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(inputs["hft_panel"])
            ).projected_view(
                (
                    "asset_id",
                    "timestamp",
                    "observed_trade_volume",
                    "observed_signed_trade_volume",
                )
            )
        )
        panel["signal_session"] = panel.timestamp.dt.normalize()
        groups = panel.groupby(["asset_id", "signal_session"], sort=False)
        fields = ["observed_trade_volume", "observed_signed_trade_volume"]
        sums = (
            groups[fields].sum().where(groups[fields].count().eq(groups.size(), axis=0))
        )
        flow = (
            sums.observed_signed_trade_volume
            / sums.observed_trade_volume.where(sums.observed_trade_volume > 0)
        ).rename("secondary_flow_imbalance")
        frame = frame.drop(columns="secondary_flow_imbalance").merge(
            flow.reset_index(), on=["asset_id", "signal_session"], how="left"
        )
    if source == "index_market":
        frame["asset_id"] = "index:" + frame.asset_id
    start, end, closed = time_window_timestamp_interval(time_window)
    selected = frame.loc[
        frame.timestamp.ge(start)
        & (frame.timestamp.le(end) if closed else frame.timestamp.lt(end)),
        list(spec.column_roles),
    ]
    metadata = {
        "producer_diagnostics": {
            "decision_clock": "09:00_Asia_Shanghai_preopen",
            "order_latency_seconds": 1,
            "label": "next_declared_session_close_over_open_minus_one",
            "label_price_adjustment": "same_bar_adjustment_cancels",
            "listing_date_inferred": False,
            "implied_variance_method": (
                "ATM_call_put_mean_variance;linear_total_variance_actual_expiries;"
                "20_60_over_252_years;no_extrapolation"
            ),
        }
    }
    yield (selected,), metadata
