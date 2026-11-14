"""Explicit preopen decisions and intraday labels for ETFs and cash indices."""

import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import observation_versions
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    next_trading_sessions,
    unified_trading_sessions,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PATH_FACTORS,
    PRICE_VOLUME_UNITS,
    price_volume_factor_definitions,
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
ETF_PRICE_VOLUME_FACTORS = tuple(
    item.output_column
    for item in price_volume_factor_definitions(
        market_id="a_share",
        required_input_data_id="etf_daily_panel",
        factor_columns=(*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS),
    )
    if set(item.input_columns) - {"filling_flag"}
    <= {"adj_close", "close", "high", "low", "volume", "amount", "period_return"}
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
    result.append(
        ProcessedDataSpec(
            data_id="etf_price_volume_factors",
            market_id="a_share",
            kind="factor",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=KEY,
            partition_columns=("year",),
            column_roles={
                **dict.fromkeys(KEY, "key"),
                "timestamp": "timestamp",
                **dict.fromkeys(ETF_PRICE_VOLUME_FACTORS, "factor"),
            },
            column_units={
                name: PRICE_VOLUME_UNITS[name] for name in ETF_PRICE_VOLUME_FACTORS
            },
            processed_input_data_ids=("etf_daily_panel",),
        )
    )
    return tuple(result)


def configured_spec(spec, config):
    if spec.data_id == "etf_price_volume_factors":
        selected = config.get("factor_columns", list(ETF_PRICE_VOLUME_FACTORS))
        params = config.get("parameters", {})
        if (
            type(selected) is not list
            or not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(ETF_PRICE_VOLUME_FACTORS)
        ):
            raise ValueError("data_operation_factor_columns_invalid")
        if (
            type(params) is not dict
            or set(params) - {"min_fresh_fraction"}
            or type(params.get("min_fresh_fraction", 0.75)) not in (int, float)
            or not 0 < params.get("min_fresh_fraction", 0.75) <= 1
        ):
            raise ValueError("price_volume_parameters_invalid")
        return replace(
            spec,
            parameters_json=json.dumps(params),
            column_roles={
                **dict.fromkeys(KEY, "key"),
                "timestamp": "timestamp",
                **dict.fromkeys(selected, "factor"),
            },
            column_units={name: PRICE_VOLUME_UNITS[name] for name in selected},
        )
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
    # ETF/index quotes use Shanghai/Shenzhen calendars, not other boards' histories.
    calendar = calendar.loc[calendar.source_market_type.isin((1, 4))]
    sessions = unified_trading_sessions(calendar)
    if actions is not None and not actions.empty:
        actions = observation_versions(
            actions,
            keys=("asset_id", "timestamp"),
            columns=("event_type", "amount", "numerator", "denominator"),
        )
    blocks = []
    for asset, group in prices.groupby("asset_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        axis = sessions[
            (sessions >= group.index.min()) & (sessions <= group.index.max())
        ]
        data = group.reindex(axis)
        following = next_trading_sessions(calendar, axis)
        # Unknown calendar intervals cannot become adjacent return observations.
        segments = pd.Series(following, index=axis).shift().ne(axis).cumsum()
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
        previous = data.close.groupby(segments).shift()
        out["period_return"] = data.close / previous.where(previous > 0) - 1
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
                known = events.loc[visible].drop_duplicates("asset_id", keep="last")
                complete = known.asset_id.nunique() == events.asset_id.nunique()
                splits = known.loc[known.event_type.eq("splits")]
                dividends = known.loc[known.event_type.eq("dividends")]
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
                    q + cash / data.close.loc[date] if complete else np.nan
                )
                if complete:
                    # Both closes are known by the current session's close.
                    # A later action version must not inherit that earlier clock.
                    out.loc[date, "return_available_at"] = max(
                        out.loc[date, "return_available_at"], known.available_at.max()
                    )
            out["adj_close"] = data.close * multiplier.groupby(segments).cumprod(
                skipna=False
            )
            # A one-session return needs only this session's action, not an
            # unknown cumulative adjustment from an earlier session.
            out["period_return"] = (
                data.close * multiplier / previous.where(previous > 0) - 1
            )
        out["return_available_at"] = out.return_available_at.where(
            out.period_return.notna()
        )
        out["eligible"] = data.close.gt(0)
        for width in (20, 60):
            out[f"volatility_{width}d"] = (
                out.period_return.groupby(segments)
                .rolling(width, min_periods=width)
                .std()
                .droplevel(0)
            )
            out[f"implied_variance_{width}d"] = np.nan
        for field in ("volume", "amount"):
            if field in data:
                out[f"avg_{field}_20d"] = (
                    data[field]
                    .groupby(segments)
                    .rolling(20, min_periods=20)
                    .mean()
                    .droplevel(0)
                )
        blocks.append(out.loc[out.timestamp.notna()].reset_index(drop=True))
    return (
        pd.concat(blocks, ignore_index=True)
        if blocks
        else pd.DataFrame(columns=[*KEY, *CONTEXT, LABEL])
    )


@contextmanager
def produce(spec, *, inputs, time_window):
    if spec.data_id == "etf_price_volume_factors":
        start, end, closed = time_window_timestamp_interval(time_window)
        source = open_dataset_session(
            **_dataset_input_options(inputs["etf_daily_panel"])
        )
        selected = tuple(
            name for name, role in spec.column_roles.items() if role == "factor"
        )
        definitions = price_volume_factor_definitions(
            market_id="a_share",
            required_input_data_id="etf_daily_panel",
            factor_columns=selected,
        )
        fields = tuple(
            dict.fromkeys(
                (
                    *KEY,
                    "signal_session",
                    *(
                        name
                        for definition in definitions
                        for name in definition.input_columns
                        if name != "filling_flag"
                    ),
                )
            )
        )
        frame = (
            materialize_dataset_session(
                source.filtered_view(
                    TimeWindow(
                        pd.Timestamp.min, end if closed else end - pd.Timedelta(1, "ns")
                    )
                ).projected_view(fields)
            )
            .sort_values(["asset_id", "timestamp"], kind="stable")
            .reset_index(drop=True)
        )
        # A native row's decision date proves its signal's next session. Missing
        # links delimit history without inventing a calendar or persisted state.
        previous_decisions = frame.groupby("asset_id", sort=False).timestamp.shift()
        segments = frame.signal_session.ne(previous_decisions.dt.normalize()).cumsum()
        keys = frame.loc[
            frame.timestamp.ge(start)
            & (frame.timestamp.le(end) if closed else frame.timestamp.lt(end)),
            [*KEY, "signal_session"],
        ]
        if keys.empty:
            result = keys.assign(**dict.fromkeys(selected, np.nan))
        else:
            # Retain causal warmup and recursive state, but calculate only the
            # signal sessions consumed by this decision window.
            calculated = calculate_price_volume_factors(
                frame.assign(
                    timestamp=frame.signal_session, _calendar_segment=segments
                ),
                factor_columns=selected,
                output_window=TimeWindow(
                    keys.signal_session.min(), keys.signal_session.max()
                ),
                history_keys=("asset_id", "_calendar_segment"),
                **json.loads(spec.parameters_json),
            )
            result = keys.merge(
                calculated.rename(columns={"timestamp": "signal_session"}),
                on=["asset_id", "signal_session"],
                how="left",
            )
        yield (
            (result.loc[:, list(spec.column_roles)],),
            {
                "producer_diagnostics": {
                    "formula_owner": "price_volume",
                    "decision_clock": "inherited_etf_daily_panel",
                    "stock_eligibility_filters": False,
                }
            },
        )
        return
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
