"""Published daily risk history for native intraday stock decisions."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.risk import (
    FactorRiskModel,
    RiskInputsUnavailable,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_real_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.stock_pool import (
    load_stock_pool,
    stock_pool_rows,
    stock_status_rows,
)
from quant_research_platform.research.datasets.risk import (
    asof_risk_frame,
    read_asof_risk_facts,
    risk_exposure_columns,
)
from quant_research_platform.research.risk import (
    FactorExposureSession,
    FactorRiskState,
    preprocess_risk_styles,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def portfolio_risk_context(config, *, window, construction):
    """Build a callable from full-market native sources, never account holdings.

    Daily return origins and maturity are source facts. Missing clocks remain
    unknown; a date-only panel timestamp is not a publication-time substitute.
    """
    config = _strict_config(
        config,
        {"inputs", "risk_exposure_input", "stock_pool"},
        {"risk_model", "daily_return_input"},
    )
    inputs = config["inputs"]
    exposure = _strict_config(
        config["risk_exposure_input"], {"data_id", "available_at_column"}, set()
    )
    source = open_dataset_session(**_dataset_input_options(inputs[exposure["data_id"]]))
    if source.spec.get("market_id") != "a_share" or source.spec["frequency"] != "1d":
        raise ValueError("portfolio_risk_source_invalid")
    declared = source.spec.get("risk_model")
    model = FactorRiskModel.from_mapping(config.get("risk_model", declared or {}))
    if declared is not None and model != FactorRiskModel.from_mapping(declared):
        raise ValueError("portfolio_risk_model_conflict")
    pool = load_stock_pool(config["stock_pool"])
    calendars = pool["calendars"]
    axes = {
        market: pd.DatetimeIndex(calendar.index[calendar.fillna(False)]).sort_values()
        for market, calendar in calendars.items()
    }
    sessions = pd.DatetimeIndex(
        sorted(set().union(*(set(axis) for axis in axes.values())))
    ).as_unit("ns")
    history_length = max(model.config.factor_window, model.config.specific_window)
    _, stop, closed = time_window_timestamp_interval(window)
    stop = pd.Timestamp(stop) - pd.Timedelta(0 if closed else 1, "ns")

    def relevant(at):
        ended = sessions[sessions < at.normalize()]
        boundaries = (
            [ended[-model.config.factor_window]]
            if len(ended) >= model.config.factor_window
            else []
        )
        for axis in axes.values():
            own = axis[axis < at.normalize()]
            if len(own):
                boundaries.append(own[max(0, len(own) - history_length)])
        return ended[ended >= min(boundaries)] if boundaries else ended

    initial = (
        relevant(pd.Timestamp(window.start))
        if construction == "absolute_risk"
        else sessions[:0]
    )
    start = initial[0] if len(initial) else pd.Timestamp(window.start).normalize()
    history_window = TimeWindow(start, stop)
    returns = pd.DataFrame()
    if construction == "absolute_risk" and "daily_return_input" in config:
        declaration = _strict_config(
            config["daily_return_input"],
            {"data_id"},
            {"column", "start_at_column", "available_at_column"},
        )
        return_source = open_dataset_session(
            **_dataset_input_options(
                inputs[declaration["data_id"]], time_window=history_window
            )
        )
        if (
            return_source.spec.get("market_id") != "a_share"
            or return_source.spec["frequency"] != "1d"
        ):
            raise ValueError("portfolio_risk_source_invalid")
        value = declaration.get("column", "period_return")
        origin = declaration.get("start_at_column", "return_start_at")
        maturity = declaration.get(
            "available_at_column",
            return_source.spec.get("label_availability_columns", {}).get(
                value, "return_available_at"
            ),
        )
        wanted = (
            "market_id",
            "asset_id",
            "timestamp",
            "listing_date",
            "is_trading",
            value,
            origin,
            maturity,
        )
        returns = materialize_dataset_session(
            return_source.projected_view(
                tuple(name for name in wanted if name in return_source.logical_columns)
            )
        ).reindex(columns=wanted)
        for name in ("timestamp", "listing_date", origin, maturity):
            returns[name] = strict_timestamp_series(
                returns[name],
                error="portfolio_risk_return_clock_invalid",
                allow_null=True,
            )
        returns["asset_return"] = strict_real_numeric_series(
            returns[value], error="portfolio_risk_return_invalid"
        )
        returns["is_trading"] = strict_bool_series(
            returns.is_trading, error="portfolio_risk_trading_invalid", allow_null=True
        )
        returns["_start_at"] = returns[origin]
        returns["_available_at"] = returns[maturity]
        returns["asset_id"] = returns.asset_id.astype("string")
        returns["timestamp"] = returns.timestamp.dt.normalize()
        returns = returns.loc[returns.timestamp.notna()].copy()
        if len(returns) and returns._start_at.notna().any():
            history_window = TimeWindow(min(start, returns._start_at.min()), stop)
    columns = (
        "market_id",
        "asset_id",
        "timestamp",
        "listing_date",
        exposure["available_at_column"],
        *risk_exposure_columns(model),
        "float_market_cap",
    )
    facts = read_asof_risk_facts(
        source.projected_view(
            tuple(name for name in columns if name in source.logical_columns)
        ),
        model=model,
        available_at_column=exposure["available_at_column"],
        time_window=history_window,
        additional_columns=("float_market_cap",),
    )
    groups = returns.groupby("timestamp", sort=False).indices if len(returns) else {}
    published = (
        returns.sort_values("_available_at", kind="stable") if len(returns) else returns
    )
    published = (
        published.loc[published._available_at.notna()] if len(published) else published
    )
    state = FactorRiskState(style_columns=model.style_columns, config=model.config)
    source_cache, prepared_days = {}, {}
    last_at, processed, published_count = None, None, 0
    estimate, estimate_key, version = None, None, 0

    def align(rows):
        return asof_risk_frame(
            rows,
            facts,
            model=model,
            available_at_column=exposure["available_at_column"],
            additional_columns=("float_market_cap",),
            cache=source_cache,
        )

    def history_day(day):
        if day in prepared_days:
            return prepared_days[day]
        rows = (
            returns.iloc[groups[day]].copy()
            if day in groups
            else returns.iloc[:0].copy()
        )
        if rows.empty:
            rows = pd.DataFrame(
                columns=[
                    "market_id",
                    "asset_id",
                    "listing_date",
                    "timestamp",
                    "_start_at",
                    "_available_at",
                    "asset_return",
                    "is_trading",
                ]
            )
            for name in ("timestamp", "listing_date", "_start_at", "_available_at"):
                rows[name] = pd.Series(dtype="datetime64[ns]")
        rows["eligible"] = pd.Series(pd.NA, index=rows.index, dtype="boolean")
        rows["_start_life"] = pd.NaT
        designs = []
        valid = rows._start_at.notna() & rows._start_at.lt(day)
        for origin, positions in (
            rows.loc[valid].groupby("_start_at", sort=False).groups.items()
        ):
            membership = stock_pool_rows(
                pd.DatetimeIndex([origin]), require_continuous=False, **pool
            )
            # A missing endpoint row cannot remove an origin stock from the
            # normalization domain. Only the response fit may lose that stock.
            normalized = preprocess_risk_styles(
                state.prepare_exposures(align(membership), include_ineligible=True),
                style_columns=model.style_columns,
            )
            membership = membership.set_index("asset_id")
            assets = rows.loc[positions, "asset_id"]
            rows.loc[positions, "eligible"] = membership.eligible.reindex(
                assets
            ).to_numpy()
            rows.loc[positions, "_start_life"] = membership.listing_date.reindex(
                assets
            ).to_numpy()
            designs.append(
                normalized.loc[normalized.asset_id.isin(assets)].assign(timestamp=day)
            )
        prepared = (
            FactorExposureSession.from_preprocessed(
                pd.concat(designs, ignore_index=True)
            )
            if designs
            else state.prepare_exposure_session(
                align(rows.assign(timestamp=day, listing_date=rows._start_life)),
                include_ineligible=True,
            )
        )
        prepared_days[day] = (rows, prepared)
        return rows, prepared

    def context(decision):
        nonlocal state, last_at, processed, published_count
        nonlocal estimate, estimate_key, version
        at = pd.Timestamp(decision.timestamp)
        if last_at is not None and at < last_at:
            raise ValueError("portfolio_risk_decision_order_invalid")
        active = relevant(at) if construction == "absolute_risk" else sessions[:0]
        count = (
            int(published._available_at.searchsorted(at, side="right"))
            if len(published)
            else 0
        )
        arrived = published.iloc[published_count:count] if len(published) else published
        revise = (
            processed is not None
            and len(arrived)
            and arrived.timestamp.between(active[0], processed).any()
            if len(active)
            else False
        )
        if revise:
            # Late realized returns change a past regression. Replay only the
            # estimator's retained own-market windows, reusing frozen designs.
            state = FactorRiskState(
                style_columns=model.style_columns, config=model.config
            )
            processed = None
        changed = False
        for day in active:
            if processed is not None and day <= processed:
                continue
            rows, prepared = history_day(day)
            ready = (
                rows._available_at.notna()
                & rows._available_at.le(at)
                & rows._available_at.ge(rows.timestamp)
                & rows._available_at.gt(rows._start_at)
                & rows.listing_date.eq(rows._start_life)
            )
            cross = rows.assign(
                asset_return=rows.asset_return.where(ready), timestamp=day
            )
            statuses = stock_status_rows(
                pd.DataFrame(
                    {
                        "asset_id": pool["stocks"].asset_id.drop_duplicates(),
                        "timestamp": day,
                    }
                ),
                stocks=pool["stocks"],
                status=pool["status"],
            )
            # Every known life advances on its real market sessions, including
            # an initial session with no response row in the return source.
            cross = cross.set_index("asset_id").reindex(statuses.asset_id).reset_index()
            cross["timestamp"] = day
            closed, opened = set(), set()
            for market, assets in statuses.groupby("market_type", sort=False):
                status = calendars.get(market, pd.Series(dtype="boolean")).get(
                    day, pd.NA
                )
                if pd.notna(status) and not status:
                    closed.update(assets.asset_id.astype(str))
                elif pd.notna(status) and status:
                    opened.update(assets.asset_id.astype(str))
            endpoint_lives = cross.asset_id.map(
                statuses.set_index("asset_id").listing_date
            )
            cross["asset_return"] = cross.asset_return.where(
                cross.asset_id.isin(opened) & cross.listing_date.eq(endpoint_lives)
            )
            cross["listing_date"] = endpoint_lives
            state.observe_return_session(
                cross, closed_assets=frozenset(closed), prepared=prepared
            )
            processed, changed = day, True
        if changed:
            version += 1
        if len(active):
            for day in tuple(prepared_days):
                if day < active[0]:
                    del prepared_days[day]
        last_at, published_count = at, count
        universe = stock_pool_rows(pd.DatetimeIndex([at]), **pool)
        raw = align(universe)
        prepared = state.prepare_exposure_session(raw, include_ineligible=True)
        exposures = state.current_exposures(prepared).set_index("asset_id")
        industries = tuple(
            name for name in prepared.industry_columns if name in exposures
        )
        key = (version, industries)
        if key != estimate_key:
            try:
                estimate = state.estimate() if construction == "absolute_risk" else None
            except RiskInputsUnavailable:
                estimate = None
            estimate_key = key
        assets = exposures.reindex(universe.asset_id).copy()
        assets["float_market_cap"] = pd.to_numeric(
            raw.set_index("asset_id").float_market_cap, errors="raise"
        )
        assets["specific_variance"] = (
            np.nan if estimate is None else estimate.specific_variance
        )
        # A newly listed life cannot inherit the previous life's residual variance.
        lives = universe.set_index("asset_id").listing_date
        historical_lives = pd.Series(state._listings, dtype="datetime64[ns]")
        assets["specific_variance"] = assets.specific_variance.where(
            lives.eq(historical_lives.reindex(lives.index))
        )
        return {
            "as_of": at,
            "assets": assets,
            "factor_covariance": None
            if estimate is None
            else estimate.factor_covariance,
            "industry_columns": industries,
            "style_columns": model.style_columns,
        }

    context.risk_model = model
    return context
