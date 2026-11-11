"""Disclosure-only fund ownership, unit flows and ETF primary-market pressure."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.asof import (
    iter_latest_report_keys,
    latest_observations,
    rolling_available_at,
)
from quant_research_platform.contracts.factors import FactorDefinition

ETF = (
    "etf_share_change_1d",
    "etf_share_change_5d",
    "etf_share_change_20d",
    "etf_share_change_surprise_20d",
    "domestic_etf_premium_discount",
    "etf_primary_secondary_pressure_divergence",
    "stock_passive_flow_pressure_1d",
    "stock_passive_flow_pressure_5d",
    "industry_passive_flow_pressure_1d",
    "industry_passive_flow_pressure_5d",
)
FUNDS = (
    "cn_fund_ownership_float_share",
    "cn_fund_holder_concentration",
    "cn_common_fund_ownership_exposure",
    "cn_disclosed_ownership_change",
    "cn_fund_equity_allocation",
    "cn_fund_cash_buffer",
    "cn_fund_net_flow_rate",
    "cn_fund_redemption_stress",
    "cn_redemption_stock_pressure",
    "ownership_crowding_x_drawdown",
    "us_institution_ownership_concentration",
    "us_institution_sector_allocation",
    "us_institution_sector_allocation_change",
    "us_crowded_sector_flow_pressure",
    "us_fund_liquidity_buffer",
    "ownership_crowding_x_liquidity_stress",
)
RECIPES = {
    **dict.fromkeys(ETF[:4], ("public_fund_shares", "trading_calendar")),
    ETF[4]: ("public_fund_nav", "public_domestic_etf_prices"),
    ETF[5]: ("public_fund_shares", "trading_calendar"),
    **dict.fromkeys(
        ETF[6:],
        ("public_fund_shares", "public_etf_pcf_components", "trading_calendar"),
    ),
    **dict.fromkeys(
        (*FUNDS[:4], FUNDS[9], FUNDS[15]),
        ("public_fund_holdings", "public_fund_holdings_reports"),
    ),
    **dict.fromkeys(FUNDS[4:8], ("public_fund_financial_cells",)),
    FUNDS[8]: (
        "public_fund_financial_cells",
        "public_fund_holdings",
        "public_fund_holdings_reports",
    ),
    **dict.fromkeys(FUNDS[10:13], ("public_13f_holdings", "public_13f_filings")),
    FUNDS[13]: (
        "public_nport_funds",
        "public_nport_holdings",
    ),
    FUNDS[14]: ("public_nport_funds",),
}
UNITS = dict.fromkeys(RECIPES, "dimensionless")
for _name in ETF[6:]:
    UNITS[_name] = "trading_days"


def definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            factor_set="standard",
            input_columns=dependencies,
            output_column=name,
            family="fund_flows",
            required_input_data_id="fund_flow_factors",
            description=(
                "Publication-visible disclosed ownership or fund-unit flow. ETF "
                "flows require historical PCF quantities and creation-unit "
                "denominators; daily shares changes are not mechanically called "
                "cash flows. Fund period unit subscriptions/redemptions are "
                "normalized by beginning shares, not nominal AUM growth. Holdings"
                " snapshots replace the whole disclosed fund report; unknown "
                "holdings outside the disclosed universe stay unknown."
            ),
            timing_semantics="available_at_before_explicit_decision",
            null_policy="only_actual_dependencies; never_fill_unknown_publication",
        )
        for name, dependencies in RECIPES.items()
    )


def share_states(frame, *, session_days, session_groups=None):
    keys = ["asset_id", "timestamp"]
    grouped = frame.groupby(keys, sort=False)
    work = (
        grouped.shares_outstanding.first()
        .where(grouped.shares_outstanding.nunique(dropna=False).le(1))
        .to_frame()
    )
    work["available_at"] = grouped.available_at.max()
    work = work.reset_index().sort_values(keys, kind="stable")
    if len(work):
        work = pd.concat(
            [
                block.set_index("timestamp")
                .reindex(
                    session_days[
                        (session_days >= block.timestamp.min())
                        & (session_days <= block.timestamp.max())
                    ]
                )
                .assign(asset_id=asset)
                .rename_axis("timestamp")
                .reset_index()
                for asset, block in work.groupby("asset_id", sort=False)
            ],
            ignore_index=True,
        )
    work["_calendar_group"] = (
        0 if session_groups is None else work.timestamp.map(session_groups)
    )
    for name in (*ETF[:4], "_share_delta_1d"):
        work[name] = np.nan
        work[f"{name}_available_at"] = pd.Series(
            pd.NaT, index=work.index, dtype="datetime64[ns]"
        )
    for _, at in work.groupby(
        ["asset_id", "_calendar_group"], sort=False
    ).indices.items():
        at = np.asarray(at)
        block = work.iloc[at]
        shares = block.shares_outstanding
        clocks = block.available_at
        for width in (1, 5, 20):
            previous = shares.shift(width)
            work.loc[at, f"etf_share_change_{width}d"] = (
                shares / previous.where(previous > 0) - 1
            ).to_numpy()
            work.loc[at, f"etf_share_change_{width}d_available_at"] = (
                pd.concat([clocks, clocks.shift(width)], axis=1)
                .max(axis=1, skipna=False)
                .to_numpy()
            )
        change = shares / shares.shift().where(shares.shift() > 0) - 1
        history = change.shift().rolling(20, min_periods=20)
        work.loc[at, "etf_share_change_surprise_20d"] = (
            (change - history.mean()) / history.std().where(lambda x: x > 0)
        ).to_numpy()
        work.loc[at, "etf_share_change_surprise_20d_available_at"] = (
            rolling_available_at(clocks, 22).to_numpy()
        )
        work.loc[at, "_share_delta_1d"] = shares.diff().to_numpy()
        work.loc[at, "_share_delta_1d_available_at"] = work.loc[
            at, "etf_share_change_1d_available_at"
        ].to_numpy()
    return work.drop(columns="_calendar_group")


def complete_mainland_stock_reports(headers):
    """Select complete mainland evidence independently of foreign holdings."""
    return headers.loc[
        headers.economic_portfolio_id.notna()
        & (
            (
                headers.disclosure_scope.eq("full_stock_holdings")
                & headers.stock_disclosure_status.isin(
                    (
                        "full_stock_table_reconciled",
                        "full_stock_table_as_reported_unreconciled",
                    )
                )
            )
            | (headers.explicit_no_stock.eq("true") & headers.stock_rows.eq(0))
            | (
                headers.disclosure_scope.eq("mainland_stock_holdings")
                & headers.stock_disclosure_status.isin(
                    (
                        "explicit_no_mainland_stock_reported",
                        "complete_all_equity_table_no_mainland_security_reported",
                    )
                )
                & headers.stock_rows.eq(0)
            )
            | (
                headers.disclosure_scope.eq("mainland_stock_holdings")
                & headers.stock_disclosure_status.eq(
                    "complete_all_equity_table_mainland_stock_domain"
                )
                & headers.stock_rows.gt(0)
            )
        )
    ]


def financial_states(cells):
    work = cells.loc[cells.value_observation_date.eq(cells.report_period_end)].copy()
    work["value"] = work.value.to_numpy(dtype=float, na_value=np.nan)
    keys = [
        "economic_portfolio_id",
        "fund_code",
        "timestamp",
        "available_at",
        "table_kind",
        "share_class_header",
        "field",
    ]
    grouped = work.groupby(keys, dropna=False, sort=False).value
    values = grouped.first().where(grouped.nunique(dropna=False).le(1)).reset_index()
    reports = []
    for (portfolio, fund, date, available), block in values.groupby(
        keys[:4], sort=False, dropna=False
    ):
        row = {
            "asset_id": str(portfolio),
            "fund_code": fund,
            "timestamp": date,
            "available_at": available,
        }
        balance = block.loc[block.table_kind.eq("balance_sheet")]
        composition = block.loc[block.table_kind.eq("portfolio_asset_composition")]
        flow = block.loc[block.table_kind.eq("fund_share_movement")]
        for field in (
            "net_assets",
            "total_assets",
            "bank_deposits",
            "settlement_reserves",
        ):
            known = balance.loc[balance.field.eq(field), "value"].drop_duplicates()
            row[field] = known.iloc[0] if len(known) == 1 else np.nan
        for field in ("stock_assets", "bank_deposits_and_settlement_reserves"):
            known = composition.loc[
                composition.field.eq(field), "value"
            ].drop_duplicates()
            row[field] = known.iloc[0] if len(known) == 1 else np.nan
        # Aggregate distinct disclosed share classes once, preserving any unknown class.
        for field in (
            "period_begin_fund_shares",
            "period_end_fund_shares",
            "period_subscription_fund_shares",
            "period_redemption_fund_shares",
            "period_split_change_fund_shares",
        ):
            selected = flow.loc[
                flow.field.eq(field), ["share_class_header", "value"]
            ].drop_duplicates()
            row[field] = (
                selected.value.sum(min_count=len(selected)) if len(selected) else np.nan
            )
        begin = row["period_begin_fund_shares"]
        end = row["period_end_fund_shares"]
        sub = row["period_subscription_fund_shares"]
        red = row["period_redemption_fund_shares"]
        split = row["period_split_change_fund_shares"]
        identity_ok = (
            not all(np.isfinite(v) for v in (begin, end, sub, red, split))
            or abs(end - (begin + sub - red + split)) <= 0.02
        )
        row["cn_fund_net_flow_rate"] = (
            (sub - red) / begin / date.month if begin > 0 and identity_ok else np.nan
        )
        row["_redemption_rate"] = red / begin if begin > 0 and identity_ok else np.nan
        row["_flow_period_start"] = pd.Timestamp(date.year, 1, 1)
        row["cn_fund_equity_allocation"] = (
            row["stock_assets"] / row["total_assets"]
            if row["total_assets"] > 0
            else np.nan
        )
        row["cn_fund_cash_buffer"] = (
            row["bank_deposits_and_settlement_reserves"] / row["total_assets"]
            if row["total_assets"] > 0
            else np.nan
        )
        reports.append(row)
    states = pd.DataFrame(reports)
    if states.empty:
        return states
    states = states.sort_values(
        ["asset_id", "timestamp", "available_at"], kind="stable"
    ).reset_index(drop=True)
    return states


def _financial_history_states(financial):
    """Recompute the latest report's stress only at actual publication events."""
    output = []
    for _, block in financial.groupby("asset_id", sort=False):
        known = {}
        for released, versions in block.groupby("available_at", sort=True):
            for row in versions.to_dict("records"):
                known[row["timestamp"]] = row
            current = max(known)
            window = sorted(known)[-9:]
            latest = known[current].copy()
            latest["available_at"] = released
            latest["cn_fund_redemption_stress"] = np.nan
            rates = pd.Series(
                [known[period]["_redemption_rate"] / period.month for period in window]
            )
            previous = rates.iloc[:-1]
            deviation = previous.std()
            if previous.count() >= 4 and deviation > 0:
                latest["cn_fund_redemption_stress"] = (
                    rates.iloc[-1] - previous.mean()
                ) / deviation
            output.append(latest)
    return pd.DataFrame(output, columns=[*financial, "cn_fund_redemption_stress"])


def us_ownership_states(holdings, filings, *, assets):
    """Update complete manager reports; supplements append and restatements replace."""
    tickers = tuple(assets)
    cusips = tuple(assets.values())
    n = len(cusips)
    if not n:
        return pd.DataFrame(
            columns=["asset_id", "timestamp", "available_at", *FUNDS[10:13]]
        )
    h = holdings.loc[
        holdings.cusip.isin(cusips)
        & holdings.share_type.eq("SH")
        & (
            holdings.put_call.isna()
            | holdings.put_call.astype("str").str.strip().eq("")
        )
    ]
    grouped = h.groupby(["accession", "cusip"], sort=False)[
        ["shares_or_principal", "value_usd"]
    ].sum(min_count=1)
    positions = {}
    index = {cusip: i for i, cusip in enumerate(cusips)}
    for accession, block in grouped.groupby(level=0, sort=False):
        matrix = np.zeros((n, 2))
        for (_, cusip), row in block.iterrows():
            matrix[index[cusip]] = row.to_numpy(float)
        positions[accession] = matrix
    f = filings.loc[
        filings.available_at.notna() & filings.filing_type.isin(("13F-HR", "13F-HR/A"))
    ].copy()
    f["_number"] = pd.to_numeric(f.amendment_number, errors="coerce")
    normal = f.filing_type.eq("13F-HR")
    f.loc[normal, "_number"] = 0
    f = f.sort_values(
        ["available_at", "timestamp", "_number", "accession"], kind="stable"
    )
    reports = {}
    current = {}
    previous = {}
    rows = []
    for clock, events in f.groupby("available_at", sort=True):
        changed = set()
        for event in events.to_dict("records"):
            manager = event["manager_cik"]
            period = event["timestamp"]
            number = event["_number"]
            quarters = reports.setdefault(manager, {})
            state = quarters.setdefault(
                period, {"base": None, "number": -1, "extras": {}}
            )
            kind = str(event["amendment_type"]).upper().strip()
            values = positions.get(event["accession"], np.zeros((n, 2)))
            if kind == "NEW HOLDINGS":
                state["extras"][event["accession"]] = (number, values)
            elif event["filing_type"] == "13F-HR" or kind == "RESTATEMENT":
                if not np.isfinite(number):
                    state["base"] = None
                elif number >= state["number"]:
                    state["base"] = values
                    state["number"] = number
            else:
                state["base"] = None
            changed.add(manager)
        for manager in changed:
            quarters = reports[manager]
            periods = sorted(quarters)

            def full(state):
                if state["base"] is None:
                    return None
                additions = [
                    v
                    for number, v in state["extras"].values()
                    if np.isfinite(number) and number > state["number"]
                ]
                if any(
                    not np.isfinite(number) for number, _ in state["extras"].values()
                ):
                    return None
                return state["base"] + sum(additions, start=np.zeros((n, 2)))

            current[manager] = full(quarters[periods[-1]])
            previous[manager] = (
                full(quarters[periods[-2]]) if len(periods) > 1 else None
            )
        complete = [value for value in current.values() if value is not None]
        if not complete:
            continue
        values = np.stack(complete)
        shares = values[:, :, 0].sum(axis=0)
        money = values[:, :, 1].sum(axis=0)
        concentration = np.divide(
            (values[:, :, 0] ** 2).sum(axis=0),
            shares**2,
            out=np.full(n, np.nan),
            where=shares > 0,
        )
        allocation = money / money.sum() if money.sum() > 0 else np.full(n, np.nan)
        cohort = [
            manager
            for manager in current
            if current[manager] is not None and previous[manager] is not None
        ]
        change = np.full(n, np.nan)
        if cohort:
            after = sum((current[m][:, 1] for m in cohort), start=np.zeros(n))
            before = sum((previous[m][:, 1] for m in cohort), start=np.zeros(n))
            if after.sum() > 0 and before.sum() > 0:
                change = after / after.sum() - before / before.sum()
        for i, ticker in enumerate(tickers):
            rows.append(
                dict(
                    asset_id=ticker,
                    timestamp=clock,
                    available_at=clock,
                    **{
                        FUNDS[10]: concentration[i],
                        FUNDS[11]: allocation[i],
                        FUNDS[12]: change[i],
                    },
                )
            )
    return pd.DataFrame(
        rows, columns=["asset_id", "timestamp", "available_at", *FUNDS[10:13]]
    )


def nport_flow_states(funds, holdings):
    """Weight visible three-month fund flows by prior disclosed long positions."""
    fields = ("balance_unit", "asset_category", "payoff_profile", "value")
    if not set(fields) <= set(holdings):
        return pd.DataFrame(
            columns=["asset_id", "timestamp", "available_at", FUNDS[13]]
        )
    h = holdings.loc[
        holdings.balance_unit.eq("NS")
        & holdings.asset_category.isin(("EC", "OTHER"))
        & holdings.payoff_profile.eq("Long")
    ]
    amounts = h.groupby(["accession", "cusip"], sort=False).value.sum(min_count=1)
    cusips = tuple(h.cusip.dropna().unique())
    n = len(cusips)
    if not n:
        return pd.DataFrame(
            columns=["asset_id", "timestamp", "available_at", FUNDS[13]]
        )
    positions = {
        accession: values.droplevel(0).reindex(cusips, fill_value=0).to_numpy(float)
        for accession, values in amounts.groupby(level=0, sort=False)
    }
    f = funds.loc[
        funds.available_at.notna() & funds.series_id.isin(holdings.series_id)
    ].copy()
    f["_amendment"] = f.filing_type.eq("NPORT-P/A")
    coordinates = ["series_id", "timestamp", "available_at"]
    rank = f.groupby(coordinates)._amendment.transform("max")
    f = f.loc[f._amendment.eq(rank)]
    # An ambiguous simultaneous amendment cannot identify a coherent holdings file.
    counts = f.groupby(coordinates).accession.transform("nunique")
    f.loc[counts.ne(1), "accession"] = pd.NA
    f = f.drop_duplicates(coordinates).sort_values(
        ["available_at", "timestamp"], kind="stable"
    )
    events = []
    for series, block in f.groupby("series_id", sort=False):
        block = block.sort_values(["timestamp", "available_at"], kind="stable")
        for row in block.itertuples():
            prior = block.loc[
                block.timestamp.le(
                    (row.timestamp.to_period("M") - 3).end_time.normalize()
                )
                & block.available_at.le(row.available_at)
            ]
            vector, rate = np.zeros(n), 0.0
            if len(prior):
                previous = prior.iloc[-1]
                flow = [
                    getattr(row, f"{kind}_month_{month}_usd")
                    for kind in ("sales", "reinvestment", "redemption")
                    for month in (1, 2, 3)
                ]
                if (
                    pd.notna(previous.accession)
                    and previous.net_assets_usd > 0
                    and np.isfinite(flow).all()
                ):
                    vector = positions.get(previous.accession, np.zeros(n))
                    vector = np.where(np.isfinite(vector) & (vector > 0), vector, 0)
                    rate = (sum(flow[:6]) - sum(flow[6:])) / previous.net_assets_usd
            events.append((row.available_at, row.timestamp, series, vector, rate))
    events.sort(key=lambda row: row[:2])
    active = {}
    weight, square, pressure = (np.zeros(n) for _ in range(3))
    rows = []
    for date, report, series, vector, rate in events:
        old_report, old, old_rate = active.get(
            series, (pd.Timestamp.min, np.zeros(n), 0)
        )
        if report < old_report:
            continue
        weight += vector - old
        square += vector**2 - old**2
        pressure += vector * rate - old * old_rate
        active[series] = report, vector, rate
        rows.append((date, weight.copy(), square.copy(), pressure.copy()))
    output = []
    for date, w, s, p in rows:
        score = np.divide(p * s, w**3, out=np.full(n, np.nan), where=w > 0)
        output.extend((cusip, date, date, score[i]) for i, cusip in enumerate(cusips))
    return pd.DataFrame(
        output, columns=["asset_id", "timestamp", "available_at", FUNDS[13]]
    ).drop_duplicates(["asset_id", "timestamp"], keep="last")


def calculate(decisions, sources, *, columns, parameters=None):
    parameters = {} if parameters is None else parameters
    work = decisions.sort_values(["timestamp", "asset_id"], kind="stable").reset_index(
        drop=True
    )
    result = work.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    for name in columns:
        result[name] = pd.Series(np.nan, index=result.index, dtype="Float64")
    missing = {
        name: [source for source in RECIPES[name] if source not in sources]
        for name in columns
    }
    usable = {name for name in columns if not missing[name]}
    session_days = session_groups = None
    if "trading_calendar" in sources:
        from quant_research_platform.markets.implementations.a_share.calendar import (
            CURRENT_MARKET_TYPES,
            next_trading_sessions,
            unified_trading_sessions,
        )

        calendar = sources["trading_calendar"].loc[
            sources["trading_calendar"].source_market_type.isin(CURRENT_MARKET_TYPES)
        ]
        session_days = unified_trading_sessions(calendar)
        successors = next_trading_sessions(calendar, session_days)
        session_groups = (
            pd.Series(session_days, index=session_days)
            .ne(pd.Series(successors, index=session_days).shift())
            .cumsum()
        )
    shares = (
        share_states(
            sources["public_fund_shares"].loc[
                sources["public_fund_shares"].asset_id.isin(
                    (set(work.asset_id) if usable & set(ETF[:4] + ETF[5:6]) else set())
                    | (
                        set(sources["public_etf_pcf_components"].fund_code)
                        if usable & set(ETF[6:])
                        else set()
                    )
                )
            ],
            session_days=session_days,
            session_groups=session_groups,
        )
        if usable & set(ETF[:4] + ETF[5:])
        else None
    )
    if usable & set(ETF[:4]):
        fields = tuple(usable & set(ETF[:4]))
        for name in fields:
            matched = latest_observations(
                shares.assign(available_at=shares[f"{name}_available_at"]),
                work,
                columns=(name,),
            )
            result[name] = matched[name].to_numpy()
    if ETF[5] in usable and {"secondary_flow_imbalance", "signal_session"} <= set(work):
        flow = work[
            ["asset_id", "signal_session", "timestamp", "secondary_flow_imbalance"]
        ].rename(
            columns={"timestamp": "_flow_available", "signal_session": "timestamp"}
        )
        state = shares.merge(flow, on=["asset_id", "timestamp"], how="inner")
        state["available_at"] = state[
            ["etf_share_change_1d_available_at", "_flow_available"]
        ].max(axis=1, skipna=False)
        state[ETF[5]] = state.etf_share_change_1d - state.secondary_flow_imbalance
        result[ETF[5]] = latest_observations(state, work, columns=(ETF[5],))[
            ETF[5]
        ].to_numpy()
    if ETF[4] in usable:
        prices = sources["public_domestic_etf_prices"][
            ["asset_id", "timestamp", "close", "available_at"]
        ].rename(columns={"available_at": "_price_available"})
        state = sources["public_fund_nav"][
            ["asset_id", "timestamp", "nav_cny", "available_at"]
        ].merge(prices, on=["asset_id", "timestamp"], how="inner")
        state["available_at"] = state[["available_at", "_price_available"]].max(
            axis=1, skipna=False
        )
        state[ETF[4]] = state.close / state.nav_cny.where(state.nav_cny > 0) - 1
        result[ETF[4]] = latest_observations(state, work, columns=(ETF[4],))[
            ETF[4]
        ].to_numpy()
    if usable & set(ETF[6:]):
        parts = sources["public_etf_pcf_components"].loc[
            :,
            [
                "fund_code",
                "timestamp",
                "available_at",
                "component_code",
                "component_kind",
                "component_quantity_unit",
                "component_quantity",
                "creation_unit_fund_shares",
            ],
        ]
        baskets = parts.groupby(["fund_code", "timestamp"], sort=False).indices
        equity_funds = parts.loc[
            parts.component_kind.eq("equity_security"), "fund_code"
        ].unique()
        complete_at = pd.Series(dtype="datetime64[ns]")
        if "public_etf_pcf_baskets" in sources:
            keys = ["fund_code", "timestamp"]
            grouped = parts.groupby(keys, sort=False)
            counts = grouped.component_code.agg(["size", "nunique"])
            counts["_components_available"] = grouped.available_at.max().where(
                grouped.available_at.count().eq(counts["size"])
            )
            headers = sources["public_etf_pcf_baskets"]
            proof = headers.merge(counts, on=keys, how="inner")
            proof = proof.loc[
                proof.declared_component_count.eq(proof["size"])
                & proof["size"].eq(proof["nunique"])
            ].copy()
            proof["_complete_at"] = proof[
                ["available_at", "_components_available"]
            ].max(axis=1, skipna=False)
            complete_at = proof.groupby(keys, sort=False)._complete_at.min()
        for date, at in work.groupby("timestamp", sort=False).indices.items():
            block = work.iloc[at]
            probes = pd.DataFrame({"asset_id": equity_funds, "timestamp": date})
            latest = latest_observations(
                shares.assign(available_at=shares._share_delta_1d_available_at),
                probes,
                columns=("_share_delta_1d",),
            )
            flow = probes.assign(_date=latest.observation_date.to_numpy())
            if "avg_volume_20d" not in block:
                continue
            for width in (1, 5):
                history = []
                for fund, cutoff in flow[["asset_id", "_date"]].itertuples(
                    index=False, name=None
                ):
                    tail = shares.loc[
                        shares.asset_id.eq(fund) & shares.timestamp.le(cutoff)
                    ].tail(width)
                    if (
                        len(tail) == width
                        and tail._share_delta_1d.notna().all()
                        and tail._share_delta_1d_available_at.le(date).all()
                    ):
                        history.append(
                            tail[["asset_id", "timestamp", "_share_delta_1d"]]
                        )
                if not history:
                    continue
                delta = pd.concat(history, ignore_index=True)
                positions = [
                    baskets.get((fund, day), np.empty(0, dtype=int))
                    for fund, day in delta[["asset_id", "timestamp"]].itertuples(
                        index=False, name=None
                    )
                ]
                eligible = parts.iloc[np.concatenate(positions)]
                eligible = eligible.loc[eligible.available_at.le(date)]
                p = delta.merge(
                    eligible,
                    left_on=["asset_id", "timestamp"],
                    right_on=["fund_code", "timestamp"],
                    how="inner",
                )
                p = p.loc[p.component_kind.eq("equity_security")]
                # Every constituent day must have its actual historical creation basket.
                covered = (
                    p[["asset_id", "timestamp"]]
                    .drop_duplicates()
                    .groupby("asset_id")
                    .size()
                )
                p = p.loc[p.asset_id.map(covered).eq(width)]
                units = pd.to_numeric(p.creation_unit_fund_shares)
                contributions = (
                    pd.to_numeric(p.component_quantity)
                    * p._share_delta_1d
                    / units.where((units > 0) & p.component_quantity_unit.eq("share"))
                ).groupby(p.component_code)
                quantity = contributions.sum(min_count=1).where(
                    contributions.count().eq(contributions.size())
                )
                mapped = block.asset_id.map(quantity)
                cohort = pd.MultiIndex.from_frame(
                    delta[["asset_id", "timestamp"]].rename(
                        columns={"asset_id": "fund_code"}
                    )
                )
                known = complete_at.reindex(cohort)
                if known.notna().all() and known.le(date).all():
                    # Absence is zero only in a complete, publication-visible PCF list.
                    mapped = mapped.where(block.asset_id.isin(quantity.index), 0)
                pressure = mapped / block.avg_volume_20d.where(block.avg_volume_20d > 0)
                name = f"stock_passive_flow_pressure_{width}d"
                if name in usable:
                    result.loc[at, name] = pressure.to_numpy()
                name = f"industry_passive_flow_pressure_{width}d"
                if name in usable and {
                    "industry_standard_id",
                    "industry_level1_code",
                } <= set(block):
                    result.loc[at, name] = (
                        pressure.groupby(
                            [block.industry_standard_id, block.industry_level1_code],
                            dropna=False,
                        )
                        .transform(lambda s: s.sum(min_count=len(s)))
                        .where(
                            block.industry_standard_id.notna()
                            & block.industry_level1_code.notna()
                        )
                        .to_numpy(dtype=float, na_value=np.nan)
                    )
    financial = (
        financial_states(sources["public_fund_financial_cells"])
        if usable & set((*FUNDS[4:9],))
        else None
    )
    if financial is not None and not financial.empty:
        dates = work.timestamp.drop_duplicates().tolist()
        aggregates = (
            _financial_history_states(financial)
            if "cn_fund_redemption_stress" in usable
            else financial
        )
        report_keys = iter_latest_report_keys(aggregates, dates, entity="asset_id")
        for _date, at in work.groupby("timestamp", sort=False).indices.items():
            known = next(report_keys)
            latest = known.merge(
                aggregates,
                left_on=["asset_id", "_report", "_available"],
                right_on=["asset_id", "timestamp", "available_at"],
                how="inner",
                suffixes=("_decision", ""),
            )
            for name in usable & set(FUNDS[4:8]):
                field = (
                    "total_assets" if name in FUNDS[4:6] else "period_begin_fund_shares"
                )
                valid = latest[name].notna() & latest[field].gt(0)
                if valid.any():
                    result.loc[at, name] = np.average(
                        latest.loc[valid, name], weights=latest.loc[valid, field]
                    )
    holdings = sources.get("public_fund_holdings")
    ownership = set((*FUNDS[:4], FUNDS[8], FUNDS[9], FUNDS[15]))
    if holdings is not None and usable & ownership:
        headers = complete_mainland_stock_reports(
            sources["public_fund_holdings_reports"]
        )
        h = holdings.loc[
            holdings.stock_market.eq("CN_MAINLAND")
            & holdings.report_version_identity.isin(headers.report_version_identity)
        ]
        reports = iter_latest_report_keys(
            headers,
            work.timestamp.drop_duplicates().tolist(),
            entity="economic_portfolio_id",
        )
        for date, at in work.groupby("timestamp", sort=False).indices.items():
            block = work.iloc[at]
            keys = next(reports)
            latest_headers = keys.merge(
                headers,
                left_on=["economic_portfolio_id", "_report", "_available"],
                right_on=["economic_portfolio_id", "timestamp", "available_at"],
                how="inner",
                suffixes=("_decision", ""),
            )
            if latest_headers.empty:
                continue
            known = h.loc[
                h.report_version_identity.isin(latest_headers.report_version_identity)
            ]
            known = (
                known.groupby(["economic_portfolio_id", "stock_code"], sort=False)
                .shares_held.sum(min_count=1)
                .reset_index()
            )
            total = known.groupby("stock_code", sort=False).shares_held.sum(min_count=1)
            squared = (
                (known.shares_held**2)
                .groupby(known.stock_code, sort=False)
                .sum(min_count=1)
            )
            hhi = squared / total.where(total > 0) ** 2
            values = block.asset_id.map(total).fillna(0)
            for name in usable & ownership:
                if name == FUNDS[0] and "a_float_shares" in block:
                    result.loc[at, name] = (
                        values / block.a_float_shares.where(block.a_float_shares > 0)
                    ).to_numpy()
                elif name == FUNDS[1]:
                    result.loc[at, name] = block.asset_id.map(hhi).to_numpy()
                elif name == FUNDS[2] and "a_float_shares" in block:
                    floats = block.set_index("asset_id").a_float_shares
                    fractions = known.shares_held / known.stock_code.map(floats).where(
                        lambda x: x > 0
                    )
                    others = (
                        fractions.groupby(known.economic_portfolio_id).transform(
                            lambda s: s.sum(min_count=len(s))
                        )
                        - fractions
                    )
                    grouped = (fractions * others).groupby(known.stock_code)
                    exposure = grouped.sum(min_count=1).where(
                        grouped.count().eq(grouped.size())
                    )
                    result.loc[at, name] = (
                        block.asset_id.map(exposure)
                        .where(block.asset_id.isin(exposure.index), 0)
                        .to_numpy()
                    )
                elif name == FUNDS[3]:
                    earlier = headers.loc[headers.available_at.le(date)].merge(
                        keys[["economic_portfolio_id", "_report"]],
                        on="economic_portfolio_id",
                        how="inner",
                    )
                    earlier = earlier.loc[earlier.timestamp.lt(earlier._report)]
                    previous_period = earlier.groupby(
                        "economic_portfolio_id"
                    ).timestamp.transform("max")
                    earlier = earlier.loc[earlier.timestamp.eq(previous_period)]
                    previous_release = earlier.groupby(
                        "economic_portfolio_id"
                    ).available_at.transform("max")
                    earlier = earlier.loc[earlier.available_at.eq(previous_release)]
                    if earlier.empty:
                        continue
                    common = known.loc[
                        known.economic_portfolio_id.isin(earlier.economic_portfolio_id)
                    ]
                    earlier = h.loc[
                        h.report_version_identity.isin(earlier.report_version_identity)
                    ]
                    changes = (
                        common.groupby("stock_code")
                        .shares_held.sum(min_count=1)
                        .subtract(
                            earlier.groupby("stock_code").shares_held.sum(min_count=1),
                            fill_value=0,
                        )
                    )
                    if "a_float_shares" in block:
                        result.loc[at, name] = (
                            block.asset_id.map(changes).fillna(0)
                            / block.a_float_shares.where(block.a_float_shares > 0)
                        ).to_numpy()
                elif name == FUNDS[9] and "peak_drawdown_20d" in block:
                    result.loc[at, name] = (
                        block.asset_id.map(hhi) * block.peak_drawdown_20d
                    ).to_numpy()
                elif name == FUNDS[15] and {"amihud_20d", "amihud_60d"} <= set(block):
                    result.loc[at, name] = (
                        block.asset_id.map(hhi)
                        * (
                            block.amihud_20d
                            / block.amihud_60d.where(block.amihud_60d > 0)
                            - 1
                        ).clip(lower=0)
                    ).to_numpy()
            if (
                FUNDS[8] in usable
                and financial is not None
                and not financial.empty
                and "avg_volume_20d" in block
            ):
                visible = financial.loc[financial.available_at.le(date)]
                current = visible.loc[
                    visible.timestamp.eq(
                        visible.groupby("asset_id").timestamp.transform("max")
                    )
                ]
                current = current.loc[
                    current.available_at.eq(
                        current.groupby("asset_id").available_at.transform("max")
                    )
                ]
                prior = headers.merge(
                    current[
                        [
                            "asset_id",
                            "timestamp",
                            "available_at",
                            "_redemption_rate",
                            "_flow_period_start",
                        ]
                    ],
                    left_on="economic_portfolio_id",
                    right_on="asset_id",
                    suffixes=("", "_flow"),
                )
                prior = prior.loc[
                    prior.timestamp.lt(prior._flow_period_start)
                    & prior.available_at.le(prior._flow_period_start)
                ]
                prior = prior.loc[
                    prior.timestamp.eq(
                        prior.groupby("economic_portfolio_id").timestamp.transform(
                            "max"
                        )
                    )
                ]
                prior = prior.loc[
                    prior.available_at.eq(
                        prior.groupby("economic_portfolio_id").available_at.transform(
                            "max"
                        )
                    )
                ]
                if prior.empty:
                    continue
                prior = h.merge(
                    prior[["report_version_identity", "_redemption_rate"]],
                    on="report_version_identity",
                    how="inner",
                )
                grouped = (prior.shares_held * prior._redemption_rate).groupby(
                    prior.stock_code
                )
                quantities = grouped.sum(min_count=1).where(
                    grouped.count().eq(grouped.size())
                )
                mapped = block.asset_id.map(quantities).where(
                    block.asset_id.isin(quantities.index), 0
                )
                result.loc[at, FUNDS[8]] = (
                    mapped / block.avg_volume_20d.where(block.avg_volume_20d > 0)
                ).to_numpy()
    if FUNDS[14] in usable:
        f = sources["public_nport_funds"]
        states = f.rename(columns={"series_id": "_series"})
        reports = iter_latest_report_keys(
            states, work.timestamp.drop_duplicates().tolist(), entity="_series"
        )
        for _date, at in work.groupby("timestamp", sort=False).indices.items():
            known = next(reports).merge(
                states,
                left_on=["_series", "_report", "_available"],
                right_on=["_series", "timestamp", "available_at"],
                how="inner",
            )
            valid = known.cash_outside_holdings_usd.notna() & known.total_assets_usd.gt(
                0
            )
            if valid.any():
                result.loc[at, FUNDS[14]] = (
                    known.loc[valid, "cash_outside_holdings_usd"].sum()
                    / known.loc[valid, "total_assets_usd"].sum()
                )
    if usable & set(FUNDS[10:13]):
        states = us_ownership_states(
            sources["public_13f_holdings"],
            sources["public_13f_filings"],
            assets=parameters.get("us_assets", {}),
        )
        if not states.empty:
            tickers = work.asset_id.map(parameters.get("asset_us_tickers", {})).fillna(
                work.asset_id
            )
            query = work.assign(asset_id=tickers)
            fields = tuple(usable & set(FUNDS[10:13]))
            result[list(fields)] = latest_observations(states, query, columns=fields)[
                list(fields)
            ].to_numpy()
    if FUNDS[13] in usable:
        states = nport_flow_states(
            sources["public_nport_funds"], sources["public_nport_holdings"]
        )
        if not states.empty:
            ids = (
                work.asset_id.map(parameters.get("asset_us_tickers", {}))
                .fillna(work.asset_id)
                .map(parameters.get("us_assets", {}))
            )
            query = work.assign(asset_id=ids)
            result[FUNDS[13]] = latest_observations(
                states, query, columns=(FUNDS[13],)
            )[FUNDS[13]].to_numpy()
    result.attrs["missing_source_dependencies"] = {
        k: v for k, v in missing.items() if v
    }
    return result
