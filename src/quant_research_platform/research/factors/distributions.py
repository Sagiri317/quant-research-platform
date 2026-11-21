"""Stock cash-dividend yields at actual ex dates and original-share prices."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition

RECIPES = dict.fromkeys(
    ("cash_dividend_yield_1y", "persistent_dividend_low_risk"),
    ("corporate_actions", "market_panel_1d"),
)
UNITS = {
    "cash_dividend_yield_1y": "decimal_return",
    "persistent_dividend_low_risk": "dimensionless",
}


def definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            family="cash_distributions",
            output_column=name,
            input_columns=(
                "corporate_actions",
                "close",
                "execution_session",
                "listing_date",
                *(
                    ("volatility_60d",)
                    if name == "persistent_dividend_low_risk"
                    else ()
                ),
            ),
            required_input_data_id="cash_distribution_factors",
            description=(
                "Sum of 365-day implemented dividends per original share, each "
                "divided by its pre-ex close."
                if name == "cash_dividend_yield_1y"
                else "Trailing yield per unit of annualized 60-session volatility, "
                "with a positive dividend in each of three consecutive 365-day windows."
            ),
            timing_semantics="ex_date_and_date_only_announcement_visible_at_decision",
            null_policy="unknown_cash_amount_or_price_or_unproved_dividend_history",
        )
        for name in RECIPES
    )


def calculate(decisions, sources, *, columns, parameters=None):
    result = decisions[["market_id", "asset_id", "timestamp"]].copy()
    for name in columns:
        result[name] = np.nan
    if "corporate_actions" not in sources or "market_panel_1d" not in sources:
        return result
    events = sources["corporate_actions"]
    events = events.loc[events.kind.eq("CA")].rename(
        columns={"source_asset_id": "asset_id", "timestamp": "ex_date"}
    )
    if events.empty:
        return result
    # The preceding close must belong to the session whose actual successor is ex date.
    reference = decisions[
        ["asset_id", "signal_session", "execution_session", "listing_date"]
    ].dropna(subset=["execution_session"])
    events = events.merge(
        reference,
        left_on=["asset_id", "ex_date"],
        right_on=["asset_id", "execution_session"],
        how="left",
        validate="many_to_one",
    )
    prices = sources["market_panel_1d"][
        ["asset_id", "timestamp", "listing_date", "close"]
    ].rename(columns={"timestamp": "signal_session"})
    events = events.merge(
        prices,
        on=["asset_id", "signal_session", "listing_date"],
        how="left",
        validate="many_to_one",
    )
    by_asset = {asset: block for asset, block in events.groupby("asset_id", sort=False)}
    for (asset, life), block in decisions.groupby(
        ["asset_id", "listing_date"], sort=False
    ):
        if asset not in by_asset:
            continue
        block = block.sort_values("signal_session")
        sessions = pd.DatetimeIndex(block.signal_session)
        decision_at = pd.DatetimeIndex(block.timestamp)
        size = len(block)
        sums = np.zeros(size + 1)
        counts = np.zeros((3, size + 1), dtype=int)
        unknown = np.zeros(size + 1, dtype=int)
        for event in by_asset[asset].itertuples():
            if event.ex_date < life or event.ex_date > sessions[-1]:
                continue
            if pd.isna(event.announcement_date):
                continue
            visible = decision_at.searchsorted(
                event.announcement_date + pd.Timedelta(days=1)
            )
            limits = sessions.searchsorted(
                [event.ex_date + pd.Timedelta(days=365 * year) for year in range(4)]
            )
            start, stop = max(limits[0], visible), limits[1]
            known = (
                pd.notna(event.listing_date)
                and event.listing_date == life
                and pd.notna(event.amount)
                and event.amount >= 0
                and pd.notna(event.close)
                and event.close > 0
            )
            if start < stop:
                if known:
                    event_yield = event.amount / event.close
                    sums[start] += event_yield
                    sums[stop] -= event_yield
                else:
                    unknown[start] += 1
                    unknown[stop] -= 1
            # Historical payout evidence depends on cash, not a pre-ex price.
            if pd.notna(event.amount) and event.amount > 0:
                for year in range(3):
                    start, stop = max(limits[year], visible), limits[year + 1]
                    if start < stop:
                        counts[year, start] += 1
                        counts[year, stop] -= 1
        paid = counts[:, :-1].cumsum(axis=1) > 0
        full_year = sessions >= max(life, sessions[0]) + pd.Timedelta(days=365)
        yield_ = np.where(
            paid[0] & full_year & (unknown[:-1].cumsum() == 0),
            sums[:-1].cumsum(),
            np.nan,
        )
        if "cash_dividend_yield_1y" in columns:
            result.loc[block.index, "cash_dividend_yield_1y"] = yield_
        if "persistent_dividend_low_risk" in columns and "volatility_60d" in block:
            volatility = block.volatility_60d.to_numpy(
                dtype=float, na_value=np.nan
            ) * np.sqrt(252)
            result.loc[block.index, "persistent_dividend_low_risk"] = np.divide(
                yield_,
                volatility,
                out=np.full(size, np.nan),
                where=paid.all(axis=0) & (volatility > 0),
            )
    return result
