"""Pure common fundamental factor mathematics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
)

from .definitions import FUNDAMENTAL_FACTORS


def ttm_from_ytd(
    current_ytd: object,
    prior_fy: object,
    prior_same_period_ytd: object,
) -> pd.Series:
    """Construct TTM as YTD_t + FY_{t-1} - YTD_{t-1,same period}."""

    values = tuple(
        _numeric(value, error="invalid_ttm_value")
        for value in (current_ytd, prior_fy, prior_same_period_ytd)
    )
    return values[0] + values[1] - values[2]


def construct_ttm_events(
    events: pd.DataFrame,
    *,
    value_column: str = "value",
) -> pd.DataFrame:
    """Add event-level TTM values before any daily as-of join."""

    required = {
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "version_identity",
        value_column,
    }
    if not required.issubset(events.columns):
        raise ValueError("ttm_event_columns_invalid")
    result = events.copy()
    for column in ("report_period", "available_at"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result["version_identity"] = strict_string_series(
        result["version_identity"], error="ttm_version_identity_invalid"
    )
    result[value_column] = strict_finite_numeric_series(
        result[value_column], error="invalid_ttm_value", allow_null=True
    )
    event_key = [
        "asset_id",
        "metric_id",
        "report_period",
        "available_at",
        "version_identity",
    ]
    if result.duplicated(event_key, keep=False).any():
        raise ValueError("ttm_event_key_duplicate")
    result["ttm_value"] = np.nan
    for (_asset_id, _metric_id), positions in result.groupby(
        ["asset_id", "metric_id"], sort=False
    ).groups.items():
        group = result.loc[positions].sort_values(
            ["available_at", "report_period", "version_identity"],
            kind="mergesort",
        )
        for position, row in group.iterrows():
            period = pd.Timestamp(row["report_period"])
            value = row[value_column]
            if pd.isna(value):
                continue
            if period.month == 12:
                ttm = value
            else:
                cutoff = pd.Timestamp(row["available_at"])
                prior_fy = _latest_legal_event_value(
                    group,
                    report_period=pd.Timestamp(year=period.year - 1, month=12, day=31),
                    cutoff=cutoff,
                    value_column=value_column,
                )
                prior_ytd = _latest_legal_event_value(
                    group,
                    report_period=pd.Timestamp(
                        year=period.year - 1, month=period.month, day=period.day
                    ),
                    cutoff=cutoff,
                    value_column=value_column,
                )
                if prior_fy is None or prior_ytd is None:
                    continue
                ttm = value + prior_fy - prior_ytd
            result.loc[position, "ttm_value"] = ttm
    return result


def _latest_legal_event_value(
    events: pd.DataFrame,
    *,
    report_period: pd.Timestamp,
    cutoff: pd.Timestamp,
    value_column: str,
) -> float | None:
    candidates = events.loc[
        events["report_period"].eq(report_period) & events["available_at"].le(cutoff)
    ]
    if candidates.empty:
        return None
    row = candidates.sort_values(
        ["available_at", "version_identity"], kind="mergesort"
    ).iloc[-1]
    return None if pd.isna(row[value_column]) else float(row[value_column])


def calculate_fundamental_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate the 19 registered factors from PIT/derived accounting facts."""

    requested = tuple(dict.fromkeys(factor_columns or FUNDAMENTAL_FACTORS))
    unknown = sorted(set(requested).difference(FUNDAMENTAL_FACTORS))
    if unknown:
        raise ValueError("unknown_fundamental_factor: " + ", ".join(unknown))
    required = {"asset_id", "timestamp"}
    if not required.issubset(panel.columns):
        raise ValueError("fundamental_factor_keys_invalid")
    result = panel.loc[:, ("asset_id", "timestamp")].copy()
    values = {column: _series(panel, column) for column in _required_columns(panel)}

    assets = values.get("total_assets")
    prior_assets = _first(values, "prior_year_total_assets")
    avg_assets = _pick(values, panel, "average_total_assets", "avg_total_assets")
    parent_equity = values.get("parent_equity")
    prior_equity = _first(values, "prior_year_parent_equity")
    avg_equity = _pick(values, panel, "average_parent_equity", "avg_parent_equity")
    market_cap = _first(values, "total_market_cap", "market_cap")
    revenue = _ttm(values, panel, "operating_revenue")
    prior_revenue = _pick(values, panel, "prior_year_operating_revenue_ttm")
    operating_cost = _ttm(values, panel, "operating_cost")
    consolidated_income = _ttm(values, panel, "consolidated_net_income")
    parent_income = _ttm(values, panel, "parent_net_income")
    prior_parent_income = _pick(values, panel, "prior_year_parent_net_income_ttm")
    cfo = _ttm(values, panel, "operating_cash_flow")
    capex = _ttm(values, panel, "capex_cash_paid")

    gross_profit = _finite(revenue - operating_cost)
    debt = _sum_existing(
        values,
        (
            "short_term_borrowings",
            "current_portion_noncurrent_liabilities",
            "long_term_borrowings",
            "bonds_payable",
        ),
        index=panel.index,
    )
    cash = values.get("cash")
    net_debt = _finite(debt - cash)

    formulas = {
        "earnings_yield": _ratio(parent_income, market_cap),
        "book_to_market": _ratio(parent_equity, market_cap),
        "sales_to_price": _ratio(revenue, market_cap),
        "cashflow_to_price": _ratio(cfo, market_cap),
        "roe": _ratio(parent_income, avg_equity),
        "roa": _ratio(consolidated_income, avg_assets),
        "gross_profitability": _ratio(gross_profit, avg_assets),
        "gross_margin": _ratio(gross_profit, revenue),
        "cfo_to_assets": _ratio(cfo, avg_assets),
        "accruals": _ratio(_finite(consolidated_income - cfo), avg_assets),
        "asset_growth_yoy": _ratio(assets, prior_assets) - 1.0,
        "revenue_growth_ttm_yoy": _ratio(revenue, prior_revenue) - 1.0,
        "earnings_change_to_assets": _ratio(
            _finite(parent_income - prior_parent_income), avg_assets
        ),
        "parent_equity_growth_yoy": _ratio(parent_equity, prior_equity) - 1.0,
        "capex_to_assets": _ratio(capex.where(capex.ge(0.0)), avg_assets),
        "liabilities_to_assets": _ratio(values.get("total_liabilities"), assets),
        "debt_to_assets": _ratio(debt, assets),
        "cash_to_assets": _ratio(cash, assets),
        "net_debt_to_assets": _ratio(net_debt, assets),
    }
    for column in requested:
        result[column] = _finite(formulas[column]).astype("float64")
    return result


def _required_columns(panel: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in (
            "total_assets",
            "total_liabilities",
            "parent_equity",
            "cash",
            "short_term_borrowings",
            "current_portion_noncurrent_liabilities",
            "long_term_borrowings",
            "bonds_payable",
            "operating_revenue",
            "operating_cost",
            "consolidated_net_income",
            "parent_net_income",
            "operating_cash_flow",
            "capex_cash_paid",
            "total_market_cap",
            "market_cap",
            "beginning_total_assets",
            "prior_year_total_assets",
            "average_total_assets",
            "avg_total_assets",
            "beginning_parent_equity",
            "prior_year_parent_equity",
            "average_parent_equity",
            "avg_parent_equity",
            "prior_year_operating_revenue_ttm",
            "prior_year_parent_net_income_ttm",
            "operating_revenue_ttm",
            "operating_cost_ttm",
            "consolidated_net_income_ttm",
            "parent_net_income_ttm",
            "operating_cash_flow_ttm",
            "capex_cash_paid_ttm",
        )
        if column in panel.columns
    )


def _series(panel: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in panel:
        return None
    return _numeric(panel[column], error=f"invalid_fundamental_value: {column}")


def _pick(
    values: dict[str, pd.Series | None],
    panel: pd.DataFrame,
    *names: str,
) -> pd.Series | None:
    for name in names:
        if name in panel:
            value = values.get(name)
            return value if value is not None else _series(panel, name)
    return None


def _first(values: dict[str, pd.Series | None], *names: str) -> pd.Series | None:
    for name in names:
        value = values.get(name)
        if value is not None:
            return value
    return None


def _ttm(
    values: dict[str, pd.Series | None], panel: pd.DataFrame, name: str
) -> pd.Series | None:
    direct = _pick(values, panel, f"{name}_ttm")
    if direct is not None:
        return direct
    return values.get(name)


def _sum_existing(
    values: dict[str, pd.Series | None],
    names: tuple[str, ...],
    *,
    index: pd.Index,
) -> pd.Series:
    present = [values[name] for name in names if values.get(name) is not None]
    if not present:
        return pd.Series(np.nan, index=index, dtype="float64")
    result = present[0]
    for value in present[1:]:
        result = result + value
    return _finite(result)


def _ratio(
    numerator: pd.Series | None,
    denominator: pd.Series | None,
) -> pd.Series:
    if numerator is None or denominator is None:
        if numerator is not None:
            return pd.Series(np.nan, index=numerator.index, dtype="float64")
        if denominator is not None:
            return pd.Series(np.nan, index=denominator.index, dtype="float64")
        return pd.Series(dtype="float64")
    valid_denominator = _finite(denominator).where(_finite(denominator).gt(0.0))
    return _finite(numerator).div(valid_denominator)


def _numeric(value: object, *, error: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return strict_finite_numeric_series(value, error=error, allow_null=True).astype(
            "float64"
        )
    return pd.Series(value, dtype="float64")


def _finite(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)
