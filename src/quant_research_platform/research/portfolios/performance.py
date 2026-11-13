"""Shared performance formulas for daily observations of native stock accounts."""

import numpy as np

from quant_research_platform.compute.portfolio_economics import _nav_return_roundoff
from quant_research_platform.research.statistics import (
    _return_sharpes,
    return_standard_deviation,
)


def performance_rows(
    selected, *, capital, economic, daily_complete, sample_role, extra_values=None
):
    """Use one set of formulas on caller-established daily account observations."""
    candidate_id = capital.candidate_id
    access = sample_role
    returns_defined = selected.return_status.eq("DEFINED").all()
    count = len(selected)
    deviation = float(
        return_standard_deviation(
            selected.net_return,
            roundoff=_nav_return_roundoff(
                selected.nav_before, selected.nav_end, selected.net_return
            ),
        )
    )
    endpoint_return = (
        selected.nav_end.iloc[-1] - capital.initial_capital
    ) / capital.initial_capital
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cagr = float(np.expm1(np.log1p(endpoint_return) * (252 / count)))
    if not np.isfinite(cagr):
        cagr = np.nan
    values = {
        "terminal_nav": selected.nav_end.iloc[-1],
        "capital_return": np.nan
        if economic["capital_return"] is None
        else economic["capital_return"],
        "cagr": cagr,
        "annualized_volatility": deviation * np.sqrt(252),
        "sharpe": float(
            _return_sharpes(
                selected.net_return,
                roundoff=_nav_return_roundoff(
                    selected.nav_before, selected.nav_end, selected.net_return
                ),
            )
            * np.sqrt(252)
        ),
        "max_drawdown": float(
            (
                selected.nav_end
                / selected.nav_end.cummax().clip(lower=capital.initial_capital)
                - 1
            ).min()
        ),
        "mean_cash_weight_post": float(selected.cash_weight_post.mean(skipna=False)),
        "terminal_cash_weight_post": float(selected.cash_weight_post.iloc[-1]),
        "terminal_cash_post": float(selected.cash_post.iloc[-1]),
        "target_turnover": float(selected.target_turnover.mean(skipna=False)),
        "requested_turnover": float(selected.requested_turnover.mean(skipna=False)),
        "realized_turnover": float(selected.realized_turnover.mean(skipna=False)),
        "gross_pnl": economic["gross_pnl"],
        "net_pnl": economic["net_pnl"],
        "equity_rounding_pnl": np.nan
        if economic["equity_rounding_pnl"] is None
        else economic["equity_rounding_pnl"],
        "cost": economic["fees"],
    }
    if extra_values is not None:
        values.update(extra_values)
    rows = []
    undefined_reason = (
        "nonpositive_start_nav"
        if selected.nav_before.eq(0).any()
        else "nonfinite_result"
        if not returns_defined
        or selected[
            [
                "gross_return",
                "cost_drag",
                "target_turnover",
                "requested_turnover",
                "realized_turnover",
                "cash_weight_post",
            ]
        ]
        .isna()
        .any()
        .any()
        else "statistic_not_identified"
    )
    for name, value in values.items():
        missing_daily = not daily_complete and name in {
            "cagr",
            "annualized_volatility",
            "sharpe",
        }
        if missing_daily:
            value = np.nan
        rows.append(
            dict(
                candidate_id=candidate_id,
                metric_id=name,
                value=value if np.isfinite(value) else np.nan,
                status="DEFINED" if np.isfinite(value) else "UNDEFINED",
                reason=""
                if np.isfinite(value)
                else economic["equity_rounding_pnl_reason"]
                if name == "equity_rounding_pnl"
                else "incomplete_daily_axis"
                if missing_daily
                else "nonfinite_result"
                if np.isinf(value)
                or name
                in {
                    "capital_return",
                    "cagr",
                }
                or name.endswith(("_capital_return", "_target_notional"))
                else undefined_reason,
                observation_count=count,
                sample_role=access,
            )
        )
    return rows
