"""Market-independent continuous-weight execution and accounting primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.research.portfolios.schedules import rebalance_dates


@dataclass(frozen=True)
class ExecutionAccounting:
    target_weights: pd.Series
    pre_trade_weights: pd.Series
    realized_weights: pd.Series
    cash_weight: float
    executions: pd.DataFrame
    target_turnover: float
    realized_turnover: float
    total_cost: float


@dataclass(frozen=True)
class AlphaCalibration:
    slope: float
    intercept: float
    expected_alpha: pd.Series
    valid_rebalance_dates: int
    scheduled_rebalance_dates: int
    benchmark_only: bool


def alpha_score(scores: pd.Series) -> pd.Series:
    """Center average ranks so each valid cross-section has mean zero."""

    values = pd.to_numeric(scores, errors="raise").astype("float64")
    if values.empty or values.isna().any() or not np.isfinite(values).all():
        raise ValueError("alpha_scores_require_finite_values")
    n = len(values)
    return (values.rank(method="average") - (n + 1.0) / 2.0) / n


def calibrate_expected_alpha(
    frame: pd.DataFrame,
    *,
    alpha_column: str = "alpha_score",
    return_column: str = "forward_return_1d",
    schedule: str = "weekly-last-trading-day",
    minimum_coverage: float = 0.8,
    minimum_valid_dates: int = 6,
) -> AlphaCalibration:
    """Calibrate expected alpha with equal-date weighted pooled OLS."""

    required = {"timestamp", alpha_column, return_column}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("alpha_calibration_columns_invalid")
    if (
        type(minimum_coverage) is not float
        or not np.isfinite(minimum_coverage)
        or not 0.0 < minimum_coverage <= 1.0
        or type(minimum_valid_dates) is not int
        or minimum_valid_dates <= 0
    ):
        raise ValueError("alpha_calibration_configuration_invalid")
    dates = pd.to_datetime(frame["timestamp"], errors="raise").dt.normalize()
    scheduled = pd.DatetimeIndex(dates.drop_duplicates()).sort_values()
    if schedule not in {
        "daily",
        "weekly-last-trading-day",
        "monthly-last-trading-day",
    }:
        raise ValueError("alpha_calibration_schedule_invalid")
    scheduled = rebalance_dates(scheduled, schedule)
    alpha = pd.to_numeric(frame[alpha_column], errors="raise").astype("float64")
    returns = pd.to_numeric(frame[return_column], errors="raise").astype("float64")
    valid = alpha.notna() & returns.notna() & np.isfinite(alpha) & np.isfinite(returns)
    valid_dates = dates.loc[valid].isin(scheduled)
    date_count = int(dates.loc[valid & valid_dates].nunique())
    enough_coverage = date_count >= minimum_valid_dates and (
        date_count / max(1, len(scheduled)) >= minimum_coverage
    )
    if not enough_coverage:
        return AlphaCalibration(
            slope=0.0,
            intercept=0.0,
            expected_alpha=pd.Series(0.0, index=frame.index, dtype="float64"),
            valid_rebalance_dates=date_count,
            scheduled_rebalance_dates=len(scheduled),
            benchmark_only=True,
        )
    selected = valid & valid_dates
    counts = dates.loc[selected].groupby(dates.loc[selected]).transform("size")
    weights = 1.0 / (date_count * counts.to_numpy(dtype="float64"))
    design = np.column_stack(
        (
            np.ones(int(selected.sum()), dtype="float64"),
            alpha.loc[selected].to_numpy(dtype="float64"),
        )
    )
    fit = np.sqrt(weights)[:, None] * design
    target = np.sqrt(weights) * returns.loc[selected].to_numpy(dtype="float64")
    coefficients, *_ = np.linalg.lstsq(fit, target, rcond=None)
    slope = float(coefficients[1])
    intercept = float(coefficients[0])
    benchmark_only = slope <= 0.0
    expected = pd.Series(0.0, index=frame.index, dtype="float64")
    if not benchmark_only:
        expected.loc[valid] = slope * alpha.loc[valid]
    return AlphaCalibration(
        slope=slope,
        intercept=intercept,
        expected_alpha=expected,
        valid_rebalance_dates=date_count,
        scheduled_rebalance_dates=len(scheduled),
        benchmark_only=benchmark_only,
    )


def one_way_turnover(
    target_weights: pd.Series,
    current_weights: pd.Series,
    *,
    include_cash: bool = True,
) -> float:
    assets = target_weights.index.union(current_weights.index)
    target = target_weights.reindex(assets, fill_value=0.0).astype("float64")
    current = current_weights.reindex(assets, fill_value=0.0).astype("float64")
    if (
        not np.isfinite(target.to_numpy()).all()
        or not np.isfinite(current.to_numpy()).all()
    ):
        raise ValueError("turnover_weights_invalid")
    stock = 0.5 * (target - current).abs().sum()
    if not include_cash:
        return float(stock)
    target_cash = 1.0 - float(target.sum())
    current_cash = 1.0 - float(current.sum())
    return float(stock + 0.5 * abs(target_cash - current_cash))


def sqrt_impact_cost(
    trade_notional: float,
    average_daily_volume: float,
    *,
    impact_bps: float = 25.0,
) -> float:
    if (
        not np.isfinite(trade_notional)
        or not np.isfinite(average_daily_volume)
        or trade_notional < 0.0
        or average_daily_volume <= 0.0
    ):
        raise ValueError("impact_cost_inputs_invalid")
    if not np.isfinite(impact_bps) or impact_bps < 0.0:
        raise ValueError("impact_cost_inputs_invalid")
    return float(
        trade_notional
        * impact_bps
        / 10_000.0
        * np.sqrt(trade_notional / average_daily_volume)
    )


def execute_continuous_target(
    current_weights: pd.Series,
    target_weights: pd.Series,
    *,
    nav: float,
    average_daily_volume: pd.Series,
    can_enter: pd.Series,
    can_exit: pd.Series,
    participation: float = 0.05,
    commission_bps: float = 3.0,
    slippage_bps: float = 5.0,
    impact_bps: float = 25.0,
) -> ExecutionAccounting:
    """Sell first, then buy with ADV/cash limits; unfilled buys are not queued."""

    assets = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(assets, fill_value=0.0).astype("float64")
    target = target_weights.reindex(assets, fill_value=0.0).astype("float64")
    adv = average_daily_volume.reindex(assets)
    enter = can_enter.reindex(assets, fill_value=False).astype(bool)
    exit_ = can_exit.reindex(assets, fill_value=False).astype(bool)
    if (
        not np.isfinite(nav)
        or not np.isfinite(participation)
        or nav <= 0.0
        or participation <= 0.0
        or participation > 1.0
    ):
        raise ValueError("execution_inputs_invalid")
    if adv.isna().any() or adv.le(0.0).any() or not np.isfinite(adv).all():
        raise ValueError("execution_adv_invalid")
    realized = current.copy()
    initial_cash = 1.0 - float(current.sum())
    cash = initial_cash
    cash_before_cost = initial_cash
    execution_rows: list[dict[str, object]] = []
    total_cost = 0.0
    for side in ("sell", "buy"):
        for asset in assets:
            desired_delta = float(target[asset] - current[asset])
            if (side == "sell" and desired_delta >= 0.0) or (
                side == "buy" and desired_delta <= 0.0
            ):
                continue
            allowed = exit_[asset] if side == "sell" else enter[asset]
            if not allowed:
                execution_rows.append(
                    {
                        "asset_id": asset,
                        "side": side,
                        "target_delta": desired_delta,
                        "realized_delta": 0.0,
                        "blocked": True,
                    }
                )
                continue
            limit = float(adv[asset]) * participation / nav
            delta = min(abs(desired_delta), limit)
            if side == "buy":
                delta = min(delta, max(0.0, cash))
            delta = -delta if side == "sell" else delta
            quantity = abs(delta) * nav
            cost = quantity * (commission_bps + slippage_bps) / 10_000.0
            cost += sqrt_impact_cost(quantity, float(adv[asset]), impact_bps=impact_bps)
            realized[asset] += delta
            cash -= delta
            cash_before_cost -= delta
            cash -= cost / nav
            total_cost += cost
            execution_rows.append(
                {
                    "asset_id": asset,
                    "side": side,
                    "target_delta": desired_delta,
                    "realized_delta": delta,
                    "blocked": False,
                }
            )
    realized_stock_turnover = sum(
        abs(float(row["realized_delta"])) for row in execution_rows
    )
    realized_turnover = 0.5 * (
        realized_stock_turnover + abs(cash_before_cost - initial_cash)
    )
    executions = pd.DataFrame.from_records(
        execution_rows,
        columns=("asset_id", "side", "target_delta", "realized_delta", "blocked"),
    )
    return ExecutionAccounting(
        target_weights=target,
        pre_trade_weights=current,
        realized_weights=realized,
        cash_weight=float(cash),
        executions=executions,
        target_turnover=one_way_turnover(target, current),
        realized_turnover=float(realized_turnover),
        total_cost=float(total_cost),
    )


def capacity_quantiles(
    capacity_by_rebalance_date: pd.Series,
) -> dict[str, object]:
    values = pd.to_numeric(capacity_by_rebalance_date, errors="raise").dropna()
    if values.empty:
        raise ValueError("capacity_observations_empty")
    annual = values.groupby(values.index.year).quantile(0.05).to_dict()
    return {
        "formal_p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "annual_p05": {str(year): float(value) for year, value in annual.items()},
    }
