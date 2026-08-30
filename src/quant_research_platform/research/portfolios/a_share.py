"""Factor-form A-share active risk, date-aware costs, and SLSQP portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_research_platform.contracts.validation import strict_finite_numeric_series


@dataclass(frozen=True)
class FeeSchedule:
    effective_date: date
    buy_commission_bps: float
    sell_commission_bps: float
    stamp_duty_bps: float
    transfer_fee_bps: float


@dataclass(frozen=True)
class PortfolioOptimizationConfig:
    rebalance_frequency: str
    portfolio_notional: float
    max_weight: float
    tracking_error_limit: float
    turnover_limit: float
    adv_participation_limit: float
    active_exposure_limit: float
    risk_penalty: float
    transaction_cost_penalty: float
    spread_slippage_bps: float
    liquidity_impact_bps: float


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    weights: pd.Series
    predicted_active_risk: float
    factor_variance: float
    specific_variance: float
    turnover: float
    estimated_cost: float
    maximum_adv_participation: float
    rebalance_frequency: str


def optimize_a_share_portfolio(
    assets: pd.DataFrame,
    *,
    exposure_columns: tuple[str, ...],
    factor_covariance: pd.DataFrame,
    fee_schedules: tuple[FeeSchedule, ...],
    trade_date: date,
    config: PortfolioOptimizationConfig,
) -> PortfolioOptimizationResult:
    """Optimize alpha minus factor risk and date-aware transaction costs."""

    required = {
        "asset_id",
        "alpha",
        "current_weight",
        "benchmark_weight",
        "specific_variance",
        "adv",
        "eligible",
        "can_enter",
        "can_exit",
        *exposure_columns,
    }
    if not isinstance(assets, pd.DataFrame) or not required.issubset(assets.columns):
        raise ValueError("a_share_optimizer_columns_invalid")
    if (
        not exposure_columns
        or factor_covariance.index.tolist() != list(exposure_columns)
        or factor_covariance.columns.tolist() != list(exposure_columns)
    ):
        raise ValueError("a_share_factor_covariance_invalid")
    data = assets.copy().reset_index(drop=True)
    if bool(data["asset_id"].duplicated(keep=False).any()):
        raise ValueError("a_share_optimizer_asset_duplicate")
    numeric_columns = (
        "alpha",
        "current_weight",
        "benchmark_weight",
        "specific_variance",
        "adv",
        *exposure_columns,
    )
    for column in numeric_columns:
        data[column] = strict_finite_numeric_series(
            data[column], error="a_share_optimizer_numeric_invalid"
        )
    for column in ("eligible", "can_enter", "can_exit"):
        if data[column].dtype != bool:
            raise ValueError("a_share_optimizer_boolean_invalid")
    _validate_config(config)
    if bool((data["specific_variance"] < 0.0).any()) or bool(
        (data["adv"] <= 0.0).any()
    ):
        raise ValueError("a_share_optimizer_risk_liquidity_invalid")
    current = data["current_weight"].to_numpy(dtype="float64")
    benchmark = data["benchmark_weight"].to_numpy(dtype="float64")
    if not np.isclose(current.sum(), 1.0) or not np.isclose(benchmark.sum(), 1.0):
        raise ValueError("a_share_optimizer_weight_sum_invalid")
    if bool((current < 0.0).any()) or bool((benchmark < 0.0).any()):
        raise ValueError("a_share_optimizer_long_only_input_invalid")
    alpha = data["alpha"].to_numpy(dtype="float64")
    specific = data["specific_variance"].to_numpy(dtype="float64")
    exposures = data.loc[:, exposure_columns].to_numpy(dtype="float64")
    factor_cov = factor_covariance.to_numpy(dtype="float64")
    if not bool(np.isfinite(factor_cov).all()) or not np.allclose(
        factor_cov, factor_cov.T
    ):
        raise ValueError("a_share_factor_covariance_invalid")
    if float(np.linalg.eigvalsh(factor_cov).min(initial=0.0)) < -1e-12:
        raise ValueError("a_share_factor_covariance_not_psd")
    buy_rate, sell_rate = _cost_rates(fee_schedules, trade_date, config)
    bounds = _weight_bounds(data, current, config.max_weight)

    def components(weights: np.ndarray) -> tuple[float, float, float]:
        active = weights - benchmark
        factor_active = exposures.T @ active
        factor_risk = float(factor_active @ factor_cov @ factor_active)
        specific_risk = float(np.sum(specific * active * active))
        return factor_risk + specific_risk, factor_risk, specific_risk

    def transaction_cost(weights: np.ndarray) -> float:
        change = weights - current
        rates = np.where(change >= 0.0, buy_rate, sell_rate)
        return float(np.sum(np.abs(change) * rates))

    def objective(weights: np.ndarray) -> float:
        risk, _, _ = components(weights)
        return float(
            -(alpha @ weights)
            + config.risk_penalty * risk
            + config.transaction_cost_penalty * transaction_cost(weights)
        )

    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
        {
            "type": "ineq",
            "fun": lambda weights: float(
                config.tracking_error_limit**2 - components(weights)[0]
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                config.turnover_limit - np.abs(weights - current).sum()
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.active_exposure_limit
                - np.abs(exposures.T @ (weights - benchmark))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.adv_participation_limit
                - np.abs(weights - current)
                * config.portfolio_notional
                / data["adv"].to_numpy(dtype="float64")
            ),
        },
    ]
    result = minimize(
        objective,
        current,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise ValueError("a_share_optimizer_failed")
    weights = np.asarray(result.x, dtype="float64")
    risk, factor_risk, specific_risk = components(weights)
    participation = (
        np.abs(weights - current)
        * config.portfolio_notional
        / data["adv"].to_numpy(dtype="float64")
    )
    _verify_solution(
        weights,
        bounds=bounds,
        current=current,
        benchmark=benchmark,
        exposures=exposures,
        adv=data["adv"].to_numpy(dtype="float64"),
        risk=risk,
        config=config,
    )
    return PortfolioOptimizationResult(
        weights=pd.Series(weights, index=data["asset_id"], name="weight"),
        predicted_active_risk=float(np.sqrt(risk)),
        factor_variance=factor_risk,
        specific_variance=specific_risk,
        turnover=float(np.abs(weights - current).sum()),
        estimated_cost=transaction_cost(weights),
        maximum_adv_participation=float(participation.max(initial=0.0)),
        rebalance_frequency=config.rebalance_frequency,
    )


def _weight_bounds(
    data: pd.DataFrame, current: np.ndarray, max_weight: float
) -> tuple[tuple[float, float], ...]:
    bounds = []
    for position, row in data.iterrows():
        lower = 0.0
        upper = max_weight
        if current[position] > max_weight and not bool(row["can_exit"]):
            raise ValueError("a_share_optimizer_overweight_cannot_exit")
        if not bool(row["eligible"]):
            upper = min(upper, current[position])
        if not bool(row["can_enter"]):
            upper = min(upper, current[position])
        if not bool(row["can_exit"]):
            lower = current[position]
        if lower > upper:
            raise ValueError("a_share_optimizer_tradability_infeasible")
        bounds.append((lower, upper))
    return tuple(bounds)


def _cost_rates(
    schedules: tuple[FeeSchedule, ...],
    trade_date: date,
    config: PortfolioOptimizationConfig,
) -> tuple[float, float]:
    eligible = sorted(
        (schedule for schedule in schedules if schedule.effective_date <= trade_date),
        key=lambda schedule: schedule.effective_date,
    )
    if not eligible:
        raise ValueError("a_share_fee_schedule_missing")
    if len({schedule.effective_date for schedule in schedules}) != len(schedules):
        raise ValueError("a_share_fee_schedule_date_duplicate")
    schedule = eligible[-1]
    buy_bps = (
        schedule.buy_commission_bps
        + schedule.transfer_fee_bps
        + config.spread_slippage_bps
        + config.liquidity_impact_bps
    )
    sell_bps = (
        schedule.sell_commission_bps
        + schedule.stamp_duty_bps
        + schedule.transfer_fee_bps
        + config.spread_slippage_bps
        + config.liquidity_impact_bps
    )
    if buy_bps <= 0.0 or sell_bps <= 0.0:
        raise ValueError("a_share_zero_cost_forbidden")
    return buy_bps / 10_000.0, sell_bps / 10_000.0


def _validate_config(config: PortfolioOptimizationConfig) -> None:
    if config.rebalance_frequency not in {"1d", "1w", "1m"}:
        raise ValueError("a_share_rebalance_frequency_invalid")
    for value in (
        config.portfolio_notional,
        config.max_weight,
        config.tracking_error_limit,
        config.turnover_limit,
        config.adv_participation_limit,
        config.active_exposure_limit,
        config.risk_penalty,
        config.transaction_cost_penalty,
        config.spread_slippage_bps,
        config.liquidity_impact_bps,
    ):
        if type(value) is not float or not np.isfinite(value) or value <= 0.0:
            raise ValueError("a_share_optimizer_config_invalid")


def _verify_solution(
    weights: np.ndarray,
    *,
    bounds: tuple[tuple[float, float], ...],
    current: np.ndarray,
    benchmark: np.ndarray,
    exposures: np.ndarray,
    adv: np.ndarray,
    risk: float,
    config: PortfolioOptimizationConfig,
) -> None:
    tolerance = 1e-7
    if not np.isclose(weights.sum(), 1.0, atol=tolerance):
        raise ValueError("a_share_optimizer_solution_invalid")
    if any(
        weight < lower - tolerance or weight > upper + tolerance
        for weight, (lower, upper) in zip(weights, bounds, strict=True)
    ):
        raise ValueError("a_share_optimizer_solution_invalid")
    if risk < -tolerance or risk > config.tracking_error_limit**2 + tolerance:
        raise ValueError("a_share_optimizer_solution_invalid")
    if np.abs(weights - current).sum() > config.turnover_limit + tolerance:
        raise ValueError("a_share_optimizer_solution_invalid")
    if bool(
        (
            np.abs(exposures.T @ (weights - benchmark))
            > config.active_exposure_limit + tolerance
        ).any()
    ):
        raise ValueError("a_share_optimizer_solution_invalid")
    participation = np.abs(weights - current) * config.portfolio_notional / adv
    if bool((participation > config.adv_participation_limit + tolerance).any()):
        raise ValueError("a_share_optimizer_solution_invalid")
