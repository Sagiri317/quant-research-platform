"""Factor-form A-share active risk, date-aware costs, and SLSQP portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_string_series,
    strict_timestamp_series,
)


@dataclass(frozen=True)
class FeeSchedule:
    effective_date: date
    buy_commission_bps: float
    sell_commission_bps: float
    stamp_duty_bps: float
    transfer_fee_bps: float

    def __post_init__(self) -> None:
        if not isinstance(self.effective_date, date) or any(
            type(value) is not float or not np.isfinite(value) or value < 0.0
            for value in (
                self.buy_commission_bps,
                self.sell_commission_bps,
                self.stamp_duty_bps,
                self.transfer_fee_bps,
            )
        ):
            raise ValueError("a_share_fee_schedule_invalid")


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
    marginal_active_risk: pd.Series
    component_active_risk: pd.Series


@dataclass(frozen=True)
class AShareFactorRiskEstimate:
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    ewma_half_life: float
    diagonal_shrinkage: float
    annualization_factor: float


def estimate_a_share_factor_risk(
    history: pd.DataFrame,
    *,
    exposure_columns: tuple[str, ...],
    weight_column: str,
    ewma_half_life: float,
    diagonal_shrinkage: float,
    annualization_factor: float,
) -> AShareFactorRiskEstimate:
    """Estimate factor returns by WLS, then EWMA/shrunk factor and specific risk."""

    required = {
        "timestamp",
        "asset_id",
        "asset_return",
        weight_column,
        *exposure_columns,
    }
    if (
        not isinstance(history, pd.DataFrame)
        or not exposure_columns
        or not required.issubset(history.columns)
        or type(ewma_half_life) is not float
        or not np.isfinite(ewma_half_life)
        or ewma_half_life <= 0.0
        or type(diagonal_shrinkage) is not float
        or not 0.0 <= diagonal_shrinkage <= 1.0
        or type(annualization_factor) is not float
        or not np.isfinite(annualization_factor)
        or annualization_factor <= 0.0
    ):
        raise ValueError("a_share_risk_model_contract_invalid")
    frame = history.loc[:, list(required)].copy()
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="a_share_risk_model_timestamp_invalid"
    )
    frame["asset_id"] = strict_string_series(
        frame["asset_id"], error="a_share_risk_model_asset_invalid"
    )
    if bool(frame.duplicated(["timestamp", "asset_id"], keep=False).any()):
        raise ValueError("a_share_risk_model_primary_key_duplicate")
    for column in ("asset_return", weight_column, *exposure_columns):
        frame[column] = strict_finite_numeric_series(
            frame[column], error="a_share_risk_model_numeric_invalid"
        )
    if bool((frame[weight_column] <= 0.0).any()):
        raise ValueError("a_share_risk_model_weight_invalid")
    factor_rows = []
    residual_rows = []
    for timestamp, cross_section in frame.groupby("timestamp", sort=True):
        if len(cross_section) <= len(exposure_columns):
            raise ValueError("a_share_risk_model_cross_section_insufficient")
        exposures = cross_section.loc[:, exposure_columns].to_numpy(dtype="float64")
        returns = cross_section["asset_return"].to_numpy(dtype="float64")
        root_weights = np.sqrt(cross_section[weight_column].to_numpy(dtype="float64"))
        weighted_exposures = exposures * root_weights[:, None]
        if np.linalg.matrix_rank(weighted_exposures) != len(exposure_columns):
            raise ValueError("a_share_risk_model_exposure_rank_invalid")
        factor_return = np.linalg.lstsq(
            weighted_exposures,
            returns * root_weights,
            rcond=None,
        )[0]
        residuals = returns - exposures @ factor_return
        factor_rows.append(
            {
                "timestamp": timestamp,
                **dict(zip(exposure_columns, factor_return, strict=True)),
            }
        )
        residual_rows.extend(
            {
                "timestamp": timestamp,
                "asset_id": asset_id,
                "residual": residual,
            }
            for asset_id, residual in zip(
                cross_section["asset_id"], residuals, strict=True
            )
        )
    factor_returns = pd.DataFrame(factor_rows)
    if len(factor_returns) < 2:
        raise ValueError("a_share_risk_model_history_insufficient")
    factor_values = factor_returns.loc[:, exposure_columns].to_numpy(dtype="float64")
    ewma_weights = _ewma_weights(len(factor_values), ewma_half_life)
    centered = factor_values - np.average(factor_values, axis=0, weights=ewma_weights)
    covariance = (centered * ewma_weights[:, None]).T @ centered
    covariance = (1.0 - diagonal_shrinkage) * covariance + diagonal_shrinkage * np.diag(
        np.diag(covariance)
    )
    covariance *= annualization_factor
    factor_covariance = pd.DataFrame(
        covariance, index=exposure_columns, columns=exposure_columns
    )
    residual_frame = pd.DataFrame(residual_rows)
    specific_variance = residual_frame.groupby("asset_id", sort=True)["residual"].apply(
        lambda values: float(
            np.dot(_ewma_weights(len(values), ewma_half_life), values.to_numpy() ** 2)
            * annualization_factor
        )
    )
    specific_variance.name = "specific_variance"
    return AShareFactorRiskEstimate(
        factor_returns=factor_returns,
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
        ewma_half_life=ewma_half_life,
        diagonal_shrinkage=diagonal_shrinkage,
        annualization_factor=annualization_factor,
    )


def _ewma_weights(count: int, half_life: float) -> np.ndarray:
    decay = 0.5 ** (1.0 / half_life)
    weights = decay ** np.arange(count - 1, -1, -1, dtype="float64")
    return weights / weights.sum()


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
    active = weights - benchmark
    covariance_with_portfolio = exposures @ factor_cov @ (exposures.T @ active)
    covariance_with_portfolio += specific * active
    active_risk = float(np.sqrt(risk))
    marginal = (
        covariance_with_portfolio / active_risk
        if active_risk
        else np.zeros_like(active)
    )
    component = active * marginal
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
        marginal_active_risk=pd.Series(
            marginal, index=data["asset_id"], name="marginal_active_risk"
        ),
        component_active_risk=pd.Series(
            component, index=data["asset_id"], name="component_active_risk"
        ),
    )


def run_a_share_portfolio_backtest(
    panel: pd.DataFrame,
    *,
    benchmark_returns: pd.DataFrame,
    exposure_columns: tuple[str, ...],
    factor_covariance: pd.DataFrame,
    fee_schedules: tuple[FeeSchedule, ...],
    config: PortfolioOptimizationConfig,
) -> dict[str, object]:
    """Run one pre-registered rebalance schedule on a fixed daily panel."""

    required = {
        "timestamp",
        "asset_return",
        "current_weight",
        "benchmark_weight",
        "adv",
        "asset_id",
        "alpha",
        "specific_variance",
        "eligible",
        "can_enter",
        "can_exit",
        *exposure_columns,
    }
    if not isinstance(panel, pd.DataFrame) or not required.issubset(panel.columns):
        raise ValueError("a_share_backtest_columns_invalid")
    frame = panel.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if bool(frame.duplicated(["timestamp", "asset_id"], keep=False).any()):
        raise ValueError("a_share_backtest_primary_key_duplicate")
    frame.sort_values(["timestamp", "asset_id"], kind="mergesort", inplace=True)
    dates = pd.DatetimeIndex(frame["timestamp"].drop_duplicates())
    if not isinstance(benchmark_returns, pd.DataFrame) or not {
        "timestamp",
        "benchmark_return",
    }.issubset(benchmark_returns.columns):
        raise ValueError("a_share_benchmark_return_columns_invalid")
    benchmark_frame = benchmark_returns.loc[:, ("timestamp", "benchmark_return")].copy()
    benchmark_frame["timestamp"] = strict_timestamp_series(
        benchmark_frame["timestamp"], error="a_share_benchmark_return_timestamp_invalid"
    )
    benchmark_frame["benchmark_return"] = strict_finite_numeric_series(
        benchmark_frame["benchmark_return"],
        error="a_share_benchmark_return_value_invalid",
    )
    if bool(benchmark_frame["timestamp"].duplicated(keep=False).any()):
        raise ValueError("a_share_benchmark_return_key_duplicate")
    benchmark_by_date = benchmark_frame.set_index("timestamp")["benchmark_return"]
    if set(benchmark_by_date.index) != set(dates):
        raise ValueError("a_share_benchmark_return_alignment_invalid")
    rebalance_dates = _rebalance_dates(dates, config.rebalance_frequency)
    first = frame.loc[frame["timestamp"].eq(dates[0])]
    current = first.set_index("asset_id")["current_weight"].astype("float64")
    if not np.isclose(current.sum(), 1.0):
        raise ValueError("a_share_backtest_initial_weight_invalid")
    nav = benchmark_nav = 1.0
    cumulative_pnl = 0.0
    factor_variance = specific_variance = 0.0
    predicted_active_risk = 0.0
    marginal_active_risk: dict[str, float] = {}
    component_active_risk: dict[str, float] = {}
    path_rows: list[dict[str, object]] = []
    exposure_rows: list[dict[str, object]] = []
    capacity_values = {limit: [] for limit in (0.01, 0.02, 0.05, 0.10)}
    for timestamp, daily in frame.groupby("timestamp", sort=True):
        daily = daily.copy()
        daily.index = daily["asset_id"]
        missing_assets = current.index.difference(daily.index)
        if bool((current.reindex(missing_assets).abs() > 1e-12).any()):
            raise ValueError("a_share_backtest_held_asset_missing")
        current = current.drop(missing_assets)
        new_assets = daily.index.difference(current.index)
        if len(new_assets):
            current = pd.concat(
                (current, pd.Series(0.0, index=new_assets, dtype="float64"))
            )
        current = current.reindex(daily.index)
        daily["current_weight"] = current
        turnover = cost = 0.0
        if pd.Timestamp(timestamp) in rebalance_dates:
            result = optimize_a_share_portfolio(
                daily.reset_index(drop=True),
                exposure_columns=exposure_columns,
                factor_covariance=factor_covariance,
                fee_schedules=fee_schedules,
                trade_date=pd.Timestamp(timestamp).date(),
                config=config,
            )
            next_weights = result.weights.reindex(daily.index)
            changes = (next_weights - current.reindex(daily.index)).abs()
            turnover = result.turnover
            cost = result.estimated_cost
            factor_variance = result.factor_variance
            specific_variance = result.specific_variance
            predicted_active_risk = result.predicted_active_risk
            marginal_active_risk = result.marginal_active_risk.to_dict()
            component_active_risk = result.component_active_risk.to_dict()
            for limit in capacity_values:
                traded = changes.gt(1e-12)
                if bool(traded.any()):
                    capacity_values[limit].append(
                        float(
                            (daily.loc[traded, "adv"] * limit / changes[traded]).min()
                        )
                    )
            current = next_weights
        weights = current.reindex(daily.index).to_numpy(dtype="float64")
        benchmark = daily["benchmark_weight"].to_numpy(dtype="float64")
        asset_returns = daily["asset_return"].to_numpy(dtype="float64")
        gross_return = float(weights @ asset_returns)
        benchmark_return = float(benchmark_by_date.loc[pd.Timestamp(timestamp)])
        net_return = gross_return - cost
        active_return = net_return - benchmark_return
        cumulative_pnl += config.portfolio_notional * net_return
        nav *= 1.0 + net_return
        benchmark_nav *= 1.0 + benchmark_return
        path_rows.append(
            {
                "timestamp": pd.Timestamp(timestamp),
                "gross_return": gross_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "active_return": active_return,
                "constant_notional_additive_pnl": cumulative_pnl,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "active_nav": nav / benchmark_nav,
                "turnover": turnover,
                "cost": cost,
                "factor_variance": factor_variance,
                "specific_variance": specific_variance,
                "predicted_active_risk": predicted_active_risk,
            }
        )
        active_weights = weights - benchmark
        for column in exposure_columns:
            exposure_rows.append(
                {
                    "timestamp": pd.Timestamp(timestamp),
                    "exposure": column,
                    "active_value": float(
                        active_weights @ daily[column].to_numpy(dtype="float64")
                    ),
                }
            )
        ending_values = weights * (1.0 + asset_returns)
        ending_total = float(ending_values.sum())
        if ending_total <= 0.0:
            raise ValueError("a_share_backtest_non_positive_holdings")
        current = pd.Series(
            ending_values / ending_total,
            index=daily.index,
            dtype="float64",
        )
    path = pd.DataFrame(path_rows)
    path["drawdown"] = path["nav"] / path["nav"].cummax() - 1.0
    active = path["active_return"].to_numpy(dtype="float64")
    tracking_error = float(np.std(active, ddof=1) * np.sqrt(252.0))
    information_ratio = (
        None
        if tracking_error == 0.0
        else float(np.mean(active) * 252.0 / tracking_error)
    )
    return {
        "path": path,
        "active_exposures": pd.DataFrame(exposure_rows),
        "summary": {
            "rebalance_frequency": config.rebalance_frequency,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "total_turnover": float(path["turnover"].sum()),
            "total_cost": float(path["cost"].sum()),
            "maximum_drawdown": float(path["drawdown"].min()),
            "mean_factor_variance": float(path["factor_variance"].mean()),
            "mean_specific_variance": float(path["specific_variance"].mean()),
            "latest_predicted_active_risk": predicted_active_risk,
            "marginal_active_risk": marginal_active_risk,
            "component_active_risk": component_active_risk,
            "constant_notional_additive_pnl": cumulative_pnl,
            "compounded_nav": nav,
            "benchmark_nav": benchmark_nav,
            "capacity_notional": {
                f"{int(limit * 100)}pct_adv": (
                    None if not values else float(min(values))
                )
                for limit, values in capacity_values.items()
            },
        },
    }


def _rebalance_dates(
    dates: pd.DatetimeIndex, frequency: str
) -> frozenset[pd.Timestamp]:
    if frequency == "1d":
        return frozenset(dates)
    periods = dates.to_period("W" if frequency == "1w" else "M")
    return frozenset(dates[~periods.duplicated()])


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
