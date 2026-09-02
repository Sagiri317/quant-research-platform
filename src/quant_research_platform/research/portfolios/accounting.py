"""Continuous-weight portfolio construction support and economic ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_BLOCK_REASONS,
    ExecutionAssumptions,
    TradeCostComponents,
)
from quant_research_platform.research.portfolios.schedules import rebalance_dates

_LEDGER_COLUMNS = (
    "execution_date",
    "asset_id",
    "pre_trade_notional",
    "target_notional",
    "executed_delta_notional",
    "post_trade_notional",
    "reason",
    "commission_cost",
    "stamp_duty_cost",
    "transfer_fee_cost",
    "slippage_cost",
    "impact_cost",
)
_COST_COLUMNS = (
    "commission_cost",
    "stamp_duty_cost",
    "transfer_fee_cost",
    "slippage_cost",
    "impact_cost",
)


@dataclass(frozen=True, slots=True)
class ExecutionAccounting:
    """One transition value backed by one canonical per-asset notional ledger."""

    _execution_date: date
    _asset_ledger: pd.DataFrame
    _nav_before: float
    _pre_trade_cash: float
    _post_trade_cash: float
    cost_source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self._asset_ledger, pd.DataFrame):
            raise ValueError("execution_ledger_frame_invalid")
        ledger = self._asset_ledger.copy(deep=True)
        if type(self._execution_date) is not date:
            raise ValueError("execution_ledger_date_invalid")
        if tuple(ledger.columns) != _LEDGER_COLUMNS:
            raise ValueError("execution_ledger_columns_invalid")
        if ledger["asset_id"].duplicated().any():
            raise ValueError("execution_ledger_asset_keys_invalid")
        if not ledger["asset_id"].is_monotonic_increasing:
            raise ValueError("execution_ledger_order_invalid")
        if not bool(
            ledger["asset_id"]
            .map(lambda value: type(value) is str and bool(value))
            .all()
        ):
            raise ValueError("execution_ledger_asset_keys_invalid")
        if not bool(
            ledger["execution_date"].map(lambda value: type(value) is date).all()
        ) or not bool(ledger["execution_date"].eq(self._execution_date).all()):
            raise ValueError("execution_ledger_date_invalid")
        if set(ledger["reason"].dropna()) - PORTFOLIO_BLOCK_REASONS:
            raise ValueError("execution_ledger_reason_invalid")
        raw_numeric = ledger.loc[
            :,
            (
                "pre_trade_notional",
                "target_notional",
                "executed_delta_notional",
                "post_trade_notional",
                *_COST_COLUMNS,
            ),
        ]
        if (
            raw_numeric.map(lambda value: isinstance(value, (bool, np.bool_)))
            .any()
            .any()
        ):
            raise ValueError("execution_ledger_numeric_invalid")
        numeric = raw_numeric.apply(pd.to_numeric, errors="raise")
        for column in numeric.columns:
            ledger[column] = numeric[column].astype("float64")
        numeric_values = numeric.to_numpy(dtype="float64")
        if not np.isfinite(numeric_values).all():
            raise ValueError("execution_ledger_numeric_invalid")
        if (
            (
                numeric.loc[
                    :, ["pre_trade_notional", "target_notional", "post_trade_notional"]
                ]
                < 0.0
            )
            .any()
            .any()
        ):
            raise ValueError("execution_ledger_notional_invalid")
        if (numeric.loc[:, list(_COST_COLUMNS)] < 0.0).any().any():
            raise ValueError("execution_ledger_cost_invalid")
        nav_before = _finite_positive(self._nav_before, "execution_nav_invalid")
        tolerance = _scaled_tolerance(nav_before)
        desired = numeric["target_notional"] - numeric["pre_trade_notional"]
        executed = numeric["executed_delta_notional"]
        if (
            ((desired > 0.0) & (executed < 0.0))
            | ((desired < 0.0) & (executed > 0.0))
            | ((desired == 0.0) & (executed != 0.0))
        ).any() or (executed.abs() > desired.abs() + tolerance).any():
            raise ValueError("execution_ledger_direction_invalid")
        if (
            numeric.loc[:, list(_COST_COLUMNS)].gt(0.0).any(axis=1) & executed.eq(0.0)
        ).any():
            raise ValueError("execution_ledger_phantom_cost_invalid")
        if not np.allclose(
            numeric["pre_trade_notional"].to_numpy(dtype="float64")
            + numeric["executed_delta_notional"].to_numpy(dtype="float64"),
            numeric["post_trade_notional"].to_numpy(dtype="float64"),
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("execution_ledger_state_invalid")
        mismatch = (
            numeric["target_notional"] - numeric["post_trade_notional"]
        ).abs() > tolerance
        if not bool((ledger["reason"].notna() == mismatch).all()):
            raise ValueError("execution_ledger_reason_invalid")
        pre_cash = _finite_nonnegative(self._pre_trade_cash, "execution_cash_invalid")
        post_cash = _finite_nonnegative(self._post_trade_cash, "execution_cash_invalid")
        total_cost = float(numeric.loc[:, list(_COST_COLUMNS)].sum().sum())
        nav_after = float(numeric["post_trade_notional"].sum() + post_cash)
        if not np.isclose(
            numeric["pre_trade_notional"].sum() + pre_cash,
            nav_before,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("execution_pre_trade_state_invalid")
        if (
            numeric["target_notional"].sum() > nav_before + tolerance
            or numeric["post_trade_notional"].sum() > nav_after + tolerance
        ):
            raise ValueError("execution_ledger_notional_invalid")
        if not np.isclose(nav_after, nav_before - total_cost, rtol=0.0, atol=tolerance):
            raise ValueError("execution_self_financing_invalid")
        if (
            type(self.cost_source_identity) is not str
            or len(self.cost_source_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.cost_source_identity
            )
        ):
            raise ValueError("execution_fee_identity_invalid")
        object.__setattr__(self, "_asset_ledger", ledger)
        object.__setattr__(self, "_nav_before", nav_before)
        object.__setattr__(self, "_pre_trade_cash", pre_cash)
        object.__setattr__(self, "_post_trade_cash", post_cash)

    @property
    def asset_ledger(self) -> pd.DataFrame:
        return self._asset_ledger.copy(deep=True)

    @property
    def execution_date(self) -> date:
        return self._execution_date

    @property
    def nav_before(self) -> float:
        return float(self._nav_before)

    @property
    def nav_after(self) -> float:
        return float(
            self._asset_ledger["post_trade_notional"].sum() + self._post_trade_cash
        )

    @property
    def pre_trade_cash(self) -> float:
        return float(self._pre_trade_cash)

    @property
    def post_trade_cash(self) -> float:
        return float(self._post_trade_cash)

    @property
    def cash_weight(self) -> float:
        return float(self._post_trade_cash / self.nav_after)

    @property
    def target_weights(self) -> pd.Series:
        return (
            self._asset_ledger.set_index("asset_id")["target_notional"]
            .div(self.nav_before)
            .rename("target_weight")
            .copy(deep=True)
        )

    @property
    def pre_trade_weights(self) -> pd.Series:
        return (
            self._asset_ledger.set_index("asset_id")["pre_trade_notional"]
            .div(self.nav_before)
            .rename("pre_trade_weight")
            .copy(deep=True)
        )

    @property
    def realized_weights(self) -> pd.Series:
        return (
            self._asset_ledger.set_index("asset_id")["post_trade_notional"]
            .div(self.nav_after)
            .rename("realized_weight")
            .copy(deep=True)
        )

    @property
    def total_cost(self) -> float:
        return float(self._asset_ledger.loc[:, list(_COST_COLUMNS)].sum().sum())

    @property
    def target_turnover(self) -> float:
        target = self._asset_ledger["target_notional"]
        current = self._asset_ledger["pre_trade_notional"]
        target_cash = self.nav_before - float(target.sum())
        current_cash = self.nav_before - float(current.sum())
        return float(
            0.5
            * ((target - current).abs().sum() + abs(target_cash - current_cash))
            / self.nav_before
        )

    @property
    def realized_turnover(self) -> float:
        deltas = self._asset_ledger["executed_delta_notional"]
        return float(
            0.5 * (deltas.abs().sum() + abs(float(deltas.sum()))) / self.nav_before
        )

    @property
    def session_totals(self) -> dict[str, float]:
        return {
            "nav_before": self.nav_before,
            "nav_after": self.nav_after,
            "pre_trade_cash": self.pre_trade_cash,
            "post_trade_cash": self.post_trade_cash,
            "total_cost": self.total_cost,
            "target_turnover": self.target_turnover,
            "realized_turnover": self.realized_turnover,
        }


@dataclass(frozen=True)
class AlphaCalibration:
    slope: float
    intercept: float
    expected_alpha: pd.Series
    valid_rebalance_dates: int
    scheduled_rebalance_dates: int
    benchmark_only: bool


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


def execute_continuous_target(
    current_weights: pd.Series,
    target_weights: pd.Series,
    *,
    nav: float,
    average_daily_volume: pd.Series,
    can_enter: pd.Series,
    can_exit: pd.Series,
    execution_date: date,
    cost_for_trade: Callable[
        [date, str, str, float, float, ExecutionAssumptions],
        TradeCostComponents,
    ],
    assumptions: ExecutionAssumptions,
    cost_source_identity: str,
) -> ExecutionAccounting:
    """Execute sell-first, cash-constrained continuous-weight orders."""

    if type(execution_date) is not date:
        raise ValueError("execution_date_invalid")
    if not callable(cost_for_trade) or not isinstance(
        assumptions, ExecutionAssumptions
    ):
        raise ValueError("execution_inputs_invalid")
    nav_value = _finite_positive(nav, "execution_nav_invalid")
    assets = _canonical_assets(current_weights, target_weights, allow_empty=True)
    current = _weight_values(current_weights, assets, "current_weights_invalid")
    target = _weight_values(target_weights, assets, "target_weights_invalid")
    if current.lt(0.0).any() or target.lt(0.0).any():
        raise ValueError("execution_weights_invalid")
    tolerance = _scaled_tolerance(nav_value)
    if (
        current.sum() * nav_value > nav_value + tolerance
        or target.sum() * nav_value > nav_value + tolerance
    ):
        raise ValueError("execution_weights_sum_invalid")
    enter = _bool_values(can_enter, assets, "execution_entry_mask_invalid")
    exit_ = _bool_values(can_exit, assets, "execution_exit_mask_invalid")
    desired = (target - current) * nav_value
    trading_assets = {
        str(asset)
        for asset in assets
        if (desired[asset] < 0.0 and bool(exit_[asset]))
        or (desired[asset] > 0.0 and bool(enter[asset]))
    }
    adv = _asset_values(
        average_daily_volume,
        assets,
        "execution_adv_invalid",
        required_assets=trading_assets,
    )
    participation = assumptions.adv_participation
    capacity = adv * participation
    records = {
        str(asset): _empty_ledger_row(
            asset, execution_date, current[asset] * nav_value, target[asset] * nav_value
        )
        for asset in assets
    }
    sell_delta = pd.Series(0.0, index=assets, dtype="float64")
    sell_cost_total = 0.0
    for asset in assets:
        desired_delta = float(desired[asset])
        if desired_delta >= 0.0:
            continue
        record = records[str(asset)]
        if not bool(exit_[asset]):
            record["reason"] = "market_blocked"
            continue
        amount = min(-desired_delta, float(capacity[asset]))
        cost = None
        if amount > 0.0:
            cost = _validated_trade_cost(
                cost_for_trade,
                execution_date=execution_date,
                asset=str(asset),
                side="sell",
                notional=amount,
                adv20=float(adv[asset]),
                source_identity=cost_source_identity,
                assumptions=assumptions,
            )
            sell_cost_total += cost.total_cost
        sell_delta[asset] = -amount
        _update_ledger_row(
            record,
            executed_delta=-amount,
            reason="adv_capacity" if amount < -desired_delta - tolerance else None,
            cost=cost,
        )
    pre_trade_cash = nav_value - float((current * nav_value).sum())
    sell_notional = float(-sell_delta.sum())
    cash_after_sells = pre_trade_cash + sell_notional - sell_cost_total
    if cash_after_sells < -tolerance:
        raise ValueError("execution_cash_after_sells_invalid")

    buy_candidates: dict[str, float] = {}
    for asset in assets:
        desired_delta = float(desired[asset])
        if desired_delta <= 0.0:
            continue
        record = records[str(asset)]
        if not bool(enter[asset]):
            record["reason"] = "market_blocked"
            continue
        buy_candidates[str(asset)] = min(desired_delta, float(capacity[asset]))

    def buy_spend(
        scale: float,
    ) -> tuple[float, float, dict[str, TradeCostComponents]]:
        notional = 0.0
        costs: dict[str, TradeCostComponents] = {}
        for asset, candidate in buy_candidates.items():
            amount = candidate * scale
            if amount <= 0.0:
                continue
            cost = _validated_trade_cost(
                cost_for_trade,
                execution_date=execution_date,
                asset=asset,
                side="buy",
                notional=amount,
                adv20=float(adv[asset]),
                source_identity=cost_source_identity,
                assumptions=assumptions,
            )
            notional += amount
            costs[asset] = cost
        return notional, sum(cost.total_cost for cost in costs.values()), costs

    scale = 0.0
    if buy_candidates and cash_after_sells > 0.0:
        full_notional, full_cost, _ = buy_spend(1.0)
        if full_notional + full_cost <= cash_after_sells + tolerance:
            scale = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(80):
                middle = (low + high) / 2.0
                middle_notional, middle_cost, _ = buy_spend(middle)
                if middle_notional + middle_cost <= cash_after_sells:
                    low = middle
                else:
                    high = middle
            scale = low
    buy_notional, buy_cost_total, buy_costs = buy_spend(scale)
    buy_delta = pd.Series(0.0, index=assets, dtype="float64")
    for asset, candidate in buy_candidates.items():
        amount = candidate * scale
        buy_delta[asset] = amount
        target_delta = float(desired[asset])
        blocked = amount < target_delta - tolerance
        if candidate == 0.0 or (
            candidate < target_delta - tolerance and scale >= 1.0 - 1e-15
        ):
            reason = "adv_capacity"
        elif candidate < target_delta - tolerance and blocked:
            reason = "adv_capacity+cash_pro_rata"
        elif candidate < target_delta - tolerance:
            reason = "adv_capacity"
        elif blocked:
            reason = "cash_pro_rata"
        else:
            reason = None
        _update_ledger_row(
            records[asset],
            executed_delta=amount,
            reason=reason,
            cost=buy_costs.get(asset),
        )
    total_cost = float(sell_cost_total + buy_cost_total)
    post_trade_cash = cash_after_sells - buy_notional - buy_cost_total
    if -tolerance <= post_trade_cash < 0.0:
        post_trade_cash = 0.0
    post_trade = current * nav_value + sell_delta + buy_delta
    nav_after = float(post_trade.sum() + post_trade_cash)
    expected_nav_after = nav_value - total_cost
    if post_trade_cash < -tolerance or nav_after <= tolerance:
        raise ValueError("execution_cash_invalid")
    if not np.isclose(nav_after, expected_nav_after, rtol=0.0, atol=tolerance):
        raise ValueError("execution_self_financing_invalid")
    for asset in assets:
        records[str(asset)]["post_trade_notional"] = float(post_trade[asset])
    ledger = pd.DataFrame.from_records(
        [records[str(asset)] for asset in assets],
        columns=_LEDGER_COLUMNS,
    )
    return ExecutionAccounting(
        _execution_date=execution_date,
        _asset_ledger=ledger,
        _nav_before=nav_value,
        _pre_trade_cash=pre_trade_cash,
        _post_trade_cash=post_trade_cash,
        cost_source_identity=cost_source_identity,
    )


def drift_continuous_holdings(
    holdings_notional: pd.Series,
    cash_notional: float,
    asset_returns: pd.Series,
) -> tuple[pd.Series, float, float]:
    """Apply open-to-open returns while preserving cash and producing no trade."""

    if set(holdings_notional.index) != set(asset_returns.index):
        raise ValueError("drift_inputs_invalid")
    assets = _canonical_assets(holdings_notional, asset_returns, allow_empty=True)
    holdings = _weight_values(holdings_notional, assets, "drift_holdings_invalid")
    returns = _numeric_series(asset_returns, assets, "drift_returns_invalid")
    cash = _finite_nonnegative(cash_notional, "drift_cash_invalid")
    if (returns < -1.0).any():
        raise ValueError("drift_returns_invalid")
    drifted = (holdings * (1.0 + returns)).sort_index()
    nav = float(drifted.sum() + cash)
    if not np.isfinite(nav) or nav <= 0.0:
        raise ValueError("drift_nav_invalid")
    return drifted.copy(deep=True), cash, nav


def _empty_ledger_row(
    asset: str,
    execution_date: date,
    pre_trade_notional: float,
    target_notional: float,
) -> dict[str, object]:
    return {
        "execution_date": execution_date,
        "asset_id": asset,
        "pre_trade_notional": float(pre_trade_notional),
        "target_notional": float(target_notional),
        "executed_delta_notional": 0.0,
        "post_trade_notional": float(pre_trade_notional),
        "reason": None,
        **{column: 0.0 for column in _COST_COLUMNS},
    }


def _update_ledger_row(
    record: dict[str, object],
    *,
    executed_delta: float,
    reason: str | None,
    cost: TradeCostComponents | None,
) -> None:
    record["executed_delta_notional"] = float(executed_delta)
    record["reason"] = reason
    if cost is not None:
        for column in _COST_COLUMNS:
            record[column] = float(getattr(cost, column))


def _canonical_assets(
    *values: pd.Series,
    allow_empty: bool = False,
) -> pd.Index:
    if not values or any(
        not isinstance(value, pd.Series)
        or not value.index.is_unique
        or any(type(asset) is not str or not asset for asset in value.index)
        for value in values
    ):
        raise ValueError("asset_index_invalid")
    labels = {asset for value in values for asset in value.index}
    if not labels and not allow_empty:
        raise ValueError("asset_index_invalid")
    return pd.Index(sorted(labels), dtype="string")


def _weight_values(values: pd.Series, assets: pd.Index, error: str) -> pd.Series:
    result = values.reindex(assets, fill_value=0.0)
    return _numeric_series(result, assets, error)


def _numeric_series(values: pd.Series, assets: pd.Index, error: str) -> pd.Series:
    raw = values.reindex(assets)
    if raw.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(error)
    try:
        result = pd.to_numeric(raw, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if result.isna().any() or not np.isfinite(result.to_numpy()).all():
        raise ValueError(error)
    return result


def _asset_values(
    values: pd.Series,
    assets: pd.Index,
    error: str,
    *,
    required_assets: set[str],
) -> pd.Series:
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError(error)
    raw = values.reindex(assets)
    if raw.dropna().map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(error)
    try:
        result = pd.to_numeric(raw, errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    present = pd.Series(assets.isin(values.index), index=assets)
    required_missing = (~present) & present.index.to_series().isin(required_assets)
    if required_missing.any():
        raise ValueError(error)
    present_values = result.loc[present]
    if (
        present_values.isna().any()
        or not np.isfinite(present_values.to_numpy()).all()
        or present_values.lt(0.0).any()
    ):
        raise ValueError(error)
    return result


def _bool_values(values: pd.Series, assets: pd.Index, error: str) -> pd.Series:
    if (
        not isinstance(values, pd.Series)
        or not values.index.is_unique
        or not pd.api.types.is_bool_dtype(values)
        or any(type(asset) is not str or not asset for asset in values.index)
    ):
        raise ValueError(error)
    result = values.reindex(assets)
    if result.isna().any():
        raise ValueError(error)
    return result.astype(bool)


def _validated_trade_cost(
    cost_for_trade: Callable[
        [date, str, str, float, float, ExecutionAssumptions],
        TradeCostComponents,
    ],
    *,
    execution_date: date,
    asset: str,
    side: str,
    notional: float,
    adv20: float,
    source_identity: str,
    assumptions: ExecutionAssumptions,
) -> TradeCostComponents:
    value = cost_for_trade(
        execution_date,
        asset,
        side,
        notional,
        adv20,
        assumptions,
    )
    if not isinstance(value, TradeCostComponents):
        raise ValueError("execution_cost_provider_invalid")
    if value.source_identity != source_identity or value.notional != notional:
        raise ValueError("execution_cost_provider_identity_mismatch")
    return value


def _finite_positive(value: float, error: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if isinstance(value, bool) or not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(error)
    return numeric


def _finite_nonnegative(value: float, error: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if isinstance(value, bool) or not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(error)
    return numeric


def _scaled_tolerance(nav: float) -> float:
    return max(np.finfo("float64").eps * max(nav, 1.0) * 256.0, 1e-12)


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
