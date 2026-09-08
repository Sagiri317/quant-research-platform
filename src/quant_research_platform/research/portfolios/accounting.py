"""Continuous-weight portfolio construction support and economic ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from quant_research_platform.contracts.portfolios import (
    CONTINUOUS_EXECUTION_COLUMNS,
    PORTFOLIO_BLOCK_REASONS,
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    ExecutedTrade,
    ExecutionAssumptions,
    FundingEvent,
    PortfolioCapital,
    PortfolioCashFlow,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioTargetBlocked,
    PortfolioValuation,
    TradeCostComponents,
)
from quant_research_platform.research.portfolios.schedules import is_rebalance_session

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


def continuous_ledger_batch(
    frame: pd.DataFrame, sequence: int, *, end_timestamp: pd.Timestamp
) -> PortfolioLedgerBatch:
    """Project the continuous execution transition into canonical economic facts."""
    if tuple(frame.columns) != CONTINUOUS_EXECUTION_COLUMNS:
        raise ValueError("portfolio_ledger_schema_invalid")
    if frame.map(lambda value: isinstance(value, (bool, np.bool_))).any().any():
        raise ValueError("portfolio_ledger_numeric_invalid")
    trades: list[ExecutedTrade] = []
    cashflows: list[PortfolioCashFlow] = []
    valuations: list[PortfolioValuation] = []
    diagnostics: list[PortfolioDiagnostic] = []
    economic_columns = {
        "decision_timestamp",
        "execution_date",
        "asset_id",
        "pre_trade_notional",
        "executed_delta_notional",
        "post_trade_notional",
        "end_notional",
        "holding_return",
        *PORTFOLIO_LEDGER_COST_COLUMNS,
    }
    for row in frame.to_dict("records"):
        asset = str(row["asset_id"])
        timestamp = int(pd.Timestamp(row["execution_date"]).value)
        fill_id = None
        delta = float(row["executed_delta_notional"])
        if asset != PORTFOLIO_CASH_ASSET_ID and delta:
            fill_id = f"{sequence}:{asset}"
            trades.append(
                ExecutedTrade(fill_id, fill_id, timestamp, asset, delta, delta)
            )
            cashflows.append(
                PortfolioCashFlow(
                    f"{fill_id}:principal",
                    timestamp,
                    asset,
                    fill_id,
                    "principal",
                    -delta,
                )
            )
        for cost in PORTFOLIO_LEDGER_COST_COLUMNS:
            amount = float(row[cost])
            if amount:
                if fill_id is None:
                    raise ValueError("portfolio_ledger_phantom_cost_invalid")
                cashflows.append(
                    PortfolioCashFlow(
                        f"{fill_id}:{cost}",
                        timestamp,
                        asset,
                        fill_id,
                        cost,
                        -amount,
                    )
                )
        for phase, column in (
            ("before", "pre_trade_notional"),
            ("after", "post_trade_notional"),
            ("end", "end_notional"),
        ):
            value = float(row[column])
            quantity = float(row["post_trade_notional"]) if phase == "end" else value
            mark = 1.0 + float(row["holding_return"]) if phase == "end" else 1.0
            valuations.append(
                PortfolioValuation(
                    int(end_timestamp.value) if phase == "end" else timestamp,
                    asset,
                    phase,
                    quantity,
                    value,
                    mark,
                )
            )
        diagnostics.append(
            PortfolioDiagnostic(
                asset,
                fill_id,
                "decision_timestamp",
                text=pd.Timestamp(row["decision_timestamp"]).isoformat(),
            )
        )
        for name, value in row.items():
            if name in economic_columns or pd.isna(value):
                continue
            diagnostics.append(
                PortfolioDiagnostic(
                    asset,
                    fill_id,
                    name,
                    number=float(value) if not isinstance(value, str) else None,
                    text=value if isinstance(value, str) else None,
                )
            )
    return PortfolioLedgerBatch(
        sequence, tuple(trades), tuple(cashflows), tuple(valuations), tuple(diagnostics)
    )


class PortfolioEconomics:
    """Bounded linear-perpetual state shared by maker and taker execution."""

    def __init__(
        self,
        capital: PortfolioCapital,
        asset_id: str,
        emit: Callable[[PortfolioLedgerBatch], None] | None,
    ) -> None:
        if capital.valuation_method != "linear_perpetual" or not asset_id:
            raise ValueError("portfolio_valuation_method_invalid")
        self.capital = capital
        self.asset_id = asset_id
        self.emit = emit
        self.quantity = 0.0
        self.cost_basis = 0.0
        self.cash = capital.initial_capital
        self.cost_cashflows = 0.0
        self.funding_cashflows = 0.0
        self.sequence = 0
        self.fill_count = 0
        self.last_timestamp = 0
        self.last_mark: float | None = None

    def _values(
        self, timestamp: int, phase: str, mark: float | None
    ) -> tuple[PortfolioValuation, ...]:
        cash = PortfolioValuation(
            timestamp, PORTFOLIO_CASH_ASSET_ID, phase, self.cash, self.cash
        )
        if mark is None:
            return (cash,)
        return (
            PortfolioValuation(
                timestamp,
                self.asset_id,
                phase,
                self.quantity,
                self.quantity * mark - self.cost_basis,
                mark,
                self.cost_basis,
            ),
            cash,
        )

    def fill(
        self,
        *,
        timestamp_ns: int,
        order_id: str,
        quantity: float,
        price: float,
        mark_price: float,
        fee: float,
    ) -> str:
        if (
            not np.isfinite([quantity, price, mark_price, fee]).all()
            or not quantity
            or min(price, mark_price) <= 0
        ):
            raise ValueError("portfolio_execution_fill_invalid")
        before = self._values(timestamp_ns, "before", mark_price)
        self.fill_count += 1
        fill_id = f"fill:{self.fill_count}"
        trade = ExecutedTrade(
            fill_id, order_id, timestamp_ns, self.asset_id, quantity, quantity * price
        )
        realized = 0.0
        if self.quantity and self.quantity * quantity < 0:
            closed = min(abs(self.quantity), abs(quantity))
            realized = (
                closed
                * np.sign(self.quantity)
                * (price - self.cost_basis / self.quantity)
            )
            remaining = self.quantity + quantity
            self.cost_basis = remaining * (
                self.cost_basis / self.quantity
                if remaining * self.quantity > 0
                else price
            )
        else:
            self.cost_basis += quantity * price
        self.quantity += quantity
        self.cash += realized - fee
        self.cost_cashflows -= fee
        cashflows = (
            PortfolioCashFlow(
                f"{fill_id}:settlement",
                timestamp_ns,
                self.asset_id,
                fill_id,
                "realized_pnl",
                float(realized),
            ),
            PortfolioCashFlow(
                f"{fill_id}:fee",
                timestamp_ns,
                self.asset_id,
                fill_id,
                "commission_cost",
                -fee,
            ),
        )
        self._emit((trade,), cashflows, before, timestamp_ns, mark_price)
        return fill_id

    def funding(self, event: FundingEvent) -> None:
        if self.quantity == 0:
            return
        if event.rate is None or event.mark_price is None:
            raise PortfolioTargetBlocked("portfolio_required_funding_missing")
        before = self._values(event.timestamp_ns, "before", event.mark_price)
        amount = -self.quantity * event.mark_price * event.rate
        self.cash += amount
        self.funding_cashflows += amount
        self._emit(
            (),
            (
                PortfolioCashFlow(
                    f"funding:{event.timestamp_ns}",
                    event.timestamp_ns,
                    self.asset_id,
                    None,
                    "funding",
                    amount,
                ),
            ),
            before,
            event.timestamp_ns,
            event.mark_price,
        )

    def mark(
        self,
        timestamp_ns: int,
        mark_price: float | None,
        *,
        diagnostics: tuple[PortfolioDiagnostic, ...] = (),
    ) -> dict[str, float]:
        if (mark_price is None and self.quantity != 0.0) or (
            mark_price is not None and (not np.isfinite(mark_price) or mark_price <= 0)
        ):
            raise ValueError("portfolio_valuation_mark_invalid")
        before = self._values(timestamp_ns, "before", mark_price)
        self._emit((), (), before, timestamp_ns, mark_price, diagnostics)
        nav = sum(row.value for row in before)
        net = nav - self.capital.initial_capital
        return {
            "gross_pnl": net - self.cost_cashflows - self.funding_cashflows,
            "net_pnl": net,
            "nav": nav,
            "capital_return": net / self.capital.initial_capital,
            "fees": -self.cost_cashflows,
            "funding": self.funding_cashflows,
        }

    def diagnostic(self, fill: dict[str, object]) -> None:
        fill_id = str(fill["fill_id"])
        diagnostics = tuple(
            PortfolioDiagnostic(
                self.asset_id,
                fill_id,
                name,
                number=float(value) if type(value) in {int, float} else None,
                text=value if isinstance(value, str) else None,
            )
            for name, value in fill.items()
            if name not in {"fill_id", "fill_price", "fill_qty", "notional", "fee"}
            and (value is None or type(value) in {int, float, str})
        )
        before = self._values(self.last_timestamp, "before", self.last_mark)
        self._emit((), (), before, self.last_timestamp, self.last_mark, diagnostics)

    def _emit(self, trades, cashflows, before, timestamp, mark, diagnostics=()) -> None:
        batch = PortfolioLedgerBatch(
            self.sequence,
            trades,
            cashflows,
            (
                *before,
                *self._values(timestamp, "after", mark),
                *self._values(timestamp, "end", mark),
            ),
            diagnostics,
        )
        self.sequence += 1
        self.last_timestamp = timestamp
        self.last_mark = mark
        if self.emit is not None:
            self.emit(batch)


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
                *PORTFOLIO_LEDGER_COST_COLUMNS,
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
        if (numeric.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)] < 0.0).any().any():
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
            numeric.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].gt(0.0).any(axis=1)
            & executed.eq(0.0)
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
        total_cost = float(
            numeric.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum()
        )
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
        return float(
            self._asset_ledger.loc[:, list(PORTFOLIO_LEDGER_COST_COLUMNS)].sum().sum()
        )

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
    slope_standard_error: float
    slope_t_statistic: float
    slope_pvalue: float
    slope_ci_low: float
    slope_ci_high: float
    weighted_r2: float
    coverage: float
    valid_rebalance_dates: int
    scheduled_rebalance_dates: int
    benchmark_only: bool


def alpha_calibration_intervals(
    sessions: pd.DatetimeIndex, schedule: str
) -> pd.DataFrame:
    """Pair adjacent planned execution opens using the actual trading calendar."""
    positions = [
        index
        for index in range(len(sessions) - 1)
        if is_rebalance_session(sessions[index], sessions[index + 1], schedule)
    ]
    return pd.DataFrame(
        {
            "timestamp": sessions[positions[:-1]],
            "entry_at": sessions[np.asarray(positions[:-1], dtype=int) + 1],
            "available_at": sessions[np.asarray(positions[1:], dtype=int) + 1],
        }
    )


def calibrate_expected_alpha(
    frame: pd.DataFrame,
    *,
    alpha_column: str = "alpha_score",
    return_column: str = "holding_return",
    as_of: pd.Timestamp,
    scheduled_intervals: pd.DataFrame,
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
    cutoff = pd.Timestamp(as_of).normalize()
    if pd.isna(cutoff) or set(scheduled_intervals) != {
        "timestamp",
        "entry_at",
        "available_at",
    }:
        raise ValueError("alpha_calibration_schedule_invalid")
    intervals = scheduled_intervals.apply(pd.to_datetime, errors="raise")
    if (
        intervals.isna().any().any()
        or intervals.timestamp.duplicated().any()
        or not (
            intervals.timestamp.lt(intervals.entry_at)
            & intervals.entry_at.lt(intervals.available_at)
        ).all()
    ):
        raise ValueError("alpha_calibration_schedule_invalid")
    scheduled = pd.DatetimeIndex(
        intervals.loc[
            intervals.timestamp.ge(cutoff - pd.DateOffset(months=12))
            & intervals.available_at.le(cutoff),
            "timestamp",
        ]
    )
    alpha = pd.to_numeric(frame[alpha_column], errors="raise").astype("float64")
    returns = pd.to_numeric(frame[return_column], errors="raise").astype("float64")
    valid = alpha.notna() & returns.notna() & np.isfinite(alpha) & np.isfinite(returns)
    valid_dates = dates.loc[valid].isin(scheduled)
    date_count = 0 if scheduled.empty else int(dates.loc[valid & valid_dates].nunique())
    coverage = date_count / max(1, len(scheduled))
    enough_coverage = date_count >= minimum_valid_dates and coverage >= minimum_coverage
    if not enough_coverage:
        return AlphaCalibration(
            slope=0.0,
            intercept=0.0,
            slope_standard_error=0.0,
            slope_t_statistic=0.0,
            slope_pvalue=1.0,
            slope_ci_low=0.0,
            slope_ci_high=0.0,
            weighted_r2=0.0,
            coverage=coverage,
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
    bread = np.zeros((2, 2), dtype="float64")
    cluster_scores = []
    actual = returns.loc[selected].to_numpy(dtype="float64")
    residual = actual - design @ coefficients
    selected_dates = dates.loc[selected].to_numpy()
    _, cluster_codes = np.unique(selected_dates, return_inverse=True)
    for cluster in np.unique(cluster_codes):
        rows = np.flatnonzero(cluster_codes == cluster)
        score = design[rows].T @ (weights[rows] * residual[rows])
        cluster_scores.append(score)
        bread += design[rows].T @ (weights[rows, None] * design[rows])
    meat = sum(score[:, None] @ score[None, :] for score in cluster_scores)
    covariance = np.linalg.pinv(bread) @ meat @ np.linalg.pinv(bread)
    standard_error = float(np.sqrt(max(0.0, covariance[1, 1])))
    degrees_freedom = max(1, len(cluster_scores) - 2)
    if standard_error > 0.0:
        statistic = slope / standard_error
        pvalue = float(2.0 * student_t.sf(abs(statistic), df=degrees_freedom))
        critical = float(student_t.ppf(0.975, df=degrees_freedom))
        ci_low = slope - critical * standard_error
        ci_high = slope + critical * standard_error
        t_statistic = statistic
    else:
        statistic = 0.0 if slope == 0.0 else np.inf
        pvalue = 1.0 if slope == 0.0 else 0.0
        ci_low = ci_high = slope
        t_statistic = statistic
    centered = actual - np.average(actual, weights=weights)
    denominator = float(np.dot(weights, centered * centered))
    weighted_r2 = (
        1.0 - float(np.dot(weights, residual * residual)) / denominator
        if denominator > 0.0
        else 0.0
    )
    return AlphaCalibration(
        slope=slope,
        intercept=intercept,
        slope_standard_error=standard_error,
        slope_t_statistic=float(t_statistic),
        slope_pvalue=pvalue,
        slope_ci_low=float(ci_low),
        slope_ci_high=float(ci_high),
        weighted_r2=float(weighted_r2),
        coverage=coverage,
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
    if -tolerance <= pre_trade_cash < 0.0:
        pre_trade_cash = 0.0
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
                if middle == low or middle == high:
                    break
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
    """Mark continuous holdings by adjusted-price returns without a trade."""

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
        **{column: 0.0 for column in PORTFOLIO_LEDGER_COST_COLUMNS},
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
        for column in PORTFOLIO_LEDGER_COST_COLUMNS:
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
    required_missing = result.isna() & present.index.to_series().isin(required_assets)
    if required_missing.any():
        raise ValueError(f"{error}:missing_required_assets")
    present_values = result.loc[present & result.notna()]
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


def portfolio_interval_economics(batch: PortfolioLedgerBatch) -> dict[str, object]:
    """One actual holding interval; costs remain at its entry execution open."""
    before = {
        row.asset_id: row.value for row in batch.valuations if row.phase == "before"
    }
    after = {
        row.asset_id: row.value for row in batch.valuations if row.phase == "after"
    }
    end = {row.asset_id: row.value for row in batch.valuations if row.phase == "end"}
    starts = {row.timestamp_ns for row in batch.valuations if row.phase == "after"}
    ends = {row.timestamp_ns for row in batch.valuations if row.phase == "end"}
    if (
        len(starts) != 1
        or len(ends) != 1
        or set(before) != set(after)
        or set(after) != set(end)
    ):
        raise ValueError("portfolio_interval_valuation_invalid")
    entry, endpoint = pd.Timestamp(next(iter(starts))), pd.Timestamp(next(iter(ends)))
    b, a, e = (
        float(sum(before.values())),
        float(sum(after.values())),
        float(sum(end.values())),
    )
    costs = -sum(
        row.amount
        for row in batch.cashflows
        if row.kind in PORTFOLIO_LEDGER_COST_COLUMNS
    )
    other_flows = [
        row
        for row in batch.cashflows
        if row.kind not in {*PORTFOLIO_LEDGER_COST_COLUMNS, "principal"}
    ]
    if other_flows or endpoint <= entry or b <= 0 or a <= 0 or e <= 0:
        raise ValueError("portfolio_interval_economic_scope_invalid")
    tolerance = _scaled_tolerance(b)
    if not np.isclose(a, b - costs, rtol=0, atol=tolerance):
        raise ValueError("portfolio_interval_cost_reconciliation_invalid")
    diagnostics = {(row.asset_id, row.name): row for row in batch.diagnostics}
    target = {asset: diagnostics[(asset, "target_notional")].number for asset in before}
    decision = {
        asset: diagnostics[(asset, "decision_delta_weight")].number for asset in before
    }
    if any(value is None for value in (*target.values(), *decision.values())):
        raise ValueError("portfolio_interval_decision_facts_missing")
    executed = {}
    for trade in batch.trades:
        executed[trade.asset_id] = executed.get(trade.asset_id, 0.0) + trade.notional
    actual_turnover = (
        0.5
        * (sum(abs(value) for value in executed.values()) + abs(sum(executed.values())))
        / b
    )
    return {
        "entry_at": entry,
        "end_at": endpoint,
        "nav_before": b,
        "nav_post": a,
        "nav_end": e,
        "gross_return": (e - a) / b,
        "net_return": (e - b) / b,
        "cost_drag": float(costs / b),
        "target_turnover": 0.5 * sum(abs(value) for value in decision.values()),
        "requested_turnover": 0.5
        * sum(abs(target[asset] - before[asset]) for asset in before)
        / b,
        "realized_turnover": actual_turnover,
    }
