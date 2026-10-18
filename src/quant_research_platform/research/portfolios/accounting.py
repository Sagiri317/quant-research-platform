"""Continuous-weight portfolio construction support and economic ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import date
from fractions import Fraction
from math import fsum

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from quant_research_platform.compute.portfolio_economics import (
    cash_equity_pnl,
    cash_equity_positions,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_BLOCK_REASONS,
    PORTFOLIO_CASH_ASSET_ID,
    PORTFOLIO_LEDGER_COST_COLUMNS,
    REBALANCE_SCHEDULES,
    ExecutionAssumptions,
    PortfolioLedgerBatch,
    PortfolioRightsAdjustment,
    PortfolioTargetBlocked,
    TradeCostComponents,
    portfolio_notional_tolerance,
)
from quant_research_platform.contracts.validation import (
    finite_exact_float,
    strict_finite_float,
    strict_finite_numeric_series,
    strict_real_numeric_series,
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


def adjust_carried_share_mark(
    mark, corporate_events, rights_events, *, asset, listing, timestamp, mark_at
):
    """Carry an unquoted raw-share claim through evidenced ex-date changes."""
    grouped = {}
    evidence = []
    for event in corporate_events.itertuples(index=False):
        if event.asset_id != asset or event.listing_date != listing:
            continue
        effective = pd.Timestamp(event.effective_at)
        if (
            not mark_at < effective <= timestamp
            or pd.Timestamp(event.available_at) > timestamp
        ):
            continue
        evidence.append(event.evidence_identity)
        if not event.entitlement_proved:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        cash, shares, ratios = grouped.setdefault(
            effective, [Fraction(), Fraction(), []]
        )
        if event.kind == "cash_dividend":
            cash += Fraction(event.cash_per_share)
        elif event.kind == "stock_distribution":
            shares += Fraction(event.additional_shares_per_share)
        else:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        if cash < 0 or shares < 0:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        grouped[effective] = [cash, shares, ratios]
    for event in rights_events.itertuples(index=False):
        if (
            event.asset_id != asset
            or event.listing_date != listing
            or event.kind != "share_change"
        ):
            continue
        effective = pd.Timestamp(event.effective_at)
        if (
            not mark_at < effective <= timestamp
            or pd.Timestamp(event.available_at) > timestamp
        ):
            continue
        evidence.append(event.evidence_identity)
        ratio = Fraction(event.remaining_old_units_per_old_unit)
        if not event.entitlement_proved or ratio <= 0:
            raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
        grouped.setdefault(effective, [Fraction(), Fraction(), []])[2].append(ratio)
    result = Fraction(mark)
    for effective in sorted(grouped):
        cash, shares, ratios = grouped[effective]
        result = (result - cash) / (1 + shares)
        for ratio in ratios:
            result /= ratio
        if result < 0:
            raise PortfolioTargetBlocked("portfolio_unquoted_ex_price_invalid")
    return result, tuple(dict.fromkeys(evidence))


class EquityEntitlements:
    """One owner for registered raw-share rights, receivables and share locks."""

    def __init__(self):
        self.registered = {}
        self.registration_emitted = set()
        self.completed = set()
        self._receivables = {}
        self.locked_quantities = {}
        self._claims = {}

    @property
    def receivables(self):
        values = {
            asset: finite_exact_float(value)
            for asset, value in self._receivables.items()
        }
        if any(value is None for value in values.values()):
            raise PortfolioTargetBlocked("portfolio_dividend_amount_unrepresentable")
        return values

    def _adjustment(
        self,
        event,
        units,
        timestamp,
        *,
        stage,
        phase,
        quantity_after=None,
        cash=0.0,
        receivable_after=None,
        locked_after=None,
    ):
        asset = event.asset_id
        quantity = Fraction(units.get(asset, 0))
        receivable = self.receivables.get(asset, 0.0)
        locked = self.locked_quantities.get(asset, Fraction())
        exact_receivable = (
            self._receivables.get(asset, Fraction())
            if receivable_after is None
            else Fraction(receivable_after)
        )
        rounded_receivable = finite_exact_float(exact_receivable)
        if rounded_receivable is None or exact_receivable < 0:
            raise PortfolioTargetBlocked("portfolio_dividend_amount_unrepresentable")
        record = PortfolioRightsAdjustment(
            event_id=f"{event.event_id}:{stage}",
            timestamp_ns=int(timestamp.value),
            asset_id=asset,
            lifecycle_id=pd.Timestamp(event.listing_date).date().isoformat(),
            quantity_before=quantity,
            quantity_after=quantity if quantity_after is None else quantity_after,
            cash_amount=cash,
            target_asset_id=None,
            target_lifecycle_id=None,
            target_quantity=Fraction(),
            evidence_identity=event.evidence_identity,
            kind=(
                "dividend"
                if event.kind == "cash_dividend" and stage != "registration"
                else "stock_distribution"
                if stage == "effective"
                else stage
            ),
            phase=phase,
            entitlement_quantity=self.registered[event.event_id],
            entitlement_record_at_ns=int(pd.Timestamp(event.record_at).value),
            receivable_before=receivable,
            receivable_after=rounded_receivable,
            locked_quantity_before=locked,
            locked_quantity_after=locked if locked_after is None else locked_after,
        )
        units.loc[asset] = record.quantity_after
        self._receivables[asset] = exact_receivable
        self.locked_quantities[asset] = record.locked_quantity_after
        return record

    def register(self, corporate_events, units, lives, timestamp, *, phase="end"):
        """Snapshot holdings after record-date trading, before any ex-date changes."""
        adjustments = []
        for event in corporate_events.itertuples(index=False):
            if (
                event.event_id in self.registered
                or pd.Timestamp(event.record_at) != timestamp
            ):
                continue
            quantity = (
                Fraction(units.get(event.asset_id, 0))
                if lives.get(event.asset_id) == event.listing_date
                else Fraction()
            )
            available = pd.Timestamp(event.available_at) <= timestamp
            self.registered[event.event_id] = quantity
            if quantity and available and event.entitlement_proved:
                adjustments.append(
                    self._adjustment(
                        event,
                        units.copy(),
                        timestamp,
                        stage="registration",
                        phase=phase,
                    )
                )
                self.registration_emitted.add(event.event_id)
        return tuple(adjustments)

    def process(
        self, corporate_events, rights_events, units, lives, timestamp, *, phase="after"
    ):
        """Apply opening entitlements; a sold original holding keeps its claim."""
        updated, identities = units.astype("object").copy(), lives.copy()
        adjustments = []
        cash = Fraction()
        for event in corporate_events.itertuples(index=False):
            if pd.Timestamp(event.available_at) > timestamp:
                continue
            held = Fraction(updated.get(event.asset_id, 0))
            relevant_holding = (
                held and identities.get(event.asset_id) == event.listing_date
            )
            entitlement = self.registered.get(event.event_id)
            if entitlement == 0:
                continue
            if timestamp < pd.Timestamp(event.effective_at):
                continue
            if not event.entitlement_proved:
                if relevant_holding or entitlement:
                    raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                continue
            if entitlement is None:
                if relevant_holding:
                    raise PortfolioTargetBlocked(
                        "portfolio_registration_snapshot_missing"
                    )
                self.registered[event.event_id] = Fraction()
                continue
            if not entitlement:
                continue
            if event.event_id not in self.registration_emitted:
                adjustments.append(
                    self._adjustment(
                        event,
                        updated,
                        timestamp,
                        stage="registration",
                        phase=phase,
                    )
                )
                self.registration_emitted.add(event.event_id)
            key = (event.event_id, "effective")
            if key not in self.completed:
                if event.kind == "cash_dividend":
                    claim = finite_exact_float(
                        entitlement * Fraction(event.cash_per_share)
                    )
                    if claim is None or claim < 0:
                        raise PortfolioTargetBlocked(
                            "portfolio_dividend_amount_unrepresentable"
                        )
                    self._claims[event.event_id] = claim
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="effective",
                            phase=phase,
                            receivable_after=self._receivables.get(
                                event.asset_id, Fraction()
                            )
                            + Fraction(claim),
                        )
                    )
                elif event.kind == "stock_distribution":
                    shares = entitlement * Fraction(event.additional_shares_per_share)
                    if shares < 0 or (
                        held and identities.get(event.asset_id) != event.listing_date
                    ):
                        raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                    identities.loc[event.asset_id] = event.listing_date
                    self._claims[event.event_id] = shares
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="effective",
                            phase=phase,
                            quantity_after=held + shares,
                            locked_after=self.locked_quantities.get(
                                event.asset_id, Fraction()
                            )
                            + shares,
                        )
                    )
                else:
                    raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
                self.completed.add(key)
            key = (event.event_id, "settlement")
            if (
                pd.notna(event.settlement_at)
                and timestamp >= pd.Timestamp(event.settlement_at)
                and key not in self.completed
            ):
                claim = self._claims.pop(event.event_id)
                if event.kind == "cash_dividend":
                    cash += Fraction(claim)
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="settlement",
                            phase=phase,
                            cash=claim,
                            receivable_after=self._receivables[event.asset_id]
                            - Fraction(claim),
                        )
                    )
                else:
                    adjustments.append(
                        self._adjustment(
                            event,
                            updated,
                            timestamp,
                            stage="unlock",
                            phase=phase,
                            locked_after=self.locked_quantities[event.asset_id] - claim,
                        )
                    )
                self.completed.add(key)
        for event in rights_events.itertuples(index=False):
            if event.kind == "quote" or (event.event_id, "rights") in self.completed:
                continue
            quantity = Fraction(updated.get(event.asset_id, 0))
            if not quantity or identities.get(event.asset_id) != event.listing_date:
                continue
            if not event.entitlement_proved:
                raise PortfolioTargetBlocked("portfolio_held_rights_unproved")
            if self.locked_quantities.get(event.asset_id, 0):
                raise PortfolioTargetBlocked("portfolio_locked_reorganization_unproved")
            remainder = quantity * Fraction(event.remaining_old_units_per_old_unit)
            amount = (
                finite_exact_float(quantity * Fraction(event.value_per_old_unit))
                if event.kind == "cash_recovery"
                else 0.0
            )
            if amount is None:
                raise PortfolioTargetBlocked(
                    "portfolio_recovery_amount_unrepresentable"
                )
            target = event.successor_asset_id if event.kind == "conversion" else None
            target_life = event.successor_listing_date if target is not None else None
            target_quantity = (
                quantity * Fraction(event.successor_units_per_old_unit)
                if target is not None
                else Fraction()
            )
            updated.loc[event.asset_id] = remainder
            if target is not None:
                if (
                    updated.get(target, 0) != 0
                    and identities.get(target) != target_life
                ):
                    raise PortfolioTargetBlocked(
                        "portfolio_simultaneous_lifecycles_unrepresentable"
                    )
                updated.loc[target] = Fraction(updated.get(target, 0)) + target_quantity
                identities.loc[target] = target_life
            adjustments.append(
                PortfolioRightsAdjustment(
                    event.event_id,
                    int(timestamp.value),
                    event.asset_id,
                    pd.Timestamp(event.listing_date).date().isoformat(),
                    quantity,
                    remainder,
                    amount,
                    target,
                    pd.Timestamp(target_life).date().isoformat()
                    if target is not None
                    else None,
                    target_quantity,
                    event.evidence_identity,
                    phase=phase,
                    receivable_before=self.receivables.get(event.asset_id, 0.0),
                    receivable_after=self.receivables.get(event.asset_id, 0.0),
                )
            )
            self.completed.add((event.event_id, "rights"))
            cash += Fraction(amount)
        result = finite_exact_float(cash)
        if result is None:
            raise PortfolioTargetBlocked("portfolio_recovery_amount_unrepresentable")
        return updated, identities, result, tuple(adjustments)


@dataclass(frozen=True, slots=True)
class ExecutionAccounting:
    """One transition value backed by one canonical per-asset notional ledger."""

    _execution_date: date
    _asset_ledger: pd.DataFrame
    _nav_before: float
    _pre_trade_cash: float
    _post_trade_cash: float
    cost_source_identity: str
    receivable: float = 0.0

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
        numeric = raw_numeric.apply(
            strict_finite_numeric_series, error="execution_ledger_numeric_invalid"
        )
        for column in numeric.columns:
            ledger[column] = numeric[column]
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
        receivable = _finite_nonnegative(
            self.receivable, "execution_receivable_invalid"
        )
        nav_before = _finite_nonnegative(self._nav_before, "execution_nav_invalid")
        tolerance = portfolio_notional_tolerance(nav_before)
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
        nav_after = float(numeric["post_trade_notional"].sum() + post_cash + receivable)
        if not np.isclose(
            numeric["pre_trade_notional"].sum() + pre_cash + receivable,
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
        if type(self.cost_source_identity) is not str or not self.cost_source_identity:
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
            self._asset_ledger["post_trade_notional"].sum()
            + self._post_trade_cash
            + self.receivable
        )

    @property
    def pre_trade_cash(self) -> float:
        return float(self._pre_trade_cash)

    @property
    def post_trade_cash(self) -> float:
        return float(self._post_trade_cash)


@dataclass(frozen=True)
class AlphaCalibration:
    slope: float | None
    intercept: float | None
    slope_standard_error: float | None
    slope_t_statistic: float | None
    slope_pvalue: float | None
    slope_ci_low: float | None
    slope_ci_high: float | None
    weighted_r2: float | None
    coverage: float
    valid_rebalance_dates: int
    scheduled_rebalance_dates: int
    undefined_reason: str | None

    @property
    def status(self) -> str:
        return "UNDEFINED" if self.undefined_reason else "DEFINED"

    @property
    def inference_status(self) -> str:
        return "UNDEFINED" if self.slope_t_statistic is None else "DEFINED"

    @property
    def inference_reason(self) -> str:
        return self.undefined_reason or (
            "alpha_calibration_zero_standard_error"
            if self.slope_t_statistic is None
            else ""
        )

    @property
    def r2_status(self) -> str:
        return "UNDEFINED" if self.weighted_r2 is None else "DEFINED"

    @property
    def r2_reason(self) -> str:
        return self.undefined_reason or (
            "alpha_calibration_constant_return" if self.weighted_r2 is None else ""
        )

    @property
    def benchmark_only(self) -> bool:
        return self.undefined_reason is not None or self.slope <= 0.0


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


def alpha_calibration_start(sessions: pd.DatetimeIndex, *, as_of) -> pd.Timestamp:
    """Start of the 252-session decision window on the complete market clock."""
    axis = pd.DatetimeIndex(sessions)
    cutoff = pd.Timestamp(as_of).normalize()
    if (
        axis.empty
        or axis.hasnans
        or not axis.is_unique
        or not axis.is_monotonic_increasing
        or pd.isna(cutoff)
    ):
        raise ValueError("alpha_calibration_calendar_invalid")
    end = axis.searchsorted(cutoff, side="right")
    if not end:
        raise ValueError("alpha_calibration_calendar_invalid")
    return axis[max(0, end - 252)]


def calibrate_expected_alpha(
    frame: pd.DataFrame,
    *,
    alpha_column: str = "prediction_rank_score",
    return_column: str = "holding_return",
    as_of: pd.Timestamp,
    scheduled_intervals: pd.DataFrame,
    trading_sessions: pd.DatetimeIndex,
    minimum_coverage: float = 0.8,
    minimum_valid_dates: int = 6,
) -> AlphaCalibration:
    """Calibrate expected alpha with equal-timestamp weighted pooled OLS."""

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
            intervals.timestamp.ge(
                alpha_calibration_start(trading_sessions, as_of=cutoff)
            )
            & intervals.available_at.le(cutoff),
            "timestamp",
        ]
    )
    alpha = strict_real_numeric_series(
        frame[alpha_column], error="alpha_calibration_numeric_invalid"
    )
    returns = strict_real_numeric_series(
        frame[return_column], error="alpha_calibration_numeric_invalid"
    )
    valid = alpha.notna() & returns.notna() & np.isfinite(alpha) & np.isfinite(returns)
    valid_dates = dates.loc[valid].isin(scheduled)
    date_count = 0 if scheduled.empty else int(dates.loc[valid & valid_dates].nunique())
    coverage = date_count / max(1, len(scheduled))
    enough_coverage = date_count >= minimum_valid_dates and coverage >= minimum_coverage

    def undefined(reason: str) -> AlphaCalibration:
        return AlphaCalibration(
            slope=None,
            intercept=None,
            slope_standard_error=None,
            slope_t_statistic=None,
            slope_pvalue=None,
            slope_ci_low=None,
            slope_ci_high=None,
            weighted_r2=None,
            coverage=coverage,
            valid_rebalance_dates=date_count,
            scheduled_rebalance_dates=len(scheduled),
            undefined_reason=reason,
        )

    if not enough_coverage:
        return undefined("alpha_calibration_insufficient_coverage")
    selected = valid & valid_dates
    left = alpha.loc[selected].to_numpy(dtype="float64")
    right = returns.loc[selected].to_numpy(dtype="float64")
    _, cluster_codes = np.unique(dates.loc[selected].to_numpy(), return_inverse=True)
    counts = np.bincount(cluster_codes)
    cluster_count = len(counts)
    weights = 1.0 / counts[cluster_codes]
    # Power-of-two scaling preserves inputs; centering avoids raw-moment cancellation.
    left_exponent = int(np.frexp(np.max(np.abs(left)))[1])
    right_exponent = int(np.frexp(np.max(np.abs(right)))[1])
    left = np.ldexp(left, -left_exponent)
    right = np.ldexp(right, -right_exponent)
    left_offset, right_offset = left[0], right[0]
    left_shift, right_shift = left - left_offset, right - right_offset
    left_mean = (
        np.sum(np.bincount(cluster_codes, weights=left_shift) / counts) / cluster_count
    )
    right_mean = (
        np.sum(np.bincount(cluster_codes, weights=right_shift) / counts) / cluster_count
    )
    centered_left = left_shift - left_mean
    centered_right = right_shift - right_mean
    weighted_left = weights * centered_left
    total_left = centered_left @ weighted_left
    if total_left == 0.0:
        return undefined("alpha_calibration_rank_deficient")
    slope = (weighted_left @ centered_right) / total_left
    intercept = right_offset + right_mean - slope * (left_offset + left_mean)
    fitted = intercept + slope * left
    residual = right - fitted
    epsilon = np.finfo(np.float64).eps
    # Include affine operands, not just their possibly cancelled final value.
    # These are the same scaled coordinates used for the centered fit above.
    residual_resolution = (
        8
        * epsilon
        * (
            np.abs(right)
            + abs(right_offset)
            + abs(right_mean)
            + abs(intercept)
            + np.abs(fitted)
            + abs(slope) * (np.abs(left) + abs(left_offset) + abs(left_mean))
        )
    )
    resolved_zero = np.abs(residual) <= residual_resolution
    residual[resolved_zero] = 0.0
    residual_resolution[resolved_zero] = 0.0
    score_terms = centered_left * residual
    scores = np.bincount(cluster_codes, weights=score_terms)
    # Propagate only this date's active residual and dot-product rounding bounds.
    score_resolution = np.bincount(
        cluster_codes, weights=np.abs(centered_left) * residual_resolution
    ) + (counts * epsilon / (1 - counts * epsilon)) * np.bincount(
        cluster_codes, weights=np.abs(score_terms)
    )
    scores[np.abs(scores) <= score_resolution] = 0.0
    influence = scores / counts / total_left
    scaled_standard_error = np.linalg.norm(influence)
    slope_value = float(np.ldexp(slope, right_exponent - left_exponent))
    standard_error = float(
        np.ldexp(scaled_standard_error, right_exponent - left_exponent)
    )
    if slope and slope_value == 0.0:
        raise ValueError("alpha_calibration_slope_unrepresentable")
    if scaled_standard_error and standard_error == 0.0:
        raise ValueError("alpha_calibration_standard_error_unrepresentable")
    t_statistic = (
        float(slope / scaled_standard_error) if scaled_standard_error else None
    )
    if t_statistic is not None:
        degrees_freedom = max(1, cluster_count - 2)
        pvalue = float(2.0 * student_t.sf(abs(t_statistic), df=degrees_freedom))
        critical = float(student_t.ppf(0.975, df=degrees_freedom))
        ci_low = slope_value - critical * standard_error
        ci_high = slope_value + critical * standard_error
    else:
        pvalue = ci_low = ci_high = None
    total_variation = centered_right @ (weights * centered_right)
    residual_variation = residual @ (weights * residual)
    return AlphaCalibration(
        slope=slope_value,
        intercept=float(np.ldexp(intercept, right_exponent)),
        slope_standard_error=standard_error,
        slope_t_statistic=t_statistic,
        slope_pvalue=pvalue,
        slope_ci_low=ci_low,
        slope_ci_high=ci_high,
        weighted_r2=float(1 - residual_variation / total_variation)
        if total_variation
        else None,
        coverage=coverage,
        valid_rebalance_dates=date_count,
        scheduled_rebalance_dates=len(scheduled),
        undefined_reason=None,
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
    cost_totals_for_buys: Callable[
        [tuple[str, ...], np.ndarray, np.ndarray], np.ndarray
    ],
    assumptions: ExecutionAssumptions,
    cost_source_identity: str,
    unquoted_assets: frozenset[str] = frozenset(),
    receivable: float = 0.0,
    locked_notional: pd.Series | None = None,
    share_quantities: pd.Series | None = None,
    locked_share_quantities: pd.Series | None = None,
    execution_prices: pd.Series | None = None,
) -> ExecutionAccounting:
    """Execute sell-first, cash-constrained continuous-weight orders."""

    if type(execution_date) is not date:
        raise ValueError("execution_date_invalid")
    if not callable(cost_for_trade) or not isinstance(
        assumptions, ExecutionAssumptions
    ):
        raise ValueError("execution_inputs_invalid")
    nav_value = _finite_nonnegative(nav, "execution_nav_invalid")
    receivable = _finite_nonnegative(receivable, "execution_receivable_invalid")
    assets = _canonical_assets(current_weights, target_weights, allow_empty=True)
    current = _weight_values(current_weights, assets, "current_weights_invalid")
    target = _weight_values(target_weights, assets, "target_weights_invalid")
    if current.lt(0.0).any() or target.lt(0.0).any():
        raise ValueError("execution_weights_invalid")
    weight_bound = 1.0 + 512.0 * np.finfo(float).eps
    if (
        sum(map(float, current)) > weight_bound
        or sum(map(float, target)) > weight_bound
    ):
        raise ValueError("execution_weights_sum_invalid")
    tolerance = portfolio_notional_tolerance(nav_value)
    enter = _bool_values(can_enter, assets, "execution_entry_mask_invalid")
    exit_ = _bool_values(can_exit, assets, "execution_exit_mask_invalid")
    if not isinstance(unquoted_assets, frozenset) or not unquoted_assets.issubset(
        assets
    ):
        raise ValueError("execution_unquoted_assets_invalid")

    def blocked_reason(asset, permission):
        if asset in unquoted_assets:
            return "research_unquoted_valuation"
        if pd.isna(permission):
            return "execution_fact_unknown"
        if not permission:
            return "market_blocked"
        return "adv_unavailable" if pd.isna(adv[asset]) else None

    pre_notional = current * nav_value
    target_notional = target * nav_value
    locked = (
        pd.Series(0.0, index=assets)
        if locked_notional is None
        else _weight_values(locked_notional, assets, "execution_locked_invalid")
    )
    if locked.lt(0).any() or (locked > pre_notional + tolerance).any():
        raise ValueError("execution_locked_invalid")
    exact_sale_limits = None
    if share_quantities is not None:
        if execution_prices is None or locked_share_quantities is None:
            raise ValueError("execution_share_inputs_invalid")
        exact_sale_limits = {}
        for asset in assets:
            quantity = Fraction(share_quantities.get(asset, 0))
            locked_quantity = Fraction(locked_share_quantities.get(asset, 0))
            price = Fraction(execution_prices.get(asset, 0))
            if not 0 <= locked_quantity <= quantity or price < 0:
                raise ValueError("execution_share_inputs_invalid")
            exact_limit = (quantity - locked_quantity) * price
            limit = float(exact_limit)
            if Fraction(limit) > exact_limit:
                limit = np.nextafter(limit, -np.inf)
            exact_sale_limits[asset] = limit
    elif locked_share_quantities is not None or execution_prices is not None:
        raise ValueError("execution_share_inputs_invalid")
    desired = target_notional - pre_notional
    adv = _asset_values(average_daily_volume, assets, "execution_adv_invalid")
    participation = assumptions.adv_participation
    capacity = adv * participation
    records = {
        str(asset): _empty_ledger_row(
            asset, execution_date, pre_notional[asset], target_notional[asset]
        )
        for asset in assets
    }
    sell_delta = pd.Series(0.0, index=assets, dtype="float64")
    for asset in assets:
        desired_delta = float(desired[asset])
        if desired_delta >= 0.0:
            continue
        record = records[str(asset)]
        if reason := blocked_reason(asset, exit_[asset]):
            record["reason"] = reason
            continue
        amount = min(
            -desired_delta,
            float(capacity[asset]),
            max(0.0, float(pre_notional[asset] - locked[asset])),
            exact_sale_limits[asset] if exact_sale_limits is not None else np.inf,
        )
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
        sell_delta[asset] = -amount
        _update_ledger_row(
            record,
            executed_delta=-amount,
            reason=(
                "locked_shares"
                if locked[asset] > 0
                and amount >= pre_notional[asset] - locked[asset] - tolerance
                else "adv_capacity"
            )
            if amount < -desired_delta - tolerance
            else None,
            cost=cost,
        )
    pre_trade_cash = fsum((nav_value, -receivable, *(-pre_notional)))
    if -tolerance <= pre_trade_cash < 0.0:
        pre_trade_cash = 0.0
    cash_after_sells = fsum(
        (
            pre_trade_cash,
            *(-sell_delta),
            *(
                -record[name]
                for record in records.values()
                for name in PORTFOLIO_LEDGER_COST_COLUMNS
            ),
        )
    )
    if cash_after_sells < -tolerance:
        raise ValueError("execution_cash_after_sells_invalid")

    buy_candidates: dict[str, float] = {}
    for asset in assets:
        desired_delta = float(desired[asset])
        if desired_delta <= 0.0:
            continue
        record = records[str(asset)]
        if reason := blocked_reason(asset, enter[asset]):
            record["reason"] = reason
            continue
        buy_candidates[str(asset)] = min(desired_delta, float(capacity[asset]))

    buy_assets = tuple(buy_candidates)
    buy_amounts = np.asarray(tuple(buy_candidates.values()), dtype="float64")
    buy_adv = adv.reindex(buy_assets).to_numpy(dtype="float64")

    def buy_spend_total(scale: float) -> float:
        amounts = buy_amounts * scale
        costs = np.asarray(cost_totals_for_buys(buy_assets, amounts, buy_adv))
        if (
            costs.shape != amounts.shape
            or not np.isfinite(costs).all()
            or (costs < 0).any()
        ):
            raise ValueError("execution_cost_provider_invalid")
        try:
            return fsum((*amounts, *costs))
        except OverflowError:
            # Every term is nonnegative: this trial exceeds any finite cash.
            return np.inf

    scale = 0.0
    if buy_candidates and cash_after_sells > 0.0:
        if buy_spend_total(1.0) <= cash_after_sells:
            scale = 1.0
        else:
            low, high = 0.0, 1.0
            for _ in range(80):
                middle = (low + high) / 2.0
                if middle == low or middle == high:
                    break
                if buy_spend_total(middle) <= cash_after_sells:
                    low = middle
                else:
                    high = middle
            scale = low
    buy_delta = pd.Series(0.0, index=assets, dtype="float64")
    for asset, candidate in buy_candidates.items():
        amount = candidate * scale
        cost = None
        if amount > 0.0:
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
            cost=cost,
        )
    total_cost = fsum(
        record[name]
        for record in records.values()
        for name in PORTFOLIO_LEDGER_COST_COLUMNS
    )
    # Reconcile cash from the same component amounts that become native flows.
    # Repeated float accumulation loses NAV ULPs as the number of fills grows.
    post_trade_cash = fsum(
        (
            pre_trade_cash,
            *(-sell_delta),
            *(-buy_delta),
            *(
                -record[name]
                for record in records.values()
                for name in PORTFOLIO_LEDGER_COST_COLUMNS
            ),
        )
    )
    if -tolerance <= post_trade_cash < 0.0:
        post_trade_cash = 0.0
    post_trade = pre_notional + sell_delta + buy_delta
    nav_after = float(post_trade.sum() + post_trade_cash + receivable)
    expected_nav_after = nav_value - total_cost
    if post_trade_cash < -tolerance or not np.isfinite(nav_after) or nav_after < 0.0:
        raise ValueError("execution_cash_invalid")
    if not np.isclose(nav_after, expected_nav_after, rtol=0.0, atol=tolerance):
        raise ValueError("execution_self_financing_invalid")
    for asset in assets:
        record = records[str(asset)]
        record["post_trade_notional"] = float(post_trade[asset])
        if abs(record["target_notional"] - record["post_trade_notional"]) <= tolerance:
            record["reason"] = None
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
        receivable=receivable,
    )


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
    return strict_finite_numeric_series(values.reindex(assets), error=error)


def _asset_values(
    values: pd.Series,
    assets: pd.Index,
    error: str,
) -> pd.Series:
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError(error)
    result = strict_finite_numeric_series(
        values.reindex(assets), error=error, allow_null=True
    ).astype("float64")
    if result.lt(0.0).any():
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
    if not assets.isin(values.index).all():
        raise ValueError(error)
    return values.reindex(assets).astype("boolean")


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


def _finite_nonnegative(value: float, error: str) -> float:
    numeric = strict_finite_float(value, error=error)
    if numeric < 0.0:
        raise ValueError(error)
    return numeric


def capacity_quantiles(
    capacity_by_rebalance_date: pd.Series,
) -> dict[str, object]:
    values = strict_real_numeric_series(
        capacity_by_rebalance_date, error="capacity_numeric_invalid"
    ).dropna()
    if values.empty:
        raise ValueError("capacity_observations_empty")
    return {
        "formal_p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "minimum": float(values.min()),
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
        fsum(before.values()),
        fsum(after.values()),
        fsum(end.values()),
    )
    costs = -sum(
        Fraction(row.amount)
        for row in batch.cashflows
        if row.kind in PORTFOLIO_LEDGER_COST_COLUMNS
    )
    other_flows = [
        row
        for row in batch.cashflows
        if row.kind
        not in {*PORTFOLIO_LEDGER_COST_COLUMNS, "principal", "recovery", "dividend"}
    ]
    if other_flows or endpoint <= entry or b < 0 or a < 0 or e < 0:
        raise ValueError("portfolio_interval_economic_scope_invalid")
    tolerance = portfolio_notional_tolerance(b)
    initial = {row.asset_id: row for row in batch.valuations if row.phase == "before"}
    post = {row.asset_id: row for row in batch.valuations if row.phase == "after"}
    traded_quantities, executed = {}, {}
    for trade in batch.trades:
        asset = trade.asset_id
        traded_quantities[asset] = traded_quantities.get(asset, 0) + Fraction(
            trade.quantity
        )
        executed[asset] = executed.get(asset, 0.0) + trade.notional
    rights_change = sum(
        float(
            (Fraction(post[asset].quantity) - traded_quantities.get(asset, 0))
            * Fraction(post[asset].mark_price)
        )
        + post[asset].receivable
        - initial[asset].value
        for asset in initial
        if asset != PORTFOLIO_CASH_ASSET_ID
    ) + sum(
        row.amount
        for row in batch.cashflows
        if row.kind in {"recovery", "dividend"} and row.timestamp_ns <= entry.value
    )
    if not np.isclose(
        a,
        b + rights_change - float(costs),
        rtol=0,
        atol=max(tolerance, portfolio_notional_tolerance(a)),
    ):
        raise ValueError("portfolio_interval_cost_reconciliation_invalid")
    diagnostics = {(row.asset_id, row.name): row for row in batch.diagnostics}
    target = {asset: diagnostics[(asset, "target_notional")].number for asset in before}
    decision = {
        asset: diagnostics[(asset, "decision_delta_weight")].number for asset in before
    }
    if any(value is None for value in (*target.values(), *decision.values())):
        raise ValueError("portfolio_interval_decision_facts_missing")

    def ratio(numerator, denominator):
        if denominator == 0:
            return np.nan
        value = finite_exact_float(Fraction(numerator) / Fraction(denominator))
        return np.nan if value is None else value

    net_pnl = cash_equity_pnl(
        cash_equity_positions(row for row in batch.valuations if row.phase == "before"),
        cash_equity_positions(row for row in batch.valuations if row.phase == "end"),
        batch.cashflows,
        batch.rights_adjustments,
    )
    turnover = (
        sum((abs(Fraction(value)) for value in executed.values()), Fraction())
        + abs(sum((Fraction(value) for value in executed.values()), Fraction()))
    ) / 2
    requested_stock = [
        Fraction(target[asset])
        - Fraction(value.share_value)
        + Fraction(executed.get(asset, 0.0))
        for asset, value in post.items()
        if asset != PORTFOLIO_CASH_ASSET_ID
    ]
    # Requests are measured at the execution open, after nontrade rights changes.
    # The cash leg offsets net stock requests; receivables are never traded cash.
    requested = (
        sum(map(abs, requested_stock), Fraction())
        + abs(sum(requested_stock, Fraction()))
    ) / 2
    return {
        "entry_at": entry,
        "end_at": endpoint,
        "nav_before": b,
        "nav_post": a,
        "cash_post": after[PORTFOLIO_CASH_ASSET_ID],
        "cash_weight_post": ratio(after[PORTFOLIO_CASH_ASSET_ID], a),
        "nav_end": e,
        "gross_return": ratio(net_pnl + Fraction(costs), b),
        "net_return": ratio(net_pnl, b),
        "cost_drag": ratio(costs, b),
        "target_turnover": 0.5 * sum(abs(value) for value in decision.values())
        if b > 0
        else np.nan,
        "requested_turnover": ratio(requested, b),
        "realized_turnover": ratio(turnover, b),
    }


def alpha_interval_semantics(rebalance_schedule):
    """Only the scientific return horizon, independent of candidate identity."""
    if (
        type(rebalance_schedule) is not str
        or rebalance_schedule not in REBALANCE_SCHEDULES
    ):
        raise ValueError("unsupported_rebalance_schedule")
    return {
        "rebalance_schedule": rebalance_schedule,
        "return_convention": (
            "next_execution_open_to_following_rebalance_execution_open"
        ),
        "calendar_convention": "a_share_unified_trading_sessions",
    }


def frozen_alpha_calibration(payload, *, cutoff, selection_group, rebalance_schedule):
    """Validate the final validation calibration bound to its actual group inputs."""
    if (
        type(payload) is not dict
        or set(payload)
        != {
            "schema_id",
            "selection_group",
            "cutoff",
            "inputs",
            "calibration",
            "interval_semantics",
        }
        or (
            payload["interval_semantics"]
            != alpha_interval_semantics(rebalance_schedule)
            or payload["schema_id"] != "qrp-final-alpha-calibration"
            or payload["selection_group"] != selection_group
            or pd.Timestamp(payload["cutoff"]) != pd.Timestamp(cutoff)
            or (type(payload["calibration"]) is not dict)
            or (
                set(payload["calibration"])
                != {field.name for field in fields(AlphaCalibration)}
            )
        )
    ):
        raise ValueError("test_alpha_calibration_binding_invalid")
    values = payload["calibration"]
    calibration = AlphaCalibration(**values)
    if calibration.status != "DEFINED":
        raise PortfolioTargetBlocked("test_alpha_calibration_undefined")
    if calibration.valid_rebalance_dates < 6 or calibration.coverage < 0.8:
        raise PortfolioTargetBlocked("test_alpha_calibration_insufficient")
    required = ("slope", "intercept", "slope_standard_error", "coverage")
    if any(
        type(values[name]) not in {int, float} or not np.isfinite(values[name])
        for name in required
    ):
        raise ValueError("test_alpha_calibration_statistics_invalid")
    if (
        calibration.slope_standard_error < 0
        or not 0 <= calibration.coverage <= 1
        or type(calibration.valid_rebalance_dates) is not int
        or (type(calibration.scheduled_rebalance_dates) is not int)
        or (calibration.scheduled_rebalance_dates < calibration.valid_rebalance_dates)
    ):
        raise ValueError("test_alpha_calibration_statistics_invalid")
    optional = (
        "slope_t_statistic",
        "slope_pvalue",
        "slope_ci_low",
        "slope_ci_high",
        "weighted_r2",
    )
    if any(
        values[name] is not None
        and (type(values[name]) not in {int, float} or not np.isfinite(values[name]))
        for name in optional
    ):
        raise ValueError("test_alpha_calibration_statistics_invalid")
    if (
        (calibration.slope_standard_error == 0)
        != (calibration.slope_t_statistic is None)
        or len({values[name] is None for name in optional[:4]}) != 1
        or (
            calibration.slope_pvalue is not None
            and (not 0 <= calibration.slope_pvalue <= 1)
        )
        or (
            calibration.slope_ci_low is not None
            and calibration.slope_ci_low > calibration.slope_ci_high
        )
    ):
        raise ValueError("test_alpha_calibration_statistics_invalid")
    return (calibration, payload["cutoff"])
