"""Point-in-time calibration of prediction ranks to realized holding returns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from quant_research_platform.contracts.portfolios import (
    REBALANCE_SCHEDULES,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.validation import strict_real_numeric_series
from quant_research_platform.research.portfolios.schedules import is_rebalance_session


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


def alpha_interval_semantics(rebalance_schedule):
    """Return the scientific holding-return horizon independently of candidate
    identity.
    """
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
    """Bind a published native calibration to this consumer's group and horizon."""
    if (
        payload["schema_id"] != "qrp-final-alpha-calibration"
        or payload["interval_semantics"] != alpha_interval_semantics(rebalance_schedule)
        or payload["selection_group"] != selection_group
        or pd.Timestamp(payload["cutoff"]) != pd.Timestamp(cutoff)
    ):
        raise ValueError("test_alpha_calibration_binding_invalid")
    calibration = AlphaCalibration(**payload["calibration"])
    if calibration.status != "DEFINED":
        raise PortfolioTargetBlocked("test_alpha_calibration_undefined")
    return calibration, payload["cutoff"]
