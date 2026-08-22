"""Portfolio constant-notional additive return and performance diagnostics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_constants import (
    RETURNS_COLUMNS,
)
from quant_research_platform.compute.grouping import normalize_group_key
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_hashable_key_series,
    strict_string_series,
    strict_timestamp_series,
)

ADDITIVE_CUMULATIVE_RETURN_COLUMNS = [
    "timestamp",
    "signal",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "additive_cumulative_gross_return",
    "additive_cumulative_net_return",
    "additive_cumulative_cost",
    "high_water_mark",
    "drawdown",
]
PERFORMANCE_SUMMARY_COLUMNS = [
    "signal",
    "periods",
    "final_additive_cumulative_gross_return",
    "final_additive_cumulative_net_return",
    "mean_net_portfolio_return",
    "volatility_net_portfolio_return",
    "downside_volatility",
    "sharpe_like",
    "sortino_like",
    "calmar_like",
    "best_period_return",
    "worst_period_return",
    "skew",
    "kurtosis",
    "var_95",
    "cvar_95",
    "tail_ratio",
    "positive_return_rate",
    "negative_return_rate",
    "win_rate",
    "loss_rate",
    "profit_factor",
    "payoff_ratio",
    "average_win",
    "average_loss",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "max_drawdown",
    "drawdown_duration",
    "max_drawdown_start",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    "mean_turnover",
    "max_turnover",
    "total_cost",
    "mean_cost",
    "cost_to_gross_return",
    "cost_to_net_return",
    "sharpe_like_reason",
    "sortino_like_reason",
    "calmar_like_reason",
    "skew_reason",
    "kurtosis_reason",
    "tail_ratio_reason",
    "profit_factor_reason",
    "payoff_ratio_reason",
    "average_win_reason",
    "average_loss_reason",
    "cost_to_gross_return_reason",
    "cost_to_net_return_reason",
]
_PERFORMANCE_SUMMARY_INTEGER_COLUMNS = (
    "periods",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "drawdown_duration",
)
_PERFORMANCE_SUMMARY_STRING_COLUMNS = (
    "signal",
    "max_drawdown_start",
    "max_drawdown_valley",
    "max_drawdown_recovery",
    *(column for column in PERFORMANCE_SUMMARY_COLUMNS if column.endswith("_reason")),
)


def _new_performance_stats() -> dict[str, object]:
    """Create the single streaming/full performance state."""

    return {
        "periods": 0,
        "gross_sum": 0.0,
        "cumulative_gross": 0.0,
        "net_sum": 0.0,
        "cumulative_net": 0.0,
        "net_square_sum": 0.0,
        "downside_sum": 0.0,
        "downside_square_sum": 0.0,
        "wins": 0,
        "losses": 0,
        "win_sum": 0.0,
        "loss_sum": 0.0,
        "best": None,
        "worst": None,
        "turnover_sum": 0.0,
        "turnover_max": None,
        "cost_sum": 0.0,
        "high_water_mark": 0.0,
        "max_drawdown": 0.0,
        "drawdown_peak_value": None,
        "drawdown_start": "",
        "drawdown_valley": "",
        "drawdown_recovery": "",
        "drawdown_start_index": -1,
        "drawdown_recovery_index": None,
        "peak_timestamp": "",
        "peak_index": -1,
        "current_win_streak": 0,
        "current_loss_streak": 0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "net_sum_cubes": 0.0,
        "net_sum_quads": 0.0,
    }


def _update_performance_stats_batch(
    stats: dict[str, object],
    *,
    timestamps: np.ndarray,
    gross_returns: np.ndarray,
    net_returns: np.ndarray,
    costs: np.ndarray,
    turnovers: np.ndarray,
) -> None:
    """Update the authoritative performance state for an ordered batch."""

    gross = np.asarray(gross_returns, dtype="float64")
    net = np.asarray(net_returns, dtype="float64")
    cost = np.asarray(costs, dtype="float64")
    turnover = np.asarray(turnovers, dtype="float64")
    count = len(net)
    if count == 0:
        return
    if not (
        len(timestamps) == count
        and len(gross) == count
        and len(cost) == count
        and len(turnover) == count
    ):
        raise ValueError("performance update batch columns must have equal length")

    period_offset = int(stats["periods"])
    previous_net = float(stats["net_sum"])
    previous_cumulative_gross = float(stats["cumulative_gross"])
    previous_cumulative_net = float(stats["cumulative_net"])
    previous_max_drawdown = float(stats["max_drawdown"])
    previous_peak_index = int(stats["peak_index"])
    previous_peak_timestamp = str(stats["peak_timestamp"])
    previous_recovery_index = stats["drawdown_recovery_index"]
    previous_peak_value = stats["drawdown_peak_value"]
    previous_win_streak = int(stats["current_win_streak"])
    previous_loss_streak = int(stats["current_loss_streak"])

    downside = np.minimum(net, 0.0)
    wins = net > 0
    losses = net < 0
    stats["periods"] = period_offset + count
    stats["gross_sum"] = float(stats["gross_sum"]) + float(gross.sum())
    stats["net_sum"] = previous_net + float(net.sum())
    stats["net_square_sum"] = float(stats["net_square_sum"]) + float(
        np.square(net).sum()
    )
    stats["net_sum_cubes"] = float(stats["net_sum_cubes"]) + float(
        np.power(net, 3).sum()
    )
    stats["net_sum_quads"] = float(stats["net_sum_quads"]) + float(
        np.power(net, 4).sum()
    )
    stats["downside_sum"] = float(stats["downside_sum"]) + float(downside.sum())
    stats["downside_square_sum"] = float(stats["downside_square_sum"]) + float(
        np.square(downside).sum()
    )
    stats["wins"] = int(stats["wins"]) + int(wins.sum())
    stats["losses"] = int(stats["losses"]) + int(losses.sum())
    stats["win_sum"] = float(stats["win_sum"]) + float(net[wins].sum())
    stats["loss_sum"] = float(stats["loss_sum"]) + float(net[losses].sum())
    stats["best"] = (
        float(net.max())
        if stats["best"] is None
        else max(float(stats["best"]), float(net.max()))
    )
    stats["worst"] = (
        float(net.min())
        if stats["worst"] is None
        else min(float(stats["worst"]), float(net.min()))
    )
    stats["current_win_streak"] = _trailing_true_run(
        wins,
        previous_win_streak,
    )
    stats["current_loss_streak"] = _trailing_true_run(
        losses,
        previous_loss_streak,
    )
    stats["max_consecutive_wins"] = max(
        int(stats["max_consecutive_wins"]),
        _max_true_run(wins, previous_win_streak),
    )
    stats["max_consecutive_losses"] = max(
        int(stats["max_consecutive_losses"]),
        _max_true_run(losses, previous_loss_streak),
    )
    stats["cost_sum"] = float(stats["cost_sum"]) + float(cost.sum())
    stats["turnover_sum"] = float(stats["turnover_sum"]) + float(turnover.sum())
    stats["turnover_max"] = (
        float(turnover.max())
        if stats["turnover_max"] is None
        else max(float(stats["turnover_max"]), float(turnover.max()))
    )

    gross_cumulative = _ordered_cumulative(
        gross,
        initial=previous_cumulative_gross,
    )
    stats["cumulative_gross"] = float(gross_cumulative[-1])
    cumulative = _ordered_cumulative(net, initial=previous_cumulative_net)
    stats["cumulative_net"] = float(cumulative[-1])
    previous_high_water = float(stats["high_water_mark"])
    high_water = np.maximum.accumulate(
        np.concatenate(([previous_high_water], cumulative))
    )[1:]
    high_water_before = np.concatenate(([previous_high_water], high_water[:-1]))
    peak_updates = cumulative >= high_water_before
    peak_positions = np.flatnonzero(peak_updates)
    latest_peak_positions = np.maximum.accumulate(
        np.where(
            peak_updates,
            np.arange(count, dtype="int64"),
            -1,
        )
    )
    if len(peak_positions):
        peak_position = int(peak_positions[-1])
        stats["peak_timestamp"] = _timestamp_iso(timestamps, peak_position)
        stats["peak_index"] = period_offset + peak_position
    stats["high_water_mark"] = float(high_water[-1])

    drawdown = cumulative - high_water
    prior_drawdown_min = np.minimum.accumulate(
        np.concatenate(([previous_max_drawdown], drawdown))
    )[:-1]
    new_drawdown_positions = np.flatnonzero(drawdown < prior_drawdown_min)
    if len(new_drawdown_positions):
        drawdown_position = int(new_drawdown_positions[-1])
        peak_value = float(high_water[drawdown_position])
        peak_position = int(latest_peak_positions[drawdown_position])
        stats["max_drawdown"] = float(drawdown[drawdown_position])
        stats["drawdown_peak_value"] = peak_value
        stats["drawdown_start"] = (
            _timestamp_iso(timestamps, peak_position)
            if peak_position >= 0
            else previous_peak_timestamp
        )
        stats["drawdown_valley"] = _timestamp_iso(
            timestamps,
            drawdown_position,
        )
        stats["drawdown_recovery"] = ""
        stats["drawdown_start_index"] = (
            period_offset + peak_position
            if peak_position >= 0
            else previous_peak_index
        )
        stats["drawdown_recovery_index"] = None
        recovery_positions = np.flatnonzero(
            cumulative[drawdown_position + 1 :] >= peak_value
        )
        if len(recovery_positions):
            recovery_position = drawdown_position + 1 + int(recovery_positions[0])
            stats["drawdown_recovery"] = _timestamp_iso(
                timestamps,
                recovery_position,
            )
            stats["drawdown_recovery_index"] = period_offset + recovery_position
    elif (
        previous_max_drawdown < 0
        and previous_recovery_index is None
        and previous_peak_value is not None
    ):
        recovery_positions = np.flatnonzero(cumulative >= float(previous_peak_value))
        if len(recovery_positions):
            recovery_position = int(recovery_positions[0])
            stats["drawdown_recovery"] = _timestamp_iso(
                timestamps,
                recovery_position,
            )
            stats["drawdown_recovery_index"] = period_offset + recovery_position


def _max_true_run(mask: np.ndarray, initial: int) -> int:
    if not bool(mask.any()):
        return 0
    changes = np.diff(mask.astype("int8"))
    starts = np.flatnonzero(np.concatenate(([mask[0]], changes == 1)))
    ends = np.flatnonzero(np.concatenate((changes == -1, [mask[-1]]))) + 1
    lengths = (ends - starts)[mask[starts]]
    if bool(mask[0]):
        lengths[0] += initial
    return int(lengths.max())


def _trailing_true_run(mask: np.ndarray, initial: int) -> int:
    if not bool(mask[-1]):
        return 0
    false_positions = np.flatnonzero(~mask)
    last_false = int(false_positions[-1]) if len(false_positions) else -1
    return len(mask) - last_false - 1 + (initial if last_false < 0 else 0)


def _timestamp_iso(timestamps: np.ndarray, index: int) -> str:
    return pd.Timestamp(timestamps[index]).isoformat()


def _ordered_cumulative(values: np.ndarray, *, initial: float = 0.0) -> np.ndarray:
    return np.cumsum(
        np.concatenate(([initial], np.asarray(values, dtype="float64"))),
        dtype="float64",
    )[1:]


def calculate_additive_cumulative_returns(
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate constant-notional additive return and drawdown series."""

    result = _validate_returns_frame(
        returns,
        segment_columns=segment_columns,
        include_performance_columns=False,
    )
    output_columns = [*segment_columns, *ADDITIVE_CUMULATIVE_RETURN_COLUMNS]
    if result.empty:
        return pd.DataFrame(columns=output_columns)

    sort_columns = [*segment_columns, "signal", "timestamp"]
    if "gross_portfolio_return" not in result:
        result["gross_portfolio_return"] = result["net_portfolio_return"]
    if "cost" not in result:
        result["cost"] = 0.0
    result = result.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    group_columns = [*segment_columns, "signal"]
    cumulative_columns = (
        ("gross_portfolio_return", "additive_cumulative_gross_return"),
        ("net_portfolio_return", "additive_cumulative_net_return"),
        ("cost", "additive_cumulative_cost"),
    )
    for positions in result.groupby(group_columns, sort=False).indices.values():
        for source_column, output_column in cumulative_columns:
            result.loc[positions, output_column] = _ordered_cumulative(
                result.loc[positions, source_column].to_numpy(copy=False)
            )
    result["high_water_mark"] = result.groupby(group_columns, sort=False)[
        "additive_cumulative_net_return"
    ].cummax().clip(lower=0.0)
    result["drawdown"] = (
        result["additive_cumulative_net_return"] - result["high_water_mark"]
    )
    return result[output_columns]


def summarize_performance(
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize frequency-neutral portfolio performance diagnostics."""

    frame = _validate_returns_frame(
        returns,
        segment_columns=segment_columns,
        include_performance_columns=True,
    )
    if frame.empty:
        return _performance_summary_frame([], segment_columns=segment_columns)

    sort_columns = [*segment_columns, "signal", "timestamp"]
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "signal"]
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        group_values = normalize_group_key(group_key, group_columns)
        stats = _new_performance_stats()
        _update_performance_stats_batch(
            stats,
            timestamps=group["timestamp"].to_numpy(copy=False),
            gross_returns=group["gross_portfolio_return"].to_numpy(copy=False),
            net_returns=group["net_portfolio_return"].to_numpy(copy=False),
            costs=group["cost"].to_numpy(copy=False),
            turnovers=group["turnover"].to_numpy(copy=False),
        )
        net_returns = group["net_portfolio_return"]
        var_95 = float(net_returns.quantile(0.05))
        q95 = float(net_returns.quantile(0.95))
        cvar_values = net_returns[net_returns <= var_95]
        record = _performance_summary_record(
            str(group_values["signal"]),
            stats,
            var_95=var_95,
            q95=q95,
            cvar_95=float(cvar_values.mean()),
        )
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                **record,
            }
        )
    return _performance_summary_frame(records, segment_columns=segment_columns)


def _performance_summary_frame(
    records: list[dict[str, object]],
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build a stable nullable-numeric performance summary frame."""

    output_columns = [*segment_columns, *PERFORMANCE_SUMMARY_COLUMNS]
    frame = pd.DataFrame.from_records(records, columns=output_columns)
    for column in _PERFORMANCE_SUMMARY_INTEGER_COLUMNS:
        frame[column] = frame[column].astype("int64")
    string_columns = set(_PERFORMANCE_SUMMARY_STRING_COLUMNS)
    for column in PERFORMANCE_SUMMARY_COLUMNS:
        if column in string_columns or column in _PERFORMANCE_SUMMARY_INTEGER_COLUMNS:
            continue
        frame[column] = frame[column].astype("float64")
    for column in _PERFORMANCE_SUMMARY_STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    return frame


def _performance_summary_record(
    signal: str,
    stats: dict[str, object],
    *,
    var_95: float,
    q95: float,
    cvar_95: float,
) -> dict[str, object]:
    periods = int(stats["periods"])
    mean_net = float(stats["net_sum"]) / periods
    volatility = _population_std(
        float(stats["net_square_sum"]), mean_net, periods
    )
    downside_mean = float(stats["downside_sum"]) / periods
    downside_volatility = _population_std(
        float(stats["downside_square_sum"]), downside_mean, periods
    )
    max_drawdown = float(stats["max_drawdown"])
    drawdown_recovery_index = stats["drawdown_recovery_index"]
    end_index = (
        int(drawdown_recovery_index)
        if drawdown_recovery_index is not None
        else periods - 1
    )
    drawdown_duration = (
        end_index - int(stats["drawdown_start_index"])
        if max_drawdown < 0
        else 0
    )

    sharpe, sharpe_reason = _ratio_result(mean_net, volatility)
    sortino, sortino_reason = _ratio_result(mean_net, downside_volatility)
    calmar, calmar_reason = _ratio_result(
        float(stats["net_sum"]), abs(max_drawdown)
    )
    skew, skew_reason = _moment_result(stats, order=3)
    kurtosis, kurtosis_reason = _moment_result(stats, order=4)
    tail_ratio, tail_ratio_reason = _ratio_result(abs(q95), abs(var_95))
    wins = int(stats["wins"])
    losses = int(stats["losses"])
    average_win = (
        float(stats["win_sum"]) / wins if wins else None
    )
    average_loss = (
        float(stats["loss_sum"]) / losses if losses else None
    )
    average_win_reason = None if wins else "no_positive_returns"
    average_loss_reason = None if losses else "no_negative_returns"
    if not losses:
        profit_factor, profit_factor_reason = None, "no_negative_returns"
    else:
        profit_factor, profit_factor_reason = _ratio_result(
            float(stats["win_sum"]), abs(float(stats["loss_sum"]))
        )
    if not wins:
        payoff_ratio, payoff_ratio_reason = None, "no_positive_returns"
    elif not losses:
        payoff_ratio, payoff_ratio_reason = None, "no_negative_returns"
    else:
        payoff_ratio, payoff_ratio_reason = _ratio_result(
            float(average_win), abs(float(average_loss))
        )
    cost_to_gross, cost_to_gross_reason = _ratio_result(
        float(stats["cost_sum"]), abs(float(stats["gross_sum"]))
    )
    cost_to_net, cost_to_net_reason = _ratio_result(
        float(stats["cost_sum"]), abs(float(stats["net_sum"]))
    )
    return {
        "signal": signal,
        "periods": periods,
        "final_additive_cumulative_gross_return": float(stats["cumulative_gross"]),
        "final_additive_cumulative_net_return": float(stats["cumulative_net"]),
        "mean_net_portfolio_return": mean_net,
        "volatility_net_portfolio_return": volatility,
        "downside_volatility": downside_volatility,
        "sharpe_like": sharpe,
        "sortino_like": sortino,
        "calmar_like": calmar,
        "best_period_return": float(stats["best"]),
        "worst_period_return": float(stats["worst"]),
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "tail_ratio": tail_ratio,
        "positive_return_rate": wins / periods,
        "negative_return_rate": losses / periods,
        "win_rate": wins / periods,
        "loss_rate": losses / periods,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "average_win": average_win,
        "average_loss": average_loss,
        "max_consecutive_wins": int(stats["max_consecutive_wins"]),
        "max_consecutive_losses": int(stats["max_consecutive_losses"]),
        "max_drawdown": max_drawdown,
        "drawdown_duration": drawdown_duration,
        "max_drawdown_start": str(stats["drawdown_start"]) if max_drawdown < 0 else "",
        "max_drawdown_valley": (
            str(stats["drawdown_valley"]) if max_drawdown < 0 else ""
        ),
        "max_drawdown_recovery": (
            str(stats["drawdown_recovery"]) if max_drawdown < 0 else ""
        ),
        "mean_turnover": float(stats["turnover_sum"]) / periods,
        "max_turnover": float(stats["turnover_max"]),
        "total_cost": float(stats["cost_sum"]),
        "mean_cost": float(stats["cost_sum"]) / periods,
        "cost_to_gross_return": cost_to_gross,
        "cost_to_net_return": cost_to_net,
        "sharpe_like_reason": sharpe_reason,
        "sortino_like_reason": sortino_reason,
        "calmar_like_reason": calmar_reason,
        "skew_reason": skew_reason,
        "kurtosis_reason": kurtosis_reason,
        "tail_ratio_reason": tail_ratio_reason,
        "profit_factor_reason": profit_factor_reason,
        "payoff_ratio_reason": payoff_ratio_reason,
        "average_win_reason": average_win_reason,
        "average_loss_reason": average_loss_reason,
        "cost_to_gross_return_reason": cost_to_gross_reason,
        "cost_to_net_return_reason": cost_to_net_reason,
    }


def _ratio_result(
    numerator: float,
    denominator: float,
    *,
    zero_reason: str = "zero_denominator",
) -> tuple[float | None, str | None]:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None, "nonfinite_input"
    if denominator == 0:
        return None, zero_reason
    value = float(numerator / denominator)
    if not np.isfinite(value):
        return None, "nonfinite_result"
    return value, None


def _moment_result(
    stats: dict[str, object],
    *,
    order: int,
) -> tuple[float | None, str | None]:
    count = int(stats["periods"])
    minimum = 3 if order == 3 else 4
    if count < minimum:
        return None, "insufficient_observations"
    mean = float(stats["net_sum"]) / count
    m2 = _central_moment(stats, mean, 2)
    if not np.isfinite(m2):
        return None, "nonfinite_result"
    if m2 <= 0:
        return None, "zero_variance"
    if order == 3:
        value = np.sqrt(count * (count - 1)) / (count - 2) * _central_moment(
            stats, mean, 3
        ) / m2**1.5
    else:
        m4 = _central_moment(stats, mean, 4)
        g2 = m4 / m2**2 - 3.0
        value = ((count - 1) / ((count - 2) * (count - 3))) * (
            (count + 1) * g2 + 6.0
        )
    if not np.isfinite(value):
        return None, "nonfinite_result"
    return float(value), None


def _central_moment(stats: dict[str, object], mean: float, order: int) -> float:
    count = int(stats["periods"])
    if order == 2:
        raw = float(stats["net_square_sum"]) / count
        return raw - 2 * mean * float(stats["net_sum"]) / count + mean**2
    if order == 3:
        return (
            float(stats["net_sum_cubes"]) / count
            - 3 * mean * float(stats["net_square_sum"]) / count
            + 3 * mean**2 * float(stats["net_sum"]) / count
            - mean**3
        )
    return (
        float(stats["net_sum_quads"]) / count
        - 4 * mean * float(stats["net_sum_cubes"]) / count
        + 6 * mean**2 * float(stats["net_square_sum"]) / count
        - 4 * mean**3 * float(stats["net_sum"]) / count
        + mean**4
    )


def _population_std(square_sum: float, mean: float, count: int) -> float:
    variance = max(0.0, square_sum / count - mean * mean)
    return math.sqrt(variance)


def _validate_returns_frame(
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...],
    include_performance_columns: bool,
    required_numeric_columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if required_numeric_columns is None:
        required_numeric_columns = ("net_portfolio_return",)
    fixed_columns = {"timestamp", "signal"}
    required_columns = {*segment_columns, *fixed_columns, *required_numeric_columns}
    if include_performance_columns:
        required_columns.update({"gross_portfolio_return", "turnover", "cost"})
    missing_columns = sorted(required_columns.difference(returns.columns))
    if missing_columns:
        raise ValueError(f"returns is missing columns: {missing_columns}.")
    if len(set(segment_columns)) != len(segment_columns):
        raise ValueError("return segment columns must not contain duplicates.")
    overlap = set(segment_columns).intersection(RETURNS_COLUMNS)
    if overlap:
        raise ValueError(
            "return segment columns must not overlap return columns: "
            + ", ".join(sorted(overlap))
        )
    result = returns.copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="returns contain invalid timestamps."
    )
    result["signal"] = strict_string_series(
        result["signal"], error="returns contain invalid signals.", strip=False
    )
    for column in segment_columns:
        result[column] = strict_hashable_key_series(
            result[column],
            error=f"returns segment column {column!r} contains invalid keys.",
        )
    numeric_columns = list(required_numeric_columns)
    if include_performance_columns:
        numeric_columns.extend(
            ["gross_portfolio_return", "turnover", "cost"]
        )
    else:
        numeric_columns.extend(
            column
            for column in ("gross_portfolio_return", "cost")
            if column in result
        )
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise ValueError(
                f"returns column {column!r} must contain numeric values."
            )
        result[column] = strict_finite_numeric_series(
            result[column],
            error=f"returns column {column!r} must contain finite numeric values.",
        ).astype("float64")
    key_columns = [*segment_columns, "timestamp", "signal"]
    if bool(result.duplicated(key_columns).any()):
        raise ValueError("returns must have unique segment-timestamp-signal rows.")
    return result
