"""Portfolio cumulative return and performance diagnostics."""

from __future__ import annotations

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

CUMULATIVE_RETURN_COLUMNS = [
    "timestamp",
    "signal",
    "gross_portfolio_return",
    "net_portfolio_return",
    "cost",
    "cumulative_gross_return",
    "cumulative_net_return",
    "cumulative_cost",
    "high_water_mark",
    "drawdown",
]
PERFORMANCE_SUMMARY_COLUMNS = [
    "signal",
    "periods",
    "final_cumulative_gross_return",
    "final_cumulative_net_return",
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
]


def calculate_cumulative_returns(
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate cumulative return and drawdown series."""

    result = _validate_returns_frame(
        returns,
        segment_columns=segment_columns,
        include_performance_columns=False,
    )
    output_columns = [*segment_columns, *CUMULATIVE_RETURN_COLUMNS]
    if result.empty:
        return pd.DataFrame(columns=output_columns)

    sort_columns = [*segment_columns, "signal", "timestamp"]
    if "gross_portfolio_return" not in result:
        result["gross_portfolio_return"] = result["net_portfolio_return"]
    if "cost" not in result:
        result["cost"] = 0.0
    result = result.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    group_columns = [*segment_columns, "signal"]
    result["cumulative_gross_return"] = result.groupby(group_columns, sort=False)[
        "gross_portfolio_return"
    ].cumsum()
    result["cumulative_net_return"] = result.groupby(group_columns, sort=False)[
        "net_portfolio_return"
    ].cumsum()
    result["cumulative_cost"] = result.groupby(group_columns, sort=False)[
        "cost"
    ].cumsum()
    result["high_water_mark"] = result.groupby(group_columns, sort=False)[
        "cumulative_net_return"
    ].cummax()
    result["drawdown"] = result["cumulative_net_return"] - result["high_water_mark"]
    return result[output_columns]


def summarize_performance(
    returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize frequency-neutral portfolio performance diagnostics."""

    frame = _validate_returns_frame(
        returns,
        segment_columns=segment_columns,
        include_performance_columns=True,
    )
    output_columns = [*segment_columns, *PERFORMANCE_SUMMARY_COLUMNS]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)

    sort_columns = [*segment_columns, "signal", "timestamp"]
    frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    cumulative_frame = cumulative_returns.copy()
    records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "signal"]
    for group_key, group in frame.groupby(group_columns, sort=True, dropna=False):
        group_values = normalize_group_key(group_key, group_columns)
        filters = cumulative_frame["signal"] == group_values["signal"]
        for segment in segment_columns:
            filters &= cumulative_frame[segment] == group_values[segment]

        net_returns = group["net_portfolio_return"]
        gross_returns = group["gross_portfolio_return"]
        costs = group["cost"]
        turnover = group["turnover"]
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        volatility = float(net_returns.std(ddof=0))
        downside_returns = net_returns.clip(upper=0)
        downside_volatility = float(downside_returns.std(ddof=0))
        total_gross = float(gross_returns.sum())
        total_net = float(net_returns.sum())
        total_cost = float(costs.sum())
        drawdown_stats = _max_drawdown_stats(cumulative_frame.loc[filters])
        var_95 = float(net_returns.quantile(0.05))
        q95 = float(net_returns.quantile(0.95))
        cvar_values = net_returns[net_returns <= var_95]
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "signal": group_values["signal"],
                "periods": int(len(group)),
                "final_cumulative_gross_return": total_gross,
                "final_cumulative_net_return": total_net,
                "mean_net_portfolio_return": float(net_returns.mean()),
                "volatility_net_portfolio_return": volatility,
                "downside_volatility": downside_volatility,
                "sharpe_like": _safe_ratio(float(net_returns.mean()), volatility),
                "sortino_like": _safe_ratio(
                    float(net_returns.mean()),
                    downside_volatility,
                ),
                "calmar_like": _safe_ratio(
                    total_net,
                    abs(float(drawdown_stats["max_drawdown"])),
                ),
                "best_period_return": float(net_returns.max()),
                "worst_period_return": float(net_returns.min()),
                "skew": _finite_stat(net_returns.skew()),
                "kurtosis": _finite_stat(net_returns.kurt()),
                "var_95": var_95,
                "cvar_95": float(cvar_values.mean()) if len(cvar_values) else 0.0,
                "tail_ratio": _safe_ratio(abs(q95), abs(var_95)),
                "positive_return_rate": float((net_returns > 0).mean()),
                "negative_return_rate": float((net_returns < 0).mean()),
                "win_rate": float((net_returns > 0).mean()),
                "loss_rate": float((net_returns < 0).mean()),
                "profit_factor": _safe_ratio(
                    float(wins.sum()),
                    abs(float(losses.sum())),
                ),
                "payoff_ratio": _safe_ratio(
                    float(wins.mean()) if len(wins) else 0.0,
                    abs(float(losses.mean())) if len(losses) else 0.0,
                ),
                "average_win": float(wins.mean()) if len(wins) else 0.0,
                "average_loss": float(losses.mean()) if len(losses) else 0.0,
                "max_consecutive_wins": _max_consecutive(net_returns > 0),
                "max_consecutive_losses": _max_consecutive(net_returns < 0),
                **drawdown_stats,
                "mean_turnover": float(turnover.mean()),
                "max_turnover": float(turnover.max()),
                "total_cost": total_cost,
                "mean_cost": float(costs.mean()),
                "cost_to_gross_return": _safe_ratio(total_cost, abs(total_gross)),
                "cost_to_net_return": _safe_ratio(total_cost, abs(total_net)),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def _max_drawdown_stats(cumulative_returns: pd.DataFrame) -> dict[str, object]:
    if cumulative_returns.empty:
        return {
            "max_drawdown": 0.0,
            "drawdown_duration": 0,
            "max_drawdown_start": "",
            "max_drawdown_valley": "",
            "max_drawdown_recovery": "",
        }

    frame = cumulative_returns.copy()
    frame["timestamp"] = strict_timestamp_series(
        frame["timestamp"], error="cumulative returns contain invalid timestamps."
    )
    frame["cumulative_net_return"] = strict_finite_numeric_series(
        frame["cumulative_net_return"],
        error="cumulative returns contain invalid net values.",
    ).astype("float64")
    frame["high_water_mark"] = strict_finite_numeric_series(
        frame["high_water_mark"],
        error="cumulative returns contain invalid high-water values.",
    ).astype("float64")
    frame["drawdown"] = strict_finite_numeric_series(
        frame["drawdown"],
        error="cumulative returns contain invalid drawdown values.",
    ).astype("float64")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    max_drawdown = float(frame["drawdown"].min())
    if max_drawdown >= 0:
        return {
            "max_drawdown": max_drawdown,
            "drawdown_duration": 0,
            "max_drawdown_start": "",
            "max_drawdown_valley": "",
            "max_drawdown_recovery": "",
        }

    valley_idx = int(frame["drawdown"].idxmin())
    valley = frame.loc[valley_idx]
    peak_value = float(valley["high_water_mark"])
    before_valley = frame.loc[:valley_idx]
    peak_rows = before_valley[before_valley["cumulative_net_return"] == peak_value]
    start_idx = int(peak_rows.index[-1]) if not peak_rows.empty else 0
    after_valley = frame.loc[valley_idx + 1 :]
    recovery_rows = after_valley[after_valley["cumulative_net_return"] >= peak_value]
    recovery_idx = int(recovery_rows.index[0]) if not recovery_rows.empty else None
    end_idx = recovery_idx if recovery_idx is not None else int(frame.index[-1])
    return {
        "max_drawdown": max_drawdown,
        "drawdown_duration": int(end_idx - start_idx),
        "max_drawdown_start": _timestamp_text(frame.loc[start_idx, "timestamp"]),
        "max_drawdown_valley": _timestamp_text(valley["timestamp"]),
        "max_drawdown_recovery": (
            _timestamp_text(frame.loc[recovery_idx, "timestamp"])
            if recovery_idx is not None
            else ""
        ),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return 0.0
    return float(numerator / denominator)


def _finite_stat(value: object) -> float:
    result = float(value)
    return result if np.isfinite(result) else 0.0


def _max_consecutive(mask: pd.Series) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    return timestamp.isoformat()


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
