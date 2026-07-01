"""Portfolio construction engine for SignalSpec-driven research panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.exposure import summarize_exposure
from quant_research_platform.compute.dataframe.performance import (
    calculate_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    simulate_weighted_long_short_returns_and_cost_from_validated_panel,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import (
    SUPPORTED_REBALANCE_FREQUENCIES,
    PortfolioSpec,
    normalize_rebalance_frequency,
)
from quant_research_platform.research.portfolios.diagnostics import (
    build_capacity_diagnostics,
    build_categorical_exposure_diagnostics,
    build_numeric_exposure_diagnostics,
)
from quant_research_platform.research.signals.materialization import (
    materialize_signal_frame,
)


@dataclass(frozen=True)
class PortfolioComputationResult:
    spec: PortfolioSpec
    positions: pd.DataFrame
    turnover: pd.DataFrame
    cost_breakdown: pd.DataFrame
    portfolio_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    performance_summary: pd.DataFrame
    exposure_summary: pd.DataFrame
    concentration: pd.DataFrame
    period_breakdown: pd.DataFrame
    diagnostics: pd.DataFrame
    capacity_diagnostics: pd.DataFrame
    numeric_exposure_diagnostics: pd.DataFrame
    categorical_exposure_diagnostics: pd.DataFrame


def construct_portfolio_positions(
    frame: pd.DataFrame,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> PortfolioComputationResult:
    """Construct portfolio diagnostics from normalized SignalSpec scores."""

    if (
        normalize_rebalance_frequency(spec.rebalance_frequency)
        not in SUPPORTED_REBALANCE_FREQUENCIES
    ):
        raise ValueError("unsupported_portfolio_rebalance_frequency")
    panel = _portfolio_panel(
        frame,
        spec,
        source_columns_by_kind=source_columns_by_kind,
    )
    signal_ids = tuple(signal.signal_id for signal in spec.signals)
    compute_spec = ComputePanelSpec(
        signal_columns=signal_ids,
        label_column=spec.return_source_column,
        timestamp_column="timestamp",
        asset_column="asset_id",
    )
    positions, turnover, portfolio_returns, diagnostics, cost_breakdown = (
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            compute_spec,
            spec.weight_config,
            portfolio_signals=signal_ids,
            tradability_config=spec.tradability_config,
            cost_config=spec.cost_config,
            return_source_column=spec.return_source_column,
        )
    )
    cumulative_returns = compute_cumulative_returns(portfolio_returns)
    performance_summary = compute_performance_summary(
        portfolio_returns,
        cumulative_returns,
    )
    exposure_summary = compute_exposure_summary(positions, portfolio_returns)
    concentration = summarize_concentration(positions)
    period_breakdown = summarize_period_breakdown(portfolio_returns)
    capacity_diagnostics = build_capacity_diagnostics(
        positions,
        panel,
        spec.diagnostics_config,
    )
    numeric_exposure_diagnostics = build_numeric_exposure_diagnostics(
        positions,
        panel,
        spec.diagnostics_config,
    )
    categorical_exposure_diagnostics = build_categorical_exposure_diagnostics(
        positions,
        panel,
        spec.diagnostics_config,
    )
    return PortfolioComputationResult(
        spec=spec,
        positions=positions,
        turnover=turnover,
        cost_breakdown=cost_breakdown,
        portfolio_returns=portfolio_returns,
        cumulative_returns=cumulative_returns,
        performance_summary=performance_summary,
        exposure_summary=exposure_summary,
        concentration=concentration,
        period_breakdown=period_breakdown,
        diagnostics=diagnostics,
        capacity_diagnostics=capacity_diagnostics,
        numeric_exposure_diagnostics=numeric_exposure_diagnostics,
        categorical_exposure_diagnostics=categorical_exposure_diagnostics,
    )


def compute_portfolio_turnover(result: PortfolioComputationResult) -> pd.DataFrame:
    return result.turnover.copy()


def compute_portfolio_costs(result: PortfolioComputationResult) -> pd.DataFrame:
    return result.cost_breakdown.copy()


def compute_portfolio_returns(result: PortfolioComputationResult) -> pd.DataFrame:
    return result.portfolio_returns.copy()


def compute_cumulative_returns(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    return calculate_cumulative_returns(portfolio_returns)


def compute_drawdowns(cumulative_returns: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in ("timestamp", "signal", "high_water_mark", "drawdown")
        if column in cumulative_returns.columns
    ]
    return cumulative_returns.loc[:, columns].copy()


def compute_performance_summary(
    portfolio_returns: pd.DataFrame,
    cumulative_returns: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_performance(portfolio_returns, cumulative_returns)


def compute_exposure_summary(
    positions: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    return summarize_exposure(positions, portfolio_returns)


def summarize_concentration(
    positions: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    output_columns = [
        *segment_columns,
        "signal",
        "periods",
        "mean_hhi",
        "max_hhi",
        "mean_top_weight",
        "max_top_weight",
        "mean_position_count",
    ]
    if positions.empty:
        return pd.DataFrame(columns=output_columns)

    frame = positions.copy()
    frame["abs_position"] = pd.to_numeric(frame["position"], errors="coerce").abs()
    period_columns = [*segment_columns, "signal", "timestamp"]
    totals = frame.groupby(period_columns, sort=True)["abs_position"].transform("sum")
    frame["abs_weight_share"] = frame["abs_position"].div(totals.replace(0, np.nan))
    period = (
        frame.groupby(period_columns, sort=True)
        .agg(
            hhi=("abs_weight_share", lambda values: float(np.square(values).sum())),
            top_weight=("abs_weight_share", "max"),
            position_count=("abs_position", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "signal"]
    for group_key, group in period.groupby(group_columns, sort=True, dropna=False):
        group_values = normalize_group_key(group_key, group_columns)
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "signal": group_values["signal"],
                "periods": int(len(group)),
                "mean_hhi": float(group["hhi"].mean()),
                "max_hhi": float(group["hhi"].max()),
                "mean_top_weight": float(group["top_weight"].mean()),
                "max_top_weight": float(group["top_weight"].max()),
                "mean_position_count": float(group["position_count"].mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def summarize_period_breakdown(
    portfolio_returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    output_columns = [
        *segment_columns,
        "signal",
        "year",
        "month",
        "periods",
        "gross_portfolio_return",
        "net_portfolio_return",
        "cost",
        "turnover",
        "hit_rate",
    ]
    if portfolio_returns.empty:
        return pd.DataFrame(columns=output_columns)

    frame = portfolio_returns.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["year"] = timestamps.dt.year.astype("int64")
    frame["month"] = timestamps.dt.month.astype("int64")
    group_columns = [*segment_columns, "signal", "year", "month"]
    result = (
        frame.groupby(group_columns, sort=True, dropna=False)
        .agg(
            periods=("net_portfolio_return", "size"),
            gross_portfolio_return=("gross_portfolio_return", "sum"),
            net_portfolio_return=("net_portfolio_return", "sum"),
            cost=("cost", "sum"),
            turnover=("turnover", "mean"),
            hit_rate=(
                "net_portfolio_return",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index()
    )
    return result.loc[:, output_columns]


def _portfolio_panel(
    frame: pd.DataFrame,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    optional_columns = (
        spec.weight_config.risk_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
        spec.diagnostics_config.capacity_amount_column,
        spec.diagnostics_config.capacity_float_market_cap_column,
        *spec.diagnostics_config.numeric_exposure_columns,
        *spec.diagnostics_config.categorical_exposure_columns,
    )
    required = {
        "asset_id",
        "timestamp",
        spec.return_source_column,
        *(signal.source_column for signal in spec.signals),
        *(column for column in optional_columns if column is not None),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_portfolio_columns: {', '.join(missing)}")
    panel = materialize_signal_frame(
        frame,
        spec.signals,
        source_columns_by_kind=source_columns_by_kind,
    )
    if spec.execution_lag_steps:
        signal_ids = [signal.signal_id for signal in spec.signals]
        ordered = panel.sort_values(["asset_id", "timestamp"])
        panel.loc[ordered.index, signal_ids] = (
            ordered.groupby("asset_id", sort=False)[signal_ids]
            .shift(spec.execution_lag_steps)
            .to_numpy()
        )
    return panel


def normalize_group_key(
    group_key: object,
    group_columns: list[str],
) -> dict[str, object]:
    if len(group_columns) == 1:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
    else:
        values = tuple(group_key)
    return dict(zip(group_columns, values, strict=True))


__all__ = [
    "PortfolioComputationResult",
    "construct_portfolio_positions",
    "compute_portfolio_turnover",
    "compute_portfolio_costs",
    "compute_portfolio_returns",
    "compute_cumulative_returns",
    "compute_drawdowns",
    "compute_performance_summary",
    "compute_exposure_summary",
    "summarize_concentration",
    "summarize_period_breakdown",
]
