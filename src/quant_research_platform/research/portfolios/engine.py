"""Portfolio construction engine for SignalSpec-driven research panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.exposure import summarize_exposure
from quant_research_platform.compute.dataframe.performance import (
    _validate_returns_frame,
    calculate_additive_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    _stream_weighted_portfolio_from_validated_panel,
)
from quant_research_platform.compute.dataframe.position_math import validate_positions
from quant_research_platform.compute.grouping import normalize_group_key
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import (
    SUPPORTED_REBALANCE_FREQUENCIES,
    PortfolioSpec,
    normalize_rebalance_frequency,
)
from quant_research_platform.contracts.validation import (
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.research.portfolios.diagnostics import (
    build_capacity_diagnostics,
    build_categorical_exposure_diagnostics,
    build_numeric_exposure_diagnostics,
)
from quant_research_platform.research.signals.materialization import (
    normalize_signal_frame,
)
from quant_research_platform.runtime.resources import MemoryLease


@dataclass(frozen=True)
class PortfolioComputationResult:
    spec: PortfolioSpec
    positions: pd.DataFrame
    turnover: pd.DataFrame
    cost_breakdown: pd.DataFrame
    portfolio_returns: pd.DataFrame
    additive_cumulative_returns: pd.DataFrame
    performance_summary: pd.DataFrame
    exposure_summary: pd.DataFrame
    concentration: pd.DataFrame
    period_breakdown: pd.DataFrame
    diagnostics: pd.DataFrame
    capacity_diagnostics: pd.DataFrame
    numeric_exposure_diagnostics: pd.DataFrame
    categorical_exposure_diagnostics: pd.DataFrame


@dataclass
class _PortfolioTemporalState:
    lag_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    previous_positions: dict[
        tuple[object, ...], dict[str, dict[str, float]]
    ] = field(default_factory=dict)


def _construct_portfolio_positions(
    frame: pd.DataFrame,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind: Mapping[str, Sequence[str]] | None,
    temporal_state: _PortfolioTemporalState | None = None,
    resource_lease: MemoryLease | None = None,
) -> PortfolioComputationResult:

    if temporal_state is None:
        temporal_state = _PortfolioTemporalState()
    if (
        normalize_rebalance_frequency(spec.rebalance_frequency)
        not in SUPPORTED_REBALANCE_FREQUENCIES
    ):
        raise ValueError("unsupported_portfolio_rebalance_frequency")
    panel = _portfolio_panel(
        frame,
        spec,
        source_columns_by_kind=source_columns_by_kind,
        temporal_state=temporal_state,
        resource_lease=resource_lease,
    )
    signal_ids = tuple(signal.signal_id for signal in spec.signals)
    compute_spec = ComputePanelSpec(
        signal_columns=signal_ids,
        label_column=spec.return_source_column,
        timestamp_column="timestamp",
        asset_column="asset_id",
    )
    positions, turnover, portfolio_returns, diagnostics, cost_breakdown = (
        _stream_weighted_portfolio_from_validated_panel(
            panel,
            compute_spec,
            spec.weight_config,
            portfolio_signals=signal_ids,
            tradability_config=spec.tradability_config,
            cost_config=spec.cost_config,
            return_source_column=spec.return_source_column,
            previous_positions=temporal_state.previous_positions,
        )
    )
    additive_cumulative_returns = calculate_additive_cumulative_returns(
        portfolio_returns
    )
    performance_summary = summarize_performance(portfolio_returns)
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
        additive_cumulative_returns=additive_cumulative_returns,
        performance_summary=performance_summary,
        exposure_summary=exposure_summary,
        concentration=concentration,
        period_breakdown=period_breakdown,
        diagnostics=diagnostics,
        capacity_diagnostics=capacity_diagnostics,
        numeric_exposure_diagnostics=numeric_exposure_diagnostics,
        categorical_exposure_diagnostics=categorical_exposure_diagnostics,
    )


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

    frame = validate_positions(positions, segment_columns=segment_columns)
    frame["abs_position"] = frame["position"].abs()
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

    frame = _validate_returns_frame(
        portfolio_returns,
        segment_columns=segment_columns,
        include_performance_columns=True,
    )
    timestamps = frame["timestamp"]
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
    temporal_state: _PortfolioTemporalState,
    resource_lease: MemoryLease | None,
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
    timestamps = strict_timestamp_series(
        frame["timestamp"], error="invalid_portfolio_timestamp_values"
    )
    assets = strict_string_series(
        frame["asset_id"], error="invalid_portfolio_asset_values", strip=False
    )
    if pd.MultiIndex.from_arrays([timestamps, assets]).has_duplicates:
        raise ValueError("duplicate_portfolio_timestamp_asset_keys")
    panel = normalize_signal_frame(
        frame,
        spec.signals,
        source_columns_by_kind=source_columns_by_kind,
        resource_lease=resource_lease,
    )
    panel["timestamp"] = timestamps
    panel["asset_id"] = assets
    _apply_execution_lag(panel, spec, temporal_state)
    return panel


def _apply_execution_lag(
    panel: pd.DataFrame,
    spec: PortfolioSpec,
    state: _PortfolioTemporalState,
) -> None:
    lag = int(spec.execution_lag_steps)
    if lag == 0:
        return
    signal_ids = [signal.signal_id for signal in spec.signals]
    columns = ["asset_id", "timestamp", *signal_ids]
    current = panel.loc[:, columns].copy()
    current["_portfolio_row"] = np.arange(len(current), dtype="int64")
    if state.lag_history.empty:
        working = current
    else:
        history = state.lag_history.copy()
        history["_portfolio_row"] = -1
        working = pd.concat([history, current], ignore_index=True)
    ordered = working.sort_values(["asset_id", "timestamp"], kind="mergesort")
    shifted = ordered.groupby("asset_id", sort=False)[signal_ids].shift(lag)
    current_mask = ordered["_portfolio_row"].ge(0).to_numpy()
    row_positions = ordered.loc[current_mask, "_portfolio_row"].to_numpy(
        dtype="int64",
        copy=False,
    )
    output = np.empty((len(panel), len(signal_ids)), dtype="float64")
    output[row_positions] = shifted.loc[current_mask, signal_ids].to_numpy()
    panel.loc[:, signal_ids] = output
    state.lag_history = (
        ordered.groupby("asset_id", sort=False, group_keys=False)
        .tail(lag)
        .loc[:, columns]
        .reset_index(drop=True)
    )
