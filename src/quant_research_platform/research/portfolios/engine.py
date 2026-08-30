"""Portfolio construction engine for SignalSpec-driven research panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_returns import (
    _stream_weighted_portfolio_from_validated_panel,
)
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
    diagnostics: pd.DataFrame
    capacity_diagnostics: pd.DataFrame
    numeric_exposure_diagnostics: pd.DataFrame
    categorical_exposure_diagnostics: pd.DataFrame


@dataclass
class _PortfolioTemporalState:
    lag_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = field(
        default_factory=dict
    )


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
        diagnostics=diagnostics,
        capacity_diagnostics=capacity_diagnostics,
        numeric_exposure_diagnostics=numeric_exposure_diagnostics,
        categorical_exposure_diagnostics=categorical_exposure_diagnostics,
    )


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
