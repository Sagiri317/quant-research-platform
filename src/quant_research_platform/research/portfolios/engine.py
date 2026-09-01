"""Portfolio construction engine for SignalSpec-driven research panels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXECUTION_COLUMNS,
    NAV_RETURN_COLUMNS,
    REALIZED_HOLDINGS_CASH_COLUMNS,
    TARGET_WEIGHT_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_returns import (
    _stream_weighted_portfolio_from_validated_panel,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import (
    EXECUTION_RETURN_COLUMN,
    REBALANCE_SCHEDULES,
    PortfolioSpec,
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
    target_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    executions: pd.DataFrame = field(default_factory=pd.DataFrame)
    realized_holdings_cash: pd.DataFrame = field(default_factory=pd.DataFrame)
    nav_returns: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class _PortfolioTemporalState:
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    cash_weights: dict[tuple[object, ...], dict[str, float]] = field(
        default_factory=dict
    )
    previous_returns: dict[tuple[object, ...], dict[str, dict[str, float]]] = field(
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
    if spec.rebalance_schedule not in REBALANCE_SCHEDULES:
        raise ValueError("unsupported_portfolio_rebalance_schedule")
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
        label_column=EXECUTION_RETURN_COLUMN,
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
            economic_return_column=EXECUTION_RETURN_COLUMN,
            previous_positions=temporal_state.previous_positions,
            cash_weights=temporal_state.cash_weights,
            previous_returns=temporal_state.previous_returns,
            rebalance_schedule=spec.rebalance_schedule,
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
    target_weights, executions, realized_holdings_cash = _accounting_frames(
        positions,
        turnover,
    )
    nav_returns = _nav_return_frame(portfolio_returns)
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
        target_weights=target_weights,
        executions=executions,
        realized_holdings_cash=realized_holdings_cash,
        nav_returns=nav_returns,
    )


def _accounting_frames(
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = positions.rename(columns={"position": "target_weight"}).loc[
        :, TARGET_WEIGHT_COLUMNS
    ]
    if positions.empty:
        empty = pd.DataFrame(columns=list(EXECUTION_COLUMNS))
        holdings = pd.DataFrame(columns=list(REALIZED_HOLDINGS_CASH_COLUMNS))
        return target, empty, holdings
    sorted_positions = positions.sort_values(
        ["signal", "timestamp", "asset"], kind="mergesort"
    )
    turnover_lookup = {
        (row.signal, pd.Timestamp(row.timestamp)): float(row.turnover)
        for row in turnover.itertuples(index=False)
    }
    execution_rows: list[dict[str, object]] = []
    holdings_rows: list[dict[str, object]] = []
    previous_by_signal: dict[str, dict[str, float]] = {}
    for (signal, timestamp), group in sorted_positions.groupby(
        ["signal", "timestamp"], sort=True
    ):
        previous = previous_by_signal.get(str(signal), {})
        current = dict(zip(group["asset"], group["position"], strict=True))
        assets = sorted(set(previous).union(current))
        is_trade = turnover_lookup.get((signal, pd.Timestamp(timestamp)), 0.0) > 0.0
        cash_weight = 1.0 - sum(float(value) for value in current.values())
        for asset in assets:
            pre_trade = float(previous.get(asset, 0.0))
            realized = float(current.get(asset, 0.0))
            delta = realized - pre_trade
            execution_rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "signal": signal,
                    "side": (
                        "buy"
                        if is_trade and delta > 0.0
                        else "sell"
                        if is_trade and delta < 0.0
                        else "none"
                    ),
                    "pre_trade_weight": pre_trade,
                    "target_weight": realized,
                    "realized_weight": realized,
                    "blocked": False,
                }
            )
            holdings_rows.append(
                {
                    "timestamp": timestamp,
                    "asset": asset,
                    "signal": signal,
                    "realized_weight": realized,
                    "cash_weight": cash_weight,
                }
            )
        holdings_rows.append(
            {
                "timestamp": timestamp,
                "asset": "__cash__",
                "signal": signal,
                "realized_weight": cash_weight,
                "cash_weight": cash_weight,
            }
        )
        previous = current
        previous_by_signal[str(signal)] = previous
    executions = pd.DataFrame.from_records(execution_rows, columns=EXECUTION_COLUMNS)
    holdings = pd.DataFrame.from_records(
        holdings_rows,
        columns=REALIZED_HOLDINGS_CASH_COLUMNS,
    )
    return target, executions, holdings


def _nav_return_frame(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    if portfolio_returns.empty:
        return pd.DataFrame(columns=NAV_RETURN_COLUMNS)
    result = portfolio_returns.loc[
        :,
        (
            "timestamp",
            "signal",
            "gross_portfolio_return",
            "net_portfolio_return",
            "cost",
            "turnover",
        ),
    ].copy()
    result["nav"] = (
        result.groupby("signal", sort=False)["net_portfolio_return"]
        .transform(lambda values: (1.0 + values).cumprod())
        .astype("float64")
    )
    result["cumulative_cost"] = result.groupby("signal", sort=False)["cost"].cumsum()
    return result.loc[:, NAV_RETURN_COLUMNS]


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
        EXECUTION_RETURN_COLUMN,
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
    return panel
