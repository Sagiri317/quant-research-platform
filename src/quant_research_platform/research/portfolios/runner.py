"""Spec-driven portfolio runner from ResearchDataset inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.contracts.portfolios import PortfolioSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.reports.report_artifacts import (
    write_portfolio_report_artifacts,
)
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.datasets.io import _iter_research_dataset_chunks
from quant_research_platform.research.portfolios.engine import (
    PortfolioResult,
    compute_cumulative_returns,
    compute_exposure_summary,
    compute_performance_summary,
    construct_portfolio_positions,
    summarize_concentration,
    summarize_period_breakdown,
)
from quant_research_platform.runtime.resources import ChunkAxis
from quant_research_platform.storage.layout import StorageLayout


def run_portfolio_from_research_dataset(
    *,
    layout: StorageLayout,
    dataset_catalog: ResearchDatasetCatalog,
    portfolio_catalog: PortfolioCatalog,
    portfolio_id: str,
    time_window: TimeWindow | None = None,
) -> Path:
    spec = portfolio_catalog.get(portfolio_id)
    dataset_spec = dataset_catalog.get(spec.research_dataset_id)
    read_columns = _portfolio_read_columns(spec)
    result = _construct_portfolio_from_chunks(
        iter_complete_time_sized_frames(
            _iter_research_dataset_chunks(
                layout,
                market_id=dataset_spec.market_id,
                dataset_id=dataset_spec.data_id,
                time_window=time_window,
                columns=read_columns,
            ),
            timestamp_column="timestamp",
            axis=ChunkAxis.TIME,
        ),
        spec,
        source_columns_by_kind={
            "factor": tuple(dataset_spec.factor_columns),
            "feature": tuple(dataset_spec.feature_columns),
            "prediction": tuple(dataset_spec.context_columns),
        },
    )
    report_spec = ReportSpec(
        report_id="portfolio_report",
        report_definition_id="portfolio_report_private",
        run_id=spec.portfolio_id,
        market_id=spec.market_id,
        run_type=spec.run_type,
    )
    return write_portfolio_report_artifacts(
        positions=result.positions,
        turnover=result.turnover,
        cost_breakdown=result.cost_breakdown,
        portfolio_returns=result.portfolio_returns,
        cumulative_returns=result.cumulative_returns,
        performance_summary=result.performance_summary,
        exposure_summary=result.exposure_summary,
        concentration=result.concentration,
        period_breakdown=result.period_breakdown,
        diagnostics=result.diagnostics,
        spec=result.spec,
        report_spec=report_spec,
        layout=layout,
    )


def _portfolio_read_columns(spec: PortfolioSpec) -> tuple[str, ...]:
    optional_columns = (
        spec.weight_config.risk_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
    )
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(signal.source_column for signal in spec.signals),
                spec.return_source_column,
                *(column for column in optional_columns if column is not None),
            )
        )
    )


def _construct_portfolio_from_chunks(
    chunks,
    spec: PortfolioSpec,
    *,
    source_columns_by_kind,
):
    context = pd.DataFrame()
    context_periods = max(1, int(spec.execution_lag_steps) + 1)
    artifact_parts: dict[str, list[pd.DataFrame]] = {
        "positions": [],
        "turnover": [],
        "cost_breakdown": [],
        "portfolio_returns": [],
        "diagnostics": [],
    }
    saw_rows = False

    for chunk in chunks:
        if chunk.empty:
            continue
        saw_rows = True
        actual_timestamps = _timestamp_set(chunk)
        working = _combine_context_and_chunk(context, chunk)
        result = construct_portfolio_positions(
            working,
            spec,
            source_columns_by_kind=source_columns_by_kind,
        )
        artifact_parts["positions"].append(
            _select_actual_timestamps(result.positions, actual_timestamps)
        )
        artifact_parts["turnover"].append(
            _select_actual_timestamps(result.turnover, actual_timestamps)
        )
        artifact_parts["cost_breakdown"].append(
            _select_actual_timestamps(result.cost_breakdown, actual_timestamps)
        )
        artifact_parts["portfolio_returns"].append(
            _select_actual_timestamps(result.portfolio_returns, actual_timestamps)
        )
        artifact_parts["diagnostics"].append(
            _select_actual_timestamps(result.diagnostics, actual_timestamps)
        )
        context = _trailing_timestamp_context(working, periods=context_periods)

    if not saw_rows:
        raise ValueError("empty_portfolio_input")

    positions = _combine_artifact_parts(artifact_parts["positions"])
    turnover = _combine_artifact_parts(artifact_parts["turnover"])
    cost_breakdown = _combine_artifact_parts(artifact_parts["cost_breakdown"])
    portfolio_returns = _combine_artifact_parts(artifact_parts["portfolio_returns"])
    diagnostics = _combine_artifact_parts(artifact_parts["diagnostics"])
    cumulative_returns = compute_cumulative_returns(portfolio_returns)
    performance_summary = compute_performance_summary(
        portfolio_returns,
        cumulative_returns,
    )
    exposure_summary = compute_exposure_summary(positions, portfolio_returns)
    concentration = summarize_concentration(positions)
    period_breakdown = summarize_period_breakdown(portfolio_returns)
    return PortfolioResult(
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
    )


def _combine_context_and_chunk(
    context: pd.DataFrame,
    chunk: pd.DataFrame,
) -> pd.DataFrame:
    if context.empty:
        return chunk
    # Bounded context carries only recent timestamps for lag/turnover continuity.
    return pd.concat([context, chunk], ignore_index=True)


def _timestamp_set(frame: pd.DataFrame) -> set[pd.Timestamp]:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_portfolio_timestamp_values")
    return set(pd.Index(timestamps).unique())


def _select_actual_timestamps(
    frame: pd.DataFrame,
    actual_timestamps: set[pd.Timestamp],
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_portfolio_timestamp_values")
    return frame.loc[timestamps.isin(actual_timestamps)].copy()


def _trailing_timestamp_context(frame: pd.DataFrame, *, periods: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("invalid_portfolio_timestamp_values")
    trailing = pd.Index(timestamps.unique()).sort_values().tolist()[-periods:]
    return frame.loc[timestamps.isin(trailing)].copy()


def _combine_artifact_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0].reset_index(drop=True)
    # Report artifacts are still materialized; research input chunks are not.
    return pd.concat(parts, ignore_index=True)


__all__ = ["run_portfolio_from_research_dataset"]
