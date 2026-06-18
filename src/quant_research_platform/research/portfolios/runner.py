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
from quant_research_platform.research.datasets.io import _iter_research_dataset_chunks
from quant_research_platform.research.portfolios.engine import simulate_portfolio_frame
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
    chunks = _iter_research_dataset_chunks(
        layout,
        market_id=dataset_spec.market_id,
        dataset_id=dataset_spec.data_id,
        time_window=time_window,
        columns=read_columns,
    )
    frame = pd.concat(chunks, ignore_index=True)
    if frame.empty:
        raise ValueError("empty_portfolio_input")
    result = simulate_portfolio_frame(frame, spec)
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
                *(signal.column for signal in spec.signals),
                spec.label_column,
                *(column for column in optional_columns if column is not None),
            )
        )
    )


__all__ = ["run_portfolio_from_research_dataset"]
