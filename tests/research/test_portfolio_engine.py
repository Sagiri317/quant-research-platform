from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.portfolios as portfolios_package
import quant_research_platform.research.portfolios.runner as portfolio_runner
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    PortfolioSpec,
    WeightConfig,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios import (
    construct_portfolio_positions,
    run_portfolio_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_research_portfolio_package_exports_final_names() -> None:
    assert not hasattr(portfolios_package, "construct_portfolio_frame")
    for name in (
        "construct_portfolio_positions",
        "compute_portfolio_turnover",
        "compute_portfolio_costs",
        "compute_portfolio_returns",
        "compute_cumulative_returns",
        "compute_drawdowns",
        "compute_performance_summary",
        "compute_exposure_summary",
    ):
        assert hasattr(portfolios_package, name)


def test_portfolio_engine_produces_required_signal_artifacts() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-02")] * 4
            + [pd.Timestamp("2024-02-02")] * 4,
            "raw_risk": [4.0, 3.0, 2.0, 1.0, 4.0, 2.5, 2.0, 1.0],
            "forward_return": [0.00, 0.01, 0.02, 0.04, -0.01, 0.01, 0.02, 0.05],
            "eligible": [True] * 8,
        }
    )
    spec = PortfolioSpec(
        portfolio_id="risk_signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="risk_signal",
                source_column="raw_risk",
                source_kind="factor",
                direction="lower_is_better",
                transform="rank",
            ),
        ),
        return_source_column="forward_return",
        frequency="1d",
        execution_lag_steps=0,
        weight_config=WeightConfig(
            method="rank_weighted_quantile",
            top_quantile=0.25,
            bottom_quantile=0.25,
        ),
        cost_config=CostConfig(slippage_bps=10.0),
    )

    result = construct_portfolio_positions(frame, spec)

    assert set(result.positions["signal"]) == {"risk_signal"}
    assert result.turnover["turnover"].tolist() == pytest.approx([2.0, 0.0])
    assert result.cost_breakdown["total_cost"].tolist() == pytest.approx([0.002, 0.0])
    assert result.portfolio_returns["net_portfolio_return"].tolist() == pytest.approx(
        [0.038, 0.06]
    )
    assert result.cumulative_returns["cumulative_net_return"].tolist() == pytest.approx(
        [0.038, 0.098]
    )
    assert result.performance_summary.loc[0, "max_drawdown"] == pytest.approx(0.0)
    assert result.performance_summary.loc[0, "win_rate"] == pytest.approx(1.0)
    assert result.exposure_summary.loc[0, "mean_gross_exposure"] == pytest.approx(2.0)
    assert result.concentration.loc[0, "mean_hhi"] == pytest.approx(0.5)
    assert result.period_breakdown["year"].tolist() == [2024, 2024]
    assert result.period_breakdown["month"].tolist() == [1, 2]
    assert result.period_breakdown["hit_rate"].tolist() == pytest.approx([1.0, 1.0])
    assert result.diagnostics["method"].tolist() == [
        "rank_weighted_quantile",
        "rank_weighted_quantile",
    ]


def test_portfolio_engine_keeps_two_signal_inputs_separate() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 4,
            "raw_a": [1.0, 2.0, 3.0, 4.0],
            "raw_b": [4.0, 3.0, 2.0, 1.0],
            "forward_return": [0.00, 0.01, 0.02, 0.04],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="two_signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="raw_a",
                source_column="raw_a",
                source_kind="factor",
                transform="rank",
            ),
            SignalSpec(
                signal_id="ranked_raw_b",
                source_column="raw_b",
                source_kind="factor",
                transform="rank",
            ),
        ),
        return_source_column="forward_return",
        frequency="1d",
        execution_lag_steps=0,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
    )

    result = construct_portfolio_positions(frame, spec)

    assert set(result.positions["signal"]) == {"raw_a", "ranked_raw_b"}
    assert set(result.portfolio_returns["signal"]) == {"raw_a", "ranked_raw_b"}
    assert result.performance_summary["periods"].tolist() == [1, 1]


def test_portfolio_engine_applies_execution_lag_steps_to_signal_scores() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-02")] * 4
            + [pd.Timestamp("2024-01-03")] * 4,
            "raw_signal": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0, 0.00, 0.01, 0.02, 0.06],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="lagged_signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="ranked_signal",
                source_column="raw_signal",
                source_kind="factor",
                transform="rank",
            ),
        ),
        return_source_column="forward_return",
        frequency="1d",
        execution_lag_steps=1,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
    )

    result = construct_portfolio_positions(frame, spec)

    assert result.portfolio_returns["timestamp"].tolist() == [
        pd.Timestamp("2024-01-03")
    ]
    assert result.portfolio_returns["gross_portfolio_return"].tolist() == pytest.approx(
        [0.06]
    )


def test_portfolio_spec_accepts_non_empty_frequency_without_engine_claim() -> None:
    spec = PortfolioSpec(
        portfolio_id="weekly_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="ranked_signal",
                source_column="raw_signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        frequency="weekly",
        execution_lag_steps=0,
    )

    assert spec.frequency == "weekly"


def test_portfolio_engine_rejects_unsupported_frequency_at_runtime() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 4,
            "raw_signal": [1.0, 2.0, 3.0, 4.0],
            "forward_return": [0.00, 0.01, 0.02, 0.04],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="weekly_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="ranked_signal",
                source_column="raw_signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        frequency="weekly",
        execution_lag_steps=0,
    )

    with pytest.raises(ValueError, match="unsupported_portfolio_frequency"):
        construct_portfolio_positions(frame, spec)


def test_portfolio_runner_writes_private_report_artifacts(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(dataset_root, _portfolio_frame(), spec=_dataset_spec())

    run_root = run_portfolio_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="risk_signal_portfolio",
        time_window=TimeWindow("2024-01-02", "2024-02-02"),
    )

    assert run_root.as_posix().endswith(
        "reports/private_real/a_share/portfolio/risk_signal_portfolio"
    )
    for filename in (
        "positions.parquet",
        "turnover.parquet",
        "cost_breakdown.parquet",
        "portfolio_returns.parquet",
        "cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "manifest.json",
    ):
        assert (run_root / filename).is_file()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frequency"] == "1d"
    assert manifest["execution_lag_steps"] == 0
    returns = pd.read_parquet(run_root / "portfolio_returns.parquet")
    assert "return_source_column" in returns.columns
    assert "label" not in returns.columns


def test_portfolio_runner_rejects_empty_time_window_before_artifact_write(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(dataset_root, _portfolio_frame(), spec=_dataset_spec())

    with pytest.raises(ValueError, match="empty_portfolio_input"):
        run_portfolio_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="risk_signal_portfolio",
            time_window=TimeWindow("2023-01-01", "2023-12-31"),
        )
    run_root = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "portfolio"
        / "risk_signal_portfolio"
    )
    assert not run_root.exists()
    assert not list(run_root.parent.glob(".tmp-*"))


def test_portfolio_runner_chunked_output_matches_single_chunk_path(
    monkeypatch,
    tmp_path,
) -> None:
    normal_root = _run_demo_portfolio(tmp_path / "normal")
    original_input_chunks = portfolio_runner._portfolio_input_chunks

    def timestamp_input_chunks(**kwargs):  # noqa: ANN003
        for chunk in original_input_chunks(**kwargs):
            timestamps = pd.to_datetime(chunk["timestamp"], errors="raise")
            for timestamp in pd.Index(timestamps.unique()).sort_values():
                yield chunk.loc[timestamps == timestamp].copy()

    monkeypatch.setattr(
        portfolio_runner,
        "_portfolio_input_chunks",
        timestamp_input_chunks,
    )
    chunked_root = _run_demo_portfolio(tmp_path / "chunked")

    for filename in (
        "positions.parquet",
        "turnover.parquet",
        "cost_breakdown.parquet",
        "portfolio_returns.parquet",
        "cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
    ):
        pd.testing.assert_frame_equal(
            _sorted_artifact_frame(pd.read_parquet(normal_root / filename)),
            _sorted_artifact_frame(pd.read_parquet(chunked_root / filename)),
            check_dtype=False,
        )


def _portfolio_spec() -> PortfolioSpec:
    return PortfolioSpec(
        portfolio_id="risk_signal_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="risk_signal",
                source_column="raw_risk",
                source_kind="factor",
                direction="lower_is_better",
                transform="rank",
            ),
        ),
        return_source_column="forward_return",
        frequency="1d",
        execution_lag_steps=0,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
    )


def _dataset_spec() -> ResearchDatasetSpec:
    return ResearchDatasetSpec(
        data_id="demo",
        market_id=MarketId("a_share"),
        input_data_ids=("factor", "label"),
        base_input_data_id="factor",
        factor_columns=("raw_risk",),
        label_columns=("forward_return",),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "raw_risk": ColumnRole.FACTOR,
            "forward_return": ColumnRole.LABEL,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )


def _portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-02")] * 4
            + [pd.Timestamp("2024-02-02")] * 4,
            "raw_risk": [4.0, 3.0, 2.0, 1.0, 4.0, 2.5, 2.0, 1.0],
            "forward_return": [0.00, 0.01, 0.02, 0.04, -0.01, 0.01, 0.02, 0.05],
        }
    )


def _run_demo_portfolio(project_root) -> Path:
    layout = StorageLayout(project_root=project_root)
    dataset_catalog = ResearchDatasetCatalog()
    dataset_catalog.register(_dataset_spec())
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(dataset_root, _portfolio_frame(), spec=_dataset_spec())
    return run_portfolio_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="risk_signal_portfolio",
        time_window=TimeWindow("2024-01-02", "2024-02-02"),
    )


def _sorted_artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    sort_columns = [column for column in frame.columns if column != "fallback_reason"]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
