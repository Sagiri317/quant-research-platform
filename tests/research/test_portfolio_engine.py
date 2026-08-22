from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.research.portfolios.runner as portfolio_runner
from quant_research_platform.artifacts._portfolio_outputs import (
    write_portfolio_outputs,
)
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    PortfolioSpec,
    WeightConfig,
)
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.engine import (
    construct_portfolio_positions,
)
from quant_research_platform.research.portfolios.runner import (
    build_portfolio_result_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.research_specs import (
    make_portfolio_research_dataset_spec,
    make_research_dataset_catalog,
    make_signal_spec,
)


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
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
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
    assert result.additive_cumulative_returns[
        "additive_cumulative_net_return"
    ].tolist() == pytest.approx([0.038, 0.098])
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


def test_portfolio_engine_reports_buy_sell_cost_breakdown() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-02")] * 4
            + [pd.Timestamp("2024-01-03")] * 4,
            "raw_signal": [1.0, 2.0, 3.0, 4.0] * 2,
            "forward_return": [0.00, 0.01, 0.02, 0.04] * 2,
        }
    )
    spec = PortfolioSpec(
        portfolio_id="cost_breakdown_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("raw_signal"),),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
        cost_config=CostConfig(
            buy_cost_bps=5.0,
            sell_cost_bps=10.0,
            slippage_bps=1.0,
            spread_bps=2.0,
        ),
    )

    result = construct_portfolio_positions(frame, spec)
    first_cost = result.cost_breakdown.iloc[0]
    second_cost = result.cost_breakdown.iloc[1]

    assert first_cost["buy_turnover"] == pytest.approx(1.0)
    assert first_cost["sell_turnover"] == pytest.approx(1.0)
    assert first_cost["turnover"] == pytest.approx(2.0)
    assert first_cost["buy_cost"] == pytest.approx(0.0005)
    assert first_cost["sell_cost"] == pytest.approx(0.001)
    assert first_cost["slippage_cost"] == pytest.approx(0.0002)
    assert first_cost["spread_cost"] == pytest.approx(0.0004)
    assert first_cost["total_cost"] == pytest.approx(0.0021)
    assert first_cost["cost_model"] == "detailed_bps"
    assert second_cost["turnover"] == pytest.approx(0.0)
    assert second_cost["total_cost"] == pytest.approx(0.0)
    assert result.portfolio_returns.loc[0, "cost"] == pytest.approx(0.0021)


def test_portfolio_engine_long_only_positions_do_not_short_assets() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 4,
            "raw_signal": [1.0, 2.0, 3.0, 4.0],
            "forward_return": [0.00, 0.01, 0.02, 0.04],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="long_only_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("raw_signal"),),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        weight_config=WeightConfig(
            portfolio_mode="long_only",
            top_quantile=0.5,
            bottom_quantile=0.25,
        ),
    )

    result = construct_portfolio_positions(frame, spec)

    assert result.positions["position"].ge(0.0).all()
    assert result.positions["position"].sum() == pytest.approx(1.0)
    assert result.portfolio_returns.loc[0, "portfolio_mode"] == "long_only"
    assert result.portfolio_returns.loc[0, "short_count"] == 0
    assert result.exposure_summary.loc[0, "mean_net_exposure"] == pytest.approx(1.0)


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
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
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
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=1,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
    )

    result = construct_portfolio_positions(frame, spec)

    assert result.portfolio_returns["timestamp"].tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert result.portfolio_returns["gross_portfolio_return"].tolist() == pytest.approx(
        [0.0, 0.06]
    )


@pytest.mark.parametrize("mutation", ["null_asset", "null_timestamp", "duplicate"])
def test_portfolio_engine_rejects_keys_before_execution_lag_groupby(
    mutation: str,
) -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d"],
            "timestamp": [pd.Timestamp("2024-01-02")] * 4,
            "raw_signal": [1.0, 2.0, 3.0, 4.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    if mutation == "null_asset":
        frame.loc[0, "asset_id"] = None
    elif mutation == "null_timestamp":
        frame.loc[0, "timestamp"] = pd.NaT
    else:
        frame.loc[1, "asset_id"] = "a"
    spec = PortfolioSpec(
        portfolio_id="lagged_key_validation_portfolio",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(make_signal_spec("raw_signal"),),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=1,
    )

    with pytest.raises(ValueError):
        construct_portfolio_positions(frame, spec)


def test_portfolio_spec_rejects_unsupported_rebalance_frequency() -> None:
    with pytest.raises(ValueError, match="unsupported_portfolio_rebalance_frequency"):
        PortfolioSpec(
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
            rebalance_frequency="weekly",
            execution_lag_steps=0,
        )


def test_portfolio_runner_writes_private_outputs(tmp_path) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_portfolio_research_dataset_spec()
    )
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(
        dataset_root,
        _portfolio_frame(),
        spec=make_portfolio_research_dataset_spec(),
    )

    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="risk_signal_portfolio",
        time_window=TimeWindow("2024-01-02", "2024-02-02"),
    )
    output_bundle = _write_portfolio_result(layout, result)

    assert output_bundle.as_posix().endswith("outputs/risk_signal_portfolio")
    for filename in (
        "positions.parquet",
        "turnover.parquet",
        "cost_breakdown.parquet",
        "portfolio_returns.parquet",
        "additive_cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "capacity_diagnostics.parquet",
        "numeric_exposure_diagnostics.parquet",
        "categorical_exposure_diagnostics.parquet",
        "manifest.json",
    ):
        assert (output_bundle / filename).is_file()
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["metadata"]["rebalance_frequency"]
        == frequency_ids.DAILY_FREQUENCY_SLUG
    )
    assert manifest["metadata"]["execution_lag_steps"] == 0
    returns = pd.read_parquet(output_bundle / "portfolio_returns.parquet")
    assert "return_source_column" in returns.columns
    assert "label" not in returns.columns


def test_portfolio_runner_rejects_empty_time_window_before_artifact_write(
    tmp_path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_portfolio_research_dataset_spec()
    )
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(
        dataset_root,
        _portfolio_frame(),
        spec=make_portfolio_research_dataset_spec(),
    )

    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="risk_signal_portfolio",
        time_window=TimeWindow("2023-01-01", "2023-12-31"),
    )
    with pytest.raises(ValueError, match="empty_portfolio_input"):
        _write_portfolio_result(layout, result)
    output_bundle = tmp_path / "outputs" / "risk_signal_portfolio"
    assert not output_bundle.exists()
    assert not list(output_bundle.parent.glob(".tmp-*"))


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
        "additive_cumulative_returns.parquet",
        "performance_summary.parquet",
        "exposure_summary.parquet",
        "concentration.parquet",
        "period_breakdown.parquet",
        "diagnostics.parquet",
        "capacity_diagnostics.parquet",
        "numeric_exposure_diagnostics.parquet",
        "categorical_exposure_diagnostics.parquet",
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
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        weight_config=WeightConfig(top_quantile=0.25, bottom_quantile=0.25),
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
    dataset_catalog = make_research_dataset_catalog(
        make_portfolio_research_dataset_spec()
    )
    portfolio_catalog = PortfolioCatalog()
    portfolio_catalog.register(_portfolio_spec())
    dataset_root = layout.research_dataset_root(MarketId("a_share"), "demo")
    write_dataset(
        dataset_root,
        _portfolio_frame(),
        spec=make_portfolio_research_dataset_spec(),
    )
    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="risk_signal_portfolio",
        time_window=TimeWindow("2024-01-02", "2024-02-02"),
    )
    return _write_portfolio_result(layout, result)


def _write_portfolio_result(layout: StorageLayout, result) -> Path:
    with result:
        return write_portfolio_outputs(
            output_chunks=result.output_chunks,
            summary_factory=result.summary_factory,
            additive_cumulative_returns_factory=result.additive_cumulative_returns_factory,
            summary_pre_reserved=True,
            spec=result.spec,
            output_spec=OutputSpec(
                output_id=result.spec.portfolio_id,
                output_definition_id="portfolio_outputs",
                run_id="test_run",
                market_id=result.spec.market_id,
                run_type=result.spec.run_type,
                scope=OutputScope.LOCAL_ONLY,
            ),
            layout=layout,
        )


def _sorted_artifact_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    sort_columns = [column for column in frame.columns if column != "fallback_reason"]
    return frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
