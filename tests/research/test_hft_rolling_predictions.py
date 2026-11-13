"""Saved rolling forecasts feed one continuous native inventory account."""

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    write_prediction_outputs,
)
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.portfolios.high_frequency import backtest
from tests.helpers.high_frequency import parameters, quotes
from tests.helpers.output_frames import make_price_volume_prediction_output_frame
from tests.research.test_policy_learning import hft_episode


def test_rolling_forecasts_keep_cash_inventory_and_next_day_sellability(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    paths = []
    for day, scores in (
        ("2024-01-02", [1.0, -1.0, -1.0]),
        ("2024-01-03", [-1.0, -1.0, -1.0]),
    ):
        metadata = make_price_volume_prediction_output_frame().attrs[
            "prediction_metadata"
        ]
        frame = quotes(day, assets=("a",))[["timestamp", "asset_id"]].assign(
            prediction_score=scores, prediction_rank_score=0.0
        )
        previous = (pd.Timestamp(day) - pd.Timedelta(days=1)).isoformat()
        metadata.update(
            prediction_window={"start": day, "end": day},
            effective_prediction_end=frame.timestamp.max().isoformat(),
            prediction_rows=len(frame),
        )
        metadata["model_facts"].update(
            fit_windows=[{"start": previous, "end": previous}],
            fit_cutoff=previous,
        )
        metadata["model_metadata"].update(
            train_start=previous, train_end=previous, label_available_max=previous
        )
        paths.append(
            str(
                write_prediction_outputs(
                    output_path=tmp_path / day,
                    output_chunks=[frame],
                    metadata=metadata,
                )
            )
        )
    config = {
        **episode["account"],
        "parameters": parameters(("a",)),
        "window": {"start": "2024-01-02", "end": "2024-01-03"},
        "predictions": paths,
        "output": str(tmp_path / "rolling"),
        "candidate_id": "rolling",
    }
    ledger = read_portfolio_ledger_artifact(backtest(config))
    assert ledger.capital.candidate_id == "rolling"
    batches = list(ledger.iter_chunks())
    trades = [trade for batch in batches for trade in batch.trades]
    assert [trade.quantity for trade in trades] == [200, -100, -200]
    assert pd.Timestamp(trades[-1].timestamp_ns).date().isoformat() == "2024-01-03"
    summary = summarize_portfolio_ledger(ledger.capital, iter(batches))
    # Three fills lose 500 shares times the 0.01 CNY half-spread.
    assert summary["net_pnl"] == pytest.approx(-5.0)
    assert summary["nav"] == pytest.approx(4995.0)
    with pytest.raises(ValueError, match="hft_prediction_keys_overlap"):
        backtest(
            {
                **config,
                "predictions": paths + paths,
                "output": str(tmp_path / "overlap"),
            }
        )
    holding_config = {**config, "predictions": [], "output": str(tmp_path / "holding")}
    del holding_config["candidate_id"]
    held = read_portfolio_ledger_artifact(backtest(holding_config))
    assert held.capital.candidate_id == "hft_inventory"
    statistics = read_statistics_artifact(
        portfolio_statistics(
            {
                "ledgers": [config["output"], holding_config["output"]],
                "window": config["window"],
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    with statistics.table("portfolio_performance.parquet") as performance:
        values = (
            performance.loc[performance.metric_id.eq("net_pnl")]
            .set_index("candidate_id")
            .value
        )
        assert values.to_dict() == pytest.approx(
            {"rolling": -5.0, "hft_inventory": 0.0}
        )
    assert summarize_portfolio_ledger(held.capital, held.iter_chunks())["nav"] == 5000.0
    from quant_research_platform.research.portfolios.inventory_analysis import (
        analyze_inventory,
    )
    from quant_research_platform.storage.datasets import read_dataset

    analysis = read_dataset(
        analyze_inventory(
            {
                "ledger": config["output"],
                "benchmark": str(tmp_path / "holding"),
                "output": str(tmp_path / "analysis"),
            }
        )
    )
    assert analysis.incremental_net_pnl.sum() == pytest.approx(-5.0)
    assert analysis.execution_to_mark_pnl.sum() == pytest.approx(-5.0)
    assert analysis.intraday_inventory_pnl.sum() == pytest.approx(0.0)
    assert analysis.decomposition_residual.abs().max() < 1e-9
    assert analysis.fill_events.sum() == 3
    assert analysis.maximum_fill_wait_seconds.max() > 0
    different_capital = {
        **holding_config,
        "parameters": {**holding_config["parameters"], "initial_capital": 6000.0},
        "output": str(tmp_path / "different-capital"),
    }
    backtest(different_capital)
    with pytest.raises(ValueError, match="inventory_benchmark_scope_mismatch"):
        analyze_inventory(
            {
                "ledger": config["output"],
                "benchmark": different_capital["output"],
                "output": str(tmp_path / "invalid-comparison"),
            }
        )
    assert not (tmp_path / "invalid-comparison").exists()
