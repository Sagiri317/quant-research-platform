from __future__ import annotations

import pytest

from quant_research_platform.artifacts.comparison import compare_evaluation_sets
from tests.api.helpers import write_evaluation_set


def test_compare_evaluation_sets_summarizes_multiple_output_directories(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    write_evaluation_set(
        run_a,
        evaluation_id="run_a",
        ic_values=[0.2, 0.2],
        rankic_values=[0.3, 0.3],
        net_returns=[0.01, 0.02],
        turnover=[2.0, 4.0],
        cost=[0.001, 0.002],
        n_rows_evaluated=20,
    )
    write_evaluation_set(
        run_b,
        evaluation_id="run_b",
        ic_values=[0.4, 0.4],
        rankic_values=[0.5, 0.5],
        net_returns=[0.03, 0.04],
        turnover=[1.0, 3.0],
        cost=[0.0, 0.001],
        n_rows_evaluated=20,
    )

    comparison = compare_evaluation_sets([run_a, run_b])

    assert comparison["evaluation_id"].tolist() == ["run_a", "run_b"]
    assert comparison["mean_ic"].round(6).tolist() == [0.2, 0.4]
    assert comparison["mean_rankic"].round(6).tolist() == [0.3, 0.5]
    assert comparison["mean_final_return"].round(6).tolist() == [0.03, 0.07]
    assert comparison["mean_turnover"].round(6).tolist() == [3.0, 2.0]
    assert comparison["total_cost"].round(6).tolist() == [0.003, 0.001]
    assert comparison["rank_mean_ic"].tolist() == [2, 1]
    assert comparison["rank_mean_rankic"].tolist() == [2, 1]
    assert comparison["rank_mean_final_return"].tolist() == [2, 1]
    assert comparison["rank_mean_turnover"].tolist() == [2, 1]
    assert comparison["rank_total_cost"].tolist() == [2, 1]
    assert "overall_score" not in comparison.columns


def test_compare_evaluation_sets_requires_at_least_one_run_directory():
    with pytest.raises(ValueError, match="at least one run directory"):
        compare_evaluation_sets([])
