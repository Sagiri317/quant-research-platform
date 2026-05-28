from __future__ import annotations

from quant_research_platform.artifacts.artifact_inventory import index_evaluation_sets
from tests.interfaces.api.artifact_run_fixtures import write_evaluation_set


def test_index_evaluation_sets_returns_future_api_run_list(tmp_path):
    write_evaluation_set(
        tmp_path / "run_b",
        run_id="run_b",
        project="quant-research-platform",
        domain="a_share",
        net_returns=[0.02, 0.03],
        turnover=[1.0, 2.0],
        cost=[0.001, 0.002],
        n_rows_evaluated=24,
    )
    write_evaluation_set(
        tmp_path / "run_a",
        run_id="run_a",
        project="crypto-perp-alpha-lab",
        domain="crypto_perp",
        net_returns=[0.01, 0.02],
        turnover=[2.0, 4.0],
        cost=[0.0, 0.001],
        n_rows_evaluated=12,
    )
    (tmp_path / "notes.md").write_text("not a run", encoding="utf-8")

    index = index_evaluation_sets(tmp_path)

    assert index["run_id"].tolist() == ["run_a", "run_b"]
    assert "evaluation_id" not in index.columns
    assert index["project"].tolist() == [
        "crypto-perp-alpha-lab",
        "quant-research-platform",
    ]
    assert index["domain"].tolist() == ["crypto_perp", "a_share"]
    assert index["data_stage"].tolist() == ["synthetic_only", "synthetic_only"]
    assert index["uses_external_data"].tolist() == [False, False]
    assert index["n_rows_evaluated"].tolist() == [12, 24]
    assert index["weight_method"].tolist() == [
        "equal_weight_quantile",
        "equal_weight_quantile",
    ]
    assert index["mean_final_return"].round(6).tolist() == [0.03, 0.05]
    assert index["mean_turnover"].round(6).tolist() == [3.0, 1.5]
