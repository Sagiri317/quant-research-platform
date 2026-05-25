from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.evaluation import (
    compare_factor_summaries,
)
from quant_research_platform.markets.implementations.a_share.evaluation.build_factor_summary import (  # noqa: E501
    PERFORMANCE_COLUMNS,
    build_factor_summary_frame,
    main,
    write_factor_summary_csv,
    write_factor_summary_markdown,
)
from tests.interfaces.api.artifact_run_fixtures import write_evaluation_set


def test_build_factor_summary_reads_schema_06_direct_run_root(tmp_path) -> None:
    report_root = tmp_path / "n7_eval"
    write_evaluation_set(
        report_root,
        run_id="n7_eval",
        time_breakdowns=("year",),
        n_rows_evaluated=1234,
    )

    summary = build_factor_summary_frame(report_root)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["run_id"] == "n7_eval"
    assert row["year"] == 2026
    assert row["feature"] == "factor"
    assert row["label"] == "forward_return"
    assert row["kernel"] == "dataframe"
    assert row["portfolio_mode"] == "long_short"
    assert row["n_rows_evaluated"] == 1234


def test_build_factor_summary_discovers_schema_06_child_runs(tmp_path) -> None:
    report_root = tmp_path / "factor_evaluation"
    write_evaluation_set(report_root / "run_a", run_id="run_a")
    write_evaluation_set(report_root / "run_b", run_id="run_b")

    summary = build_factor_summary_frame(report_root)

    assert summary["run_id"].tolist() == ["run_a", "run_b"]
    assert summary["feature"].tolist() == ["factor", "factor"]


def test_build_factor_summary_cli_defaults_to_full_report_root(
    tmp_path,
    monkeypatch,
) -> None:
    report_root = (
        tmp_path
        / "reports"
        / "private_real"
        / "factor_evaluation"
        / "full_2024_2025"
    )
    write_evaluation_set(report_root, run_id="full_2024_2025")
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == 0
    assert (report_root / "factor_summary.csv").exists()


def test_factor_summary_writers_use_stable_outputs_and_excel_safe_csv(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "year": [2025],
            "label": ["forward_return_20d"],
            "feature": ["amihud_20d"],
            "kernel": ["dataframe"],
            "mean_ic": [0.01],
            "icir": [0.5],
            "positive_ic_rate": [0.6],
            "mean_rankic": [0.03],
            "rankicir": [0.75],
            "positive_rankic_rate": [0.65],
            "top_bottom_spread_mean": [0.03],
            "top_bottom_spread_hit_rate": [0.55],
            "group_monotonicity_score": [0.8],
            "portfolio_mode": ["long_only"],
            "mean_net_portfolio_return": [0.004],
            "sharpe_like": [1.2],
            "max_drawdown": [-0.08],
            "mean_turnover": [0.3],
            "label_non_null_rate": [0.9],
            "eligible_rows": [1700],
            "can_enter_rows": [1600],
            "can_exit_rows": [1500],
            "n_rows_evaluated": [1234],
            "run_id": ["csmar_2025_forward_return_20d"],
        }
    )
    csv_path = tmp_path / "factor_summary.csv"
    md_path = tmp_path / "factor_summary.md"

    write_factor_summary_csv(csv_path, frame)
    write_factor_summary_markdown(md_path, frame)

    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    markdown = md_path.read_text(encoding="utf-8")
    assert "amihud_20d" in markdown


def test_factor_summary_contract_excludes_retired_performance_columns() -> None:
    retired = {
        "total_gross_portfolio_return",
        "total_net_portfolio_return",
        "mean_net_long_short_return",
        "weight_id",
        "cost_id",
    }

    assert retired.isdisjoint(PERFORMANCE_COLUMNS)


def test_build_comparison_frame_reports_full_minus_yearly_deltas() -> None:
    yearly = pd.DataFrame(
        {
            "year": [2024],
            "label": ["forward_return_20d"],
            "feature": ["mom_20d"],
            "mean_rankic": [0.01],
            "sharpe_like": [0.5],
            "run_id": ["yearly"],
        }
    )
    full = pd.DataFrame(
        {
            "year": [2024],
            "label": ["forward_return_20d"],
            "feature": ["mom_20d"],
            "mean_rankic": [0.015],
            "sharpe_like": [0.2],
            "run_id": ["full"],
        }
    )

    comparison = compare_factor_summaries.build_comparison_frame(yearly, full)

    rankic = comparison[comparison["metric"].eq("mean_rankic")].iloc[0]
    sharpe = comparison[comparison["metric"].eq("sharpe_like")].iloc[0]
    assert rankic["delta_full_minus_yearly"] == pytest.approx(0.005)
    assert sharpe["delta_full_minus_yearly"] == pytest.approx(-0.3)


def test_compare_factor_summaries_cli_defaults_to_current_diagnostic(
    tmp_path,
    monkeypatch,
) -> None:
    comparison_dir = (
        tmp_path
        / "reports"
        / "private_real"
        / "factor_evaluation"
        / "comparisons"
        / "full_vs_yearly_current_2024_2025"
    )
    full_dir = (
        tmp_path
        / "reports"
        / "private_real"
        / "factor_evaluation"
        / "full_2024_2025"
    )
    comparison_dir.mkdir(parents=True)
    full_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "year": [2024],
            "label": ["forward_return_20d"],
            "feature": ["mom_20d"],
            "mean_rankic": [0.01],
        }
    ).to_csv(comparison_dir / "yearly_current_factor_summary.csv", index=False)
    pd.DataFrame(
        {
            "year": [2024],
            "label": ["forward_return_20d"],
            "feature": ["mom_20d"],
            "mean_rankic": [0.02],
        }
    ).to_csv(full_dir / "factor_summary.csv", index=False)
    monkeypatch.chdir(tmp_path)

    exit_code = compare_factor_summaries.main([])

    assert exit_code == 0
    assert (comparison_dir / "metric_delta_long.csv").exists()
    assert (comparison_dir / "comparison_summary.md").exists()
