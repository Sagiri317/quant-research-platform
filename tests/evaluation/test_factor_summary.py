from __future__ import annotations

import json

import pandas as pd
import pytest
from evaluation.build_factor_summary import (
    build_factor_summary_frame,
    main,
    write_factor_summary_csv,
    write_factor_summary_markdown,
)
from evaluation.compare_factor_summaries import build_comparison_frame
from evaluation.compare_factor_summaries import main as compare_main


def test_build_factor_summary_merges_core_metrics_and_labels(tmp_path) -> None:
    report_root = tmp_path / "factor_evaluation"
    run_dir = report_root / "core_runs" / "csmar_2024_forward_return_20d"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)

    (report_root / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "core_runs": [
                    {
                        "year": 2024,
                        "label": "forward_return_20d",
                        "features": ["mom_20d"],
                        "output_dir": str(run_dir),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "csmar_2024_forward_return_20d",
                "n_rows_evaluated": 1234,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "year": [2024],
            "label": ["forward_return_20d"],
            "rows": [2000],
            "non_null_rows": [1800],
            "non_null_rate": [0.9],
            "eligible_rows": [1700],
            "can_enter_rows": [1600],
            "can_exit_rows": [1500],
        }
    ).to_csv(report_root / "label_availability.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "label": ["forward_return_20d"],
            "mean_ic": [0.01],
            "std_ic": [0.02],
            "icir": [0.5],
            "positive_ic_rate": [0.6],
            "mean_rankic": [0.03],
            "std_rankic": [0.04],
            "rankicir": [0.75],
            "positive_rankic_rate": [0.65],
        }
    ).to_csv(metrics_dir / "ic_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "label": ["forward_return_20d"],
            "top_group_mean_return": [0.02],
            "bottom_group_mean_return": [-0.01],
            "top_bottom_spread_mean": [0.03],
            "top_bottom_spread_hit_rate": [0.55],
            "group_monotonicity_score": [0.8],
        }
    ).to_csv(metrics_dir / "group_spread_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "periods": [250],
            "mean_net_long_short_return": [0.004],
            "sharpe_like": [1.2],
            "max_drawdown": [-0.08],
            "mean_turnover": [0.3],
        }
    ).to_csv(metrics_dir / "performance_summary.csv", index=False)

    summary = build_factor_summary_frame(report_root)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["year"] == 2024
    assert row["feature"] == "mom_20d"
    assert row["mean_rankic"] == 0.03
    assert row["top_bottom_spread_mean"] == 0.03
    assert row["mean_net_long_short_return"] == 0.004
    assert row["label_non_null_rate"] == 0.9
    assert row["n_rows_evaluated"] == 1234


def test_build_factor_summary_handles_full_sample_manifest(tmp_path) -> None:
    report_root = tmp_path / "factor_evaluation"
    run_dir = report_root / "core_runs" / "csmar_2024_2025_forward_return_20d"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (report_root / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "core_runs": [
                    {
                        "run_id": "csmar_2024_2025_forward_return_20d",
                        "start_year": 2024,
                        "end_year": 2025,
                        "label": "forward_return_20d",
                        "features": ["mom_20d"],
                        "output_dir": str(run_dir),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"evaluation_id": "csmar_2024_2025_forward_return_20d"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "year": [2024, 2025],
            "label": ["forward_return_20d", "forward_return_20d"],
            "non_null_rate": [0.9, 0.8],
            "eligible_rows": [1700, 1800],
            "can_enter_rows": [1600, 1700],
            "can_exit_rows": [1500, 1600],
        }
    ).to_csv(report_root / "label_availability.csv", index=False)
    breakdown_dir = run_dir / "metrics_by_year"
    breakdown_dir.mkdir()
    pd.DataFrame(
        {
            "feature": ["mom_20d", "mom_20d"],
            "label": ["forward_return_20d", "forward_return_20d"],
            "year": [2024, 2025],
            "mean_ic": [0.01, 0.02],
            "icir": [0.5, 0.6],
            "positive_ic_rate": [0.6, 0.7],
            "mean_rankic": [0.03, 0.04],
            "rankicir": [0.75, 0.85],
            "positive_rankic_rate": [0.65, 0.75],
        }
    ).to_csv(breakdown_dir / "ic_summary_by_year.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d", "mom_20d"],
            "label": ["forward_return_20d", "forward_return_20d"],
            "year": [2024, 2025],
            "top_bottom_spread_mean": [0.03, 0.04],
            "top_bottom_spread_hit_rate": [0.55, 0.65],
            "group_monotonicity_score": [0.8, 0.9],
        }
    ).to_csv(breakdown_dir / "group_spread_summary_by_year.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d", "mom_20d"],
            "year": [2024, 2025],
            "mean_net_long_short_return": [0.004, 0.005],
            "sharpe_like": [1.2, 1.3],
            "max_drawdown": [-0.08, -0.09],
            "mean_turnover": [0.3, 0.4],
        }
    ).to_csv(breakdown_dir / "performance_summary_by_year.csv", index=False)

    summary = build_factor_summary_frame(report_root)

    assert len(summary) == 2
    assert summary["year"].tolist() == [2024, 2025]
    assert summary["run_id"].tolist() == [
        "csmar_2024_2025_forward_return_20d",
        "csmar_2024_2025_forward_return_20d",
    ]


def test_build_factor_summary_cli_defaults_to_full_report_root(
    tmp_path,
    monkeypatch,
) -> None:
    report_root = tmp_path / "reports" / "private_real" / "factor_evaluation"
    run_dir = report_root / "full_2024_2025" / "core_runs" / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (report_root / "full_2024_2025" / "evaluation_manifest.json").write_text(
        json.dumps(
            {
                "core_runs": [
                    {
                        "year": 2024,
                        "label": "forward_return_20d",
                        "features": ["mom_20d"],
                        "output_dir": str(run_dir),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "label": ["forward_return_20d"],
            "mean_ic": [0.01],
            "icir": [0.5],
            "positive_ic_rate": [0.6],
            "mean_rankic": [0.03],
            "rankicir": [0.75],
            "positive_rankic_rate": [0.65],
        }
    ).to_csv(metrics_dir / "ic_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "label": ["forward_return_20d"],
            "top_bottom_spread_mean": [0.03],
            "top_bottom_spread_hit_rate": [0.55],
            "group_monotonicity_score": [0.8],
        }
    ).to_csv(metrics_dir / "group_spread_summary.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["mom_20d"],
            "mean_net_long_short_return": [0.004],
            "sharpe_like": [1.2],
            "max_drawdown": [-0.08],
            "mean_turnover": [0.3],
        }
    ).to_csv(metrics_dir / "performance_summary.csv", index=False)
    monkeypatch.chdir(tmp_path)

    exit_code = main([])

    assert exit_code == 0
    assert (report_root / "full_2024_2025" / "factor_summary.csv").exists()


def test_factor_summary_writers_use_stable_outputs_and_excel_safe_csv(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "year": [2025],
            "label": ["forward_return_20d"],
            "feature": ["amihud_20d"],
            "mean_ic": [0.01],
            "icir": [0.5],
            "positive_ic_rate": [0.6],
            "mean_rankic": [0.03],
            "rankicir": [0.75],
            "positive_rankic_rate": [0.65],
            "top_bottom_spread_mean": [0.03],
            "top_bottom_spread_hit_rate": [0.55],
            "group_monotonicity_score": [0.8],
            "mean_net_long_short_return": [0.004],
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
    assert "# 因子评估汇总" in markdown
    assert "amihud_20d" in markdown


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

    comparison = build_comparison_frame(yearly, full)

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

    exit_code = compare_main([])

    assert exit_code == 0
    assert (comparison_dir / "metric_delta_long.csv").exists()
    assert (comparison_dir / "comparison_summary.md").exists()
