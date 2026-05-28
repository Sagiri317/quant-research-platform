import json

import pandas as pd
import pytest

from quant_research_platform.artifacts import load_evaluation_set
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    EvaluationSpec,
    WeightConfig,
)


def test_time_breakdowns_write_year_quarter_month_summary_artifacts(tmp_path):
    spec = EvaluationSpec(
        panel=_two_year_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata={
            "run_id": "period_breakdown_run",
            "project": "quant-research-platform",
            "kind": "factor",
            "domain": "evaluation_synthetic",
            "data_stage": "synthetic_only",
            "uses_external_data": False,
        },
        weight_configs=(WeightConfig(top_quantile=0.5, bottom_quantile=0.5),),
        n_groups=2,
        time_breakdowns=("year", "quarter", "month"),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    base = summary["base_evaluation"]
    assert base["breakdown_enabled"] is True
    assert base["breakdown_columns"] == ["year", "quarter", "month"]
    assert (
        base["period_breakdown_policy"]
        == "slice_overall_returns_rebase_period_drawdown"
    )
    manifest_base = manifest["base_evaluation"]
    assert manifest_base["breakdown_enabled"] is True
    assert manifest_base["breakdown_columns"] == ["year", "quarter", "month"]
    assert (
        manifest_base["period_breakdown_policy"]
        == "slice_overall_returns_rebase_period_drawdown"
    )

    for period in ("year", "quarter", "month"):
        directory = tmp_path / "breakdowns" / period
        name = f"evaluation_metrics_summary_by_{period}"
        assert (directory / "evaluation_metrics_summary.csv").exists()
        assert name in result.metrics
        assert result.metrics[name]["period_type"].eq(period).all()
        assert (
            manifest["artifacts"][f"breakdowns/{period}"][name]
            == f"breakdowns/{period}/evaluation_metrics_summary.csv"
        )
        display_tables = {
            table["key"]: table for table in manifest["display"]["tables"]
        }
        display_downloads = manifest["display"]["downloads"]
        download_items = {
            (item["kind"], item["key"]) for item in display_downloads["items"]
        }
        assert display_tables[name]["path"] == (
            f"breakdowns/{period}/evaluation_metrics_summary.csv"
        )
        assert ("table", name) in download_items
        all_preset = next(
            preset for preset in display_downloads["presets"] if preset["key"] == "all"
        )
        assert {"kind": "table", "key": name} in all_preset["items"]

    performance_by_year = pd.read_csv(
        tmp_path / "breakdowns" / "year" / "evaluation_metrics_summary.csv"
    )
    periods = performance_by_year[
        performance_by_year["metric_name"].eq("periods")
    ].set_index("period")["metric_value"]
    assert set(periods.index.astype(int)) == {2025, 2026}
    assert periods.astype(float).to_dict() == {
        2025: 2,
        2026: 2,
    }
    drawdowns = performance_by_year[
        performance_by_year["metric_name"].eq("max_drawdown")
    ].set_index("period")["metric_value"]
    drawdowns = pd.to_numeric(drawdowns)
    assert drawdowns[2025] == pytest.approx(-0.2)
    assert drawdowns[2026] == pytest.approx(-0.1)

    loaded = load_evaluation_set(tmp_path)
    assert "evaluation_metrics_summary_by_year" in loaded.tables
    assert "evaluation_metrics_summary_by_month" in loaded.tables


def test_monthly_ic_breakdown_is_written_and_display_declared(tmp_path):
    spec = EvaluationSpec(
        panel=_two_year_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata={
            "run_id": "monthly_ic_breakdown_run",
            "project": "quant-research-platform",
            "kind": "factor",
            "domain": "evaluation_synthetic",
            "data_stage": "synthetic_only",
            "uses_external_data": False,
        },
        weight_configs=(WeightConfig(top_quantile=0.5, bottom_quantile=0.5),),
        n_groups=2,
        time_breakdowns=("month",),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    output_path = tmp_path / "breakdowns" / "month" / "ic.csv"
    assert output_path.exists()
    assert "ic_breakdown_by_month" in result.metrics
    monthly_ic = pd.read_csv(output_path)
    assert list(monthly_ic.columns) == [
        "combination_id",
        "period_type",
        "period",
        "period_start",
        "period_end",
        "mean_ic",
        "std_ic",
        "icir",
        "positive_ic_rate",
        "mean_rankic",
        "std_rankic",
        "rankicir",
        "positive_rankic_rate",
        "n_observations",
        "diagnostic_status",
        "status_reason",
        "warning",
    ]
    assert monthly_ic["period_type"].eq("month").all()
    assert set(monthly_ic["period"]) == {"2025-12", "2026-01"}

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["artifacts"]["breakdowns/month"]["ic_breakdown_by_month"]
        == "breakdowns/month/ic.csv"
    )
    display_tables = {table["key"]: table for table in manifest["display"]["tables"]}
    assert display_tables["ic_breakdown_by_month"]["path"] == (
        "breakdowns/month/ic.csv"
    )
    download_items = {
        (item["kind"], item["key"])
        for item in manifest["display"]["downloads"]["items"]
    }
    assert ("table", "ic_breakdown_by_month") in download_items


def test_time_breakdown_performance_slices_overall_returns_without_reweighting():
    from quant_research_platform.evaluations.period_breakdowns import (
        build_time_breakdown_metrics,
    )

    returns = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-03"]),
            "feature": ["factor", "factor", "factor"],
            "label": ["forward_return", "forward_return", "forward_return"],
            "long_return": [0.4, -0.2, 0.1],
            "short_return": [0.0, 0.0, 0.0],
            "portfolio_mode": ["long_short", "long_short", "long_short"],
            "gross_portfolio_return": [0.4, -0.2, 0.1],
            "turnover": [2.0, 0.25, 0.75],
            "cost": [0.02, 0.01, 0.03],
            "net_portfolio_return": [0.38, -0.21, 0.07],
            "long_count": [1, 1, 1],
            "short_count": [1, 1, 1],
        }
    )
    metrics = {
        "ic": pd.DataFrame(
            {
                "timestamp": returns["timestamp"],
                "feature": ["factor", "factor", "factor"],
                "label": ["forward_return", "forward_return", "forward_return"],
                "ic": [1.0, -1.0, 1.0],
                "rankic": [1.0, -1.0, 1.0],
                "n_obs": [2, 2, 2],
            }
        ),
        "grouped_returns": pd.DataFrame(
            {
                "timestamp": list(returns["timestamp"]) * 2,
                "feature": ["factor"] * 6,
                "label": ["forward_return"] * 6,
                "group": [1, 1, 1, 2, 2, 2],
                "mean_forward_return": [0.0, 0.0, 0.0, 0.4, -0.2, 0.1],
                "n_obs": [1, 1, 1, 1, 1, 1],
            }
        ),
        "returns": returns,
    }

    breakdowns = build_time_breakdown_metrics(metrics, time_breakdowns=("year",))
    by_year = breakdowns["performance_summary_by_year"]

    row_2026 = by_year.loc[by_year["year"] == 2026].iloc[0]
    assert row_2026["periods"] == 2
    assert row_2026["final_cumulative_net_return"] == pytest.approx(-0.14)
    assert row_2026["total_cost"] == pytest.approx(0.04)
    assert row_2026["mean_turnover"] == pytest.approx(0.5)
    assert row_2026["max_drawdown"] == pytest.approx(-0.21)


def test_time_breakdown_values_are_validated(tmp_path):
    with pytest.raises(ValueError, match="time_breakdowns"):
        EvaluationSpec(
            panel=_two_year_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata={
                "project": "quant-research-platform",
                "kind": "factor",
                "domain": "evaluation_synthetic",
                "data_stage": "synthetic_only",
                "uses_external_data": False,
            },
            time_breakdowns=("week",),
            output_dir=tmp_path,
        )


def _two_year_panel() -> pd.DataFrame:
    rows = []
    returns = {
        "2025-12-30": 0.1,
        "2025-12-31": -0.2,
        "2026-01-02": 0.05,
        "2026-01-03": -0.1,
    }
    for timestamp, long_asset_return in returns.items():
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "A",
                "factor": 1.0,
                "forward_return": long_asset_return,
                "forward_return_1d": long_asset_return,
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "asset": "B",
                "factor": -1.0,
                "forward_return": 0.0,
                "forward_return_1d": 0.0,
            }
        )
    return pd.DataFrame.from_records(rows)
