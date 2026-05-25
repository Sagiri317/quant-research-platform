from __future__ import annotations

import pytest

import quant_research_platform.artifacts.artifact_comparison as comparison_module
from quant_research_platform.artifacts.artifact_comparison import (
    compare_evaluation_sets,
)
from quant_research_platform.artifacts.artifact_loader import load_evaluation_set
from quant_research_platform.artifacts.evaluation_comparison_report import (
    write_evaluation_comparison_report,
)
from tests.interfaces.api.artifact_run_fixtures import write_evaluation_set


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


def test_compare_evaluation_sets_uses_low_memory_file_summary(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_low_memory"
    write_evaluation_set(run_dir, evaluation_id="run_low_memory")

    def fail_load(*_args, **_kwargs):
        raise AssertionError("compare_evaluation_sets must not summarize loaded runs")

    monkeypatch.setattr(comparison_module, "_summarize_run", fail_load)

    comparison = compare_evaluation_sets([run_dir])

    assert comparison["evaluation_id"].tolist() == ["run_low_memory"]


def test_write_evaluation_comparison_report_outputs_file_only_formats(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    report_dir = tmp_path / "comparison_report"
    write_evaluation_set(run_a, evaluation_id="run_a")
    write_evaluation_set(run_b, evaluation_id="run_b")

    outputs = write_evaluation_comparison_report([run_a, run_b], report_dir)

    assert set(outputs) == {
        "summary_csv",
        "combination_csv",
        "kernel_parity_csv",
        "json",
        "markdown",
    }
    assert outputs["summary_csv"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert outputs["combination_csv"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert outputs["kernel_parity_csv"].read_bytes().startswith(b"\xef\xbb\xbf")
    assert '"run_count": 2' in outputs["json"].read_text(encoding="utf-8")
    assert '"combination_comparison"' in outputs["json"].read_text(encoding="utf-8")
    assert "factor__forward_return__dataframe__long_short" in outputs[
        "combination_csv"
    ].read_text(encoding="utf-8-sig")
    markdown = outputs["markdown"].read_text(
        encoding="utf-8"
    )
    assert "# Evaluation Comparison Report" in markdown
    assert "## Combination Comparison" in markdown
    assert "## Kernel Parity" in markdown
    assert not (report_dir / "manifest.json").exists()


@pytest.mark.parametrize("report_name", ["../outside", "nested/report", "bad\\name"])
def test_write_evaluation_comparison_report_rejects_unsafe_report_name(
    tmp_path,
    report_name,
):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    report_dir = tmp_path / "comparison_report"
    write_evaluation_set(run_a, evaluation_id="run_a")
    write_evaluation_set(run_b, evaluation_id="run_b")

    with pytest.raises(ValueError, match="report_name"):
        write_evaluation_comparison_report(
            [run_a, run_b],
            report_dir,
            report_name=report_name,
        )

    assert not (tmp_path / "outside.csv").exists()
    assert not (report_dir / "nested").exists()


def test_compare_evaluation_sets_requires_schema_evaluation_id(tmp_path):
    run_dir = tmp_path / "run_without_id"
    write_evaluation_set(run_dir, evaluation_id="run_without_id")
    loaded = load_evaluation_set(run_dir)
    loaded.summary.pop("evaluation_id")

    with pytest.raises(ValueError, match="evaluation_id"):
        comparison_module._summarize_run(loaded)


@pytest.mark.parametrize(
    "field",
    ["project", "kind", "n_rows_evaluated", "n_combinations"],
)
def test_compare_evaluation_sets_requires_current_summary_fields(tmp_path, field):
    run_dir = tmp_path / f"run_without_{field}"
    write_evaluation_set(run_dir, evaluation_id="run_without_field")
    loaded = load_evaluation_set(run_dir)
    loaded.summary.pop(field)

    with pytest.raises(ValueError, match=field):
        comparison_module._summarize_run(loaded)


def test_compare_evaluation_sets_requires_weight_method(tmp_path):
    run_dir = tmp_path / "run_without_weight_method"
    write_evaluation_set(run_dir, evaluation_id="run_without_weight_method")
    loaded = load_evaluation_set(run_dir)
    loaded.config["weight_configs"] = [{}]

    with pytest.raises(ValueError, match="weight_configs"):
        comparison_module._summarize_run(loaded)


def test_compare_evaluation_sets_reports_scalar_weight_cost_bps(tmp_path):
    run_dir = tmp_path / "run_with_scalar_weight_cost_bps"
    write_evaluation_set(run_dir, evaluation_id="run_with_scalar_weight_cost_bps")
    loaded = load_evaluation_set(run_dir)
    loaded.config["cost_configs"] = [{}]
    loaded.config["weight_configs"] = [
        {"method": "equal_weight_quantile", "cost_bps": 99.0}
    ]

    record = comparison_module._summarize_run(loaded)

    assert record["cost_bps"] == 99.0


def test_compare_evaluation_sets_rejects_legacy_schema_without_full_load(
    tmp_path,
    monkeypatch,
):
    run_dir = tmp_path / "legacy_schema"
    write_evaluation_set(run_dir, evaluation_id="legacy_schema")
    for filename in ("manifest.json", "summary.json"):
        path = run_dir / filename
        payload = path.read_text(encoding="utf-8").replace(
            '"artifact_schema_version": "0.6"',
            '"artifact_schema_version": "0.5"',
        )
        path.write_text(payload, encoding="utf-8")

    def fail_load(*_args, **_kwargs):
        raise AssertionError("low-memory comparison must not full-load run tables")

    monkeypatch.setattr(comparison_module, "_summarize_run", fail_load)

    with pytest.raises(ValueError, match="unsupported artifact_schema_version"):
        compare_evaluation_sets([run_dir])
