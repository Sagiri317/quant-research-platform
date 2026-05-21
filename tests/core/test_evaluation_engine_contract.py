import json

import pandas as pd
import pytest

from quant_research_platform.core.contracts import (
    CostConfig,
    EvaluationSpec,
    ExecutionConfig,
    NeutralizationConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.evaluation import run_evaluation


def test_pnl_uses_realized_return_column_while_diagnostics_use_label(tmp_path):
    panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 4),
            "asset": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "factor": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "forward_return_20d": [-0.50, 0.10, 0.20, 0.40, -0.60, 0.15, 0.25, 0.45],
            "forward_return_1d": [0.01, 0.02, 0.03, 0.04, 0.02, 0.02, 0.03, 0.05],
        }
    )
    spec = EvaluationSpec(
        panel=panel,
        feature_columns=("factor",),
        label_columns=("forward_return_20d",),
        metadata=_metadata("split_label_pnl_run"),
        n_groups=2,
        weight_configs=(
            WeightConfig(
                method="equal_weight_quantile",
                direction="long_high",
                top_quantile=0.25,
                bottom_quantile=0.25,
            ),
        ),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    assert set(result.metrics["ic"]["label"]) == {"forward_return_20d"}
    assert set(result.metrics["grouped_returns"]["label"]) == {"forward_return_20d"}
    assert set(result.metrics["returns"]["label"]) == {"forward_return_20d"}
    assert set(result.metrics["returns"]["pnl_return_column"]) == {
        "forward_return_1d"
    }
    pnl = result.metrics["pnl"].sort_values("timestamp").reset_index(drop=True)
    assert pnl["net_portfolio_return"].tolist() == pytest.approx([0.03, 0.03])
    assert pnl["cumulative_pnl"].tolist() == pytest.approx([0.03, 0.06])
    assert result.summary["schema"]["label_columns"] == ["forward_return_20d"]
    assert result.summary["schema"]["pnl_return_column"] == "forward_return_1d"


def test_run_evaluation_writes_evaluation_set_artifacts(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor", "alt_factor"),
        label_columns=("forward_return",),
        metadata=_metadata("unit_factor_run"),
        domain_notes={"boundary": "Synthetic test fixture only."},
        n_groups=2,
        weight_configs=(
            WeightConfig(
                method="equal_weight_quantile",
                direction="long_high",
                top_quantile=0.25,
                bottom_quantile=0.25,
            ),
        ),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    expected_paths = {
        "manifest": tmp_path / "manifest.json",
        "config": tmp_path / "config.json",
        "summary": tmp_path / "summary.json",
        "combinations": tmp_path / "combinations.csv",
        "ic": tmp_path / "metrics" / "ic.csv",
        "performance_summary": tmp_path / "metrics" / "performance_summary.csv",
        "positions": tmp_path / "portfolio" / "positions.csv",
        "pnl": tmp_path / "portfolio" / "pnl.csv",
        "report": tmp_path / "report.md",
    }
    for path in expected_paths.values():
        assert path.exists(), path

    manifest = json.loads(expected_paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(expected_paths["summary"].read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == "0.5"
    assert manifest["artifact_type"] == "evaluation_set"
    assert manifest["evaluation_id"] == "unit_factor_run"
    assert manifest["artifacts"]["combinations"] == "combinations.csv"
    display = manifest["display"]
    table_keys = {table["key"] for table in display["tables"]}
    assert {"combinations", "ic", "returns", "pnl"}.issubset(table_keys)
    assert {
        chart["id"] for chart in display["charts"]
    } == {"pnl", "ic_rankic", "return_turnover_cost"}
    all_download = next(
        preset for preset in display["downloads"]["presets"] if preset["key"] == "all"
    )
    assert "portfolio/pnl.csv" in all_download["items"]
    assert "artifacts.xlsx" not in all_download["items"]
    assert all(
        item["item"] != "artifacts.xlsx"
        for item in display["downloads"]["items"]
    )
    assert summary["n_combinations"] == 2
    assert set(result.metrics) >= {
        "ic",
        "ic_summary",
        "grouped_returns",
        "group_spread_summary",
        "performance_summary",
        "exposure_summary",
        "positions",
        "turnover",
        "returns",
        "pnl",
        "cost_breakdown",
        "weight_diagnostics",
        "neutralization_diagnostics",
    }
    assert set(pd.read_csv(expected_paths["positions"])["feature"]) == {
        "factor",
        "alt_factor",
    }
    performance_summary = pd.read_csv(expected_paths["performance_summary"])
    combinations = pd.read_csv(expected_paths["combinations"])
    assert "final_cumulative_gross_return" in performance_summary.columns
    assert "final_cumulative_net_return" in performance_summary.columns
    assert "total_gross_portfolio_return" not in performance_summary.columns
    assert "total_net_portfolio_return" not in performance_summary.columns
    assert "final_cumulative_net_return" in combinations.columns
    assert "total_net_portfolio_return" not in combinations.columns


def test_run_evaluation_evaluates_cross_section_once_per_combination(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.core.evaluation as evaluation

    calls = 0
    original_evaluator = evaluation.evaluate_ic_and_grouped_returns_from_validated_panel

    def counting_cross_section_metric(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        evaluation,
        "evaluate_ic_and_grouped_returns_from_validated_panel",
        counting_cross_section_metric,
    )
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("combined_cross_section"),
        n_groups=2,
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    assert result.metrics["ic"]["ic"].round(6).tolist() == [0.833809, 0.615918]
    assert set(result.metrics["grouped_returns"]["group"]) == {1, 2}
    assert calls == 1


def test_run_evaluation_reuses_validated_panel_for_same_label(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.core.evaluation as evaluation

    calls = 0
    original_validator = evaluation.validate_panel

    def counting_validate_panel(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_validator(*args, **kwargs)

    monkeypatch.setattr(evaluation, "validate_panel", counting_validate_panel)
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor", "alt_factor"),
        label_columns=("forward_return",),
        metadata=_metadata("shared_validation_run"),
        n_groups=2,
        output_dir=tmp_path,
    )

    run_evaluation(spec)

    assert calls == 1


def test_run_evaluation_does_not_revalidate_inside_cross_section(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.engines.dataframe.cross_section as cross_section

    def fail_revalidation(*args, **kwargs):
        raise AssertionError("run_evaluation should pass a prevalidated panel")

    monkeypatch.setattr(cross_section, "validate_panel", fail_revalidation)
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor", "alt_factor"),
        label_columns=("forward_return",),
        metadata=_metadata("cross_section_validation_reuse_run"),
        n_groups=2,
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    assert result.summary["n_combinations"] == 2


def test_run_evaluation_batches_features_for_same_label_weight_and_cost(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.core.evaluation as evaluation

    cross_section_calls = 0
    portfolio_calls = 0
    original_cross_section = (
        evaluation.evaluate_ic_and_grouped_returns_from_validated_panel
    )
    original_portfolio = (
        evaluation._simulate_weighted_long_short_returns_and_cost_from_validated_panel
    )

    def counting_cross_section(*args, **kwargs):
        nonlocal cross_section_calls
        cross_section_calls += 1
        return original_cross_section(*args, **kwargs)

    def counting_portfolio(*args, **kwargs):
        nonlocal portfolio_calls
        portfolio_calls += 1
        return original_portfolio(*args, **kwargs)

    monkeypatch.setattr(
        evaluation,
        "evaluate_ic_and_grouped_returns_from_validated_panel",
        counting_cross_section,
    )
    monkeypatch.setattr(
        evaluation,
        "_simulate_weighted_long_short_returns_and_cost_from_validated_panel",
        counting_portfolio,
    )
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor", "alt_factor"),
        label_columns=("forward_return",),
        metadata=_metadata("feature_set_batch_run"),
        n_groups=2,
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    assert result.summary["n_combinations"] == 2
    assert cross_section_calls == 1
    assert portfolio_calls == 1


def test_feature_set_combination_ids_are_mapped_once_per_feature(monkeypatch):
    import quant_research_platform.core.evaluation as evaluation

    calls = 0
    original_combination_id = evaluation._combination_id

    def counting_combination_id(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_combination_id(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_combination_id", counting_combination_id)
    frame = pd.DataFrame(
        {
            "feature": ["factor", "factor", "alt_factor"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    result = evaluation._add_feature_set_combination_columns(
        frame,
        label="forward_return",
        weight_id="w0",
        cost_id="c0",
    )

    assert calls == 2
    assert result["combination_id"].tolist() == [
        "factor__forward_return__w0__c0",
        "factor__forward_return__w0__c0",
        "alt_factor__forward_return__w0__c0",
    ]


def test_inverse_vol_scaled_weights_use_risk_column(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("risk_weight_run"),
        n_groups=2,
        weight_configs=(
            WeightConfig(
                method="inverse_vol_scaled_quantile",
                direction="long_high",
                top_quantile=0.5,
                bottom_quantile=0.5,
                risk_column="risk",
            ),
        ),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    positions = result.metrics["positions"]
    first_day = positions[positions["timestamp"] == pd.Timestamp("2026-01-01")]
    long_weights = first_day[first_day["position"] > 0].set_index("asset")["position"]
    assert long_weights["D"] > long_weights["B"]


def test_explicit_execution_tradability_and_cost_configs_are_recorded(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel().assign(
            eligible=True,
            can_enter=True,
            can_exit=True,
            next_open=1.0,
        ),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("execution_config_run"),
        n_groups=2,
        weight_configs=(WeightConfig(top_quantile=0.5, bottom_quantile=0.5),),
        execution_config=ExecutionConfig(
            execution_lag=1,
            execution_price_column="next_open",
        ),
        tradability_config=TradabilityConfig(
            eligibility_column="eligible",
            can_enter_column="can_enter",
            can_exit_column="can_exit",
            blocked_entry_policy="hold_cash",
            blocked_exit_policy="carry_forward",
        ),
        cost_configs=(
            CostConfig(
                buy_cost_bps=10,
                sell_cost_bps=20,
                slippage_bps=2,
                spread_bps=1,
                impact_model="linear_proxy",
            ),
        ),
        output_dir=tmp_path,
    )

    run_evaluation(spec)

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert config["execution_config"]["execution_lag"] == 1
    assert config["tradability_config"]["eligibility_column"] == "eligible"
    assert config["cost_configs"][0]["impact_model"] == "linear_proxy"
    assert summary["execution_config"] == config["execution_config"]
    assert summary["tradability_config"] == config["tradability_config"]
    assert summary["cost_configs"] == config["cost_configs"]


def test_execution_and_tradability_config_columns_must_exist(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("missing_execution_columns"),
        n_groups=2,
        execution_config=ExecutionConfig(execution_price_column="next_open"),
        tradability_config=TradabilityConfig(eligibility_column="eligible"),
        output_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="evaluation panel is missing columns"):
        run_evaluation(spec)


def test_evaluation_spec_applies_tradability_config_to_positions(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel().assign(
            eligible=[True, True, False, True, True, True, True, True],
        ),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("tradability_behavior_run"),
        n_groups=2,
        weight_configs=(WeightConfig(top_quantile=0.25, bottom_quantile=0.25),),
        tradability_config=TradabilityConfig(eligibility_column="eligible"),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    first_day = result.metrics["positions"][
        result.metrics["positions"]["timestamp"] == pd.Timestamp("2026-01-01")
    ]
    assert first_day.set_index("asset")["position"].to_dict() == {"A": -1.0, "D": 1.0}


def test_evaluation_spec_applies_neutralization_before_scoring(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("neutralized_behavior_run"),
        n_groups=2,
        neutralization_config=NeutralizationConfig(
            method="demean_by_group",
            group_columns=("industry",),
        ),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    diagnostics = result.metrics["neutralization_diagnostics"]
    assert set(diagnostics["method"]) == {"demean_by_group"}
    assert set(diagnostics["group_columns"]) == {"industry"}
    assert "neutralization_config" in result.summary


def test_segmented_prediction_turnover_resets_between_splits(tmp_path):
    prediction_panel = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                ]
            ),
            "asset": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "split": ["validation"] * 4 + ["test"] * 4,
            "prediction": [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
            "forward_return": [0.02, -0.01, -0.02, 0.03, 0.04, -0.01, 0.01, 0.0],
            "forward_return_1d": [0.02, -0.01, -0.02, 0.03, 0.04, -0.01, 0.01, 0.0],
        }
    )
    spec = EvaluationSpec(
        panel=prediction_panel,
        feature_columns=("prediction",),
        label_columns=("forward_return",),
        metadata={
            **_metadata("prediction_segment_run"),
            "kind": "prediction",
        },
        n_groups=2,
        weight_configs=(
            WeightConfig(
                method="equal_weight_quantile",
                direction="long_high",
                top_quantile=0.5,
                bottom_quantile=0.5,
            ),
        ),
        segment_columns=("split",),
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    ic = result.metrics["ic"]
    grouped_returns = result.metrics["grouped_returns"]
    assert "split" in ic.columns
    assert "split" in grouped_returns.columns
    assert set(ic["split"]) == {"validation", "test"}
    assert set(grouped_returns["split"]) == {"validation", "test"}

    turnover = result.metrics["turnover"]
    first_by_split = turnover.sort_values("timestamp").groupby("split").first()
    assert first_by_split.loc["validation", "turnover"] == pytest.approx(2.0)
    assert first_by_split.loc["test", "turnover"] == pytest.approx(2.0)


def test_evaluation_spec_requires_minimum_run_metadata() -> None:
    with pytest.raises(ValueError, match="metadata is missing required fields: domain"):
        EvaluationSpec(
            panel=_factor_panel(),
            feature_columns=("factor",),
            label_columns=("forward_return",),
            metadata={
                "project": "quant-research-platform",
                "kind": "factor",
                "data_stage": "synthetic_only",
                "uses_external_data": False,
            },
        )


def _metadata(run_id: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "core_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _factor_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-02",
                ]
            ),
            "asset": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "industry": [
                "tech",
                "tech",
                "bank",
                "bank",
                "tech",
                "tech",
                "bank",
                "bank",
            ],
            "factor": [0.1, 0.4, -0.2, 0.8, 0.3, -0.4, 0.7, 0.2],
            "alt_factor": [1.0, 0.5, -0.5, -1.0, 1.2, 0.1, -0.2, -0.7],
            "risk": [1.0, 2.0, 1.5, 0.5, 1.0, 1.2, 0.8, 1.4],
            "forward_return": [0.02, 0.01, -0.01, 0.03, 0.04, -0.02, 0.01, 0.0],
            "forward_return_1d": [0.02, 0.01, -0.01, 0.03, 0.04, -0.02, 0.01, 0.0],
        }
    )
