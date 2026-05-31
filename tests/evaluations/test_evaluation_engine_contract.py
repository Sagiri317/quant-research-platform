import json

import pandas as pd
import pytest

from quant_research_platform.artifacts import load_evaluation_set
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    CostConfig,
    DiagnosticsSpec,
    EvaluationSpec,
    ExecutionConfig,
    NeutralizationConfig,
    TensorBatchConfig,
    TradabilityConfig,
    WeightConfig,
)


def test_cumulative_returns_use_return_source_column_for_portfolio_path(
    tmp_path,
):
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
        metadata=_metadata("split_label_return_source_run"),
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

    assert set(result.combinations["label"]) == {"forward_return_20d"}
    assert set(result.metrics["returns"]["return_source_column"]) == {
        "forward_return_1d"
    }
    returns = result.metrics["returns"].sort_values("timestamp").reset_index(drop=True)
    assert returns["net_return"].tolist() == pytest.approx([0.03, 0.03])
    assert (1.0 + returns["net_return"]).cumprod().sub(1.0).tolist() == pytest.approx(
        [0.03, 0.0609]
    )
    base = result.summary["base_evaluation"]
    assert base["label_columns"] == ["forward_return_20d"]
    assert base["return_source_column"] == "forward_return_1d"


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
        "run_context": tmp_path / "run_context.json",
        "summary": tmp_path / "summary.json",
        "combinations": tmp_path / "dimensions" / "combinations.csv",
        "ic": tmp_path / "metrics" / "ic.csv",
        "evaluation_metrics_summary": (
            tmp_path / "metrics" / "evaluation_metrics_summary.csv"
        ),
        "returns": tmp_path / "portfolio" / "returns.csv",
        "diagnostic_summary": tmp_path / "diagnostics" / "diagnostic_summary.csv",
        "report": tmp_path / "report.md",
    }
    for path in expected_paths.values():
        assert path.exists(), path

    manifest = json.loads(expected_paths["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(expected_paths["summary"].read_text(encoding="utf-8"))
    run_context = json.loads(expected_paths["run_context"].read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == "0.7"
    assert manifest["artifact_type"] == "evaluation_set"
    assert manifest["run_id"] == "unit_factor_run"
    assert "evaluation_id" not in manifest
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "lineage.json").exists()
    assert manifest["artifacts"]["dimensions"]["combinations"] == (
        "dimensions/combinations.csv"
    )
    display = manifest["display"]
    table_keys = {table["key"] for table in display["tables"]}
    assert {
        "combinations",
        "ic",
        "returns",
        "evaluation_metrics_summary",
        "coverage_summary",
        "diagnostic_summary",
        "diagnostic_metrics",
    }.issubset(table_keys)
    assert {chart["id"] for chart in display["charts"]} == {
        "primary_cumulative_return",
    }
    all_download = next(
        preset for preset in display["downloads"]["presets"] if preset["key"] == "all"
    )
    assert {"kind": "table", "key": "returns"} in all_download["items"]
    assert {"kind": "metadata_file", "key": "run_context"} in all_download["items"]
    assert "artifacts.xlsx" not in all_download["items"]
    assert all(
        item.get("key") != "artifacts.xlsx" for item in display["downloads"]["items"]
    )
    assert summary["n_combinations"] == 2
    assert set(result.metrics) >= {
        "ic",
        "grouped_returns",
        "evaluation_metrics_summary",
        "coverage_summary",
        "returns",
        "diagnostic_policies",
        "diagnostic_summary",
        "diagnostic_metrics",
    }
    assert summary["run_id"] == run_context["provenance"]["run_id"]
    metrics = pd.read_csv(expected_paths["evaluation_metrics_summary"])
    combinations = pd.read_csv(expected_paths["combinations"])
    assert "final_cumulative_net_return" in set(metrics["metric_name"])
    assert "performance_summary" not in table_keys
    assert "cumulative_returns" not in table_keys
    assert "total_net_portfolio_return" not in combinations.columns


def test_run_evaluation_evaluates_cross_section_once_per_combination(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.evaluations.evaluation_runner as evaluation

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
    import quant_research_platform.evaluations.evaluation_runner as evaluation

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
    import quant_research_platform.kernels.dataframe.cross_section as cross_section

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


def test_run_evaluation_batches_features_for_same_label_run_level_configs(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.evaluations.evaluation_runner as evaluation

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


def test_tensor_kernel_uses_tensor_metric_path(tmp_path, monkeypatch):
    import quant_research_platform.evaluations.evaluation_runner as evaluation

    def fail_dataframe_cross_section(*args, **kwargs):
        raise AssertionError("tensor kernel must not use dataframe cross-section")

    def fail_dataframe_portfolio(*args, **kwargs):
        raise AssertionError("tensor kernel must not use dataframe portfolio")

    monkeypatch.setattr(
        evaluation,
        "evaluate_ic_and_grouped_returns_from_validated_panel",
        fail_dataframe_cross_section,
    )
    monkeypatch.setattr(
        evaluation,
        "_simulate_weighted_long_short_returns_and_cost_from_validated_panel",
        fail_dataframe_portfolio,
    )
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("tensor_kernel_run"),
        n_groups=2,
        kernel="tensor",
        output_dir=tmp_path,
    )

    result = run_evaluation(spec)

    assert result.summary["base_evaluation"]["kernel"] == "tensor"
    assert not result.metrics["returns"].empty
    assert not result.metrics["diagnostic_summary"].empty


def test_tensor_kernel_passes_configured_value_dtype_to_tensor_conversion(
    tmp_path,
    monkeypatch,
):
    import quant_research_platform.evaluations.evaluation_runner as evaluation

    cross_section_dtypes: list[str] = []
    portfolio_dtypes: list[str] = []
    original_panel_to_tensor = evaluation.panel_to_tensor
    original_portfolio = evaluation.build_tensor_portfolio_artifacts

    def spy_panel_to_tensor(*args, **kwargs):
        cross_section_dtypes.append(kwargs.get("value_dtype", "float64"))
        return original_panel_to_tensor(*args, **kwargs)

    def spy_tensor_portfolio(*args, **kwargs):
        portfolio_dtypes.append(kwargs.get("value_dtype", "float64"))
        return original_portfolio(*args, **kwargs)

    monkeypatch.setattr(evaluation, "panel_to_tensor", spy_panel_to_tensor)
    monkeypatch.setattr(
        evaluation,
        "build_tensor_portfolio_artifacts",
        spy_tensor_portfolio,
    )
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("tensor_dtype_run"),
        n_groups=2,
        kernel="tensor",
        tensor_config=TensorBatchConfig(value_dtype="float32"),
        output_dir=tmp_path,
        diagnostics_spec=DiagnosticsSpec.from_request(
            {"enabled": True, "materialized_views": ["weights"]}
        ),
    )

    run_evaluation(spec)

    assert cross_section_dtypes
    assert portfolio_dtypes
    assert set(cross_section_dtypes) == {"float32"}
    assert set(portfolio_dtypes) == {"float32"}


def test_feature_set_combination_ids_are_mapped_once_per_feature(monkeypatch):
    import quant_research_platform.evaluations.evaluation_runner as evaluation

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
        kernel="dataframe",
        portfolio_mode="long_only",
    )

    assert calls == 2
    assert result["combination_id"].tolist() == [
        "factor__forward_return__dataframe__long_only",
        "factor__forward_return__dataframe__long_only",
        "alt_factor__forward_return__dataframe__long_only",
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
        diagnostics_spec=DiagnosticsSpec.from_request(
            {"enabled": True, "materialized_views": ["weights"]}
        ),
    )

    result = run_evaluation(spec)

    weights = result.metrics["weights"]
    first_day = weights[weights["timestamp"] == pd.Timestamp("2026-01-01")]
    long_weights = first_day[first_day["target_weight"] > 0].set_index("asset")[
        "target_weight"
    ]
    assert long_weights["D"] > long_weights["B"]
    run_context = json.loads((tmp_path / "run_context.json").read_text())
    assert (
        run_context["provenance"]["diagnostic_policy_fingerprint"]
        == "n5_v1_descriptive_proxy"
    )


def test_missing_optional_context_diagnostic_metrics_validate_schema_07(tmp_path):
    spec = EvaluationSpec(
        panel=_factor_panel(),
        feature_columns=("factor",),
        label_columns=("forward_return",),
        metadata=_metadata("missing_optional_context_diagnostics"),
        n_groups=2,
        output_dir=tmp_path,
        diagnostics_spec=DiagnosticsSpec.from_request(
            {
                "enabled": True,
                "materialized_views": ["exposure_capacity_summary"],
            }
        ),
    )

    result = run_evaluation(spec)
    loaded = load_evaluation_set(tmp_path)

    metrics = result.metrics["diagnostic_metrics"]
    invalid = metrics[
        metrics["diagnostic_status"].eq("partial")
        & metrics["metric_value"].isna()
    ]
    assert invalid.empty
    missing_optional = metrics[
        metrics["metric_name"].isin(
            {
                "liquidity_context_coverage_proxy",
                "median_amount_proxy",
                "median_turnover_liquidity_proxy",
                "market_cap_context_coverage_proxy",
                "median_total_market_cap_proxy",
                "median_float_market_cap_proxy",
            }
        )
    ]
    assert set(missing_optional["diagnostic_status"]) == {"unavailable"}
    assert "diagnostic_metrics" in loaded.tables
    assert "exposure_capacity_summary" in loaded.tables


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

    run_context = json.loads(
        (tmp_path / "run_context.json").read_text(encoding="utf-8")
    )
    config = run_context["config"]["evaluation_spec"]
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert config["execution_config"]["execution_lag"] == 1
    assert config["tradability_config"]["eligibility_column"] == "eligible"
    assert config["cost_configs"][0]["impact_model"] == "linear_proxy"
    assert "execution_config" not in summary
    assert "tradability_config" not in summary
    assert "cost_configs" not in summary
    assert "weight_configs" not in summary


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
        diagnostics_spec=DiagnosticsSpec.from_request(
            {"enabled": True, "materialized_views": ["weights"]}
        ),
    )

    result = run_evaluation(spec)

    first_day = result.metrics["weights"][
        result.metrics["weights"]["timestamp"] == pd.Timestamp("2026-01-01")
    ]
    assert first_day.set_index("asset")["target_weight"].to_dict() == {
        "A": -1.0,
        "D": 1.0,
    }


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

    neutralization = result.config["evaluation_spec"]["neutralization_config"]
    assert neutralization["method"] == "demean_by_group"
    assert neutralization["group_columns"] == ("industry",)
    assert "neutralization_config" not in result.summary


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
        metadata=_prediction_metadata("prediction_segment_run"),
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

    returns = result.metrics["returns"]
    first_by_split = returns.sort_values("timestamp").groupby("split").first()
    assert first_by_split.loc["validation", "turnover"] == pytest.approx(2.0)
    assert first_by_split.loc["test", "turnover"] == pytest.approx(2.0)


def test_segmented_combinations_use_aggregate_summary_metrics(tmp_path):
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
        metadata=_prediction_metadata("prediction_segment_summary_run"),
        n_groups=2,
        weight_configs=(
            WeightConfig(
                method="equal_weight_quantile",
                direction="long_high",
                top_quantile=0.5,
                bottom_quantile=0.5,
                cost_bps=10.0,
            ),
        ),
        segment_columns=("split",),
        output_dir=tmp_path,
    )

    run_evaluation(spec)
    combinations = pd.read_csv(tmp_path / "dimensions" / "combinations.csv")
    evaluation_metrics = pd.read_csv(
        tmp_path / "metrics" / "evaluation_metrics_summary.csv"
    )

    assert len(combinations) == 1
    metric_values = evaluation_metrics.set_index("metric_name")["metric_value"]
    assert "final_cumulative_net_return" in metric_values
    assert "total_cost" in metric_values
    assert "mean_ic" in metric_values


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
        "domain": "evaluation_synthetic",
        "data_stage": "synthetic_only",
        "uses_external_data": False,
    }


def _prediction_metadata(run_id: str) -> dict[str, object]:
    return {
        **_metadata(run_id),
        "kind": "prediction",
        "prediction_handoff_version": "n6_mvs_v1",
        "prediction_columns": ["prediction"],
        "label_columns": ["forward_return"],
        "prediction_time_semantics": {
            "decision_time_column": "timestamp",
            "as_of_time_columns": {"prediction": "timestamp"},
        },
        "source_ref": "synthetic_prediction_panel_v1",
        "model_ref": "opaque_model_ref_v1",
        "training_scope": "external_or_upstream_not_run_by_qrp",
        "prediction_boundary": (
            "prediction evidence is not trading, not risk, not capacity, "
            "not execution simulation, not paper trading, and not live trading "
            "evidence"
        ),
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
