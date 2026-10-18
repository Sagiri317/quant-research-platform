"""Frozen training-domain baselines through native model and prediction artifacts."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.models import (
    ModelSpec,
    TrainingWeighting,
    validate_model_metadata_core,
)
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.statistics_production import (
    _prediction_quality,
    model_statistics,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.research.direct_model_test_support import fit_frame
from tests.research.test_published_model_metric_semantics import (
    _case,
    _defined_target_case,
)


@pytest.mark.parametrize("algorithm", ["linear_regression", "decision_tree_regressor"])
@pytest.mark.parametrize("half_life, expected", [(None, 4.0), (1, 14 / 3)])
def test_mean_uses_only_actual_training_rows_and_weights(
    tmp_path, algorithm, half_life, expected
):
    dates = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 4)
    frame = pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id=["a", "b", "c", "a", "b", "c", "d"],
            market_type=1,
            x=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            label=[2.0, 2.0, 2.0, 6.0, 1000.0, 1000.0, 1000.0],
            eligible=[True, True, True, True, False, True, True],
            label_available_at=[*dates[:5], pd.Timestamp("2020-01-03"), dates[-1]],
        )
    )
    case = fit_frame(
        tmp_path,
        frame,
        ModelSpec(
            model_id="mean",
            model_algorithm=algorithm,
            input_factor_columns=("x",),
            target_column="label",
            training_weighting=TrainingWeighting(half_life_periods=half_life),
        ),
        trading_calendar=(
            pd.DataFrame(
                dict(
                    calendar_date=pd.date_range("2020-01-01", periods=2),
                    source_market_type=1,
                    is_trading_day=True,
                )
            )
            if half_life is not None
            else None
        ),
    )
    assert case.result.metadata["rows_after_input_label_dropna"] == 4
    assert case.result.metadata["training_target_mean"] == pytest.approx(expected)


def test_period_specific_frozen_means_survive_changed_evaluation_targets(tmp_path):
    first, _, _, first_meta, config = _case(tmp_path / "first")
    second, _, _, second_meta, other = _case(
        tmp_path / "second", intercept=1.5, slope=3.0
    )
    assert first_meta["model_metadata"]["training_target_mean"] == pytest.approx(0.1)
    assert second_meta["model_metadata"]["training_target_mean"] == pytest.approx(1.5)
    labels = pd.concat(
        [
            first.loc[first.timestamp.eq("2020-01-03")],
            second.loc[second.timestamp.ge("2020-01-04")],
        ]
    )
    labels["label"] += 1.0
    labels.to_parquet(config["dataset"], index=False)
    config["evaluation"]["baseline"] = "training_mean"
    config["periods"][0]["window"] = {"start": "2020-01-03", "end": "2020-01-03"}
    config["periods"].append(
        dict(
            period_id="second",
            window={"start": "2020-01-04", "end": "2020-01-05"},
            predictions=other["periods"][0]["predictions"],
        )
    )
    output = model_statistics(config)
    with read_statistics_artifact(output).table(
        "model_prediction_metrics.parquet"
    ) as metrics:
        expected = [1 + 0.2**2 * 5 / 64, *[1 + 3.0**2 * 5 / 64] * 2]
        assert metrics.baseline_mse.tolist() == pytest.approx(expected)
        assert metrics.mse.tolist() == pytest.approx([1.0] * 3)
        assert metrics.oos_r2.tolist() == pytest.approx(1 - 1 / np.array(expected))
    metadata = json.loads((output / "manifest.json").read_text())["metadata"]
    assert metadata["prediction_metrics"]["baseline"] == "training_mean"
    assert (
        metadata["prediction_metrics"]["baseline_weighting"]
        == "frozen_model_training_weighting"
    )


def test_requested_training_mean_requires_saved_training_evidence(tmp_path):
    _, _, prediction, _, config = _case(tmp_path)
    path = prediction / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["metadata"]["model_metadata"]["training_target_mean"]
    path.write_text(json.dumps(manifest))
    config["evaluation"]["baseline"] = "training_mean"
    with pytest.raises(ValueError, match="statistics_training_target_mean_missing"):
        model_statistics(config)
    ic_only = {**config, "analyses": ["prediction_hac"], "statistics": {"hac_lag": 0}}
    ic_only["output"] = str(tmp_path / "ic_only")
    model_statistics(ic_only)
    config["evaluation"]["baseline"] = "zero_prediction"
    model_statistics(config)


@pytest.mark.parametrize("invalid", [True, "1.0", None])
def test_saved_training_mean_must_be_a_finite_number(tmp_path, invalid):
    _, case, prediction, _, _ = _case(tmp_path)
    with pytest.raises(ValueError, match="invalid_model_training_target_mean"):
        validate_model_metadata_core(
            {**case.result.metadata, "training_target_mean": invalid}
        )
    path = prediction / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["model_metadata"]["training_target_mean"] = invalid
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="prediction_training_target_mean_invalid"):
        read_predictions(prediction)


@pytest.mark.parametrize("baseline", [None, "evaluation_mean", 1, []])
def test_baseline_configuration_rejects_unknown_choices(tmp_path, baseline):
    *_, config = _case(tmp_path)
    config["evaluation"]["baseline"] = baseline
    with pytest.raises(ValueError, match="model_statistics_baseline_invalid"):
        model_statistics(config)


def test_training_mean_zero_error_has_explicit_undefined_r2():
    target = pd.Series([2.0, 2.0])
    result = _prediction_quality(
        target, target, target, head_fraction=1.0, baseline_prediction=2.0
    )
    assert result["baseline_mse"] == 0.0
    assert np.isnan(result["oos_r2"])
    assert result["oos_r2_reason"] == "zero_baseline_error"


@pytest.mark.parametrize("baseline", ["zero_prediction", "training_mean"])
@pytest.mark.parametrize("mismatch", ["unit", "definition"])
def test_training_mean_requires_comparable_target_units_and_definition(
    tmp_path, baseline, mismatch
):
    frame, spec, metadata, model, _, _ = _defined_target_case(tmp_path)
    if mismatch == "unit":
        frame["model_target"] *= 100
        spec = replace(
            spec, column_units={**spec.column_units, "model_target": "percent"}
        )
        reason = "prediction_target_unit_mismatch"
    else:
        spec = replace(spec, target_clip="mad")
        reason = "prediction_target_definition_mismatch"
    source = tmp_path / "evaluation"
    write_dataset(source, frame, spec=spec, metadata=metadata)
    window = {"start": "2020-01-03", "end": "2020-01-05"}
    prediction = predict(
        dict(
            model=str(model),
            inputs=str(source),
            output=str(tmp_path / "pred"),
            prediction_window=window,
            sample_role="validation",
        )
    )
    artifact = read_statistics_artifact(
        model_statistics(
            dict(
                dataset=str(source),
                output=str(tmp_path / "stats"),
                window=window,
                label_cutoff="2020-01-06",
                family_ids=["model"],
                analyses=["model_losses", "prediction_hac"],
                evaluation=dict(primary_label="model_target", baseline=baseline),
                statistics={"hac_lag": 0},
                periods=[
                    dict(period_id="p", window=window, predictions=[str(prediction)])
                ],
            )
        )
    )
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.mse.isna().all()
        assert metrics.oos_r2.isna().all()
        assert metrics.mse_reason.eq(reason).all()
        assert metrics.oos_r2_reason.eq(reason).all()
        assert metrics.baseline_mse.isna().eq(baseline == "training_mean").all()
        assert metrics.head_mean_raw_return.notna().all()
    with artifact.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics["mean"].to_numpy() == pytest.approx(1.0)
