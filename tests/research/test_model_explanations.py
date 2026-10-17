"""Real eleven-model prediction, explanation, persistence and PIT regressions."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.explanations import read_explanation
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models.explanations import explain
from tests.research.direct_model_test_support import fit_frame, predict_frame

_ALGORITHMS = {
    "linear_regression": "affine",
    "ridge_regression": "affine",
    "lasso_regression": "affine",
    "huber_regression": "affine",
    "decision_tree_regressor": "tree_shap_path",
    "random_forest_regressor": "tree_shap_path",
    "gradient_boosting_regressor": "tree_shap_path",
    "xgboost_regressor": "native_tree",
    "lightgbm_regressor": "native_tree",
    "equal_weight_rank_ensemble": "rank",
    "ic_weighted_rank_ensemble": "rank",
}


def _case(tmp_path, algorithm="linear_regression", *, parameters=None):
    rng = np.random.default_rng(18)
    dates = pd.date_range("2020-01-01", periods=10).repeat(8)
    x, y = rng.normal(size=(2, len(dates)))
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": list("abcdefgh") * 10,
            "x": x + 10,
            "y": y - 8,
            "label": 0.02 * x - 0.04 * y,
            "eligible": True,
            "label_available_at": dates,
        }
    )
    params = {
        "lasso_regression": {"alpha": 0.001},
        "huber_regression": {"max_iter": 1000},
        "decision_tree_regressor": {"max_depth": 3},
        "random_forest_regressor": {"n_estimators": 4, "max_depth": 3, "n_jobs": 1},
        "gradient_boosting_regressor": {"n_estimators": 4, "max_depth": 2},
        "xgboost_regressor": {"n_estimators": 4, "max_depth": 2, "n_jobs": 1},
        "lightgbm_regressor": {
            "n_estimators": 4,
            "max_depth": 2,
            "n_jobs": 1,
            "verbosity": -1,
            "min_child_samples": 2,
        },
    }.get(algorithm, {})
    params.update(parameters or {})
    if params.get("objective") == "reg:logistic":
        frame["label"] = 1.0 / (1.0 + np.exp(-frame["label"]))
    spec = ModelSpec(
        model_id=algorithm,
        model_algorithm=algorithm,
        input_feature_columns=("x", "y"),
        target_column="label",
        random_seed=19,
        hyperparameters=params,
    )
    case = fit_frame(
        tmp_path / algorithm, frame.loc[frame.timestamp.le("2020-01-05")], spec
    )
    future = tmp_path / "all.parquet"
    frame.to_parquet(future, index=False)
    declared = copy.deepcopy(case.config["input_spec"])
    declared["column_units"] = {"label": "decimal_return"}
    config = {
        "model": str(case.root),
        "inputs": str(future),
        "input_spec": declared,
        "output": str(tmp_path / "explained"),
        "window": {"start": "2020-01-06", "end": "2020-01-10"},
        "sample_role": "test",
        "methods": ["native_importance", _ALGORITHMS[algorithm]],
        "selection": {"max_rows": 24, "max_dates": 3, "root_seed": 17},
        "output_space": "model_output",
    }
    return case, frame, config


@pytest.mark.parametrize("algorithm", list(_ALGORITHMS))
def test_real_eleven_models_roundtrip_matches_production_prediction(
    tmp_path, algorithm
):
    case, frame, config = _case(tmp_path, algorithm)
    predictions, _, _ = predict_frame(
        tmp_path / "predict", frame, case, window=config["window"]
    )
    metadata, tables = read_explanation(explain(config))
    actual = tables["predictions"].merge(
        predictions,
        on=["timestamp", "asset_id"],
        suffixes=("", "_production"),
        validate="one_to_one",
    )
    assert len(actual) == 24
    np.testing.assert_allclose(
        actual.prediction_score,
        actual.prediction_score_production,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        actual.explained_output, actual.prediction_score, rtol=1e-12, atol=1e-14
    )
    assert actual.additivity_residual.abs().le(actual.additivity_tolerance).all()
    assert metadata["attribution"]["method"] == _ALGORITHMS[algorithm]
    assert metadata["context_row_count"] == 24
    assert set(tables["summary"].period) == {"all", "2020-01"}
    before = Path(config["model"], "predictor.pkl").read_bytes()
    with pytest.raises(ValueError, match="output_already_exists"):
        explain(config)
    assert Path(config["model"], "predictor.pkl").read_bytes() == before


@pytest.mark.parametrize("algorithm", list(_ALGORITHMS)[:4])
def test_linear_shap_uses_exact_explicit_historical_mean(tmp_path, algorithm):
    case, frame, config = _case(tmp_path, algorithm)
    config["methods"] = ["linear_shap"]
    config["background"] = {
        "inputs": config["inputs"],
        "input_spec": config["input_spec"],
        "window": {"start": "2020-01-01", "end": "2020-01-05"},
        "max_rows": 40,
        "root_seed": 1,
    }
    metadata, tables = read_explanation(explain(config))
    predictor = case.result.prediction_model
    matrix = predictor.scaler.transform(
        frame.loc[frame.timestamp.le("2020-01-05"), ["x", "y"]]
    )
    coef = np.asarray(predictor.coef_)
    center = matrix.mean().to_numpy()
    expected_baseline = (
        predictor.coefficients[0]
        if algorithm == "linear_regression"
        else predictor.estimator.intercept_
    ) + center @ coef
    np.testing.assert_allclose(
        tables["predictions"].base_value, expected_baseline, rtol=1e-12, atol=1e-14
    )
    assert metadata["attribution"]["background"]["kind"] == "historical_reference"
    assert metadata["attribution"]["background"]["root_seed"] == 1
    assert (
        metadata["attribution"]["background"]["training_sample_reconstructed"] is False
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        "decision_tree_regressor",
        "random_forest_regressor",
        "gradient_boosting_regressor",
    ],
)
def test_real_tree_interventional_shap(tmp_path, algorithm):
    _, _, config = _case(tmp_path, algorithm)
    config["methods"] = ["tree_shap_interventional"]
    config["background"] = {
        "inputs": config["inputs"],
        "input_spec": config["input_spec"],
        "window": {"start": "2020-01-01", "end": "2020-01-05"},
        "max_rows": 20,
        "root_seed": 1,
    }
    metadata, tables = read_explanation(explain(config))
    assert metadata["attribution"]["background"]["row_count"] == 20
    assert (
        tables["predictions"]
        .additivity_residual.abs()
        .le(tables["predictions"].additivity_tolerance)
        .all()
    )


@pytest.mark.parametrize(
    "algorithm", ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"]
)
def test_rank_keeps_full_context_before_sampling_and_frozen_calibration(
    tmp_path, algorithm
):
    case, _, config = _case(tmp_path, algorithm)
    config["selection"].update(max_rows=2, asset_ids=["a"])
    metadata, tables = read_explanation(explain(config))
    assert metadata["context_row_count"] == 24
    assert len(tables["predictions"]) == 2
    assert tables["predictions"].asset_id.eq("a").all()
    config["output"] += "-calibrated"
    config["output_space"] = "calibrated_label"
    calibrated_meta, calibrated = read_explanation(explain(config))
    intercept, slope = case.result.prediction_model.label_calibration
    np.testing.assert_allclose(
        calibrated["contributions"].contribution,
        tables["contributions"].contribution * slope,
    )
    np.testing.assert_allclose(calibrated["predictions"].base_value, intercept)
    assert calibrated_meta["unit"] == "decimal_return"
    assert metadata["unit"] == "dimensionless_rank_score"


def test_native_importance_needs_no_inputs_shap_or_fitting(tmp_path, monkeypatch):
    import quant_research_platform.research.models.explanations as owner

    case, _, _ = _case(tmp_path)
    monkeypatch.setattr(owner, "_session", lambda *args, **kw: pytest.fail("read X"))
    metadata, tables = read_explanation(
        explain(
            {
                "model": str(case.root),
                "output": str(tmp_path / "native"),
                "methods": ["native_importance"],
            }
        )
    )
    assert set(tables) == {"native_importance"}
    assert (
        tables["native_importance"].set_index("feature").value.to_dict()
        == case.result.metadata["feature_importance"]
    )
    assert "attribution" not in metadata


def test_background_future_gap_and_assets_are_rejected(tmp_path):
    case, _, config = _case(tmp_path)
    config["methods"] = ["linear_shap"]
    config["background"] = {
        "inputs": config["inputs"],
        "input_spec": config["input_spec"],
        "window": {"start": "2020-01-01", "end": "2020-01-06"},
        "max_rows": 20,
        "root_seed": 1,
    }
    with pytest.raises(ValueError, match="after_fit_cutoff"):
        explain(config)
    config["background"]["window"]["end"] = "2020-01-05"
    source = pd.read_parquet(config["inputs"])
    source.loc[0, "asset_id"] = "outside_training"
    source.to_parquet(config["inputs"], index=False)
    with pytest.raises(ValueError, match="outside_fit_domain"):
        explain(config)
    manifest = case.root / "model.json"
    payload = json.loads(manifest.read_text())
    payload["facts"]["fit_windows"] = [
        {"start": "2020-01-01", "end": "2020-01-02"},
        {"start": "2020-01-04", "end": "2020-01-05"},
    ]
    manifest.write_text(json.dumps(payload))
    source.loc[0, "asset_id"] = "a"
    source.to_parquet(config["inputs"], index=False)
    with pytest.raises(ValueError, match="outside_fit_domain"):
        explain(config)


def _permutation(config, *, metrics=None):
    config["methods"] = ["permutation"]
    config["permutation"] = {
        "metrics": metrics or ["ic", "rankic", "mse"],
        "cutoff": "2020-01-10",
        "axis": "within_timestamp",
        "groups": {"both": ["x", "y"]},
        "repeats": 3,
        "root_seed": 12,
    }
    return config


def test_permutation_maturity_grouped_values_determinism_and_units(tmp_path):
    _, frame, config = _case(tmp_path)
    config = _permutation(config)
    frame.loc[frame.asset_id.eq("a"), "label_available_at"] = pd.Timestamp("2030-01-01")
    frame.to_parquet(config["inputs"], index=False)
    metadata, tables = read_explanation(explain(config))
    assert metadata["context_row_count"] == 24
    assert metadata["permutation"]["mature_scoring_rows"] == 21
    assert tables["permutation_summary"].observation_count.eq(21).all()
    assert tables["permutation_summary"].delta_mean.gt(0).all()
    config["output"] += "-repeat"
    _, repeated = read_explanation(explain(config))
    pd.testing.assert_frame_equal(
        tables["permutation_repeats"], repeated["permutation_repeats"]
    )
    config["output"] += "-unit"
    config["input_spec"]["column_units"]["label"] = "percent_return"
    _, mismatched = read_explanation(explain(config))
    rows = mismatched["permutation_repeats"]
    mse = rows.loc[rows.metric.eq("mse")]
    assert mse.status.eq("UNDEFINED").all()
    assert mse.reason.eq("prediction_target_unit_mismatch").all()
    assert mse[["baseline", "perturbed", "delta"]].isna().all().all()
    assert rows.loc[~rows.metric.eq("mse"), "status"].eq("DEFINED").all()


def test_permutation_requires_calibrated_rank_mse(tmp_path):
    _, _, config = _case(tmp_path, "equal_weight_rank_ensemble")
    _permutation(config)
    with pytest.raises(ValueError, match="mse_requires_label_output"):
        explain(config)
    config["output_space"] = "calibrated_label"
    _, tables = read_explanation(explain(config))
    assert tables["permutation_summary"].status.eq("DEFINED").all()


@pytest.mark.parametrize("step_seconds", [86400, 0.1])
def test_temporal_blocks_split_episodes_and_reject_gaps(tmp_path, step_seconds):
    _, frame, config = _case(tmp_path)
    frame = frame.loc[frame.asset_id.eq("a")].copy()
    frame["episode"] = "first"
    future = frame.timestamp.ge("2020-01-06")
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns]")
    frame.loc[future, "timestamp"] = pd.date_range(
        "2020-01-06",
        periods=int(future.sum()),
        freq=pd.Timedelta(seconds=step_seconds),
    ).as_unit("ns")
    frame["label_available_at"] = frame.timestamp
    frame.to_parquet(config["inputs"], index=False)
    config["input_spec"]["column_roles"]["episode"] = "context"
    config["selection"].update(max_dates=None, max_rows=5)
    config["methods"] = ["permutation"]
    config["permutation"] = {
        "metrics": ["ic", "mse"],
        "cutoff": "2020-01-10",
        "axis": "temporal_blocks",
        "groups": {"both": ["x", "y"]},
        "repeats": 3,
        "root_seed": 12,
        "episode_column": "episode",
        "block_size": 2,
        "step_seconds": step_seconds,
    }
    metadata, tables = read_explanation(explain(config))
    assert metadata["permutation"]["config"]["step_seconds"] == step_seconds
    assert tables["permutation_summary"].period_count.eq(1).all()
    config["output"] += "-gap"
    frame = frame.drop(frame.index[7])
    frame.to_parquet(config["inputs"], index=False)
    with pytest.raises(ValueError, match="episode_too_short"):
        explain(config)


def test_nonidentity_xgboost_raw_margin_units_and_no_silent_fallback(tmp_path):
    _, _, config = _case(
        tmp_path, "xgboost_regressor", parameters={"objective": "reg:logistic"}
    )
    with pytest.raises(ValueError, match="nonidentity_link"):
        explain(config)
    config["output_space"] = "raw_margin"
    metadata, tables = read_explanation(explain(config))
    assert metadata["unit"] == "raw_margin"
    assert len(tables["predictions"]) == 24
    assert not np.allclose(
        tables["predictions"].explained_output, tables["predictions"].prediction_score
    )


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"methods": ["rank"]}, "rank_model_required"),
        ({"methods": ["affine", "linear_shap"]}, "single_attribution"),
        ({"output_space": "prediction_rank_score"}, "output_space_invalid"),
        ({"unconsumed": 1}, "config_fields_invalid"),
    ],
)
def test_reject_unsupported_or_unused_options(tmp_path, change, reason):
    _, _, config = _case(tmp_path)
    config.update(change)
    with pytest.raises(ValueError, match=reason):
        explain(config)


def test_rank_hand_calculation_ties_missing_values_and_single_asset(tmp_path):
    case, frame, config = _case(tmp_path, "equal_weight_rank_ensemble")
    future = frame.loc[
        frame.timestamp.eq("2020-01-06") & frame.asset_id.isin(["a", "b", "c"])
    ].copy()
    future["x"] = [1.0, 1.0, 3.0]
    future["y"] = [np.nan, 4.0, 2.0]
    future.to_parquet(config["inputs"], index=False)
    config["selection"].update(asset_ids=["a"], max_rows=1)
    _, tables = read_explanation(explain(config))
    weights = np.asarray(case.result.prediction_model.directions) / 2
    actual = tables["contributions"].set_index("feature").contribution
    assert actual["x"] == pytest.approx(-weights[0] / 6)
    assert actual["y"] == 0
    config["output"] += "-single"
    future.iloc[:1].to_parquet(config["inputs"], index=False)
    _, single = read_explanation(explain(config))
    assert single["contributions"].contribution.eq(0).all()
    assert single["summary"].absolute_share.isna().all()


def test_ic_rank_undefined_ic_feature_retains_zero_importance(tmp_path):
    _, frame, _ = _case(tmp_path)
    frame["y"] = 1.0
    spec = ModelSpec(
        model_id="ic",
        model_algorithm="ic_weighted_rank_ensemble",
        input_feature_columns=("x", "y"),
        target_column="label",
        random_seed=19,
    )
    case = fit_frame(tmp_path / "ic", frame.loc[frame.timestamp.le("2020-01-05")], spec)
    source = tmp_path / "constant.parquet"
    frame.to_parquet(source, index=False)
    config = {
        "model": str(case.root),
        "inputs": str(source),
        "input_spec": case.config["input_spec"],
        "output": str(tmp_path / "excluded"),
        "methods": ["native_importance", "rank"],
        "window": {"start": "2020-01-06", "end": "2020-01-10"},
        "sample_role": "test",
        "selection": {"max_rows": 24, "max_dates": 3, "root_seed": 17},
        "output_space": "model_output",
    }
    _, tables = read_explanation(explain(config))
    excluded = tables["native_importance"].set_index("feature").loc["y"]
    assert excluded.status == "DEFINED"
    assert excluded.value == 0.0
    assert (
        tables["contributions"]
        .loc[lambda x: x.feature.eq("y"), "contribution"]
        .eq(0)
        .all()
    )


def test_only_selected_metrics_run_on_extreme_scale(tmp_path):
    _, frame, _ = _case(tmp_path)
    frame["label"] *= 1e200
    spec = ModelSpec(
        model_id="large",
        model_algorithm="linear_regression",
        input_feature_columns=("x", "y"),
        target_column="label",
        random_seed=19,
    )
    case = fit_frame(
        tmp_path / "large", frame.loc[frame.timestamp.le("2020-01-05")], spec
    )
    source = tmp_path / "large.parquet"
    frame.to_parquet(source, index=False)
    declared = copy.deepcopy(case.config["input_spec"])
    declared["column_units"] = {"label": "decimal_return"}
    config = {
        "model": str(case.root),
        "inputs": str(source),
        "input_spec": declared,
        "output": str(tmp_path / "large-explained"),
        "methods": ["permutation"],
        "window": {"start": "2020-01-06", "end": "2020-01-10"},
        "sample_role": "test",
        "selection": {"max_rows": 24, "max_dates": 3, "root_seed": 17},
        "output_space": "model_output",
    }
    _permutation(config, metrics=["ic", "rankic"])
    _, tables = read_explanation(explain(config))
    assert tables["permutation_summary"].status.eq("DEFINED").all()
    config["output"] += "-mse"
    config["permutation"]["metrics"] = ["mse"]
    with pytest.raises(ValueError, match="mse_unrepresentable"):
        explain(config)
    assert not Path(config["output"]).exists()


def test_explanation_without_labels_and_no_implicit_fit(tmp_path, monkeypatch):
    import quant_research_platform.research.models.fitting as fitting

    _, frame, config = _case(tmp_path)
    monkeypatch.setattr(
        fitting,
        "_fit_model_result",
        lambda *a, **kw: pytest.fail("explain must not fit"),
    )
    frame = frame.drop(columns=["label", "label_available_at"])
    for column in ("label", "label_available_at"):
        config["input_spec"]["column_roles"].pop(column)
    config["input_spec"].pop("column_units")
    frame.to_parquet(config["inputs"], index=False)
    _, tables = read_explanation(explain(config))
    assert len(tables["predictions"]) == 24


def test_read_rejects_tampered_contribution_additivity(tmp_path):
    _, _, config = _case(tmp_path)
    output = explain(config)
    path = output / "contributions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "contribution"] += 1
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    pq.write_table(
        pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path
    )
    with pytest.raises(ValueError, match="additivity_failed"):
        read_explanation(output)


def test_cli_is_same_explain_operation(tmp_path):
    from typer.testing import CliRunner

    from quant_research_platform.interfaces.cli.commands import app

    case, _, _ = _case(tmp_path)
    config = {
        "model": str(case.root),
        "output": str(tmp_path / "cli"),
        "methods": ["native_importance"],
    }
    path = tmp_path / "explain.json"
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(app, ["explain", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert set(read_explanation(Path(config["output"]))[1]) == {"native_importance"}


def test_finite_prediction_survives_unrepresentable_contribution_and_subnormal(
    tmp_path,
):
    from quant_research_platform.research.models.fitting import ModelFitResult
    from quant_research_platform.research.models.persisted import save_model
    from quant_research_platform.research.models.scaler import (
        LinearScaledPredictor,
        ModelMatrixScaler,
    )

    case, _, config = _case(tmp_path)
    scaler = ModelMatrixScaler.fit(pd.DataFrame({"x": [-1.0, 1.0], "y": [-1.0, 1.0]}))
    predictor = LinearScaledPredictor((1.0, 1e308, -1e308), scaler)
    metadata = copy.deepcopy(case.result.metadata)
    metadata.update(
        scaler=scaler.to_payload(), feature_importance={"x": 1e308, "y": 1e308}
    )
    result = ModelFitResult(metadata, predictor)
    model = save_model(
        tmp_path / "numeric-model",
        result,
        facts=case.facts,
        trial={
            "status": "completed",
            "trial_id": "numeric",
            "result": {"model_metadata": metadata},
        },
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
            "asset_id": ["a"] * 3,
            "x": [10.0, 1.0, 1e-320],
            "y": [10.0, 1.0, 0.0],
            "label": [0.0] * 3,
            "label_available_at": pd.to_datetime(
                ["2020-01-06", "2020-01-07", "2020-01-08"]
            ),
            "eligible": True,
        }
    )
    frame.to_parquet(config["inputs"], index=False)
    config.update(
        model=str(model),
        methods=["affine"],
        selection={"max_rows": 3, "max_dates": None, "root_seed": 1},
    )
    _, tables = read_explanation(explain(config))
    actual = tables["predictions"].sort_values("timestamp")
    assert actual.status.tolist() == ["UNDEFINED", "DEFINED", "DEFINED"]
    assert actual.reason.iloc[0] == "nonrepresentable_contribution_or_baseline"
    np.testing.assert_array_equal(
        actual.prediction_score, predictor.predict(frame[["x", "y"]])
    )
    assert actual.prediction_score.iloc[0] == 1.0
    subnormal = (
        tables["contributions"]
        .loc[
            lambda rows: rows.timestamp.eq("2020-01-08") & rows.feature.eq("x"),
            "contribution",
        ]
        .iloc[0]
    )
    assert subnormal == pytest.approx(1e308 * 1e-320, rel=1e-12)
    summary = tables["summary"].loc[lambda rows: rows.period.eq("all")]
    assert summary.sample_count.eq(3).all()
    assert summary.observation_count.eq(2).all()
    assert np.isfinite(summary.mean_abs_contribution).all()
    assert actual.additivity_tolerance.dropna().map(np.isfinite).all()


def test_shap_nonfinite_row_is_explicit_without_hiding_other_predictions(
    tmp_path, monkeypatch
):
    import shap

    _, _, config = _case(tmp_path, "decision_tree_regressor")
    original = shap.TreeExplainer.__call__

    def partially_nonfinite(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result.values[0, 0] = np.nan
        return result

    monkeypatch.setattr(shap.TreeExplainer, "__call__", partially_nonfinite)
    _, tables = read_explanation(explain(config))
    assert tables["predictions"].status.value_counts().to_dict() == {
        "DEFINED": 23,
        "UNDEFINED": 1,
    }
    assert tables["predictions"].prediction_score.notna().all()
    assert (
        tables["summary"]
        .loc[lambda frame: frame.period.eq("all"), "observation_count"]
        .eq(23)
        .all()
    )


def test_date_only_background_end_cannot_admit_after_exact_fit_cutoff(tmp_path):
    _, frame, config = _case(tmp_path)
    after = frame.iloc[:1].copy()
    after["timestamp"] = pd.Timestamp("2020-01-05T12:00:00")
    pd.concat([frame, after], ignore_index=True).sort_values(
        ["timestamp", "asset_id"]
    ).to_parquet(config["inputs"], index=False)
    config.update(
        methods=["linear_shap"],
        background={
            "inputs": config["inputs"],
            "input_spec": config["input_spec"],
            "window": {"start": "2020-01-01", "end": "2020-01-05"},
            "max_rows": 100,
            "root_seed": 1,
        },
    )
    with pytest.raises(ValueError, match="outside_fit_domain"):
        explain(config)


def test_large_finite_permutation_mse_keeps_finite_repeat_dispersion(tmp_path):
    _, frame, config = _case(tmp_path)
    frame["label"] *= 1e150
    case = fit_frame(
        tmp_path / "large-mse",
        frame.loc[frame.timestamp.le("2020-01-05")],
        ModelSpec(
            model_id="large",
            model_algorithm="linear_regression",
            input_feature_columns=("x", "y"),
            target_column="label",
            random_seed=19,
        ),
    )
    frame.to_parquet(config["inputs"], index=False)
    config["model"] = str(case.root)
    _permutation(config, metrics=["mse"])
    _, tables = read_explanation(explain(config))
    summary = tables["permutation_summary"].iloc[0]
    assert np.isfinite(summary.delta_std) and summary.delta_std > 0
    deltas = tables["permutation_repeats"].delta.to_numpy()
    scale = np.max(np.abs(deltas))
    assert summary.delta_std / scale == pytest.approx((deltas / scale).std(ddof=0))


@pytest.mark.parametrize(
    "mutation", ["unknown_method", "missing_unit", "missing_space", "baseline"]
)
def test_read_rejects_missing_or_relabelled_scientific_metadata(tmp_path, mutation):
    _, _, config = _case(tmp_path)
    output = explain(config)
    path = output / "metadata.json"
    payload = json.loads(path.read_text())
    if mutation == "unknown_method":
        payload["methods"] = ["unknown_method"]
    elif mutation == "missing_unit":
        payload.pop("unit")
    elif mutation == "missing_space":
        payload.pop("output_space")
    else:
        payload["attribution"]["baseline"] = "invented"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="metadata_invalid"):
        read_explanation(output)


def test_summary_retains_minimum_subnormal_contributions():
    from quant_research_platform.research.models.explanations import _summaries

    minimum = np.nextafter(0.0, 1.0)
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "asset_id": ["a", "a"],
            "feature": ["x", "x"],
            "contribution": [minimum, minimum],
        }
    )
    summary = _summaries(rows, "month")
    assert summary.mean_abs_contribution.eq(minimum).all()
    assert summary.mean_signed_contribution.eq(minimum).all()
    assert summary.absolute_share.eq(1).all()


def test_cancellation_loss_cannot_be_certified_by_large_contribution_tolerance(
    tmp_path,
):
    from fractions import Fraction

    from quant_research_platform.research.models.fitting import ModelFitResult
    from quant_research_platform.research.models.persisted import save_model
    from quant_research_platform.research.models.scaler import (
        LinearScaledPredictor,
        ModelMatrixScaler,
    )

    case, frame, config = _case(tmp_path)
    scaler = ModelMatrixScaler(
        ("x", "y"), (0.0, 0.0), (Fraction(), Fraction()), (1.0, 1.0)
    )
    predictor = LinearScaledPredictor((1.0, 1e308, -1e308), scaler)
    metadata = copy.deepcopy(case.result.metadata)
    metadata.update(
        scaler=scaler.to_payload(), feature_importance={"x": 1e308, "y": 1e308}
    )
    result = ModelFitResult(metadata, predictor)
    model = save_model(
        tmp_path / "cancellation",
        result,
        facts=case.facts,
        trial={
            "status": "completed",
            "trial_id": "numeric",
            "result": {"model_metadata": metadata},
        },
    )
    frame = frame.iloc[:1].copy()
    frame["timestamp"] = pd.Timestamp("2020-01-06")
    frame[["x", "y"]] = 1.0000000000000002
    frame.to_parquet(config["inputs"], index=False)
    config.update(model=str(model), methods=["affine"])
    _, tables = read_explanation(explain(config))
    row = tables["predictions"].iloc[0]
    # BLAS implementations may cancel exactly, but a nonzero output-scale loss
    # must never be certified by a contribution-scale backward error alone.
    actual = predictor.predict(frame[["x", "y"]])[0]
    assert row.prediction_score == actual
    if actual != 1.0:
        assert row.status == "UNDEFINED"
        assert row.reason == "cancellation_precision_loss"
        assert tables["contributions"].contribution.isna().all()
        assert tables["summary"].observation_count.eq(0).all()


def test_read_binds_explanation_method_to_actual_model_kind(tmp_path):
    _, _, config = _case(tmp_path, "equal_weight_rank_ensemble")
    output = explain(config)
    path = output / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(model_algorithm="linear_regression", unit=metadata["label_unit"])
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="rank_model_required"):
        read_explanation(output)


@pytest.mark.parametrize(
    "member,column",
    [
        ("native_importance", "unit"),
        ("native_importance", "kind"),
        ("native_importance", "reason"),
        ("permutation_summary", "unit"),
        ("permutation_repeats", "reason"),
    ],
)
def test_artifact_requires_string_units_and_explicit_reasons(tmp_path, member, column):
    from quant_research_platform.artifacts.explanations import write_explanation

    _, _, config = _case(tmp_path)
    if member.startswith("permutation"):
        _permutation(config)
    metadata, tables = read_explanation(explain(config))
    tables[member].loc[0, column] = None
    with pytest.raises(ValueError, match="table_string_invalid"):
        write_explanation(tmp_path / "invalid", metadata=metadata, tables=tables)


def test_xgboost_squared_log_error_still_has_identity_prediction_output(tmp_path):
    _, _, config = _case(
        tmp_path, "xgboost_regressor", parameters={"objective": "reg:squaredlogerror"}
    )
    metadata, tables = read_explanation(explain(config))
    assert metadata["unit"] == "decimal_return"
    np.testing.assert_array_equal(
        tables["predictions"].explained_output, tables["predictions"].prediction_score
    )
    assert tables["predictions"].status.eq("DEFINED").all()


@pytest.mark.parametrize("step_seconds", [0.1, 0.000000001, 1, 1.5])
def test_temporal_permutation_exact_subsecond_continuity(step_seconds):
    from quant_research_platform.research.models.explanations import (
        _permutation_config,
        _permutation_segments,
    )

    config = _permutation_config(
        {
            "metrics": ["ic"],
            "cutoff": "2020-01-10",
            "axis": "temporal_blocks",
            "groups": {"x": ["x"]},
            "repeats": 1,
            "root_seed": 12,
            "episode_column": "episode",
            "block_size": 2,
            "step_seconds": step_seconds,
        },
        ["x"],
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2020-01-06", periods=6, freq=pd.Timedelta(seconds=step_seconds)
            ).as_unit("ns"),
            "asset_id": "a",
            "episode": "first",
        }
    )
    assert [part.tolist() for part in _permutation_segments(frame, config)] == [
        list(range(6))
    ]
    frame.loc[3:, "timestamp"] += pd.Timedelta(1, unit="ns")
    assert [part.tolist() for part in _permutation_segments(frame, config)] == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    frame.loc[1, "timestamp"] += pd.Timedelta(1, unit="ns")
    with pytest.raises(ValueError, match="episode_too_short"):
        _permutation_segments(frame, config)


@pytest.mark.parametrize(
    "step", [True, "0.1", 0, -0.1, np.nan, np.inf, 1e-10, 0.1000000001]
)
def test_temporal_permutation_rejects_unrepresentable_steps(step):
    from quant_research_platform.research.models.explanations import _permutation_config

    with pytest.raises(
        ValueError, match="step_seconds_invalid|qrp_json_nonfinite_number"
    ):
        _permutation_config(
            {
                "metrics": ["ic"],
                "cutoff": "2020-01-10",
                "axis": "temporal_blocks",
                "groups": {"x": ["x"]},
                "repeats": 1,
                "root_seed": 12,
                "episode_column": "episode",
                "block_size": 2,
                "step_seconds": step,
            },
            ["x"],
        )


def test_explanation_default_seed_and_stable_permutation_streams(tmp_path):
    _, _, config = _case(tmp_path)
    _permutation(config, metrics=["mse"])
    config["selection"].pop("root_seed")
    config["permutation"].pop("root_seed")
    config["permutation"]["groups"] = {"x": ["x"], "y": ["y"]}
    state = np.random.get_state()
    metadata, first = read_explanation(explain(config))
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert metadata["selection"]["root_seed"] == 42
    assert metadata["permutation"]["config"]["root_seed"] == 42
    config["output"] += "-explicit"
    config["selection"]["root_seed"] = 42
    config["permutation"]["root_seed"] = 42
    _, explicit = read_explanation(explain(config))
    for name in first:
        pd.testing.assert_frame_equal(first[name], explicit[name])
    config["output"] += "-reordered-extended"
    config["permutation"]["groups"] = {"y": ["y"], "x": ["x"]}
    config["permutation"]["repeats"] = 5
    _, extended = read_explanation(explain(config))
    keys = ["group", "repeat", "metric"]
    expected = first["permutation_repeats"].sort_values(keys).reset_index(drop=True)
    actual = extended["permutation_repeats"]
    actual = actual.loc[actual["repeat"].lt(3)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


@pytest.mark.parametrize("section", ["selection", "permutation"])
def test_explanation_explicit_null_seed_is_rejected(tmp_path, section):
    _, _, config = _case(tmp_path)
    _permutation(config)
    config[section]["root_seed"] = None
    with pytest.raises(ValueError, match="explanation_seed_invalid"):
        explain(config)


def test_background_seed_default_matches_explicit_and_rejects_null(tmp_path):
    _, _, config = _case(tmp_path)
    config["methods"] = ["linear_shap"]
    config["background"] = {
        "inputs": config["inputs"],
        "input_spec": config["input_spec"],
        "window": {"start": "2020-01-01", "end": "2020-01-05"},
        "max_rows": 10,
    }
    metadata, first = read_explanation(explain(config))
    assert metadata["attribution"]["background"]["root_seed"] == 42
    config["output"] += "-explicit"
    config["background"]["root_seed"] = 42
    _, second = read_explanation(explain(config))
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    config["output"] += "-null"
    config["background"]["root_seed"] = None
    with pytest.raises(ValueError, match="explanation_seed_invalid"):
        explain(config)
