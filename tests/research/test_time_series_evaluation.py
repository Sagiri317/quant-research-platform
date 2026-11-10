"""Daily metrics and common samples from real saved model predictions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.evaluations.time_series import (
    daily_metric_summary,
    daily_signal_metrics,
    evaluate_time_series,
)
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.storage.datasets import read_dataset
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def test_daily_equal_weight_metrics_pairing_and_missing_clock(tmp_path):
    inputs = data(tmp_path)
    frame = pd.read_parquet(inputs)
    # Retain the model relation in three intraday observations of one asset.
    frame["timestamp"] += pd.to_timedelta(
        frame.asset_id.map({"a": 0, "b": 3, "c": 6}), unit="s"
    )
    frame["label_available_at"] = frame.timestamp
    frame["asset_id"] = "a"
    frame["sparse"] = frame.x.where(~frame.timestamp.dt.day.eq(5))
    frame["constant"] = 1.0
    frame.to_parquet(inputs, index=False)
    declaration = input_spec()
    declaration["column_roles"]["sparse"] = "factor"
    declaration["column_roles"]["constant"] = "feature"
    declaration["label_availability_columns"] = {"label": "label_available_at"}
    declaration["column_units"] = {"label": "decimal_return"}
    predictions = {}
    for name, last in (("a", 3), ("b", 2), ("constant", 3)):
        config = fit_config(tmp_path, inputs, model_id=name)
        if name == "constant":
            config["model"]["input_feature_columns"] = ["constant"]
        config.update(input_spec=declaration, fit_cutoff=f"2020-01-0{last}T23:59:59")
        config["fit_windows"][0]["end"] = f"2020-01-0{last}"
        model = fit(config)
        predictions[name] = [
            str(
                predict(
                    {
                        "model": str(model),
                        "inputs": str(inputs),
                        "input_spec": declaration,
                        "prediction_window": {
                            "start": "2020-01-04",
                            "end": "2020-01-06",
                        },
                        "output": str(tmp_path / f"pred-{name}"),
                    }
                )
            )
        ]
    config = {
        "inputs": {"path": str(inputs), "spec": declaration},
        "predictions": predictions,
        "r2_baselines": ["zero", "b"],
        "factor_columns": ["sparse"],
        "label": "label",
        "window": {"start": "2020-01-04", "end": "2020-01-06"},
        "cutoff": "2020-01-06T23:59:59",
        "sessions": ["2020-01-04", "2020-01-05", "2020-01-06"],
        "asset_ids": ["a"],
        "comparison_groups": {
            "models": ["a", "b"],
            "factor_pair": ["a", "sparse"],
            "constant_pair": ["a", "constant"],
        },
        "comparisons": [
            {
                "comparison_id": "a_minus_b",
                "left": "a",
                "right": "b",
                "sample_scope": "models",
            }
        ],
        "inference_groups": {
            "models": ["a", "b"],
            "delta": ["a_minus_b"],
            "factors": ["sparse"],
        },
        "statistics": {
            "hac_lag": 1,
            "block_size": 1,
            "repetitions": 100,
            "seed": 42,
            "confidence": 0.95,
            "fdr_method": "BY",
            "alpha": 0.05,
        },
        "output": str(tmp_path / "evaluation"),
    }
    output = evaluate_time_series(config)
    daily = read_dataset(output / "daily")
    own = daily.loc[
        daily.candidate_id.eq("a")
        & daily.sample_scope.eq("own")
        & daily.metric.eq("mse")
    ]
    np.testing.assert_allclose(own.value, [4, 9, 16])
    constant_pair = daily.loc[daily.sample_scope.eq("constant_pair")]
    assert (
        constant_pair.loc[constant_pair.metric.isin(["ic", "rank_ic"]), "value"]
        .isna()
        .all()
    )
    assert (
        constant_pair.loc[constant_pair.metric.isin(["ic", "rank_ic"]), "reason"]
        .eq("no_common_defined_blocks")
        .all()
    )
    constant_pair_mse = constant_pair.loc[
        constant_pair.candidate_id.eq("a") & constant_pair.metric.eq("mse")
    ]
    np.testing.assert_allclose(constant_pair_mse.value, [4, 9, 16])
    delta = daily.loc[daily.candidate_id.eq("a_minus_b") & daily.metric.eq("mse")]
    np.testing.assert_allclose(delta.value, [-2.25, -3.25, -4.25])
    sparse_day = daily.loc[
        daily.sample_scope.eq("factor_pair") & daily.timestamp.eq("2020-01-05")
    ]
    assert sparse_day.observation_count.eq(0).all()
    inference = read_dataset(output / "inference")
    own_mean = inference.loc[
        inference.candidate_id.eq("a")
        & inference.sample_scope.eq("own")
        & inference.metric.eq("mse")
    ].iloc[0]
    assert own_mean["mean"] == pytest.approx(29 / 3)
    assert own_mean.status == "DEFINED"
    summary = read_dataset(output / "summary")
    summary = summary.loc[
        summary.candidate_id.eq("a") & summary.sample_scope.eq("own")
    ].set_index("metric")
    assert summary.loc["mse", "value"] == pytest.approx(29 / 3)
    assert summary.loc["rmse", "value"] == pytest.approx(np.sqrt(29 / 3))
    assert summary.loc["mae", "value"] == pytest.approx(3)
    assert summary.loc["r2_vs_b", "value"] == pytest.approx(39 / 155)
    incomplete = inference.loc[
        inference.candidate_id.eq("sparse") & inference.metric.eq("ic")
    ]
    assert incomplete.reason.eq("incomplete_session_clock").all()
    dense_output = evaluate_time_series(
        {
            **config,
            "predictions": {},
            "r2_baselines": ["zero"],
            "window": {"start": "2020-01-04", "end": "2020-01-04"},
            "sessions": ["2020-01-04"],
            "comparison_groups": {},
            "comparisons": [],
            "inference_groups": {"factors": ["sparse"]},
            "output": str(tmp_path / "defined-factor-day"),
        }
    )
    dense = read_dataset(dense_output / "daily")
    assert dense.reason.isna().all()
    assert dense.value.eq(1).all()


def test_cross_section_daily_errors_and_matched_r2_use_two_stage_weights():
    # Unequal cross sections and a single-observation timestamp distinguish
    # equal timestamps from pooled rows, and loss coverage from IC coverage.
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-04 09:30"] * 2
                + ["2020-01-04 09:31"] * 3
                + ["2020-01-04 09:32"]
                + ["2020-01-08 09:30"] * 2
            ),
            "asset_id": ["a", "b", "a", "b", "c", "a", "a", "b"],
            "signal": [2.0, 3.0, 0.0, 1.0, 2.0, 9.0, 1.0, 2.0],
            "label": [0.0, 2.0, 4.0, 2.0, 0.0, 10.0, 3.0, 4.0],
            "ols": [np.nan, 2.0, np.nan, 1.0, 4.0, np.nan, 0.0, 0.0],
        }
    )
    daily = daily_signal_metrics(
        frame,
        evaluation_axis="cross_sectional",
        sessions=["2020-01-04", "2020-01-06", "2020-01-08"],
        asset_ids=["a", "b", "c"],
        regression=True,
        baseline_columns=["zero", "ols"],
    )
    assert daily.asset_id.eq("__cross_section__").all()
    day = daily.loc[daily.timestamp.eq("2020-01-04")].set_index("metric")
    assert day.loc["ic", "value"] == pytest.approx(0.0)
    assert day.loc["rank_ic", "value"] == pytest.approx(0.0)
    assert day.loc["ic", "valid_block_count"] == 2
    assert day.block_count.eq(3).all()
    assert day.loc["mse", "value"] == pytest.approx(3.5)
    assert day.loc["rmse", "value"] == pytest.approx(np.sqrt(3.5))
    assert day.loc["mae", "value"] == pytest.approx(29 / 18)
    assert day.loc["matched_model_mse_vs_ols", "value"] == pytest.approx(1.75)
    assert day.loc["matched_baseline_mse_vs_ols", "value"] == pytest.approx(4.25)
    assert day.loc["r2_vs_ols", "value"] == pytest.approx(10 / 17)
    assert day.loc["mse", "observation_count"] == 6
    assert day.loc["matched_model_mse_vs_ols", "observation_count"] == 3
    assert day.loc["r2_vs_ols", "valid_block_count"] == 2
    missing = daily.loc[daily.timestamp.eq("2020-01-06")]
    assert missing.value.isna().all()
    assert missing.input_count.eq(0).all() and missing.block_count.eq(0).all()
    summary = daily_metric_summary(
        daily.assign(candidate_id="model", sample_scope="own")
    ).set_index("metric")
    assert summary.loc["mse", "value"] == pytest.approx(15 / 4)
    assert summary.loc["rmse", "value"] == pytest.approx(np.sqrt(15 / 4))
    assert summary.loc["mae", "value"] == pytest.approx(65 / 36)
    # Ratio of day-weighted matched losses, never the average of day R-squared.
    assert summary.loc["r2_vs_ols", "value"] == pytest.approx(44 / 67)
    assert summary.loc["r2_vs_ols", "aggregation"] == "ratio_daily_equal_weight_mse"
    assert summary.session_count.eq(3).all()
    assert summary.valid_session_count.eq(2).all()


def test_intraday_correlations_are_per_asset_day_and_keep_declared_gaps():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-04 09:30", "2020-01-04 09:31", "2020-01-04 09:32"] * 2
                + ["2020-01-08 09:30", "2020-01-08 09:31", "2020-01-08 09:32"]
            ),
            "asset_id": ["a"] * 3 + ["b"] * 3 + ["a"] * 3,
            "signal": [1.0, 2, 3, 1, 2, 3, 1, 3, 2],
            "label": [1.0, 4, 9, 3, 2, 1, 3, 2, 1],
        }
    )
    daily = daily_signal_metrics(
        frame,
        evaluation_axis="single_asset_time_series",
        sessions=["2020-01-04", "2020-01-08"],
        asset_ids=["a", "b"],
    ).set_index(["asset_id", "timestamp", "metric"])
    assert len(daily) == 8
    first, last = pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-08")
    assert daily.loc[("a", first, "ic"), "value"] == pytest.approx(4 * np.sqrt(3) / 7)
    assert daily.loc[("a", first, "rank_ic"), "value"] == pytest.approx(1)
    assert daily.loc[("b", first, "ic"), "value"] == pytest.approx(-1)
    assert daily.loc[("a", last, "rank_ic"), "value"] == pytest.approx(-0.5)
    assert pd.isna(daily.loc[("b", last, "ic"), "value"])
    assert daily.loc[("b", last, "ic"), "reason"] == "insufficient_observations"


def test_zero_and_missing_baselines_do_not_remove_own_errors_or_empty_sessions():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-04 09:30", "2020-01-04 09:31"]),
            "asset_id": ["a", "a"],
            "signal": [1.0, 2.0],
            "label": [0.0, 0.0],
            "missing": [np.nan, np.nan],
        }
    )
    kwargs = dict(
        evaluation_axis="single_asset_time_series",
        sessions=["2020-01-04", "2020-01-08"],
        asset_ids=["a", "b"],
        regression=True,
        baseline_columns=["zero", "missing"],
    )
    daily = daily_signal_metrics(frame, **kwargs)
    first = daily.loc[daily.asset_id.eq("a") & daily.timestamp.eq("2020-01-04")]
    first = first.set_index("metric")
    assert first.loc["mse", "value"] == pytest.approx(2.5)
    assert first.loc["r2_vs_zero", "reason"] == "zero_baseline_error"
    assert first.loc["matched_model_mse_vs_missing", "observation_count"] == 0
    assert first.loc["r2_vs_missing", "reason"] == "no_observations"
    summary = daily_metric_summary(
        daily.assign(candidate_id="model", sample_scope="own")
    )
    zero = summary.loc[summary.asset_id.eq("a") & summary.metric.eq("r2_vs_zero")].iloc[
        0
    ]
    assert pd.isna(zero.value) and zero.reason == "zero_baseline_error"
    empty = daily_signal_metrics(frame.iloc[:0], **kwargs)
    assert len(empty) == 4 * 11
    assert empty.value.isna().all()
    assert empty.observation_count.eq(0).all()
    with pytest.raises(ValueError, match="evaluation_asset_id_reserved"):
        daily_signal_metrics(frame, **{**kwargs, "asset_ids": ["__cross_section__"]})


def _factor_evaluation_config(tmp_path, frame):
    path = tmp_path / "factors.parquet"
    frame.to_parquet(path, index=False)
    declaration = input_spec()
    declaration["column_roles"].update(peer="factor")
    declaration["label_availability_columns"] = {"label": "label_available_at"}
    declaration["column_units"] = {"label": "decimal_return"}
    return {
        "inputs": {"path": str(path), "spec": declaration},
        "predictions": {},
        "factor_columns": ["x", "peer"],
        "label": "label",
        "window": {"start": "2020-01-04", "end": "2020-01-08"},
        "cutoff": "2020-01-08T23:59:59",
        "sessions": ["2020-01-04", "2020-01-08"],
        "asset_ids": sorted(frame.asset_id.unique()),
        "evaluation_axis": "cross_sectional",
        "comparison_groups": {"factors": ["x", "peer"]},
        "inference_groups": {"factors": ["x", "peer"]},
        "statistics": {
            "hac_lag": 0,
            "block_size": 1,
            "repetitions": 100,
            "seed": 42,
            "confidence": 0.95,
            "fdr_method": "BY",
            "alpha": 0.05,
        },
        "output": str(tmp_path / "evaluation"),
    }


@pytest.mark.parametrize("axis", ["cross_sectional", "single_asset_time_series"])
def test_empty_session_axis_roundtrips_native_schemas_and_rejects_observations(
    tmp_path, axis
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-04 09:30", "2020-01-04 09:31"]),
            "asset_id": ["a", "a"],
            "x": [1.0, 2.0],
            "peer": [2.0, 1.0],
            "label": [1.0, 2.0],
            "eligible": [True, True],
            "label_available_at": pd.to_datetime(["2020-01-04"] * 2),
        }
    )
    config = _factor_evaluation_config(tmp_path, frame)
    config.update(
        evaluation_axis=axis,
        comparisons=[
            {
                "comparison_id": "delta",
                "left": "x",
                "right": "peer",
                "sample_scope": "factors",
            }
        ],
    )
    populated = evaluate_time_series(config)
    config.update(sessions=[], output=str(tmp_path / "empty-evaluation"))
    with pytest.raises(ValueError, match="evaluation_input_session_not_declared"):
        evaluate_time_series(config)
    assert not (tmp_path / "empty-evaluation").exists()

    # Empty means no observations, not observations excluded by finite eligibility.
    signal = frame.rename(columns={"x": "signal"})
    kwargs = dict(
        evaluation_axis=axis,
        sessions=[],
        asset_ids=["a"],
        regression=True,
        baseline_columns=["zero"],
    )
    with pytest.raises(ValueError, match="evaluation_input_session_not_declared"):
        daily_signal_metrics(signal.assign(signal=np.nan, label=np.nan), **kwargs)
    empty_daily = daily_signal_metrics(signal.iloc[:0], **kwargs)
    assert empty_daily.empty
    assert pd.api.types.is_datetime64_any_dtype(empty_daily.timestamp)
    assert pd.api.types.is_float_dtype(empty_daily.value)
    assert pd.api.types.is_integer_dtype(empty_daily.observation_count)

    frame.iloc[:0].to_parquet(tmp_path / "factors.parquet", index=False)
    empty = evaluate_time_series(config)
    for name in ("daily", "inference", "summary"):
        actual = read_dataset(empty / name)
        reference = read_dataset(populated / name)
        assert actual.empty
        assert list(actual) == list(reference)
        assert pd.api.types.is_integer_dtype(actual.observation_count)
        assert pd.api.types.is_float_dtype(
            actual["mean" if name == "inference" else "value"]
        )
    assert pd.api.types.is_datetime64_any_dtype(read_dataset(empty / "daily").timestamp)


def test_native_common_ranks_recompute_after_finite_pit_and_unknown_eligibility(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-04 09:30"] * 6),
            "asset_id": list("abcdef"),
            "x": [0.0, 1.0, 2.0, 3.0, 99.0, 100.0],
            "peer": [0.0, np.nan, 2.0, 3.0, 99.0, 100.0],
            "label": [0.0, 3.0, 1.0, 2.0, -99.0, -100.0],
            "eligible": pd.array([True, True, True, True, None, True], dtype="boolean"),
            "label_available_at": pd.to_datetime(["2020-01-04"] * 5 + ["2020-01-09"]),
        }
    )
    config = _factor_evaluation_config(tmp_path, frame)
    config["window"] = {"start": "2020-01-04T09:30:00", "end": "2020-01-08T09:31:00"}
    output = evaluate_time_series(config)
    daily = read_dataset(output / "daily")
    first = daily.loc[daily.timestamp.eq("2020-01-04") & daily.candidate_id.eq("x")]
    own = first.loc[first.sample_scope.eq("own") & first.metric.eq("rank_ic")].iloc[0]
    common = first.loc[
        first.sample_scope.eq("factors") & first.metric.eq("rank_ic")
    ].iloc[0]
    assert own.value == pytest.approx(0.4)
    assert own.observation_count == 4 and own.input_count == 6
    assert common.value == pytest.approx(1.0)
    assert common.observation_count == 3 and common.input_count == 6
    assert daily.loc[daily.timestamp.eq("2020-01-08"), "value"].isna().all()
    inference = read_dataset(output / "inference")
    assert inference.reason.eq("incomplete_session_clock").all()
    assert pd.read_parquet(tmp_path / "factors.parquet").eligible.isna().sum() == 1


def test_native_probability_scores_are_not_return_predictions_or_r2_baselines(tmp_path):
    from copy import deepcopy

    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.models import (
        DEEPLOB_INPUT_COLUMNS,
        PROBABILITY_COLUMNS,
        ModelSpec,
        model_spec_payload,
    )
    from tests.helpers.output_frames import _price_volume_prediction_metadata

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-04 09:30"] * 3),
            "asset_id": list("abc"),
            "x": [-0.006, 0.006, 0.0],
            "peer": [0.0, 0.0, 0.0],
            "label": [-0.01, 0.01, 0.0],
            "eligible": [True] * 3,
            "label_available_at": pd.to_datetime(["2020-01-04 09:31"] * 3),
        }
    )
    config = _factor_evaluation_config(tmp_path, frame)
    config.update(
        factor_columns=[],
        comparison_groups={},
        inference_groups={"models": ["regression", "classifier", "wrong_unit"]},
        r2_baselines=["baseline_regression", "regression", "zero", "classifier"],
    )
    predictions = {}
    for name in ("regression", "classifier", "wrong_unit"):
        classifier = name == "classifier"
        metadata = deepcopy(_price_volume_prediction_metadata())
        model = model_spec_payload(
            ModelSpec(
                model_id=name,
                model_algorithm="deeplob_classifier"
                if classifier
                else "linear_regression",
                input_feature_columns=DEEPLOB_INPUT_COLUMNS if classifier else ("x",),
                target_column="label",
                hyperparameters={"class_threshold": 0.005} if classifier else {},
            )
        )
        unit = "basis_points" if name == "wrong_unit" else "decimal_return"
        metadata.update(
            model_id=name,
            family_id=name,
            label_unit=unit,
            prediction_window=config["window"],
            prediction_rows=3,
            effective_prediction_end="2020-01-04T09:30:00",
        )
        metadata["model_metadata"]["label_availability_column"] = "label_available_at"
        metadata["model_facts"].update(
            family_id=name,
            model=model,
            label_unit=unit,
            input_units={},
            candidate_parameters=model["hyperparameters"],
            target_definition={
                "source_label": "label",
                "target_transform": "raw",
                "target_clip": "none",
                "label_semantics": None,
            },
        )
        predicted = frame[["timestamp", "asset_id"]].assign(
            prediction_score=[-0.7, 0.7, 0.0] if classifier else frame.x,
            prediction_rank_score=[-0.5, 0.5, 0.0],
        )
        if classifier:
            predicted[list(PROBABILITY_COLUMNS)] = [
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
                [0.1, 0.8, 0.1],
            ]
        predictions[name] = [
            str(
                write_prediction_outputs(
                    output_path=tmp_path / name,
                    output_chunks=[predicted],
                    metadata=metadata,
                )
            )
        ]
    predictions["baseline_regression"] = predictions["regression"]
    config["predictions"] = predictions
    output = evaluate_time_series(config)
    daily = read_dataset(output / "daily")
    first = daily.loc[daily.timestamp.eq("2020-01-04")]
    classifier = first.loc[first.candidate_id.eq("classifier")].set_index("metric")
    assert not {"mse", "rmse", "mae", "r2_vs_zero"} & set(classifier.index)
    assert classifier.loc["log_loss", "value"] == pytest.approx(-np.log(0.8))
    assert classifier.loc["brier", "value"] == pytest.approx(0.06)
    assert classifier.loc["accuracy", "value"] == 1
    assert classifier.loc["macro_f1", "value"] == 1
    regression = first.loc[first.candidate_id.eq("regression")].set_index("metric")
    assert regression.loc["mse", "value"] == pytest.approx(2 * 0.004**2 / 3)
    assert regression.loc["r2_vs_zero", "value"] == pytest.approx(0.84)
    assert regression.loc["r2_vs_regression", "value"] == pytest.approx(0)
    assert regression.loc["r2_vs_baseline_regression", "value"] == pytest.approx(0)
    assert (
        regression.loc["r2_vs_classifier", "reason"]
        == "prediction_direction_score_not_return"
    )
    wrong = first.loc[first.candidate_id.eq("wrong_unit")].set_index("metric")
    assert wrong.loc["ic", "value"] == pytest.approx(1)
    assert pd.isna(wrong.loc["mse", "value"])
    assert wrong.loc["mse", "reason"] == "prediction_target_unit_mismatch"
    empty_output = evaluate_time_series(
        {
            **config,
            "window": {"start": "2020-01-08", "end": "2020-01-08"},
            "sessions": [],
            "output": str(tmp_path / "empty-prediction-evaluation"),
        }
    )
    for name in ("daily", "inference", "summary"):
        assert read_dataset(empty_output / name).empty


def test_economic_prediction_column_has_its_own_finite_loss_domain():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-04 09:30"] * 2 + ["2020-01-04 09:31"] * 2
            ),
            "asset_id": ["a", "b", "a", "b"],
            "signal": [1.0, 2.0, 3.0, 4.0],
            "label": [1.0, 2.0, 3.0, 4.0],
            "economic": [1.0, np.nan, np.nan, 8.0],
            "ols": [np.nan, 0.0, 0.0, 0.0],
        }
    )
    kwargs = dict(
        evaluation_axis="cross_sectional",
        sessions=["2020-01-04"],
        asset_ids=["a", "b"],
        regression=True,
        prediction_column="economic",
        baseline_columns=["ols"],
    )
    daily = daily_signal_metrics(frame, **kwargs).set_index("metric")
    assert daily.loc["ic", "value"] == pytest.approx(1)
    assert daily.loc["ic", "observation_count"] == 4
    assert daily.loc["mse", "value"] == pytest.approx(8)
    assert daily.loc["mse", "observation_count"] == 2
    assert daily.loc["matched_model_mse_vs_ols", "value"] == pytest.approx(16)
    assert daily.loc["matched_baseline_mse_vs_ols", "value"] == pytest.approx(16)
    assert daily.loc["r2_vs_ols", "value"] == pytest.approx(0)
    assert daily.loc["r2_vs_ols", "observation_count"] == 1
    empty = daily_signal_metrics(frame.assign(economic=np.nan), **kwargs).set_index(
        "metric"
    )
    assert empty.loc["ic", "observation_count"] == 4
    assert pd.isna(empty.loc["mse", "value"])
    assert empty.loc["mse", "reason"] == "no_observations"


def test_overall_r2_includes_paired_zero_baseline_error_days():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2020-01-04 09:30",
                    "2020-01-04 09:31",
                    "2020-01-08 09:30",
                    "2020-01-08 09:31",
                ]
            ),
            "asset_id": ["a"] * 4,
            "signal": [1.0, 2.0, 0.0, 0.0],
            "label": [0.0, 0.0, 1.0, 2.0],
        }
    )
    daily = daily_signal_metrics(
        frame,
        evaluation_axis="single_asset_time_series",
        sessions=["2020-01-04", "2020-01-08"],
        asset_ids=["a"],
        regression=True,
        baseline_columns=["zero"],
    )
    r2 = daily.loc[daily.metric.eq("r2_vs_zero")]
    assert pd.isna(r2.iloc[0].value) and r2.iloc[0].reason == "zero_baseline_error"
    assert r2.iloc[1].value == pytest.approx(0)
    summary = daily_metric_summary(daily.assign(candidate_id="m", sample_scope="own"))
    overall = summary.set_index("metric").loc["r2_vs_zero"]
    # Both days contribute paired losses even though the first day R-squared
    # is undefined: 1 - mean(2.5, 2.5) / mean(0, 2.5) = -1.
    assert overall.value == pytest.approx(-1)
    assert overall.valid_session_count == 2
    assert overall.valid_block_count == 2 and overall.observation_count == 4


def test_common_daily_correlation_delta_uses_shared_defined_timestamps(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-04 09:30"] * 3 + ["2020-01-04 09:31"] * 3
            ),
            "asset_id": list("abc") * 2,
            "x": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "peer": [5.0, 5.0, 5.0, 1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0] * 2,
            "eligible": True,
            "label_available_at": pd.to_datetime(["2020-01-04 09:32"] * 6),
        }
    )
    config = _factor_evaluation_config(tmp_path, frame)
    config["comparisons"] = [
        {
            "comparison_id": "delta",
            "left": "x",
            "right": "peer",
            "sample_scope": "factors",
        }
    ]
    output = evaluate_time_series(config)
    daily = read_dataset(output / "daily")
    first = daily.loc[daily.timestamp.eq("2020-01-04")].set_index(
        ["sample_scope", "candidate_id", "metric"]
    )
    for metric in ("ic", "rank_ic"):
        # Own x uses both timestamps: mean(+1, -1)=0. The common comparison
        # must omit timestamp 09:30 for both candidates because peer is constant.
        assert first.loc[("own", "x", metric), "value"] == pytest.approx(0)
        assert first.loc[("own", "x", metric), "observation_count"] == 6
        assert first.loc[("factors", "x", metric), "value"] == pytest.approx(-1)
        assert first.loc[("factors", "peer", metric), "value"] == pytest.approx(1)
        assert first.loc[("factors", "delta", metric), "value"] == pytest.approx(-2)
        assert first.loc[("factors", "x", metric), "observation_count"] == 3
        assert first.loc[("factors", "x", metric), "valid_block_count"] == 1
        assert first.loc[("factors", "x", metric), "block_count"] == 2
    assert daily.loc[daily.timestamp.eq("2020-01-08"), "value"].isna().all()


def test_common_metric_families_match_errors_without_using_inapplicable_candidates():
    from quant_research_platform.research.evaluations.time_series import (
        _common_metric_blocks,
        _daily_signal_metrics_from_blocks,
        _signal_metric_blocks,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2020-01-04 09:30"] * 3 + ["2020-01-04 09:31"] * 3
            ),
            "asset_id": list("abc") * 2,
            "signal": [1.0, 2.0, 3.0] * 2,
            "label": [1.0, 2.0, 3.0] * 2,
        }
    )
    blocks = []
    for name, economic in (
        ("a", [4.0, 5.0, 6.0, 2.0, 3.0, 4.0]),
        ("b", [np.nan, np.nan, np.nan, 3.0, 4.0, 5.0]),
        ("inapplicable", [np.nan] * 6),
    ):
        blocks.append(
            _signal_metric_blocks(
                frame.assign(economic=economic),
                evaluation_axis="cross_sectional",
                sessions=["2020-01-04"],
                asset_ids=list("abc"),
                regression=True,
                prediction_column="economic",
            ).assign(candidate_id=name)
        )
    own = _daily_signal_metrics_from_blocks(blocks[0]).set_index("metric")
    assert own.loc["mse", "value"] == pytest.approx(5)
    matched = _common_metric_blocks(
        pd.concat(blocks, ignore_index=True),
        metric_candidates={
            "ic": ["a", "b", "inapplicable"],
            "rank_ic": ["a", "b", "inapplicable"],
            "mse": ["a", "b"],
            "mae": ["a", "b"],
        },
    )
    common = {
        candidate: _daily_signal_metrics_from_blocks(part).set_index("metric")
        for candidate, part in matched.groupby("candidate_id", sort=False)
    }
    # Only the second timestamp supports both economic forecasts. An explicitly
    # inapplicable forecast cannot remove another model's valid error evidence.
    assert common["a"].loc["mse", "value"] == pytest.approx(1)
    assert common["b"].loc["mse", "value"] == pytest.approx(4)
    assert common["a"].loc["rmse", "value"] == pytest.approx(1)
    assert common["a"].loc["mae", "value"] == pytest.approx(1)
    assert common["a"].loc["mse", "observation_count"] == 3
    assert common["a"].loc["ic", "observation_count"] == 6
    assert common["inapplicable"].loc["mse", "reason"] == "no_observations"
