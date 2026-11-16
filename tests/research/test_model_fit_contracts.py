"""Model fitting, legal sample retention and owned preprocessing buffers."""

from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.estimators import (
    build_estimator_from_model_spec,
)


def test_default_native_inputs_freeze_one_representation_and_preserve_explicit_choice(
    tmp_path,
):
    from copy import deepcopy

    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from tests.correctness.test_direct_model_operations import data, fit_config

    path = data(tmp_path)
    frame = pd.read_parquet(path)
    frame["z"] = np.sin(frame.timestamp.dt.day + frame.x)
    frame["feature_x"] = frame.x**2
    frame["feature_x_other"] = frame.label * 100
    frame.to_parquet(path, index=False)
    explicit = fit_config(tmp_path, path)
    explicit["inputs"]["spec"]["column_roles"].update(
        x="factor", z="factor", feature_x="feature", feature_x_other="feature"
    )
    explicit["model"].update(
        input_factor_columns=["z"], input_feature_columns=["feature_x"]
    )
    reference, reference_facts = load_model(fit(explicit))
    default = deepcopy(explicit)
    default["output"] = str(tmp_path / "default")
    default["model"].pop("input_factor_columns")
    default["model"].pop("input_feature_columns")
    actual, facts = load_model(fit(default))
    assert facts["model"] == reference_facts["model"]
    assert actual.metadata["input_columns"] == ["z", "feature_x"]
    np.testing.assert_array_equal(
        actual.prediction_model.predict(frame[["z", "feature_x"]]),
        reference.prediction_model.predict(frame[["z", "feature_x"]]),
    )
    subset = deepcopy(default)
    subset["output"] = str(tmp_path / "subset")
    subset["model"]["input_feature_columns"] = ["feature_x"]
    selected, _ = load_model(fit(subset))
    assert selected.metadata["input_columns"] == ["feature_x"]
    subset["output"] = str(tmp_path / "empty")
    subset["model"]["input_feature_columns"] = []
    with pytest.raises(ValueError, match="model_input_columns_required"):
        fit(subset)


@pytest.mark.parametrize(
    "parameters",
    [
        {"external_memory": 1},
        {"external_memory": True, "device": "cpu"},
        {"external_memory": True, "tree_method": "exact"},
    ],
)
def test_external_memory_requires_an_explicit_gpu_hist_protocol(parameters):
    with pytest.raises(ValueError, match="model_external_memory"):
        ModelSpec(
            model_id="external",
            model_algorithm="xgboost_regressor",
            input_factor_columns=("x",),
            target_column="label",
            hyperparameters=parameters,
        )


def test_gpu_external_memory_matches_native_quantile_and_survives_roundtrip(tmp_path):
    from dataclasses import replace

    from quant_research_platform.research.models.persisted import load_model, save_model
    from quant_research_platform.research.models.production import fit_model

    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable")
    rng = np.random.default_rng(81)
    training_rows = 16403
    x = rng.integers(-3, 4, size=(training_rows + 32, 3)).astype(float)
    y = 0.7 * x[:, 0] - 0.2 * x[:, 2]
    x[::7, 1] = np.nan
    frame = pd.DataFrame(x, columns=["x", "y", "z"])
    frame["timestamp"] = pd.to_datetime(
        ["2020-01-01"] * 4000
        + ["2020-01-02"] * (training_rows - 4000)
        + ["2020-01-04"] * 32
    )
    frame["label_available_at"] = frame.timestamp
    frame["label"] = y
    frame["asset_id"] = [str(index) for index in range(len(frame))]
    frame["eligible"] = True
    spec = ModelSpec(
        model_id="external",
        model_algorithm="xgboost_regressor",
        input_factor_columns=("x", "y", "z"),
        target_column="label",
        hyperparameters={
            "device": "cuda:0",
            "tree_method": "hist",
            "n_estimators": 8,
            "early_stopping_rounds": 2,
            "max_depth": 2,
            "max_bin": 16,
            "colsample_bytree": 0.8,
            "feature_weights": [1.0, 0.0, 1.0],
            "n_jobs": 2,
        },
    )
    results = []
    allocator = cupy.cuda.get_allocator()
    cached_bytes = cupy.get_default_memory_pool().total_bytes()
    for external in (False, True, True):
        candidate = replace(
            spec, hyperparameters={**spec.hyperparameters, "external_memory": external}
        )
        result = fit_model(
            [frame.iloc[:training_rows]],
            candidate,
            fit_windows=[{"start": "2020-01-01", "end": "2020-01-02"}],
            fit_cutoff="2020-01-03",
            validation_frames=[frame.iloc[training_rows:]],
            validation_cutoff="2020-01-05",
        )
        assert (
            result.metadata["resolved_hyperparameters"]["external_memory"] is external
        )
        assert cupy.cuda.get_allocator() is allocator
        assert cupy.get_default_memory_pool().total_bytes() == cached_bytes
        results.append(result)
    inputs = frame.loc[:, spec.input_columns]
    np.testing.assert_allclose(
        results[0].prediction_model.predict(inputs),
        results[1].prediction_model.predict(inputs),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        results[1].prediction_model.predict(inputs),
        results[2].prediction_model.predict(inputs),
        rtol=2e-6,
        atol=2e-6,
    )
    save_model(tmp_path / "external", results[1], facts={}, trial_id="external")
    restored, _ = load_model(tmp_path / "external")
    np.testing.assert_array_equal(
        restored.prediction_model.predict(inputs),
        results[1].prediction_model.predict(inputs),
    )


@pytest.mark.parametrize("failure", [None, "source", "imputation", "validation"])
def test_fit_closes_temporary_inputs_on_success_and_failure(monkeypatch, failure):
    from quant_research_platform.research.models.production import fit_model

    handles = []
    original = fitting.TemporaryFile

    def opened(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(fitting, "TemporaryFile", opened)
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "asset_id": "a",
            "x": [1.0, 2.0],
            "z": [1.0, 3.0],
            "label": [0.1, 0.2],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-03"),
        }
    )
    if failure == "imputation":
        frame["z"] = np.nan
    spec = ModelSpec(
        model_id="closed",
        model_algorithm="xgboost_regressor"
        if failure == "validation"
        else "linear_regression",
        input_factor_columns=("x", "z"),
        target_column="label",
    )

    def source():
        yield frame
        if failure == "source":
            raise RuntimeError("source_read_failed")

    options = dict(
        fit_windows=[{"start": "2020-01-01", "end": "2020-01-03"}],
        fit_cutoff="2020-01-03",
    )
    if failure == "validation":
        options.update(validation_frames=[frame], validation_cutoff="2020-01-04")
    if failure is None:
        result = fit_model(source(), spec, **options)
        assert np.isfinite(result.prediction_model.predict(frame[["x", "z"]])).all()
    else:
        error = {
            "source": "source_read_failed",
            "imputation": "model_training_feature_all_missing",
            "validation": "model_validation_training_overlap",
        }[failure]
        with pytest.raises((RuntimeError, ValueError), match=error):
            fit_model(source(), spec, **options)
    assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize(
    "scaling,intercept,positive,selection",
    [
        (True, True, False, "cyclic"),
        (False, True, True, "random"),
        (False, False, False, "cyclic"),
    ],
)
def test_precomputed_lasso_reuses_statistics_and_matches_native_weighted_fit(
    tmp_path, scaling, intercept, positive, selection
):
    from dataclasses import replace

    from sklearn.linear_model import Lasso

    from quant_research_platform.research.models.persisted import load_model, save_model

    rng = np.random.default_rng(79)
    count = 16403
    values = rng.normal(size=(count, 3)) + [2, 3, 4]
    target = 3 + values @ [0.7, -0.4, 0.2] + rng.normal(0, 0.1, count)
    values[::17, 1] = np.nan
    frame = pd.DataFrame(values, columns=["x", "y", "z"])
    frame["label"] = target
    frame["timestamp"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        np.sqrt(np.arange(count)).astype(int), unit="D"
    )
    frame["asset_id"] = [str(index) for index in range(count)]
    frame["eligible"] = True
    frame["label_available_at"] = frame.timestamp
    spec = ModelSpec(
        model_id="gram",
        model_algorithm="lasso_regression",
        input_factor_columns=("x", "y", "z"),
        target_column="label",
        training_scaling=scaling,
        hyperparameters={
            "precompute": True,
            "max_iter": None,
            "tol": 1e-9,
            "fit_intercept": intercept,
            "positive": positive,
            "selection": selection,
            "warm_start": True,
        },
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2021-01-01"
    )
    raw = prepared.features.iloc[prepared.training_rows].to_numpy(
        dtype="float64", copy=True
    )
    weights = prepared.preprocessing[0]
    target = prepared.training.label.to_numpy(copy=True)
    medians = np.nanmedian(raw, axis=0)
    np.copyto(raw, medians, where=np.isnan(raw))
    if scaling:
        mean = np.average(raw, axis=0, weights=weights)
        raw -= mean
        raw /= np.sqrt(np.average(raw**2, axis=0, weights=weights))
    statistics = None
    for alpha in (0.03, 0.01, 0.001):
        candidate = replace(
            spec, hyperparameters={**spec.hyperparameters, "alpha": alpha}
        )
        result = fitting._fit_model_result(
            prepared, candidate, estimator=build_estimator_from_model_spec(candidate)
        )
        expected = Lasso(
            alpha=alpha,
            max_iter=100000,
            tol=1e-9,
            fit_intercept=intercept,
            positive=positive,
            selection=selection,
            random_state=spec.random_seed,
            precompute=True,
        ).fit(raw, target, sample_weight=weights)
        actual = result.prediction_model.estimator
        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=2e-7, atol=2e-9)
        assert actual.intercept_ == pytest.approx(
            expected.intercept_, rel=2e-7, abs=2e-9
        )
        assert actual.dual_gap_ <= 1e-9 * np.average(
            (target - (np.average(target, weights=weights) if intercept else 0)) ** 2,
            weights=weights,
        )
        if statistics is None:
            statistics = prepared.lasso_gram
        assert prepared.lasso_gram is statistics
        assert "matrix" not in prepared.__dict__
    prepared.close()
    save_model(tmp_path / "model", result, facts={}, trial_id="gram")
    restored, _ = load_model(tmp_path / "model")
    np.testing.assert_array_equal(
        restored.prediction_model.predict(frame.loc[:, spec.input_columns]),
        result.prediction_model.predict(frame.loc[:, spec.input_columns]),
    )


def test_fitting_counts_raw_filter_and_label_rows_once() -> None:
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="model_target",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "asset_id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "model_target": [1.0, 2.0, 3.0, 4.0],
            "eligible": [True, True, True, False],
            "model_target_available_at": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2019-01-03", "2018-01-04"]
            ),
        }
    )
    with closing(
        fitting._collect_training_frame((frame,), spec, fitting_cutoff="2018-12-31")
    ) as prepared:
        result = fitting._fit_model_result(
            prepared, spec, estimator=build_estimator_from_model_spec(spec)
        )
    model_result = result
    assert model_result.metadata["rows_input"] == 4
    assert model_result.metadata["rows_after_filter_mode"] == 3
    assert model_result.metadata["rows_after_input_label_dropna"] == 2


def test_convergence_warning_is_a_model_fit_failure() -> None:
    phase = np.arange(40) * (2 * np.pi / 40)
    common = np.sqrt(2) * np.sin(phase)
    difference = 0.01 * np.sqrt(2) * np.cos(phase)
    spec = ModelSpec(
        model_id="linear_warning",
        model_algorithm="lasso_regression",
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters={"alpha": 0.01, "tol": 1e-12, "max_iter": 1},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2018-01-01", periods=40, freq="s"),
            "x": common + difference,
            "z": common - difference,
            "label": 2 * common,
        }
    )
    with closing(
        fitting._collect_training_frame(
            [frame.assign(eligible=True, label_available_at=frame.timestamp)],
            spec,
            fitting_cutoff="2018-01-02",
        )
    ) as prepared:
        with pytest.raises(ValueError, match="model_fit_non_converged"):
            fitting._fit_predictor(
                spec,
                prepared_training=prepared,
                estimator=build_estimator_from_model_spec(spec),
            )


def test_training_retention_preserves_all_legal_rows_in_canonical_key_order():
    chunk_rows = 5
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2017-01-02", periods=16).repeat(2),
            "label_available_at": pd.Timestamp("2017-02-01"),
            "asset_id": ["b", "a"] * 16,
            "x": np.arange(32.0),
            "label": np.arange(32.0) * 2 + 1,
            "eligible": True,
        }
    )
    frame.loc[frame.index % 7 == 0, "label_available_at"] = pd.Timestamp("2018-01-01")
    expected = frame.loc[
        frame.index % 7 != 0, ["timestamp", "asset_id", "x", "label"]
    ].copy()
    expected["timestamp"] = expected["timestamp"].dt.as_unit("ns")
    prepared = fitting._collect_training_frame(
        (frame.iloc[start : start + chunk_rows] for start in range(0, 32, chunk_rows)),
        spec,
        fitting_cutoff="2017-12-31",
    )
    pd.testing.assert_frame_equal(
        prepared.training,
        expected.drop(columns=list(spec.input_columns)).sort_values(
            ["timestamp", "asset_id"], ignore_index=True
        ),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        prepared.features.iloc[prepared.training_rows].to_numpy(dtype="float64"),
        expected.sort_values(["timestamp", "asset_id"])
        .loc[:, spec.input_columns]
        .to_numpy(),
    )
    prepared.close()


@pytest.mark.parametrize(
    "algorithm,parameters",
    [("lasso_regression", {"alpha": 0.01, "copy_X": copy}) for copy in (True, False)]
    + [("ridge_regression", {"alpha": 0.1, "copy_X": copy}) for copy in (True, False)]
    + [
        (
            "xgboost_regressor",
            {
                "device": "cpu",
                "n_estimators": 3,
                "max_depth": 2,
                "feature_weights": [1.0, 0.0],
                "colsample_bytree": 0.5,
            },
        )
    ],
)
def test_real_estimators_consume_prepared_matrix_without_changing_source(
    algorithm, parameters
):
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters=parameters,
        random_seed=37,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 2),
            "x": np.arange(6.0),
            "z": [2.0, 5.0, 1.0, 4.0, 9.0, 3.0],
            "label": np.sin(np.arange(6.0)),
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-03"),
        }
    )
    source = frame.copy(deep=True)
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-03"
    )
    training = prepared.training.copy(deep=True)
    weights, scaler = prepared.preprocessing
    original_weights = weights.copy()
    original_matrix = prepared.matrix.copy(deep=True)
    reference = build_estimator_from_model_spec(spec)
    reference.fit(
        prepared.matrix.copy(deep=True),
        training.label.to_numpy(copy=True),
        sample_weight=weights.copy(),
    )
    actual = fitting._fit_model_result(
        prepared, spec, estimator=build_estimator_from_model_spec(spec)
    )
    expected = reference.predict(scaler.transform(frame[["x", "z"]]))
    np.testing.assert_allclose(
        actual.prediction_model.predict(frame[["x", "z"]]),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )
    assert actual.metadata["training_target_mean"] == pytest.approx(
        np.average(training.label, weights=original_weights)
    )
    pd.testing.assert_frame_equal(frame, source)
    pd.testing.assert_frame_equal(prepared.training, training)
    pd.testing.assert_frame_equal(prepared.matrix, original_matrix, check_exact=True)
    np.testing.assert_array_equal(weights, original_weights)
    prepared.close()


@pytest.mark.parametrize("algorithm", ["lasso_regression", "ridge_regression"])
@pytest.mark.parametrize("copy_X", [True, False])
def test_shared_training_keeps_mutating_candidates_on_the_original_scale(
    algorithm, copy_X
):
    from dataclasses import replace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.models.production import (
        _fit_prepared,
        _prepare_fit,
        fit_model,
    )

    values = np.arange(1.0, 7.0)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=6),
            "asset_id": "a",
            "x": values,
            "label": 3 * values + 2,
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-06"),
        }
    )
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
        input_factor_columns=("x",),
        target_column="label",
        training_scaling=False,
        hyperparameters={"alpha": 0.01, "copy_X": copy_X},
    )
    windows = (TimeWindow("2020-01-01", "2020-01-06"),)
    prepared, _, _ = _prepare_fit(
        [frame], spec, windows=windows, cutoff=pd.Timestamp("2020-01-06")
    )
    source = frame.copy(deep=True)
    loss_rows = prepared.training.copy(deep=True)
    matrix = prepared.matrix
    original = matrix.copy(deep=True)
    weights, scaler = prepared.preprocessing
    for alpha in (0.01, 0.001, 0.0001):
        candidate = replace(spec, hyperparameters={"alpha": alpha, "copy_X": copy_X})
        actual = _fit_prepared(prepared, candidate)
        fresh = fit_model(
            [frame], candidate, fit_windows=windows, fit_cutoff="2020-01-06"
        )
        slope = (
            3 - alpha / np.var(values)
            if algorithm == "lasso_regression"
            else 3 * 17.5 / (17.5 + alpha)
        )
        expected = slope * (values - 3.5) + 12.5
        np.testing.assert_allclose(
            actual.prediction_model.predict(frame[["x"]]),
            expected,
            rtol=1e-12,
            atol=1e-14,
        )
        assert actual.metadata == fresh.metadata
        assert prepared.matrix is matrix
        assert prepared.preprocessing[0] is weights
        assert prepared.preprocessing[1] is scaler
        pd.testing.assert_frame_equal(prepared.matrix, original, check_exact=True)
        pd.testing.assert_frame_equal(prepared.training, loss_rows, check_exact=True)
    pd.testing.assert_frame_equal(frame, source, check_exact=True)


def test_training_filter_handles_duplicate_dataframe_index():

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [False, True],
            "label_available_at": pd.Timestamp("2020-01-02"),
        },
        index=[7, 7],
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-02"
    )
    assert prepared.training.asset_id.tolist() == ["b"]
    assert prepared.row_counts["rows_input"] == 2
    assert prepared.row_counts["rows_after_filter_mode"] == 1
    prepared.close()


@pytest.mark.parametrize(
    "column,value",
    (
        ("x", np.inf),
        ("label", np.inf),
        ("eligible", None),
        ("label_available_at", "not-a-date"),
    ),
)
def test_filtered_raw_training_rows_still_reject_invalid_values(column, value):

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [True, False],
            "label_available_at": pd.Timestamp("2020-01-02"),
        }
    )
    if column in {"eligible", "label_available_at"}:
        frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    with pytest.raises(ValueError):
        fitting._collect_training_frame([frame], spec, fitting_cutoff="2020-01-02")


@pytest.mark.parametrize(
    "algorithm",
    [
        "xgboost_regressor",
        "lstm_regressor",
        "transformer_regressor",
        "deeplob_regressor",
        "deeplob_classifier",
    ],
)
def test_shared_candidate_inputs_match_fresh_fits_with_distinct_histories(algorithm):
    from dataclasses import replace
    from itertools import groupby

    from quant_research_platform.contracts.models import DEEPLOB_INPUT_COLUMNS
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.models.production import (
        _fit_input_key,
        _fit_prepared,
        _prepare_fit,
        _prepare_model_inputs,
        fit_model,
    )

    dates = pd.date_range("2020-01-01T09:30", periods=24, freq="s")
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(2),
            "asset_id": ["a", "b"] * len(dates),
            "x": np.sin(np.arange(48)),
            "z": np.cos(np.arange(48)),
            "label": np.sin(np.arange(48)) / 200,
            "eligible": True,
            "label_available_at": dates.repeat(2),
            "regime": np.repeat(np.arange(24) // 10, 2),
        }
    )
    deep = algorithm.startswith("deeplob")
    columns = DEEPLOB_INPUT_COLUMNS if deep else ("x", "z")
    if deep:
        for level in range(1, 11):
            for side, sign in (("ask", 1), ("bid", -1)):
                frame[f"{side}_px{level}"] = 10 + sign * level / 100 + frame.x / 1000
                frame[f"{side}_vol{level}"] = 100 + level + frame.z
        frame["session_date"] = frame.timestamp.dt.normalize()
        frame["segment_id"] = frame.regime.astype(str)
    frame.loc[5, list(columns)] = np.nan
    parameters = (
        {
            "device": "cpu",
            "n_jobs": 1,
            "n_estimators": 6,
            "early_stopping_rounds": 2,
            "max_depth": 1,
            "max_bin": 4,
        }
        if algorithm == "xgboost_regressor"
        else {
            "device": "cpu",
            "epochs": 2,
            "batch_size": 16,
            "hidden_size": 4,
            "sequence_length": 3,
            **(
                {"conv_channels": 2, "inception_channels": 2}
                if deep
                else {"sequence_group_columns": ["regime"]}
            ),
            **({"n_heads": 2} if algorithm == "transformer_regressor" else {}),
            **({"class_threshold": 0.001} if algorithm == "deeplob_classifier" else {}),
        }
    )
    spec = ModelSpec(
        model_id="shared",
        model_algorithm=algorithm,
        input_feature_columns=columns,
        target_column="label",
        hyperparameters=parameters,
        training_scaling=algorithm != "xgboost_regressor",
        random_seed=13,
    )
    variants = (
        ({"max_depth": 1}, {"max_depth": 2}, {"max_bin": 8})
        if algorithm == "xgboost_regressor"
        else ({"sequence_length": 3}, {"sequence_length": 7})
    )
    candidates = [
        replace(spec, hyperparameters={**parameters, **item}) for item in variants
    ]
    train = frame.loc[frame.timestamp <= dates[15]]
    held = frame.loc[
        frame.timestamp >= dates[16 if algorithm == "xgboost_regressor" else 14]
    ]
    windows = (TimeWindow(dates[0], dates[15]),)
    validation_windows = (TimeWindow(dates[16], dates[-1]),)
    groups = [list(items) for _, items in groupby(candidates, _fit_input_key)]
    assert len(groups) == (2 if algorithm == "xgboost_regressor" else 1)
    for family in groups:
        prepared, validation, cutoff = _prepare_fit(
            [train],
            family[0],
            windows=windows,
            cutoff=dates[15],
            validation_frames=[held],
            validation_cutoff=dates[-1],
            validation_windows=validation_windows,
        )
        with closing(prepared), closing(validation):
            inputs = _prepare_model_inputs(prepared, family, validation)
            prepared.close()
            validation.close()
            for candidate in family:
                actual = _fit_prepared(
                    prepared,
                    candidate,
                    validation=validation,
                    validation_cutoff=cutoff,
                    fit_inputs=inputs,
                )
                fresh = fit_model(
                    [train],
                    candidate,
                    fit_windows=windows,
                    fit_cutoff=dates[15],
                    validation_frames=[held],
                    validation_cutoff=dates[-1],
                    validation_windows=validation_windows,
                )
                np.testing.assert_allclose(
                    fitting.prediction_callable(columns, actual.prediction_model)(held),
                    fitting.prediction_callable(columns, fresh.prediction_model)(held),
                    rtol=1e-6,
                    atol=1e-8,
                )
                if algorithm != "xgboost_regressor":
                    pd.testing.assert_frame_equal(
                        actual.prediction_model.history.reset_index(drop=True),
                        fresh.prediction_model.history.reset_index(drop=True),
                    )
                    for left, right in zip(
                        actual.metadata["learning_curve"]["epochs"],
                        fresh.metadata["learning_curve"]["epochs"],
                        strict=True,
                    ):
                        assert left == pytest.approx(right, rel=1e-6, abs=1e-8)
            del inputs
