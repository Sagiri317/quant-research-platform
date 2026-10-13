"""Preserved fitting invariants after removing automatic quarterly workflow fixtures."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.fitting import training_scope_chunks
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
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
    result = fitting._fit_model_result(
        fitting._collect_training_frame((frame,), spec, fitting_cutoff="2018-12-31"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    model_result = result
    assert model_result.metadata["rows_input"] == 4
    assert model_result.metadata["rows_after_filter_mode"] == 3
    assert model_result.metadata["rows_after_input_label_dropna"] == 2


def test_training_asset_scope_preserves_consumer_primary_on_early_close() -> None:
    frame = pd.DataFrame({"asset_id": ["a", "b"], "value": [1.0, 2.0]})
    primary = RuntimeError("consumer_primary")
    scoped = training_scope_chunks((frame,), asset_ids=("b",))
    next(scoped)
    with pytest.raises(RuntimeError) as raised:
        scoped.throw(primary)
    assert raised.value is primary


def test_rank_missing_contributes_neutral_without_renormalization() -> None:
    predictor = RankEnsemblePredictor(columns=("x", "y"), directions=(1.0, 1.0))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01"] * 2),
            "x": [1.0, 2.0],
            "y": [np.nan, 4.0],
        }
    )
    assert predictor.predict(frame).tolist() == pytest.approx([-0.125, 0.125])


def test_trial_selection_prefers_defined_negative_and_tie_breaks_undefined() -> None:
    partial = pd.DataFrame(
        {"trial_id": ["defined", "undefined"], "validation_rankic": [-0.4, np.nan]}
    )
    assert (
        select_best_validation_trial(
            partial,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("defined", "undefined"),
        )
        == "defined"
    )
    all_undefined = partial.assign(validation_rankic=np.nan)
    assert (
        select_best_validation_trial(
            all_undefined,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("undefined", "defined"),
        )
        == "undefined"
    )


def test_convergence_warning_is_a_model_fit_failure() -> None:
    from sklearn.exceptions import ConvergenceWarning

    class WarningEstimator:
        def fit(self, _matrix, _target, **_kwargs):
            warnings.warn("not converged", ConvergenceWarning, stacklevel=2)

    spec = ModelSpec(
        model_id="ridge_warning",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        target_column="label",
        hyperparameters={"alpha": 1.0},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="model_fit_non_converged"):
        fitting._fit_predictor(
            spec,
            prepared_training=fitting._collect_training_frame(
                [frame.assign(eligible=True, label_available_at=frame.timestamp)],
                spec,
                fitting_cutoff="2018-01-02",
            ),
            estimator=WarningEstimator(),
            threads=1,
        )


@pytest.mark.parametrize("cap", (None, 13))
@pytest.mark.parametrize("chunk_rows", (5, 17))
def test_training_retention_preserves_trailing_rows_and_same_day_order(cap, chunk_rows):
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
        max_train_rows=cap,
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
    if cap is not None:
        expected = expected.tail(cap)
    prepared = fitting._collect_training_frame(
        (frame.iloc[start : start + chunk_rows] for start in range(0, 32, chunk_rows)),
        spec,
        fitting_cutoff="2017-12-31",
    )
    pd.testing.assert_frame_equal(
        prepared.training, expected.reset_index(drop=True), check_exact=True
    )


def test_estimator_without_fit_cannot_be_returned_as_fitted():
    spec = ModelSpec(
        model_id="missing_fit",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        target_column="label",
        hyperparameters={"alpha": 1.0},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2018-01-01", "2018-01-02"]),
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
        }
    )
    with pytest.raises(AttributeError, match="fit"):
        fitting._fit_predictor(
            spec,
            prepared_training=fitting._collect_training_frame(
                [frame.assign(eligible=True, label_available_at=frame.timestamp)],
                spec,
                fitting_cutoff="2018-01-02",
            ),
            estimator=object(),
            threads=1,
        )


def test_equal_rank_single_undefined_direction_names_required_input():
    from quant_research_platform.research.models.rank_ensemble import (
        EqualWeightRankEnsembleFactory,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"] * 3),
            "valid": [1.0, 2.0, 3.0],
            "constant": [1.0] * 3,
            "label": [1.0, 2.0, 3.0],
        }
    )
    estimator = EqualWeightRankEnsembleFactory(("valid", "constant"), "label")
    with pytest.raises(
        ValueError,
        match="ensemble_direction_undefined:constant:no_valid_training_rankic",
    ):
        estimator.fit(frame, frame.label.to_numpy())
    frame["constant"] = [3.0, 2.0, 1.0]
    fitted = estimator.fit(frame, frame.label.to_numpy())
    assert fitted.columns == ("valid", "constant")
    assert fitted.directions == (1.0, -1.0)


def test_common_pk_identity_distinguishes_equal_count_and_ignores_chunking():
    from quant_research_platform.research.models.trial_selection import (
        ModelSampleCoverage,
    )

    asset_major = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01", "2019-01-02"] * 2),
            "asset_id": ["a", "a", "b", "b"],
        }
    )
    time_major = asset_major.iloc[[0, 2, 1, 3]]
    whole, pieces, changed, duplicate, invalid = (
        ModelSampleCoverage({"scope": "oracle"}) for _ in range(5)
    )
    whole.add(asset_major)
    pieces.add(time_major.iloc[:2])
    pieces.add(time_major.iloc[2:])
    changed.add(asset_major.assign(asset_id=["a", "a", "c", "c"]))
    assert whole.row_count == pieces.row_count == changed.row_count == 4
    assert whole.semantic_identity == pieces.semantic_identity
    assert whole.semantic_identity != changed.semantic_identity
    duplicate.add(asset_major)
    with pytest.raises(ValueError, match="model_coverage_key_order_invalid"):
        duplicate.add(asset_major.iloc[-1:])
    with pytest.raises(ValueError, match="model_coverage_key_invalid"):
        invalid.add(asset_major.assign(asset_id=["a", "a", None, "b"]))


def test_shared_preprocessing_matches_independent_candidate_states(monkeypatch):
    specs = tuple(
        ModelSpec(
            model_id=f"ridge_{index}",
            model_algorithm="ridge_regression",
            input_factor_columns=("x",),
            target_column="label",
            hyperparameters={"alpha": alpha},
            random_seed=37,
        )
        for index, alpha in enumerate((0.01, 0.1, 1.0, 10.0))
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=12).repeat(3),
            "x": np.arange(36.0),
            "label": np.sin(np.arange(36.0)),
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-13"),
        }
    )
    expected = [
        fitting._fit_model_result(
            fitting._collect_training_frame([frame], spec, fitting_cutoff="2020-01-13"),
            spec,
            estimator=build_estimator_from_model_spec(spec),
        )
        for spec in specs
    ]
    counts = {"weights": 0, "scaler": 0, "transform": 0}
    originals = (
        fitting.equal_timestamp_sample_weights,
        fitting.ModelMatrixScaler.fit,
        fitting.ModelMatrixScaler.transform,
    )

    def weights(*args, **kwargs):
        counts["weights"] += 1
        return originals[0](*args, **kwargs)

    def scaler(*args, **kwargs):
        counts["scaler"] += 1
        return originals[1](*args, **kwargs)

    def transform(*args, **kwargs):
        counts["transform"] += 1
        return originals[2](*args, **kwargs)

    monkeypatch.setattr(fitting, "equal_timestamp_sample_weights", weights)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", scaler)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "transform", transform)
    prepared = fitting._collect_training_frame(
        [frame], specs[0], fitting_cutoff="2020-01-13"
    )
    actual = [
        fitting._fit_model_result(
            prepared, spec, estimator=build_estimator_from_model_spec(spec)
        )
        for spec in specs
    ]
    assert counts == {"weights": 1, "scaler": 1, "transform": 1}
    for reference, result in zip(expected, actual, strict=True):
        assert result.metadata == reference.metadata
        assert pickle.dumps(result.prediction_model) == pickle.dumps(
            reference.prediction_model
        )
        restored = pickle.loads(pickle.dumps(result.prediction_model))
        fitting.validate_model_state(restored, result.metadata)
        np.testing.assert_array_equal(
            restored.predict(frame[["x"]]),
            reference.prediction_model.predict(frame[["x"]]),
        )


@pytest.mark.parametrize(
    "algorithm", ("ridge_regression", "equal_weight_rank_ensemble")
)
def test_estimator_mutation_cannot_change_shared_preprocessing(algorithm):
    spec = ModelSpec(
        model_id="isolated",
        model_algorithm=algorithm,
        input_factor_columns=("x",),
        target_column="label",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "x": [1.0, 2.0, 3.0],
            "label": [3.0, 1.0, 2.0],
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-02"),
        }
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-02"
    )
    original = pickle.dumps(prepared.preprocessing)
    training = prepared.training.copy(deep=True)

    class MutatingEstimator:
        def fit(self, matrix, target, *, sample_weight):
            matrix.loc[:, "x"] = 999.0
            target[:] = 999.0
            sample_weight[:] = 999.0
            raise RuntimeError("mutated_estimator_inputs")

    with pytest.raises(RuntimeError, match="mutated_estimator_inputs"):
        fitting._fit_predictor(
            spec, prepared_training=prepared, estimator=MutatingEstimator(), threads=1
        )
    assert pickle.dumps(prepared.preprocessing) == original
    pd.testing.assert_frame_equal(prepared.training, training, check_exact=True)


def test_training_raw_validation_runs_once_with_duplicate_dataframe_index(monkeypatch):

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
    calls = []
    original = fitting.legal_training_mask

    def validate(raw, **kwargs):
        calls.append(len(raw))
        return original(raw, **kwargs)

    monkeypatch.setattr(fitting, "legal_training_mask", validate)
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-02"
    )
    assert calls == [2]
    assert prepared.training.asset_id.tolist() == ["b"]
    assert prepared.row_counts["rows_input"] == 2
    assert prepared.row_counts["rows_after_filter_mode"] == 1


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
