from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform._version import __version__ as qrp_version
from quant_research_platform.contracts.models import (
    ModelSpec,
    _model_algorithm_ids,
    _model_smoke_hyperparameters,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    resolved_model_hyperparameters,
)
from quant_research_platform.research.predictions.outputs import (
    build_prediction_result_from_frames,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
)


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "factor": [0.1, 0.2, 0.2, 0.4, 0.3, 0.6],
            "feature": [-1.0, 1.0, -0.5, 0.5, -0.25, 0.25],
            "label": [0.01, 0.02, 0.02, 0.04, 0.03, 0.06],
            "eligible": [True] * 6,
            "asset_id": ["a", "b"] * 3,
        }
    )


@pytest.mark.parametrize("algorithm", sorted(_model_algorithm_ids()))
def test_all_model_algorithm_ids_fit_and_predict_synthetic_data(
    algorithm: str,
) -> None:
    frame = _model_frame()
    single_factor = algorithm == "single_factor_baseline"
    spec = ModelSpec(
        model_id=f"{algorithm}_demo",
        model_algorithm=algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=() if single_factor else ("feature",),
        label_column="label",
        hyperparameters=_model_smoke_hyperparameters(algorithm),
        random_seed=7,
        max_train_rows=len(frame),
    )

    estimator = build_estimator_from_model_spec(spec)
    prediction_spec = PredictionSpec(
        prediction_id=f"{algorithm}_prediction",
        model_id=spec.model_id,
        research_dataset_id="research_demo",
        train_end="2024-01-03",
        test_start="2024-01-04",
        test_end="2024-01-04",
    )
    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame.loc[frame["timestamp"] <= "2024-01-03"],),
            spec,
            estimator=estimator,
        )
        try:
            result = result_owned.value
            predictions = prediction_frame_from_artifact(
                frame,
                prediction_spec,
                result.coefficients,
                spec.input_columns,
                prediction_model=result.prediction_model,
                model_metadata=result.metadata,
            )

            assert estimator is not None
            assert result.metadata["model_algorithm"] == algorithm
            assert result.metadata["input_factor_columns"] == ["factor"]
            assert result.metadata["input_feature_columns"] == (
                [] if single_factor else ["feature"]
            )
            assert result.metadata["label_column"] == "label"
            assert result.metadata["dependency_status"]["status"] == "available"
            assert result.metadata["test_start"] is None
            assert result.metadata["test_end"] is None
            assert result.metadata["oos_start"] is None
            assert result.metadata["oos_end"] is None
            assert len(predictions) == 2
            assert np.isfinite(predictions["prediction_score"]).any()
            if single_factor:
                assert predictions.attrs["model_metadata"]["input_columns"] == [
                    "factor"
                ]
                assert predictions.attrs["model_metadata"][
                    "feature_importance"
                ] == {"factor": 1.0}
        finally:
            result_owned.lease.close()


def test_training_controls_leave_future_rows_out_of_fit_metadata() -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id="linear_regression_demo",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=4,
        purge_gap_steps=1,
    )
    estimator = build_estimator_from_model_spec(spec)

    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,), spec, estimator=estimator
        )
        try:
            result_train_end = result_owned.value.metadata["train_end"]
            result_test_start = result_owned.value.metadata["test_start"]
            result_test_end = result_owned.value.metadata["test_end"]
        finally:
            result_owned.lease.close()

    assert pd.Timestamp(result_train_end) < frame["timestamp"].max()
    assert result_test_start is None
    assert result_test_end is None


def test_model_metadata_records_resolved_parameters_and_backend_version() -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id="ridge_regression_demo",
        model_algorithm="ridge_regression",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters={"alpha": 7.0},
        random_seed=11,
        max_train_rows=len(frame),
        purge_gap_steps=1,
    )
    estimator = build_estimator_from_model_spec(spec)

    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=estimator,
        )
        try:
            result_metadata = dict(result_owned.value.metadata)
        finally:
            result_owned.lease.close()

    assert result_metadata["hyperparameters"] == {"alpha": 7.0}
    assert result_metadata["resolved_hyperparameters"] == {"alpha": 7.0}
    assert result_metadata["random_seed"] == 11
    assert result_metadata["purge_gap_steps"] == 1
    assert result_metadata["train_start"] == "2024-01-02"
    assert result_metadata["train_end"] == "2024-01-03"
    assert result_metadata["dependency_status"] == {
        "status": "available",
        "package": "sklearn",
        "version": pytest.importorskip("sklearn").__version__,
    }


def test_model_spec_snapshots_nested_parameters_and_thaws_backend_payload() -> None:
    nested_list = [np.int64(1), np.float64(0.5)]
    nested_dict = {"values": nested_list}
    spec = ModelSpec(
        model_id="linear_nested_parameters",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        hyperparameters={"nested": nested_dict},
    )
    nested_list[0] = 99
    nested_dict["values"] = ["changed"]

    resolved = resolved_model_hyperparameters(spec)
    assert resolved == {"nested": {"values": [1, 0.5]}}
    assert spec.hyperparameters["nested"]["values"] == (1, 0.5)  # type: ignore[index]

    metadata = model_fitting.model_metadata(
        spec,
        filter_metadata={
            "filter_columns_used": [],
            "filter_definition": "none",
            "rows_input": 1,
            "rows_after_filter_mode": 1,
        },
        row_counts={
            "rows_after_input_label_dropna": 1,
            "train_start": "2024-01-01",
            "train_end": "2024-01-01",
        },
        coefficient_count=1,
    )
    assert metadata["hyperparameters"] == resolved
    resolved["nested"]["values"][0] = 7  # type: ignore[index]
    assert spec.hyperparameters["nested"]["values"] == (1, 0.5)  # type: ignore[index]

    backend_spec = ModelSpec(
        model_id="hist_nested_parameters",
        model_algorithm="hist_gradient_boosting_regressor",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters={
            "monotonic_cst": [1, -1],
            "max_iter": np.int64(2),
        },
    )
    estimator = build_estimator_from_model_spec(backend_spec)
    assert list(estimator.monotonic_cst) == [1, -1]
    assert type(resolved_model_hyperparameters(backend_spec)["monotonic_cst"]) is list


def test_prediction_accepts_actual_bounded_parallel_model_threads() -> None:
    frame = _model_frame()
    model_spec = ModelSpec(
        model_id="extra_trees_demo",
        model_algorithm="extra_trees_regressor",
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        hyperparameters=_model_smoke_hyperparameters("extra_trees_regressor"),
        random_seed=7,
        max_train_rows=len(frame),
    )
    estimator = build_estimator_from_model_spec(model_spec)
    prediction_spec = PredictionSpec(
        prediction_id="extra_trees_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="research_demo",
        train_end="2024-01-03",
        test_start="2024-01-04",
        test_end="2024-01-04",
    )
    with runtime_resource_policy_scope():
        fit_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            model_spec,
            estimator=estimator,
        )
        try:
            fit = fit_owned.value
            result = build_prediction_result_from_frames(
                (),
                prediction_spec,
                fit.coefficients,
                model_spec.input_columns,
                prediction_model=fit.prediction_model,
                model_metadata=fit.metadata,
            )

            assert int(result.model_metadata["resolved_hyperparameters"]["n_jobs"]) > 0  # type: ignore[index]

            bounded_request = {
                **fit.metadata["hyperparameters"],
                "n_jobs": 4,
            }
            bounded_resolved = {
                **fit.metadata["resolved_hyperparameters"],
                "n_jobs": 2,
            }
            bounded = {
                **fit.metadata,
                "hyperparameters": bounded_request,
                "resolved_hyperparameters": bounded_resolved,
            }
            assert (
                build_prediction_result_from_frames(
                    (),
                    prediction_spec,
                    fit.coefficients,
                    model_spec.input_columns,
                    prediction_model=fit.prediction_model,
                    model_metadata=bounded,
                ).model_metadata["resolved_hyperparameters"]
                == bounded_resolved
            )

            forged = {
                **fit.metadata,
                "hyperparameters": {**fit.metadata["hyperparameters"], "n_jobs": 1},
                "resolved_hyperparameters": {
                    **fit.metadata["resolved_hyperparameters"],
                    "n_jobs": 999_999,
                },
            }
            with pytest.raises(ValueError, match="invalid_model_metadata"):
                build_prediction_result_from_frames(
                    (),
                    prediction_spec,
                    fit.coefficients,
                    model_spec.input_columns,
                    prediction_model=fit.prediction_model,
                    model_metadata=forged,
                )

            for invalid_threads in (0, -1, True, 1.0):
                resolved = {
                    **fit.metadata["resolved_hyperparameters"],
                    "n_jobs": invalid_threads,
                }
                forged = {**fit.metadata, "resolved_hyperparameters": resolved}
                with pytest.raises(ValueError, match="invalid_model_metadata"):
                    build_prediction_result_from_frames(
                        (),
                        prediction_spec,
                        fit.coefficients,
                        model_spec.input_columns,
                        prediction_model=fit.prediction_model,
                        model_metadata=forged,
                    )
        finally:
            fit_owned.lease.close()


@pytest.mark.parametrize("physical_cores", (16, 4))
def test_persisted_parallel_model_threads_are_reader_host_independent(
    monkeypatch: pytest.MonkeyPatch,
    physical_cores: int,
) -> None:
    hyperparameters = _model_smoke_hyperparameters("extra_trees_regressor")
    resolved_hyperparameters = {**hyperparameters, "n_jobs": 8}
    monkeypatch.setattr(
        "quant_research_platform.contracts.models.psutil.cpu_count",
        lambda *, logical: physical_cores,
    )

    result = build_prediction_result_from_frames(
        (),
        PredictionSpec(
            prediction_id="portable_extra_trees_prediction",
            model_id="extra_trees_demo",
            research_dataset_id="research_demo",
            train_end="2024-01-03",
            test_start="2024-01-04",
        ),
        np.array([0.0, 0.0]),
        ("factor", "feature"),
        model_metadata={
            "model_algorithm": "extra_trees_regressor",
            "hyperparameters": hyperparameters,
            "resolved_hyperparameters": resolved_hyperparameters,
            "dependency_status": {"package": "sklearn"},
        },
    )

    assert result.model_metadata["resolved_hyperparameters"] == (
        resolved_hyperparameters
    )


def test_model_contract_rejects_conflicting_effective_seed() -> None:
    with pytest.raises(
        ValueError,
        match="model_random_seed_conflicts_with_random_state",
    ):
        ModelSpec(
            model_id="decision_tree_demo",
            model_algorithm="decision_tree_regressor",
            input_factor_columns=("factor",),
            label_column="label",
            random_seed=11,
            hyperparameters={"random_state": 99},
        )


@pytest.mark.parametrize(
    ("algorithm", "module_prefix", "package"),
    (
        ("linear_regression", "quant_research_platform.", "internal"),
        ("ridge_regression", "sklearn.", "sklearn"),
    ),
)
def test_model_metadata_matches_actual_execution_owner(
    algorithm: str,
    module_prefix: str,
    package: str,
) -> None:
    frame = _model_frame()
    spec = ModelSpec(
        model_id=f"{algorithm}_demo",
        model_algorithm=algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )

    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,), spec, estimator=estimator
        )
        try:
            dependency_status = dict(
                result_owned.value.metadata["dependency_status"]
            )
        finally:
            result_owned.lease.close()

    assert type(estimator).__module__.startswith(module_prefix)
    assert dependency_status["package"] == package
    if package == "internal":
        assert dependency_status["version"] == qrp_version


def test_prediction_spec_rejects_non_canonical_output_schema() -> None:
    with pytest.raises(ValueError, match="prediction_output_columns_must_be_fixed"):
        PredictionSpec(
            prediction_id="bad_prediction_schema",
            model_id="linear_regression_demo",
            research_dataset_id="research_demo",
            prediction_column="prediction",
            primary_key=("timestamp",),
            train_end="2024-01-02",
            test_start="2024-01-03",
        )
