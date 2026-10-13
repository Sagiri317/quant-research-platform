"""Derived Ridge seeds survive native fitting, publication and state loading."""

from pathlib import Path

import numpy as np
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting, persisted
from quant_research_platform.storage.identity import native_filesystem_path
from tests.research.workflow_test_support import shared_model_case


@pytest.mark.parametrize("solver", ("sag", "saga"))
def test_ridge_native_seed_is_independent_of_global_rng_and_persisted(tmp_path, solver):
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={
            "solver": solver,
            "alpha": 1,
            "max_iter": 10000,
            "tol": 1e-10,
        },
    )
    state = np.random.get_state()
    try:
        np.random.seed(1)
        first = shared_model_case(
            tmp_path / "first", specs=(spec,), development_end="2019-03-31"
        )
        np.random.seed(991)
        second = shared_model_case(
            tmp_path / "second", specs=(spec,), development_end="2019-03-31"
        )
    finally:
        np.random.set_state(state)
    assert first.artifact.selections == second.artifact.selections
    for first_entry, second_entry in zip(
        first.artifact.final_models, second.artifact.final_models, strict=True
    ):
        assert first_entry["model_metadata"] == second_entry["model_metadata"]
        group = first_entry["selection_group"]
        left = persisted.read_quarterly_predictor(first.artifact, group, "ridge")
        right = persisted.read_quarterly_predictor(second.artifact, group, "ridge")
        metadata = first_entry["model_metadata"]
        assert type(left.estimator.random_state) is int
        assert left.estimator.random_state == metadata["random_seed"]
        assert right.estimator.random_state == metadata["random_seed"]
        inputs = first.frame.loc[:, list(spec.input_columns)]
        np.testing.assert_array_equal(left.predict(inputs), right.predict(inputs))
        left.estimator.random_state = None
        with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
            fitting.validate_model_state(left, metadata)


@pytest.mark.parametrize("stage", ("candidate", "final"))
@pytest.mark.parametrize("field", ("hyperparameters", "resolved_hyperparameters"))
def test_model_reader_rejects_equal_numeric_values_with_different_parameter_types(
    tmp_path, stage, field
):
    from tests.research.test_quarterly_model_artifacts import _republish

    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={"alpha": 1},
    )
    good = shared_model_case(
        tmp_path / "good", specs=(spec,), development_end="2019-03-31"
    )
    original = {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    }
    mutated = []

    def change(relative, document):
        key = "candidate_models" if stage == "candidate" else "final_models"
        if key in document and not mutated:
            metadata = document[key][0]["model_metadata"]
            metadata[field]["alpha"] = 1.0
            if field == "hyperparameters":
                metadata["resolved_hyperparameters"]["alpha"] = 1.0
            mutated.append(relative)

    bad = _republish(good, tmp_path / "bad", mutate=change)
    assert len(mutated) == 1
    error = (
        "invalid_model_metadata"
        if field == "resolved_hyperparameters"
        else "model_trial_parameters_misaligned"
    )
    with pytest.raises(ValueError, match=f"^{error}$"):
        persisted.read_quarterly_model_artifact(
            bad,
            expected_identity=good.envelope.instance_identity,
            trial_layout=good.layout,
        )
    assert {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    } == original


@pytest.mark.parametrize("solver", ("sag", "saga"))
@pytest.mark.parametrize("boundary", ("resolved", "state_alpha", "state_seed"))
def test_ridge_resolved_and_native_state_parameters_keep_typed_identity(
    solver, boundary
):
    import copy
    import pickle

    import pandas as pd

    from quant_research_platform.contracts.models import validate_model_metadata_core
    from quant_research_platform.research.models.supervised import (
        build_estimator_from_model_spec,
    )

    rng = np.random.default_rng(14)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120),
            "x": rng.normal(size=120),
            "label": rng.normal(size=120),
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=731,
        hyperparameters={"solver": solver, "alpha": 1, "max_iter": 10000, "tol": 1e-8},
    )
    fitted = fitting._fit_model_result(
        fitting._collect_training_frame(
            (frame,), spec, fitting_cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
        ),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    metadata = validate_model_metadata_core(fitted.metadata)
    predictor = pickle.loads(pickle.dumps(fitted.prediction_model))
    fitting.validate_model_state(predictor, metadata)
    original = pickle.dumps(predictor)
    changed = copy.deepcopy(metadata)
    if boundary == "resolved":
        changed["resolved_hyperparameters"]["alpha"] = 1.0
        with pytest.raises(ValueError, match="^invalid_model_metadata$"):
            validate_model_metadata_core(changed)
    else:
        key, value = (
            ("alpha", 1.0) if boundary == "state_alpha" else ("random_state", 731.0)
        )
        predictor.estimator.set_params(**{key: value})
        with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
            fitting.validate_model_state(predictor, metadata)
    fitting.validate_model_state(pickle.loads(original), metadata)


@pytest.mark.parametrize("seed,alias", ((1, True), (0, 0.0)))
def test_ridge_native_seed_type_aliases_are_rejected(seed, alias):
    import pandas as pd

    from quant_research_platform.research.models.supervised import (
        build_estimator_from_model_spec,
    )

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=40),
            "x": np.arange(40.0),
            "label": np.arange(40.0) / 2,
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=seed,
    )
    fitted = fitting._fit_model_result(
        fitting._collect_training_frame(
            (frame,), spec, fitting_cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
        ),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    fitted.prediction_model.estimator.random_state = alias
    with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
        fitting.validate_model_state(fitted.prediction_model, fitted.metadata)


def test_resolved_parameters_preserve_only_legal_thread_resolution_and_float_zero():
    from quant_research_platform.contracts.models import (
        _resolved_model_hyperparameters_match,
    )

    assert _resolved_model_hyperparameters_match(
        "ridge_regression", {"alpha": -0.0}, {"alpha": 0.0}
    )
    assert _resolved_model_hyperparameters_match(
        "random_forest_regressor",
        {"n_jobs": -1, "min_samples_split": 2},
        {"n_jobs": 2, "min_samples_split": 2},
    )
    assert _resolved_model_hyperparameters_match(
        "random_forest_regressor", {"n_jobs": 4}, {"n_jobs": 2}
    )
    assert not _resolved_model_hyperparameters_match(
        "random_forest_regressor", {"n_jobs": 1}, {"n_jobs": 2}
    )
    assert not _resolved_model_hyperparameters_match(
        "random_forest_regressor", {"n_jobs": 2}, {"n_jobs": 2.0}
    )
    assert not _resolved_model_hyperparameters_match(
        "random_forest_regressor",
        {"n_jobs": -1, "min_samples_split": 2},
        {"n_jobs": 2, "min_samples_split": 2.0},
    )


@pytest.mark.parametrize(
    "algorithm",
    (
        "ridge_regression",
        "lasso_regression",
        "huber_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    ),
)
def test_native_estimator_defaults_and_nan_pass_typed_state_checks(algorithm):
    import pickle

    import pandas as pd

    from quant_research_platform.contracts.models import validate_model_metadata_core
    from quant_research_platform.research.models.supervised import (
        build_estimator_from_model_spec,
    )

    rng = np.random.default_rng(27)
    x, z = rng.normal(size=(2, 120))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=120),
            "x": x,
            "z": z,
            "label": 0.3 * x + 0.2 * z + rng.normal(scale=0.01, size=120),
            "eligible": True,
        }
    )
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(days=1)
    params = {}
    if algorithm in {
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }:
        params.update(n_estimators=2, max_depth=2)
    if algorithm in {
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }:
        params["n_jobs"] = 1
    if algorithm == "lightgbm_regressor":
        params["verbosity"] = -1
    if algorithm == "ridge_regression":
        params["alpha"] = 0.0
    spec = ModelSpec(
        model_id="probe",
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        label_column="label",
        random_seed=731,
        hyperparameters=params,
    )
    fitted = fitting._fit_model_result(
        fitting._collect_training_frame(
            (frame,), spec, fitting_cutoff=frame.timestamp.max() + pd.Timedelta(days=2)
        ),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    metadata = validate_model_metadata_core(fitted.metadata)
    predictor = pickle.loads(pickle.dumps(fitted.prediction_model))
    if algorithm == "ridge_regression":
        predictor.estimator.alpha = -0.0
    fitting.validate_model_state(predictor, metadata)


@pytest.mark.parametrize("boundary", ("resolved", "state"))
def test_model_publication_rejects_typed_parameter_drift_and_preserves_good(
    tmp_path, boundary
):
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={"alpha": 1},
    )
    good = shared_model_case(
        tmp_path / "good", specs=(spec,), development_end="2019-03-31"
    )
    original = {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    }

    def wrap(builder):
        def run(**callbacks):
            on_candidate = callbacks["on_candidate"]

            def mutate(fold, family, actual_spec, result):
                if boundary == "resolved":
                    result.metadata["resolved_hyperparameters"]["alpha"] = 1.0
                else:
                    result.prediction_model.estimator.alpha = 1.0
                on_candidate(fold, family, actual_spec, result)

            callbacks["on_candidate"] = mutate
            yield from builder(**callbacks)

        return run

    error = (
        "invalid_model_metadata"
        if boundary == "resolved"
        else "model_state_metadata_mismatch"
    )
    with pytest.raises(ValueError) as caught:
        shared_model_case(
            tmp_path / "bad",
            specs=(spec,),
            development_end="2019-03-31",
            model_results_wrapper=wrap,
        )
    assert caught.value.args == (error,)
    assert {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    } == original
    assert not list((tmp_path / "bad/models").rglob("model.json"))


def test_model_reader_rejects_rehashed_typed_native_state_drift(tmp_path):
    import pickle

    from quant_research_platform.storage.identity import (
        filesystem_mkdir,
        filesystem_write_bytes,
    )
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact_record,
        write_semantic_artifact,
    )

    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={"alpha": 1},
    )
    good = shared_model_case(
        tmp_path / "good", specs=(spec,), development_end="2019-03-31"
    )
    original = {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    }
    changed = []

    def write(stage):
        for member in good.artifact.member_records:
            relative = member["path"]
            payload = Path(native_filesystem_path(good.root / relative)).read_bytes()
            if relative.endswith(".pkl") and not changed:
                predictor = pickle.loads(payload)
                predictor.estimator.alpha = 1.0
                payload = pickle.dumps(predictor)
                changed.append(relative)
            destination = stage / relative
            filesystem_mkdir(destination.parent, parents=True, exist_ok=True)
            filesystem_write_bytes(destination, payload)

    operational = {}
    read_semantic_artifact_record(good.root, operational_metadata=operational)
    bad = write_semantic_artifact(
        tmp_path / "bad",
        good.envelope,
        write,
        operational_metadata=operational,
    )
    assert len(changed) == 1
    fresh = persisted.read_quarterly_model_artifact(
        bad,
        expected_identity=good.envelope.instance_identity,
        trial_layout=good.layout,
    )
    with pytest.raises(ValueError, match="^model_state_metadata_mismatch$"):
        entry = fresh.final_models[0]
        persisted.read_quarterly_predictor(
            fresh, entry["selection_group"], entry["family_id"]
        )
    assert {
        path.relative_to(good.root): path.read_bytes()
        for path in good.root.rglob("*")
        if path.is_file()
    } == original
