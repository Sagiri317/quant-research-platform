from __future__ import annotations

import json
import pickle
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.models import fitting, persisted
from quant_research_platform.research.models.fitting import (
    ModelFitResult,
    training_scope_chunks,
)
from quant_research_platform.research.models.persisted import (
    read_quarterly_model_artifact,
    read_quarterly_predictor,
)
from quant_research_platform.research.models.production import (
    _fit_selected_model_result,
    _refit_model_spec,
)
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
    preflight_model_specs,
)
from quant_research_platform.research.models.trial_selection import (
    select_best_validation_trial,
)
from quant_research_platform.research.predictions import runner
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.experiments import (
    read_model_trial_ledger_reference,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_walk,
    filesystem_write_bytes,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.semantic_artifacts import (
    read_semantic_artifact_operational_metadata,
    write_semantic_artifact,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_result
from tests.research.workflow_test_support import (
    _column_predictor,
    _metadata,
    shared_model_case,
    shared_model_protocol,
    shared_model_rows,
)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


def _mock_prediction_dataset(monkeypatch, frame):
    import pyarrow as pa

    @dataclass
    class Session:
        logical_columns: tuple = ()
        physical_columns: tuple = ()
        owns_inventory: bool = False
        owns_lease: bool = False

        @property
        def arrow_schema(self):
            return pa.Schema.from_pandas(frame, preserve_index=False)

        @property
        def spec(self):
            return {"timestamp_column": "timestamp"}

        @contextmanager
        def filtered_view(self, _window):
            yield self

    @contextmanager
    def open_session(*_args, **_kwargs):
        yield Session()

    monkeypatch.setattr(runner, "open_dataset_session", open_session)
    monkeypatch.setattr(
        runner, "iter_dataset_chunks", lambda *_args, **_kwargs: iter((frame,))
    )


def test_a_share_model_folds_require_formal_history_start_year() -> None:
    protocol = a_share.protocol(scenario_id="pv_all", access_window="development")
    manifest = protocol.manifest()
    sessions = pd.date_range("2017-01-01", "2019-03-31", freq="B")
    with pytest.raises(ValueError, match="model_chronology_formal_history_missing"):
        a_share.model_folds(
            sessions,
            chronology=manifest["model_chronology"],
            formal_dataset_window=manifest["formal_dataset_window"],
            development_window=manifest["development_window"],
        )


def test_fitting_counts_raw_filter_and_label_rows_once() -> None:
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="model_target",
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
            "forward_return_20d_available_at": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2019-01-03", "2018-01-04"]
            ),
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        result = fitting._fit_model_result_owned_from_frames(  # noqa: SLF001
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2018-12-31",
        )
        model_result = result.value if isinstance(result, OwnedValue) else result
        try:
            assert model_result.metadata["rows_input"] == 4
            assert model_result.metadata["rows_after_filter_mode"] == 3
            assert model_result.metadata["rows_after_input_label_dropna"] == 2
        finally:
            if isinstance(result, OwnedValue) and result.lease.active:
                result.release(owner="test.model.fitting")


def test_training_asset_scope_owns_scoped_payload_until_consumer_release() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        scoped = training_scope_chunks((frame,), asset_ids=("b",))
        selected = next(scoped)
        assert selected["asset_id"].tolist() == ["b"]
        assert policy.used_memory_bytes > 0
        scoped.close()
        assert policy.used_memory_bytes == 0


def test_training_asset_scope_preserves_consumer_primary_on_early_close() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "value": [1.0, 2.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=8 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )
    primary = RuntimeError("consumer_primary")
    with runtime_resource_policy_scope(policy):
        scoped = training_scope_chunks((frame,), asset_ids=("b",))
        next(scoped)
        with pytest.raises(RuntimeError) as raised:
            scoped.throw(primary)
        assert raised.value is primary
        assert policy.used_memory_bytes == 0


def test_rank_missing_contributes_neutral_without_renormalization() -> None:
    predictor = RankEnsemblePredictor(
        columns=("x", "y"),
        directions=(1.0, 1.0),
    )
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
        {
            "trial_id": ["defined", "undefined"],
            "validation_rankic": [-0.4, np.nan],
        }
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


def test_seed_changes_by_quarter_component_and_refit_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trial_q1 = derive_seed(42, "a_share_model_trial", "ridge", "config", "2019Q1", 0)
    trial_q2 = derive_seed(42, "a_share_model_trial", "ridge", "config", "2019Q2", 0)
    refit_q1 = derive_seed(42, "a_share_model_refit", "ridge", "config", "2019Q1")
    assert trial_q1 == derive_seed(
        42, "a_share_model_trial", "ridge", "config", "2019Q1", 0
    )
    assert len({trial_q1, trial_q2, refit_q1}) == 3
    base = ModelSpec(
        model_id="ridge_family",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=999,
    )
    refits = [
        _refit_model_spec(
            base,
            n_train=10,
            candidate_parameters={"regularization_strength": 0.1},
            quarter_id=quarter,
            root_seed=42,
        )
        for quarter in ("2019Q1", "2019Q2")
    ]
    assert refits[0].random_seed != refits[1].random_seed
    assert refits[0].random_seed != base.random_seed
    assert dict(refits[0].hyperparameters) == {"alpha": 1.0}

    captured: list[int] = []

    def fit_owned(_frames, spec, **_kwargs):  # noqa: ANN001, ANN003
        captured.append(spec.random_seed)
        return ModelFitResult(
            metadata=_metadata(("x",)),
            prediction_model=_column_predictor("x"),
        )

    monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", fit_owned)
    _fit_selected_model_result(
        training_counts={},
        dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
        selected_spec=base,
        candidate_parameters=dict(base.hyperparameters),
        chunks=lambda *_args, **_kwargs: (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2018-01-01"]),
                    "asset_id": ["a"],
                    "x": [1.0],
                    "label": [2.0],
                    "label_available_at": pd.to_datetime(["2018-01-01"]),
                    "eligible": [True],
                }
            ),
        ),
        fitting_cutoff="2018-12-31",
        quarter_id="2019Q1",
        root_seed=42,
    )
    assert captured == [
        derive_seed(
            42, "a_share_model_refit", "ridge_regression", "ridge_family", "2019Q1"
        )
    ]


def test_a_share_evaluation_catalog_contains_all_formal_scenarios() -> None:
    assert {
        "pv_all_evaluation",
        "pv_expanded_evaluation",
        "fusion_nf_core_evaluation",
        "fusion_nf_expectations_evaluation",
    } == {spec.evaluation_id for spec in research_profile().evaluations.list()}


def test_convergence_warning_is_a_model_fit_failure() -> None:
    from sklearn.exceptions import ConvergenceWarning

    class WarningEstimator:
        def fit(self, _matrix, _target, **_kwargs):
            warnings.warn("not converged", ConvergenceWarning, stacklevel=2)

    spec = ModelSpec(
        model_id="ridge_warning",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
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
            training=frame,
            y=np.array([1.0, 2.0]),
            estimator=WarningEstimator(),
            threads=1,
        )


@pytest.mark.parametrize("max_train_rows", (None, 4))
def test_linear_workspace_cap_rejects_before_weights_and_solver(
    monkeypatch, max_train_rows
):
    from tests.helpers.runtime_resources import capture_test_policy

    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=max_train_rows,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2017-01-02", periods=8),
            "label_available_at": pd.bdate_range("2017-01-02", periods=8),
            "x": np.arange(8.0),
            "label": np.arange(8.0) * 2 + 1,
            "eligible": True,
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=1_000_000, explicit_cap_bytes=200_000
    )
    actual_collect = fitting._collect_training_frame_owned
    blockers = []

    def collect(*args, **kwargs):
        owned = actual_collect(*args, **kwargs)
        assert policy.used_memory_bytes == (
            owned.lease.reserved_bytes
            + fitting._linear_model_state_upper_bound_bytes(spec)
        )
        blockers.append(
            policy.lease("test.blocker", 200_000 - policy.used_memory_bytes - 1)
        )
        return owned

    def forbidden(*args, **kwargs):
        raise AssertionError("linear_allocation_before_admission")

    monkeypatch.setattr(fitting, "_collect_training_frame_owned", collect)
    monkeypatch.setattr(fitting, "equal_timestamp_sample_weights", forbidden)
    monkeypatch.setattr(fitting.np.linalg, "lstsq", forbidden)
    with runtime_resource_policy_scope(policy):
        try:
            with pytest.raises(
                ValueError,
                match="model_training_working_set_exceeds_task_memory_budget",
            ):
                fitting._fit_linear_model_result_owned_from_frames(
                    (frame,), spec, fitting_cutoff="2018-12-31"
                )
            assert policy.used_memory_bytes == blockers[0].reserved_bytes
        finally:
            for blocker in blockers:
                blocker.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("boundary", ("soft", "captured", "explicit"))
def test_estimator_admission_uses_capacity_not_soft_thread_planning(
    monkeypatch, boundary
):
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024**2,
        explicit_cap_bytes=12 * 1024**2 if boundary == "explicit" else None,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-01-01", periods=1000),
            "label_available_at": pd.date_range("2015-01-01", periods=1000),
            "x": np.arange(1000, dtype="float64"),
            "label": np.arange(1000, dtype="float64") / 10,
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
    )
    estimator = build_estimator_from_model_spec(spec)
    original_collect = fitting._collect_training_frame_owned
    original_fit = fitting._fit_predictor
    blockers = []
    fits = []

    def collect(*args, **kwargs):
        owned = original_collect(*args, **kwargs)
        training = owned.value[0]
        required = (
            len(training) * 8
            + fitting._model_state_upper_bound_bytes(
                training, spec, estimator=estimator
            )
            + fitting._estimator_fit_working_set_bytes(training, spec, threads=1)
        )
        limit = (
            policy.memory_budget_bytes
            if boundary == "soft"
            else policy.captured_available_memory_bytes
            if boundary == "captured"
            else policy.explicit_memory_cap_bytes
        )
        blockers.append(
            policy.lease(
                "test.other_work",
                limit - policy.used_memory_bytes - required + required // 4,
            )
        )
        assert policy.used_memory_bytes + required > limit
        return owned

    def fit(*args, **kwargs):
        fits.append(kwargs["threads"])
        return original_fit(*args, **kwargs)

    with runtime_resource_policy_scope(policy):
        baseline = fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=build_estimator_from_model_spec(spec),
            fitting_cutoff="2018-12-31",
        )
        try:
            expected = baseline.value.prediction_model.predict(
                frame.loc[:, spec.input_columns]
            )
        finally:
            baseline.release(owner="test.baseline")
        monkeypatch.setattr(fitting, "_collect_training_frame_owned", collect)
        monkeypatch.setattr(fitting, "_fit_predictor", fit)
        try:
            if boundary == "soft":
                fitted = fitting._fit_model_result_owned_from_frames(
                    (frame,), spec, estimator=estimator, fitting_cutoff="2018-12-31"
                )
                try:
                    np.testing.assert_array_equal(
                        fitted.value.prediction_model.predict(
                            frame.loc[:, spec.input_columns]
                        ),
                        expected,
                    )
                finally:
                    fitted.release(owner="test.soft.fit")
                assert fits == [1]
            else:
                with pytest.raises(
                    ValueError,
                    match="model_training_working_set_exceeds_task_memory_budget",
                ):
                    fitting._fit_model_result_owned_from_frames(
                        (frame,), spec, estimator=estimator, fitting_cutoff="2018-12-31"
                    )
                assert fits == []
            assert policy.used_memory_bytes == blockers[0].reserved_bytes
        finally:
            for blocker in blockers:
                blocker.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cap", (None, 13))
@pytest.mark.parametrize("chunk_rows", (5, 17))
def test_training_retention_preserves_trailing_rows_and_same_day_order(cap, chunk_rows):
    from tests.helpers.runtime_resources import capture_test_policy

    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        label_column="label",
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
    policy = capture_test_policy(available_memory_bytes=1024**2)
    with runtime_resource_policy_scope(policy):
        owned = fitting._collect_training_frame_owned(
            (
                frame.iloc[start : start + chunk_rows]
                for start in range(0, 32, chunk_rows)
            ),
            spec,
            fitting_cutoff="2017-12-31",
        )
        try:
            actual, _, _ = owned.value
            pd.testing.assert_frame_equal(
                actual, expected.reset_index(drop=True), check_exact=True
            )
            assert policy.used_memory_bytes == owned.lease.reserved_bytes
        finally:
            owned.lease.close()
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "fraction,depth,expected_nodes", ((0.25, 6, 7), (0.0001, 2, 7), (1e-320, 3, 15))
)
def test_rf_state_bound_uses_declared_leaf_weight_without_losing_row_or_depth_limits(
    fraction, depth, expected_nodes
):
    frame = pd.DataFrame({"x": np.arange(128, dtype="float64")})
    spec = ModelSpec(
        model_id="rf",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("x",),
        label_column="label",
        hyperparameters={
            "n_estimators": 3,
            "max_depth": depth,
            "min_weight_fraction_leaf": fraction,
            "bootstrap": False,
        },
        random_seed=42,
    )
    estimator = build_estimator_from_model_spec(spec)
    with runtime_resource_policy_scope() as policy:
        with policy.lease("test.rf_state", 1024**2):
            estimator.fit(
                frame, np.sin(frame.x), sample_weight=np.linspace(1.0, 2.0, len(frame))
            )
            assert all(
                tree.tree_.node_count <= expected_nodes
                for tree in estimator.estimators_
            )
            assert fitting._model_state_upper_bound_bytes(
                frame, spec, estimator=estimator
            ) == 3 * (4096 + expected_nodes * 256)
        assert policy.used_memory_bytes == 0


def test_estimator_without_fit_cannot_be_returned_as_fitted():
    spec = ModelSpec(
        model_id="missing_fit",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
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
            training=frame,
            y=frame.label.to_numpy(),
            estimator=object(),
            threads=1,
        )


@pytest.mark.parametrize("arrow_features", [False, True])
@pytest.mark.parametrize("missing_position", [0, 200, 501, 997, 999])
def test_training_admits_materialized_filter_index_without_changing_rows(
    missing_position,
    arrow_features,
):
    from tests.helpers.runtime_resources import capture_test_policy

    positions = np.arange(1000, dtype="float64")
    features = tuple(f"x{index}" for index in range(6))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=1000, freq="100ms"),
            "label_available_at": pd.Timestamp("2024-01-02"),
            "asset_id": ["asset_b", "asset_a"] * 500,
            "eligible": True,
            "label": positions * 2 + 3,
            **{name: positions + index for index, name in enumerate(features)},
        }
    )
    if arrow_features:
        frame = frame.astype({name: "float64[pyarrow]" for name in features})
    frame.loc[missing_position, "label"] = np.nan
    original = frame.copy(deep=True)
    expected_positions = [
        position for position in range(1000) if position != missing_position
    ]
    spec = ModelSpec(
        model_id="sparse_training",
        model_algorithm="linear_regression",
        input_factor_columns=features,
        label_column="label",
    )
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    with runtime_resource_policy_scope(policy):
        owned = fitting._collect_training_frame_owned(
            [frame], spec, fitting_cutoff="2024-01-03"
        )
        try:
            actual, counts, _metadata = owned.value
            assert actual.x0.tolist() == expected_positions
            assert actual.label.tolist() == [
                position * 2 + 3 for position in expected_positions
            ]
            assert actual.timestamp.tolist() == [
                frame.timestamp.iloc[position] for position in expected_positions
            ]
            assert counts["rows_after_input_label_dropna"] == 999
            pd.testing.assert_frame_equal(frame, original, check_exact=True)
            assert policy.used_memory_bytes == owned.lease.reserved_bytes
            actual = None
        finally:
            owned.release(owner="test.sparse_training")
    assert policy.used_memory_bytes == 0


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
    ) as caught:
        estimator.fit(frame, frame.label.to_numpy())
    from quant_research_platform.workflows.research import _semantic_failure_code

    assert _semantic_failure_code(caught.value) == str(caught.value)
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


def _prediction_result(
    case,
    policy,
    *,
    group="global",
    start="2019-01-01",
    end="2019-06-30",
    access="development",
):
    return runner.build_prediction_result_from_persisted_model(
        layout=case.layout,
        dataset_catalog=ResearchDatasetCatalog((case.dataset,)),
        prediction_spec=PredictionSpec(
            prediction_id=f"prediction_{group}",
            model_id="model-artifact",
            research_dataset_id=case.dataset.data_id,
            train_end=None,
            test_start=start,
            test_end=end,
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        ),
        persisted_model=case.artifact,
        resource_policy=policy,
        selection_group=group,
        access_window=access,
    )


def _republish(case, logical, *, mutate, operational=None):
    def write(stage):
        for member in case.artifact.member_records:
            relative = member["path"]
            destination = stage / relative
            filesystem_mkdir(destination.parent, parents=True, exist_ok=True)
            payload = filesystem_read_bytes(case.root / relative)
            if relative.endswith(".json"):
                document = json.loads(payload)
                mutate(relative, document)
                payload = json.dumps(document).encode()
            filesystem_write_bytes(destination, payload)

    return write_semantic_artifact(
        logical,
        case.envelope,
        write,
        operational_metadata=operational
        or read_semantic_artifact_operational_metadata(case.root),
    )


def test_candidates_fit_once_and_both_groups_have_real_final_models(
    tmp_path, monkeypatch
):
    specs = (
        ModelSpec(
            model_id="linear_x",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
        ModelSpec(
            model_id="ridge_y",
            model_algorithm="ridge_regression",
            input_factor_columns=("y",),
            label_column="label",
        ),
    )
    configurations = [
        {"model_id": specs[0].model_id, "hyperparameters": {}},
        {
            "model_id": specs[1].model_id,
            "hyperparameters": {},
            "parameter_candidates": [
                {"regularization_strength": 0.01},
                {"regularization_strength": 0.1},
            ],
        },
    ]
    calls = []
    original = fitting._fit_model_result_owned_from_frames

    def fit(frames, spec, **kwargs):
        calls.append((spec.model_id, kwargs["fitting_cutoff"]))
        return original(frames, spec, **kwargs)

    monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", fit)
    preflight_model_specs(specs)
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(
            tmp_path,
            specs=specs,
            protocol=shared_model_protocol(specs, configurations=configurations),
        )
        assert len(calls) == 6 * 3 + 2 * 2
        assert len(case.artifact.candidate_models) == 18
        assert len(case.artifact.final_models) == 4
        assert set(case.artifact.selections["global"]) == {"final"}
        assert set(case.artifact.selections["quarterly"]) == {
            "2019Q1",
            "2019Q2",
            "final",
        }
        assert {entry["quarter_id"] for entry in case.artifact.candidate_models} == {
            "2018Q1",
            "2018Q2",
            "2018Q3",
            "2018Q4",
            "2019Q1",
            "2019Q2",
        }
        for family in specs:
            entries = [
                entry
                for entry in case.artifact.final_models
                if entry["family_id"] == family.model_id
            ]
            assert (
                entries[0]["fit_windows"]
                == entries[1]["fit_windows"]
                == [{"start": "2014-01-01", "end": "2019-06-30"}]
            )
            assert entries[0]["model_metadata"] == entries[1]["model_metadata"]
            with read_quarterly_predictor(
                case.artifact, "global", family.model_id, resource_policy=policy
            ) as g:
                with read_quarterly_predictor(
                    case.artifact, "quarterly", family.model_id, resource_policy=policy
                ) as q:
                    np.testing.assert_array_equal(
                        g.predict(case.frame.loc[:, list(family.input_columns)]),
                        q.predict(case.frame.loc[:, list(family.input_columns)]),
                    )
        for reference in case.artifact.trial_ledgers:
            with read_model_trial_ledger_reference(case.layout, reference) as (
                plan,
                records,
            ):
                if plan.stage == "refit":
                    assert plan.folds == ()
                    assert plan.selection_group in {"global", "quarterly"}
                    assert plan.selection_metric is None
                    assert set(records[0].metrics) == {"n_train"}
        assert policy.used_memory_bytes == 0


def test_candidate_own_prediction_rows_survive_shared_scoring_domain(tmp_path):
    specs = tuple(
        ModelSpec(
            model_id=f"linear_{column}",
            model_algorithm="linear_regression",
            input_factor_columns=(column,),
            label_column="label",
        )
        for column in ("x", "y")
    )
    frame = shared_model_rows(specs)
    frame.loc[frame.asset_id.eq("asset_0"), "y"] = np.nan
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path, specs=specs, frame=frame)
        for family, count in (("linear_x", 8), ("linear_y", 7)):
            entry = next(
                e
                for e in case.artifact.candidate_models
                if e["quarter_id"] == "2019Q1" and e["family_id"] == family
            )
            saved = pd.concat(case.artifact.validation_chunks(entry), ignore_index=True)
            assert saved.groupby("timestamp").size().tolist() == [count, count]
            assert saved.groupby("timestamp").score_eligible.sum().tolist() == [7, 7]
        for group in ("global", "quarterly"):
            result = _prediction_result(case, policy, group=group)
            bundle = write_prediction_result(case.layout, result)
            prediction = pd.read_parquet(bundle / "prediction.parquet")
            assert (
                prediction.groupby(["timestamp", "family_id"])
                .size()
                .unstack()["linear_x"]
                .tolist()
                == [8] * 4
            )
            assert (
                prediction.groupby(["timestamp", "family_id"])
                .size()
                .unstack()["linear_y"]
                .tolist()
                == [7] * 4
            )
        for reference in case.artifact.trial_ledgers:
            with read_model_trial_ledger_reference(case.layout, reference) as (
                plan,
                records,
            ):
                if plan.stage == "selection":
                    # Only the first fixture session matures by each fold's last day.
                    assert records[0].metrics["validation_rows"] == 7
        assert policy.used_memory_bytes == 0


def test_common_pk_readback_is_independent_of_chunking(tmp_path):
    with runtime_resource_policy_scope() as policy:
        first = shared_model_case(tmp_path / "single", chunk_size=1)
        second = shared_model_case(tmp_path / "whole", chunk_size=1024)
        for group in ("global", "quarterly"):
            for key, decision in first.artifact.selections[group].items():
                other = second.artifact.selections[group][key]
                assert (
                    decision["common_validation_coverage"]
                    == other["common_validation_coverage"]
                )
                assert decision["families"] == other["families"]
        assert policy.used_memory_bytes == 0


def test_future_validation_changes_do_not_change_prior_quarter_selection(tmp_path):
    specs = (
        ModelSpec(
            model_id="positive",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            label_column="label",
        ),
        ModelSpec(
            model_id="negative",
            model_algorithm="linear_regression",
            input_factor_columns=("y",),
            label_column="label",
        ),
    )
    original = shared_model_rows(specs)
    original["y"] = np.where(
        original.timestamp.lt("2019-01-01"), original.x, -original.x
    )
    changed = original.copy()
    changed.loc[changed.timestamp.ge("2019-04-01"), "label"] *= -13.0
    with runtime_resource_policy_scope() as policy:
        before = shared_model_case(tmp_path / "before", specs=specs, frame=original)
        after = shared_model_case(tmp_path / "after", specs=specs, frame=changed)
        for quarter in ("2019Q1", "2019Q2"):
            assert (
                before.artifact.selections["quarterly"][quarter]
                == after.artifact.selections["quarterly"][quarter]
            )
        assert (
            before.artifact.selections["global"]["final"]
            != after.artifact.selections["global"]["final"]
        )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("group", ["global", "quarterly"])
def test_prediction_request_clips_saved_rows_without_retraining(
    tmp_path, monkeypatch, group
):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path)

        def forbidden(*args, **kwargs):
            raise AssertionError("development_prediction_refit")

        monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", forbidden)
        result = _prediction_result(
            case, policy, group=group, start="2019-01-03", end="2019-04-02"
        )
        bundle = write_prediction_result(case.layout, result)
        frame = pd.read_parquet(bundle / "prediction.parquet")
        assert frame.timestamp.min() >= pd.Timestamp("2019-01-03")
        assert frame.timestamp.max() <= pd.Timestamp("2019-04-02")
        assert frame.model_id.eq("linear_regression").all()
        assert policy.used_memory_bytes == 0


def test_prediction_rejects_uncovered_quarter(tmp_path):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path)
        with pytest.raises(PermissionError, match="a_share_prediction_window_invalid"):
            _prediction_result(case, policy, start="2018-10-01")
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "candidate_seed",
        "candidate_columns",
        "candidate_rows",
        "candidate_trial",
        "quarter_coordinate",
        "quarter_matrix",
        "selection_winner",
        "selection_score",
        "selection_cutoff",
        "selection_coverage",
        "final_group",
        "final_cutoff",
        "final_seed",
        "final_fit_window",
        "final_trial",
        "unknown_member",
    ],
)
def test_reader_rejects_self_consistent_publication_drift(tmp_path, mutation):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path / "good")
        old_good = _file_bytes(case.root)

        def change(relative, payload):
            if relative == "models/2019Q1/metadata.json":
                entry = payload["candidate_models"][0]
                if mutation == "candidate_seed":
                    entry["model_metadata"]["random_seed"] += 1
                elif mutation == "candidate_columns":
                    entry["model_metadata"]["input_columns"] = ["other"]
                elif mutation == "candidate_rows":
                    entry["model_metadata"]["rows_after_input_label_dropna"] -= 1
                elif mutation == "candidate_trial":
                    entry["trial_semantic_identity"] = "e" * 64
                elif mutation == "quarter_coordinate":
                    entry["quarter_id"] = "2019Q2"
                elif mutation == "quarter_matrix":
                    payload["candidate_models"] = []
            if relative == "selections/global/final.json":
                decision = payload["selection"]
                if mutation == "selection_winner":
                    decision["selected_model_id"] = "other"
                elif mutation == "selection_score":
                    decision["families"][0]["trial_metrics"][0][
                        "validation_rankic"
                    ] = -0.75
                elif mutation == "selection_cutoff":
                    decision["cutoff"] = "2024-01-01"
                elif mutation == "selection_coverage":
                    decision["common_validation_coverage"]["row_count"] += 1
            if relative == "models/final/global/metadata.json":
                entry = payload["final_models"][0]
                if mutation == "final_group":
                    entry["selection_group"] = "quarterly"
                elif mutation == "final_cutoff":
                    entry["fit_cutoff"] = "2024-01-01"
                elif mutation == "final_seed":
                    entry["model_metadata"]["random_seed"] += 1
                elif mutation == "final_fit_window":
                    entry["fit_windows"][0]["start"] = "2015-01-01"
                elif mutation == "final_trial":
                    entry["refit_trial_semantic_identity"] = "e" * 64
            if relative == "model.json" and mutation == "unknown_member":
                payload["extra"] = True

        root = _republish(case, tmp_path / "corrupt", mutate=change)
        with pytest.raises(ValueError):
            read_quarterly_model_artifact(root, trial_layout=case.layout)
        assert _file_bytes(case.root) == old_good
        assert policy.used_memory_bytes == 0


def test_model_member_mutation_is_rejected_before_loading_predictor(
    tmp_path, monkeypatch
):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path)
        path = case.root / "models/final/global/linear_regression.pkl"
        filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")
        monkeypatch.setattr(
            persisted.pickle, "load", lambda *_: pytest.fail("load before identity")
        )
        with pytest.raises(ValueError, match="model_artifact_model_invalid"):
            with read_quarterly_predictor(
                case.artifact, "global", "linear_regression", resource_policy=policy
            ):
                pytest.fail("mutated member accepted")
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("consumer", ["staged", "persisted", "crypto"])
@pytest.mark.parametrize(
    "failure", ["open", "load", "close", "missing", "notdir", "corrupt"]
)
def test_model_member_io_preserves_failure_class(
    tmp_path, monkeypatch, consumer, failure
):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path / "good")
        original_bytes = _file_bytes(case.root)
        metadata = case.artifact.final_models[0]["model_metadata"]
        if consumer == "crypto":

            def write(stage):
                target = stage / "models/final/linear.pkl"
                filesystem_mkdir(target.parent, parents=True)
                filesystem_write_bytes(
                    target,
                    filesystem_read_bytes(
                        case.root / "models/final/global/linear_regression.pkl"
                    ),
                )

            crypto_root = write_semantic_artifact(
                tmp_path / "crypto", case.envelope, write
            )
            from quant_research_platform.storage.semantic_artifacts import (
                read_semantic_artifact_record,
            )

            identities = {}
            _, members = read_semantic_artifact_record(
                crypto_root, member_identities=identities
            )
            crypto = SimpleNamespace(
                instance_root=crypto_root,
                member_records=members,
                member_identities=identities,
            )
            entry = {"model_id": "linear", "model_metadata": metadata}
        primary = (
            FileNotFoundError("model_missing")
            if failure == "missing"
            else NotADirectoryError("model_not_directory")
            if failure == "notdir"
            else pickle.UnpicklingError("model_corrupt")
            if failure == "corrupt"
            else OSError(5, "model_system_io")
        )
        actual_open, actual_load = persisted.open_filesystem_file, persisted.pickle.load
        handles = []

        class CloseFailure:
            def __init__(self, handle):
                self.handle = handle

            def __getattr__(self, name):
                return getattr(self.handle, name)

            def close(self):
                self.handle.close()
                raise primary

        def open_member(path, mode="rb", **kwargs):
            if Path(path).suffix != ".pkl" or mode != "rb":
                return actual_open(path, mode, **kwargs)
            if failure in {"open", "missing", "notdir"}:
                raise primary
            handle = actual_open(path, mode, **kwargs)
            handles.append(handle)
            return CloseFailure(handle) if failure == "close" else handle

        def load_member(handle):
            if failure in {"load", "corrupt"}:
                raise primary
            return actual_load(handle)

        monkeypatch.setattr(persisted, "open_filesystem_file", open_member)
        monkeypatch.setattr(persisted.pickle, "load", load_member)
        invalid = failure in {"missing", "notdir", "corrupt"}
        with pytest.raises(ValueError if invalid else OSError) as caught:
            if consumer == "staged":
                shared_model_case(tmp_path / "staged")
            else:
                context = (
                    persisted.read_crypto_model_member(
                        crypto, entry, resource_policy=policy
                    )
                    if consumer == "crypto"
                    else read_quarterly_predictor(
                        case.artifact,
                        "global",
                        "linear_regression",
                        resource_policy=policy,
                    )
                )
                with context:
                    pytest.fail("model I/O failure swallowed")
        assert caught.value.__cause__ is primary if invalid else caught.value is primary
        assert all(handle.closed for handle in handles)
        assert _file_bytes(case.root) == original_bytes
        assert not tuple(filesystem_glob(tmp_path / "staged", "**/.a-*"))
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cancellation", [False, True])
def test_refit_scaler_failure_is_recorded_before_learning(
    tmp_path, monkeypatch, cancellation
):
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.storage.experiments import read_trial_ledger

    attempts = []
    original_begin, original_fit = TrialRecorder.begin, fitting.ModelMatrixScaler.fit

    def begin(recorder, *args, **kwargs):
        record = original_begin(recorder, *args, **kwargs)
        attempts.append((recorder, record))
        return record

    primary = (
        KeyboardInterrupt("refit_scaler_oracle")
        if cancellation
        else RuntimeError("refit_scaler_oracle")
    )

    def fit(*args, **kwargs):
        if attempts and attempts[-1][0].plan.stage == "refit":
            raise primary
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(TrialRecorder, "begin", begin)
    monkeypatch.setattr(fitting.ModelMatrixScaler, "fit", fit)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(type(primary)) as caught:
            shared_model_case(tmp_path)
        assert caught.value is primary
        recorder, _ = attempts[-1]
        with read_trial_ledger(recorder.root, recorder.plan) as (_, events):
            assert events[-1].status == ("abandoned" if cancellation else "failed")
            assert events[-1].metrics == {}
            assert events[-1].undefined_reasons == {"failure": type(primary).__name__}
        assert not tuple(filesystem_glob(tmp_path / "models", "**/.a-*"))
        assert policy.used_memory_bytes == 0


def test_failed_source_verification_preserves_published_model_and_releases_state(
    tmp_path,
):
    with runtime_resource_policy_scope() as policy:
        good = shared_model_case(tmp_path / "good")
        before = _file_bytes(good.root)
        primary = RuntimeError("source_identity_changed")

        def verify():
            raise primary

        with pytest.raises(RuntimeError) as caught:
            shared_model_case(tmp_path / "failed", verify_source=verify)
        assert caught.value is primary
        assert _file_bytes(good.root) == before
        assert not tuple(filesystem_glob(tmp_path / "failed/models", "**/.a-*"))
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("max_train_rows", [None, 12])
def test_final_refit_strength_and_state_follow_actual_rows(tmp_path, max_train_rows):
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        max_train_rows=max_train_rows,
    )
    configuration = [
        {
            "model_id": "ridge",
            "hyperparameters": {},
            "parameter_candidates": [{"regularization_strength": 0.1}],
        }
    ]
    preflight_model_specs((spec,))
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(
            tmp_path,
            specs=(spec,),
            protocol=shared_model_protocol((spec,), configurations=configuration),
        )
        for entry in case.artifact.final_models:
            count = entry["model_metadata"]["rows_after_input_label_dropna"]
            assert count == (
                len(case.frame) if max_train_rows is None else max_train_rows
            )
            assert entry["model_metadata"]["hyperparameters"]["alpha"] == pytest.approx(
                count * 0.1
            )
            with read_quarterly_predictor(
                case.artifact, entry["selection_group"], "ridge", resource_policy=policy
            ) as predictor:
                assert np.isfinite(
                    predictor.predict(case.frame.loc[:, list(spec.input_columns)])
                ).all()
        assert policy.used_memory_bytes == 0


def test_disk_admission_precedes_pickle_and_preserves_old_good(tmp_path, monkeypatch):
    from quant_research_platform.runtime import resources

    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path / "good")
        before = _file_bytes(case.root)
        calls = []
        monkeypatch.setattr(
            persisted.pickle, "dump", lambda *_args, **_kwargs: calls.append(True)
        )
        monkeypatch.setattr(
            resources.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1)
        )
        with pytest.raises(RuntimeError, match="insufficient_free_disk"):
            shared_model_case(tmp_path / "failed")
        assert calls == []
        assert _file_bytes(case.root) == before
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "failure_type", [FileNotFoundError, NotADirectoryError, OSError]
)
def test_final_model_post_load_stat_preserves_system_error(
    tmp_path, monkeypatch, failure_type
):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path)
        primary = failure_type("model_stat_failure")
        actual_load, actual_stat = persisted.pickle.load, persisted.os.stat
        armed = False

        def load(handle):
            nonlocal armed
            value = actual_load(handle)
            armed = True
            return value

        def stat(path, *args, **kwargs):
            if armed and not isinstance(path, int) and Path(path).suffix == ".pkl":
                raise primary
            return actual_stat(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(persisted.pickle, "load", load)
            patch.setattr(persisted.os, "stat", stat)
            with pytest.raises(
                ValueError if failure_type is not OSError else OSError
            ) as caught:
                with read_quarterly_predictor(
                    case.artifact, "global", "linear_regression", resource_policy=policy
                ):
                    pass
            assert armed
            if failure_type is OSError:
                assert caught.value is primary
            else:
                assert caught.value.__cause__ is primary
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "mutation",
    ("part", "spec", "manifest", "unselected_year", "cancel", "exception", "staged"),
)
def test_model_selection_reuses_snapshot_and_rejects_between_scan_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.datasets.io import (
        _research_dataset_access_window,
    )
    from quant_research_platform.research.models import production
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from quant_research_platform.storage.datasets.writing import write_dataset

    dataset = make_demo_research_dataset_spec(
        data_id="snapshot",
        factor_columns=("x",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("label_available_at",),
    )
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 3,
            "asset_id": ["a"] * 3,
            "timestamp": pd.to_datetime(["2014-01-02", "2018-01-02", "2024-01-02"]),
            "label_available_at": pd.to_datetime(
                ["2014-02-03", "2018-02-02", "2024-02-02"]
            ),
            "x": [1.0, 2.0, 3.0],
            "label": [0.1, 0.2, 0.3],
            "eligible": True,
        }
    )
    frame = frame.astype(
        {"timestamp": "datetime64[ns]", "label_available_at": "datetime64[ns]"}
    )
    layout = StorageLayout(tmp_path)
    root = layout.research_dataset_root(dataset.market_id, dataset.data_id)
    with runtime_resource_policy_scope() as policy:
        write_dataset(
            root, frame, spec=dataset, resource_policy=policy, borrowed_input=True
        )
        with open_dataset_session(
            root,
            columns=None,
            time_window=_research_dataset_access_window(dataset.market_id, None),
            resource_policy=policy,
        ) as session:
            scratch = session.inventory.scratch_path
            baseline = policy.used_memory_bytes
            observed = []

            def exercise(**kwargs):
                assert list(kwargs["sessions"]) == list(frame["timestamp"][:2])

                def read():
                    return kwargs["chunks"](
                        tuple(frame.columns),
                        input_columns=("x",),
                        start=None,
                        end="2018-12-31",
                    )

                first = pd.concat(read(), ignore_index=True)
                pd.testing.assert_frame_equal(
                    first, frame.iloc[:2].reset_index(drop=True)
                )
                observed.append(True)
                if mutation == "staged":
                    good = shared_model_case(
                        tmp_path / "good-model", verify_source=session.verify
                    )
                    old_good = _file_bytes(good.root)

                    def mutate_after_fit(actual):
                        def results(**callbacks):
                            yield from actual(**callbacks)
                            path = root / "spec.json"
                            filesystem_write_bytes(
                                path, filesystem_read_bytes(path) + b" "
                            )

                        return results

                    with pytest.raises(
                        ValueError, match="dataset_spec_identity_changed"
                    ):
                        shared_model_case(
                            tmp_path / "failed-model",
                            verify_source=session.verify,
                            model_results_wrapper=mutate_after_fit,
                        )
                    assert _file_bytes(good.root) == old_good
                    return ()
                if mutation in {"cancel", "exception"}:
                    iterator = iter(read())
                    try:
                        next(iterator)
                        if mutation == "exception":
                            raise RuntimeError("consumer_failure")
                    finally:
                        iterator.close()
                    return ()
                if mutation == "part":
                    part = session.read_paths[0]
                    filesystem_write_bytes(part, filesystem_read_bytes(part) + b" ")
                elif mutation in {"spec", "manifest"}:
                    path = (
                        root / "spec.json"
                        if mutation == "spec"
                        else root / "_manifest" / "manifest.json"
                    )
                    filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")
                else:
                    path = root / "data" / "year=2025" / "part-00000.parquet"
                    filesystem_mkdir(path.parent)
                    filesystem_write_bytes(
                        path, filesystem_read_bytes(session.all_paths[0])
                    )
                return tuple(read())

            monkeypatch.setattr(production, "_build_a_share_models", exercise)

            def run():
                return production.build_model_selection_result(
                    layout=layout,
                    dataset_catalog=ResearchDatasetCatalog((dataset,)),
                    model_catalog=research_profile().models,
                    dataset_id=dataset.data_id,
                    model_ids=("linear",),
                    tie_break_order=("linear",),
                    time_window=TimeWindow("2019-01-01", "2019-03-31"),
                    resource_policy=policy,
                    protocol_manifest={
                        **a_share.protocol("pv_all").manifest(),
                        "research_configuration": {
                            "models": [{"model_id": "linear", "hyperparameters": {}}]
                        },
                    },
                    run_id="snapshot",
                    research_protocol_identity="a" * 64,
                    model_target_id="snapshot_model",
                    backend_versions={},
                    dataset_session=session,
                    on_validation=lambda *_args: pytest.fail("unexpected_validation"),
                    on_candidate=lambda *_args: pytest.fail("unexpected_candidate"),
                    on_trial=lambda *_args: pytest.fail("unexpected_trial"),
                    select_models=lambda *_args: pytest.fail("unexpected_selection"),
                    on_observation=lambda *_args: pytest.fail("unexpected_observation"),
                )

            if mutation in {"cancel", "staged"}:
                assert run() == ()
            elif mutation == "exception":
                with pytest.raises(RuntimeError, match="^consumer_failure$"):
                    run()
            else:
                with pytest.raises(
                    ValueError,
                    match="dataset_(file_identity|spec_identity|manifest_identity|inventory)_changed",
                ):
                    run()
            assert observed == [True]
            assert policy.used_memory_bytes == baseline
        assert not filesystem_exists(scratch)
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "mutation", ["dataset", "target", "protocol", "fold", "family"]
)
def test_publication_rejects_valid_ledger_from_wrong_artifact_scope(tmp_path, mutation):
    from dataclasses import replace
    from uuid import uuid4

    from quant_research_platform.contracts.experiments import ExperimentFold
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.experiments import TrialRecorder

    bad_layout = StorageLayout(tmp_path / "bad")
    substituted = []

    def wrap(builder):
        def run(**callbacks):
            on_candidate, on_trial = callbacks["on_candidate"], callbacks["on_trial"]
            metadata = {}

            def candidate(fold, family, spec, result):
                metadata.clear()
                metadata.update(result.metadata)
                on_candidate(fold, family, spec, result)

            def trial(fold, family, reference):
                if substituted:
                    on_trial(fold, family, reference)
                    return
                with read_model_trial_ledger_reference(bad_layout, reference) as (
                    plan,
                    records,
                ):
                    record = records[0]
                changes = {"experiment_id": "experiment_" + uuid4().hex}
                if mutation == "dataset":
                    changes.update(
                        research_dataset_id="other_dataset",
                        data_snapshot_id="e" * 64,
                        input_lineage=("e" * 64,),
                    )
                elif mutation == "target":
                    changes["target_id"] = "other_model"
                elif mutation == "protocol":
                    changes["research_protocol_identity"] = "e" * 64
                elif mutation == "family":
                    changes["candidate_family_id"] = "other_family"
                else:
                    changes["folds"] = (
                        ExperimentFold(
                            fold_id="2018-q2",
                            train=fold.train,
                            validation=TimeWindow(
                                pd.Timestamp("2018-04-02"), pd.Timestamp("2018-04-03")
                            ),
                        ),
                    )
                wrong = replace(plan, **changes)
                payload = record.payload()["normalized_parameters"]
                for key in ("input_factor_columns", "input_feature_columns"):
                    payload[key] = tuple(payload[key])
                actual_spec = ModelSpec(**payload)
                recorder = TrialRecorder(bad_layout, "run_wrong_scope", wrong)
                attempt = recorder.begin(
                    actual_spec, common_coverage_id=record.common_coverage_id
                )
                recorder.finish(
                    attempt, metrics=dict(record.metrics), model_metadata=metadata
                )
                invalid_reference = recorder.reference()
                with read_model_trial_ledger_reference(
                    bad_layout, invalid_reference
                ) as (restored, completed):
                    assert restored == wrong
                    assert completed[0].status == "completed"
                substituted.append(invalid_reference)
                on_trial(fold, family, invalid_reference)

            callbacks.update(on_candidate=candidate, on_trial=trial)
            yield from builder(**callbacks)

        return run

    with runtime_resource_policy_scope() as policy:
        good = shared_model_case(tmp_path / "good")
        old_good = _file_bytes(good.root)
        expected = (
            "model_trial_fold_binding_invalid"
            if mutation == "fold"
            else "model_trial_artifact_binding_invalid"
        )
        with pytest.raises(ValueError, match=expected):
            shared_model_case(tmp_path / "bad", model_results_wrapper=wrap)
        assert len(substituted) == 1
        assert _file_bytes(good.root) == old_good
        assert not tuple(filesystem_glob(tmp_path / "bad/models", "**/model.json"))
        assert policy.used_memory_bytes == 0


def test_full_scope_family_metadata_fits_quarter_shards_and_protocol_member_limit(
    tmp_path,
):
    from dataclasses import replace

    from quant_research_platform.research.features.transforms import (
        model_input_validity_columns,
    )
    from quant_research_platform.research.models import supervised
    from quant_research_platform.storage.identity import filesystem_stat
    from tests.helpers.research_configuration import research_configuration

    configuration = research_configuration(
        name="fusion_nf_expectations", small_models=True
    )
    specs = tuple(
        ModelSpec(
            model_id=item["model_id"],
            model_algorithm=item["model_algorithm"],
            input_factor_columns=tuple(item["input_factor_columns"]),
            input_feature_columns=tuple(item["input_feature_columns"]),
            label_column=item["label_column"],
            hyperparameters=item["hyperparameters"],
        )
        for item in configuration["models"]
    )
    preflight_model_specs(specs)
    frame = shared_model_rows(specs, development_end="2019-03-31")
    for column in frame:
        if column.startswith("feature_missing_"):
            frame[column] = 0.0
    validity = tuple(
        dict.fromkeys(
            column
            for spec in specs
            for column in model_input_validity_columns(
                spec.input_columns, spec.model_algorithm
            )
            if column not in frame.columns
        )
    )
    for column in validity:
        frame[column] = True
    dataset = a_share.research_dataset_catalog().get("fusion_nf_expectations")
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(
            tmp_path,
            specs=specs,
            frame=frame,
            dataset=dataset,
            development_end="2019-03-31",
        )
        assert len(case.artifact.candidate_models) == 5 * 8
        assert len(case.artifact.final_models) == 2 * 8
        for member in case.artifact.member_records:
            if member["path"].endswith(".json"):
                assert filesystem_stat(case.root / member["path"]).st_size < 1024**2
        shard = json.loads(
            filesystem_read_bytes(case.root / "models/2019Q1/metadata.json")
        )
        assert len(shard["candidate_models"]) == 8
        assert any(
            item["model_metadata"]["scaler"] is not None
            for item in shard["candidate_models"]
        )
        for entry in case.artifact.final_models:
            metadata = entry["model_metadata"]
            algorithm = metadata["model_algorithm"]
            with read_quarterly_predictor(
                case.artifact,
                entry["selection_group"],
                entry["family_id"],
                resource_policy=policy,
            ) as actual:
                if algorithm in {
                    "equal_weight_rank_ensemble",
                    "ic_weighted_rank_ensemble",
                }:
                    changed = (
                        replace(
                            actual,
                            directions=tuple(-value for value in actual.directions),
                        )
                        if algorithm == "equal_weight_rank_ensemble"
                        else replace(
                            actual, weights=tuple(-value for value in actual.weights)
                        )
                    )
                    with pytest.raises(
                        ValueError, match="model_state_metadata_mismatch"
                    ):
                        fitting.validate_model_state(changed, metadata)
                    changed = None
                elif algorithm == "random_forest_regressor":
                    with pytest.MonkeyPatch.context() as patch:
                        patch.setattr(
                            supervised,
                            "model_cpu_capacity",
                            lambda: pytest.fail(
                                "persisted model validation rediscovered host capacity"
                            ),
                        )
                        fitting.validate_model_state(actual, metadata)
                    assert (
                        actual.n_jobs == metadata["resolved_hyperparameters"]["n_jobs"]
                    )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "field,value",
    (
        (
            "dependency_status",
            {"status": "available", "package": "sklearn", "version": "test"},
        ),
        ("rows_input", 1),
        ("filter_columns_used", []),
    ),
)
def test_quarterly_model_core_rejects_contract_drift(tmp_path, field, value):
    with runtime_resource_policy_scope() as policy:
        case = shared_model_case(tmp_path / "good")
        before = _file_bytes(case.root)

        def mutate(relative, payload):
            if relative == "models/2019Q1/metadata.json":
                payload["candidate_models"][0]["model_metadata"][field] = value

        root = _republish(case, tmp_path / "bad", mutate=mutate)
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            read_quarterly_model_artifact(root, trial_layout=case.layout)
        assert _file_bytes(case.root) == before
        assert policy.used_memory_bytes == 0


def test_model_metadata_failure_releases_copies_and_preserves_old_good(
    tmp_path, monkeypatch
):
    policy = capture_test_policy(
        available_memory_bytes=256 * 1024**2,
        explicit_cap_bytes=128 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        good = shared_model_case(tmp_path)
        before = _file_bytes(good.root)
        instances = set(good.root.parent.iterdir())
        primary = RuntimeError("model_metadata_boundary_primary")
        copied_metadata_ids = set()
        metadata_validations = 0
        original_validate = persisted.validate_model_metadata_core
        original_write = persisted.write_internal_artifact_manifest

        def admitted_validate(value, **kwargs):
            nonlocal metadata_validations
            metadata_validations += 1
            assert any(
                lease.owner == "research.model.artifact.metadata"
                and lease.reserved_bytes > 0
                for lease in policy._leases
            )
            return original_validate(value, **kwargs)

        def fail_metadata(path, payload, **kwargs):
            if path.name == "metadata.json":
                for entry in payload["candidate_models"]:
                    copied_metadata_ids.add(id(entry["model_metadata"]))
                assert kwargs["_admit"] is not None
                raise primary
            return original_write(path, payload, **kwargs)

        monkeypatch.setattr(
            persisted, "validate_model_metadata_core", admitted_validate
        )
        monkeypatch.setattr(
            persisted, "write_internal_artifact_manifest", fail_metadata
        )
        with pytest.raises(RuntimeError) as captured:
            shared_model_case(tmp_path, software_identity="e" * 64)
        assert captured.value is primary
        assert metadata_validations > 0 and copied_metadata_ids
        traceback = primary.__traceback__
        failed_frame_cleared = False
        while traceback is not None:
            values = traceback.tb_frame.f_locals
            if traceback.tb_frame.f_code.co_name == "fail_metadata":
                failed_frame_cleared = not values
            assert not any(
                id(value) in copied_metadata_ids for value in values.values()
            )
            if traceback.tb_frame.f_code.co_name == "write":
                assert not values.get("candidate_models")
                assert not values.get("final_models")
                assert not values.get("selections")
            traceback = traceback.tb_next
        assert failed_frame_cleared
        assert policy.to_payload()["used_memory_bytes"] == 0
        with policy.lease("test.after_model_metadata_failure", 64 * 1024**2):
            assert _file_bytes(good.root) == before
        assert set(good.root.parent.iterdir()) == instances
