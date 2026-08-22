from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.contracts.models as model_contracts
import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import (
    ModelSpec,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    label_catalog,
)
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.runtime.resources import (
    MemoryLease,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
    make_filter_mode_prediction_output_scores,
)


def test_filter_columns_cannot_be_features() -> None:
    with pytest.raises(ValueError, match="filter_column_in_input_columns: eligible"):
        EvaluationSpec(
            run_id="bad_eval_features",
            market_id=MarketId("a_share"),
            research_dataset_id="demo_filter",
            signals=(make_signal_spec("eligible"),),
            label_columns=("label",),
        )
    with pytest.raises(ValueError, match="filter_column_in_input_columns: can_exit"):
        ModelSpec(
            model_id="bad_model_features",
            model_algorithm="linear_regression",
            input_factor_columns=("can_exit",),
            label_column="label",
        )


def test_model_sample_filter_modes_and_metadata_ignore_tradability_masks() -> None:
    frame = make_filter_mode_frame()
    eligible_spec = ModelSpec(
        model_id="linear_eligible",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    unfiltered_spec = ModelSpec(
        model_id="linear_unfiltered",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        max_train_rows=len(frame),
    )
    eligible_estimator = build_estimator_from_model_spec(eligible_spec)
    unfiltered_estimator = build_estimator_from_model_spec(unfiltered_spec)

    with runtime_resource_policy_scope():
        eligible_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            eligible_spec,
            estimator=eligible_estimator,
        )
        try:
            eligible = eligible_owned.value
            eligible_metadata = dict(eligible.metadata)
            eligible_coefficients = eligible.coefficients.copy()
        finally:
            eligible_owned.lease.close()
    with runtime_resource_policy_scope():
        unfiltered_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            unfiltered_spec,
            estimator=unfiltered_estimator,
        )
        try:
            unfiltered = unfiltered_owned.value
            unfiltered_metadata = dict(unfiltered.metadata)
            unfiltered_coefficients = unfiltered.coefficients.copy()
        finally:
            unfiltered_owned.lease.close()

    assert (
        eligible_metadata
        | {
            key: eligible_metadata[key]
            for key in (
                "dependency_status",
                "feature_importance",
                "hyperparameters",
                "max_train_rows",
                "purge_gap_steps",
                "random_seed",
            )
        }
        == eligible_metadata
    )
    assert {
        key: eligible_metadata[key]
        for key in (
            "model_algorithm",
            "sample_filter_mode",
            "filter_columns_used",
            "filter_definition",
            "rows_input",
            "rows_after_filter_mode",
            "rows_after_input_label_dropna",
            "train_start",
            "train_end",
            "input_factor_columns",
            "input_feature_columns",
            "input_columns",
            "input_count",
            "label_column",
            "coefficient_count",
        )
    } == {
        "model_algorithm": "linear_regression",
        "sample_filter_mode": "eligible",
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 6,
        "rows_after_filter_mode": 5,
        "rows_after_input_label_dropna": 4,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
        "input_factor_columns": ["factor"],
        "input_feature_columns": [],
        "input_columns": ["factor"],
        "input_count": 1,
        "label_column": "label",
        "coefficient_count": 2,
    }
    assert eligible_metadata["dependency_status"]["status"] == "available"
    assert eligible_metadata["hyperparameters"] == {}
    assert eligible_metadata["random_seed"] is None
    assert eligible_metadata["max_train_rows"] == len(frame)
    assert eligible_metadata["purge_gap_steps"] == 0
    assert eligible_metadata["feature_importance"] == pytest.approx({"factor": 0.01})
    assert unfiltered_metadata["rows_after_filter_mode"] == 6
    assert unfiltered_metadata["rows_after_input_label_dropna"] == 5
    assert eligible_coefficients.tolist() == pytest.approx([0.0, 0.01])
    assert unfiltered_coefficients.tolist() == pytest.approx([0.0, 0.01])

    all_tradability_blocked = frame.assign(can_enter=False, can_exit=False)
    still_trains_estimator = build_estimator_from_model_spec(eligible_spec)
    with runtime_resource_policy_scope():
        still_trains_owned = model_fitting._fit_model_result_owned_from_frames(
            (all_tradability_blocked,),
            eligible_spec,
            estimator=still_trains_estimator,
        )
        try:
            still_trains_rows = still_trains_owned.value.metadata[
                "rows_after_input_label_dropna"
            ]
        finally:
            still_trains_owned.lease.close()
    assert still_trains_rows == 4


@pytest.mark.parametrize(
    ("metadata_name", "field", "value"),
    [
        ("filter_metadata", "rows_input", True),
        ("filter_metadata", "rows_input", 1.5),
        ("filter_metadata", "rows_input", -1),
        ("filter_metadata", "rows_after_filter_mode", True),
        ("filter_metadata", "rows_after_filter_mode", 1.5),
        ("row_counts", "rows_after_input_label_dropna", True),
        ("row_counts", "rows_after_input_label_dropna", 1.5),
        ("row_counts", "rows_after_input_label_dropna", -1),
    ],
)
def test_model_metadata_rejects_invalid_count_values(
    metadata_name: str,
    field: str,
    value: object,
) -> None:
    import quant_research_platform.research.models.fitting as model_fitting

    spec = ModelSpec(
        model_id="linear_bad_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )
    filter_metadata = {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }
    row_counts = {
        "rows_after_input_label_dropna": 2,
        "train_start": "2024-01-02",
        "train_end": "2024-01-02",
    }
    target = filter_metadata if metadata_name == "filter_metadata" else row_counts
    target[field] = value

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        model_fitting.model_metadata(
            spec,
            filter_metadata=filter_metadata,
            row_counts=row_counts,
            coefficient_count=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_model_streaming_fit_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    import quant_research_platform.research.models.fitting as model_fitting

    metadata = _model_filter_metadata()
    metadata[field] = value

    def fake_filter_mode_scope(frame: pd.DataFrame, **_: object):
        return nullcontext((frame, metadata))

    monkeypatch.setattr(model_fitting, "_filter_mode_scope", fake_filter_mode_scope)
    spec = ModelSpec(
        model_id="linear_bad_streaming_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )
    estimator = build_estimator_from_model_spec(spec)

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        with runtime_resource_policy_scope():
            model_fitting._fit_model_result_owned_from_frames(
                (make_filter_mode_frame(),),
                spec,
                estimator=estimator,
            )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows_input", True),
        ("rows_input", 1.5),
        ("rows_input", -1),
        ("rows_after_filter_mode", True),
        ("rows_after_filter_mode", 1.5),
        ("rows_after_filter_mode", -1),
    ],
)
def test_model_training_collection_rejects_invalid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    import quant_research_platform.research.models.fitting as model_fitting

    metadata = _model_filter_metadata()
    metadata[field] = value

    def fake_filter_mode_scope(frame: pd.DataFrame, **_: object):
        return nullcontext((frame, metadata))

    monkeypatch.setattr(model_fitting, "_filter_mode_scope", fake_filter_mode_scope)
    spec = ModelSpec(
        model_id="linear_bad_collected_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(make_filter_mode_frame()),
    )

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        with runtime_resource_policy_scope():
            model_fitting._collect_training_frame_owned(
                (make_filter_mode_frame(),), spec
            )


@pytest.mark.parametrize("model_algorithm", ["linear_regression", "ridge_regression"])
def test_model_training_controls_apply_to_linear_and_estimators(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "factor": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "eligible": [True, True, True],
        }
    )
    max_rows_spec = ModelSpec(
        model_id=f"{model_algorithm}_max_rows",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=1,
    )
    purge_spec = ModelSpec(
        model_id=f"{model_algorithm}_purge",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        purge_gap_steps=1,
        max_train_rows=len(frame),
    )
    max_rows_estimator = build_estimator_from_model_spec(max_rows_spec)
    purge_estimator = build_estimator_from_model_spec(purge_spec)

    with runtime_resource_policy_scope():
        max_rows_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            max_rows_spec,
            estimator=max_rows_estimator,
        )
        try:
            max_rows_metadata = dict(max_rows_owned.value.metadata)
        finally:
            max_rows_owned.lease.close()
    with runtime_resource_policy_scope():
        purge_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            purge_spec,
            estimator=purge_estimator,
        )
        try:
            purge_metadata = dict(purge_owned.value.metadata)
        finally:
            purge_owned.lease.close()

    assert max_rows_metadata["rows_after_input_label_dropna"] == 1
    assert max_rows_metadata["train_start"] == "2024-01-03"
    assert max_rows_metadata["train_end"] == "2024-01-03"
    assert purge_metadata["rows_after_input_label_dropna"] == 2
    assert purge_metadata["train_start"] == "2024-01-01"
    assert purge_metadata["train_end"] == "2024-01-02"


def test_model_training_purge_uses_each_assets_valid_label_rows() -> None:
    dates = pd.bdate_range("2024-01-02", periods=5)
    panel = pd.DataFrame.from_records(
        [
            {
                "market_id": "a_share",
                "asset_id": asset_id,
                "timestamp": timestamp,
                "adj_open": float(100 + offset),
            }
            for asset_id, asset_dates in (
                ("dense", dates),
                ("sparse", dates[[0, 2, 4]]),
            )
            for offset, timestamp in enumerate(asset_dates)
        ]
    )
    definition = label_catalog().get("forward_return_1d")
    frame = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=(definition,),
    )
    frame["factor"] = np.where(frame["asset_id"].eq("dense"), 1.0, 99.0)
    frame["eligible"] = True
    spec = ModelSpec(
        model_id="linear_valid_bar_purge",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column=definition.output_column,
        purge_gap_steps=definition.lookahead,
        max_train_rows=len(frame),
    )

    with runtime_resource_policy_scope():
        training_owned = model_fitting._collect_training_frame_owned((frame,), spec)
        try:
            training = training_owned.value[0].copy()
        finally:
            training_owned.lease.close()

    assert definition.lookahead == 2
    assert training["factor"].tolist() == [1.0]


def test_model_max_train_rows_streaming_matches_multi_asset_full_training() -> None:
    import quant_research_platform.research.models.fitting as model_fitting
    import quant_research_platform.research.models.validation_split as validation_split

    dates = pd.bdate_range("2024-01-02", periods=5)
    chunk_keys = (
        (
            (0, "A"),
            (0, "B"),
            (0, "D"),
            (1, "A"),
            (1, "B"),
            (1, "D"),
            (2, "B"),
            (2, "D"),
        ),
        ((3, "A"), (3, "B")),
        ((4, "A"), (4, "B"), (4, "D")),
    )
    chunks = tuple(
        pd.DataFrame.from_records(
            [
                {
                    "timestamp": dates[offset],
                    "asset_id": asset_id,
                    "factor": float(ord(asset_id) * 10 + offset),
                    "label": float(ord(asset_id) * 20 + offset),
                    "eligible": True,
                }
                for offset, asset_id in keys
            ]
        )
        for keys in chunk_keys
    )
    frame = pd.concat(chunks, ignore_index=True)
    spec = ModelSpec(
        model_id="linear_streaming_max_rows",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=6,
        purge_gap_steps=2,
    )
    expected = validation_split.apply_training_controls(
        frame.loc[:, ["timestamp", "asset_id", "factor", "label"]],
        spec,
    ).drop(columns="asset_id")

    with runtime_resource_policy_scope():
        training_owned = model_fitting._collect_training_frame_owned(chunks, spec)
        try:
            training = training_owned.value[0].copy()
            row_counts = dict(training_owned.value[1])
        finally:
            training_owned.lease.close()

    pd.testing.assert_frame_equal(training, expected)
    assert row_counts["rows_after_input_label_dropna"] == len(expected)


def test_model_training_collection_rejects_unbounded_collection() -> None:
    import quant_research_platform.research.models.fitting as model_fitting

    frame = pd.concat(
        [
            make_filter_mode_frame().assign(timestamp=pd.Timestamp(value))
            for value in ("2024-01-02", "2024-01-03", "2024-01-04")
        ],
        ignore_index=True,
    )
    spec = ModelSpec(
        model_id="linear_uncapped_streaming",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=None,
    )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        with runtime_resource_policy_scope():
            model_fitting._collect_training_frame_owned(
                (frame,),
                spec,
            )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        with runtime_resource_policy_scope():
            model_fitting._collect_training_frame_owned(
                (chunk for _, chunk in frame.groupby("timestamp")),
                spec,
            )


def test_linear_matrix_admission_precedes_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research.models.fitting as model_fitting
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    frame = make_filter_mode_frame().loc[:, ["timestamp", "factor", "label"]]
    spec = ModelSpec(
        model_id="linear_matrix_admission",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(frame),
    )
    entered: list[str] = []

    def reject_linear_matrix(*_args, **_kwargs):  # noqa: ANN002, ANN003
        entered.append("linear")
        raise AssertionError("linear matrix allocated before admission")

    monkeypatch.setattr(model_fitting, "linear_design_matrix", reject_linear_matrix)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("outer_owner", policy.memory_budget_bytes):
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                model_fitting.fit_linear_coefficients_exact(frame, spec)

    assert entered == []


def test_model_native_state_admission_uses_resolved_tree_size_before_fit() -> None:
    import quant_research_platform.research.models.fitting as model_fitting
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    frame = make_filter_mode_frame().loc[:, ["timestamp", "factor", "label"]]
    spec = ModelSpec(
        model_id="forest_native_state_admission",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        hyperparameters={"n_estimators": np.int64(100_000), "n_jobs": 1},
        max_train_rows=len(frame),
    )

    class Estimator:
        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("estimator fit entered before native-state admission")

    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=Estimator(),
            )


def test_numpy_integral_tree_size_matches_native_parameter_and_state_bound() -> None:
    frame = make_filter_mode_frame().loc[:, ["timestamp", "factor", "label"]]
    fitted: list[tuple[object, dict[str, object], int]] = []
    for n_estimators in (100, np.int64(100)):
        spec = ModelSpec(
            model_id=f"forest_integral_{type(n_estimators).__name__}",
            model_algorithm="random_forest_regressor",
            input_factor_columns=("factor",),
            label_column="label",
            sample_filter_mode="unfiltered",
            hyperparameters={"n_estimators": n_estimators, "n_jobs": 1},
            max_train_rows=len(frame),
        )
        estimator = build_estimator_from_model_spec(spec)
        with runtime_resource_policy_scope():
            result_owned = model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=estimator,
            )
            try:
                resolved_hyperparameters = dict(
                    result_owned.value.metadata["resolved_hyperparameters"]
                )
            finally:
                result_owned.lease.close()
        fitted.append(
            (
                estimator,
                resolved_hyperparameters,
                model_fitting._model_state_upper_bound_bytes(  # noqa: SLF001
                    frame,
                    spec,
                    estimator=estimator,
                ),
            )
        )

    assert all(type(estimator.n_estimators) is int for estimator, _, _ in fitted)
    assert all(
        resolved["n_estimators"] == 100 and type(resolved["n_estimators"]) is int
        for _, resolved, _ in fitted
    )
    assert fitted[0][2] == fitted[1][2]


@pytest.mark.parametrize(
    ("model_algorithm", "maximum_bytes"),
    [
        ("gradient_boosting_regressor", 1024 * 1024),
        ("hist_gradient_boosting_regressor", 3 * 1024 * 1024),
    ],
)
def test_model_state_bound_uses_real_resolved_estimator_defaults(
    model_algorithm: str,
    maximum_bytes: int,
) -> None:
    rows = 100_000
    frame = pd.DataFrame({"factor": np.zeros(rows)})
    spec = ModelSpec(
        model_id=f"{model_algorithm}_resolved_defaults",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=rows,
    )
    estimator = build_estimator_from_model_spec(spec)

    assert model_fitting._model_state_upper_bound_bytes(  # noqa: SLF001
        frame,
        spec,
        estimator=estimator,
    ) <= maximum_bytes


def test_model_fit_envelope_does_not_recharge_preloaded_backend_image() -> None:
    rows = 32
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": np.linspace(1.0, 2.0, rows),
        }
    )
    spec = ModelSpec(
        model_id="forest_native_fit_admission",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        hyperparameters={"n_estimators": 1, "max_depth": 2, "n_jobs": 1},
        max_train_rows=rows,
    )
    estimator = build_estimator_from_model_spec(spec)
    feature_frame = model_fitting._selected_frame_payload_bytes(  # noqa: SLF001
        frame,
        list(spec.input_columns),
    )
    dense_features = rows * len(spec.input_columns) * np.dtype(np.float64).itemsize
    work_bytes = (
        feature_frame
        + 2 * dense_features
        + rows * np.dtype(np.float64).itemsize
        + rows * np.dtype(np.intp).itemsize
        + (len(spec.input_columns) + 1) ** 2 * np.dtype(np.float64).itemsize
        + model_fitting._parallel_estimator_workspace_bytes(  # noqa: SLF001
            rows=rows,
            inputs=len(spec.input_columns),
            spec=spec,
        )
    )
    admitted_bytes = (
        pandas_frame_payload_bytes(frame)
        + int(frame["label"].to_numpy().nbytes)
        + work_bytes
        + model_fitting._model_state_upper_bound_bytes(frame, spec)  # noqa: SLF001
    )
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=admitted_bytes + 64 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=estimator,
        )
        try:
            assert owned.value.prediction_model is estimator
        finally:
            owned.lease.close()


def test_model_threads_are_bound_to_physical_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_contracts.psutil, "cpu_count", lambda **_kwargs: 2)
    all_core = ModelSpec(
        model_id="forest_all_physical_cores",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        hyperparameters={"n_estimators": 1, "n_jobs": -1},
        max_train_rows=8,
    )
    too_many = ModelSpec(
        model_id="forest_too_many_threads",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        hyperparameters={"n_estimators": 1, "n_jobs": 3},
        max_train_rows=8,
    )

    assert build_estimator_from_model_spec(all_core).n_jobs == 2
    with pytest.raises(ValueError, match="model_threads_exceed_physical_cores"):
        build_estimator_from_model_spec(too_many)


def test_model_threads_reject_numpy_integral_request() -> None:
    spec = ModelSpec(
        model_id="forest_numpy_thread_request",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        hyperparameters={"n_estimators": 1, "n_jobs": np.int64(1)},
        max_train_rows=8,
    )

    with pytest.raises(ValueError, match="invalid_model_thread_count"):
        build_estimator_from_model_spec(spec)


def test_model_threads_adapt_to_live_planning_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_contracts.psutil, "cpu_count", lambda **_kwargs: 4)
    rows = 32
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": np.linspace(1.0, 2.0, rows),
        }
    )
    spec = ModelSpec(
        model_id="adaptive_forest",
        model_algorithm="random_forest_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        hyperparameters={"n_estimators": 1, "n_jobs": -1},
        max_train_rows=rows,
    )
    base = model_fitting._estimator_fit_working_set_bytes(  # noqa: SLF001
        frame,
        spec,
        threads=0,
    )
    one = model_fitting._estimator_fit_working_set_bytes(  # noqa: SLF001
        frame,
        spec,
        threads=1,
    )
    input_bytes = pandas_frame_payload_bytes(frame) + frame["label"].to_numpy().nbytes
    state_bytes = model_fitting._model_state_upper_bound_bytes(frame, spec)  # noqa: SLF001
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=(
            input_bytes + state_bytes + base + 2 * (one - base)
        ),
    )

    class Estimator:
        n_jobs = 4

        def set_params(self, **params):  # noqa: ANN003, ANN202
            self.n_jobs = params["n_jobs"]
            return self

        def fit(self, *_args):  # noqa: ANN002, ANN202
            assert self.n_jobs == 2

    estimator = Estimator()
    with runtime_resource_policy_scope(policy):
        owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=estimator,
        )
        try:
            assert owned.value.prediction_model is estimator
        finally:
            owned.lease.close()


def test_linear_tsqr_reserves_first_frame_across_second_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 512
    first = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": np.linspace(1.0, 2.0, rows),
            "eligible": True,
        }
    )
    second = first.assign(timestamp=first["timestamp"] + pd.Timedelta(days=1))
    third = first.assign(timestamp=first["timestamp"] + pd.Timedelta(days=2))
    spec = ModelSpec(
        model_id="linear_tsqr_retained_frame",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="eligible",
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)
    input_ids = {id(first), id(second), id(third)}
    ephemeral_refs: list[weakref.ReferenceType[pd.DataFrame]] = []
    original_payload = model_fitting.pandas_frame_payload_bytes

    def tracked_payload(frame: pd.DataFrame) -> int:
        if id(frame) not in input_ids:
            ephemeral_refs.append(weakref.ref(frame))
        return original_payload(frame)

    monkeypatch.setattr(model_fitting, "pandas_frame_payload_bytes", tracked_payload)

    def frames():
        yield first
        assert ephemeral_refs
        assert all(reference() is None for reference in ephemeral_refs[:-1])
        retained = ephemeral_refs[-1]()
        assert retained is not None
        assert policy.used_memory_bytes >= original_payload(retained)
        del retained
        yield second
        assert all(reference() is None for reference in ephemeral_refs)
        yield third

    with runtime_resource_policy_scope(policy):
        coefficients, row_counts, _metadata = model_fitting.fit_linear_coefficients(
            frames(),
            spec=spec,
            fit_columns=("timestamp", "factor", "label"),
        )

    assert len(coefficients) == 2
    assert row_counts["rows_after_input_label_dropna"] == 3 * rows


def test_linear_qr_failure_clears_work_arrays_before_lease_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 32
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": np.linspace(1.0, 2.0, rows),
        }
    )
    spec = ModelSpec(
        model_id="linear_qr_failure_aliases",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
    )
    qr_refs: list[weakref.ReferenceType[object]] = []

    class FailingTranspose:
        def __matmul__(self, _other):  # noqa: ANN001
            raise RuntimeError("qr_after_allocation")

    class QFactor:
        @property
        def T(self) -> FailingTranspose:
            return FailingTranspose()

    def failing_qr(matrix, *, mode):  # noqa: ANN001
        del mode
        del matrix
        q_factor = QFactor()
        r_factor = np.ones((2, 2), dtype="float64")
        qr_refs.extend((weakref.ref(q_factor), weakref.ref(r_factor)))
        return q_factor, r_factor

    monkeypatch.setattr(model_fitting.np.linalg, "qr", failing_qr)
    original_close = MemoryLease.close
    close_checks: list[bool] = []

    def observing_close(self: MemoryLease) -> None:
        if self.owner == "research.model.qr_work":
            gc.collect()
            close_checks.append(all(reference() is None for reference in qr_refs))
            assert close_checks[-1]
        original_close(self)

    monkeypatch.setattr(MemoryLease, "close", observing_close)
    policy = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)

    with pytest.raises(RuntimeError, match="qr_after_allocation"):
        with runtime_resource_policy_scope(policy):
            model_fitting.fit_linear_coefficients(
                (frame,),
                spec=spec,
                fit_columns=("timestamp", "factor", "label"),
            )

    assert close_checks == [True]
    assert policy.used_memory_bytes == 0


def test_estimator_target_and_model_state_stay_reserved_through_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 64
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": pd.Series(range(rows), dtype="Int64"),
        }
    )
    spec = ModelSpec(
        model_id="tree_retained_state",
        model_algorithm="decision_tree_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        max_train_rows=rows,
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)
    expected_training_bytes = pandas_frame_payload_bytes(frame)
    expected_model_bytes = model_fitting._model_state_upper_bound_bytes(frame, spec)
    original_to_numpy = pd.Series.to_numpy
    observations: list[int] = []

    def admitted_to_numpy(series, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        dtype = kwargs.get("dtype", args[0] if args else None)
        if (
            series.name == "label"
            and dtype is not None
            and np.dtype(dtype) == np.dtype("float64")
        ):
            assert (
                policy.used_memory_bytes
                >= len(series) * np.dtype("float64").itemsize
            )
        return original_to_numpy(series, *args, **kwargs)

    class Estimator:
        coefficient_count = 1
        coefficients = np.array([1.0])

        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        @property
        def feature_importances_(self) -> np.ndarray:
            observations.append(policy.used_memory_bytes)
            assert policy.used_memory_bytes >= (
                expected_training_bytes + expected_model_bytes
            )
            return np.array([1.0])

    monkeypatch.setattr(pd.Series, "to_numpy", admitted_to_numpy)

    with runtime_resource_policy_scope(policy):
        owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,),
            spec,
            estimator=Estimator(),
        )
        try:
            assert owned.value.metadata["feature_importance"] == {"factor": 1.0}
        finally:
            owned.lease.close()

    assert observations


@pytest.mark.parametrize(
    ("model_algorithm", "hyperparameters"),
    (
        ("xgboost_regressor", {"n_estimators": 100, "max_depth": 0, "n_jobs": 1}),
        (
            "hist_gradient_boosting_regressor",
            {"max_iter": 100, "max_leaf_nodes": None},
        ),
        (
            "gradient_boosting_regressor",
            {"n_estimators": 100, "max_depth": None},
        ),
    ),
)
def test_unbounded_tree_sentinels_reject_before_estimator_fit(
    model_algorithm: str,
    hyperparameters: dict[str, object],
) -> None:
    rows = 1_000
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "factor": np.linspace(0.0, 1.0, rows),
            "label": np.linspace(1.0, 2.0, rows),
        }
    )
    spec = ModelSpec(
        model_id=f"{model_algorithm}_unbounded_state",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        hyperparameters=hyperparameters,
        max_train_rows=rows,
    )

    class Estimator:
        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError(
                "estimator fit entered before sentinel-aware admission"
            )

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024 * 1024,
        explicit_cap_bytes=8 * 1024 * 1024,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=Estimator(),
            )


def test_gradient_tree_zero_depth_rejects_before_estimator_fit() -> None:
    frame = make_filter_mode_frame().loc[:, ["timestamp", "factor", "label"]]
    spec = ModelSpec(
        model_id="gradient_zero_depth",
        model_algorithm="gradient_boosting_regressor",
        input_factor_columns=("factor",),
        label_column="label",
        sample_filter_mode="unfiltered",
        hyperparameters={"n_estimators": 1, "max_depth": 0},
        max_train_rows=len(frame),
    )

    class Estimator:
        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            raise AssertionError("estimator fit entered before state validation")

    with pytest.raises(ValueError, match="invalid_model_state_sizing_parameter"):
        with runtime_resource_policy_scope():
            model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=Estimator(),
            )
    with runtime_resource_policy_scope():
        pass


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensemble_prediction_outputs_are_chunk_boundary_invariant(
    model_algorithm: str,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset_id": ["asset_a", "asset_b", "asset_a", "asset_b"],
            "factor": [1.0, 2.0, 1.0, 2.0],
            "feature": [10.0, 20.0, 10.0, 20.0],
            "label": [0.0, 1.0, 0.0, 1.0],
            "eligible": [True, True, True, True],
        }
    )
    model_spec = ModelSpec(
        model_id=f"{model_algorithm}_chunk_boundary",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_chunk_boundary_prediction",
        model_id=model_spec.model_id,
        research_dataset_id="rank_boundary_dataset",
        train_end="2024-01-01",
        test_start="2024-01-02",
    )
    estimator = build_estimator_from_model_spec(model_spec)
    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,), model_spec, estimator=estimator
        )
        try:
            result = result_owned.value
            one_chunk = make_filter_mode_prediction_output_scores(
                [frame],
                prediction_spec=prediction_spec,
                model_spec=model_spec,
                result=result,
            )
            split_chunks = make_filter_mode_prediction_output_scores(
                [chunk for _, chunk in frame.groupby("timestamp", sort=True)],
                prediction_spec=prediction_spec,
                model_spec=model_spec,
                result=result,
            )

            assert one_chunk["prediction_score"].tolist() == pytest.approx(
                [0.5, 1.0, 0.5, 1.0]
            )
            assert split_chunks["prediction_score"].tolist() == pytest.approx(
                one_chunk["prediction_score"].tolist()
            )
        finally:
            result_owned.lease.close()


def test_model_rank_deficient_fit_matches_direct_lstsq_design() -> None:
    x1 = [
        0.1257302210933933,
        -0.1321048632913019,
        0.6404226504432821,
        0.10490011715303971,
        -0.535669373161111,
        0.36159505490948474,
        1.3040000451301372,
        0.9470809631292422,
    ]
    x2 = [
        0.12573021405604093,
        -0.1321048759455166,
        0.6404226442105374,
        0.1049001175662995,
        -0.5356693964114188,
        0.3615950527215681,
        1.3040000326710277,
        0.9470809558065686,
    ]
    y = [
        -0.5442589828573099,
        -0.31630015636915454,
        0.4116305363741328,
        1.0425133694426776,
        -0.12853466294403426,
        1.3664634705496859,
        -0.6651946734866135,
        0.3515100700930197,
    ]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-05",
                ]
            ),
            "asset_id": [f"asset_{idx}" for idx in range(len(x1))],
            "factor": x1,
            "factor_shadow": x2,
            "label": y,
            "eligible": [True] * len(x1),
        }
    )
    spec = ModelSpec(
        model_id="linear_rank_deficient",
        model_algorithm="linear_regression",
        input_factor_columns=("factor", "factor_shadow"),
        label_column="label",
        max_train_rows=len(frame),
    )
    usable = frame.loc[frame["eligible"]].dropna(
        subset=[*spec.input_columns, spec.label_column]
    )
    x = usable.loc[:, spec.input_columns].to_numpy(dtype="float64")
    y = usable.loc[:, spec.label_column].to_numpy(dtype="float64")
    design = np.empty((len(x), len(spec.input_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    expected, *_ = np.linalg.lstsq(design, y, rcond=None)
    estimator = build_estimator_from_model_spec(spec)

    with runtime_resource_policy_scope():
        result_owned = model_fitting._fit_model_result_owned_from_frames(
            (frame,), spec, estimator=estimator
        )
        try:
            result_coefficients = result_owned.value.coefficients.tolist()
        finally:
            result_owned.lease.close()

    assert result_coefficients == pytest.approx(expected.tolist())


def test_model_rejects_tradable_sample_filter_mode() -> None:
    with pytest.raises(ValueError, match="unsupported_model_sample_filter_mode"):
        ModelSpec(
            model_id="linear_tradable",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            sample_filter_mode="tradable",
        )


def test_model_rejects_unsupported_contract_version() -> None:
    with pytest.raises(ValueError, match="unsupported_model_contract_version"):
        ModelSpec(
            model_id="linear_bad_contract",
            model_algorithm="linear_regression",
            input_factor_columns=("factor",),
            label_column="label",
            contract_version="9.9",
        )


def _model_filter_metadata() -> dict[str, object]:
    return {
        "filter_columns_used": ["eligible"],
        "filter_definition": "eligible == True",
        "rows_input": 2,
        "rows_after_filter_mode": 2,
    }
