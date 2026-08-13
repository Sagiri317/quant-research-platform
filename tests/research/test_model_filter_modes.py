from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
from quant_research_platform.contracts.evaluations import (
    EvaluationSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import (
    PREDICTION_SCORE_COLUMN,
    PredictionSpec,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    label_catalog,
)
from quant_research_platform.research.models.algorithm_facts import (
    model_algorithm_ids,
)
from quant_research_platform.research.models.prediction import predict_model_frame
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.runtime.resources import (
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from tests.helpers.research_specs import (
    make_signal_spec,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.filter_mode_fixtures import (
    make_filter_mode_frame,
    make_filter_mode_model_hyperparameters,
    make_filter_mode_prediction_output_scores,
)
from tests.research.prediction_output_helpers import (
    prediction_frame_from_artifact,
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

    eligible = model_fitting.fit_model_result_from_frames(
        (frame,),
        eligible_spec,
        estimator=build_estimator_from_model_spec(eligible_spec),
    )
    unfiltered = model_fitting.fit_model_result_from_frames(
        (frame,),
        unfiltered_spec,
        estimator=build_estimator_from_model_spec(unfiltered_spec),
    )

    assert (
        eligible.metadata
        | {
            key: eligible.metadata[key]
            for key in (
                "dependency_status",
                "feature_importance",
                "hyperparameters",
                "max_train_rows",
                "purge_gap_steps",
                "random_seed",
            )
        }
        == eligible.metadata
    )
    assert {
        key: eligible.metadata[key]
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
    assert eligible.metadata["dependency_status"]["status"] == "available"
    assert eligible.metadata["hyperparameters"] == {}
    assert eligible.metadata["random_seed"] is None
    assert eligible.metadata["max_train_rows"] == len(frame)
    assert eligible.metadata["purge_gap_steps"] == 0
    assert eligible.metadata["feature_importance"] == pytest.approx({"factor": 0.01})
    assert unfiltered.metadata["rows_after_filter_mode"] == 6
    assert unfiltered.metadata["rows_after_input_label_dropna"] == 5
    assert eligible.coefficients.tolist() == pytest.approx([0.0, 0.01])
    assert unfiltered.coefficients.tolist() == pytest.approx([0.0, 0.01])

    all_tradability_blocked = frame.assign(can_enter=False, can_exit=False)
    still_trains = model_fitting.fit_model_result_from_frames(
        (all_tradability_blocked,),
        eligible_spec,
        estimator=build_estimator_from_model_spec(eligible_spec),
    )
    assert still_trains.metadata["rows_after_input_label_dropna"] == 4


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

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, metadata

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_bad_streaming_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        model_fitting.fit_model_result_from_frames(
            (make_filter_mode_frame(),),
            spec,
            estimator=build_estimator_from_model_spec(spec),
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

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, metadata

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_bad_collected_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=len(make_filter_mode_frame()),
    )

    with pytest.raises(ValueError, match="invalid_model_metadata_count"):
        model_fitting.collect_training_frame((make_filter_mode_frame(),), spec)


def test_model_fit_accepts_valid_filter_metadata_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_research_platform.research.models.fitting as model_fitting

    def fake_apply_filter_mode(frame: pd.DataFrame, **_: object):
        return frame, _model_filter_metadata()

    monkeypatch.setattr(model_fitting, "apply_filter_mode", fake_apply_filter_mode)
    spec = ModelSpec(
        model_id="linear_valid_metadata_count",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
    )

    result = model_fitting.fit_model_result_from_frames(
        (make_filter_mode_frame(),),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )

    assert result.metadata["rows_input"] == 2
    assert result.metadata["rows_after_filter_mode"] == 2


@pytest.mark.parametrize(
    "model_algorithm",
    model_algorithm_ids(),
)
def test_required_model_algorithm_ids_fit_and_predict(model_algorithm: str) -> None:
    train = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": [f"asset_{idx}" for idx in range(8)],
            "timestamp": [pd.Timestamp("2024-01-02")] * 8,
            "factor": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "feature": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "label": [0.11, 0.19, 0.31, 0.41, 0.49, 0.61, 0.69, 0.81],
            "eligible": [True] * 8,
        }
    )
    test = train.assign(
        timestamp=pd.Timestamp("2024-01-03"),
        label=np.nan,
    )
    frame = pd.concat([train, test], ignore_index=True)
    spec = ModelSpec(
        model_id=f"{model_algorithm}_demo",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        random_seed=7,
        hyperparameters=make_filter_mode_model_hyperparameters(model_algorithm),
        max_train_rows=len(train),
    )
    prediction_spec = PredictionSpec(
        prediction_id=f"{model_algorithm}_pred",
        model_id=spec.model_id,
        research_dataset_id="demo_filter",
        train_end="2024-01-02",
        test_start="2024-01-03",
        test_end="2024-01-03",
    )

    result = model_fitting.fit_model_result_from_frames(
        (frame.loc[frame["timestamp"].lt("2024-01-03")],),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    prediction = prediction_frame_from_artifact(
        frame,
        prediction_spec,
        result.coefficients,
        spec.input_columns,
        prediction_model=result.prediction_model,
    )

    assert result.metadata["model_algorithm"] == model_algorithm
    assert result.metadata["input_columns"] == ["factor", "feature"]
    assert result.metadata["rows_after_input_label_dropna"] == 8
    assert len(prediction) == 8
    assert prediction[PREDICTION_SCORE_COLUMN].notna().all()


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

    max_rows = model_fitting.fit_model_result_from_frames(
        (frame,),
        max_rows_spec,
        estimator=build_estimator_from_model_spec(max_rows_spec),
    )
    purge = model_fitting.fit_model_result_from_frames(
        (frame,), purge_spec, estimator=build_estimator_from_model_spec(purge_spec)
    )

    assert max_rows.metadata["rows_after_input_label_dropna"] == 1
    assert max_rows.metadata["train_start"] == "2024-01-03"
    assert max_rows.metadata["train_end"] == "2024-01-03"
    assert purge.metadata["rows_after_input_label_dropna"] == 2
    assert purge.metadata["train_start"] == "2024-01-01"
    assert purge.metadata["train_end"] == "2024-01-02"


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

    training, _row_counts, _metadata = model_fitting.collect_training_frame(
        (frame,), spec
    )

    assert definition.lookahead == 2
    assert training["factor"].tolist() == [1.0]


def test_model_max_train_rows_streaming_matches_multi_asset_full_training() -> None:
    import quant_research_platform.research.models.fitting as model_fitting
    import quant_research_platform.research.models.validation_split as validation_split

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "asset_id": ["asset_a", "asset_b"] * 3 + ["asset_a", "asset_a"],
            "factor": [1.0, 101.0, 2.0, 102.0, 3.0, 103.0, 4.0, 5.0],
            "label": [2.0, 202.0, 4.0, 204.0, 6.0, 206.0, 8.0, 10.0],
            "eligible": True,
        }
    )
    spec = ModelSpec(
        model_id="linear_streaming_max_rows",
        model_algorithm="linear_regression",
        input_factor_columns=("factor",),
        label_column="label",
        max_train_rows=2,
        purge_gap_steps=2,
    )
    expected = validation_split.apply_training_controls(
        frame.loc[:, ["timestamp", "asset_id", "factor", "label"]],
        spec,
    ).drop(columns="asset_id")

    training, row_counts, _metadata = model_fitting.collect_training_frame(
        (frame.iloc[:6], frame.iloc[6:]),
        spec,
    )

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
        model_fitting.collect_training_frame(
            (frame,),
            spec,
        )

    with pytest.raises(ValueError, match="unbounded_model_training_collection"):
        model_fitting.collect_training_frame(
            (chunk for _, chunk in frame.groupby("timestamp")),
            spec,
        )


def test_model_matrix_admission_precedes_linear_and_estimator_allocation(
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

    class Estimator:
        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            entered.append("estimator")
            raise AssertionError("estimator fit entered before admission")

    monkeypatch.setattr(model_fitting, "linear_design_matrix", reject_linear_matrix)
    for operation in ("linear", "estimator"):
        policy = capture_test_policy(
            available_memory_bytes=16 * 1024 * 1024,
            explicit_cap_bytes=1 * 1024 * 1024,
        )
        with runtime_resource_policy_scope(policy):
            with policy.reserve("outer_owner", policy.task_memory_budget_bytes):
                with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                    if operation == "linear":
                        model_fitting.fit_linear_coefficients_exact(frame, spec)
                    else:
                        model_fitting.fit_predictor(
                            spec,
                            training=frame,
                            y=frame["label"].to_numpy(),
                            estimator=Estimator(),
                        )

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
        hyperparameters={"n_estimators": 100_000, "n_jobs": 1},
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
            model_fitting.fit_predictor(
                spec,
                training=frame,
                y=frame["label"].to_numpy(),
                estimator=Estimator(),
            )


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
        assert len(ephemeral_refs) == 2
        assert ephemeral_refs[0]() is None
        retained = ephemeral_refs[1]()
        assert retained is not None
        assert policy.reserved_memory_bytes >= original_payload(retained)
        del retained
        yield second
        assert len(ephemeral_refs) == 4
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
        if series.name == "label" and dtype is not None and np.dtype(dtype) == np.dtype(
            "float64"
        ):
            assert policy.reserved_memory_bytes >= len(series) * np.dtype(
                "float64"
            ).itemsize
        return original_to_numpy(series, *args, **kwargs)

    class Estimator:
        coefficient_count = 1
        coefficients = np.array([1.0])

        def fit(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
            return None

        @property
        def feature_importances_(self) -> np.ndarray:
            observations.append(policy.reserved_memory_bytes)
            assert policy.reserved_memory_bytes >= (
                expected_training_bytes + expected_model_bytes
            )
            return np.array([1.0])

    monkeypatch.setattr(pd.Series, "to_numpy", admitted_to_numpy)

    with runtime_resource_policy_scope(policy):
        result = model_fitting.fit_estimator_model_result_from_frames(
            (frame,),
            spec,
            estimator=Estimator(),
        )

    assert observations
    assert result.metadata["feature_importance"] == {"factor": 1.0}


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
            model_fitting.fit_predictor(
                spec,
                training=frame,
                y=frame["label"].to_numpy(),
                estimator=Estimator(),
            )


@pytest.mark.parametrize(
    "model_algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"],
)
def test_rank_ensembles_predict_cross_sectional_ranks_per_timestamp(
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
    spec = ModelSpec(
        model_id=f"{model_algorithm}_rank_scope",
        model_algorithm=model_algorithm,
        input_factor_columns=("factor",),
        input_feature_columns=("feature",),
        label_column="label",
        max_train_rows=len(frame),
    )

    result = model_fitting.fit_model_result_from_frames(
        (frame,), spec, estimator=build_estimator_from_model_spec(spec)
    )
    predicted = result.prediction_model.predict(frame)
    predicted_frame = predict_model_frame(frame, spec, result)

    assert predicted.tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert predicted_frame["prediction_score"].tolist() == pytest.approx(
        [0.5, 1.0, 0.5, 1.0]
    )


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
    result = model_fitting.fit_model_result_from_frames(
        (frame,), model_spec, estimator=build_estimator_from_model_spec(model_spec)
    )

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

    assert one_chunk["prediction_score"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert split_chunks["prediction_score"].tolist() == pytest.approx(
        one_chunk["prediction_score"].tolist()
    )


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

    result = model_fitting.fit_model_result_from_frames(
        (frame,), spec, estimator=build_estimator_from_model_spec(spec)
    )

    assert result.coefficients.tolist() == pytest.approx(expected.tolist())

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
