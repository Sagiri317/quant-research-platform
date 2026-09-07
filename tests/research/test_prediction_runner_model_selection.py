from __future__ import annotations

import warnings
import weakref
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.fitting as model_fitting
import quant_research_platform.research.models.trial_selection as model_selection_owner
import quant_research_platform.research.predictions.runner as prediction_runner
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.models.supervised import (
    build_estimator_from_model_spec,
)
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_prediction_inputs import (
    make_prediction_frame,
)
from tests.helpers.research_specs import (
    make_demo_research_dataset_spec,
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.workflow_test_support import _column_predictor


def test_prediction_runner_selects_model_using_validation_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        )
    )
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="prediction_demo",
                model_id="base_model",
                research_dataset_id="research_demo",
                train_end="2023-01-03",
                test_start="2023-01-07",
                test_end="2023-01-08",
                candidate_model_ids=("candidate_model",),
                model_selection_metric="validation_mse",
                model_selection_validation_start="2023-01-04",
                model_selection_validation_end="2023-01-06",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        _development_frame(),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        ),
    )

    original_to_dict = pd.DataFrame.to_dict

    def reject_validation_row_graph(frame, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        orient = args[0] if args else kwargs.get("orient", "dict")
        if orient == "records" and {
            "timestamp",
            "asset_id",
            "label",
            "prediction_score",
        }.issubset(frame.columns):
            raise AssertionError(
                "validation predictions converted to a Python row graph"
            )
        return original_to_dict(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", reject_validation_row_graph)

    with runtime_resource_policy_scope():
        selected, model_selection = model_selection_owner.select_prediction_model_spec(
            prediction_spec=prediction_catalog.get("prediction_demo"),
            dataset_spec=dataset_catalog.get("research_demo"),
            model_catalog=model_catalog,
            chunks=lambda columns, *, input_columns, start, end: (
                iter_research_dataset_chunks(
                    layout,
                    market_id=MarketId("a_share"),
                    dataset_id="research_demo",
                    columns=columns,
                )
            ),
        )

    assert model_selection["selected_model_id"] == "candidate_model"
    assert model_selection["selection_metric"] == "validation_mse"
    assert model_selection["selection_direction"] == "minimize"
    assert model_selection["candidate_model_ids"] == ["base_model", "candidate_model"]
    assert selected.input_columns == ("factor_b",)
    assert model_selection["selected_model_input_columns"] == ["factor_b"]


def test_model_selection_compares_common_validation_primary_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _development_frame()
    frame.loc[frame["timestamp"].eq("2023-01-04"), "factor_a"] = np.nan
    frame.loc[frame["timestamp"].eq("2023-01-05"), "factor_b"] = np.nan
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="base_model",
        research_dataset_id="research_demo",
        train_end="2023-01-03",
        test_start="2023-01-07",
        test_end="2023-01-08",
        candidate_model_ids=("candidate_model",),
        model_selection_metric="validation_mse",
        model_selection_validation_start="2023-01-04",
        model_selection_validation_end="2023-01-06",
    )
    validation_keys: list[tuple[tuple[pd.Timestamp, str], ...]] = []
    original_metrics = model_selection_owner.evaluate_model_trial_metrics_from_chunks

    def chunks(columns, *, input_columns, start, end):  # noqa: ANN001
        del input_columns
        return (frame.loc[:, list(columns)],)

    def record_validation_keys(chunks, **kwargs):  # noqa: ANN001, ANN003
        keys: list[tuple[pd.Timestamp, str]] = []

        def observed_chunks():
            for item in chunks:
                frame_value = (
                    item.value
                    if isinstance(item, prediction_runner.OwnedValue)  # noqa: SLF001
                    else item
                )
                keys.extend(
                    frame_value[["timestamp", "asset_id"]].itertuples(
                        index=False,
                        name=None,
                    )
                )
                yield item

        result = original_metrics(observed_chunks(), **kwargs)
        validation_keys.append(tuple(keys))
        return result

    monkeypatch.setattr(
        model_selection_owner,
        "evaluate_model_trial_metrics_from_chunks",
        record_validation_keys,
    )

    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with prediction_runner.runtime_resource_policy_scope(policy):
        model_selection_owner.select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
            model_catalog=model_catalog,
            chunks=chunks,
        )

    assert len(validation_keys) == 2
    assert validation_keys[0] == validation_keys[1]


def test_empty_validation_chunk_releases_its_lease() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "asset_id": ["asset"],
            "factor_a": [1.0],
            "label": [1.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024 * 1024,
    )
    candidate_spec = ModelSpec(
        model_id="empty_validation",
        model_algorithm="linear_regression",
        input_factor_columns=("factor_a",),
        label_column="label",
    )
    prediction_spec = SimpleNamespace(
        sample_filter_mode="unfiltered",
        prediction_column="prediction",
    )
    model_result = model_fitting.ModelFitResult(
        metadata={},
        prediction_model=_column_predictor("factor_a"),
    )

    with prediction_runner.runtime_resource_policy_scope(policy):
        result = list(
            model_selection_owner.iter_validation_prediction_chunks(
                lambda: iter((frame,)),
                prediction_spec=prediction_spec,
                candidate_spec=candidate_spec,
                model_result=model_result,
                validation_start=pd.Timestamp("2023-01-02"),
                validation_end=pd.Timestamp("2023-01-03"),
            )
        )

    assert result == []
    assert policy.used_memory_bytes == 0


def test_model_selection_drops_candidate_model_alias_before_lease_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _development_frame()
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="base_model",
        research_dataset_id="research_demo",
        train_end="2023-01-03",
        test_start="2023-01-07",
        test_end="2023-01-08",
        candidate_model_ids=("candidate_model",),
        model_selection_metric="validation_mse",
        model_selection_validation_start="2023-01-04",
        model_selection_validation_end="2023-01-06",
    )
    created_refs: list[weakref.ReferenceType[object]] = []

    class Predictor:
        def predict(self, values: pd.DataFrame) -> np.ndarray:
            return np.ones(len(values), dtype="float64")

    def fit_owned(*_args, **_kwargs):  # noqa: ANN002, ANN003
        predictor = Predictor()
        created_refs.append(weakref.ref(predictor))
        lease = current_runtime_resource_policy().lease(
            "research.test.candidate_model",
            1024,
        )
        return prediction_runner.OwnedValue(  # noqa: SLF001
            model_fitting.ModelFitResult(
                metadata={
                    "resolved_hyperparameters": {},
                    "random_seed": None,
                    "rows_after_input_label_dropna": 1,
                    "dependency_status": {
                        "status": "available",
                        "package": "internal",
                        "version": "test",
                    },
                },
                prediction_model=predictor,
            ),
            lease,
        )

    monkeypatch.setattr(
        model_fitting,
        "_fit_model_result_owned_from_frames",
        fit_owned,
    )

    def chunks(columns, *, input_columns, start, end):  # noqa: ANN001
        del input_columns
        return (frame.loc[:, list(columns)],)

    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        model_selection_owner.select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
            model_catalog=model_catalog,
            chunks=chunks,
        )

    assert created_refs
    assert all(reference() is None for reference in created_refs)
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -np.inf])
def test_model_selection_rejects_candidate_specific_non_finite_predictions(
    invalid_prediction: float,
) -> None:
    validation = pd.DataFrame(
        {
            "label": [1.0, 2.0, 3.0],
            "prediction": [1.0, invalid_prediction, 3.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            (
                validation.assign(
                    timestamp=pd.date_range(
                        "2023-01-01", periods=len(validation), freq="D"
                    ),
                    asset_id="asset",
                ),
            ),
            model_id="candidate_model",
            label_column="label",
            prediction_column="prediction",
        )


def test_prediction_runner_applies_sample_filter_to_validation_metrics(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        )
    )
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="prediction_demo",
                model_id="base_model",
                research_dataset_id="research_demo",
                train_end="2023-01-03",
                test_start="2023-01-07",
                test_end="2023-01-08",
                candidate_model_ids=("candidate_model",),
                model_selection_metric="validation_mse",
                model_selection_validation_start="2023-01-04",
                model_selection_validation_end="2023-01-06",
                sample_filter_mode="eligible",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        _development_frame(filter_flip=True),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        ),
    )

    with runtime_resource_policy_scope():
        selected, model_selection = model_selection_owner.select_prediction_model_spec(
            prediction_spec=prediction_catalog.get("prediction_demo"),
            dataset_spec=dataset_catalog.get("research_demo"),
            model_catalog=model_catalog,
            chunks=lambda columns, *, input_columns, start, end: (
                iter_research_dataset_chunks(
                    layout,
                    market_id=MarketId("a_share"),
                    dataset_id="research_demo",
                    columns=columns,
                )
            ),
        )
    assert selected.input_columns == ("factor_b",)

    assert model_selection["selected_model_id"] == "candidate_model"
    assert model_selection["selected_model_input_columns"] == ["factor_b"]


def test_prediction_runner_fails_closed_on_candidate_label_mismatch(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        )
    )
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",), label_column="label"),
        _model("candidate_model", ("factor_b",), label_column="other_label"),
    )
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
                prediction_id="prediction_demo",
                model_id="base_model",
                research_dataset_id="research_demo",
                train_end="2023-01-03",
                test_start="2023-01-07",
                candidate_model_ids=("candidate_model",),
                model_selection_validation_start="2023-01-04",
                model_selection_validation_end="2023-01-06",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        _development_frame().assign(other_label=1.0),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label", "other_label"),
            filter_columns=("eligible",),
            context_columns=("label_available_at",),
        ),
    )

    with (
        runtime_resource_policy_scope(),
        pytest.raises(ValueError, match="model_selection_label_column_mismatch"),
    ):
        model_selection_owner.select_prediction_model_spec(
            prediction_spec=prediction_catalog.get("prediction_demo"),
            dataset_spec=dataset_catalog.get("research_demo"),
            model_catalog=model_catalog,
            chunks=lambda columns, *, input_columns, start, end: (
                iter_research_dataset_chunks(
                    layout,
                    market_id=MarketId("a_share"),
                    dataset_id="research_demo",
                    columns=columns,
                )
            ),
        )


def test_prediction_runner_accepts_equivalent_timezone_representations() -> None:
    chunk = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2023-01-15T12:00:00",
                    tz=timezone(timedelta(hours=-5)),
                ),
                pd.Timestamp(
                    "2023-01-15T12:01:00",
                    tz=ZoneInfo("America/New_York"),
                ),
            ]
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timestamps = strict_timestamp_series(
            chunk["timestamp"],
            error="invalid_prediction_timestamp_values",
        )

    assert timestamps.tolist() == [
        pd.Timestamp("2023-01-15T17:00:00"),
        pd.Timestamp("2023-01-15T17:01:00"),
    ]


def test_prediction_result_context_preserves_body_primary() -> None:
    primary = ValueError("body_primary")
    closed: list[bool] = []
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    scope = prediction_runner.runtime_resource_policy_scope(policy)
    scope.__enter__()

    def close() -> None:
        closed.append(True)
        scope.__exit__(None, None, None)
        raise MemoryError("cleanup_secondary")

    result = prediction_runner.PredictionResult(
        spec=SimpleNamespace(),
        output_chunks=(),
        model_metadata={},
        prediction_metadata_factory=dict,
        _close=close,
    )
    with pytest.raises(ValueError) as raised:
        with result:
            raise primary

    assert raised.value is primary
    assert raised.value.__notes__ == [
        "prediction result cleanup also failed: MemoryError"
    ]
    assert closed == [True]
    assert current_runtime_resource_policy() is None


def _model_catalog(*models: ModelSpec) -> ModelCatalog:
    return ModelCatalog(models)


@pytest.mark.parametrize(
    ("future_label", "missing_input"), [(False, False), (True, False), (True, True)]
)
def test_real_rank_validation_predictions_and_winner_preserve_split_sessions(
    tmp_path: Path, future_label: bool, missing_input: bool
) -> None:
    dates = pd.date_range("2023-01-01", periods=5)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "timestamp": dates.repeat(4),
            "asset_id": ["a", "b", "c", "d"] * 5,
            "factor_a": [1.0, 2.0, 3.0, 4.0] * 5,
            "label": [1.0, 2.0, 3.0, 4.0] * 5,
            "label_available_at": dates.repeat(4),
            "eligible": True,
        }
    )
    validation_rows = frame.timestamp.between(dates[2], dates[3])
    if future_label:
        future_rows = validation_rows & frame.asset_id.eq("d")
        frame.loc[future_rows, "label_available_at"] = dates[4]
        frame.loc[future_rows, "label"] = 1e9
    if missing_input:
        frame.loc[validation_rows & frame.asset_id.eq("b"), "factor_a"] = np.nan
    layout = StorageLayout(tmp_path)
    dataset = make_demo_research_dataset_spec(
        data_id="research",
        input_data_ids=("source",),
        base_input_data_id="source",
        factor_columns=("factor_a",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("label_available_at",),
    )
    root = layout.research_dataset_root(MarketId("a_share"), dataset.data_id)
    write_dataset(root, frame, spec=dataset)
    frame = read_dataset(root)
    specs = tuple(
        ModelSpec(
            model_id=algorithm,
            model_algorithm=algorithm,
            input_factor_columns=("factor_a",),
            label_column="label",
        )
        for algorithm in ("equal_weight_rank_ensemble", "ic_weighted_rank_ensemble")
    )
    prediction = PredictionSpec(
        prediction_id="rank",
        model_id=specs[0].model_id,
        candidate_model_ids=(specs[1].model_id,),
        research_dataset_id="research",
        train_end="2023-01-02",
        test_start="2023-01-05",
        model_selection_validation_start="2023-01-03",
        model_selection_validation_end="2023-01-04",
        model_selection_metric="validation_mse",
    )
    policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with runtime_resource_policy_scope(policy):
        for spec in specs:
            fitted = model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                spec,
                estimator=build_estimator_from_model_spec(spec),
                fitting_cutoff=prediction.train_end,
            )
            try:
                outputs = []
                for step in (len(frame), 2):
                    pieces = []
                    for item in model_selection_owner.iter_validation_prediction_chunks(
                        lambda step=step: (
                            frame.iloc[i : i + step] for i in range(0, len(frame), step)
                        ),
                        prediction_spec=prediction,
                        candidate_spec=spec,
                        model_result=fitted.value,
                        validation_start=pd.Timestamp("2023-01-03"),
                        validation_end=pd.Timestamp("2023-01-04"),
                    ):
                        pieces.append(item.value.copy())
                        item.release(owner="test.rank.validation")
                    outputs.append(pd.concat(pieces, ignore_index=True))
                pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)
                expected = (
                    [-1 / 3, 0.0, 0.0]
                    if missing_input
                    else [-0.375, -0.125, 0.125]
                    if future_label
                    else [-0.375, -0.125, 0.125, 0.375]
                )
                assert outputs[0][
                    prediction.prediction_column
                ].tolist() == pytest.approx(expected * 2)
            finally:
                fitted.release(owner="test.rank.fit")
        selections = []
        for step in (len(frame), 2):
            selected, metadata = model_selection_owner.select_prediction_model_spec(
                prediction_spec=prediction,
                dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
                model_catalog=ModelCatalog(specs),
                chunks=lambda columns, *, input_columns, start, end, step=step: (
                    frame.iloc[i : i + step].loc[:, list(columns)]
                    for i in range(0, len(frame), step)
                ),
            )
            selections.append((selected.model_id, metadata))
        assert selections[0] == selections[1]
        assert selections[0][0] == specs[0].model_id
    assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("label_column", ["forward_return_20d", "custom_target"])
def test_dataset_training_preserves_label_maturity(
    tmp_path: Path, label_column: str
) -> None:
    availability = (
        "forward_return_20d_available_at"
        if label_column == "forward_return_20d"
        else "label_available_at"
    )
    dataset = make_demo_research_dataset_spec(
        data_id="research",
        input_data_ids=("source",),
        base_input_data_id="source",
        factor_columns=("factor_a",),
        label_columns=(label_column,),
        filter_columns=("eligible",),
        context_columns=(availability,),
    )
    model = _model("model", ("factor_a",), label_column=label_column)
    prediction = PredictionSpec(
        prediction_id="prediction",
        model_id=model.model_id,
        research_dataset_id=dataset.data_id,
        train_end="2023-01-03",
        test_start="2023-01-04",
        test_end="2023-01-04",
    )
    outputs = []
    for future_label in (0.3, 30.0):
        layout = StorageLayout(tmp_path / str(future_label))
        frame = pd.DataFrame(
            {
                "market_id": "a_share",
                "asset_id": "asset",
                "timestamp": pd.to_datetime(
                    ["2022-12-01", "2022-12-02", "2023-01-02", "2023-01-04"]
                ),
                "factor_a": [1.0, 2.0, 3.0, 4.0],
                label_column: [0.1, 0.2, future_label, 0.4],
                "eligible": True,
                availability: pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-02-02", "2023-02-04"]
                ),
            }
        )
        write_dataset(
            layout.research_dataset_root(MarketId("a_share"), dataset.data_id),
            frame,
            spec=dataset,
        )
        with runtime_resource_policy_scope():
            owned = model_fitting._fit_model_result_owned_from_frames(
                iter_research_dataset_chunks(
                    layout,
                    market_id=MarketId("a_share"),
                    dataset_id=dataset.data_id,
                ),
                model,
                estimator=build_estimator_from_model_spec(model),
                fitting_cutoff=prediction.train_end,
            )
            try:
                assert owned.value.metadata["rows_after_input_label_dropna"] == 2
                predict = model_fitting.prediction_callable(
                    model.input_columns, owned.value.prediction_model
                )
                outputs.append(
                    pd.DataFrame({prediction.prediction_column: predict(frame.tail(1))})
                )
            finally:
                owned.release(owner="test.dataset.fit")
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)
    assert outputs[0][prediction.prediction_column].tolist() == pytest.approx([0.4])


@pytest.mark.parametrize(
    "label_column",
    ["label", "forward_return_20d", "model_target", "future_mid_return_1s"],
)
def test_training_requires_its_explicit_availability(label_column: str) -> None:
    frame = _development_frame().rename(columns={"label": label_column})
    if label_column == "label":
        frame = frame.drop(columns="label_available_at")
    model = _model("model", ("factor_a",), label_column=label_column)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError, match="label_availability_column_required"):
            model_fitting._fit_model_result_owned_from_frames(
                (frame,),
                model,
                estimator=build_estimator_from_model_spec(model),
                fitting_cutoff="2023-01-03",
            )
        assert policy.used_memory_bytes == 0


def _model(
    model_id: str,
    input_columns: tuple[str, ...],
    *,
    label_column: str = "label",
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        model_algorithm="linear_regression",
        input_factor_columns=input_columns,
        label_column=label_column,
    )


def _development_frame(*, filter_flip: bool = False) -> pd.DataFrame:
    frame = make_prediction_frame(filter_flip=filter_flip)
    for column in ("timestamp", "label_available_at"):
        frame[column] = frame[column] - pd.DateOffset(years=1)
    return frame
