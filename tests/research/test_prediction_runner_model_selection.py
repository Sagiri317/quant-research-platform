from __future__ import annotations

import gc
import json
import warnings
import weakref
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.predictions.runner as prediction_runner
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_prediction_inputs import (
    make_prediction_frame,
)
from tests.helpers.research_specs import (
    make_demo_research_dataset_spec,
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_result


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
            context_columns=(),
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
            train_end="2024-01-03",
            test_start="2024-01-07",
            test_end="2024-01-08",
            candidate_model_ids=("candidate_model",),
            model_selection_metric="validation_mse",
            model_selection_validation_start="2024-01-04",
            model_selection_validation_end="2024-01-06",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        make_prediction_frame(),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=(),
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

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="prediction_demo",
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    model_selection = manifest["metadata"]["model_metadata"]["model_selection"]

    assert model_selection["selected_model_id"] == "candidate_model"
    assert model_selection["selection_metric"] == "validation_mse"
    assert model_selection["selection_direction"] == "minimize"
    assert model_selection["candidate_model_ids"] == ["base_model", "candidate_model"]
    assert manifest["metadata"]["model_metadata"]["input_columns"] == ["factor_b"]
    assert model_selection["selected_model_input_columns"] == ["factor_b"]


def test_prediction_runner_preloads_selected_models_before_root_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_catalog = _model_catalog(_model("base_model", ("factor_a",)))
    prediction_catalog = PredictionCatalog(
        (
            PredictionSpec(
            prediction_id="prediction_demo",
            model_id="base_model",
            research_dataset_id="research_demo",
            train_end="2024-01-03",
            test_start="2024-01-04",
            ),
        )
    )
    events: list[str] = []
    monkeypatch.setattr(
        prediction_runner,
        "preflight_model_specs",
        lambda specs: events.extend(spec.model_id for spec in specs),
    )

    def reject_operation_scope(_policy):  # noqa: ANN001
        assert events == ["base_model"]
        raise RuntimeError("operation_scope_after_model_preload")

    monkeypatch.setattr(
        prediction_runner,
        "runtime_resource_policy_scope",
        reject_operation_scope,
    )

    with pytest.raises(RuntimeError, match="operation_scope_after_model_preload"):
        build_prediction_result_from_research_dataset(
            layout=StorageLayout(project_root=tmp_path),
            dataset_catalog=ResearchDatasetCatalog(),
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="prediction_demo",
        )


def test_model_selection_compares_common_validation_primary_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_prediction_frame()
    frame.loc[frame["timestamp"].eq("2024-01-04"), "factor_a"] = np.nan
    frame.loc[frame["timestamp"].eq("2024-01-05"), "factor_b"] = np.nan
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="base_model",
        research_dataset_id="research_demo",
        train_end="2024-01-03",
        test_start="2024-01-07",
        test_end="2024-01-08",
        candidate_model_ids=("candidate_model",),
        model_selection_metric="validation_mse",
        model_selection_validation_start="2024-01-04",
        model_selection_validation_end="2024-01-06",
    )
    validation_keys: list[tuple[tuple[pd.Timestamp, str], ...]] = []
    original_metrics = prediction_runner.evaluate_model_trial_metrics_from_chunks

    def chunks(columns, *, input_columns):  # noqa: ANN001
        del input_columns
        return (frame.loc[:, list(columns)],)

    def record_validation_keys(chunks, **kwargs):  # noqa: ANN001, ANN003
        keys: list[tuple[pd.Timestamp, str]] = []

        def observed_chunks():
            for item in chunks:
                frame_value = (
                    item.value
                    if isinstance(item, prediction_runner._OwnedValue)  # noqa: SLF001
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
        prediction_runner,
        "evaluate_model_trial_metrics_from_chunks",
        record_validation_keys,
    )

    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with prediction_runner.runtime_resource_policy_scope(policy):
        prediction_runner._select_prediction_model_spec(
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
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "asset_id": ["asset"],
            "factor_a": [1.0],
            "label": [1.0],
        }
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024 * 1024,
    )
    candidate_spec = SimpleNamespace(
        input_columns=("factor_a",),
        label_column="label",
    )
    prediction_spec = SimpleNamespace(
        sample_filter_mode="unfiltered",
        prediction_column="prediction",
    )
    model_result = prediction_runner.fitting.ModelFitResult(
        coefficients=np.array([0.0, 1.0]),
        metadata={},
    )

    with prediction_runner.runtime_resource_policy_scope(policy):
        result = list(
            prediction_runner._iter_validation_prediction_chunks(  # noqa: SLF001
                lambda: iter((frame,)),
                prediction_spec=prediction_spec,
                candidate_spec=candidate_spec,
                model_result=model_result,
                validation_start=pd.Timestamp("2024-01-02"),
                validation_end=pd.Timestamp("2024-01-03"),
            )
        )

    assert result == []
    assert policy.used_memory_bytes == 0


def test_model_selection_drops_candidate_model_alias_before_lease_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = make_prediction_frame()
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="base_model",
        research_dataset_id="research_demo",
        train_end="2024-01-03",
        test_start="2024-01-07",
        test_end="2024-01-08",
        candidate_model_ids=("candidate_model",),
        model_selection_metric="validation_mse",
        model_selection_validation_start="2024-01-04",
        model_selection_validation_end="2024-01-06",
    )
    created_refs: list[weakref.ReferenceType[object]] = []

    class Predictor:
        def predict(self, values: pd.DataFrame) -> np.ndarray:
            return np.ones(len(values), dtype="float64")

    def fit_owned(*_args, **_kwargs):  # noqa: ANN002, ANN003
        predictor = Predictor()
        created_refs.append(weakref.ref(predictor))
        lease = prediction_runner.current_runtime_resource_policy().lease(
            "research.test.candidate_model",
            1024,
        )
        return prediction_runner._OwnedValue(  # noqa: SLF001
            prediction_runner.fitting.ModelFitResult(
                coefficients=np.array([0.0, 1.0]),
                metadata={},
                prediction_model=predictor,
            ),
            lease,
        )

    monkeypatch.setattr(
        prediction_runner.fitting,
        "_fit_model_result_owned_from_frames",
        fit_owned,
    )

    def chunks(columns, *, input_columns):  # noqa: ANN001
        del input_columns
        return (frame.loc[:, list(columns)],)

    policy = capture_test_policy(
        available_memory_bytes=128 * 1024**2,
        explicit_cap_bytes=64 * 1024**2,
    )
    with runtime_resource_policy_scope(policy):
        prediction_runner._select_prediction_model_spec(
            prediction_spec=prediction_spec,
            dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
            model_catalog=model_catalog,
            chunks=chunks,
        )

    assert created_refs
    assert all(reference() is None for reference in created_refs)
    assert policy.used_memory_bytes == 0


def test_prediction_build_failure_drops_model_alias_before_lease_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_spec = _model("base_model", ("factor_a",))
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="base_model",
        research_dataset_id="research_demo",
        train_end="2024-01-03",
        test_start="2024-01-07",
        test_end="2024-01-08",
    )
    dataset_spec = SimpleNamespace(
        primary_key=("timestamp", "asset_id"),
        market_id=MarketId("a_share"),
        data_id="research_demo",
    )
    created_refs: list[weakref.ReferenceType[object]] = []

    class Predictor:
        def predict(self, values: pd.DataFrame) -> np.ndarray:
            return np.ones(len(values), dtype="float64")

    closed_leases: list[object] = []

    class ObservingLease:
        active = True
        close_count = 0

        def close(self) -> None:
            assert self.active
            self.active = False
            self.close_count += 1
            closed_leases.append(self)

    def fit_owned(*_args, **_kwargs):  # noqa: ANN002, ANN003
        predictor = Predictor()
        created_refs.append(weakref.ref(predictor))
        lease = ObservingLease()
        return prediction_runner._OwnedValue(  # noqa: SLF001
            prediction_runner.fitting.ModelFitResult(
                coefficients=np.array([0.0, 1.0]),
                metadata={},
                prediction_model=predictor,
            ),
            lease,
        )

    def fail_build(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("prediction_build_failure")

    monkeypatch.setattr(
        prediction_runner.fitting,
        "_fit_model_result_owned_from_frames",
        fit_owned,
    )
    monkeypatch.setattr(
        prediction_runner,
        "build_prediction_result_from_frames",
        fail_build,
    )

    with pytest.raises(RuntimeError, match="prediction_build_failure"):
        prediction_runner._build_prediction_result_from_research_dataset(  # noqa: SLF001
            layout=SimpleNamespace(),
            dataset_catalog=SimpleNamespace(get=lambda _dataset_id: dataset_spec),
            model_catalog=SimpleNamespace(get=lambda _model_id: model_spec),
            prediction_catalog=SimpleNamespace(
                get=lambda _prediction_id: prediction_spec
            ),
            prediction_id="prediction_demo",
            time_window=None,
            prediction_train_end=None,
            prediction_test_start=None,
            prediction_test_end=None,
        )

    assert created_refs
    gc.collect()
    assert all(reference() is None for reference in created_refs)
    assert len(closed_leases) == 1
    assert closed_leases[0].close_count == 1
    assert not closed_leases[0].active


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
                        "2024-01-01", periods=len(validation), freq="D"
                    ),
                    asset_id="asset",
                ),
            ),
            model_id="candidate_model",
            label_column="label",
            prediction_column="prediction",
            purge_gap_steps=0,
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
            context_columns=(),
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
            train_end="2024-01-03",
            test_start="2024-01-07",
            test_end="2024-01-08",
            candidate_model_ids=("candidate_model",),
            model_selection_metric="validation_mse",
            model_selection_validation_start="2024-01-04",
            model_selection_validation_end="2024-01-06",
            sample_filter_mode="eligible",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        make_prediction_frame(filter_flip=True),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label",),
            filter_columns=("eligible",),
            context_columns=(),
        ),
    )

    result = build_prediction_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        model_catalog=model_catalog,
        prediction_catalog=prediction_catalog,
        prediction_id="prediction_demo",
    )
    output_bundle = write_prediction_result(layout, result)
    manifest = json.loads((output_bundle / "manifest.json").read_text(encoding="utf-8"))
    model_selection = manifest["metadata"]["model_metadata"]["model_selection"]

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
            context_columns=(),
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
            train_end="2024-01-03",
            test_start="2024-01-07",
            candidate_model_ids=("candidate_model",),
            model_selection_validation_start="2024-01-04",
            model_selection_validation_end="2024-01-06",
            ),
        )
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), "research_demo"),
        make_prediction_frame().assign(other_label=1.0),
        spec=make_demo_research_dataset_spec(
            data_id="research_demo",
            input_data_ids=("processed_demo",),
            base_input_data_id="processed_demo",
            factor_columns=("factor_a", "factor_b"),
            label_columns=("label", "other_label"),
            filter_columns=("eligible",),
            context_columns=(),
        ),
    )

    with pytest.raises(ValueError, match="model_selection_label_column_mismatch"):
        build_prediction_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=model_catalog,
            prediction_catalog=prediction_catalog,
            prediction_id="prediction_demo",
        )


def test_prediction_runner_accepts_equivalent_timezone_representations() -> None:
    chunk = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp(
                    "2024-01-15T12:00:00",
                    tz=timezone(timedelta(hours=-5)),
                ),
                pd.Timestamp(
                    "2024-01-15T12:01:00",
                    tz=ZoneInfo("America/New_York"),
                ),
            ]
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        timestamps = prediction_runner._prediction_timestamps(chunk)

    assert timestamps.tolist() == [
        pd.Timestamp("2024-01-15T17:00:00"),
        pd.Timestamp("2024-01-15T17:01:00"),
    ]


def test_prediction_lazy_root_waits_for_chunks_and_metadata(monkeypatch) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    configured = SimpleNamespace(model_id="model", candidate_model_ids=())
    monkeypatch.setattr(prediction_runner, "preflight_model_specs", lambda _specs: None)
    monkeypatch.setattr(
        prediction_runner,
        "_build_prediction_result_from_research_dataset",
        lambda **_kwargs: prediction_runner.PredictionResult(
            spec=SimpleNamespace(),
            output_chunks=(pd.DataFrame({"prediction": [1.0]}),),
            model_metadata={},
            prediction_metadata_factory=lambda: {"rows": 1},
        ),
    )

    result = build_prediction_result_from_research_dataset(
        layout=SimpleNamespace(),
        dataset_catalog=SimpleNamespace(),
        model_catalog=SimpleNamespace(),
        prediction_catalog=SimpleNamespace(get=lambda _prediction_id: configured),
        prediction_id="prediction",
        resource_policy=policy,
    )

    with pytest.raises(
        RuntimeError, match="prediction_metadata_before_output_complete"
    ):
        result.prediction_metadata_factory()
    assert prediction_runner.current_runtime_resource_policy() is policy
    assert len(list(result.output_chunks)) == 1
    assert prediction_runner.current_runtime_resource_policy() is policy
    assert result.prediction_metadata_factory() == {"rows": 1}
    assert prediction_runner.current_runtime_resource_policy() is policy
    result.close()
    assert prediction_runner.current_runtime_resource_policy() is None


def test_prediction_result_explicit_close_owns_operation_root(monkeypatch) -> None:
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    configured = SimpleNamespace(model_id="model", candidate_model_ids=())
    monkeypatch.setattr(prediction_runner, "preflight_model_specs", lambda _specs: None)
    monkeypatch.setattr(
        prediction_runner,
        "_build_prediction_result_from_research_dataset",
        lambda **_kwargs: prediction_runner.PredictionResult(
            spec=SimpleNamespace(),
            output_chunks=(pd.DataFrame({"prediction": [1.0]}),),
            model_metadata={},
            prediction_metadata_factory=lambda: {"rows": 1},
        ),
    )
    result = build_prediction_result_from_research_dataset(
        layout=SimpleNamespace(),
        dataset_catalog=SimpleNamespace(),
        model_catalog=SimpleNamespace(),
        prediction_catalog=SimpleNamespace(get=lambda _prediction_id: configured),
        prediction_id="prediction",
        resource_policy=policy,
    )

    assert prediction_runner.current_runtime_resource_policy() is policy
    result.close()
    result.close()
    assert prediction_runner.current_runtime_resource_policy() is None
    with pytest.raises(RuntimeError, match="prediction_result_closed"):
        list(result.output_chunks)
    with pytest.raises(RuntimeError, match="prediction_result_closed"):
        result.prediction_metadata_factory()


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
    assert prediction_runner.current_runtime_resource_policy() is None


def _model_catalog(*models: ModelSpec) -> ModelCatalog:
    return ModelCatalog(models)


def _model(
    model_id: str,
    input_columns: tuple[str, ...],
    *,
    label_column: str = "label",
    purge_gap_steps: int = 0,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        model_algorithm="linear_regression",
        input_factor_columns=input_columns,
        label_column=label_column,
        purge_gap_steps=purge_gap_steps,
    )
