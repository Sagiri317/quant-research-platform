from __future__ import annotations

import json
import warnings
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
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_prediction_inputs import (
    make_prediction_frame,
)
from tests.helpers.research_specs import (
    make_demo_research_dataset_spec,
    make_research_dataset_catalog,
)
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
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
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
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="prediction_demo",
            model_id="base_model",
            research_dataset_id="research_demo",
            train_end="2024-01-03",
            test_start="2024-01-04",
        )
    )
    events: list[str] = []
    monkeypatch.setattr(
        prediction_runner,
        "preflight_model_specs",
        lambda specs: events.extend(spec.model_id for spec in specs),
    )

    def reject_capture(cls, **_kwargs):  # noqa: ANN001
        assert events == ["base_model"]
        raise RuntimeError("capture_after_model_preload")

    monkeypatch.setattr(
        prediction_runner.RuntimeResourcePolicy,
        "capture",
        classmethod(reject_capture),
    )

    with pytest.raises(RuntimeError, match="capture_after_model_preload"):
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
    original_metrics = prediction_runner.evaluate_model_trial_metrics

    def chunks(columns, *, input_columns):  # noqa: ANN001
        del input_columns
        return (frame.loc[:, list(columns)],)

    def record_validation_keys(validation, **kwargs):  # noqa: ANN001, ANN003
        validation_keys.append(
            tuple(
                validation[["timestamp", "asset_id"]].itertuples(
                    index=False,
                    name=None,
                )
            )
        )
        return original_metrics(validation, **kwargs)

    monkeypatch.setattr(
        prediction_runner,
        "evaluate_model_trial_metrics",
        record_validation_keys,
    )

    prediction_runner._select_prediction_model_spec(
        prediction_spec=prediction_spec,
        dataset_spec=SimpleNamespace(primary_key=("timestamp", "asset_id")),
        model_catalog=model_catalog,
        chunks=chunks,
    )

    assert len(validation_keys) == 2
    assert validation_keys[0] == validation_keys[1]


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
        prediction_runner.evaluate_model_trial_metrics(
            validation,
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
            context_columns=(),
        )
    )
    model_catalog = _model_catalog(
        _model("base_model", ("factor_a",)),
        _model("candidate_model", ("factor_b",)),
    )
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
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
    prediction_catalog = PredictionCatalog()
    prediction_catalog.register(
        PredictionSpec(
            prediction_id="prediction_demo",
            model_id="base_model",
            research_dataset_id="research_demo",
            train_end="2024-01-03",
            test_start="2024-01-07",
            candidate_model_ids=("candidate_model",),
            model_selection_validation_start="2024-01-04",
            model_selection_validation_end="2024-01-06",
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


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    ("purge_gap_steps", "expected_keys"),
    [
        (0, (("dense", 1), ("dense", 2), ("dense", 3), ("sparse", 1))),
        (2, (("dense", 1),)),
        (3, None),
    ],
)
def test_model_selection_purges_each_assets_validation_tail_before_test_window(
    chunked: bool,
    purge_gap_steps: int,
    expected_keys: tuple[tuple[str, int], ...] | None,
) -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    dense = pd.DataFrame(
        {
            "timestamp": dates,
            "asset_id": "dense",
            "factor_a": np.arange(len(dates), dtype="float64"),
            "adj_open": np.arange(10.0, 16.0),
            "eligible": True,
        }
    )
    sparse = pd.DataFrame(
        {
            "timestamp": dates[[0, 1, 4, 5]],
            "asset_id": "sparse",
            "factor_a": np.arange(4, dtype="float64"),
            "adj_open": [20.0, 21.0, 50.0, 60.0],
            "eligible": True,
        }
    )
    labeled = add_forward_return_labels(
        pd.concat([dense, sparse], ignore_index=True),
        horizons=(1,),
    )
    sparse_validation_label = labeled.loc[
        labeled["asset_id"].eq("sparse")
        & labeled["timestamp"].eq(dates[1]),
        "forward_return_1d",
    ].item()
    assert sparse_validation_label == pytest.approx(60.0 / 50.0 - 1.0)

    candidate_spec = _model(
        "linear_model",
        ("factor_a",),
        label_column="forward_return_1d",
        purge_gap_steps=purge_gap_steps,
    )
    prediction_spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id=candidate_spec.model_id,
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-05",
        test_end="2024-01-06",
        candidate_model_ids=("candidate_model",),
        model_selection_validation_start="2024-01-02",
        model_selection_validation_end="2024-01-04",
    )
    chunks = (
        (
            labeled.loc[labeled["asset_id"].eq("dense")],
            labeled.loc[labeled["asset_id"].eq("sparse")],
        )
        if chunked
        else (labeled,)
    )

    def validation_frame() -> pd.DataFrame:
        return prediction_runner._validation_prediction_frame(
            chunks,
            prediction_spec=prediction_spec,
            candidate_spec=candidate_spec,
            model_result=prediction_runner.fitting.ModelFitResult(
                coefficients=np.array([0.0, 1.0]),
                metadata={},
            ),
            validation_start=dates[1],
            validation_end=dates[3],
        )

    if expected_keys is None:
        with pytest.raises(ValueError, match="empty_model_selection_validation_window"):
            validation_frame()
        return
    validation = validation_frame()
    assert list(validation[["asset_id", "timestamp"]].itertuples(index=False)) == [
        (asset_id, dates[date_position])
        for asset_id, date_position in expected_keys
    ]


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


def _model_catalog(*models: ModelSpec) -> ModelCatalog:
    catalog = ModelCatalog()
    for model in models:
        catalog.register(model)
    return catalog


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
