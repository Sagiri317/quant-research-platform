from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_inputs import (
    make_prediction_frame,
    make_research_dataset_catalog,
    make_research_dataset_spec,
)
from tests.research.prediction_output_helpers import write_prediction_result


def test_prediction_runner_selects_model_using_validation_metric(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_research_dataset_spec(
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
        spec=make_research_dataset_spec(
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
    assert model_selection["selection_metric"] == "validation_mse"
    assert model_selection["selection_direction"] == "minimize"
    assert model_selection["candidate_model_ids"] == ["base_model", "candidate_model"]
    assert manifest["metadata"]["model_metadata"]["input_columns"] == ["factor_b"]
    assert model_selection["selected_model_input_columns"] == ["factor_b"]


def test_prediction_runner_applies_sample_filter_to_validation_metrics(
    tmp_path: Path,
) -> None:
    layout = StorageLayout(project_root=tmp_path)
    dataset_catalog = make_research_dataset_catalog(
        make_research_dataset_spec(
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
        spec=make_research_dataset_spec(
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
        make_research_dataset_spec(
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
        spec=make_research_dataset_spec(
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
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        model_algorithm="linear_regression",
        input_factor_columns=input_columns,
        label_column=label_column,
    )
