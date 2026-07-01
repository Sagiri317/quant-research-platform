from __future__ import annotations

import pytest

from quant_research_platform.contracts.predictions import PredictionSpec


def test_prediction_spec_accepts_model_selection_contract_fields() -> None:
    spec = PredictionSpec(
        prediction_id="prediction_demo",
        model_id="linear_model",
        research_dataset_id="research_demo",
        train_end="2024-01-31",
        test_start="2024-03-01",
        candidate_model_ids=("ridge_model",),
        model_selection_metric="validation_mse",
        model_selection_validation_start="2024-02-01",
        model_selection_validation_end="2024-02-29",
    )

    assert spec.candidate_model_ids == ("ridge_model",)
    assert spec.model_selection_metric == "validation_mse"


def test_prediction_spec_rejects_invalid_model_selection_candidates() -> None:
    kwargs = {
        "prediction_id": "prediction_demo",
        "model_id": "linear_model",
        "research_dataset_id": "research_demo",
        "train_end": "2024-01-31",
        "test_start": "2024-03-01",
        "model_selection_validation_start": "2024-02-01",
        "model_selection_validation_end": "2024-02-29",
    }

    with pytest.raises(
        ValueError,
        match="candidate_model_ids_must_not_contain_duplicates",
    ):
        PredictionSpec(**kwargs, candidate_model_ids=("ridge_model", "ridge_model"))

    with pytest.raises(
        ValueError,
        match="candidate_model_ids_must_not_include_model_id",
    ):
        PredictionSpec(**kwargs, candidate_model_ids=("linear_model",))


def test_prediction_spec_rejects_invalid_model_selection_window() -> None:
    kwargs = {
        "prediction_id": "prediction_demo",
        "model_id": "linear_model",
        "research_dataset_id": "research_demo",
        "train_end": "2024-01-31",
        "test_start": "2024-03-01",
        "candidate_model_ids": ("ridge_model",),
    }

    with pytest.raises(ValueError, match="model_selection_validation_window_required"):
        PredictionSpec(**kwargs)

    with pytest.raises(ValueError, match="invalid_model_selection_metric"):
        PredictionSpec(
            **kwargs,
            model_selection_metric="test_ic",
            model_selection_validation_start="2024-02-01",
            model_selection_validation_end="2024-02-29",
        )

    with pytest.raises(
        ValueError,
        match="model_selection_window_must_follow_train_window",
    ):
        PredictionSpec(
            **kwargs,
            model_selection_validation_start="2024-01-31",
            model_selection_validation_end="2024-02-29",
        )

    with pytest.raises(
        ValueError,
        match="model_selection_window_must_precede_test_window",
    ):
        PredictionSpec(
            **kwargs,
            model_selection_validation_start="2024-02-01",
            model_selection_validation_end="2024-03-01",
        )
