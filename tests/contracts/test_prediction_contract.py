from __future__ import annotations

import pytest

from quant_research_platform.contracts.predictions import PredictionSpec


def test_prediction_mode_rejects_unimplemented_smoke_alias() -> None:
    with pytest.raises(ValueError, match="unsupported_prediction_mode"):
        PredictionSpec(
            prediction_id="prediction_demo",
            model_id="linear_model",
            research_dataset_id="research_demo",
            train_end="2024-01-31",
            test_start="2024-03-01",
            prediction_mode="oos_smoke",
        )


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


def test_prediction_spec_accepts_intraday_windows_and_normalizes_utc() -> None:
    spec = PredictionSpec(
        prediction_id="intraday_prediction",
        model_id="linear_model",
        research_dataset_id="intraday_research",
        train_end="2026-08-27T09:00:00+08:00",
        test_start="2026-08-27T09:00:00.100+08:00",
        test_end="2026-08-27T09:00:01+08:00",
    )

    assert spec.train_end == "2026-08-27T01:00:00"
    assert spec.test_start == "2026-08-27T01:00:00.100000"
    assert spec.test_end == "2026-08-27T01:00:01"


def test_prediction_spec_rejects_mixed_date_and_intraday_windows() -> None:
    with pytest.raises(ValueError, match="mixed_time_window_bounds"):
        PredictionSpec(
            prediction_id="mixed_window_prediction",
            model_id="linear_model",
            research_dataset_id="intraday_research",
            train_end="2026-08-26",
            test_start="2026-08-27T00:00:00",
        )


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


def test_prediction_spec_binds_paired_train_and_evaluation_asset_scopes() -> None:
    spec = PredictionSpec(
        prediction_id="btc_leave_out",
        model_id="ridge_regression",
        research_dataset_id="crypto_hft",
        train_end="2024-01-31T00:00:00",
        test_start="2024-02-01T00:00:00",
        training_asset_ids=("eth_usdt",),
        evaluation_asset_ids=("btc_usdt",),
    )
    assert spec.training_asset_ids == ("eth_usdt",)
    assert spec.evaluation_asset_ids == ("btc_usdt",)

    with pytest.raises(ValueError, match="prediction_asset_scopes_must_be_paired"):
        PredictionSpec(
            prediction_id="bad_scope",
            model_id="ridge_regression",
            research_dataset_id="crypto_hft",
            train_end="2024-01-31T00:00:00",
            test_start="2024-02-01T00:00:00",
            training_asset_ids=("eth_usdt",),
        )
