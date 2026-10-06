from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    select_best_validation_trial,
    validate_trial_metric_columns,
)


def test_model_selection_uses_validation_only_and_explicit_ties() -> None:
    validate_trial_metric_columns(
        selection_metric="validation_rankic",
        validation_metric_columns=("validation_rankic",),
        test_metric_columns=("test_rankic",),
    )
    trials = pd.DataFrame(
        {"trial_id": ["trial_b", "trial_a"], "validation_rankic": [0.7, 0.7]}
    )
    assert (
        select_best_validation_trial(
            trials,
            trial_id_column="trial_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
            tie_break_order=("trial_a", "trial_b"),
        )
        == "trial_a"
    )
    with pytest.raises(ValueError, match="selection_metric_must_be_validation_metric"):
        validate_trial_metric_columns(
            selection_metric="test_rankic",
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=("test_rankic",),
        )


def test_model_selection_metrics_are_cross_sectional_and_chunk_parity_stable() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            "asset_id": ["a", "b", "c"] * 2,
            "label": [1.0, 2.0, 3.0, 101.0, 102.0, 103.0],
            "prediction": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    partitioned = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:2], frame.iloc[2:5], frame.iloc[5:]),
        model_id="current",
        label_column="label",
        prediction_column="prediction",
    )
    assert partitioned == whole
    assert whole["validation_rows"] == 6
    assert whole["validation_ic"] == pytest.approx(0.0)
    assert whole["validation_rankic"] == pytest.approx(0.0)


def test_model_selection_rejects_nonfinite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 2),
            "asset_id": ["a", "b"],
            "label": [1.0, 2.0],
            "prediction": [np.nan, 1.0],
        }
    )
    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="current",
            label_column="label",
            prediction_column="prediction",
        )


def test_time_series_trial_metrics_keep_axis_across_chunks_and_degeneracy():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6),
            "asset_id": "a",
            "label": [2.0, 4.0, 1.0, 6.0, 3.0, 5.0],
            "prediction": [5.0, 9.0, 3.0, 13.0, 7.0, 11.0],
        }
    )
    arguments = dict(
        model_id="model", label_column="label", prediction_column="prediction"
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,), **arguments, validation_axis="single_asset_time_series"
    )
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:2], frame.iloc[:0], frame.iloc[2:5], frame.iloc[5:]),
        **arguments,
        validation_axis="single_asset_time_series",
    )
    assert split == whole
    assert whole["validation_ic"] == 1.0
    assert whole["validation_rankic"] is None
    assert whole["validation_rankic_reason"] == "not_applicable"
    cross_sectional = evaluate_model_trial_metrics_from_chunks((frame,), **arguments)
    assert cross_sectional["validation_ic"] is None
    assert cross_sectional["validation_ic_reason"] == "no_valid_observations"
    single = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1],), **arguments, validation_axis="single_asset_time_series"
    )
    assert single["validation_ic_reason"] == "no_valid_observations"
    assert single["validation_rankic_reason"] == "not_applicable"
    constant = evaluate_model_trial_metrics_from_chunks(
        (frame.assign(prediction=1.0),),
        **arguments,
        validation_axis="single_asset_time_series",
    )
    assert constant["validation_ic_reason"] == "zero_variance"


@pytest.mark.parametrize("residual", [1e155, 1e-155, 2e-161, 3e-161, 0.0, 2.0])
def test_model_selection_mse_scales_before_squaring(residual: float) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100),
            "asset_id": "asset",
            "label": 0.0,
            "prediction": [residual, *[0.0] * 99],
        }
    )
    whole = evaluate_model_trial_metrics_from_chunks(
        (frame,), model_id="model", label_column="label", prediction_column="prediction"
    )
    split = evaluate_model_trial_metrics_from_chunks(
        (frame.iloc[:1], frame.iloc[1:]),
        model_id="model",
        label_column="label",
        prediction_column="prediction",
    )
    assert whole == split
    assert whole["validation_mse"] == pytest.approx(
        float(Fraction.from_float(residual) ** 2 / len(frame)), rel=1e-12, abs=0.0
    )
    assert whole["validation_ic"] is None
    assert whole["validation_rankic"] is None


def test_model_selection_mse_rejects_unrepresentable_final_value() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "asset_id": "a",
            "label": 0.0,
            "prediction": 1e155,
        }
    )
    with pytest.raises(ValueError, match="model_selection_mse_unrepresentable"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="model",
            label_column="label",
            prediction_column="prediction",
        )


@pytest.mark.parametrize("field", ["validation_ic", "validation_rankic"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_persisted_trial_correlations_obey_exact_bounds(field, sign):
    import math

    from quant_research_platform.contracts.models import (
        validate_completed_model_trial_metrics,
    )

    metrics = {
        "validation_rows": 2,
        "validation_ic": 1.0,
        "validation_ic_reason": None,
        "validation_rankic": 1.0,
        "validation_rankic_reason": None,
        "validation_mse": 0.0,
        "n_train": 2,
    }
    metrics[field] = sign
    validate_completed_model_trial_metrics("model", metrics)
    metrics[field] = sign * math.nextafter(1.0, math.inf)
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        validate_completed_model_trial_metrics("model", metrics)
