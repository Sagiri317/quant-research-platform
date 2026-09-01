from __future__ import annotations

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
        {
            "trial_id": ["trial_b", "trial_a"],
            "validation_rankic": [0.7, 0.7],
        }
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
