from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics,
    select_best_validation_trial,
    validate_trial_metric_columns,
)


def test_trial_selection_requires_validation_metric_only() -> None:
    validate_trial_metric_columns(
        selection_metric="validation_ic",
        validation_metric_columns=("validation_ic",),
        test_metric_columns=("test_ic",),
    )

    with pytest.raises(ValueError, match="selection_metric_must_be_validation_metric"):
        validate_trial_metric_columns(
            selection_metric="test_ic",
            validation_metric_columns=("validation_ic",),
            test_metric_columns=("test_ic",),
        )

    with pytest.raises(ValueError, match="selection_metric_must_not_be_test_metric"):
        validate_trial_metric_columns(
            selection_metric="shared_ic",
            validation_metric_columns=("shared_ic",),
            test_metric_columns=("shared_ic",),
        )


def test_select_best_validation_trial_is_deterministic_on_ties() -> None:
    trials = pd.DataFrame(
        {
            "trial_id": ["trial_b", "trial_a", "trial_c"],
            "validation_ic": [0.7, 0.7, 0.2],
            "test_ic": [0.1, 0.9, 0.8],
        }
    )

    selected = select_best_validation_trial(
        trials,
        trial_id_column="trial_id",
        selection_metric="validation_ic",
        maximize=True,
        validation_metric_columns=("validation_ic",),
        test_metric_columns=("test_ic",),
    )

    assert selected == "trial_a"


def test_select_best_validation_trial_fails_closed_on_bad_inputs() -> None:
    kwargs = {
        "trial_id_column": "trial_id",
        "selection_metric": "validation_loss",
        "maximize": False,
        "validation_metric_columns": ("validation_loss",),
        "test_metric_columns": ("test_loss",),
    }

    with pytest.raises(ValueError, match="empty_trials"):
        select_best_validation_trial(pd.DataFrame(), **kwargs)

    with pytest.raises(ValueError, match="missing_trial_selection_columns"):
        select_best_validation_trial(pd.DataFrame({"trial_id": ["a"]}), **kwargs)

    with pytest.raises(ValueError, match="nan_trial_selection_metric"):
        select_best_validation_trial(
            pd.DataFrame({"trial_id": ["a"], "validation_loss": [float("nan")]}),
            **kwargs,
        )

    with pytest.raises(ValueError, match="duplicate_trial_id"):
        select_best_validation_trial(
            pd.DataFrame(
                {"trial_id": ["a", "a"], "validation_loss": [0.2, 0.1]}
            ),
            **kwargs,
        )


def test_evaluate_model_trial_metrics_returns_zero_correlation_for_one_row() -> None:
    metrics = evaluate_model_trial_metrics(
        pd.DataFrame({"label": [1.0], "prediction": [1.0]}),
        model_id="single_row",
        label_column="label",
        prediction_column="prediction",
    )

    assert metrics["validation_rows"] == 1
    assert metrics["validation_ic"] == 0.0
    assert metrics["validation_rankic"] == 0.0
    assert metrics["validation_mse"] == 0.0
