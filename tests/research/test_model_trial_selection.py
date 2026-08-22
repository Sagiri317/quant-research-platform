from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.research._memory import _selected_frame_payload_bytes
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
)
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics,
    select_best_validation_trial,
    validate_trial_metric_columns,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.helpers.runtime_resources import capture_test_policy


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

    with pytest.raises(ValueError, match="all_validation_metrics_undefined"):
        select_best_validation_trial(
            pd.DataFrame({"trial_id": ["a"], "validation_loss": [float("nan")]}),
            **kwargs,
        )
    with pytest.raises(ValueError, match="all_validation_metrics_undefined"):
        select_best_validation_trial(
            pd.DataFrame({"trial_id": ["a", "b"], "validation_loss": [None, None]}),
            **kwargs,
        )

    with pytest.raises(ValueError, match="duplicate_trial_id"):
        select_best_validation_trial(
            pd.DataFrame(
                {"trial_id": ["a", "a"], "validation_loss": [0.2, 0.1]}
            ),
            **kwargs,
        )


def test_evaluate_model_trial_metrics_marks_one_row_correlation_undefined() -> None:
    metrics = evaluate_model_trial_metrics(
        pd.DataFrame({"label": [1.0], "prediction": [1.0]}),
        model_id="single_row",
        label_column="label",
        prediction_column="prediction",
    )

    assert metrics["validation_rows"] == 1
    assert metrics["validation_ic"] is None
    assert metrics["validation_ic_reason"] == "insufficient_observations"
    assert metrics["validation_rankic"] is None
    assert metrics["validation_rankic_reason"] == "insufficient_observations"
    assert metrics["validation_mse"] == 0.0


def test_evaluate_model_trial_metrics_marks_constant_correlation_undefined() -> None:
    metrics = evaluate_model_trial_metrics(
        pd.DataFrame(
            {
                "label": [1.0, 1.0, 1.0],
                "prediction": [2.0, 2.0, 2.0],
            }
        ),
        model_id="constant_window",
        label_column="label",
        prediction_column="prediction",
    )

    assert metrics["validation_ic"] is None
    assert metrics["validation_ic_reason"] == "zero_variance"
    assert metrics["validation_rankic"] is None
    assert metrics["validation_rankic_reason"] == "zero_variance"


def test_select_best_validation_trial_excludes_undefined_candidates() -> None:
    selected = select_best_validation_trial(
        pd.DataFrame(
            {
                "trial_id": ["undefined", "defined"],
                "validation_ic": [float("nan"), 0.2],
            }
        ),
        trial_id_column="trial_id",
        selection_metric="validation_ic",
        maximize=True,
        validation_metric_columns=("validation_ic",),
        test_metric_columns=(),
    )
    assert selected == "defined"


def test_select_best_validation_trial_rejects_numeric_text_and_mixed_values() -> None:
    kwargs = {
        "trial_id_column": "trial_id",
        "selection_metric": "validation_ic",
        "maximize": True,
        "validation_metric_columns": ("validation_ic",),
        "test_metric_columns": (),
    }
    for values in (["0.2", "0.1"], [0.2, "0.1"], ["0.2", None]):
        with pytest.raises(ValueError, match="invalid_trial_selection_metric"):
            select_best_validation_trial(
                pd.DataFrame({"trial_id": ["a", "b"], "validation_ic": values}),
                **kwargs,
            )


def test_ic_weighted_rank_ensemble_fails_when_correlations_are_undefined() -> None:
    constant = pd.DataFrame(
        {
            "factor": [1.0, 1.0, 1.0],
            "feature": [2.0, 2.0, 2.0],
            "label": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="all_ensemble_correlations_undefined"):
        ICWeightedRankEnsemble.fit(
            constant,
            input_columns=("factor", "feature"),
            label_column="label",
        )


def test_ic_weighted_rank_ensemble_fails_when_defined_weights_are_zero() -> None:
    zero_corr = pd.DataFrame(
        {
            "factor": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "feature": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "label": [-2.0, 2.0, 1.0, 0.0, -1.0],
        }
    )

    with pytest.raises(ValueError, match="all_ensemble_correlations_zero"):
        ICWeightedRankEnsemble.fit(
            zero_corr,
            input_columns=("factor", "feature"),
            label_column="label",
        )


def test_ic_weighted_rank_ensemble_excludes_only_undefined_inputs() -> None:
    frame = pd.DataFrame(
        {
            "factor": [1.0, 2.0, 3.0, 4.0],
            "constant": [7.0, 7.0, 7.0, 7.0],
            "label": [10.0, 20.0, 30.0, 40.0],
        }
    )

    ensemble = ICWeightedRankEnsemble.fit(
        frame,
        input_columns=("factor", "constant"),
        label_column="label",
    )

    assert ensemble.columns == ("factor",)
    assert ensemble.coefficient_count == 1
    assert ensemble.weights.tolist() == pytest.approx([1.0])
    assert ensemble.predict(frame).tolist() == pytest.approx(
        [0.25, 0.5, 0.75, 1.0]
    )


def test_evaluate_model_trial_metrics_does_not_recharge_borrowed_input() -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["asset-" + "x" * 5000] * 3,
            "label": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.5, 2.5],
        }
    )
    selected_bytes = _selected_frame_payload_bytes(
        frame,
        ["label", "prediction"],
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=(
            2 * selected_bytes + len(frame) * 8 * 6 + len(frame) * 3 + 1024
        ),
    )

    with runtime_resource_policy_scope(policy):
        metrics = evaluate_model_trial_metrics(
            frame,
            model_id="borrowed_input",
            label_column="label",
            prediction_column="prediction",
        )

    assert metrics["validation_rows"] == 3
    assert policy.used_memory_bytes == 0


def test_evaluate_model_trial_metrics_rejects_before_new_selection_copy() -> None:
    frame = pd.DataFrame(
        {
            "label": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.5, 2.5],
        }
    )
    selected_bytes = _selected_frame_payload_bytes(
        frame,
        ["label", "prediction"],
    )
    requested = (
        selected_bytes * 2
        + len(frame) * 8 * 6
        + len(frame) * 3
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=requested - 1,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            evaluate_model_trial_metrics(
                frame,
                model_id="insufficient_selection_work",
                label_column="label",
                prediction_column="prediction",
            )

    assert policy.used_memory_bytes == 0
