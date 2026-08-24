from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.models.trial_selection as trial_selection
from quant_research_platform.research._memory import _selected_frame_payload_bytes
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
)
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    select_best_validation_trial,
    validate_trial_metric_columns,
)
from quant_research_platform.runtime.resources import (
    _OwnedValue,
    runtime_resource_policy_scope,
)
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


def test_chunked_trial_metrics_marks_one_row_correlation_undefined() -> None:
    metrics = evaluate_model_trial_metrics_from_chunks(
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01"]),
                    "asset_id": ["asset"],
                    "label": [1.0],
                    "prediction": [1.0],
                }
            ),
        ),
        model_id="single_row",
        label_column="label",
        prediction_column="prediction",
        purge_gap_steps=0,
    )

    assert metrics["validation_rows"] == 1
    assert metrics["validation_ic"] is None
    assert metrics["validation_ic_reason"] == "insufficient_observations"
    assert metrics["validation_rankic"] is None
    assert metrics["validation_rankic_reason"] == "insufficient_observations"
    assert metrics["validation_mse"] == 0.0


def test_chunked_trial_metrics_marks_constant_correlation_undefined() -> None:
    metrics = evaluate_model_trial_metrics_from_chunks(
        (
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=3, freq="D"
                    ),
                    "asset_id": ["asset"] * 3,
                    "label": [1.0, 1.0, 1.0],
                    "prediction": [2.0, 2.0, 2.0],
                }
            ),
        ),
        model_id="constant_window",
        label_column="label",
        prediction_column="prediction",
        purge_gap_steps=0,
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


def test_ic_weighted_rank_ensemble_rejects_rank_roundoff_zero() -> None:
    frame = pd.DataFrame(
        {
            "factor": range(7),
            "label": [0, 3, 5, 6, 4, 2, 1],
        }
    )

    with pytest.raises(ValueError, match="all_ensemble_correlations_zero"):
        ICWeightedRankEnsemble.fit(
            frame,
            input_columns=("factor",),
            label_column="label",
        )


def test_ic_weighted_rank_ensemble_handles_per_timestamp_rank_denominators() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01"] * 3 + ["2024-01-02"] * 6
            ),
            "factor": [0, 1, 2, 0, 1, 2, 3, 4, 5],
            "label": [1, 2, 3, 3, 4, 6, 5, 2, 1],
        }
    )

    with pytest.raises(ValueError, match="all_ensemble_correlations_zero"):
        ICWeightedRankEnsemble.fit(
            frame,
            input_columns=("factor",),
            label_column="label",
        )


def test_ic_weighted_rank_ensemble_keeps_exact_nonzero_near_roundoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "factor": range(7),
            "label": range(7),
        }
    )
    monkeypatch.setattr(
        "quant_research_platform.research.models.rank_ensemble._pearson_correlation_result",
        lambda *_args: (float(np.finfo(np.float64).eps), None),
    )

    ensemble = ICWeightedRankEnsemble.fit(
        frame,
        input_columns=("factor",),
        label_column="label",
    )

    assert ensemble.weights.tolist() == [1.0]


def test_ic_weighted_rank_ensemble_preserves_genuine_nonzero_rank_signal() -> None:
    frame = pd.DataFrame(
        {
            "factor": range(7),
            "label": [0, 1, 2, 3, 4, 6, 5],
        }
    )

    ensemble = ICWeightedRankEnsemble.fit(
        frame,
        input_columns=("factor",),
        label_column="label",
    )

    assert ensemble.columns == ("factor",)
    assert ensemble.weights[0] > 0.0


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


def test_chunked_trial_metrics_match_exact_purge_and_tie_oracle() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": list(dates) + list(dates[[0, 1, 4, 5]]),
            "asset_id": ["dense"] * 6 + ["sparse"] * 4,
            "label": [1.0, 2.0, 2.0, 4.0, 5.0, 5.0, 1.0, 1.0, 3.0, 3.0],
            "prediction": [1.0, 1.0, 2.0, 3.0, 5.0, 4.0, 1.0, 2.0, 2.0, 4.0],
        }
    )
    chunks = (frame.iloc[:5].copy(), frame.iloc[5:].copy())

    metrics = evaluate_model_trial_metrics_from_chunks(
        chunks,
        model_id="streamed",
        label_column="label",
        prediction_column="prediction",
        purge_gap_steps=2,
    )

    assert metrics["model_id"] == "streamed"
    assert metrics["validation_rows"] == 6
    assert metrics["validation_ic"] == pytest.approx(0.7682733253465358)
    assert metrics["validation_rankic"] == pytest.approx(0.5833333333333334)
    assert metrics["validation_mse"] == pytest.approx(0.5)
    assert metrics["validation_ic_reason"] is None
    assert metrics["validation_rankic_reason"] is None


def test_chunked_trial_metrics_complete_when_aggregate_exceeds_target() -> None:
    def make_chunk(chunk_index: int) -> pd.DataFrame:
        start = chunk_index * 1000
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=1000, freq="min"
                ),
                "asset_id": [
                    f"asset_{row % 10}" for row in range(start, start + 1000)
                ],
                "label": np.arange(start, start + 1000, dtype="float64"),
                "prediction": np.arange(start, start + 1000, dtype="float64") + 0.25,
            }
        )

    aggregate_bytes = 0

    def chunks():
        nonlocal aggregate_bytes
        for chunk_index in range(80):
            chunk = make_chunk(chunk_index)
            aggregate_bytes += _selected_frame_payload_bytes(
                chunk,
                ["timestamp", "asset_id", "label", "prediction"],
            )
            yield chunk

    policy = capture_test_policy(
        available_memory_bytes=4 * 1024**2,
        explicit_cap_bytes=2 * 1024**2,
    )

    with runtime_resource_policy_scope(policy):
        metrics = evaluate_model_trial_metrics_from_chunks(
            chunks(),
            model_id="large_stream",
            label_column="label",
            prediction_column="prediction",
            purge_gap_steps=0,
        )

    assert metrics["validation_rows"] == 80_000
    assert metrics["validation_mse"] == pytest.approx(0.0625)
    assert aggregate_bytes > 2 * 1024**2
    assert policy.used_memory_bytes == 0


def test_chunked_trial_metrics_releases_owned_chunk_on_primary_failure() -> None:
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024 * 1024,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D"),
            "asset_id": ["asset"] * 2,
            "label": [1.0, 2.0],
            "prediction": [1.0, 2.0],
        }
    )

    def failing_chunks():
        lease = policy.lease("tests.metric.chunk", 1024)
        yield _OwnedValue(frame, lease)
        raise RuntimeError("metric_primary")

    with runtime_resource_policy_scope(policy):
        with pytest.raises(RuntimeError, match="metric_primary"):
            evaluate_model_trial_metrics_from_chunks(
                failing_chunks(),
                model_id="failure",
                label_column="label",
                prediction_column="prediction",
                purge_gap_steps=0,
            )

    assert policy.used_memory_bytes == 0


def test_chunked_trial_metrics_preserves_primary_when_scratch_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary_directory = trial_selection.tempfile.TemporaryDirectory

    class FailingScratch(temporary_directory):
        def __exit__(self, *args):  # noqa: ANN002
            super().__exit__(*args)
            raise RuntimeError("scratch_cleanup")

    monkeypatch.setattr(trial_selection.tempfile, "TemporaryDirectory", FailingScratch)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D"),
            "asset_id": ["asset"] * 2,
            "label": [1.0, 2.0],
            "prediction": [1.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match="invalid_trial_metric_values") as raised:
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="scratch_failure",
            label_column="label",
            prediction_column="prediction",
            purge_gap_steps=0,
        )

    assert raised.value.__notes__ == [
        "model selection scratch cleanup also failed: RuntimeError"
    ]


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_chunked_trial_metrics_rejects_nonfinite_values(invalid_value: float) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "asset_id": ["asset"] * 3,
            "label": [1.0, 2.0, 3.0],
            "prediction": [1.0, invalid_value, 3.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id="invalid",
            label_column="label",
            prediction_column="prediction",
            purge_gap_steps=0,
        )


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01"]),
                    "asset_id": ["asset"],
                    "label": [1.0],
                    "prediction": [1.0],
                }
            ),
            "insufficient_observations",
        ),
        (
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=3),
                    "asset_id": ["asset"] * 3,
                    "label": [1.0, 1.0, 1.0],
                    "prediction": [2.0, 2.0, 2.0],
                }
            ),
            "zero_variance",
        ),
    ],
)
def test_chunked_trial_metrics_preserves_undefined_reasons(
    frame: pd.DataFrame,
    reason: str,
) -> None:
    metrics = evaluate_model_trial_metrics_from_chunks(
        (frame,),
        model_id="undefined",
        label_column="label",
        prediction_column="prediction",
        purge_gap_steps=0,
    )

    assert metrics["validation_ic"] is None
    assert metrics["validation_rankic"] is None
    assert metrics["validation_ic_reason"] == reason
    assert metrics["validation_rankic_reason"] == reason
