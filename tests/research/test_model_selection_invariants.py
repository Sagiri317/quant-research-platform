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
    for prefix in ("ic", "rankic"):
        assert whole[f"validation_std_{prefix}"] == 1.0
        assert whole[f"validation_{prefix}ir"] == 0.0
        assert whole[f"validation_valid_{prefix}_count"] == 2
        assert whole[f"validation_positive_{prefix}_rate"] == 0.5
        assert whole[f"validation_{prefix}ir_reason"] is None
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
    assert whole["validation_icir"] is None
    assert whole["validation_icir_reason"] == "insufficient_observations"
    assert whole["validation_valid_ic_count"] == 1
    assert whole["validation_rankicir_reason"] == "not_applicable"
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
        _validate_model_selection_trial_record,
    )

    metrics = {
        "validation_rows": 2,
        "validation_ic": 1.0,
        "validation_std_ic": 0.0,
        "validation_std_ic_reason": None,
        "validation_icir": None,
        "validation_positive_ic_rate": 1.0,
        "validation_valid_ic_count": 1,
        "validation_icir_reason": "insufficient_observations",
        "validation_std_rankic": 0.0,
        "validation_std_rankic_reason": None,
        "validation_rankicir": None,
        "validation_positive_rankic_rate": 1.0,
        "validation_valid_rankic_count": 1,
        "validation_rankicir_reason": "insufficient_observations",
        "validation_ic_reason": None,
        "validation_rankic": 1.0,
        "validation_rankic_reason": None,
        "validation_mse": 0.0,
        "model_id": "model",
    }
    metrics[field] = sign
    _validate_model_selection_trial_record(metrics)
    metrics[field] = sign * math.nextafter(1.0, math.inf)
    with pytest.raises(ValueError, match="invalid_model_metadata"):
        _validate_model_selection_trial_record(metrics)


@pytest.mark.parametrize("axis", ["single_asset_time_series", "cross_sectional"])
def test_ordered_metrics_release_consumed_frames(monkeypatch, axis):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=8),
            "asset_id": "a",
            "label": [1.0, 5.0, -2.0, 6.0, 3.0, -4.0, 7.0, 8.0],
            "prediction": [2.0, 3.0, -4.0, 5.0, 1.0, -3.0, 9.0, 7.0],
        }
    )
    kwargs = dict(
        model_id="probe",
        label_column="label",
        prediction_column="prediction",
        validation_axis=axis,
    )
    expected = evaluate_model_trial_metrics_from_chunks((frame,), **kwargs)
    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        evaluate_model_trial_metrics_from_chunks(
            (frame.iloc[4:], frame.iloc[:4]), **kwargs
        )
    closed = []

    def chunks():
        try:
            for offset in range(0, len(frame), 2):
                yield frame.iloc[offset : offset + 2]
        finally:
            closed.append(True)

    import weakref

    from quant_research_platform.research.models import trial_selection

    references = []
    original = trial_selection._metric_chunk

    def observed(*args, **kwargs):
        assert sum(ref() is not None for ref in references) <= 2
        values = original(*args, **kwargs)
        references.append(weakref.ref(values))
        return values

    monkeypatch.setattr(trial_selection, "_metric_chunk", observed)
    actual = evaluate_model_trial_metrics_from_chunks(chunks(), **kwargs)
    assert actual == expected and closed == [True]
    assert all(ref() is None for ref in references)
    with pytest.raises(ValueError, match="non_monotonic_time_chunk_order"):
        evaluate_model_trial_metrics_from_chunks(
            (frame.iloc[4:], frame.iloc[:4]), **kwargs
        )


def test_streaming_metric_failure_closes_source():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=2),
            "asset_id": "a",
            "label": [1.0, 2.0],
            "prediction": [2.0, np.inf],
        }
    )
    closed = []

    def chunks():
        try:
            yield frame.iloc[:1]
            yield frame.iloc[1:]
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="invalid_trial_metric_values"):
        evaluate_model_trial_metrics_from_chunks(
            chunks(),
            model_id="probe",
            label_column="label",
            prediction_column="prediction",
            validation_axis="single_asset_time_series",
        )
    assert closed == [True]


def test_completed_trial_keeps_representable_icir_with_unrepresentable_std():
    from quant_research_platform.contracts.models import (
        _validate_model_selection_trial_record,
    )

    tiny = float(np.nextafter(0.0, 1.0))
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
            "asset_id": list("abcd") * 2,
            "label": [0.0, 0.0, 1.0, 1.0, tiny, -tiny, 1.0, 1.0],
            "prediction": [1.0, -1.0, 1.0, -1.0] * 2,
        }
    )
    metrics = evaluate_model_trial_metrics_from_chunks(
        (frame,),
        model_id="candidate",
        label_column="label",
        prediction_column="prediction",
    )
    assert metrics["validation_std_ic"] is None
    assert metrics["validation_std_ic_reason"] == "nonfinite_result"
    assert metrics["validation_icir"] == 1.0
    assert metrics["validation_valid_ic_count"] == 2
    assert metrics["validation_icir_reason"] is None
    _validate_model_selection_trial_record(metrics)
    for changed in (
        {"validation_std_ic": 0.0},
        {"validation_std_ic": -1.0},
        {"validation_std_ic_reason": None},
        {"validation_std_ic_reason": "zero_variance"},
        {"validation_valid_ic_count": 1},
        {"validation_icir": float("inf")},
        {"validation_icir_reason": "zero_variance"},
    ):
        with pytest.raises(ValueError, match="invalid_model_metadata"):
            _validate_model_selection_trial_record(metrics | changed)
