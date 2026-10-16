"""Allocation reductions preserve the original scaler and linear-fit arithmetic."""

import tracemalloc
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import norm
from threadpoolctl import threadpool_limits

from quant_research_platform.contracts.models import TrainingWeighting
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
    _finite_matrix,
    _weights,
    equal_timestamp_sample_weights,
)


# Frozen pre-optimization arithmetic, including exact-offset fallback.
def _reference_fit(
    cls,
    frame: pd.DataFrame,
    *,
    sample_weight: object | None = None,
) -> ModelMatrixScaler:
    columns = tuple(str(column) for column in frame.columns)
    if frame.empty or not columns or frame.columns.has_duplicates:
        raise ValueError("model_scaler_columns_invalid")
    values = _finite_matrix(frame, columns)
    weights = _weights(sample_weight, len(values))
    original_weights = weights
    anchor = values[int(np.argmax(weights))]
    root_weights = np.sqrt(weights)
    root_weights /= root_weights.max()
    weights = weights / weights.max()
    total = float(weights.sum())
    with np.errstate(over="ignore"):
        differences = values - anchor
    shifted = np.isfinite(differences).all(axis=0)
    origin = np.where(shifted, anchor, 0.0)
    differences[:, ~shifted] = values[:, ~shifted]
    magnitude = np.max(np.abs(differences), axis=0)
    constant = magnitude == 0.0
    magnitude[constant] = 1.0
    normalized = differences / magnitude
    weighted = weights[:, None] * normalized
    center = weighted.sum(axis=0) / total
    offset = center * magnitude
    normalized_scale = np.array(
        [
            norm(column, check_finite=False)
            for column in (root_weights[:, None] * (normalized - center)).T
        ]
    ) / np.sqrt(total)
    # Values are in [-1, 1], so the population deviation cannot exceed one.
    scale = np.minimum(normalized_scale, 1.0) * magnitude
    # Preserve original-unit centers across underflow and anchor cancellation.
    exact_columns = ~constant & (
        ((normalized != 0) & (np.abs(weighted) < np.finfo(float).tiny)).any(axis=0)
        | (np.abs(origin + offset) <= np.finfo(float).eps * np.abs(origin))
    )
    exact_offsets = [Fraction(float(value)) for value in offset]
    if exact_columns.any():
        exact_weights = tuple(Fraction(float(weight)) for weight in original_weights)
        exact_total = sum(exact_weights)
        for column in np.flatnonzero(exact_columns):
            exact_mean = (
                sum(
                    weight * Fraction(float(value))
                    for weight, value in zip(
                        exact_weights, values[:, column], strict=True
                    )
                )
                / exact_total
            )
            exact_offset = exact_mean - Fraction(float(origin[column]))
            if abs(exact_mean) < abs(exact_offset):
                origin[column] = 0.0
                exact_offset = exact_mean
            exact_offsets[column] = exact_offset
    scale[constant] = 1.0
    if np.any(scale == 0.0):
        raise ValueError("model_scaler_scale_unrepresentable")
    return cls(
        columns=columns,
        origin_=tuple(float(value) for value in origin),
        offset_=tuple(exact_offsets),
        scale_=tuple(float(value) for value in scale),
    )


def _case(kind):
    if kind == "ordinary":
        frame = pd.DataFrame(np.random.default_rng(17).normal(size=(9, 3)))
    elif kind == "overflow":
        frame = pd.DataFrame({"x": [-1e308, 1e308, 0.0], "y": [1.0, 3.0, 2.0]})
    elif kind == "subnormal":
        tiny = np.nextafter(0.0, 1.0)
        frame = pd.DataFrame({"x": [0.0, tiny * 8, tiny * 16], "y": [0.0, 0.0, 1.0]})
    elif kind == "cancellation":
        frame = pd.DataFrame({"x": [1.0, -1.0, 0.0], "y": [1e300, -1e300, 1.0]})
    elif kind == "mixed_extremes":
        maximum, tiny = np.finfo(float).max, np.nextafter(0.0, 1.0)
        frame = pd.DataFrame(
            {
                "x": [-maximum, maximum, 0.0],
                "y": [maximum, -maximum, 1.0],
                "constant": [3.0, 3.0, 3.0],
                "subnormal": [0.0, tiny * 8, tiny * 16],
            }
        )
    else:
        frame = pd.DataFrame({"x": [1.0, 0.0, 0.0], "y": [3.0, 3.0, 3.0]})
    frame.columns = [str(column) for column in frame.columns]
    return frame


@pytest.mark.parametrize(
    "kind",
    ("ordinary", "overflow", "subnormal", "fraction", "cancellation", "mixed_extremes"),
)
@pytest.mark.parametrize(
    "weighting", ("uniform", "equal_timestamp", "decayed", "extreme")
)
def test_scaler_and_linear_fit_match_original_arithmetic(kind, weighting, monkeypatch):
    frame = _case(kind)
    if weighting == "uniform":
        weights = np.ones(len(frame))
    elif weighting in {"equal_timestamp", "decayed"}:
        timestamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            np.arange(len(frame)) // 2, unit="D"
        )
        weights = equal_timestamp_sample_weights(
            timestamps,
            TrainingWeighting(half_life_periods=2) if weighting == "decayed" else None,
        )
    else:
        weights = np.geomspace(np.nextafter(0.0, 1.0), 1.0, len(frame))
    original_frame, original_weights = frame.copy(deep=True), weights.copy()
    try:
        expected_scaler = _reference_fit(
            ModelMatrixScaler, frame, sample_weight=weights
        )
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error)):
            ModelMatrixScaler.fit(frame, sample_weight=weights)
        pd.testing.assert_frame_equal(frame, original_frame, check_exact=True)
        np.testing.assert_array_equal(weights, original_weights)
        return
    actual_scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    assert actual_scaler.columns == expected_scaler.columns
    matrix = actual_scaler.transform(frame)
    pd.testing.assert_frame_equal(
        matrix,
        expected_scaler.transform(frame),
        check_exact=False,
        rtol=1e-13,
        atol=1e-14,
    )
    before_matrix = matrix.copy(deep=True)
    target = np.linspace(-1e150, 1e150, len(frame))
    training = frame.assign(label=target)
    before_training = training.copy(deep=True)
    prepared = SimpleNamespace(
        training=training,
        row_counts={},
        filter_metadata={},
        preprocessing=(weights, actual_scaler, matrix),
    )
    spec = SimpleNamespace(target_column="label", input_columns=tuple(frame.columns))
    monkeypatch.setattr(fitting, "model_metadata", lambda *args, **kwargs: {})
    design = np.column_stack((np.ones(len(frame)), matrix.to_numpy(dtype="float64")))
    _, exponent = np.frexp(float(np.max(np.abs(target))))
    with threadpool_limits(limits=1, user_api="blas"):
        expected_beta = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            np.ldexp(target, -exponent) * np.sqrt(weights),
            rcond=None,
        )[0]
    expected_beta = np.ldexp(expected_beta, exponent)
    result = fitting._fit_linear_model_result(prepared, spec).prediction_model
    np.testing.assert_allclose(result.coefficients, expected_beta, rtol=1e-12, atol=0)
    reference = LinearScaledPredictor(tuple(expected_beta), expected_scaler)
    np.testing.assert_allclose(
        result.predict(frame), reference.predict(frame), rtol=1e-12, atol=0
    )
    pd.testing.assert_frame_equal(frame, original_frame, check_exact=True)
    pd.testing.assert_frame_equal(matrix, before_matrix, check_exact=True)
    pd.testing.assert_frame_equal(training, before_training, check_exact=True)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize(
    "shape", ((1, 1), (1, 5), (2, 1), (7, 17), (257, 17), (1025, 3))
)
@pytest.mark.parametrize("layout", ("C", "F", "mixed"))
def test_scaler_shapes_and_layouts_match_original_arithmetic(shape, layout):
    values = np.random.default_rng(59).normal(size=shape)
    if layout == "mixed":
        frame = pd.DataFrame(
            {
                str(column): values[:, column].astype("float32" if column % 2 else int)
                for column in range(shape[1])
            }
        )
    else:
        frame = pd.DataFrame(np.array(values, order=layout))
        frame.columns = frame.columns.astype(str)
    frame.index = pd.Index(np.arange(len(frame))[::-1], name="row")
    original = frame.copy(deep=True)
    weights = np.linspace(0.5, 3.0, len(frame))
    weights.flags.writeable = False
    for sample_weight in (None, weights):
        expected = _reference_fit(ModelMatrixScaler, frame, sample_weight=sample_weight)
        actual = ModelMatrixScaler.fit(frame, sample_weight=sample_weight)
        assert actual.columns == expected.columns
        pd.testing.assert_frame_equal(
            actual.transform(frame),
            expected.transform(frame),
            check_exact=False,
            rtol=1e-13,
            atol=1e-14,
        )
        coefficients = tuple(np.linspace(-3.0, 5.0, shape[1] + 1))
        np.testing.assert_allclose(
            LinearScaledPredictor(coefficients, actual).predict(frame),
            LinearScaledPredictor(coefficients, expected).predict(frame),
            rtol=1e-12,
            atol=0,
        )
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    np.testing.assert_array_equal(weights, np.linspace(0.5, 3.0, len(frame)))


def test_scaler_keeps_root_weights_before_extreme_weight_normalization():
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [1.0, 0.0]})
    weights = np.array([1e308, 1e-300])
    original, original_weights = frame.copy(deep=True), weights.copy()
    expected = _reference_fit(ModelMatrixScaler, frame, sample_weight=weights)
    actual = ModelMatrixScaler.fit(frame, sample_weight=weights)
    assert actual.columns == expected.columns
    assert 0.0 < actual.scale_[0] < np.finfo(float).tiny ** 0.5
    pd.testing.assert_frame_equal(
        actual.transform(frame),
        expected.transform(frame),
        check_exact=False,
        rtol=1e-13,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        LinearScaledPredictor((1.0, 2.0, -3.0), actual).predict(frame),
        LinearScaledPredictor((1.0, 2.0, -3.0), expected).predict(frame),
        rtol=1e-12,
        atol=0,
    )
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    np.testing.assert_array_equal(weights, original_weights)


def test_scaler_peak_scratch_reduction_is_bounded_by_column():
    frame = pd.DataFrame(np.random.default_rng(29).normal(size=(1024, 32)))
    frame.columns = frame.columns.astype(str)
    weights = np.ones(len(frame))
    _reference_fit(ModelMatrixScaler, frame, sample_weight=weights)
    ModelMatrixScaler.fit(frame, sample_weight=weights)

    def peak(function):
        tracemalloc.start()
        try:
            function(frame, sample_weight=weights)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    original = peak(
        lambda *args, **kwargs: _reference_fit(ModelMatrixScaler, *args, **kwargs)
    )
    optimized = peak(ModelMatrixScaler.fit)
    # A small 256-KiB matrix suffices; require only half a matrix of savings.
    assert original - optimized > frame.size * 4
    # Column-local norms alone still needed over five matrices. Reuse and release
    # fit buffers so only three dense matrices plus column-local scratch coexist.
    assert optimized < frame.size * 8 * 3.5


def test_unrepresentable_population_scale_remains_an_explicit_failure():
    frame = pd.DataFrame({"x": [0.0, np.nextafter(0.0, 1.0)]})
    for fit in (
        lambda frame: _reference_fit(ModelMatrixScaler, frame),
        ModelMatrixScaler.fit,
    ):
        with pytest.raises(ValueError, match="model_scaler_scale_unrepresentable"):
            fit(frame)


def test_linear_lstsq_receives_the_fresh_design_without_an_extra_copy(monkeypatch):
    frame = _case("ordinary")
    weights = np.arange(1.0, len(frame) + 1.0)
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    matrix = scaler.transform(frame)
    prepared = SimpleNamespace(
        training=frame.assign(label=np.arange(len(frame), dtype=float)),
        row_counts={},
        filter_metadata={},
        preprocessing=(weights, scaler, matrix),
    )
    spec = SimpleNamespace(target_column="label", input_columns=tuple(frame.columns))
    monkeypatch.setattr(fitting, "model_metadata", lambda *args, **kwargs: {})
    captured = []
    original_stack, original_lstsq = np.column_stack, np.linalg.lstsq

    def stack(*args, **kwargs):
        result = original_stack(*args, **kwargs)
        captured.append(result)
        assert not np.shares_memory(result, matrix.to_numpy())
        return result

    def lstsq(design, *args, **kwargs):
        assert design is captured[-1]
        return original_lstsq(design, *args, **kwargs)

    monkeypatch.setattr(np, "column_stack", stack)
    monkeypatch.setattr(np.linalg, "lstsq", lstsq)
    fitting._fit_linear_model_result(prepared, spec)
    assert len(captured) == 1
