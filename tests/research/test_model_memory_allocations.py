"""Ordinary weighted fitting preserves source inputs and reuses work buffers."""

import tracemalloc
import weakref
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from quant_research_platform.contracts.models import TrainingWeighting
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
    equal_timestamp_sample_weights,
)


def _reference_fit(cls, frame, *, sample_weight=None):
    values = frame.to_numpy(dtype="float64")
    mean = np.average(values, axis=0, weights=sample_weight)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=sample_weight))
    constant = np.all(values == values[0], axis=0)
    mean[constant], scale[constant] = values[0, constant], 1.0
    return cls(tuple(frame.columns), tuple(mean), tuple(scale))


def _case(kind):
    if kind == "ordinary":
        frame = pd.DataFrame(np.random.default_rng(17).normal(size=(9, 3)))
    else:
        frame = pd.DataFrame({"x": [1.0, 0.0, 0.0], "y": [3.0, 3.0, 3.0]})
    frame.columns = [str(column) for column in frame.columns]
    return frame


@pytest.mark.parametrize(
    "kind",
    ("ordinary", "fraction"),
)
@pytest.mark.parametrize("weighting", ("uniform", "equal_timestamp", "decayed"))
def test_scaler_and_linear_fit_match_weighted_oracle(kind, weighting, monkeypatch):
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
            trading_sessions=pd.date_range(timestamps.min(), timestamps.max()),
        )
    original_frame, original_weights = frame.copy(deep=True), weights.copy()
    expected_scaler = _reference_fit(ModelMatrixScaler, frame, sample_weight=weights)
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
    target = np.linspace(-0.05, 0.05, len(frame))
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
    with threadpool_limits(limits=1, user_api="blas"):
        expected_beta = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            target * np.sqrt(weights),
            rcond=None,
        )[0]
    result = fitting._fit_linear_model_result(prepared, spec).prediction_model
    np.testing.assert_allclose(
        result.coefficients, expected_beta, rtol=1e-12, atol=1e-14
    )
    reference = LinearScaledPredictor(tuple(expected_beta), expected_scaler)
    np.testing.assert_allclose(
        result.predict(frame), reference.predict(frame), rtol=1e-12, atol=1e-14
    )
    pd.testing.assert_frame_equal(frame, original_frame, check_exact=True)
    pd.testing.assert_frame_equal(matrix, before_matrix, check_exact=True)
    pd.testing.assert_frame_equal(training, before_training, check_exact=True)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize(
    "shape", ((1, 1), (1, 5), (2, 1), (7, 17), (257, 17), (1025, 3))
)
@pytest.mark.parametrize("layout", ("C", "F", "mixed"))
def test_scaler_shapes_and_layouts_match_weighted_oracle(shape, layout):
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
            atol=1e-13,
        )
    pd.testing.assert_frame_equal(frame, original, check_exact=True)
    np.testing.assert_array_equal(weights, np.linspace(0.5, 3.0, len(frame)))


def test_scaler_peak_scratch_reduction_is_bounded_by_column():
    frame = pd.DataFrame(np.random.default_rng(29).normal(size=(1024, 32)))
    frame.columns = frame.columns.astype(str)
    weights = np.ones(len(frame))
    ModelMatrixScaler.fit(frame, sample_weight=weights)

    def peak(function):
        tracemalloc.start()
        try:
            function(frame, sample_weight=weights)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    optimized = peak(ModelMatrixScaler.fit)
    # Keep one owned input matrix and reuse the centered/squared work buffer.
    assert optimized < frame.size * 8 * 3.5


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


@pytest.mark.parametrize("chunk_size", [1, 2, 6])
def test_collected_training_releases_chunks_and_preserves_source(
    chunk_size, monkeypatch
):
    from tests.research.test_model_missingness_contract import frame, spec

    source = frame().astype({"x": "Float64", "z": "Float64"})
    before = source.copy(deep=True)
    references = []
    original_concat = pd.concat

    def concat(objects, *args, **kwargs):
        if isinstance(objects, list):
            references.extend(weakref.ref(chunk) for chunk in objects)
        return original_concat(objects, *args, **kwargs)

    monkeypatch.setattr(pd, "concat", concat)
    prepared = fitting._collect_training_frame(
        (
            source.iloc[start : start + chunk_size]
            for start in range(0, len(source), chunk_size)
        ),
        spec(),
        fitting_cutoff="2024-01-03",
    )
    assert all(reference() is None for reference in references)
    expected = (
        source[["timestamp", "asset_id", "x", "z", "label"]]
        .astype({"timestamp": "datetime64[ns]"})
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(prepared.training, expected, check_exact=True)
    assert prepared.row_counts["rows_input"] == len(source)
    assert prepared.row_counts["rows_after_input_label_dropna"] == len(expected)
    prepared.training.loc[0, "x"] = -999.0
    pd.testing.assert_frame_equal(source, before, check_exact=True)
