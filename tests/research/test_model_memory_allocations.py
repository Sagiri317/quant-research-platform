"""Ordinary weighted fitting preserves source inputs and reuses work buffers."""

import tracemalloc
import weakref
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
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
        preprocessing=(weights, actual_scaler),
        matrix=matrix,
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
    "shape,layout",
    [
        ((1, 1), "C"),
        ((1, 5), "mixed"),
        ((2, 1), "F"),
        ((7, 17), "C"),
        ((7, 17), "F"),
        ((7, 17), "mixed"),
        ((257, 17), "F"),
        ((1025, 3), "C"),
        ((1025, 3), "mixed"),
    ],
)
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
    # Fitting keeps column-sized work arrays, independent of feature width.
    assert optimized < frame.size * 8 * 0.8


@pytest.mark.parametrize("operation", ["fit", "transform"])
def test_nullable_imputation_bounds_scratch_and_preserves_values(operation):
    frame = pd.DataFrame(
        np.random.default_rng(29).normal(size=(8192, 32)), dtype="Float64"
    )
    frame.columns = frame.columns.astype(str)
    for index, column in enumerate(frame):
        frame.loc[frame.index % 37 == index, column] = pd.NA
    scaler = ModelMatrixScaler.fit(frame, missing="impute")
    tracemalloc.start()
    try:
        if operation == "fit":
            ModelMatrixScaler.fit(frame, missing="impute")
        else:
            scaler.transform(frame)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Nullable conversion must release each column before collecting the next.
    assert peak < frame.size * 8 * 1.6


@pytest.mark.parametrize(
    "algorithm,parameters",
    [("linear_regression", {})]
    + [
        (
            "lasso_regression",
            {"alpha": 0.01, "copy_X": copy, "max_iter": None, "precompute": True},
        )
        for copy in (True, False)
    ]
    + [
        ("ridge_regression", {"alpha": 0.1, "copy_X": False}),
        ("xgboost_regressor", {"device": "cpu", "n_estimators": 2, "max_depth": 1}),
    ],
)
def test_tabular_fit_uses_mapped_features_without_a_second_raw_table(
    algorithm, parameters
):
    columns = tuple(f"x{index}" for index in range(335))
    frame = pd.DataFrame(
        np.random.default_rng(29).normal(size=(4096, len(columns))), columns=columns
    )
    for column in columns[-7:]:
        frame[column] = frame[column].astype("Float64")
        frame.loc[frame.index % 17 == 0, column] = pd.NA
    frame["timestamp"] = pd.date_range("2024-01-01", periods=len(frame), freq="s")
    frame["asset_id"] = "a"
    frame["label"] = 2 * frame.x0 + 0.5 * frame.x1
    frame["eligible"] = True
    frame["label_available_at"] = frame.timestamp + pd.Timedelta(seconds=1)
    before = frame.copy(deep=True)
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
        input_feature_columns=columns,
        target_column="label",
        hyperparameters=parameters,
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2024-01-02"
    )
    expected = prepared.training.copy(deep=True)
    assert not set(columns).intersection(prepared.training.columns)
    assert isinstance(prepared.feature_mapping, np.memmap)
    assert prepared.features.dtypes.astype(str).eq("double[pyarrow]").all()
    np.testing.assert_allclose(
        prepared.features.iloc[prepared.training_rows].to_numpy(dtype="float64"),
        frame.loc[:, columns].to_numpy(dtype="float64"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    estimator = fitting.estimators.build_estimator_from_model_spec(spec)
    result = fitting._fit_model_result(prepared, spec, estimator=estimator)
    pd.testing.assert_frame_equal(prepared.training, expected, check_exact=True)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)
    assert result.metadata["rows_after_input_label_dropna"] == len(frame)
    if algorithm in {"lasso_regression", "xgboost_regressor"}:
        assert "matrix" not in prepared.__dict__
    prepared.close()
    assert prepared.feature_file.closed


def test_linear_lstsq_receives_the_fresh_design_without_an_extra_copy(monkeypatch):
    frame = _case("ordinary")
    weights = np.arange(1.0, len(frame) + 1.0)
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    matrix = scaler.transform(frame)
    prepared = SimpleNamespace(
        training=frame.assign(label=np.arange(len(frame), dtype=float)),
        row_counts={},
        filter_metadata={},
        preprocessing=(weights, scaler),
        matrix=matrix,
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
    pd.testing.assert_frame_equal(
        prepared.training, expected.drop(columns=["x", "z"]), check_exact=True
    )
    np.testing.assert_allclose(
        prepared.features.iloc[prepared.training_rows].to_numpy(dtype="float64"),
        expected[["x", "z"]].to_numpy(dtype="float64"),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert prepared.row_counts["rows_input"] == len(source)
    assert prepared.row_counts["rows_after_input_label_dropna"] == len(expected)
    prepared.close()
    pd.testing.assert_frame_equal(source, before, check_exact=True)


@pytest.mark.parametrize("missing", [None, "impute", "native"])
def test_scaler_parses_each_source_column_once(missing, monkeypatch):
    import quant_research_platform.research.models.scaler as scaling

    frame = _case("ordinary")
    if missing is not None:
        frame.loc[0, "0"] = np.nan
    calls = []
    parse = scaling._finite_numeric_series

    def counted(values, **kwargs):
        calls.append(values.name)
        return parse(values, **kwargs)

    monkeypatch.setattr(scaling, "_finite_numeric_series", counted)
    scaler = ModelMatrixScaler.fit(frame, missing=missing, scaling=True)
    assert calls == list(frame.columns)
    calls.clear()
    scaler.transform(frame)
    assert calls == list(frame.columns)


def test_sequence_fit_only_materializes_its_context_matrix(monkeypatch):
    timestamps = pd.date_range("2024-01-01", periods=4).repeat(2)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["a", "b"] * 4,
            "x": np.arange(8.0),
            "label": np.arange(8.0) ** 2,
            "eligible": [True, False] + [True] * 6,
            "label_available_at": timestamps,
        }
    )
    spec = ModelSpec(
        model_id="sequence",
        model_algorithm="lstm_regressor",
        input_factor_columns=("x",),
        target_column="label",
        hyperparameters={
            "device": "cpu",
            "hidden_size": 2,
            "epochs": 1,
            "sequence_length": 2,
        },
    )
    prepared = fitting._collect_training_frame(
        [frame.iloc[4:], frame.iloc[:4]], spec, fitting_cutoff="2024-01-04"
    )
    assert "x" not in prepared.training
    assert len(prepared.sequence_context) == len(frame)
    assert prepared.preprocessing[1].mean_[0] == pytest.approx(
        frame.loc[frame.eligible].groupby("timestamp").x.mean().mean()
    )
    rows = []
    transform = ModelMatrixScaler._transform_numeric

    def counted(self, inputs, **kwargs):
        rows.append(len(inputs))
        return transform(self, inputs, **kwargs)

    monkeypatch.setattr(ModelMatrixScaler, "_transform_numeric", counted)
    result = fitting._fit_model_result(prepared, spec, estimator=None)
    assert rows == [len(frame)]
    assert "matrix" not in prepared.__dict__
    assert result.metadata["training_target_mean"] == pytest.approx(
        frame.loc[frame.eligible].groupby("timestamp").label.mean().mean()
    )
    future = frame.iloc[:4].copy()
    future["timestamp"] += pd.Timedelta(days=4)
    expected = result.prediction_model.predict(future)
    prepared.close()
    np.testing.assert_array_equal(result.prediction_model.predict(future), expected)


def test_sequence_matrix_float32_conversion_preserves_float64_preprocessing():
    from quant_research_platform.research.models.neural import _matrix

    rng = np.random.default_rng(19)
    frame = pd.DataFrame(rng.normal(size=(16401, 7)), columns=list("abcdefg"))
    frame["a"] += 1e9
    frame.loc[frame.index % 17 == 0, "b"] = np.nan
    frame.loc[16380:16390, :] = np.nan
    valid = frame.notna().any(axis=1).to_numpy()
    scaler = ModelMatrixScaler.fit(frame.loc[valid], missing="impute")
    expected = np.zeros(frame.shape, dtype="float32")
    expected[valid] = scaler.transform(frame.loc[valid]).to_numpy(dtype="float32")
    original = frame.copy(deep=True)
    actual = _matrix(scaler, frame, tuple(frame.columns), valid)
    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("grouped", [False, True])
def test_sequence_arrays_preserve_history_and_gaps_across_source_batches(grouped):
    from quant_research_platform.research.models.neural import _sequence_arrays

    dates = pd.bdate_range("2014-01-02", periods=2000)
    count = len(dates) * 9
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(9),
            "asset_id": np.tile(list("abcdefghi"), len(dates)),
            "x": np.arange(count, dtype="float64") + 0.25,
            "y": np.arange(count, dtype="float64") % 17,
            "regime": np.repeat(np.arange(len(dates)) >= 1000, 9),
            "label": np.arange(count) % 5 / 100,
            "eligible": np.arange(count) % 17 != 0,
            "label_available_at": dates.repeat(9),
        }
    )
    frame.loc[frame.index % 11 == 0, "y"] = np.nan
    frame.loc[[90, 17000], ["x", "y"]] = np.nan
    history = frame.iloc[:9].copy()
    history["timestamp"] = dates[0] - pd.offsets.BDay()
    history[["x", "y"]] = [-1.0, 2.0]
    spec = ModelSpec(
        model_id="sequence",
        model_algorithm="lstm_regressor",
        input_feature_columns=("x", "y"),
        target_column="label",
        hyperparameters={"sequence_group_columns": ["regime"] if grouped else []},
    )
    scaler = ModelMatrixScaler(("x", "y"), (2.0, -1.0), (4.0, 2.0), (3.0, 5.0))
    expected = pd.concat([history, frame], ignore_index=True)
    if not grouped:
        expected = expected.loc[expected[["x", "y"]].notna().any(axis=1)]
    expected = expected.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    valid = expected[["x", "y"]].notna().any(axis=1).to_numpy()
    values = expected[["x", "y"]].fillna({"x": 3.0, "y": 5.0}).to_numpy()
    values = ((values - [2.0, -1.0]) / [4.0, 2.0]).astype("float32")
    values[~valid] = 0
    lookup = {(row.asset_id, row.timestamp): i for i, row in expected.iterrows()}
    starts = []
    previous, previous_valid, start = None, False, 0
    for i, row in expected.iterrows():
        key = (row.asset_id, row.regime) if grouped else (row.asset_id,)
        if key != previous or grouped and (not valid[i] or not previous_valid):
            start = i
        starts.append(start)
        previous, previous_valid = key, valid[i]
    with closing(
        fitting._collect_training_frame(
            [frame.iloc[start : start + 1024] for start in range(0, count, 1024)],
            spec,
            fitting_cutoff=dates[-1],
        )
    ) as prepared:
        context, matrix, endpoints, group_start = _sequence_arrays(
            prepared, spec, scaler, history
        )
        pd.testing.assert_frame_equal(
            context[["asset_id", "timestamp"]].reset_index(drop=True),
            expected[["asset_id", "timestamp"]],
        )
        np.testing.assert_array_equal(matrix, values)
        np.testing.assert_array_equal(group_start, starts)
        np.testing.assert_array_equal(
            endpoints,
            [
                lookup[(row.asset_id, row.timestamp)]
                for row in prepared.training.itertuples()
            ],
        )
