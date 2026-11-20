"""Fit supported predictors on legal samples."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from itertools import count
from tempfile import TemporaryFile

import numpy as np
import pandas as pd
import pyarrow as pa

import quant_research_platform.research.models.estimators as estimators
import quant_research_platform.research.models.feature_importance as feature_importance
from quant_research_platform.contracts.experiments import (
    label_maturity_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import (
    DEEPLOB_ALGORITHMS,
    SEQUENCE_ALGORITHMS,
    ModelSpec,
    TrainingWeighting,
    _thaw_model_hyperparameters,
    sequence_context_columns,
)
from quant_research_platform.contracts.time import (
    prediction_window_mask,
    time_value_isoformat,
)
from quant_research_platform.contracts.validation import (
    _finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
    ScaledPredictor,
    equal_timestamp_sample_weights,
)


@dataclass
class ModelFitResult:
    metadata: dict[str, object]
    prediction_model: object = field(repr=False, compare=False)


@dataclass
class PreparedTraining:
    """Mapped numeric inputs and narrow metadata for legal loss rows."""

    training: pd.DataFrame
    row_counts: dict[str, object]
    filter_metadata: dict[str, object]
    input_columns: tuple[str, ...]
    training_weighting: TrainingWeighting
    target_column: str
    features: pd.DataFrame
    feature_file: object
    feature_mapping: np.memmap | None
    training_rows: np.ndarray
    native_missing: bool = False
    trading_sessions: object = None
    training_scaling: bool = True
    sequence_context: pd.DataFrame | None = None
    complete_inputs: bool = False
    context_valid: np.ndarray | None = None

    def close(self):
        self.features = None
        if self.feature_mapping is not None:
            self.feature_mapping._mmap.close()
            self.feature_mapping = None
        self.feature_file.close()

    @cached_property
    def validation_weights(self):
        return equal_timestamp_sample_weights(self.training.timestamp)

    @cached_property
    def preprocessing(self):
        weights = equal_timestamp_sample_weights(
            self.training["timestamp"],
            self.training_weighting,
            trading_sessions=self.trading_sessions,
        )
        scaler = (
            ModelMatrixScaler(
                self.input_columns,
                (0.0,) * len(self.input_columns),
                (1.0,) * len(self.input_columns),
                native_missing=True,
                scaling=False,
            )
            if self.native_missing and not self.training_scaling
            else ModelMatrixScaler.fit(
                self.features,
                sample_weight=weights,
                missing=None
                if self.complete_inputs
                else "native"
                if self.native_missing
                else "impute",
                scaling=self.training_scaling,
                rows=self.training_rows,
            )
        )
        return weights, scaler

    @cached_property
    def matrix(self):
        values = np.empty((len(self.training), len(self.input_columns)), order="F")
        for start, stop, block in self.blocks(self.preprocessing[1]):
            values[start:stop] = block
        return pd.DataFrame(values, columns=self.input_columns, copy=False)

    def blocks(self, scaler, *, dtype="float64"):
        for start in range(0, len(self.training), 16384):
            stop = min(start + 16384, len(self.training))
            values = self.feature_rows(self.training_rows[start:stop]).to_numpy(
                dtype="float64", copy=True
            )
            values = scaler._transform_numeric(values, dtype=dtype)
            yield start, stop, values

    def feature_rows(self, rows):
        # Slice before Arrow take so a small batch does not combine the whole column.
        if not len(rows):
            return self.features.iloc[:0]
        lower, upper = int(np.min(rows)), int(np.max(rows)) + 1
        return self.features.iloc[lower:upper].iloc[rows - lower]

    @cached_property
    def lasso_gram(self):
        """Reuse weighted sufficient statistics across precomputed Lasso candidates."""
        weights, scaler = self.preprocessing
        weights = weights * (len(weights) / weights.sum())
        target = self.training[self.target_column].to_numpy(dtype="float64", copy=True)
        target_mean = float(np.average(target, weights=weights))
        root_weight = np.sqrt(weights)
        centered_target = (target - target_mean) * root_weight
        width = len(self.input_columns)
        mean = np.zeros(width)

        for start, stop, values in self.blocks(scaler):
            mean += weights[start:stop] @ values
        mean /= weights.sum()
        gram = np.zeros((width, width))
        cross = np.zeros(width)
        for start, stop, values in self.blocks(scaler):
            values -= mean
            values *= root_weight[start:stop, None]
            gram += values.T @ values
            cross += values.T @ centered_target[start:stop]
        return gram, cross, mean, centered_target, target_mean, root_weight


def prediction_callable(input_columns: tuple[str, ...], prediction_model: object):
    """Build the model-owned prediction callable used by selection and output."""

    from quant_research_platform.research.models.neural import SequencePredictor

    if isinstance(prediction_model, SequencePredictor):
        return prediction_model.prediction_function()

    def predict_from_model(frame: pd.DataFrame) -> np.ndarray:
        predicted = prediction_model.predict(frame.loc[:, list(input_columns)])
        return np.asarray(predicted, dtype="float64")

    return predict_from_model


def _fit_model_result(
    prepared_training,
    spec: ModelSpec,
    *,
    estimator: object,
    validation=None,
    fit_inputs=None,
) -> ModelFitResult:
    if spec.model_algorithm == "linear_regression":
        result = _fit_linear_model_result(prepared_training, spec)
    elif spec.model_algorithm in SEQUENCE_ALGORITHMS:
        from quant_research_platform.research.models.neural import fit_sequence

        predictor, curve, stopping = fit_sequence(
            prepared_training, spec, validation, fit_inputs=fit_inputs
        )
        result = ModelFitResult(
            metadata=model_metadata(
                spec,
                filter_metadata=prepared_training.filter_metadata,
                row_counts=prepared_training.row_counts,
                scaler=predictor.scaler,
            ),
            prediction_model=predictor,
        )
        result.metadata["learning_curve"] = curve
        if stopping is not None:
            result.metadata["early_stopping"] = stopping
    else:
        result = _fit_estimator_model_result(
            prepared_training,
            spec,
            estimator=estimator,
            validation=validation,
            fit_inputs=fit_inputs,
        )
    target = prepared_training.training[spec.target_column].to_numpy(dtype="float64")
    result.metadata["training_target_mean"] = float(
        np.average(target, weights=prepared_training.preprocessing[0])
    )
    if (
        "early_stopping" in result.metadata
        and spec.model_algorithm != "deeplob_classifier"
    ):
        observed = validation.training
        errors = (
            observed[spec.target_column].to_numpy(dtype="float64")
            - result.metadata["training_target_mean"]
        )
        result.metadata["early_stopping"]["validation_mean_baseline_mse"] = float(
            np.average(errors**2, weights=validation.validation_weights)
        )
    return result


def _fit_linear_model_result(
    prepared_training,
    spec: ModelSpec,
) -> ModelFitResult:
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if prepared_training.training.empty:
        raise ValueError("model_training_sample_empty")
    weights, scaler = prepared_training.preprocessing
    transformed = prepared_training.matrix.to_numpy(dtype="float64")
    design = np.column_stack((np.ones(len(transformed)), transformed))
    weighted = np.sqrt(weights)
    target = prepared_training.training[spec.target_column].to_numpy(dtype="float64")
    design *= weighted[:, None]
    beta = np.linalg.lstsq(
        design,
        target * weighted,
        rcond=None,
    )[0]
    predictor = LinearScaledPredictor(
        coefficients=tuple(float(value) for value in beta), scaler=scaler
    )
    metadata = model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        feature_importance=feature_importance.linear_feature_importance(spec, beta),
        scaler=scaler,
    )
    result = ModelFitResult(metadata=metadata, prediction_model=predictor)
    return result


def model_metadata(
    spec: ModelSpec,
    *,
    filter_metadata: dict[str, object],
    row_counts: dict[str, object],
    feature_importance: dict[str, float] | None = None,
    feature_importance_status: str | None = None,
    resolved_hyperparameters: dict[str, object] | None = None,
    scaler: ModelMatrixScaler | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "model_algorithm": spec.model_algorithm,
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "rows_input": filter_metadata["rows_input"],
        "rows_after_filter_mode": filter_metadata["rows_after_filter_mode"],
        "rows_after_input_label_dropna": row_counts["rows_after_input_label_dropna"],
        "training_asset_ids": row_counts.get("training_asset_ids", []),
        "train_start": row_counts["train_start"],
        "train_end": row_counts["train_end"],
        "label_availability_column": row_counts["label_availability_column"],
        "label_available_max": row_counts["label_available_max"],
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "input_columns": list(spec.input_columns),
        "input_count": len(spec.input_columns),
        "target_column": spec.target_column,
        "hyperparameters": _thaw_model_hyperparameters(spec.hyperparameters),
        "resolved_hyperparameters": estimators.resolved_model_hyperparameters(spec)
        if resolved_hyperparameters is None
        else _thaw_model_hyperparameters(resolved_hyperparameters),
        "random_seed": spec.random_seed,
        "dependency_status": estimators.require_model_dependency(spec.model_algorithm),
        "training_weighting": spec.training_weighting.payload(),
        "training_scaling": spec.training_scaling,
        "scaler": None if scaler is None else scaler.to_payload(),
    }
    if feature_importance is not None:
        metadata["feature_importance"] = feature_importance
    else:
        metadata["feature_importance_status"] = (
            feature_importance_status or "not_available_for_model_algorithm"
        )
    return metadata


def _fit_estimator_model_result(
    prepared_training,
    spec: ModelSpec,
    *,
    estimator: object,
    validation=None,
    fit_inputs=None,
) -> ModelFitResult:
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if prepared_training.training.empty:
        raise ValueError("model_training_sample_empty")
    predictor = _fit_predictor(
        spec,
        prepared_training=prepared_training,
        estimator=estimator,
        validation=validation,
        fit_inputs=fit_inputs,
    )

    resolved_hyperparameters = estimators.resolved_model_hyperparameters(spec)
    metadata = model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        feature_importance=feature_importance.estimator_feature_importance(
            spec, predictor
        ),
        resolved_hyperparameters=resolved_hyperparameters,
        scaler=predictor.scaler if isinstance(predictor, ScaledPredictor) else None,
    )
    if spec.hyperparameters.get("early_stopping_rounds") is not None:
        metrics = estimator.evals_result()["validation_0"]
        metadata["early_stopping"] = {
            "best_iteration": int(estimator.best_iteration),
            "iterations": len(next(iter(metrics.values()))),
            "validation_metrics": metrics,
            "validation_weighting": "equal_timestamp",
        }
    result = ModelFitResult(metadata=metadata, prediction_model=predictor)
    return result


def _legal_training_rows(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    fitting_cutoff: object,
    availability_column: str,
):
    """Select all legal rows without learning preprocessing."""
    keep = label_maturity_mask(
        frame,
        fitting_cutoff=fitting_cutoff,
        availability_column=availability_column,
        target_column=spec.target_column,
    )
    keep &= frame.loc[:, spec.input_columns].notna().any(axis=1)
    filtered, metadata = apply_filter_mode(
        frame, mode=spec.sample_filter_mode, metadata_key="sample_filter_mode"
    )
    return filtered.loc[keep.loc[filtered.index]], metadata


def _collect_training_frame(frames, spec, **kwargs):
    # Coalesce wide IPC column buffers before writing to the filesystem.
    feature_file = TemporaryFile(buffering=8 * 1024 * 1024)
    try:
        with pa.ipc.new_file(
            feature_file,
            pa.schema((column, pa.float64()) for column in spec.input_columns),
        ) as writer:
            return _collect_training_rows(
                frames, spec, feature_file=feature_file, writer=writer, **kwargs
            )
    except BaseException:
        feature_file.close()
        raise


def _write_features(frame, columns, destination):
    for start in range(0, len(frame), 16384):
        block = frame.iloc[start : start + 16384]
        arrays = []
        for column in columns:
            values = block[column].to_numpy(dtype="float64")
            arrays.append(pa.array(values, type=pa.float64(), from_pandas=True))
        destination.write_batch(pa.RecordBatch.from_arrays(arrays, names=columns))


def _collect_training_rows(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    feature_file,
    writer,
    fitting_cutoff: object,
    availability_column: str | None = None,
    trading_calendar=None,
    sample_windows=None,
) -> PreparedTraining:
    required = set(spec.input_columns)
    required.add(spec.target_column)
    required.add("timestamp")
    required.update(required_filter_columns(spec.sample_filter_mode))
    availability_columns = (
        (availability_column,)
        if availability_column is not None
        else training_availability_columns(spec.target_column)
    )
    context_columns = sequence_context_columns(
        spec.model_algorithm, spec.hyperparameters
    )
    if set(context_columns).intersection(availability_columns):
        raise ValueError("sequence_context_uses_future_outcome")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *(
                    ("market_type",)
                    if spec.training_weighting.half_life_periods is not None
                    else ()
                ),
                *spec.input_columns,
                *context_columns,
                spec.target_column,
                *availability_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    retained: list[pd.DataFrame] = []
    sequence_context = []
    context_valid = []
    context_size = 0
    with closing_iterator(
        iter(
            iter_model_work_frames(frames, required=required, work_columns=work_columns)
        )
    ) as work_frames:
        for chunk in work_frames:
            chunk = chunk.reset_index(drop=True)
            for column in spec.input_columns:
                chunk[column] = _finite_numeric_series(
                    chunk[column],
                    error=f"invalid_training_value: {column}",
                    allow_null=True,
                )
            if spec.model_algorithm in SEQUENCE_ALGORITHMS:
                observed = (
                    pd.Series(True, index=chunk.index)
                    if context_columns
                    else chunk.loc[:, spec.input_columns].notna().any(axis=1)
                )
                observed &= pd.to_datetime(chunk.timestamp).le(
                    pd.Timestamp(fitting_cutoff)
                )
                context_positions = pd.Series(
                    np.arange(context_size, context_size + int(observed.sum())),
                    index=chunk.index[observed],
                )
                context_size += int(observed.sum())
                sequence_context.append(
                    chunk.loc[
                        observed,
                        list(
                            dict.fromkeys(
                                (
                                    "asset_id",
                                    "timestamp",
                                    *context_columns,
                                )
                            )
                        ),
                    ]
                )
                context = chunk.loc[observed]
                _write_features(context, spec.input_columns, writer)
                if spec.model_algorithm in DEEPLOB_ALGORITHMS:
                    from quant_research_platform.research.models.deeplob import (
                        complete_book_rows,
                    )

                    valid = complete_book_rows(context)
                else:
                    valid = context.loc[:, spec.input_columns].notna().any(axis=1)
                context_valid.append(valid.to_numpy())
                del context
            if sample_windows is not None:
                selected = pd.Series(False, index=chunk.index)
                for window in sample_windows:
                    selected |= prediction_window_mask(chunk.timestamp, window)
                chunk = chunk.loc[selected]
                if chunk.empty:
                    continue
            filtered, metadata = _legal_training_rows(
                chunk,
                spec,
                fitting_cutoff=fitting_cutoff,
                availability_column=availability_columns[0],
            )
            if spec.model_algorithm in DEEPLOB_ALGORITHMS:
                filtered = filtered.loc[
                    filtered.loc[:, spec.input_columns].notna().all(axis=1)
                ]
            rows_input += metadata["rows_input"]
            rows_after_filter += metadata["rows_after_filter_mode"]
            filter_columns_used = list(metadata["filter_columns_used"])
            filter_definition = str(metadata["filter_definition"])
            selected_columns = [
                "timestamp",
                *(("asset_id",) if "asset_id" in filtered.columns else ()),
                *(("market_type",) if "market_type" in filtered.columns else ()),
                spec.target_column,
                availability_columns[0],
            ]
            usable = filtered.loc[:, selected_columns].copy()
            if sequence_context:
                usable.index = context_positions.loc[filtered.index].to_numpy()
            else:
                _write_features(filtered, spec.input_columns, writer)
                usable.index = np.arange(context_size, context_size + len(filtered))
                context_size += len(filtered)
            del chunk, filtered
            if usable.empty:
                del usable
                continue
            retained.append(usable)
            del usable
    training = (
        retained[0]
        if len(retained) == 1
        else pd.concat(retained)
        if retained
        else pd.DataFrame()
    )
    retained.clear()
    train_start = train_end = None
    if not training.empty:
        timestamps = strict_timestamp_series(
            training["timestamp"], error="invalid_model_training_timestamp"
        )
        training["timestamp"] = timestamps
        training = training.sort_values(
            ["timestamp", *(("asset_id",) if "asset_id" in training else ())],
            kind="stable",
        )
        train_start = time_value_isoformat(pd.Timestamp(timestamps.min()))
        train_end = time_value_isoformat(pd.Timestamp(timestamps.max()))
    row_counts = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "rows_after_input_label_dropna": int(len(training)),
        "training_asset_ids": sorted(training.asset_id.unique())
        if "asset_id" in training
        else [],
        "train_start": train_start,
        "train_end": train_end,
        "label_availability_column": availability_columns[0],
        "label_available_max": None
        if training.empty
        else time_value_isoformat(
            pd.Timestamp(
                strict_timestamp_series(
                    training[availability_columns[0]],
                    error="invalid_model_label_maturity",
                ).max()
            ),
            preserve_datetime=True,
        ),
    }
    if not training.empty:
        training = training.drop(columns=availability_columns[0])
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    sessions = None
    if spec.training_weighting.half_life_periods is not None:
        if trading_calendar is None or "market_type" not in training:
            raise ValueError("training_decay_own_market_calendar_required")
        from quant_research_platform.markets.implementations.a_share.calendar import (
            market_type_calendar_state,
        )

        markets = training.market_type.drop_duplicates()
        if markets.empty or markets.isna().any():
            raise ValueError("training_decay_own_market_calendar_required")
        dates = pd.date_range(
            training.timestamp.min().normalize(),
            training.timestamp.max().normalize(),
            freq="D",
        )
        state = None
        for market in markets:
            observed = market_type_calendar_state(trading_calendar, market).reindex(
                dates
            )
            if observed.isna().any():
                raise ValueError("training_decay_calendar_coverage_missing")
            if state is not None and not np.array_equal(
                observed.to_numpy(), state.to_numpy()
            ):
                raise ValueError("training_decay_common_session_clock_required")
            state = observed
        assert state is not None
        sessions = state.index[state.to_numpy(dtype=bool)]
    training_rows = training.index.to_numpy(dtype="int64")
    training = training.reset_index(drop=True)
    writer.close()
    feature_file.flush()
    mapping = np.memmap(feature_file, dtype="uint8", mode="r")
    features = (
        pa.ipc.open_file(pa.BufferReader(pa.py_buffer(mapping)))
        .read_all()
        .to_pandas(types_mapper=pd.ArrowDtype)
    )
    return PreparedTraining(
        training,
        row_counts,
        filter_metadata,
        spec.input_columns,
        spec.training_weighting,
        spec.target_column,
        features,
        feature_file,
        mapping,
        training_rows,
        spec.model_algorithm == "xgboost_regressor",
        sessions,
        spec.training_scaling,
        pd.concat(sequence_context, ignore_index=True) if sequence_context else None,
        spec.model_algorithm in DEEPLOB_ALGORITHMS,
        np.concatenate(context_valid) if context_valid else None,
    )


def _fit_predictor(
    spec: ModelSpec,
    *,
    prepared_training: PreparedTraining,
    estimator: object,
    validation=None,
    fit_inputs=None,
) -> object:
    weights, scaler = prepared_training.preprocessing
    from sklearn.exceptions import ConvergenceWarning

    if spec.model_algorithm == "lasso_regression" and estimator.precompute is True:
        _fit_lasso_gram(estimator, prepared_training)
        return ScaledPredictor(estimator, scaler)
    if spec.model_algorithm == "xgboost_regressor":
        with _booster_context(spec):
            _fit_booster(
                estimator, prepared_training, spec, validation, fit_inputs=fit_inputs
            )
        return ScaledPredictor(estimator, scaler)
    scaled_matrix = prepared_training.matrix
    y = prepared_training.training[spec.target_column].to_numpy(dtype="float64")
    try:
        if spec.model_algorithm == "lasso_regression" and estimator.max_iter is None:
            _fit_lasso_until_converged(estimator, scaled_matrix, y, weights)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                estimator.fit(
                    _sklearn_fit_matrix(scaled_matrix, estimator),
                    y,
                    sample_weight=weights,
                )
    except ConvergenceWarning:
        raise ValueError("model_fit_non_converged") from None
    return ScaledPredictor(estimator, scaler)


def _sklearn_fit_matrix(matrix, estimator):
    """Give an estimator that centers in place ownership of its borrowed matrix."""
    if estimator.copy_X:
        return matrix
    return pd.DataFrame(
        matrix.to_numpy(copy=False).copy(order="K"),
        index=matrix.index,
        columns=matrix.columns,
        copy=False,
    )


def _fit_lasso_until_converged(estimator, matrix, target, weights):
    from sklearn.exceptions import ConvergenceWarning

    center = np.average(target, weights=weights) if estimator.fit_intercept else 0.0
    tolerance = estimator.tol * np.average((target - center) ** 2, weights=weights)
    warm_start = estimator.warm_start
    block, iterations = 1000, 0
    estimator.set_params(warm_start=True)
    try:
        while True:
            estimator.set_params(max_iter=block)
            with warnings.catch_warnings():
                # A block warning means continue the same coordinate-descent solve.
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(
                    _sklearn_fit_matrix(matrix, estimator),
                    target,
                    sample_weight=weights,
                )
            iterations += estimator.n_iter_
            if estimator.dual_gap_ <= tolerance:
                break
            block *= 2
    finally:
        estimator.set_params(max_iter=None, warm_start=warm_start)
    estimator.n_iter_ = iterations


def _fit_lasso_gram(estimator, prepared):
    # Lasso.fit and public lasso_path still require the full X. Use their same
    # native Gram solver without materializing X solely to provide its shape.
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram
    from sklearn.utils import check_random_state

    limit = estimator.max_iter
    block = 1000 if limit is None else limit
    estimator.set_params(max_iter=block)
    try:
        estimator._validate_params()
    finally:
        estimator.set_params(max_iter=limit)
    gram, cross, mean, target, target_mean, root_weight = prepared.lasso_gram
    count = len(target)
    if not estimator.fit_intercept:
        gram = gram + count * np.outer(mean, mean)
        cross = cross + count * mean * target_mean
        target = target + target_mean * root_weight
    coefficient = (
        estimator.coef_.copy()
        if estimator.warm_start and hasattr(estimator, "coef_")
        else np.zeros(len(mean))
    )
    iterations = 0
    tolerance = estimator.tol * (target @ target)
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            coefficient, gap, _, used = enet_coordinate_descent_gram(
                coefficient,
                estimator.alpha * count,
                0.0,
                gram,
                cross,
                target,
                block,
                estimator.tol,
                check_random_state(estimator.random_state),
                estimator.selection == "random",
                estimator.positive,
            )
        iterations += used
        if gap <= tolerance:
            break
        if estimator.max_iter is not None:
            raise ValueError("model_fit_non_converged")
        block *= 2
    estimator.coef_ = np.asarray(coefficient)
    estimator.intercept_ = (
        target_mean - mean @ estimator.coef_ if estimator.fit_intercept else 0.0
    )
    estimator.n_features_in_ = len(mean)
    estimator.feature_names_in_ = np.asarray(prepared.input_columns, dtype=object)
    estimator.n_iter_ = iterations
    estimator.dual_gap_ = gap / count


def _booster_matrix(
    estimator, prepared, scaler, *, ref=None, external=False, max_quantile_batches=None
):
    from xgboost import DataIter

    weights = prepared.preprocessing[0] if ref is None else prepared.validation_weights
    if external:
        import cupy
        from xgboost import ExtMemQuantileDMatrix

    class Batches(DataIter):
        def __init__(self):
            super().__init__(release_data=True)
            self.reset()

        def reset(self):
            self.blocks = iter(prepared.blocks(scaler))

        def next(self, input_data):
            block = next(self.blocks, None)
            if block is None:
                return False
            start, stop, values = block
            if external:
                with cupy.cuda.Device(int(estimator.device.partition(":")[2] or "0")):
                    values = cupy.asarray(values)
            input_data(
                data=values,
                label=prepared.training[prepared.target_column]
                .iloc[start:stop]
                .to_numpy(),
                weight=weights[start:stop],
                feature_names=list(prepared.input_columns),
                feature_types=estimator.feature_types,
                feature_weights=estimator.feature_weights,
            )
            return True

    if external:
        # Share the driver async pool with XGBoost instead of retaining a CuPy cache.
        with cupy.cuda.using_allocator(cupy.cuda.malloc_async):
            return ExtMemQuantileDMatrix(
                Batches(),
                ref=ref,
                missing=estimator.missing,
                nthread=estimator.n_jobs,
                max_bin=estimator.max_bin,
                max_quantile_batches=max_quantile_batches,
                cache_host_ratio=1.0,
                enable_categorical=estimator.enable_categorical,
            )
    return estimator._create_dmatrix(
        ref=ref,
        data=Batches(),
        missing=estimator.missing,
        enable_categorical=estimator.enable_categorical,
    )


def _booster_context(spec):
    from xgboost import config_context

    return config_context(
        **(
            {"use_cuda_async_pool": True}
            if spec.hyperparameters.get("external_memory")
            else {}
        )
    )


def _booster_inputs(estimator, prepared, spec, validation):
    scaler = prepared.preprocessing[1]
    external = spec.hyperparameters.get("external_memory", False)
    batches = spec.hyperparameters.get("max_quantile_batches")
    training = _booster_matrix(
        estimator, prepared, scaler, external=external, max_quantile_batches=batches
    )
    held = (
        None
        if validation is None
        else _booster_matrix(
            estimator,
            validation,
            scaler,
            ref=training,
            external=external,
            max_quantile_batches=batches,
        )
    )
    return training, held


def _fit_booster(estimator, prepared, spec, validation, *, fit_inputs=None):
    from xgboost import Booster
    from xgboost.callback import EarlyStopping

    training, held = (
        _booster_inputs(estimator, prepared, spec, validation)
        if fit_inputs is None
        else fit_inputs
    )
    booster = Booster(
        estimator.get_xgb_params(),
        cache=[training] if held is None else [training, held],
    )
    patience = spec.hyperparameters.get("early_stopping_rounds")
    callback = (
        None if patience is None else EarlyStopping(rounds=patience, save_best=True)
    )
    if callback is not None:
        booster = callback.before_training(booster)
    history = {"validation_0": {}}
    iterations = (
        count()
        if "n_estimators" in spec.hyperparameters
        and spec.hyperparameters["n_estimators"] is None
        else range(estimator.get_num_boosting_rounds())
    )
    for iteration in iterations:
        booster.update(training, iteration)
        if held is not None:
            for measurement in booster.eval_set(
                [(held, "validation_0")], iteration
            ).split()[1:]:
                name, value = measurement.split(":", 1)
                metric = name.removeprefix("validation_0-")
                history["validation_0"].setdefault(metric, []).append(float(value))
        if callback is not None and callback.after_iteration(
            booster, iteration, history
        ):
            break
    estimator._Booster = (
        booster if callback is None else callback.after_training(booster)
    )
    if held is not None:
        estimator.evals_result_ = history


def iter_model_work_frames(
    frames: Iterable[pd.DataFrame], *, required: set[str], work_columns: tuple[str, ...]
) -> Iterable[pd.DataFrame]:
    with closing_iterator(iter(frames)) as active_frames:
        for frame in active_frames:
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"missing_model_columns: {', '.join(missing)}")
            selected = [column for column in work_columns if column in frame.columns]
            yielded_value = frame.loc[:, selected]
            yield yielded_value


def model_training_read_columns(
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    *,
    label_column: str,
    sample_filter_mode,
    availability_column=None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *primary_key,
                *input_columns,
                label_column,
                *(
                    training_availability_columns(label_column)
                    if availability_column is None
                    else (availability_column,)
                ),
                *required_filter_columns(sample_filter_mode),
            )
        )
    )
