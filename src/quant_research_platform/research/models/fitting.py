"""Fit supported predictors on legal samples."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from itertools import count

import numpy as np
import pandas as pd

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
    strict_finite_numeric_series,
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


@dataclass(frozen=True)
class PreparedTraining:
    training: pd.DataFrame
    row_counts: dict[str, object]
    filter_metadata: dict[str, object]
    input_columns: tuple[str, ...]
    training_weighting: TrainingWeighting
    native_missing: bool = False
    trading_sessions: object = None
    training_scaling: bool = True
    sequence_context: pd.DataFrame | None = None
    complete_inputs: bool = False

    @cached_property
    def preprocessing(self):
        weights = equal_timestamp_sample_weights(
            self.training["timestamp"],
            self.training_weighting,
            trading_sessions=self.trading_sessions,
        )
        matrix = self.training.loc[:, self.input_columns]
        scaler = ModelMatrixScaler.fit(
            matrix,
            sample_weight=weights,
            missing=None
            if self.complete_inputs
            else "native"
            if self.native_missing
            else "impute",
            scaling=self.training_scaling,
        )
        return weights, scaler

    @cached_property
    def matrix(self):
        return self.preprocessing[1].transform(self.training.loc[:, self.input_columns])


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
) -> ModelFitResult:
    if spec.model_algorithm == "linear_regression":
        result = _fit_linear_model_result(prepared_training, spec)
    elif spec.model_algorithm in SEQUENCE_ALGORITHMS:
        from quant_research_platform.research.models.neural import fit_sequence

        predictor, curve, stopping = fit_sequence(prepared_training, spec, validation)
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
            prepared_training, spec, estimator=estimator, validation=validation
        )
    target = prepared_training.training[spec.target_column].to_numpy(dtype="float64")
    result.metadata["training_target_mean"] = float(
        np.average(target, weights=prepared_training.preprocessing[0])
    )
    return result


def _fit_linear_model_result(
    prepared_training,
    spec: ModelSpec,
) -> ModelFitResult:
    training = prepared_training.training
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if training.empty:
        raise ValueError("model_training_sample_empty")
    weights, scaler = prepared_training.preprocessing
    transformed = prepared_training.matrix.to_numpy(dtype="float64")
    design = np.column_stack((np.ones(len(transformed)), transformed))
    weighted = np.sqrt(weights)
    target = training[spec.target_column].to_numpy(dtype="float64")
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
) -> ModelFitResult:
    training = prepared_training.training
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if training.empty:
        raise ValueError("model_training_sample_empty")
    predictor = _fit_predictor(
        spec,
        prepared_training=prepared_training,
        estimator=estimator,
        validation=validation,
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
    columns = spec.input_columns
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    for column in columns:
        strict_finite_numeric_series(
            frame[column], error=f"invalid_training_value: {column}", allow_null=True
        )
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


def _collect_training_frame(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
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
    with closing_iterator(
        iter(
            iter_model_work_frames(frames, required=required, work_columns=work_columns)
        )
    ) as work_frames:
        for chunk in work_frames:
            chunk = chunk.reset_index(drop=True)
            if spec.model_algorithm in SEQUENCE_ALGORITHMS:
                observed = chunk.loc[:, spec.input_columns].notna().any(axis=1)
                if context_columns:
                    observed = pd.Series(True, index=chunk.index)
                observed &= pd.to_datetime(chunk.timestamp).le(
                    pd.Timestamp(fitting_cutoff)
                )
                sequence_context.append(
                    chunk.loc[
                        observed,
                        list(
                            dict.fromkeys(
                                (
                                    "asset_id",
                                    "timestamp",
                                    *spec.input_columns,
                                    *context_columns,
                                )
                            )
                        ),
                    ]
                )
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
                *spec.input_columns,
                spec.target_column,
                availability_columns[0],
            ]
            usable = filtered.loc[:, selected_columns].copy()
            del chunk, filtered
            if usable.empty:
                del usable
                continue
            retained.append(usable)
            del usable
    training = (
        retained[0]
        if len(retained) == 1
        else pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame()
    )
    retained.clear()
    training.index = pd.RangeIndex(len(training))
    train_start = train_end = None
    if not training.empty:
        timestamps = strict_timestamp_series(
            training["timestamp"], error="invalid_model_training_timestamp"
        )
        training["timestamp"] = timestamps
        training = training.sort_values(
            ["timestamp", *(("asset_id",) if "asset_id" in training else ())],
            kind="stable",
            ignore_index=True,
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
    return PreparedTraining(
        training,
        row_counts,
        filter_metadata,
        spec.input_columns,
        spec.training_weighting,
        spec.model_algorithm == "xgboost_regressor",
        sessions,
        spec.training_scaling,
        pd.concat(sequence_context, ignore_index=True) if sequence_context else None,
        spec.model_algorithm in DEEPLOB_ALGORITHMS,
    )


def _fit_predictor(
    spec: ModelSpec,
    *,
    prepared_training: PreparedTraining,
    estimator: object,
    validation=None,
) -> object:
    y = prepared_training.training[spec.target_column].to_numpy(dtype="float64")
    weights, scaler = prepared_training.preprocessing
    from sklearn.exceptions import ConvergenceWarning

    scaled_matrix = prepared_training.matrix
    fit_options = {}
    patience = spec.hyperparameters.get("early_stopping_rounds")
    if validation is not None:
        fit_options = {
            "eval_set": [
                (
                    scaler.transform(validation.training.loc[:, spec.input_columns]),
                    validation.training[spec.target_column].to_numpy(dtype="float64"),
                )
            ],
            "verbose": False,
        }
    callback = None
    if patience is not None:
        from xgboost.callback import EarlyStopping

        # Native save_best discards later trees for prediction and explanations.
        callback = EarlyStopping(rounds=patience, save_best=True)
        estimator.set_params(
            early_stopping_rounds=None,
            callbacks=[callback],
        )
    try:
        if spec.model_algorithm == "lasso_regression" and estimator.max_iter is None:
            _fit_lasso_until_converged(estimator, scaled_matrix, y, weights)
        elif (
            spec.model_algorithm == "xgboost_regressor"
            and "n_estimators" in spec.hyperparameters
            and spec.hyperparameters["n_estimators"] is None
        ):
            _fit_booster_until_stopped(
                estimator,
                scaled_matrix,
                y,
                weights,
                fit_options["eval_set"][0],
                callback,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                matrix = (
                    _sklearn_fit_matrix(scaled_matrix, estimator)
                    if spec.model_algorithm in {"lasso_regression", "ridge_regression"}
                    else scaled_matrix
                )
                estimator.fit(matrix, y, sample_weight=weights, **fit_options)
    except ConvergenceWarning:
        raise ValueError("model_fit_non_converged") from None
    finally:
        if patience is not None:
            estimator.set_params(early_stopping_rounds=patience, callbacks=None)
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


def _fit_booster_until_stopped(
    estimator, matrix, target, weights, validation, callback
):
    from xgboost import Booster

    options = {
        "missing": estimator.missing,
        "enable_categorical": estimator.enable_categorical,
        "feature_types": estimator.feature_types,
    }
    training = estimator._create_dmatrix(
        ref=None, data=matrix, label=target, weight=weights, **options
    )
    held = estimator._create_dmatrix(
        ref=training, data=validation[0], label=validation[1], **options
    )
    booster = Booster(estimator.get_xgb_params(), cache=[training, held])
    booster = callback.before_training(booster)
    history = {"validation_0": {}}
    for iteration in count():
        booster.update(training, iteration)
        for measurement in booster.eval_set(
            [(held, "validation_0")], iteration
        ).split()[1:]:
            name, value = measurement.split(":", 1)
            metric = name.removeprefix("validation_0-")
            history["validation_0"].setdefault(metric, []).append(float(value))
        if callback.after_iteration(booster, iteration, history):
            break
    estimator._Booster = callback.after_training(booster)
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
