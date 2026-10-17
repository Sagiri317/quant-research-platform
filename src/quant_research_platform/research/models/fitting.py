"""Supervised model fitting owners."""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import quant_research_platform.research.models.estimators as estimators
import quant_research_platform.research.models.feature_importance as feature_importance
from quant_research_platform.contracts.experiments import (
    legal_training_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
)
from quant_research_platform.contracts.models import (
    ModelSpec,
    TrainingWeighting,
    _thaw_model_hyperparameters,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    canonical_feature_validity_columns,
    feature_observation_columns,
    mask_unobserved_features,
    model_input_validity_columns,
    model_observed_input_mask,
)
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models.rank_ensemble import (
    ICWeightedRankEnsemble,
    RankEnsemblePredictor,
    rank_model_parameters,
)
from quant_research_platform.research.models.scaler import (
    LinearScaledPredictor,
    ModelMatrixScaler,
    ScaledPredictor,
    equal_timestamp_sample_weights,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames


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
    rank_inputs: bool
    native_missing: bool = False
    trading_sessions: object = None
    training_scaling: bool = True

    @cached_property
    def preprocessing(self):
        # Begin the trial before learning; candidates share only immutable fit facts.
        weights = equal_timestamp_sample_weights(
            self.training["timestamp"],
            self.training_weighting,
            trading_sessions=self.trading_sessions,
        )
        if self.rank_inputs:
            return weights, None, self.training
        matrix = self.training.loc[:, self.input_columns]
        scaler = ModelMatrixScaler.fit(
            matrix,
            sample_weight=weights,
            missing="native" if self.native_missing else "impute",
            scaling=self.training_scaling,
        )
        return weights, scaler, scaler.transform(matrix)


def prediction_callable(input_columns: tuple[str, ...], prediction_model: object):
    """Build the model-owned prediction callable used by selection and output."""

    def predict_from_model(frame: pd.DataFrame) -> np.ndarray:
        columns = input_columns
        if isinstance(
            prediction_model, (RankEnsemblePredictor, ICWeightedRankEnsemble)
        ):
            columns = tuple(dict.fromkeys(("timestamp", *input_columns)))
        predicted = prediction_model.predict(frame.loc[:, list(columns)])
        return np.asarray(predicted, dtype="float64")

    return predict_from_model


def validate_model_state(predictor: object, metadata: dict[str, object]) -> None:
    """Bind persisted model metadata to the actual trained predictor state."""
    spec = ModelSpec(
        model_id="persisted_model",
        model_algorithm=metadata["model_algorithm"],
        input_factor_columns=tuple(metadata["input_factor_columns"]),
        input_feature_columns=tuple(metadata["input_feature_columns"]),
        target_column=metadata["target_column"],
        sample_filter_mode=metadata["sample_filter_mode"],
        hyperparameters=metadata["hyperparameters"],
        random_seed=metadata["random_seed"],
        max_train_rows=metadata["max_train_rows"],
        training_weighting=metadata["training_weighting"],
        training_scaling=metadata["training_scaling"],
    )
    if spec.model_algorithm == "linear_regression":
        if type(predictor) is not LinearScaledPredictor:
            raise ValueError("model_state_metadata_mismatch")
        try:
            coefficients = np.asarray(predictor.coefficients, dtype="float64")
        except (TypeError, ValueError):
            raise ValueError("model_state_metadata_mismatch") from None
        if (
            coefficients.shape != (len(predictor.scaler.columns) + 1,)
            or not np.isfinite(coefficients).all()
        ):
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.linear_feature_importance(spec, coefficients)
    elif spec.model_algorithm in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        expected_type = (
            RankEnsemblePredictor
            if spec.model_algorithm == "equal_weight_rank_ensemble"
            else ICWeightedRankEnsemble
        )
        if type(predictor) is not expected_type:
            raise ValueError("model_state_metadata_mismatch")
        parameters = rank_model_parameters(predictor)
        if isinstance(predictor, RankEnsemblePredictor):
            if any(direction not in (-1.0, 1.0) for direction in predictor.directions):
                raise ValueError("model_state_metadata_mismatch")
        elif not np.isclose(
            sum(abs(weight) for weight in predictor.weights),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("model_state_metadata_mismatch")
        if (
            metadata.get("rank_parameters") != parameters
            or metadata["scaler"] is not None
        ):
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.estimator_feature_importance(spec, predictor)
    else:
        if type(predictor) is not ScaledPredictor:
            raise ValueError("model_state_metadata_mismatch")
        expected = estimators._MODEL_ESTIMATOR_FACTORIES[spec.model_algorithm](
            spec, dict(metadata["resolved_hyperparameters"])
        )
        actual = predictor.estimator
        if type(actual) is not type(expected):
            raise ValueError("model_state_metadata_mismatch")
        actual_parameters, expected_parameters = (
            actual.get_params(deep=False),
            expected.get_params(deep=False),
        )
        if set(actual_parameters) != set(expected_parameters):
            raise ValueError("model_state_metadata_mismatch")
        for name, expected_value in expected_parameters.items():
            actual_value = actual_parameters[name]
            if isinstance(expected_value, float) and np.isnan(expected_value):
                matches = isinstance(actual_value, float) and np.isnan(actual_value)
            else:
                try:
                    matches = canonical_identity_bytes(
                        actual_value
                    ) == canonical_identity_bytes(expected_value)
                except ValueError:
                    matches = False
            if not matches:
                raise ValueError("model_state_metadata_mismatch")
        try:
            inputs_match = (
                actual.n_features_in_ == len(predictor.scaler.columns)
                and tuple(actual.feature_names_in_) == predictor.scaler.columns
            )
        except AttributeError:
            raise ValueError("model_state_metadata_mismatch") from None
        if not inputs_match:
            raise ValueError("model_state_metadata_mismatch")
        importance = feature_importance.estimator_feature_importance(spec, predictor)
    if isinstance(predictor, (LinearScaledPredictor, ScaledPredictor)):
        if (
            type(predictor.scaler) is not ModelMatrixScaler
            or any(type(value) is not Fraction for value in predictor.scaler.offset_)
            or predictor.scaler.scaling != spec.training_scaling
            or predictor.scaler.to_payload() != metadata["scaler"]
        ):
            raise ValueError("model_state_metadata_mismatch")
    saved_importance = metadata.get("feature_importance")
    if (importance is None) != (saved_importance is None) or (
        importance is not None
        and (
            set(importance) != set(saved_importance)
            or not np.allclose(
                list(importance.values()),
                [saved_importance[column] for column in importance],
                rtol=1e-12,
                atol=1e-15 * max(map(abs, importance.values()), default=0.0)
                + np.nextafter(0.0, 1.0),
            )
        )
    ):
        raise ValueError("model_state_metadata_mismatch")


def _fit_model_result(
    prepared_training,
    spec: ModelSpec,
    *,
    estimator: object,
) -> ModelFitResult:
    if spec.model_algorithm == "linear_regression":
        return _fit_linear_model_result(prepared_training, spec)
    return _fit_estimator_model_result(
        prepared_training,
        spec,
        estimator=estimator,
    )


def _fit_linear_model_result(
    prepared_training,
    spec: ModelSpec,
) -> ModelFitResult:
    training = prepared_training.training
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if training.empty:
        raise ValueError("model_training_sample_empty")
    weights, scaler, matrix = prepared_training.preprocessing
    transformed = matrix.to_numpy(dtype="float64")
    design = np.column_stack((np.ones(len(transformed)), transformed))
    weighted = np.sqrt(weights)
    target = training[spec.target_column].to_numpy(dtype="float64")
    _, target_exponent = np.frexp(float(np.max(np.abs(target))))
    design *= weighted[:, None]
    with threadpool_limits(limits=1, user_api="blas"):
        beta = np.linalg.lstsq(
            design,
            np.ldexp(target, -target_exponent) * weighted,
            rcond=None,
        )[0]
    beta = np.ldexp(beta, target_exponent)
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
        "rows_input": _strict_non_negative_count_value(filter_metadata["rows_input"]),
        "rows_after_filter_mode": _strict_non_negative_count_value(
            filter_metadata["rows_after_filter_mode"]
        ),
        "rows_after_input_label_dropna": _strict_non_negative_count_value(
            row_counts["rows_after_input_label_dropna"]
        ),
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
        "max_train_rows": spec.max_train_rows,
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
) -> ModelFitResult:
    training = prepared_training.training
    row_counts = prepared_training.row_counts
    filter_metadata = prepared_training.filter_metadata
    if training.empty:
        raise ValueError("model_training_sample_empty")
    threads = estimators.model_thread_count(spec)
    predictor = _fit_predictor(
        spec, prepared_training=prepared_training, estimator=estimator, threads=threads
    )

    resolved_hyperparameters = estimators.resolved_model_hyperparameters(spec)
    if spec.model_algorithm in estimators._PARALLEL_MODEL_ALGORITHMS:
        fitted_threads = getattr(predictor, "n_jobs", None)
        if type(fitted_threads) is not int or fitted_threads <= 0:
            raise RuntimeError("invalid_fitted_model_thread_count")
        resolved_hyperparameters["n_jobs"] = fitted_threads
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
    if isinstance(predictor, (RankEnsemblePredictor, ICWeightedRankEnsemble)):
        metadata["rank_parameters"] = rank_model_parameters(predictor)
    result = ModelFitResult(metadata=metadata, prediction_model=predictor)
    return result


def _legal_training_rows(
    frame: pd.DataFrame,
    spec: ModelSpec,
    *,
    fitting_cutoff: object,
    availability_column: str,
    declared_columns=(),
):
    """Select legal rows without learning preprocessing or applying a row cap."""
    validity = model_input_validity_columns(
        spec.input_columns, spec.model_algorithm, declared_columns
    )
    columns = tuple(dict.fromkeys((*spec.input_columns, *validity)))
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    for column in columns:
        strict_finite_numeric_series(
            frame[column], error=f"invalid_training_value: {column}", allow_null=True
        )
    keep = legal_training_mask(
        frame,
        fitting_cutoff=fitting_cutoff,
        availability_column=availability_column,
        feature_columns=(),
        target_column=spec.target_column,
    )
    keep &= model_observed_input_mask(frame, spec.input_columns, declared_columns)
    masked = mask_unobserved_features(frame, spec.input_columns, declared_columns)
    filtered, metadata = apply_filter_mode(
        masked, mode=spec.sample_filter_mode, metadata_key="sample_filter_mode"
    )
    return filtered.loc[keep.loc[filtered.index]], metadata


def _collect_training_frame(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    fitting_cutoff: object,
    availability_column: str | None = None,
    declared_columns=(),
    trading_calendar=None,
) -> PreparedTraining:
    validity_columns = model_input_validity_columns(
        spec.input_columns, spec.model_algorithm, declared_columns
    )
    required = set((*spec.input_columns, *validity_columns))
    required.add(spec.target_column)
    required.add("timestamp")
    required.add("eligible")
    availability_columns = (
        (availability_column,)
        if availability_column is not None
        else training_availability_columns(spec.target_column)
    )
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
                *validity_columns,
                spec.target_column,
                "eligible",
                *availability_columns,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    retained: deque[pd.DataFrame] = deque()
    retained_rows = 0
    with closing_iterator(
        iter(
            iter_model_work_frames(frames, required=required, work_columns=work_columns)
        )
    ) as work_frames:
        for chunk in work_frames:
            chunk = chunk.reset_index(drop=True)
            filtered, metadata = _legal_training_rows(
                chunk,
                spec,
                fitting_cutoff=fitting_cutoff,
                availability_column=availability_columns[0],
                declared_columns=declared_columns,
            )
            rows_input += _strict_non_negative_count_value(metadata["rows_input"])
            rows_after_filter += _strict_non_negative_count_value(
                metadata["rows_after_filter_mode"]
            )
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
            blocks = (
                (usable,)
                if spec.max_train_rows is None
                else (block for _, block in usable.groupby("timestamp", sort=False))
            )
            for block in blocks:
                retained.append(block)
                retained_rows += len(block)
                if spec.max_train_rows is not None:
                    while retained_rows > spec.max_train_rows and len(retained) > 1:
                        retained_rows -= len(retained.popleft())
            del usable, blocks, block
    training = (
        retained[0]
        if len(retained) == 1
        else pd.concat(retained, ignore_index=True)
        if retained
        else pd.DataFrame()
    )
    retained.clear()
    training = _apply_training_controls(training, spec)
    train_start = train_end = None
    if not training.empty:
        timestamps = strict_timestamp_series(
            training["timestamp"], error="invalid_model_training_timestamp"
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
        spec.model_algorithm
        in {"equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"},
        _native_missing(spec),
        sessions,
        spec.training_scaling,
    )


def _fit_predictor(
    spec: ModelSpec,
    *,
    prepared_training: PreparedTraining,
    estimator: object,
    threads: int,
) -> object:
    y = prepared_training.training[spec.target_column].to_numpy(
        dtype="float64", copy=True
    )
    weights, scaler, matrix = prepared_training.preprocessing
    weights = weights.copy()
    if spec.model_algorithm in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        return estimator.fit(
            prepared_training.training.copy(deep=True), y, sample_weight=weights
        )
    from sklearn.exceptions import ConvergenceWarning

    if spec.model_algorithm in estimators._PARALLEL_MODEL_ALGORITHMS:
        set_params = getattr(estimator, "set_params", None)
        if not callable(set_params):
            raise RuntimeError("model_thread_configuration_unavailable")
        set_params(n_jobs=threads)
    openmp_threads = (
        threads
        if spec.model_algorithm in {"xgboost_regressor", "lightgbm_regressor"}
        else 1
    )
    scaled_matrix = matrix.copy(deep=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            with threadpool_limits(limits=1, user_api="blas"):
                with threadpool_limits(limits=openmp_threads, user_api="openmp"):
                    estimator.fit(scaled_matrix, y, sample_weight=weights)
    except ConvergenceWarning:
        raise ValueError("model_fit_non_converged") from None
    return ScaledPredictor(estimator, scaler)


def iter_model_work_frames(
    frames: Iterable[pd.DataFrame], *, required: set[str], work_columns: tuple[str, ...]
) -> Iterable[pd.DataFrame]:
    complete_frames = iter_complete_time_frames(
        iter(frames), timestamp_column="timestamp"
    )
    with closing_iterator(complete_frames) as active_frames:
        for frame in active_frames:
            missing = sorted(required.difference(frame.columns))
            if missing:
                raise ValueError(f"missing_model_columns: {', '.join(missing)}")
            selected = [column for column in work_columns if column in frame.columns]
            yielded_value = frame.loc[:, selected]
            try:
                yield yielded_value
            except GeneratorExit:
                return


def model_training_read_columns(
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    *,
    label_column: str,
    sample_filter_mode,
    availability_column=None,
    declared_columns=(),
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *primary_key,
                *input_columns,
                *canonical_feature_validity_columns(input_columns, declared_columns),
                *feature_observation_columns(input_columns, declared_columns),
                *feature_observation_columns(
                    canonical_feature_validity_columns(input_columns, declared_columns),
                    declared_columns,
                ),
                label_column,
                "eligible",
                *(
                    training_availability_columns(label_column)
                    if availability_column is None
                    else (availability_column,)
                ),
                *required_filter_columns(sample_filter_mode),
            )
        )
    )


def _strict_non_negative_count_value(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_model_metadata_count")
    return value


def _apply_training_controls(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Apply only the generic, explicitly requested row cap."""
    if frame.empty:
        return frame
    result = frame.copy(deep=False)
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_model_training_timestamp"
    )
    if spec.max_train_rows is not None:
        counts = result.groupby("timestamp", sort=True).size()
        retained = counts.iloc[::-1].cumsum() <= spec.max_train_rows
        result = result.loc[result.timestamp.isin(retained.index[retained])]
    result.index = pd.RangeIndex(len(result))
    return result


def _native_missing(spec):
    if spec.model_algorithm == "linear_regression":
        return False
    estimator = estimators.build_estimator_from_model_spec(spec)
    if spec.model_algorithm in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }:
        return False
    from sklearn.utils import get_tags

    return bool(get_tags(estimator).input_tags.allow_nan)
