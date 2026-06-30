"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from importlib import import_module

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.research.models import (
    estimators,
    feature_importance,
    validation_split,
)
from quant_research_platform.research.models.prediction import predict_model_frame
from quant_research_platform.runtime.resources import ChunkAxis


@dataclass(frozen=True)
class ModelFitResult:
    coefficients: np.ndarray
    metadata: dict[str, object]
    prediction_model: object | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class _AlgorithmFacts:
    algorithm: str
    dependency_package: str | None = None
    estimator_factory: Callable[[ModelSpec, dict[str, object]], object] | None = None
    default_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    smoke_hyperparameters: Mapping[str, object] = field(default_factory=dict)
    input_kind: str = "feature_inputs"


_MODEL_ALGORITHM_FACTS = {
    "linear_regression": _AlgorithmFacts(
        "linear_regression",
        "sklearn",
        estimator_factory=estimators._linear_regression_estimator,
    ),
    "ridge_regression": _AlgorithmFacts(
        "ridge_regression",
        "sklearn",
        estimator_factory=estimators._ridge_regression_estimator,
        default_hyperparameters={"alpha": 1.0},
    ),
    "lasso_regression": _AlgorithmFacts(
        "lasso_regression",
        "sklearn",
        estimator_factory=estimators._lasso_regression_estimator,
        default_hyperparameters={"alpha": 0.001, "max_iter": 10_000},
    ),
    "elastic_net": _AlgorithmFacts(
        "elastic_net",
        "sklearn",
        estimator_factory=estimators._elastic_net_estimator,
        default_hyperparameters={
            "alpha": 0.001,
            "l1_ratio": 0.5,
            "max_iter": 10_000,
        },
    ),
    "huber_regression": _AlgorithmFacts(
        "huber_regression",
        "sklearn",
        estimator_factory=estimators._huber_regression_estimator,
        default_hyperparameters={"max_iter": 1_000},
    ),
    "decision_tree_regressor": _AlgorithmFacts(
        "decision_tree_regressor",
        "sklearn",
        estimator_factory=estimators._decision_tree_regressor,
    ),
    "random_forest_regressor": _AlgorithmFacts(
        "random_forest_regressor",
        "sklearn",
        estimator_factory=estimators._random_forest_regressor,
        default_hyperparameters={"n_estimators": 20, "n_jobs": 1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
    ),
    "extra_trees_regressor": _AlgorithmFacts(
        "extra_trees_regressor",
        "sklearn",
        estimator_factory=estimators._extra_trees_regressor,
        default_hyperparameters={"n_estimators": 20, "n_jobs": 1},
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 3, "n_jobs": 1},
    ),
    "gradient_boosting_regressor": _AlgorithmFacts(
        "gradient_boosting_regressor",
        "sklearn",
        estimator_factory=estimators._gradient_boosting_regressor,
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2},
    ),
    "hist_gradient_boosting_regressor": _AlgorithmFacts(
        "hist_gradient_boosting_regressor",
        "sklearn",
        estimator_factory=estimators._hist_gradient_boosting_regressor,
        smoke_hyperparameters={"max_iter": 5, "max_leaf_nodes": 7},
    ),
    "xgboost_regressor": _AlgorithmFacts(
        "xgboost_regressor",
        "xgboost",
        estimator_factory=estimators._xgboost_regressor,
        default_hyperparameters={
            "n_estimators": 20,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "device": "cpu",
            "n_jobs": 1,
        },
        smoke_hyperparameters={"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
    ),
    "lightgbm_regressor": _AlgorithmFacts(
        "lightgbm_regressor",
        "lightgbm",
        estimator_factory=estimators._lightgbm_regressor,
        default_hyperparameters={
            "n_estimators": 20,
            "learning_rate": 0.1,
            "verbosity": -1,
            "n_jobs": 1,
        },
        smoke_hyperparameters={
            "n_estimators": 5,
            "num_leaves": 7,
            "n_jobs": 1,
            "verbosity": -1,
        },
    ),
    "single_factor_baseline": _AlgorithmFacts(
        "single_factor_baseline",
        estimator_factory=estimators._single_factor_baseline,
        input_kind="single_factor_input",
    ),
    "equal_weight_rank_ensemble": _AlgorithmFacts(
        "equal_weight_rank_ensemble",
        estimator_factory=estimators._equal_weight_rank_ensemble,
        input_kind="rank_ensemble_inputs",
    ),
    "ic_weighted_rank_ensemble": _AlgorithmFacts(
        "ic_weighted_rank_ensemble",
        estimator_factory=estimators._ic_weighted_rank_ensemble,
        input_kind="rank_ensemble_inputs",
    ),
}


def model_algorithm_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ALGORITHM_FACTS))


def model_algorithm_definition(algorithm: str) -> _AlgorithmFacts:
    return _algorithm_facts(algorithm)


def model_default_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(_algorithm_facts(algorithm).default_hyperparameters)


def model_smoke_hyperparameters(algorithm: str) -> dict[str, object]:
    return dict(_algorithm_facts(algorithm).smoke_hyperparameters)


def model_dependency_status(algorithm: str) -> dict[str, str]:
    facts = _algorithm_facts(algorithm)
    package = facts.dependency_package
    if package is None:
        return {"status": "available", "package": "internal"}
    try:
        import_module(package)
    except ImportError:
        return {"status": "missing", "package": package}
    return {"status": "available", "package": package}


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    facts = _algorithm_facts(spec.model_algorithm)
    _dependency_status(facts.algorithm)
    if facts.estimator_factory is None:
        raise ValueError("unsupported_model_algorithm")
    params = dict(facts.default_hyperparameters)
    params.update(spec.hyperparameters)
    return facts.estimator_factory(spec, params)


def fit_model_result(frame: pd.DataFrame, spec: ModelSpec) -> ModelFitResult:
    return _fit_model_result_from_frames((frame,), spec)


def _fit_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    estimator = build_estimator_from_model_spec(spec)
    fit_model_result_from_frames = getattr(
        estimator,
        "fit_model_result_from_frames",
        None,
    )
    if callable(fit_model_result_from_frames):
        return fit_model_result_from_frames(frames, spec)
    return _fit_estimator_model_result_from_frames(
        frames,
        spec,
        estimator=estimator,
    )


def _fit_linear_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    if spec.max_train_rows is not None or spec.purge_gap_steps > 0:
        return _fit_controlled_linear_model_result_from_frames(frames, spec)
    input_columns = spec.input_columns
    required = set(input_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                *input_columns,
                spec.label_column,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    fit_columns = tuple(
        column
        for column in ("timestamp", *input_columns, spec.label_column)
        if column in work_columns
    )
    beta, row_counts, filter_metadata = _fit_linear_coefficients(
        _iter_model_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        spec=spec,
        fit_columns=fit_columns,
    )
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    metadata = _model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=int(len(beta)),
        feature_importance=feature_importance.linear_feature_importance(spec, beta),
    )
    return ModelFitResult(coefficients=beta, metadata=metadata)


def _model_metadata(
    spec: ModelSpec,
    *,
    filter_metadata: dict[str, object],
    row_counts: dict[str, object],
    coefficient_count: int,
    feature_importance: dict[str, float] | None = None,
    feature_importance_status: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "contract_version": spec.contract_version,
        "model_algorithm": spec.model_algorithm,
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": list(filter_metadata["filter_columns_used"]),
        "filter_definition": str(filter_metadata["filter_definition"]),
        "rows_input": int(filter_metadata["rows_input"]),
        "rows_after_filter_mode": int(filter_metadata["rows_after_filter_mode"]),
        "rows_after_input_label_dropna": int(
            row_counts["rows_after_input_label_dropna"]
        ),
        "train_start": row_counts["train_start"],
        "train_end": row_counts["train_end"],
        "test_start": None,
        "test_end": None,
        "oos_start": None,
        "oos_end": None,
        "input_factor_columns": list(spec.input_factor_columns),
        "input_feature_columns": list(spec.input_feature_columns),
        "input_columns": list(spec.input_columns),
        "input_count": len(spec.input_columns),
        "label_column": spec.label_column,
        "hyperparameters": dict(spec.hyperparameters),
        "random_seed": spec.random_seed,
        "max_train_rows": spec.max_train_rows,
        "purge_gap_steps": spec.purge_gap_steps,
        "dependency_status": _dependency_status(spec.model_algorithm),
        "coefficient_count": coefficient_count,
    }
    if feature_importance is not None:
        metadata["feature_importance"] = feature_importance
    else:
        metadata["feature_importance_status"] = (
            feature_importance_status or "not_available_for_model_algorithm"
        )
    return metadata


def _dependency_status(algorithm: str) -> dict[str, str]:
    status = model_dependency_status(algorithm)
    if status["status"] == "missing":
        raise ImportError(f"model_dependency_unavailable: {status['package']}")
    return status


def _algorithm_facts(algorithm: str) -> _AlgorithmFacts:
    try:
        return _MODEL_ALGORITHM_FACTS[algorithm]
    except KeyError as exc:
        raise ValueError("unsupported_model_algorithm") from exc


def _fit_controlled_linear_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    training, row_counts, filter_metadata = _collect_training_frame(frames, spec)
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    beta = _fit_linear_coefficients_exact(training, spec)
    metadata = _model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=int(len(beta)),
        feature_importance=feature_importance.linear_feature_importance(spec, beta),
    )
    return ModelFitResult(coefficients=beta, metadata=metadata)


def _fit_linear_coefficients(
    frames: Iterable[pd.DataFrame],
    *,
    spec: ModelSpec,
    fit_columns: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    input_count = len(spec.input_columns)
    chunk_count = 0
    row_count = 0
    rows_input = 0
    rows_after_filter = 0
    first_usable: pd.DataFrame | None = None
    filter_columns_used: list[str] = []
    filter_definition = ""
    r_factor: np.ndarray | None = None
    qt_y: np.ndarray | None = None
    train_start: str | None = None
    train_end: str | None = None

    for chunk in iter_complete_time_sized_frames(
        iter(frames),
        timestamp_column="timestamp",
        axis=ChunkAxis.TIME,
    ):
        chunk_count += 1
        filtered, metadata = apply_filter_mode(
            chunk,
            mode=spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        rows_input += int(metadata["rows_input"])
        rows_after_filter += int(metadata["rows_after_filter_mode"])
        filter_columns_used = list(metadata["filter_columns_used"])
        filter_definition = str(metadata["filter_definition"])
        usable = filtered.dropna(
            subset=[*spec.input_columns, spec.label_column]
        ).loc[:, list(fit_columns)]
        if first_usable is None:
            first_usable = usable
        elif not first_usable.empty:
            first_usable = pd.DataFrame()
        y = usable.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
        row_count += int(len(y))
        if len(y) == 0:
            continue
        usable_timestamps = pd.to_datetime(usable["timestamp"], errors="coerce")
        chunk_start = _date_string(usable_timestamps.min())
        chunk_end = _date_string(usable_timestamps.max())
        train_start = (
            chunk_start
            if train_start is None
            else min(train_start, chunk_start)
        )
        train_end = (
            chunk_end
            if train_end is None
            else max(train_end, chunk_end)
        )
        chunk_design = _linear_design_matrix(usable, spec)
        if r_factor is None or qt_y is None:
            stacked_design = chunk_design
            stacked_y = y
        else:
            stacked_design = np.vstack([r_factor, chunk_design])
            stacked_y = np.concatenate([qt_y, y])
        q_factor, r_factor = np.linalg.qr(stacked_design, mode="reduced")
        qt_y = q_factor.T @ stacked_y

    row_counts = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "rows_after_input_label_dropna": row_count,
        "train_start": train_start,
        "train_end": train_end,
    }
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    if row_count == 0:
        return np.empty(input_count + 1, dtype="float64"), row_counts, filter_metadata
    if chunk_count == 1 and first_usable is not None:
        return (
            _fit_linear_coefficients_exact(first_usable, spec),
            row_counts,
            filter_metadata,
        )
    if r_factor is None or qt_y is None:
        return np.empty(input_count + 1, dtype="float64"), row_counts, filter_metadata
    return (
        _fit_linear_coefficients_from_tsqr(r_factor, qt_y),
        row_counts,
        filter_metadata,
    )


def _fit_estimator_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
    *,
    estimator: object | None = None,
) -> ModelFitResult:
    training, row_counts, filter_metadata = _collect_training_frame(frames, spec)
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    y = training.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
    predictor = _fit_predictor(spec, training=training, y=y, estimator=estimator)
    coefficient_count = int(
        getattr(predictor, "coefficient_count", len(spec.input_columns))
    )
    metadata = _model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=coefficient_count,
        feature_importance=feature_importance.estimator_feature_importance(
            spec,
            predictor,
        ),
    )
    coefficients = np.asarray(
        getattr(predictor, "coefficients", np.full(coefficient_count, np.nan)),
        dtype="float64",
    )
    return ModelFitResult(
        coefficients=coefficients,
        metadata=metadata,
        prediction_model=predictor,
    )


def _collect_training_frame(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    if spec.max_train_rows is None:
        raise ValueError("unbounded_model_training_collection")
    required = set(spec.input_columns)
    required.add(spec.label_column)
    required.add("timestamp")
    work_columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                *spec.input_columns,
                spec.label_column,
                *required_filter_columns(spec.sample_filter_mode),
            )
        )
    )
    rows_input = 0
    rows_after_filter = 0
    filter_columns_used: list[str] = []
    filter_definition = ""
    capped_training: pd.DataFrame | None = None
    for chunk in iter_complete_time_sized_frames(
        _iter_model_work_frames(
            frames,
            required=required,
            work_columns=work_columns,
        ),
        timestamp_column="timestamp",
        axis=ChunkAxis.TIME,
    ):
        filtered, metadata = apply_filter_mode(
            chunk,
            mode=spec.sample_filter_mode,
            metadata_key="sample_filter_mode",
        )
        rows_input += int(metadata["rows_input"])
        rows_after_filter += int(metadata["rows_after_filter_mode"])
        filter_columns_used = list(metadata["filter_columns_used"])
        filter_definition = str(metadata["filter_definition"])
        usable = filtered.dropna(subset=[*spec.input_columns, spec.label_column]).loc[
            :, ["timestamp", *spec.input_columns, spec.label_column]
        ]
        if not usable.empty:
            capped_training = validation_split.trim_training_buffer(
                usable
                if capped_training is None
                else pd.concat([capped_training, usable], ignore_index=True),
                spec,
            )
    training = capped_training if capped_training is not None else pd.DataFrame()
    training = validation_split.apply_training_controls(training, spec)
    train_start = None
    train_end = None
    if not training.empty:
        timestamps = pd.to_datetime(training["timestamp"], errors="coerce")
        train_start = _date_string(timestamps.min())
        train_end = _date_string(timestamps.max())
    row_counts = {
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
        "rows_after_input_label_dropna": int(len(training)),
        "train_start": train_start,
        "train_end": train_end,
    }
    filter_metadata = {
        "sample_filter_mode": spec.sample_filter_mode,
        "filter_columns_used": filter_columns_used,
        "filter_definition": filter_definition,
        "rows_input": rows_input,
        "rows_after_filter_mode": rows_after_filter,
    }
    return training, row_counts, filter_metadata


def _fit_predictor(
    spec: ModelSpec,
    *,
    training: pd.DataFrame,
    y: np.ndarray,
    estimator: object | None = None,
) -> object:
    if estimator is None:
        estimator = build_estimator_from_model_spec(spec)
    fit_training_frame = getattr(estimator, "fit_training_frame", None)
    if callable(fit_training_frame):
        return fit_training_frame(training)
    if not hasattr(estimator, "fit"):
        return estimator
    estimator.fit(training.loc[:, spec.input_columns], y)
    return estimator


def _fit_linear_coefficients_exact(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    design = _linear_design_matrix(frame, spec)
    y = frame.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta


def _iter_model_work_frames(
    frames: Iterable[pd.DataFrame],
    *,
    required: set[str],
    work_columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for frame in frames:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"missing_model_columns: {', '.join(missing)}")
        selected = [column for column in work_columns if column in frame.columns]
        yield frame.loc[:, selected]


def _linear_design_matrix(
    frame: pd.DataFrame,
    spec: ModelSpec,
) -> np.ndarray:
    x = frame.loc[:, spec.input_columns].to_numpy(dtype="float64", copy=False)
    design = np.empty((len(x), len(spec.input_columns) + 1), dtype="float64")
    design[:, 0] = 1.0
    design[:, 1:] = x
    return design


def _fit_linear_coefficients_from_tsqr(
    r_factor: np.ndarray,
    qt_y: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(r_factor, qt_y, rcond=None)
    return beta


def _date_string(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


__all__ = [
    "ModelFitResult",
    "build_estimator_from_model_spec",
    "fit_model_result",
    "predict_model_frame",
]
