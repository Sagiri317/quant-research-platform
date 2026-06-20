"""Deterministic supervised model helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import import_module

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research._memory import iter_complete_time_sized_frames
from quant_research_platform.research.filtering import apply_filter_mode
from quant_research_platform.runtime.resources import ChunkAxis


@dataclass(frozen=True)
class ModelFitResult:
    coefficients: np.ndarray
    metadata: dict[str, object]
    prediction_model: object | None = field(default=None, compare=False, repr=False)


def build_estimator_from_model_spec(spec: ModelSpec) -> object:
    _dependency_status(spec.model_algorithm)
    algorithm = spec.model_algorithm
    if algorithm == "linear_regression":
        from sklearn.linear_model import LinearRegression

        return LinearRegression()
    if algorithm == "single_factor_baseline":
        return _SingleFactorBaseline(spec.input_columns[0])
    if algorithm == "equal_weight_rank_ensemble":
        return _RankEnsemblePredictor(spec.input_columns)
    if algorithm == "ic_weighted_rank_ensemble":
        return _ICWeightedRankEnsembleFactory(spec.input_columns, spec.label_column)
    return _build_regression_estimator(spec)


def fit_model_result(frame: pd.DataFrame, spec: ModelSpec) -> ModelFitResult:
    return _fit_model_result_from_frames((frame,), spec)


def predict_model_frame(
    frame: pd.DataFrame,
    spec: ModelSpec,
    result: ModelFitResult,
) -> pd.DataFrame:
    required = set(spec.input_columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_model_columns: {', '.join(missing)}")
    output = frame.copy()
    output["prediction_score"] = _predict_scores(
        frame,
        coefficients=result.coefficients,
        input_columns=spec.input_columns,
        prediction_model=result.prediction_model,
    )
    return output


def _fit_model_result_from_frames(
    frames: Iterable[pd.DataFrame],
    spec: ModelSpec,
) -> ModelFitResult:
    if spec.model_algorithm != "linear_regression":
        return _fit_estimator_model_result_from_frames(frames, spec)
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
        feature_importance=_linear_feature_importance(spec, beta),
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
    dependency_by_algorithm = {
        "linear_regression": "sklearn",
        "ridge_regression": "sklearn",
        "lasso_regression": "sklearn",
        "elastic_net": "sklearn",
        "huber_regression": "sklearn",
        "decision_tree_regressor": "sklearn",
        "random_forest_regressor": "sklearn",
        "extra_trees_regressor": "sklearn",
        "gradient_boosting_regressor": "sklearn",
        "hist_gradient_boosting_regressor": "sklearn",
        "xgboost_regressor": "xgboost",
        "lightgbm_regressor": "lightgbm",
    }
    package = dependency_by_algorithm.get(algorithm)
    if package is None:
        return {"status": "available", "package": "internal"}
    try:
        import_module(package)
    except ImportError as exc:
        raise ImportError(f"model_dependency_unavailable: {package}") from exc
    return {"status": "available", "package": package}


def _linear_feature_importance(
    spec: ModelSpec,
    coefficients: np.ndarray,
) -> dict[str, float]:
    values = coefficients[1 : len(spec.input_columns) + 1]
    return {
        column: float(abs(value))
        for column, value in zip(spec.input_columns, values, strict=True)
    }


def _estimator_feature_importance(
    spec: ModelSpec,
    predictor: object,
) -> dict[str, float] | None:
    if isinstance(predictor, _SingleFactorBaseline):
        return {predictor.column: 1.0}
    if isinstance(predictor, _RankEnsemblePredictor):
        weight = 1.0 / len(predictor.columns)
        return {column: weight for column in predictor.columns}
    if isinstance(predictor, _ICWeightedRankEnsemble):
        return {
            column: float(abs(weight))
            for column, weight in zip(predictor.columns, predictor.weights, strict=True)
        }
    if hasattr(predictor, "feature_importances_"):
        values = np.asarray(predictor.feature_importances_, dtype="float64")
        return {
            column: float(value)
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    if hasattr(predictor, "coef_"):
        values = np.ravel(np.asarray(predictor.coef_, dtype="float64"))
        return {
            column: float(abs(value))
            for column, value in zip(spec.input_columns, values, strict=True)
        }
    return None


def _predict_scores(
    frame: pd.DataFrame,
    *,
    coefficients: np.ndarray,
    input_columns: tuple[str, ...],
    prediction_model: object | None,
) -> np.ndarray:
    if prediction_model is not None:
        predicted = prediction_model.predict(frame.loc[:, input_columns])
        return np.asarray(predicted, dtype="float64")
    x = frame.loc[:, input_columns].to_numpy(dtype="float64", copy=False)
    return np.asarray(float(coefficients[0]) + x @ coefficients[1:])


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
        feature_importance=_linear_feature_importance(spec, beta),
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
) -> ModelFitResult:
    training, row_counts, filter_metadata = _collect_training_frame(frames, spec)
    if row_counts["rows_after_input_label_dropna"] == 0:
        raise ValueError("model_training_sample_empty")
    y = training.loc[:, spec.label_column].to_numpy(dtype="float64", copy=False)
    predictor = _fit_predictor(spec, training=training, y=y)
    coefficient_count = int(
        getattr(predictor, "coefficient_count", len(spec.input_columns))
    )
    metadata = _model_metadata(
        spec,
        filter_metadata=filter_metadata,
        row_counts=row_counts,
        coefficient_count=coefficient_count,
        feature_importance=_estimator_feature_importance(spec, predictor),
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
            capped_training = _trim_training_buffer(
                usable
                if capped_training is None
                else pd.concat([capped_training, usable], ignore_index=True),
                spec,
            )
    training = capped_training if capped_training is not None else pd.DataFrame()
    training = _apply_training_controls(training, spec)
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


def _trim_training_buffer(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if spec.max_train_rows is None or frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if spec.purge_gap_steps <= 0:
        return result.tail(spec.max_train_rows).reset_index(drop=True)
    unique_timestamps = tuple(sorted(result["timestamp"].dropna().unique()))
    if spec.purge_gap_steps >= len(unique_timestamps):
        return result
    purge_start = unique_timestamps[-spec.purge_gap_steps]
    training_candidates = result.loc[result["timestamp"] < purge_start]
    purge_gap_rows = result.loc[result["timestamp"] >= purge_start]
    return pd.concat(
        [training_candidates.tail(spec.max_train_rows), purge_gap_rows],
        ignore_index=True,
    )


def _apply_training_controls(frame: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if spec.purge_gap_steps > 0:
        unique_timestamps = tuple(sorted(result["timestamp"].dropna().unique()))
        if spec.purge_gap_steps >= len(unique_timestamps):
            result = result.iloc[0:0]
        else:
            cutoff = unique_timestamps[-spec.purge_gap_steps]
            result = result.loc[result["timestamp"] < cutoff]
    if spec.max_train_rows is not None and len(result) > spec.max_train_rows:
        result = result.tail(spec.max_train_rows)
    return result.reset_index(drop=True)


def _fit_predictor(
    spec: ModelSpec,
    *,
    training: pd.DataFrame,
    y: np.ndarray,
) -> object:
    algorithm = spec.model_algorithm
    if algorithm == "single_factor_baseline":
        return _SingleFactorBaseline(spec.input_columns[0])
    if algorithm == "equal_weight_rank_ensemble":
        return _RankEnsemblePredictor(spec.input_columns)
    if algorithm == "ic_weighted_rank_ensemble":
        return _ICWeightedRankEnsemble.fit(
            training,
            input_columns=spec.input_columns,
            label_column=spec.label_column,
        )
    estimator = _build_regression_estimator(spec)
    estimator.fit(training.loc[:, spec.input_columns], y)
    return estimator


def _build_regression_estimator(spec: ModelSpec) -> object:
    _dependency_status(spec.model_algorithm)
    params = dict(spec.hyperparameters)
    if spec.model_algorithm == "ridge_regression":
        from sklearn.linear_model import Ridge

        params.setdefault("alpha", 1.0)
        return Ridge(**params)
    if spec.model_algorithm == "lasso_regression":
        from sklearn.linear_model import Lasso

        params.setdefault("alpha", 0.001)
        params.setdefault("max_iter", 10_000)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return Lasso(**params)
    if spec.model_algorithm == "elastic_net":
        from sklearn.linear_model import ElasticNet

        params.setdefault("alpha", 0.001)
        params.setdefault("l1_ratio", 0.5)
        params.setdefault("max_iter", 10_000)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return ElasticNet(**params)
    if spec.model_algorithm == "huber_regression":
        from sklearn.linear_model import HuberRegressor

        params.setdefault("max_iter", 1_000)
        return HuberRegressor(**params)
    if spec.model_algorithm == "decision_tree_regressor":
        from sklearn.tree import DecisionTreeRegressor

        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return DecisionTreeRegressor(**params)
    if spec.model_algorithm == "random_forest_regressor":
        from sklearn.ensemble import RandomForestRegressor

        params.setdefault("n_estimators", 20)
        params.setdefault("n_jobs", 1)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return RandomForestRegressor(**params)
    if spec.model_algorithm == "extra_trees_regressor":
        from sklearn.ensemble import ExtraTreesRegressor

        params.setdefault("n_estimators", 20)
        params.setdefault("n_jobs", 1)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return ExtraTreesRegressor(**params)
    if spec.model_algorithm == "gradient_boosting_regressor":
        from sklearn.ensemble import GradientBoostingRegressor

        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return GradientBoostingRegressor(**params)
    if spec.model_algorithm == "hist_gradient_boosting_regressor":
        from sklearn.ensemble import HistGradientBoostingRegressor

        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return HistGradientBoostingRegressor(**params)
    if spec.model_algorithm == "xgboost_regressor":
        from xgboost import XGBRegressor

        params.setdefault("n_estimators", 20)
        params.setdefault("max_depth", 3)
        params.setdefault("learning_rate", 0.1)
        params.setdefault("objective", "reg:squarederror")
        params.setdefault("device", "cpu")
        params.setdefault("n_jobs", 1)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return XGBRegressor(**params)
    if spec.model_algorithm == "lightgbm_regressor":
        from lightgbm import LGBMRegressor

        params.setdefault("n_estimators", 20)
        params.setdefault("learning_rate", 0.1)
        params.setdefault("verbosity", -1)
        params.setdefault("n_jobs", 1)
        if spec.random_seed is not None:
            params.setdefault("random_state", spec.random_seed)
        return LGBMRegressor(**params)
    raise ValueError("unsupported_model_algorithm")


@dataclass(frozen=True)
class _SingleFactorBaseline:
    column: str
    coefficient_count: int = 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, self.column].to_numpy(dtype="float64", copy=False)


@dataclass(frozen=True)
class _RankEnsemblePredictor:
    columns: tuple[str, ...]
    coefficient_count: int = 1

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return _rank_ensemble_scores(frame, self.columns)


@dataclass(frozen=True)
class _ICWeightedRankEnsembleFactory:
    columns: tuple[str, ...]
    label_column: str


@dataclass(frozen=True)
class _ICWeightedRankEnsemble:
    columns: tuple[str, ...]
    weights: np.ndarray
    coefficient_count: int

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        input_columns: tuple[str, ...],
        label_column: str,
    ) -> _ICWeightedRankEnsemble:
        weights = []
        label = _rank_series(frame, label_column)
        for column in input_columns:
            corr = _rank_series(frame, column).corr(label, method="pearson")
            weights.append(0.0 if pd.isna(corr) else float(corr))
        values = np.asarray(weights, dtype="float64")
        if np.allclose(values, 0.0):
            values = np.ones(len(input_columns), dtype="float64")
        return cls(
            columns=input_columns,
            weights=values / np.sum(np.abs(values)),
            coefficient_count=len(input_columns),
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        ranks = _rank_frame_columns(frame, self.columns)
        return ranks @ self.weights


def _rank_ensemble_scores(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    ranks = _rank_frame_columns(frame, columns)
    return ranks.mean(axis=1)


def _rank_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if "timestamp" in frame:
        ranked = frame.groupby("timestamp", sort=False)[list(columns)].rank(
            method="average",
            pct=True,
        )
    else:
        ranked = frame.loc[:, columns].rank(method="average", pct=True)
    return ranked.to_numpy(dtype="float64", copy=False)


def _rank_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if "timestamp" in frame:
        return frame.groupby("timestamp", sort=False)[column].rank(
            method="average",
            pct=True,
        )
    return frame[column].rank(method="average", pct=True)


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
