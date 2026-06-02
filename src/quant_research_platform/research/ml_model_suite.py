"""Local non-DL N6 model suite for A-share prediction handoffs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge

from quant_research_platform.markets.implementations.a_share.research_batch_safety import (  # noqa: E501
    write_json_sanitized,
)
from quant_research_platform.runtime.private_paths import (
    require_private_outputs_for_private_input,
)

A_SHARE_MODEL_SUITE_ID = "n6_a_share_sklearn_boosting_suite_v1"
PREDICTION_HANDOFF_VERSION = "n6_a_share_prediction_handoff_v1"
DEFAULT_RESEARCH_RANDOM_SEED = 42
DEFAULT_NON_DL_MODEL_IDS: tuple[str, ...] = (
    "single_factor_baselines",
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
    "ridge",
    "elastic_net",
    "huber",
    "random_forest_lite",
    "extra_trees_lite",
    "hist_gradient_boosting",
    "xgboost_lite",
    "lightgbm_lite",
)
MODEL_BOUNDARY = (
    "learned prediction signal only; not formula factor; not model registry; "
    "not feature store; not deep learning; not investment advice; not trading, "
    "not risk, not capacity, not execution simulation, not paper trading, "
    "and not live trading evidence"
)
_SENSITIVE_IDENTIFIER_PATTERN = re.compile(
    r"(credential|secret|token|api[_-]?key|password)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SuiteFoldConfig:
    """One walk-forward OOS split for the local model suite."""

    fold_id: str
    train_end: str
    test_start: str
    test_end: str | None = None
    split_name: str = "walk_forward"


@dataclass(frozen=True)
class ModelSuiteConfig:
    """Input/output config for the local non-DL model suite."""

    input_path: Path
    output_prediction_path: Path
    output_metadata_path: Path
    output_prediction_long_path: Path | None
    timestamp_column: str
    entity_column: str
    feature_columns: tuple[str, ...]
    label_column: str
    label_horizon_days: int
    folds: tuple[SuiteFoldConfig, ...]
    model_ids: tuple[str, ...] = DEFAULT_NON_DL_MODEL_IDS
    model_suite_id: str = A_SHARE_MODEL_SUITE_ID
    random_seed: int = DEFAULT_RESEARCH_RANDOM_SEED


@dataclass(frozen=True)
class ModelSuiteResult:
    """Small summary of generated suite outputs."""

    prediction_path: Path
    metadata_path: Path
    model_suite_id: str
    prediction_columns: tuple[str, ...]
    row_count: int


def run_model_suite(config: ModelSuiteConfig) -> ModelSuiteResult:
    """Train configured non-DL models and write OOS prediction columns."""

    _validate_config(config)
    frame = pd.read_parquet(
        config.input_path,
        columns=_read_columns(config),
        engine="pyarrow",
    )
    frame = frame.sort_values([config.timestamp_column, config.entity_column]).copy()
    timestamps = pd.to_datetime(frame[config.timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{config.timestamp_column} must contain valid datetimes")

    prediction_frames: list[pd.DataFrame] = []
    metadata_folds: list[dict[str, Any]] = []
    model_ids = config.model_ids or DEFAULT_NON_DL_MODEL_IDS
    prediction_columns = _prediction_columns(model_ids, config)
    model_statuses: list[dict[str, Any]] = []
    for fold in config.folds:
        split = _split_for_fold(frame, timestamps, config, fold)
        train = split["train"].dropna(
            subset=(*config.feature_columns, config.label_column)
        )
        test = split["test"].dropna(subset=config.feature_columns)
        if train.empty or test.empty:
            raise ValueError("insufficient_sample")

        predictions = test[
            [config.timestamp_column, config.entity_column, config.label_column]
        ].copy()
        predictions["fold_id"] = fold.fold_id
        predictions["split_name"] = fold.split_name
        predictions["target_horizon"] = _target_horizon(config.label_column)
        for model_id in model_ids:
            status = _add_model_predictions(
                predictions=predictions,
                train=train,
                test=test,
                config=config,
                model_id=model_id,
                random_seed=config.random_seed,
            )
            model_statuses.extend(
                {
                    **item,
                    "fold_id": fold.fold_id,
                    "split_name": fold.split_name,
                }
                for item in status
            )
        prediction_frames.append(predictions)
        metadata_folds.append(
            {
                "fold_id": fold.fold_id,
                "split_name": fold.split_name,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "purged_train_end": split["purged_train_end"].date().isoformat(),
                "train_row_count": int(len(train)),
                "test_row_count": int(len(test)),
            }
        )

    output = pd.concat(prediction_frames, ignore_index=True)
    ready_prediction_columns = tuple(
        column
        for column in prediction_columns
        if column in output.columns and output[column].notna().any()
    )
    config.output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(
        config.output_prediction_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    if config.output_prediction_long_path is not None:
        _write_long_predictions(
            output,
            config=config,
            prediction_columns=ready_prediction_columns,
        )
    _write_metadata(
        config,
        prediction_columns=ready_prediction_columns,
        folds=metadata_folds,
        model_statuses=model_statuses,
    )
    return ModelSuiteResult(
        prediction_path=config.output_prediction_path,
        metadata_path=config.output_metadata_path,
        model_suite_id=config.model_suite_id,
        prediction_columns=ready_prediction_columns,
        row_count=len(output),
    )


def _validate_config(config: ModelSuiteConfig) -> None:
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    if not config.folds:
        raise ValueError("invalid_prediction_config")
    if config.label_horizon_days < 0:
        raise ValueError("invalid_prediction_config")
    expected_horizon = _horizon_days_from_label(config.label_column)
    if expected_horizon is not None and expected_horizon != config.label_horizon_days:
        raise ValueError("invalid_prediction_config")
    if not config.feature_columns:
        raise ValueError("invalid_feature_preset")
    leakage = [
        column
        for column in config.feature_columns
        if _looks_like_label_or_forward_return(column)
    ]
    if leakage or config.label_column in config.feature_columns:
        raise ValueError("invalid_feature_preset")
    if any(
        feature_name_has_sensitive_identifier(column)
        for column in config.feature_columns
    ):
        raise ValueError("invalid_feature_preset")
    forbidden_models = {"pytorch", "tensorflow", "deep_learning"}
    if any(model_id in forbidden_models for model_id in config.model_ids):
        raise ValueError("dependency_unavailable")
    output_paths = [config.output_prediction_path, config.output_metadata_path]
    if config.output_prediction_long_path is not None:
        output_paths.append(config.output_prediction_long_path)
    require_private_outputs_for_private_input(
        config.input_path,
        tuple(output_paths),
        error_message="private_output_policy_violation",
    )
    validate_prediction_folds(config.folds)


def _read_columns(config: ModelSuiteConfig) -> list[str]:
    return list(
        dict.fromkeys(
            (
                config.timestamp_column,
                config.entity_column,
                config.label_column,
                *config.feature_columns,
            )
        )
    )


def _split_for_fold(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    config: ModelSuiteConfig,
    fold: SuiteFoldConfig,
) -> dict[str, Any]:
    train_end = pd.Timestamp(fold.train_end)
    test_start = pd.Timestamp(fold.test_start)
    test_end = pd.Timestamp(fold.test_end) if fold.test_end else None
    calendar_purged_train_end = _purged_train_end_from_trading_calendar(
        timestamps,
        test_start=test_start,
        label_horizon_days=config.label_horizon_days,
    )
    purged_train_end = min(train_end, calendar_purged_train_end)
    train = frame.loc[timestamps <= purged_train_end].copy()
    test_mask = timestamps >= test_start
    if test_end is not None:
        test_mask &= timestamps <= test_end
    test = frame.loc[test_mask].copy()
    return {
        "train": train,
        "test": test,
        "purged_train_end": pd.Timestamp(purged_train_end),
    }


def _validate_non_null(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing_prediction_field: {missing}")
    if frame[list(columns)].isna().any().any():
        raise ValueError("insufficient_sample")


def validate_prediction_folds(
    folds: tuple[SuiteFoldConfig, ...],
) -> tuple[SuiteFoldConfig, ...]:
    """Validate walk-forward folds before any row-level training or writes."""

    if not folds:
        raise ValueError("invalid_prediction_config")
    seen_fold_ids: set[str] = set()
    seen_windows: set[tuple[pd.Timestamp, pd.Timestamp | None]] = set()
    previous_start: pd.Timestamp | None = None
    previous_end: pd.Timestamp | None = None
    for idx, fold in enumerate(folds):
        fold_id = _required_fold_text(fold.fold_id)
        train_end = _parse_fold_timestamp(fold.train_end)
        test_start = _parse_fold_timestamp(fold.test_start)
        test_end = (
            None
            if fold.test_end is None or str(fold.test_end).strip() == ""
            else _parse_fold_timestamp(fold.test_end)
        )
        if fold_id in seen_fold_ids:
            raise ValueError("invalid_prediction_config")
        seen_fold_ids.add(fold_id)
        if train_end >= test_start:
            raise ValueError("invalid_prediction_config")
        if test_end is None and idx != len(folds) - 1:
            raise ValueError("invalid_prediction_config")
        if test_end is not None and test_start > test_end:
            raise ValueError("invalid_prediction_config")
        window = (test_start, test_end)
        if window in seen_windows:
            raise ValueError("invalid_prediction_config")
        seen_windows.add(window)
        if previous_start is not None and test_start <= previous_start:
            raise ValueError("invalid_prediction_config")
        if previous_end is not None and test_start <= previous_end:
            raise ValueError("invalid_prediction_config")
        previous_start = test_start
        previous_end = test_end
    return folds


def _required_fold_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_prediction_config")
    return value.strip()


def _parse_fold_timestamp(value: object) -> pd.Timestamp:
    text = _required_fold_text(value)
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError):
        raise ValueError("invalid_prediction_config") from None
    if pd.isna(timestamp):
        raise ValueError("invalid_prediction_config")
    return timestamp


def _purged_train_end_from_trading_calendar(
    timestamps: pd.Series,
    *,
    test_start: pd.Timestamp,
    label_horizon_days: int,
) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(timestamps.dropna()).dt.normalize().unique())
    dates = dates.sort_values(ignore_index=True)
    first_test_position = int(dates.searchsorted(test_start.normalize(), side="left"))
    # Labels use execution rows t+1 through t+h+1. Keep only rows whose label
    # exit date is strictly before the first OOS test date.
    max_train_position = first_test_position - label_horizon_days - 2
    if max_train_position < 0:
        raise ValueError("insufficient_sample")
    return pd.Timestamp(dates.iloc[max_train_position])


def _make_model(model_id: str, *, random_seed: int):
    if model_id == "ridge":
        return Ridge(alpha=1.0)
    if model_id == "elastic_net":
        return ElasticNet(
            alpha=0.001,
            l1_ratio=0.2,
            max_iter=1000,
            random_state=random_seed,
        )
    if model_id == "huber":
        return HuberRegressor(max_iter=100)
    if model_id == "random_forest_lite":
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=random_seed,
            n_jobs=1,
        )
    if model_id == "extra_trees_lite":
        return ExtraTreesRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=random_seed,
            n_jobs=1,
        )
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=50,
            max_depth=4,
            random_state=random_seed,
        )
    if model_id == "xgboost_lite":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            seed=random_seed,
            objective="reg:squarederror",
            n_jobs=1,
            nthread=1,
        )
    if model_id == "lightgbm_lite":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_seed,
            seed=random_seed,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(f"dependency_unavailable: {model_id}")


def _prediction_columns(
    model_ids: tuple[str, ...],
    config: ModelSuiteConfig,
) -> tuple[str, ...]:
    columns: list[str] = []
    for model_id in model_ids:
        if model_id == "single_factor_baselines":
            columns.extend(
                _prediction_column(f"single_factor_{feature}", config.label_column)
                for feature in config.feature_columns
            )
        else:
            columns.append(_prediction_column(model_id, config.label_column))
    return tuple(dict.fromkeys(columns))


def _prediction_column(model_id: str, label_column: str) -> str:
    horizon = _target_horizon(label_column)
    return f"prediction_score_{model_id}_{horizon}"


def _target_horizon(label_column: str) -> str:
    prefix = "forward_return_"
    if label_column.startswith(prefix):
        return label_column.removeprefix(prefix)
    return "label"


def _horizon_days_from_label(label_column: str) -> int | None:
    match = re.fullmatch(r"forward_return_(\d+)d", label_column)
    if match is None:
        return None
    return int(match.group(1))


def _add_model_predictions(
    *,
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelSuiteConfig,
    model_id: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    if model_id == "single_factor_baselines":
        return _add_single_factor_predictions(
            predictions=predictions,
            test=test,
            config=config,
        )
    if model_id == "equal_weight_rank_ensemble":
        column = _prediction_column(model_id, config.label_column)
        predictions[column] = _rank_ensemble(
            test,
            config.feature_columns,
            timestamp_column=config.timestamp_column,
        )
        return [{"model_id": model_id, "status": "ready", "prediction_column": column}]
    if model_id == "ic_weighted_rank_ensemble":
        column = _prediction_column(model_id, config.label_column)
        predictions[column] = _rank_ensemble(
            test,
            config.feature_columns,
            timestamp_column=config.timestamp_column,
            weights=_ic_weights(train, config),
        )
        return [{"model_id": model_id, "status": "ready", "prediction_column": column}]
    if model_id == "huber" and len(train) < max(20, len(config.feature_columns) + 2):
        return [{"model_id": model_id, "status": "skipped_insufficient_sample"}]
    column = _prediction_column(model_id, config.label_column)
    dependency = _dependency_status(model_id)
    if dependency["status"] != "available":
        return [
            {
                "model_id": model_id,
                "status": "dependency_unavailable",
                "reason": dependency["reason"],
            }
        ]
    try:
        model = _make_model(model_id, random_seed=random_seed)
        model.fit(train[list(config.feature_columns)], train[config.label_column])
        predictions[column] = model.predict(test[list(config.feature_columns)])
    except ImportError as exc:
        return [
            {
                "model_id": model_id,
                "status": "dependency_unavailable",
                "reason": str(exc),
            }
        ]
    except ValueError as exc:
        return [{"model_id": model_id, "status": "fit_failed", "reason": str(exc)}]
    payload: dict[str, Any] = {
        "model_id": model_id,
        "status": "ready",
        "prediction_column": column,
    }
    importances = _feature_importances(model, config.feature_columns)
    if importances:
        payload["feature_importances"] = importances
    return [payload]


def _add_single_factor_predictions(
    *,
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    config: ModelSuiteConfig,
) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    for feature in config.feature_columns:
        column = _prediction_column(f"single_factor_{feature}", config.label_column)
        predictions[column] = test[feature].to_numpy()
        statuses.append(
            {
                "model_id": f"single_factor_{feature}",
                "status": "ready",
                "prediction_column": column,
            }
        )
    return statuses


def _rank_ensemble(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    timestamp_column: str,
    weights: pd.Series | None = None,
) -> pd.Series:
    ranks = frame.groupby(timestamp_column)[list(feature_columns)].rank(pct=True)
    if weights is None:
        return ranks.mean(axis=1)
    aligned = weights.reindex(feature_columns).fillna(0.0)
    denominator = aligned.abs().sum()
    if denominator == 0.0:
        return ranks.mean(axis=1)
    return ranks.mul(aligned / denominator, axis=1).sum(axis=1)


def _ic_weights(train: pd.DataFrame, config: ModelSuiteConfig) -> pd.Series:
    weights = {}
    for feature in config.feature_columns:
        valid = train[[feature, config.label_column]].dropna()
        weights[feature] = (
            valid[feature].rank().corr(valid[config.label_column].rank())
            if len(valid) > 1
            else 0.0
        )
    return (
        pd.Series(weights, dtype="float64")
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
    )


def _dependency_status(model_id: str) -> dict[str, str]:
    package = None
    if model_id == "xgboost_lite":
        package = "xgboost"
    elif model_id == "lightgbm_lite":
        package = "lightgbm"
    if package is None:
        return {"status": "available", "reason": "bundled_or_sklearn"}
    if importlib_util.find_spec(package) is None:
        return {"status": "missing", "reason": f"{package} import not available"}
    return {"status": "available", "reason": f"{package} import available"}


def _feature_importances(
    model: object,
    feature_columns: tuple[str, ...],
) -> list[dict[str, object]]:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return []
    numeric = np.asarray(values, dtype="float64")
    if len(numeric) != len(feature_columns):
        return []
    return [
        {"feature": feature, "importance": float(value)}
        for feature, value in zip(feature_columns, numeric, strict=False)
    ]


def _write_long_predictions(
    frame: pd.DataFrame,
    *,
    config: ModelSuiteConfig,
    prediction_columns: tuple[str, ...],
) -> None:
    records = frame.melt(
        id_vars=(
            config.timestamp_column,
            config.entity_column,
            config.label_column,
            "fold_id",
            "split_name",
            "target_horizon",
        ),
        value_vars=list(prediction_columns),
        var_name="prediction_column",
        value_name="prediction_score",
    )
    records["model_id"] = records["prediction_column"].map(
        lambda column: _model_id_from_prediction_column(
            str(column),
            config.label_column,
        )
    )
    config.output_prediction_long_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_parquet(
        config.output_prediction_long_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )


def _model_id_from_prediction_column(column: str, label_column: str) -> str:
    prefix = "prediction_score_"
    suffix = f"_{_target_horizon(label_column)}"
    value = column.removeprefix(prefix)
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def _write_metadata(
    config: ModelSuiteConfig,
    *,
    prediction_columns: tuple[str, ...],
    folds: list[dict[str, Any]],
    model_statuses: list[dict[str, Any]],
) -> None:
    payload = {
        "prediction_handoff_version": PREDICTION_HANDOFF_VERSION,
        "model_suite_id": config.model_suite_id,
        "model_ids": list(config.model_ids),
        "requested_model_ids": list(config.model_ids or DEFAULT_NON_DL_MODEL_IDS),
        "prediction_columns": list(prediction_columns),
        "feature_columns": list(config.feature_columns),
        "target_column": config.label_column,
        "label_horizon_days": int(config.label_horizon_days),
        "folds": folds,
        "model_statuses": model_statuses,
        "random_seed": int(config.random_seed),
        "timestamp_column": config.timestamp_column,
        "entity_column": config.entity_column,
        "label_column": config.label_column,
        "boundary": MODEL_BOUNDARY,
    }
    write_json_sanitized(config.output_metadata_path, payload)


def _looks_like_label_or_forward_return(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized in {"label", "target"}
        or normalized.startswith("label_")
        or normalized.startswith("target_")
        or normalized.startswith("forward_return")
    )


def feature_name_looks_like_leakage(column: str) -> bool:
    """Return whether a feature name appears to contain target/label leakage."""

    return _looks_like_label_or_forward_return(column)


def feature_name_has_sensitive_identifier(column: str) -> bool:
    """Return whether a feature name looks like private credential metadata."""

    return bool(_SENSITIVE_IDENTIFIER_PATTERN.search(column))


__all__ = [
    "A_SHARE_MODEL_SUITE_ID",
    "DEFAULT_NON_DL_MODEL_IDS",
    "DEFAULT_RESEARCH_RANDOM_SEED",
    "MODEL_BOUNDARY",
    "ModelSuiteConfig",
    "ModelSuiteResult",
    "PREDICTION_HANDOFF_VERSION",
    "SuiteFoldConfig",
    "feature_name_has_sensitive_identifier",
    "feature_name_looks_like_leakage",
    "run_model_suite",
    "validate_prediction_folds",
]
