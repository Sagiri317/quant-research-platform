"""Local non-DL N6 model suite for A-share prediction handoffs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge

from quant_research_platform.runtime.private_paths import (
    require_private_outputs_for_private_input,
)

A_SHARE_MODEL_SUITE_ID = "n6_a_share_sklearn_boosting_suite_v1"
PREDICTION_HANDOFF_VERSION = "n6_a_share_prediction_handoff_v1"
MODEL_BOUNDARY = (
    "learned prediction signal only; not formula factor; not model registry; "
    "not feature store; not deep learning; not investment advice; not trading, "
    "not risk, not capacity, not execution simulation, not paper trading, "
    "and not live trading evidence"
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
    timestamp_column: str
    entity_column: str
    feature_columns: tuple[str, ...]
    label_column: str
    label_horizon_days: int
    folds: tuple[SuiteFoldConfig, ...]
    model_ids: tuple[str, ...] = (
        "ridge",
        "elastic_net",
        "huber",
        "random_forest_lite",
        "extra_trees_lite",
        "hist_gradient_boosting",
    )
    model_suite_id: str = A_SHARE_MODEL_SUITE_ID
    random_seed: int = 42


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
    prediction_columns = tuple(
        _prediction_column(model_id, config.label_column)
        for model_id in config.model_ids
    )
    for fold in config.folds:
        split = _split_for_fold(frame, timestamps, config, fold)
        train = split["train"]
        test = split["test"]
        if train.empty or test.empty:
            raise ValueError("insufficient_sample")
        _validate_non_null(train, (*config.feature_columns, config.label_column))
        _validate_non_null(test, config.feature_columns)

        predictions = test[
            [config.timestamp_column, config.entity_column, config.label_column]
        ].copy()
        predictions["fold_id"] = fold.fold_id
        predictions["split_name"] = fold.split_name
        for model_id, prediction_column in zip(
            config.model_ids,
            prediction_columns,
            strict=True,
        ):
            model = _make_model(model_id, random_seed=config.random_seed)
            model.fit(train[list(config.feature_columns)], train[config.label_column])
            predictions[prediction_column] = model.predict(
                test[list(config.feature_columns)]
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
    config.output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(
        config.output_prediction_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    _write_metadata(config, prediction_columns=prediction_columns, folds=metadata_folds)
    return ModelSuiteResult(
        prediction_path=config.output_prediction_path,
        metadata_path=config.output_metadata_path,
        model_suite_id=config.model_suite_id,
        prediction_columns=prediction_columns,
        row_count=len(output),
    )


def _validate_config(config: ModelSuiteConfig) -> None:
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    if not config.folds:
        raise ValueError("invalid_prediction_config")
    if config.label_horizon_days < 0:
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
    forbidden_models = {"pytorch", "tensorflow", "deep_learning"}
    if any(model_id in forbidden_models for model_id in config.model_ids):
        raise ValueError("dependency_unavailable")
    require_private_outputs_for_private_input(
        config.input_path,
        (config.output_prediction_path, config.output_metadata_path),
        error_message="private_output_policy_violation",
    )


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
    purged_train_end = min(
        train_end,
        test_start - pd.offsets.BDay(config.label_horizon_days + 1),
    )
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
        )
    if model_id == "extra_trees_lite":
        return ExtraTreesRegressor(
            n_estimators=50,
            max_depth=6,
            random_state=random_seed,
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
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_seed,
            objective="reg:squarederror",
        )
    if model_id == "lightgbm_lite":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=-1,
        )
    raise ValueError(f"dependency_unavailable: {model_id}")


def _prediction_column(model_id: str, label_column: str) -> str:
    horizon = "20d"
    prefix = "forward_return_"
    if label_column.startswith(prefix):
        horizon = label_column.removeprefix(prefix)
    return f"prediction_score_{model_id}_{horizon}"


def _write_metadata(
    config: ModelSuiteConfig,
    *,
    prediction_columns: tuple[str, ...],
    folds: list[dict[str, Any]],
) -> None:
    payload = {
        "prediction_handoff_version": PREDICTION_HANDOFF_VERSION,
        "model_suite_id": config.model_suite_id,
        "model_ids": list(config.model_ids),
        "prediction_columns": list(prediction_columns),
        "feature_columns": list(config.feature_columns),
        "target_column": config.label_column,
        "label_horizon_days": int(config.label_horizon_days),
        "folds": folds,
        "random_seed": int(config.random_seed),
        "boundary": MODEL_BOUNDARY,
    }
    config.output_metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _looks_like_label_or_forward_return(column: str) -> bool:
    normalized = column.casefold()
    return (
        normalized in {"label", "target"}
        or normalized.startswith("label_")
        or normalized.startswith("target_")
        or normalized.startswith("forward_return")
    )


__all__ = [
    "A_SHARE_MODEL_SUITE_ID",
    "MODEL_BOUNDARY",
    "ModelSuiteConfig",
    "ModelSuiteResult",
    "PREDICTION_HANDOFF_VERSION",
    "SuiteFoldConfig",
    "run_model_suite",
]
