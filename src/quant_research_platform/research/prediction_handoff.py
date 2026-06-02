"""Local A-share N6 prediction handoff orchestration."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.datasets.parquet_panel import make_parquet_evaluation_spec
from quant_research_platform.evaluations.evaluation_runner import run_evaluation
from quant_research_platform.evaluations.evaluation_spec import (
    CostConfig,
    WeightConfig,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
)
from quant_research_platform.markets.implementations.a_share.research_batch_safety import (  # noqa: E501
    build_schema_inventory,
    run_memory_preflight,
    write_json_sanitized,
)
from quant_research_platform.research.ml_model_suite import (
    A_SHARE_MODEL_SUITE_ID,
    DEFAULT_NON_DL_MODEL_IDS,
    MODEL_BOUNDARY,
    PREDICTION_HANDOFF_VERSION,
    ModelSuiteConfig,
    SuiteFoldConfig,
    feature_name_has_sensitive_identifier,
    feature_name_looks_like_leakage,
    identifier_has_unsafe_metadata_marker,
    run_model_suite,
    validate_prediction_folds,
)
from quant_research_platform.runtime.private_paths import (
    path_is_under_private_output_root,
    require_private_outputs_for_private_input,
    text_has_private_marker,
)
from quant_research_platform.runtime.run_metadata import PREDICTION_TRAINING_SCOPE

SUPPORTED_FEATURE_PRESETS = frozenset({LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID})


@dataclass(frozen=True)
class PredictionHandoffConfig:
    """Config for one local prediction handoff run."""

    input_path: Path
    output_root: Path
    run_id: str
    timestamp_column: str = "timestamp"
    entity_column: str = "asset"
    feature_columns: tuple[str, ...] = ()
    feature_preset: str | None = None
    label_column: str = "forward_return_20d"
    return_source_column: str = "forward_return_1d"
    label_horizon_days: int = 20
    folds: tuple[dict[str, str | None], ...] = ()
    model_ids: tuple[str, ...] = DEFAULT_NON_DL_MODEL_IDS
    memory_budget_gb: float = 10.0
    dry_run: bool = False
    n_groups: int = 2


@dataclass(frozen=True)
class PredictionHandoffResult:
    """Small summary of one prediction handoff run."""

    run_dir: Path
    status: str
    row_count: int
    prediction_columns: tuple[str, ...] = ()


def run_prediction_handoff(
    config: PredictionHandoffConfig,
) -> PredictionHandoffResult:
    """Generate prediction parquet, evaluate it, and write comparison/EDA outputs."""

    _validate_config(config)
    run_dir = _new_run_dir(config.output_root, config.run_id)
    try:
        _write_preflight_files(config, run_dir)
        _write_feature_set_manifest(config, run_dir)
        if config.dry_run:
            write_json_sanitized(
                run_dir / "runtime_summary.json",
                _summary(config, "dry_run"),
            )
            _write_checkpoints(run_dir)
            return PredictionHandoffResult(
                run_dir=run_dir,
                status="dry_run",
                row_count=0,
            )

        prediction_path = run_dir / "predictions.parquet"
        prediction_long_path = run_dir / "prediction_records.parquet"
        metadata_path = run_dir / "model_metadata.json"
        model_result = run_model_suite(
            ModelSuiteConfig(
                input_path=config.input_path,
                output_prediction_path=prediction_path,
                output_metadata_path=metadata_path,
                output_prediction_long_path=prediction_long_path,
                timestamp_column=config.timestamp_column,
                entity_column=config.entity_column,
                feature_columns=config.feature_columns,
                label_column=config.label_column,
                label_horizon_days=config.label_horizon_days,
                folds=_suite_folds(config.folds),
                model_ids=config.model_ids or DEFAULT_NON_DL_MODEL_IDS,
            )
        )
        _write_handoff_metadata_overlay(
            config,
            metadata_path,
            prediction_columns=model_result.prediction_columns,
        )
        prediction_frame = _add_return_source_column(config, prediction_path)
        prediction_frame.to_parquet(
            prediction_path,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _run_prediction_evaluation(
            config,
            prediction_path,
            model_result.prediction_columns,
        )
        _write_comparison_outputs(
            config,
            run_dir,
            prediction_frame,
            model_result.prediction_columns,
        )
        _write_eda_outputs(
            config,
            run_dir,
            prediction_frame,
            model_result.prediction_columns,
        )
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            {
                **_summary(config, "ready"),
                "row_count": int(len(prediction_frame)),
                "prediction_columns": list(model_result.prediction_columns),
                "model_suite_id": A_SHARE_MODEL_SUITE_ID,
                "prediction_record_format": "wide_plus_long",
            },
        )
        _write_checkpoints(run_dir)
        return PredictionHandoffResult(
            run_dir=run_dir,
            status="ready",
            row_count=len(prediction_frame),
            prediction_columns=model_result.prediction_columns,
        )
    except Exception as exc:
        if not (run_dir / "runtime_summary.json").is_file():
            write_json_sanitized(
                run_dir / "runtime_summary.json",
                _summary(config, "failed", failure_code=_failure_code(exc)),
            )
        _write_checkpoints(run_dir)
        raise


def _validate_config(config: PredictionHandoffConfig) -> None:
    ensure_prediction_run_id_available(config.output_root, config.run_id)
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    if not config.feature_columns:
        raise ValueError("invalid_feature_preset")
    if config.return_source_column == config.label_column:
        raise ValueError("invalid_prediction_config")
    if config.return_source_column.startswith("prediction_score_"):
        raise ValueError("invalid_prediction_config")
    _validate_metadata_identifiers(
        (
            config.timestamp_column,
            config.entity_column,
            config.label_column,
            config.return_source_column,
        )
    )
    leakage = [
        column
        for column in config.feature_columns
        if feature_name_looks_like_leakage(column)
    ]
    if leakage or config.label_column in config.feature_columns:
        raise ValueError("invalid_feature_preset")
    if any(
        feature_name_has_sensitive_identifier(column)
        for column in config.feature_columns
    ):
        raise ValueError("invalid_feature_preset")
    if any(
        identifier_has_unsafe_metadata_marker(column)
        for column in config.feature_columns
    ):
        raise ValueError("invalid_feature_preset")
    if any(
        feature_name_has_sensitive_identifier(model_id)
        for model_id in config.model_ids
    ):
        raise ValueError("invalid_prediction_config")
    if any(
        identifier_has_unsafe_metadata_marker(model_id)
        for model_id in config.model_ids
    ):
        raise ValueError("invalid_prediction_config")
    if config.feature_preset and config.feature_preset not in SUPPORTED_FEATURE_PRESETS:
        raise ValueError("invalid_feature_preset")
    if not config.folds:
        raise ValueError("invalid_prediction_config")
    folds = _suite_folds(config.folds)
    _validate_metadata_identifiers(
        tuple(fold.fold_id for fold in folds)
        + tuple(fold.split_name for fold in folds)
    )
    require_private_outputs_for_private_input(
        config.input_path,
        (config.output_root,),
        error_message="private_output_policy_violation",
    )


def ensure_prediction_run_id_available(output_root: Path, run_id: str) -> Path:
    """Validate that a prediction handoff run directory can be created."""

    if (
        Path(run_id).name != run_id
        or run_id in {"", ".", ".."}
        or feature_name_has_sensitive_identifier(run_id)
        or identifier_has_unsafe_metadata_marker(run_id)
        or text_has_private_marker(run_id)
    ):
        raise ValueError("invalid_prediction_config")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise ValueError("duplicate_run_id")
    return run_dir


def _new_run_dir(output_root: Path, run_id: str) -> Path:
    run_dir = ensure_prediction_run_id_available(output_root, run_id)
    run_dir.mkdir(parents=True)
    return run_dir


def _write_preflight_files(config: PredictionHandoffConfig, run_dir: Path) -> None:
    preflight = run_memory_preflight(
        config.input_path,
        required_columns=(
            config.timestamp_column,
            config.entity_column,
            config.label_column,
            config.return_source_column,
            *config.feature_columns,
        ),
        memory_budget_gb=config.memory_budget_gb,
    )
    if preflight["status"] != "passed":
        write_json_sanitized(run_dir / "memory_preflight.json", preflight)
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            _summary(config, "failed", failure_code="memory_preflight_failed"),
        )
        _write_checkpoints(run_dir)
        raise ValueError("memory_preflight_failed")
    inventory = build_schema_inventory(
        config.input_path,
        expected_columns=(
            config.timestamp_column,
            config.entity_column,
            config.label_column,
            config.return_source_column,
            *config.feature_columns,
        ),
        key_columns=(config.entity_column, config.timestamp_column),
        date_column=config.timestamp_column,
    )
    write_json_sanitized(run_dir / "memory_preflight.json", preflight)
    write_json_sanitized(run_dir / "schema_inventory.json", inventory)
    if int(inventory.get("duplicate_key_count") or 0) > 0:
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            _summary(config, "failed", failure_code="duplicate_prediction_key"),
        )
        _write_checkpoints(run_dir)
        raise ValueError("duplicate_prediction_key")


def _write_feature_set_manifest(
    config: PredictionHandoffConfig,
    run_dir: Path,
) -> None:
    write_json_sanitized(
        run_dir / "feature_set_manifest.json",
        {
            "feature_set_id": _feature_set_id(config),
            "feature_resolution": "preset" if config.feature_preset else "explicit",
            "feature_columns": list(config.feature_columns),
            "feature_count": len(config.feature_columns),
            "excludes": [
                "target",
                "label",
                "forward_return",
                "filter",
                "tradability",
                "raw_identifier",
                "timestamp",
                "entity",
            ],
        },
    )


def _suite_folds(
    raw_folds: tuple[dict[str, str | None], ...],
) -> tuple[SuiteFoldConfig, ...]:
    folds: list[SuiteFoldConfig] = []
    for raw_fold in raw_folds:
        if not isinstance(raw_fold, dict):
            raise ValueError("invalid_prediction_config")
        if not all(key in raw_fold for key in ("fold_id", "train_end", "test_start")):
            raise ValueError("invalid_prediction_config")
        fold_id = _required_fold_field(raw_fold.get("fold_id"))
        train_end = _required_fold_field(raw_fold.get("train_end"))
        test_start = _required_fold_field(raw_fold.get("test_start"))
        test_end_value = raw_fold.get("test_end")
        test_end = (
            None
            if test_end_value is None or str(test_end_value).strip() == ""
            else _required_fold_field(test_end_value)
        )
        folds.append(
            SuiteFoldConfig(
                fold_id=fold_id,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                split_name=str(raw_fold.get("split_name") or "walk_forward"),
            )
        )
    return validate_prediction_folds(tuple(folds))


def _validate_metadata_identifiers(values: tuple[str, ...]) -> None:
    if any(feature_name_has_sensitive_identifier(value) for value in values):
        raise ValueError("invalid_prediction_config")
    if any(identifier_has_unsafe_metadata_marker(value) for value in values):
        raise ValueError("invalid_prediction_config")


def _required_fold_field(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("invalid_prediction_config")
    return value.strip()


def _add_return_source_column(
    config: PredictionHandoffConfig,
    prediction_path: Path,
) -> pd.DataFrame:
    predictions = pd.read_parquet(prediction_path, engine="pyarrow")
    if config.return_source_column in predictions.columns:
        return predictions
    source = pd.read_parquet(
        config.input_path,
        columns=(
            config.timestamp_column,
            config.entity_column,
            config.return_source_column,
        ),
        engine="pyarrow",
    )
    return predictions.merge(
        source,
        on=[config.timestamp_column, config.entity_column],
        how="left",
        validate="one_to_one",
    )


def _run_prediction_evaluation(
    config: PredictionHandoffConfig,
    prediction_path: Path,
    prediction_columns: tuple[str, ...],
) -> None:
    metadata = {
        "run_id": config.run_id,
        "project": "quant-research-platform",
        "kind": "prediction",
        "domain": "prediction_handoff",
        "data_stage": "approved_offline_real_data",
        "uses_external_data": True,
        "prediction_handoff_version": PREDICTION_HANDOFF_VERSION,
        "prediction_columns": list(prediction_columns),
        "label_columns": [config.label_column],
        "prediction_time_semantics": {
            "decision_time_column": config.timestamp_column,
            "as_of_time_columns": {
                column: config.timestamp_column for column in prediction_columns
            },
        },
        "source_ref": _feature_set_id(config),
        "model_ref": A_SHARE_MODEL_SUITE_ID,
        "training_scope": PREDICTION_TRAINING_SCOPE,
        "training_execution_scope": "local_qrp_research_layer",
        "evaluation_training_boundary": "training_outside_evaluation",
        "prediction_boundary": MODEL_BOUNDARY,
    }
    spec = make_parquet_evaluation_spec(
        prediction_path,
        timestamp_column=config.timestamp_column,
        asset_column=config.entity_column,
        feature_columns=prediction_columns,
        label_columns=(config.label_column,),
        return_source_column=config.return_source_column,
        metadata=metadata,
        output_dir=config.output_root / config.run_id / "evaluation",
        n_groups=config.n_groups,
        weight_configs=(WeightConfig(),),
        cost_configs=(CostConfig(slippage_bps=0.0),),
        kernel="dataframe",
        domain_notes={"boundary": MODEL_BOUNDARY},
    )
    run_evaluation(spec)


def _write_comparison_outputs(
    config: PredictionHandoffConfig,
    run_dir: Path,
    frame: pd.DataFrame,
    prediction_columns: tuple[str, ...],
) -> None:
    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    feature_set_id = _feature_set_id(config)
    rows = [
        _metric_row(
            frame,
            signal_column=column,
            label_column=config.label_column,
            timestamp_column=config.timestamp_column,
            model_id=_model_id_from_prediction_column(column, config.label_column),
            target_horizon=config.label_column.removeprefix("forward_return_"),
            feature_set_id=feature_set_id,
        )
        for column in prediction_columns
    ]
    existing_model_ids = {str(row["model_id"]) for row in rows}
    for feature in config.feature_columns:
        model_id = f"single_factor_{feature}"
        if model_id in existing_model_ids:
            continue
        rows.append(
            _metric_row(
                frame=_merge_feature_for_comparison(config, frame, feature),
                signal_column=feature,
                label_column=config.label_column,
                timestamp_column=config.timestamp_column,
                model_id=model_id,
                target_horizon=config.label_column.removeprefix("forward_return_"),
                feature_set_id=feature_set_id,
            )
        )
    leaderboard = pd.DataFrame(rows)
    leaderboard.to_csv(comparison_dir / "model_leaderboard.csv", index=False)
    (comparison_dir / "model_leaderboard.json").write_text(
        leaderboard.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )
    leaderboard.to_csv(
        comparison_dir / "prediction_vs_single_factors.csv",
        index=False,
    )
    (comparison_dir / "prediction_vs_single_factors.json").write_text(
        leaderboard.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )


def _merge_feature_for_comparison(
    config: PredictionHandoffConfig,
    frame: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    if feature in frame.columns:
        return frame
    source = pd.read_parquet(
        config.input_path,
        columns=(config.timestamp_column, config.entity_column, feature),
        engine="pyarrow",
    )
    return frame.merge(
        source,
        on=[config.timestamp_column, config.entity_column],
        how="left",
        validate="one_to_one",
    )


def _metric_row(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    label_column: str,
    timestamp_column: str,
    model_id: str,
    target_horizon: str,
    feature_set_id: str,
) -> dict[str, object]:
    valid = frame[[timestamp_column, signal_column, label_column]].dropna()
    grouped_ic = _datewise_correlations(
        valid,
        timestamp_column=timestamp_column,
        signal_column=signal_column,
        label_column=label_column,
        rank=False,
    )
    grouped_rank_ic = _datewise_correlations(
        valid,
        timestamp_column=timestamp_column,
        signal_column=signal_column,
        label_column=label_column,
        rank=True,
    )
    return {
        "model_id": model_id,
        "target_horizon": target_horizon,
        "feature_set_id": feature_set_id,
        "mean_ic": _finite_mean(grouped_ic),
        "mean_rankic": _finite_mean(grouped_rank_ic),
        "coverage": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "n_oos_rows": int(len(valid)),
    }


def _feature_set_id(config: PredictionHandoffConfig) -> str:
    return config.feature_preset or "explicit_feature_columns"


def _write_handoff_metadata_overlay(
    config: PredictionHandoffConfig,
    metadata_path: Path,
    *,
    prediction_columns: tuple[str, ...],
) -> None:
    """Add handoff-level portable metadata needed for recovery and APIs."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "timestamp_column": config.timestamp_column,
            "entity_column": config.entity_column,
            "label_column": config.label_column,
            "target_column": config.label_column,
            "return_source_column": config.return_source_column,
            "feature_columns": list(config.feature_columns),
            "feature_preset": config.feature_preset,
            "model_ids": list(metadata.get("model_ids", [])),
            "requested_model_ids": list(config.model_ids or DEFAULT_NON_DL_MODEL_IDS),
            "model_suite_id": A_SHARE_MODEL_SUITE_ID,
            "prediction_handoff_version": PREDICTION_HANDOFF_VERSION,
            "prediction_columns": list(prediction_columns),
            "label_horizon_days": int(config.label_horizon_days),
            "training_execution_scope": "local_qrp_research_layer",
            "evaluation_training_boundary": "training_outside_evaluation",
        }
    )
    write_json_sanitized(metadata_path, metadata)


def _finite_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )
    finite = finite.dropna()
    return None if finite.empty else float(finite.mean())


def _datewise_correlations(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    signal_column: str,
    label_column: str,
    rank: bool,
) -> pd.Series:
    values = []
    for _, group in frame.groupby(timestamp_column):
        if len(group) <= 1:
            values.append(None)
            continue
        signal = group[signal_column].rank() if rank else group[signal_column]
        label = group[label_column].rank() if rank else group[label_column]
        values.append(signal.corr(label))
    return pd.Series(values, dtype="float64")


def _model_id_from_prediction_column(column: str, label_column: str) -> str:
    horizon = label_column.removeprefix("forward_return_")
    suffix = f"_{horizon}"
    value = column.removeprefix("prediction_score_")
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return value


def _write_eda_outputs(
    config: PredictionHandoffConfig,
    run_dir: Path,
    frame: pd.DataFrame,
    prediction_columns: tuple[str, ...],
) -> None:
    _write_prediction_eda_outputs(
        run_dir=run_dir,
        frame=frame,
        prediction_columns=prediction_columns,
        label_column=config.label_column,
        timestamp_column=config.timestamp_column,
    )


def _write_prediction_eda_outputs(
    *,
    run_dir: Path,
    frame: pd.DataFrame,
    prediction_columns: tuple[str, ...],
    label_column: str,
    timestamp_column: str,
) -> None:
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    years = pd.to_datetime(frame[timestamp_column]).dt.year
    coverage = (
        frame.assign(year=years)
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    coverage.to_csv(eda_dir / "data_coverage_by_year.csv", index=False)
    months = (
        pd.to_datetime(frame[timestamp_column])
        .dt.to_period("M")
        .astype(str)
    )
    monthly = (
        frame.assign(month=months)
        .groupby("month", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    monthly.to_csv(eda_dir / "data_coverage_by_month.csv", index=False)
    missingness = (
        frame[list(prediction_columns)]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missingness.to_csv(eda_dir / "missingness_by_feature.csv", index=False)
    ic_year = _ic_breakdown(
        frame,
        prediction_columns=prediction_columns,
        label_column=label_column,
        timestamp_column=timestamp_column,
        period="year",
    )
    ic_month = _ic_breakdown(
        frame,
        prediction_columns=prediction_columns,
        label_column=label_column,
        timestamp_column=timestamp_column,
        period="month",
    )
    ic_year.to_csv(eda_dir / "prediction_ic_by_year.csv", index=False)
    ic_month.to_csv(eda_dir / "prediction_ic_by_month.csv", index=False)
    importance_count = _write_feature_importance_outputs(run_dir, eda_dir)
    importance = _read_optional_csv(eda_dir / "tree_model_feature_importance.csv")
    chart_count, chart_status, chart_files = _write_optional_charts(
        eda_dir,
        coverage,
        missingness,
        ic_year=ic_year,
        ic_month=ic_month,
        leaderboard=None,
        feature_importance=importance,
    )
    write_json_sanitized(
        eda_dir / "eda_manifest.json",
        {
            "status": "ready",
            "eda_status": "experimental_non_contract",
            "artifact_count": 6 + importance_count + chart_count,
            "chart_count": chart_count,
            "chart_dependency_status": chart_status,
            "chart_files": list(chart_files),
            "feature_importance_artifact_count": importance_count,
        },
    )
    (eda_dir / "eda_report.md").write_text(
        "# Prediction handoff EDA\n\n"
        "Experimental aggregate-only diagnostics; not schema 0.7 contract.\n",
        encoding="utf-8",
    )


def refresh_prediction_handoff_eda(run_dir: Path) -> dict[str, object]:
    """Regenerate aggregate EDA outputs and charts for an existing handoff run."""

    if not path_is_under_private_output_root(
        run_dir,
        "a_share_prediction_handoffs",
    ):
        raise ValueError("private_output_policy_violation")
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_model_metadata(run_dir)
    frame = pd.read_parquet(
        run_dir / "predictions.parquet",
        engine="pyarrow",
    )
    _write_prediction_eda_outputs(
        run_dir=run_dir,
        frame=frame,
        prediction_columns=tuple(metadata["prediction_columns"]),
        label_column=str(metadata.get("label_column") or metadata["target_column"]),
        timestamp_column=str(metadata["timestamp_column"]),
    )
    manifest = json.loads(
        (eda_dir / "eda_manifest.json").read_text(encoding="utf-8")
    )
    chart_count = int(manifest.get("chart_count", 0) or 0)
    chart_status = str(manifest.get("chart_dependency_status", "unknown"))
    chart_files = tuple(manifest.get("chart_files", []))
    _write_checkpoints(run_dir)
    return {
        "status": "ready",
        "chart_count": chart_count,
        "chart_dependency_status": chart_status,
        "chart_files": tuple(chart_files),
    }


def _read_model_metadata(run_dir: Path) -> dict[str, object]:
    metadata_path = run_dir / "model_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prediction_columns = metadata.get("prediction_columns", [])
    if not isinstance(prediction_columns, list) or not all(
        isinstance(column, str) for column in prediction_columns
    ):
        raise ValueError("invalid_prediction_config")
    target_column = metadata.get("target_column")
    if not isinstance(target_column, str):
        raise ValueError("invalid_prediction_config")
    timestamp_column = metadata.get("timestamp_column", "timestamp")
    if not isinstance(timestamp_column, str):
        timestamp_column = "timestamp"
    entity_column = metadata.get("entity_column", "asset")
    if not isinstance(entity_column, str):
        entity_column = "asset"
    label_column = metadata.get("label_column", target_column)
    if not isinstance(label_column, str):
        label_column = target_column
    return_source_column = metadata.get("return_source_column")
    if return_source_column is not None and not isinstance(return_source_column, str):
        return_source_column = None
    return {
        "prediction_columns": tuple(prediction_columns),
        "target_column": target_column,
        "label_column": label_column,
        "timestamp_column": timestamp_column,
        "entity_column": entity_column,
        "return_source_column": return_source_column,
    }


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def _ic_breakdown(
    frame: pd.DataFrame,
    *,
    prediction_columns: tuple[str, ...],
    label_column: str,
    timestamp_column: str,
    period: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if period == "year":
        labels = dates.dt.year.astype("string")
    else:
        labels = dates.dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for column in prediction_columns:
        valid = frame[[column, label_column]].copy()
        valid["period"] = labels
        valid = valid.dropna()
        for value, group in valid.groupby("period"):
            if len(group) <= 1:
                continue
            rows.append(
                {
                    "period": str(value),
                    "model_id": _model_id_from_prediction_column(
                        column,
                        label_column,
                    ),
                    "mean_ic": float(group[column].corr(group[label_column])),
                    "mean_rankic": float(
                        group[column].rank().corr(group[label_column].rank())
                    ),
                    "n_oos_rows": int(len(group)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=("period", "model_id", "mean_ic", "mean_rankic", "n_oos_rows"),
    )


def _write_feature_importance_outputs(run_dir: Path, eda_dir: Path) -> int:
    metadata_path = run_dir / "model_metadata.json"
    if not metadata_path.is_file():
        return 0
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    rows = []
    for status in metadata.get("model_statuses", []):
        if not isinstance(status, dict):
            continue
        for item in status.get("feature_importances", []):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "fold_id": status.get("fold_id"),
                    "model_id": status.get("model_id"),
                    "feature": item.get("feature"),
                    "importance": item.get("importance"),
                }
            )
    if not rows:
        pd.DataFrame(
            columns=("fold_id", "model_id", "feature", "importance")
        ).to_csv(
            eda_dir / "tree_model_feature_importance.csv",
            index=False,
        )
        return 0
    pd.DataFrame(rows).to_csv(
        eda_dir / "tree_model_feature_importance.csv",
        index=False,
    )
    return 1


def _summary(
    config: PredictionHandoffConfig,
    status: str,
    *,
    failure_code: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "run_id": config.run_id,
        "prediction_handoff_version": PREDICTION_HANDOFF_VERSION,
        "model_suite_id": A_SHARE_MODEL_SUITE_ID,
        "boundary": MODEL_BOUNDARY,
    }
    if failure_code:
        payload["failure_code"] = failure_code
    return payload


def _failure_code(exc: BaseException) -> str:
    detail = str(exc)
    known = {
        "memory_preflight_failed",
        "insufficient_sample",
        "duplicate_prediction_key",
        "invalid_prediction_config",
        "invalid_feature_preset",
        "private_output_policy_violation",
        "dependency_unavailable",
        "duplicate_run_id",
    }
    if detail in known:
        return detail
    if isinstance(exc, FileNotFoundError):
        return "missing_prediction_input"
    return "prediction_handoff_failed"


def _write_checkpoints(run_dir: Path) -> None:
    checkpoints = [
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("feature_set_manifest", "feature_set_manifest.json"),
        ("predictions", "predictions.parquet"),
        ("prediction_records", "prediction_records.parquet"),
        ("model_metadata", "model_metadata.json"),
        ("evaluation", "evaluation_manifest"),
        ("comparison", "model_leaderboard"),
        ("eda", "eda_manifest"),
        ("runtime_summary", "runtime_summary.json"),
    ]
    payload = [
        {
            "key": key,
            "label": label,
            "status": "ready" if _checkpoint_ready(run_dir, key) else "pending",
        }
        for key, label in checkpoints
    ]
    write_json_sanitized(run_dir / "checkpoints.json", {"checkpoints": payload})


def _checkpoint_ready(run_dir: Path, key: str) -> bool:
    paths = {
        "schema_inventory": run_dir / "schema_inventory.json",
        "memory_preflight": run_dir / "memory_preflight.json",
        "feature_set_manifest": run_dir / "feature_set_manifest.json",
        "predictions": run_dir / "predictions.parquet",
        "prediction_records": run_dir / "prediction_records.parquet",
        "model_metadata": run_dir / "model_metadata.json",
        "evaluation": run_dir / "evaluation" / "manifest.json",
        "comparison": run_dir / "comparison" / "model_leaderboard.csv",
        "eda": run_dir / "eda" / "eda_manifest.json",
        "runtime_summary": run_dir / "runtime_summary.json",
    }
    return paths[key].is_file()


def _write_optional_charts(
    eda_dir: Path,
    coverage: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    ic_year: pd.DataFrame | None = None,
    ic_month: pd.DataFrame | None = None,
    leaderboard: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
) -> tuple[int, str, tuple[str, ...]]:
    _clear_managed_chart_files(eda_dir)
    chart_files: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, OSError, RuntimeError):
        _clear_managed_chart_files(eda_dir)
        return 0, "dependency_unavailable", ()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if not coverage.empty:
                fig, ax = plt.subplots(figsize=(6, 3))
                coverage.plot.bar(x="year", y="row_count", ax=ax, legend=False)
                ax.set_xlabel("year")
                ax.set_ylabel("row_count")
                _safe_tight_layout(fig)
                filename = "coverage_by_year.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if not missingness.empty:
                fig, ax = plt.subplots(figsize=(7, 3))
                top_missing = missingness.sort_values(
                    "missing_fraction",
                    ascending=False,
                ).head(20)
                top_missing.plot.bar(
                    x="feature",
                    y="missing_fraction",
                    ax=ax,
                    legend=False,
                )
                ax.set_xlabel("feature")
                ax.set_ylabel("missing_fraction")
                _safe_tight_layout(fig)
                filename = "missingness_top20.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if ic_year is not None and not ic_year.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                _top_metric_rows(ic_year).plot.bar(
                    x="label",
                    y="mean_rankic",
                    ax=ax,
                    legend=False,
                )
                ax.set_xlabel("model_year")
                ax.set_ylabel("mean_rankic")
                _safe_tight_layout(fig)
                filename = "prediction_ic_by_year.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if ic_month is not None and not ic_month.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                _top_metric_rows(ic_month).plot.bar(
                    x="label",
                    y="mean_rankic",
                    ax=ax,
                    legend=False,
                )
                ax.set_xlabel("model_month")
                ax.set_ylabel("mean_rankic")
                _safe_tight_layout(fig)
                filename = "prediction_ic_by_month.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if leaderboard is not None and not leaderboard.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                _top_model_rows(leaderboard).plot.bar(
                    x="model_id",
                    y="mean_rankic",
                    ax=ax,
                    legend=False,
                )
                ax.set_xlabel("model_id")
                ax.set_ylabel("mean_rankic")
                _safe_tight_layout(fig)
                filename = "model_leaderboard_top20.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
            if feature_importance is not None and not feature_importance.empty:
                fig, ax = plt.subplots(figsize=(8, 3))
                _top_feature_importance_rows(feature_importance).plot.bar(
                    x="label",
                    y="importance",
                    ax=ax,
                    legend=False,
                )
                ax.set_xlabel("model_feature")
                ax.set_ylabel("importance")
                _safe_tight_layout(fig)
                filename = "feature_importance_top20.png"
                chart_files.append(filename)
                fig.savefig(eda_dir / filename, dpi=120)
                plt.close(fig)
    except (ImportError, OSError, RuntimeError):
        _clear_managed_chart_files(eda_dir)
        return 0, "dependency_unavailable", ()
    return (
        len(chart_files),
        "ready" if chart_files else "not_generated",
        tuple(chart_files),
    )


def _clear_managed_chart_files(eda_dir: Path) -> None:
    for filename in (
        "coverage_by_year.png",
        "missingness_top20.png",
        "prediction_ic_by_year.png",
        "prediction_ic_by_month.png",
        "model_leaderboard_top20.png",
        "feature_importance_top20.png",
    ):
        (eda_dir / filename).unlink(missing_ok=True)


def _safe_tight_layout(fig: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def _top_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["label"] = (
        output["model_id"].astype(str) + "_" + output["period"].astype(str)
    )
    output["abs_rankic"] = output["mean_rankic"].abs()
    return output.sort_values("abs_rankic", ascending=False).head(20)


def _top_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["abs_rankic"] = output["mean_rankic"].abs()
    return output.sort_values("abs_rankic", ascending=False).head(20)


def _top_feature_importance_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["label"] = (
        output["model_id"].astype(str) + "_" + output["feature"].astype(str)
    )
    return output.sort_values("importance", ascending=False).head(20)


__all__ = [
    "PredictionHandoffConfig",
    "PredictionHandoffResult",
    "refresh_prediction_handoff_eda",
    "run_prediction_handoff",
]
