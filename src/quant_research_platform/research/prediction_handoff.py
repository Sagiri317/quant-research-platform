"""Local A-share N6 prediction handoff orchestration."""

from __future__ import annotations

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
    MODEL_BOUNDARY,
    PREDICTION_HANDOFF_VERSION,
    ModelSuiteConfig,
    SuiteFoldConfig,
    run_model_suite,
)
from quant_research_platform.runtime.private_paths import (
    require_private_outputs_for_private_input,
)
from quant_research_platform.runtime.run_metadata import PREDICTION_TRAINING_SCOPE


@dataclass(frozen=True)
class PredictionHandoffConfig:
    """Config for one local prediction handoff run."""

    input_path: Path
    output_root: Path
    run_id: str
    timestamp_column: str = "timestamp"
    entity_column: str = "asset"
    feature_columns: tuple[str, ...] = ()
    label_column: str = "forward_return_20d"
    return_source_column: str = "forward_return_1d"
    label_horizon_days: int = 20
    folds: tuple[dict[str, str | None], ...] = ()
    model_ids: tuple[str, ...] = ("ridge", "elastic_net", "hist_gradient_boosting")
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
    _write_preflight_files(config, run_dir)
    if config.dry_run:
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            _summary(config, "dry_run"),
        )
        _write_checkpoints(run_dir)
        return PredictionHandoffResult(run_dir=run_dir, status="dry_run", row_count=0)

    prediction_path = run_dir / "predictions.parquet"
    metadata_path = run_dir / "model_metadata.json"
    model_result = run_model_suite(
        ModelSuiteConfig(
            input_path=config.input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column=config.timestamp_column,
            entity_column=config.entity_column,
            feature_columns=config.feature_columns,
            label_column=config.label_column,
            label_horizon_days=config.label_horizon_days,
            folds=_suite_folds(config.folds),
            model_ids=config.model_ids,
        )
    )
    prediction_frame = _add_return_source_column(config, prediction_path)
    prediction_frame.to_parquet(
        prediction_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    _run_prediction_evaluation(config, prediction_path, model_result.prediction_columns)
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
        },
    )
    _write_checkpoints(run_dir)
    return PredictionHandoffResult(
        run_dir=run_dir,
        status="ready",
        row_count=len(prediction_frame),
        prediction_columns=model_result.prediction_columns,
    )


def _validate_config(config: PredictionHandoffConfig) -> None:
    if not config.input_path.is_file():
        raise FileNotFoundError(config.input_path)
    if not config.feature_columns:
        raise ValueError("invalid_feature_preset")
    if not config.folds:
        raise ValueError("invalid_prediction_config")
    require_private_outputs_for_private_input(
        config.input_path,
        (config.output_root,),
        error_message="private_output_policy_violation",
    )


def _new_run_dir(output_root: Path, run_id: str) -> Path:
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError("invalid_prediction_config")
    run_dir = output_root / run_id
    if run_dir.exists():
        raise ValueError("duplicate_run_id")
    run_dir.mkdir(parents=True)
    return run_dir


def _write_preflight_files(config: PredictionHandoffConfig, run_dir: Path) -> None:
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
    write_json_sanitized(run_dir / "schema_inventory.json", inventory)
    write_json_sanitized(run_dir / "memory_preflight.json", preflight)
    if preflight["status"] != "passed":
        write_json_sanitized(
            run_dir / "runtime_summary.json",
            _summary(config, "failed", failure_code="memory_preflight_failed"),
        )
        _write_checkpoints(run_dir)
        raise ValueError("memory_preflight_failed")


def _suite_folds(
    raw_folds: tuple[dict[str, str | None], ...],
) -> tuple[SuiteFoldConfig, ...]:
    return tuple(
        SuiteFoldConfig(
            fold_id=str(fold["fold_id"]),
            train_end=str(fold["train_end"]),
            test_start=str(fold["test_start"]),
            test_end=None if fold.get("test_end") is None else str(fold["test_end"]),
            split_name=str(fold.get("split_name") or "walk_forward"),
        )
        for fold in raw_folds
    )


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
        "source_ref": LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
        "model_ref": A_SHARE_MODEL_SUITE_ID,
        "training_scope": PREDICTION_TRAINING_SCOPE,
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
    rows = [
        _metric_row(
            frame,
            signal_column=column,
            label_column=config.label_column,
            model_id=column.removeprefix("prediction_score_").removesuffix("_20d"),
            target_horizon=config.label_column.removeprefix("forward_return_"),
            feature_set_id=LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
        )
        for column in prediction_columns
    ]
    for feature in config.feature_columns:
        rows.append(
            _metric_row(
                frame=_merge_feature_for_comparison(config, frame, feature),
                signal_column=feature,
                label_column=config.label_column,
                model_id=f"single_factor_{feature}",
                target_horizon=config.label_column.removeprefix("forward_return_"),
                feature_set_id=LITERATURE_PRICE_VOLUME_FEATURE_PRESET_ID,
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
    model_id: str,
    target_horizon: str,
    feature_set_id: str,
) -> dict[str, object]:
    valid = frame[[signal_column, label_column]].dropna()
    mean_ic = (
        float(valid[signal_column].corr(valid[label_column]))
        if len(valid) > 1
        else None
    )
    rank_ic = (
        float(valid[signal_column].rank().corr(valid[label_column].rank()))
        if len(valid) > 1
        else None
    )
    return {
        "model_id": model_id,
        "target_horizon": target_horizon,
        "feature_set_id": feature_set_id,
        "mean_ic": mean_ic,
        "mean_rankic": rank_ic,
        "coverage": float(len(valid) / len(frame)) if len(frame) else 0.0,
        "n_oos_rows": int(len(valid)),
    }


def _write_eda_outputs(
    config: PredictionHandoffConfig,
    run_dir: Path,
    frame: pd.DataFrame,
    prediction_columns: tuple[str, ...],
) -> None:
    eda_dir = run_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    years = pd.to_datetime(frame[config.timestamp_column]).dt.year
    coverage = (
        frame.assign(year=years)
        .groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )
    coverage.to_csv(eda_dir / "data_coverage_by_year.csv", index=False)
    missingness = (
        frame[list(prediction_columns)]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missingness.to_csv(eda_dir / "missingness_by_feature.csv", index=False)
    write_json_sanitized(
        eda_dir / "eda_manifest.json",
        {
            "status": "ready",
            "eda_status": "experimental_non_contract",
            "artifact_count": 2,
        },
    )
    (eda_dir / "eda_report.md").write_text(
        "# Prediction handoff EDA\n\n"
        "Experimental aggregate-only diagnostics; not schema 0.7 contract.\n",
        encoding="utf-8",
    )


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


def _write_checkpoints(run_dir: Path) -> None:
    checkpoints = [
        ("schema_inventory", "schema_inventory.json"),
        ("memory_preflight", "memory_preflight.json"),
        ("predictions", "predictions.parquet"),
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
        "predictions": run_dir / "predictions.parquet",
        "model_metadata": run_dir / "model_metadata.json",
        "evaluation": run_dir / "evaluation" / "manifest.json",
        "comparison": run_dir / "comparison" / "model_leaderboard.csv",
        "eda": run_dir / "eda" / "eda_manifest.json",
        "runtime_summary": run_dir / "runtime_summary.json",
    }
    return paths[key].is_file()


__all__ = [
    "PredictionHandoffConfig",
    "PredictionHandoffResult",
    "run_prediction_handoff",
]
