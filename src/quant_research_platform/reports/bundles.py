"""Report bundle helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.privacy import (
    is_private_boundary_path,
    sanitize_public_object,
    sanitize_public_text,
)


@dataclass(frozen=True)
class ReportBundle:
    run_id: str
    root: Path


def write_stage_profile_bundle(
    frame: pd.DataFrame,
    *,
    spec: ReportSpec,
    output_root: Path,
    report_text: str,
) -> dict[str, Path]:
    """Write a private stage-profile report bundle."""

    if spec.report_definition_id != "stage_profile_private_diagnostic":
        raise ValueError("invalid_stage_profile_report_definition")
    destination = Path(output_root)
    if spec.private_boundary_required and not is_private_boundary_path(destination):
        raise ValueError("outside_private_boundary")
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "stage_profile.csv"
    json_path = destination / "stage_profile.json"
    report_path = destination / "stage_profile_report.md"
    safe_rows = sanitize_public_object(frame.to_dict(orient="records"))
    safe_frame = pd.DataFrame.from_records(safe_rows)
    safe_frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(safe_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(sanitize_public_text(report_text), encoding="utf-8")
    write_manifest(
        destination / "manifest.json",
        {
            "contract_version": spec.contract_version,
            "report_id": spec.report_id,
            "report_definition_id": spec.report_definition_id,
            "run_id": spec.run_id,
            "market_id": str(spec.market_id),
            "run_type": spec.run_type,
            "private_boundary_required": spec.private_boundary_required,
            "artifacts": [
                "stage_profile.csv",
                "stage_profile.json",
                "stage_profile_report.md",
            ],
        },
    )
    return {"csv": csv_path, "json": json_path, "report": report_path}


def write_prediction_bundle_files(
    frame: pd.DataFrame,
    *,
    spec: PredictionSpec,
    output_root: Path,
) -> Path:
    """Write a private prediction bundle under reports/private_real."""

    if not is_private_boundary_path(output_root):
        raise ValueError("outside_private_boundary")
    if "data/processed" in output_root.as_posix():
        raise ValueError("prediction_output_must_be_report_root")
    required = (*spec.primary_key, spec.prediction_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_bundle_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_bundle_columns: {', '.join(extra)}")
    if bool(frame.duplicated(list(spec.primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    run_root = output_root / spec.prediction_id
    run_root.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(required)].to_parquet(
        run_root / "prediction.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    write_manifest(
        run_root / "manifest.json",
        {
            "contract_version": spec.contract_version,
            "prediction_id": spec.prediction_id,
            "model_id": spec.model_id,
            "research_dataset_id": spec.research_dataset_id,
            "prediction_mode": "smoke_in_sample",
            "oos": False,
            "prediction_column": spec.prediction_column,
            "primary_key": list(spec.primary_key),
        },
    )
    return run_root


def write_evaluation_report_files(
    metrics: pd.DataFrame,
    grouped_returns: pd.DataFrame,
    *,
    spec: EvaluationSpec,
    output_root: Path,
) -> Path:
    """Write a private evaluation report bundle under reports/private_real."""

    if not is_private_boundary_path(output_root):
        raise ValueError("outside_private_boundary")
    if "data/processed" in output_root.as_posix():
        raise ValueError("evaluation_output_must_be_report_root")

    _validate_report_frame(
        metrics,
        required=("timestamp", "feature", "label", "ic", "rankic"),
        error_name="evaluation_metrics",
    )
    _validate_report_frame(
        grouped_returns,
        required=("timestamp", "group", "mean_return"),
        error_name="evaluation_grouped_returns",
    )

    run_root = output_root / spec.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(
        run_root / "metrics.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    grouped_returns.to_parquet(
        run_root / "grouped_returns.parquet",
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    write_manifest(
        run_root / "manifest.json",
        {
            "contract_version": spec.contract_version,
            "run_id": spec.run_id,
            "run_type": spec.run_type,
            "market_id": str(spec.market_id),
            "research_dataset_id": spec.research_dataset_id,
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "artifacts": ["metrics.parquet", "grouped_returns.parquet"],
        },
    )
    return run_root


def _validate_report_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    error_name: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_{error_name}_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_{error_name}_columns: {', '.join(extra)}")


__all__ = [
    "ReportBundle",
    "write_evaluation_report_files",
    "write_prediction_bundle_files",
    "write_stage_profile_bundle",
]
