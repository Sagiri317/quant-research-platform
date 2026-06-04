"""Report bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
)
from quant_research_platform.storage.layout import StorageLayout
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
    report_text: str,
    layout: StorageLayout,
) -> dict[str, Path]:
    """Write a private stage-profile report bundle."""

    destination = _derive_report_run_root(
        report_spec=spec,
        layout=layout,
        expected_definition_id="stage_profile_private_diagnostic",
        expected_run_id=spec.run_id,
    )

    def _write(temp_root: Path) -> None:
        safe_rows = sanitize_public_object(frame.to_dict(orient="records"))
        safe_frame = pd.DataFrame.from_records(safe_rows)
        atomic_write_csv(safe_frame, temp_root / "stage_profile.csv", index=False)
        atomic_write_json(temp_root / "stage_profile.json", safe_rows)
        atomic_write_text(
            temp_root / "stage_profile_report.md",
            sanitize_public_text(report_text),
            encoding="utf-8",
        )
        write_manifest(
            temp_root / "manifest.json",
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

    atomic_write_directory(destination, _write)
    csv_path = destination / "stage_profile.csv"
    json_path = destination / "stage_profile.json"
    report_path = destination / "stage_profile_report.md"
    return {"csv": csv_path, "json": json_path, "report": report_path}


def write_prediction_bundle_files(
    frame: pd.DataFrame,
    *,
    spec: PredictionSpec,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    """Write a private prediction bundle under reports/private_real."""

    run_root = _derive_report_run_root(
        report_spec=report_spec,
        layout=layout,
        expected_definition_id="prediction_bundle_private",
        expected_run_id=spec.prediction_id,
    )
    required = (*spec.primary_key, spec.prediction_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_bundle_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_bundle_columns: {', '.join(extra)}")
    if bool(frame.duplicated(list(spec.primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    def _write(temp_root: Path) -> None:
        atomic_write_parquet(
            frame.loc[:, list(required)],
            temp_root / "prediction.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            {
                "contract_version": spec.contract_version,
                "prediction_id": spec.prediction_id,
                "model_id": spec.model_id,
                "research_dataset_id": spec.research_dataset_id,
                "prediction_mode": spec.prediction_mode,
                "is_oos": spec.is_oos,
                "train_end": spec.train_end,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "is_final_oos_window": spec.is_final_oos_window,
                "prediction_column": spec.prediction_column,
                "primary_key": list(spec.primary_key),
            },
        )

    atomic_write_directory(run_root, _write)
    return run_root


def write_evaluation_report_files(
    metrics: pd.DataFrame,
    grouped_returns: pd.DataFrame,
    *,
    spec: EvaluationSpec,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    """Write a private evaluation report bundle under reports/private_real."""

    run_root = _derive_report_run_root(
        report_spec=report_spec,
        layout=layout,
        expected_definition_id="evaluation_report_private",
        expected_run_id=spec.run_id,
    )

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

    def _write(temp_root: Path) -> None:
        atomic_write_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        atomic_write_parquet(
            grouped_returns,
            temp_root / "grouped_returns.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
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

    atomic_write_directory(run_root, _write)
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


def _derive_report_run_root(
    *,
    report_spec: ReportSpec,
    layout: StorageLayout,
    expected_definition_id: str,
    expected_run_id: str,
) -> Path:
    try:
        definition = ReportCatalog.bootstrap_default().get(
            report_spec.report_definition_id
        )
    except KeyError as exc:
        raise ValueError("report_spec_mismatch") from exc
    if definition.report_definition_id != expected_definition_id:
        raise ValueError("report_spec_mismatch")
    if definition.contract_version != report_spec.contract_version:
        raise ValueError("report_spec_mismatch")
    if report_spec.run_id != expected_run_id:
        raise ValueError("report_spec_mismatch")
    if definition.run_type != report_spec.run_type:
        raise ValueError("report_spec_mismatch")
    run_root = (
        layout.private_report_root(report_spec.market_id, report_spec.run_type)
        / report_spec.run_id
    )
    if report_spec.private_boundary_required and not is_private_boundary_path(run_root):
        raise ValueError("outside_private_boundary")
    if "data/processed" in run_root.as_posix():
        raise ValueError("report_output_must_be_report_root")
    return run_root


__all__ = [
    "ReportBundle",
    "write_evaluation_report_files",
    "write_prediction_bundle_files",
    "write_stage_profile_bundle",
]
