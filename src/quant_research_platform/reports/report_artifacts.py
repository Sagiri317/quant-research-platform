"""Private report artifact writers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.catalog import report_catalog
from quant_research_platform.runtime.privacy import (
    is_private_boundary_path,
    sanitize_public_object,
    sanitize_public_text,
)
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
    reject_if_exists,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest

_EVALUATION_METRICS_COLUMNS = ("timestamp", "feature", "label", "ic", "rankic", "n_obs")
_EVALUATION_GROUPED_LABEL_VALUE_COLUMNS = (
    "timestamp",
    "feature",
    "label",
    "group",
    "mean_label_value",
    "n_obs",
)
_EVALUATION_IC_SUMMARY_COLUMNS = (
    "feature",
    "label",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "valid_ic_count",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "valid_rankic_count",
)
_EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS = (
    "feature",
    "label",
    "top_group_mean_label_value",
    "bottom_group_mean_label_value",
    "top_bottom_label_value_spread_mean",
    "top_bottom_label_value_spread_hit_rate",
    "group_monotonicity_score",
)
_EVALUATION_ARTIFACTS = [
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
]
_MODEL_METADATA_KEYS = frozenset(
    {
        "contract_version",
        "model_family",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_feature_label_dropna",
        "train_start",
        "train_end",
        "feature_columns",
        "feature_count",
        "label_column",
        "coefficient_count",
    }
)
_PREDICTION_METADATA_KEYS = frozenset(
    {
        "contract_version",
        "prediction_mode",
        "sample_filter_mode",
        "filter_columns_used",
        "filter_definition",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_feature_dropna",
        "prediction_rows",
        "train_end",
        "test_start",
        "requested_test_end",
        "effective_test_end",
        "is_oos",
        "is_final_oos_window",
        "feature_columns",
        "feature_count",
        "prediction_column",
    }
)


def write_stage_profile_report_artifacts(
    frame: pd.DataFrame,
    *,
    spec: ReportSpec,
    report_text: str,
    layout: StorageLayout,
) -> dict[str, Path]:
    """Write private stage-profile report artifacts."""

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
            _stage_profile_report_manifest_payload(spec=spec),
        )

    atomic_write_directory(destination, _write)
    csv_path = destination / "stage_profile.csv"
    json_path = destination / "stage_profile.json"
    report_path = destination / "stage_profile_report.md"
    return {"csv": csv_path, "json": json_path, "report": report_path}


def write_prediction_report_artifacts(
    frame: pd.DataFrame,
    *,
    spec: PredictionSpec,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    """Write private prediction report artifacts under reports/private_real."""

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
    model_metadata = _validate_model_metadata_shape(
        _required_frame_metadata(
            frame,
            key="model_metadata",
            error_name="missing_model_metadata",
        )
    )
    prediction_metadata = _validate_prediction_metadata_shape(
        _required_frame_metadata(
            frame,
            key="prediction_metadata",
            error_name="missing_prediction_metadata",
        )
    )
    _validate_model_metadata_consistency(model_metadata, spec=spec)
    _validate_prediction_metadata_consistency(
        prediction_metadata,
        spec=spec,
        frame=frame,
    )
    _validate_model_prediction_metadata_consistency(
        model_metadata,
        prediction_metadata,
    )

    def _write(temp_root: Path) -> None:
        _atomic_write_report_parquet(
            frame.loc[:, list(required)],
            temp_root / "prediction.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _prediction_report_manifest_payload(
                spec=spec,
                model_metadata=model_metadata,
                prediction_metadata=prediction_metadata,
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def write_evaluation_report_artifacts(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    report_spec: ReportSpec,
    layout: StorageLayout,
) -> Path:
    """Write private evaluation report artifacts under reports/private_real."""

    run_root = _derive_report_run_root(
        report_spec=report_spec,
        layout=layout,
        expected_definition_id="evaluation_report_private",
        expected_run_id=spec.run_id,
    )

    _validate_report_frame(
        metrics,
        required=_EVALUATION_METRICS_COLUMNS,
        error_name="evaluation_metrics",
    )
    _validate_report_frame(
        grouped_label_values,
        required=_EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        error_name="evaluation_grouped_label_values",
    )
    _validate_report_frame(
        ic_summary,
        required=_EVALUATION_IC_SUMMARY_COLUMNS,
        error_name="evaluation_ic_summary",
    )
    _validate_report_frame(
        group_spread_summary,
        required=_EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
        error_name="evaluation_group_spread_summary",
    )
    _validate_evaluation_artifact_structure(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        spec=spec,
        n_groups=evaluation_diagnostics.n_groups,
    )
    _validate_evaluation_diagnostics_consistency(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        evaluation_diagnostics=evaluation_diagnostics,
        spec=spec,
    )

    def _write(temp_root: Path) -> None:
        _atomic_write_report_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _atomic_write_report_parquet(
            grouped_label_values,
            temp_root / "grouped_label_values.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _atomic_write_report_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _atomic_write_report_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _evaluation_report_manifest_payload(
                spec=spec,
                evaluation_diagnostics=evaluation_diagnostics,
            ),
        )

    atomic_write_directory(run_root, _write)
    return run_root


def _stage_profile_report_manifest_payload(*, spec: ReportSpec) -> dict[str, object]:
    return {
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
    }


def _prediction_report_manifest_payload(
    *,
    spec: PredictionSpec,
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
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
        "sample_filter_mode": spec.sample_filter_mode,
        "model_metadata": model_metadata,
        "prediction_metadata": prediction_metadata,
    }


def _evaluation_report_manifest_payload(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
) -> dict[str, object]:
    return {
        "contract_version": spec.contract_version,
        "run_id": spec.run_id,
        "run_type": spec.run_type,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "feature_columns": list(spec.feature_columns),
        "label_columns": list(spec.label_columns),
        "filter_mode": spec.filter_mode,
        "evaluation_diagnostics": sanitize_public_object(
            asdict(evaluation_diagnostics)
        ),
        "artifacts": list(_EVALUATION_ARTIFACTS),
    }


def _required_frame_metadata(
    frame: pd.DataFrame,
    *,
    key: str,
    error_name: str,
) -> dict[str, object]:
    if key not in frame.attrs:
        raise ValueError(error_name)
    value = frame.attrs[key]
    if not isinstance(value, Mapping) or not value:
        raise ValueError(error_name)
    sanitized = sanitize_public_object(dict(value))
    if not isinstance(sanitized, dict) or not sanitized:
        raise ValueError(error_name)
    return sanitized


def _validate_model_metadata_shape(metadata: dict[str, object]) -> dict[str, object]:
    if set(metadata) != _MODEL_METADATA_KEYS:
        raise ValueError("invalid_model_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_model_metadata")
    _require_non_negative_metadata_int(metadata, "rows_input", "invalid_model_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_model_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_feature_label_dropna",
        "invalid_model_metadata",
    )
    feature_columns = _require_metadata_string_list(
        metadata,
        "feature_columns",
        "invalid_model_metadata",
    )
    feature_count = _require_non_negative_metadata_int(
        metadata,
        "feature_count",
        "invalid_model_metadata",
    )
    coefficient_count = _require_non_negative_metadata_int(
        metadata,
        "coefficient_count",
        "invalid_model_metadata",
    )
    if feature_count != len(feature_columns) or coefficient_count < 1:
        raise ValueError("invalid_model_metadata")
    _require_metadata_string(metadata, "model_family", "invalid_model_metadata")
    _require_metadata_string(metadata, "sample_filter_mode", "invalid_model_metadata")
    _require_metadata_string(metadata, "filter_definition", "invalid_model_metadata")
    _require_metadata_string(metadata, "label_column", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_start", "invalid_model_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_model_metadata")
    _require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_model_metadata",
        allow_empty=True,
    )
    return metadata


def _validate_prediction_metadata_shape(
    metadata: dict[str, object],
) -> dict[str, object]:
    if set(metadata) != _PREDICTION_METADATA_KEYS:
        raise ValueError("invalid_prediction_metadata")
    if metadata["contract_version"] != "0.1":
        raise ValueError("invalid_prediction_metadata")
    _require_non_negative_metadata_int(
        metadata,
        "rows_input",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "rows_after_feature_dropna",
        "invalid_prediction_metadata",
    )
    _require_non_negative_metadata_int(
        metadata,
        "prediction_rows",
        "invalid_prediction_metadata",
    )
    feature_columns = _require_metadata_string_list(
        metadata,
        "feature_columns",
        "invalid_prediction_metadata",
    )
    feature_count = _require_non_negative_metadata_int(
        metadata,
        "feature_count",
        "invalid_prediction_metadata",
    )
    if feature_count != len(feature_columns):
        raise ValueError("invalid_prediction_metadata")
    _require_metadata_string(
        metadata,
        "prediction_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "sample_filter_mode",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "filter_definition",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(metadata, "test_start", "invalid_prediction_metadata")
    _require_metadata_string(metadata, "train_end", "invalid_prediction_metadata")
    if metadata["requested_test_end"] is not None:
        _require_metadata_string(
            metadata,
            "requested_test_end",
            "invalid_prediction_metadata",
        )
    _require_metadata_string(
        metadata,
        "effective_test_end",
        "invalid_prediction_metadata",
    )
    _require_metadata_string(
        metadata,
        "prediction_column",
        "invalid_prediction_metadata",
    )
    _require_metadata_string_list(
        metadata,
        "filter_columns_used",
        "invalid_prediction_metadata",
        allow_empty=True,
    )
    if not isinstance(metadata["is_oos"], bool) or not isinstance(
        metadata["is_final_oos_window"],
        bool,
    ):
        raise ValueError("invalid_prediction_metadata")
    return metadata


def _validate_model_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
) -> None:
    feature_count = int(metadata["feature_count"])
    feature_columns = metadata["feature_columns"]
    if not isinstance(feature_columns, list) or feature_count != len(feature_columns):
        raise ValueError("model_metadata_inconsistent")
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_feature_label_dropna"),
        "model_metadata_inconsistent",
    )
    _validate_filter_metadata(
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
        error_name="model_metadata_inconsistent",
    )
    train_start = pd.Timestamp(metadata["train_start"])
    train_end = pd.Timestamp(metadata["train_end"])
    if train_start > train_end or train_end > pd.Timestamp(spec.train_end):
        raise ValueError("model_metadata_inconsistent")
    if (
        metadata["model_family"] == "linear"
        and int(metadata["coefficient_count"]) != feature_count + 1
    ):
        raise ValueError("model_metadata_inconsistent")


def _validate_prediction_metadata_consistency(
    metadata: dict[str, object],
    *,
    spec: PredictionSpec,
    frame: pd.DataFrame,
) -> None:
    expected = {
        "prediction_rows": len(frame),
        "prediction_column": spec.prediction_column,
        "prediction_mode": spec.prediction_mode,
        "sample_filter_mode": spec.sample_filter_mode,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "requested_test_end": spec.test_end,
        "is_oos": spec.is_oos,
        "is_final_oos_window": spec.is_final_oos_window,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError("prediction_metadata_inconsistent")
    _validate_metadata_row_count_chain(
        metadata,
        ("rows_input", "rows_after_filter_mode", "rows_after_feature_dropna"),
        "prediction_metadata_inconsistent",
    )
    if int(metadata["prediction_rows"]) > int(metadata["rows_after_feature_dropna"]):
        raise ValueError("prediction_metadata_inconsistent")
    _validate_filter_metadata(
        mode=str(metadata["sample_filter_mode"]),
        columns=metadata["filter_columns_used"],
        definition=str(metadata["filter_definition"]),
        error_name="prediction_metadata_inconsistent",
    )
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("prediction_metadata_inconsistent")
    effective_test_end = pd.Timestamp(timestamps.max()).date().isoformat()
    if metadata["effective_test_end"] != effective_test_end:
        raise ValueError("prediction_metadata_inconsistent")
    requested_test_end = metadata["requested_test_end"]
    if requested_test_end is not None and pd.Timestamp(
        effective_test_end
    ) > pd.Timestamp(requested_test_end):
        raise ValueError("prediction_metadata_inconsistent")


def _validate_model_prediction_metadata_consistency(
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> None:
    if model_metadata["feature_columns"] != prediction_metadata["feature_columns"]:
        raise ValueError("prediction_metadata_inconsistent")
    if model_metadata["feature_count"] != prediction_metadata["feature_count"]:
        raise ValueError("prediction_metadata_inconsistent")


def _validate_evaluation_diagnostics_consistency(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
) -> None:
    _validate_metadata_row_count_chain(
        asdict(evaluation_diagnostics),
        ("rows_input", "rows_after_filter_mode", "rows_after_feature_label_dropna"),
        "evaluation_diagnostics_inconsistent",
    )
    _validate_filter_metadata(
        mode=evaluation_diagnostics.filter_mode,
        columns=list(evaluation_diagnostics.filter_columns_used),
        definition=evaluation_diagnostics.filter_definition,
        error_name="evaluation_diagnostics_inconsistent",
    )
    timestamps = pd.to_datetime(metrics["timestamp"], errors="coerce")
    if bool(timestamps.isna().any()):
        raise ValueError("evaluation_diagnostics_inconsistent")
    expected_timestamp_count = int(timestamps.nunique()) if not timestamps.empty else 0
    expected_min_timestamp = (
        pd.Timestamp(timestamps.min()).date().isoformat()
        if not timestamps.empty
        else None
    )
    expected_max_timestamp = (
        pd.Timestamp(timestamps.max()).date().isoformat()
        if not timestamps.empty
        else None
    )
    n_obs = pd.to_numeric(metrics["n_obs"], errors="coerce")
    if bool(n_obs.isna().any()):
        raise ValueError("evaluation_diagnostics_inconsistent")
    expected_skipped_timestamp_count = int(
        metrics.loc[n_obs.lt(2), "timestamp"].nunique()
    )
    expected_rows_after_feature_label_dropna = int(n_obs.sum())
    expected = {
        "filter_mode": spec.filter_mode,
        "filter_columns_used": tuple(required_filter_columns(spec.filter_mode)),
        "filter_definition": filter_definition(spec.filter_mode),
        "rows_after_feature_label_dropna": expected_rows_after_feature_label_dropna,
        "metric_rows": len(metrics),
        "grouped_label_value_rows": len(grouped_label_values),
        "ic_summary_rows": len(ic_summary),
        "group_spread_summary_rows": len(group_spread_summary),
        "timestamp_count": expected_timestamp_count,
        "min_timestamp": expected_min_timestamp,
        "max_timestamp": expected_max_timestamp,
        "skipped_timestamp_count": expected_skipped_timestamp_count,
    }
    for key, value in expected.items():
        if getattr(evaluation_diagnostics, key) != value:
            raise ValueError("evaluation_diagnostics_inconsistent")


def _validate_evaluation_artifact_structure(
    *,
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    spec: EvaluationSpec,
    n_groups: int,
) -> None:
    _reject_duplicate_keys(metrics, ("timestamp", "feature", "label"), "metrics")
    _reject_duplicate_keys(
        grouped_label_values,
        ("timestamp", "feature", "label", "group"),
        "grouped_label_values",
    )
    _reject_duplicate_keys(ic_summary, ("feature", "label"), "ic_summary")
    _reject_duplicate_keys(
        group_spread_summary,
        ("feature", "label"),
        "group_spread_summary",
    )
    for frame in (metrics, grouped_label_values, ic_summary, group_spread_summary):
        _validate_feature_label_membership(frame, spec=spec)
    groups = pd.to_numeric(grouped_label_values["group"], errors="coerce")
    if bool(groups.isna().any()):
        raise ValueError("evaluation_group_out_of_range")
    if bool((groups % 1 != 0).any()) or bool(groups.lt(1).any()) or bool(
        groups.gt(n_groups).any()
    ):
        raise ValueError("evaluation_group_out_of_range")


def _reject_duplicate_keys(
    frame: pd.DataFrame,
    keys: tuple[str, ...],
    artifact_name: str,
) -> None:
    if bool(frame.duplicated(list(keys)).any()):
        raise ValueError(f"duplicate_evaluation_{artifact_name}_keys")


def _validate_feature_label_membership(
    frame: pd.DataFrame,
    *,
    spec: EvaluationSpec,
) -> None:
    if not set(frame["feature"]).issubset(set(spec.feature_columns)) or not set(
        frame["label"]
    ).issubset(set(spec.label_columns)):
        raise ValueError("evaluation_artifact_feature_label_mismatch")


def _validate_metadata_row_count_chain(
    metadata: dict[str, object],
    keys: tuple[str, str, str],
    error_name: str,
) -> None:
    first, second, third = (int(metadata[key]) for key in keys)
    if not (0 <= third <= second <= first):
        raise ValueError(error_name)


def _validate_filter_metadata(
    *,
    mode: str,
    columns: object,
    definition: str,
    error_name: str,
) -> None:
    if list(columns) != list(required_filter_columns(mode)):
        raise ValueError(error_name)
    if definition != filter_definition(mode):
        raise ValueError(error_name)


def _require_metadata_string(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> str:
    value = metadata[key]
    if not isinstance(value, str) or not value:
        raise ValueError(error_name)
    return value


def _require_metadata_string_list(
    metadata: dict[str, object],
    key: str,
    error_name: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    value = metadata[key]
    if not isinstance(value, list):
        raise ValueError(error_name)
    if (not allow_empty and not value) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(error_name)
    return value


def _require_non_negative_metadata_int(
    metadata: dict[str, object],
    key: str,
    error_name: str,
) -> int:
    value = metadata[key]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(error_name)
    return value


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
        definition = report_catalog().get(
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
        layout.private_research_report_root(
            report_spec.market_id,
            report_spec.run_type,
        )
        / report_spec.run_id
    )
    if report_spec.private_boundary_required and not is_private_boundary_path(run_root):
        raise ValueError("outside_private_boundary")
    if "data/processed" in run_root.as_posix():
        raise ValueError("report_output_must_be_report_root")
    return run_root


def _atomic_write_report_parquet(
    frame: pd.DataFrame,
    path: Path,
    **kwargs: object,
) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid4().hex[:12]}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(temp_path, **kwargs)
        reject_if_exists(final_path)
        _rename_with_transient_permission_retry(temp_path, final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    return final_path


def _rename_with_transient_permission_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 3,
    delay_seconds: float = 0.10,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)


__all__ = [
    "write_evaluation_report_artifacts",
    "write_prediction_report_artifacts",
    "write_stage_profile_report_artifacts",
]
