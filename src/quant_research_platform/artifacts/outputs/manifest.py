"""Output manifest payload builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

from quant_research_platform.artifacts.definitions import output_catalog
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    EVALUATION_ARTIFACTS,
    validate_evaluation_output_metadata_shape,
)
from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_output_metadata_shape,
)
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.outputs import (
    OutputSpec,
    validate_output_identifier,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)

_OUTPUT_MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "output_id",
        "output_definition_id",
        "run_id",
        "metadata",
        "files",
    }
)
PREDICTION_ARTIFACTS = ("prediction.parquet",)
_OUTPUT_DEFINITION_FILES = {
    "prediction_outputs": PREDICTION_ARTIFACTS,
    "evaluation_outputs": EVALUATION_ARTIFACTS,
}


def validate_output_manifest_business_envelope(
    value: object,
) -> dict[str, object]:
    """Validate the current persisted output-manifest business envelope."""

    if type(value) is not dict or set(value) != _OUTPUT_MANIFEST_KEYS:
        raise ValueError("output_manifest_fields_invalid")
    if value["schema_id"] != "qrp-output-manifest":
        raise ValueError("output_manifest_schema_id_mismatch")

    output_id = value["output_id"]
    if type(output_id) is not str:
        raise ValueError("output_manifest_output_id_invalid")
    try:
        validate_output_identifier(output_id)
    except ValueError as exc:
        raise ValueError("output_manifest_output_id_invalid") from exc

    output_definition_id = value["output_definition_id"]
    if type(output_definition_id) is not str:
        raise ValueError("output_manifest_definition_id_invalid")
    try:
        validate_output_identifier(output_definition_id)
    except ValueError as exc:
        raise ValueError("output_manifest_definition_id_invalid") from exc
    if output_id == output_definition_id:
        raise ValueError("output_manifest_output_id_invalid")
    try:
        definition = output_catalog().get(output_definition_id)
    except KeyError as exc:
        raise ValueError("output_manifest_definition_id_invalid") from exc

    run_id = value["run_id"]
    if type(run_id) is not str:
        raise ValueError("output_manifest_run_id_invalid")
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise ValueError("output_manifest_run_id_invalid") from exc

    metadata = value["metadata"]
    if type(metadata) is not dict or _metadata_contains_schema_id(metadata):
        raise ValueError("output_manifest_metadata_invalid")
    if output_definition_id == "prediction_outputs":
        validate_prediction_output_metadata_shape(metadata)
    elif output_definition_id == "evaluation_outputs":
        validate_evaluation_output_metadata_shape(metadata)
    else:
        raise ValueError("output_manifest_definition_id_invalid")
    if "run_type" in metadata and metadata["run_type"] != definition.run_type:
        raise ValueError("output_manifest_metadata_invalid")
    if "scope" in metadata and metadata["scope"] != str(definition.scope):
        raise ValueError("output_manifest_metadata_invalid")
    evidence_payload: dict[str, object] = {}
    if output_definition_id == "prediction_outputs":
        model_metadata = metadata["model_metadata"]
        if "model_artifact_id" in model_metadata:
            evidence_payload = {
                "model_artifact_id": model_metadata["model_artifact_id"],
                "model_artifact_identity": model_metadata["model_artifact_identity"],
            }
        else:
            evidence_payload = {
                "hyperparameters": model_metadata["hyperparameters"],
                "resolved_hyperparameters": model_metadata["resolved_hyperparameters"],
                "dependency_version": model_metadata["dependency_status"]["version"],
            }
    if evidence_json_value_unsafe_reason(evidence_payload) is not None:
        raise ValueError("output_manifest_metadata_invalid")

    files = value["files"]
    if type(files) is not list or not files:
        raise ValueError("output_files_invalid")
    logical_files: set[str] = set()
    for item in files:
        if type(item) is not str or not item or "\\" in item:
            raise ValueError("output_file_path_invalid")
        path = PurePosixPath(item)
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or path.parts[0] in {"", ".", ".."}
            or path.as_posix() != item
        ):
            raise ValueError("output_file_path_invalid")
        logical_name = item.casefold()
        if logical_name in logical_files:
            raise ValueError("output_files_invalid")
        logical_files.add(logical_name)
    if files != list(_OUTPUT_DEFINITION_FILES[output_definition_id]):
        raise ValueError("output_manifest_definition_files_mismatch")
    return value


def _metadata_contains_schema_id(value: object) -> bool:
    if isinstance(value, Mapping):
        return "schema_id" in value or any(
            _metadata_contains_schema_id(item) for item in value.values()
        )
    if type(value) is list:
        return any(_metadata_contains_schema_id(item) for item in value)
    return False


def output_manifest_payload(
    *,
    output_spec: OutputSpec,
    metadata: Mapping[str, object],
    files: Iterable[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-output-manifest",
        "output_id": output_spec.output_id,
        "output_definition_id": output_spec.output_definition_id,
        "run_id": output_spec.run_id,
        "metadata": dict(metadata),
        "files": list(files),
    }
    validate_output_manifest_business_envelope(payload)
    return payload


def _evaluation_diagnostics_payload(
    diagnostics: EvaluationDiagnostics,
) -> dict[str, object]:
    return {
        "filter_mode": diagnostics.filter_mode,
        "filter_columns_used": list(diagnostics.filter_columns_used),
        "filter_definition": diagnostics.filter_definition,
        "rows_input": diagnostics.rows_input,
        "rows_after_filter_mode": diagnostics.rows_after_filter_mode,
        "rows_after_signal_label_dropna": diagnostics.rows_after_signal_label_dropna,
        "metric_rows": diagnostics.metric_rows,
        "grouped_label_value_rows": diagnostics.grouped_label_value_rows,
        "ic_summary_rows": diagnostics.ic_summary_rows,
        "group_spread_summary_rows": diagnostics.group_spread_summary_rows,
        "timestamp_count": diagnostics.timestamp_count,
        "min_timestamp": diagnostics.min_timestamp,
        "max_timestamp": diagnostics.max_timestamp,
        "n_groups": diagnostics.n_groups,
        "skipped_timestamp_count": diagnostics.skipped_timestamp_count,
    }


def prediction_output_metadata(
    *,
    spec: PredictionSpec,
    model_metadata: dict[str, object],
    prediction_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "prediction_id": spec.prediction_id,
        "model_id": spec.model_id,
        "research_dataset_id": spec.research_dataset_id,
        "prediction_mode": spec.prediction_mode,
        "is_oos": spec.is_oos,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "test_end": spec.test_end,
        "is_final_oos_window": spec.is_final_oos_window,
        "training_asset_ids": list(spec.training_asset_ids),
        "evaluation_asset_ids": list(spec.evaluation_asset_ids),
        "prediction_column": spec.prediction_column,
        "primary_key": list(spec.primary_key),
        "sample_filter_mode": spec.sample_filter_mode,
        "model_metadata": model_metadata,
        "prediction_metadata": prediction_metadata,
    }


def evaluation_output_metadata(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
) -> dict[str, object]:
    return {
        "evaluation_id": spec.evaluation_id,
        "run_type": spec.run_type,
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "evaluation_axis": spec.evaluation_axis,
        "signals": [
            {
                "signal_id": signal.signal_id,
                "source_kind": signal.source_kind,
                "source_column": signal.source_column,
                "direction": signal.direction,
                "transform": signal.transform,
            }
            for signal in spec.signals
        ],
        "label_columns": list(spec.label_columns),
        "label_availability_columns": dict(spec.label_availability_columns),
        "filter_mode": spec.filter_mode,
        "evaluation_diagnostics": _evaluation_diagnostics_payload(
            evaluation_diagnostics
        ),
    }
