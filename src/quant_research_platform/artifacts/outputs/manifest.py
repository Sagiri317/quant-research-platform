"""Output manifest payload builders."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

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
    validate_output_identifier,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.storage.identity import require_unaliased_path
from quant_research_platform.storage.manifests import read_qrp_json_file

_OUTPUT_MANIFEST_KEYS = frozenset(
    {
        "schema_id",
        "output_definition_id",
        "metadata",
        "files",
    }
)
PREDICTION_ARTIFACTS = ("prediction.parquet",)
_OUTPUT_DEFINITION_FILES = {
    "prediction_outputs": PREDICTION_ARTIFACTS,
    "evaluation_outputs": EVALUATION_ARTIFACTS,
}


def validate_output_manifest(
    value: object,
) -> dict[str, object]:
    """Validate the persisted output-manifest business envelope."""

    if type(value) is not dict or set(value) != _OUTPUT_MANIFEST_KEYS:
        raise ValueError("output_manifest_fields_invalid")
    if value["schema_id"] != "qrp-output-manifest":
        raise ValueError("output_manifest_schema_id_mismatch")

    output_definition_id = value["output_definition_id"]
    if type(output_definition_id) is not str:
        raise ValueError("output_manifest_definition_id_invalid")
    try:
        validate_output_identifier(output_definition_id)
    except ValueError as exc:
        raise ValueError("output_manifest_definition_id_invalid") from exc
    if output_definition_id not in _OUTPUT_DEFINITION_FILES:
        raise ValueError("output_manifest_definition_id_invalid")

    metadata = value["metadata"]
    if type(metadata) is not dict or _metadata_contains_schema_id(metadata):
        raise ValueError("output_manifest_metadata_invalid")
    if output_definition_id == "prediction_outputs":
        validate_prediction_output_metadata_shape(metadata)
    elif output_definition_id == "evaluation_outputs":
        validate_evaluation_output_metadata_shape(metadata)
    else:
        raise ValueError("output_manifest_definition_id_invalid")
    if "run_type" in metadata and metadata[
        "run_type"
    ] != output_definition_id.removesuffix("_outputs"):
        raise ValueError("output_manifest_metadata_invalid")
    evidence_payload: dict[str, object] = {}
    if output_definition_id == "prediction_outputs":
        facts = metadata["model_facts"]
        evidence_payload = {
            "hyperparameters": facts["model"]["hyperparameters"],
            "candidate_parameters": facts["candidate_parameters"],
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
    expected_files = list(_OUTPUT_DEFINITION_FILES[output_definition_id])
    if output_definition_id == "evaluation_outputs":
        expected_files.extend(
            f"{name}.parquet"
            for name in ("daily_coverage", "daily_strata")
            if name in metadata.get("optional_member_rows", {})
        )
    if files != expected_files:
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
    definition_id: str,
    metadata: Mapping[str, object],
    files: Iterable[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": "qrp-output-manifest",
        "output_definition_id": definition_id,
        "metadata": dict(metadata),
        "files": list(files),
    }
    validate_output_manifest(payload)
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


def evaluation_output_metadata(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
    science: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        **(dict(science) if science is not None else {}),
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


def validate_output_bundle(
    root: Path,
    *,
    expected_definition_id: str | None = None,
    selected_files: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    """Read shared metadata once and check only the selected declared paths."""
    root = require_unaliased_path(Path(root), error="output_bundle_root_invalid")
    if not root.is_dir():
        raise ValueError("output_bundle_root_invalid")
    manifest = read_qrp_json_file(root / "manifest.json")[1]
    validate_output_manifest(manifest)
    if (
        expected_definition_id is not None
        and manifest["output_definition_id"] != expected_definition_id
    ):
        raise ValueError("output_manifest_definition_id_mismatch")
    if selected_files is not None and any(
        name not in manifest["files"] for name in selected_files
    ):
        raise ValueError("output_file_path_invalid")
    members = tuple(
        root / name
        for name in (manifest["files"] if selected_files is None else selected_files)
    )
    for path in members:
        require_unaliased_path(path, error="output_member_path_invalid")
        if not path.is_file():
            raise ValueError("output_member_path_invalid")
    return manifest, members
