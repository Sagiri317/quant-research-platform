"""Build output manifests and read their saved scientific metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

EVALUATION_ARTIFACTS = (
    "metrics.parquet",
    "grouped_label_values.parquet",
    "ic_summary.parquet",
    "group_spread_summary.parquet",
)


def output_manifest_payload(
    *,
    definition_id: str,
    metadata: Mapping[str, object],
    files: Iterable[str],
) -> dict[str, object]:
    return {
        "schema_id": "qrp-output-manifest",
        "output_definition_id": definition_id,
        "metadata": dict(metadata),
        "files": list(files),
    }


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
        "market_id": str(spec.market_id),
        "research_dataset_id": spec.research_dataset_id,
        "evaluation_axis": spec.evaluation_axis,
        **(
            {"grouping_rule": "fixed_groups_shared_extreme_boundaries"}
            if spec.evaluation_axis == "cross_sectional"
            else {"grouping_rule": "frozen_reference_cutpoints"}
            if "response_diagnostics" in (science or {})
            else {}
        ),
        **(
            {"top_quantile": spec.top_quantile} if spec.top_quantile is not None else {}
        ),
        **(
            {"correlation_clock": spec.correlation_clock}
            if spec.correlation_clock != "observation"
            else {}
        ),
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


def read_output_manifest(root: Path) -> dict[str, object]:
    """Read the saved output manifest."""
    return read_qrp_json_file(Path(root) / "manifest.json")[1]
