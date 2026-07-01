"""Evaluation output writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts._evaluation_output_validation import (
    _EVALUATION_ARTIFACTS,
    _EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    _EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    _EVALUATION_IC_SUMMARY_COLUMNS,
    _EVALUATION_METRICS_COLUMNS,
    _validate_evaluation_artifact_structure,
    _validate_evaluation_diagnostics_consistency,
)
from quant_research_platform.artifacts._output_common_validation import (
    _derive_output_bundle_root,
    _validate_output_frame,
)
from quant_research_platform.artifacts._output_manifest import (
    _evaluation_output_metadata,
    _output_manifest_payload,
)
from quant_research_platform.artifacts._parquet_writers import _write_parquet
from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest


def write_evaluation_outputs(
    metrics: pd.DataFrame,
    grouped_label_values: pd.DataFrame,
    *,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    evaluation_diagnostics: EvaluationDiagnostics,
    spec: EvaluationSpec,
    output_spec: OutputSpec,
    layout: StorageLayout,
) -> Path:
    """Write local evaluation outputs into an output bundle."""

    output_bundle = _derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="evaluation_outputs",
    )

    _validate_output_frame(
        metrics,
        required=_EVALUATION_METRICS_COLUMNS,
        error_name="evaluation_metrics",
    )
    _validate_output_frame(
        grouped_label_values,
        required=_EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        error_name="evaluation_grouped_label_values",
    )
    _validate_output_frame(
        ic_summary,
        required=_EVALUATION_IC_SUMMARY_COLUMNS,
        error_name="evaluation_ic_summary",
    )
    _validate_output_frame(
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
        _write_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            grouped_label_values,
            temp_root / "grouped_label_values.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        _write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            _output_manifest_payload(
                output_spec=output_spec,
                metadata=_evaluation_output_metadata(
                    spec=spec,
                    evaluation_diagnostics=evaluation_diagnostics,
                ),
                files=_EVALUATION_ARTIFACTS,
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle
