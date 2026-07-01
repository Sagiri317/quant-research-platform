"""Evaluation output writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts._evaluation_output_validation import (
    EVALUATION_ARTIFACTS,
    EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
    EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
    EVALUATION_IC_SUMMARY_COLUMNS,
    EVALUATION_METRICS_COLUMNS,
    validate_evaluation_artifact_structure,
    validate_evaluation_diagnostics_consistency,
)
from quant_research_platform.artifacts._output_common_validation import (
    derive_output_bundle_root,
    validate_output_frame,
)
from quant_research_platform.artifacts._output_manifest import (
    evaluation_output_metadata,
    output_manifest_payload,
)
from quant_research_platform.artifacts._parquet_writers import write_parquet
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

    output_bundle = derive_output_bundle_root(
        output_spec=output_spec,
        layout=layout,
        expected_definition_id="evaluation_outputs",
    )

    validate_output_frame(
        metrics,
        required=EVALUATION_METRICS_COLUMNS,
        error_name="evaluation_metrics",
    )
    validate_output_frame(
        grouped_label_values,
        required=EVALUATION_GROUPED_LABEL_VALUE_COLUMNS,
        error_name="evaluation_grouped_label_values",
    )
    validate_output_frame(
        ic_summary,
        required=EVALUATION_IC_SUMMARY_COLUMNS,
        error_name="evaluation_ic_summary",
    )
    validate_output_frame(
        group_spread_summary,
        required=EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS,
        error_name="evaluation_group_spread_summary",
    )
    validate_evaluation_artifact_structure(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        spec=spec,
        n_groups=evaluation_diagnostics.n_groups,
    )
    validate_evaluation_diagnostics_consistency(
        metrics=metrics,
        grouped_label_values=grouped_label_values,
        ic_summary=ic_summary,
        group_spread_summary=group_spread_summary,
        evaluation_diagnostics=evaluation_diagnostics,
        spec=spec,
    )

    def _write(temp_root: Path) -> None:
        write_parquet(
            metrics,
            temp_root / "metrics.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            grouped_label_values,
            temp_root / "grouped_label_values.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            ic_summary,
            temp_root / "ic_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_parquet(
            group_spread_summary,
            temp_root / "group_spread_summary.parquet",
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        write_manifest(
            temp_root / "manifest.json",
            output_manifest_payload(
                output_spec=output_spec,
                metadata=evaluation_output_metadata(
                    spec=spec,
                    evaluation_diagnostics=evaluation_diagnostics,
                ),
                files=EVALUATION_ARTIFACTS,
            ),
        )

    atomic_write_directory(output_bundle, _write)
    return output_bundle
