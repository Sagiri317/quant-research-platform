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
    strict_non_negative_count_series,
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
        (temp_root / "summary.md").write_text(
            _render_evaluation_summary(
                spec=spec,
                evaluation_diagnostics=evaluation_diagnostics,
                metrics=metrics,
                ic_summary=ic_summary,
                group_spread_summary=group_spread_summary,
            ),
            encoding="utf-8",
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


def _render_evaluation_summary(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
    metrics: pd.DataFrame,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
) -> str:
    signals = ", ".join(signal.signal_id for signal in spec.signals)
    labels = ", ".join(spec.label_columns)
    lines = [
        "# Evaluation Summary",
        "",
        f"- run_id: `{spec.run_id}`",
        f"- market_id: `{spec.market_id}`",
        f"- research_dataset_id: `{spec.research_dataset_id}`",
        f"- signals: `{signals}`",
        f"- labels: `{labels}`",
        f"- rows_input: `{evaluation_diagnostics.rows_input}`",
        f"- rows_after_filter: `{evaluation_diagnostics.rows_after_filter_mode}`",
        f"- timestamp_count: `{evaluation_diagnostics.timestamp_count}`",
        f"- min_timestamp: `{evaluation_diagnostics.min_timestamp}`",
        f"- max_timestamp: `{evaluation_diagnostics.max_timestamp}`",
        "",
        "## IC Summary",
        "",
        _markdown_table(
            ic_summary,
            ("signal", "label", "mean_ic", "mean_rankic"),
        ),
        "",
        "## Signal/Label Coverage Summary",
        "",
        _markdown_table(
            _signal_label_coverage_summary(metrics),
            ("signal", "label", "mean_n_obs", "min_n_obs", "non_empty_period_count"),
        ),
        "",
        "## Group Spread Summary",
        "",
        _markdown_table(
            group_spread_summary,
            ("signal", "label", "top_bottom_label_value_spread_mean"),
        ),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.loc[:, list(columns)].iterrows():
        rows.append(
            "| "
            + " | ".join(_markdown_cell(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", r"\|")


def _signal_label_coverage_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["signal", "label", "mean_n_obs", "min_n_obs", "non_empty_period_count"]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    source = metrics.copy()
    if "n_obs" not in source:
        return source.loc[:, ["signal", "label"]].drop_duplicates().assign(
            mean_n_obs=pd.NA,
            min_n_obs=pd.NA,
            non_empty_period_count=pd.NA,
        )
    source["n_obs"] = strict_non_negative_count_series(source["n_obs"])
    grouped = source.groupby(["signal", "label"], as_index=False, sort=True)["n_obs"]
    summary = grouped.agg(
        mean_n_obs="mean",
        min_n_obs="min",
        non_empty_period_count=lambda values: int(values.gt(0).sum()),
    )
    return summary.loc[:, columns]
