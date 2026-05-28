"""Low-memory file report for comparing evaluation_set artifact directories."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts.artifact_comparison import (
    compare_evaluation_sets,
)


def write_evaluation_comparison_report(
    run_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    report_name: str = "evaluation_comparison",
) -> dict[str, Path]:
    """Write CSV, JSON, and Markdown comparison files without web artifacts."""

    comparison = compare_evaluation_sets(run_dirs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_name = _validate_report_name(report_name)
    summary_csv_path = destination / f"{report_name}.csv"
    combination_csv_path = destination / f"{report_name}_combinations.csv"
    kernel_parity_csv_path = destination / f"{report_name}_kernel_parity.csv"
    json_path = destination / f"{report_name}.json"
    md_path = destination / f"{report_name}.md"
    combination_comparison = _combination_comparison(run_dirs)
    kernel_parity = _kernel_parity(combination_comparison)
    comparison.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    combination_comparison.to_csv(
        combination_csv_path,
        index=False,
        encoding="utf-8-sig",
    )
    kernel_parity.to_csv(kernel_parity_csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "run_count": int(len(comparison)),
                "columns": list(comparison.columns),
                "rows": comparison.to_dict(orient="records"),
                "combination_comparison": {
                    "columns": list(combination_comparison.columns),
                    "rows": combination_comparison.to_dict(orient="records"),
                },
                "kernel_parity": {
                    "columns": list(kernel_parity.columns),
                    "rows": kernel_parity.to_dict(orient="records"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        _markdown_report(comparison, combination_comparison, kernel_parity),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_csv_path,
        "combination_csv": combination_csv_path,
        "kernel_parity_csv": kernel_parity_csv_path,
        "json": json_path,
        "markdown": md_path,
    }


def _validate_report_name(report_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", report_name):
        raise ValueError("report_name must be a safe filename stem.")
    return report_name


def _combination_comparison(run_dirs: list[str | Path]) -> pd.DataFrame:
    frames = [_read_combination_summary(Path(run_dir)) for run_dir in run_dirs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_combination_summary(run_dir: Path) -> pd.DataFrame:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    table_paths = _artifact_table_paths(manifest["artifacts"])
    combinations = pd.read_csv(run_dir / table_paths["combinations"])
    metrics_summary = pd.read_csv(run_dir / table_paths["evaluation_metrics_summary"])
    metrics = _merge_fact_table(
        combinations,
        _metrics_summary_wide(metrics_summary),
    )
    metrics.insert(0, "run_id", summary["run_id"])
    return metrics


def _metrics_summary_wide(metrics_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"combination_id", "metric_name", "metric_value"}
    missing = sorted(required.difference(metrics_summary.columns))
    if missing:
        raise ValueError(f"evaluation_metrics_summary missing columns: {missing}.")
    values = metrics_summary.copy()
    values["metric_value"] = pd.to_numeric(
        values["metric_value"],
        errors="coerce",
    )
    return (
        values.pivot_table(
            index="combination_id",
            columns="metric_name",
            values="metric_value",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(columns=None)
    )


def _merge_fact_table(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [
        key
        for key in ("combination_id", "feature", "label", "kernel", "portfolio_mode")
        if key in left.columns and key in right.columns
    ]
    if "combination_id" not in keys:
        keys.insert(0, "combination_id")
    return left.merge(right, on=keys, how="left")


def _artifact_table_paths(artifacts: dict[str, object]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for group in (
        "dimensions",
        "metrics",
        "portfolio",
        "diagnostics",
        "input_quality",
        "summaries",
    ):
        section = artifacts.get(group)
        if isinstance(section, dict):
            paths.update(
                {
                    str(name): str(path)
                    for name, path in section.items()
                    if isinstance(path, str)
                }
            )
    return paths


def _kernel_parity(combinations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_id",
        "feature",
        "label",
        "portfolio_mode",
        "metric",
        "dataframe_value",
        "tensor_value",
        "abs_delta",
    ]
    if combinations.empty or "kernel" not in combinations.columns:
        return pd.DataFrame(columns=columns)
    metrics = [
        metric
        for metric in (
            "mean_ic",
            "mean_rankic",
            "final_cumulative_net_return",
            "mean_turnover",
            "total_cost",
        )
        if metric in combinations.columns
    ]
    rows: list[dict[str, object]] = []
    grouping = ["run_id", "feature", "label", "portfolio_mode"]
    for key, group in combinations.groupby(grouping, dropna=False, sort=False):
        by_kernel = {
            str(row["kernel"]): row
            for _, row in group.iterrows()
            if pd.notna(row.get("kernel"))
        }
        if "dataframe" not in by_kernel or "tensor" not in by_kernel:
            continue
        for metric in metrics:
            dataframe_value = pd.to_numeric(
                pd.Series([by_kernel["dataframe"].get(metric)]),
                errors="coerce",
            ).iloc[0]
            tensor_value = pd.to_numeric(
                pd.Series([by_kernel["tensor"].get(metric)]),
                errors="coerce",
            ).iloc[0]
            rows.append(
                {
                    "run_id": key[0],
                    "feature": key[1],
                    "label": key[2],
                    "portfolio_mode": key[3],
                    "metric": metric,
                    "dataframe_value": dataframe_value,
                    "tensor_value": tensor_value,
                    "abs_delta": abs(dataframe_value - tensor_value)
                    if pd.notna(dataframe_value) and pd.notna(tensor_value)
                    else pd.NA,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _markdown_report(
    comparison: pd.DataFrame,
    combination_comparison: pd.DataFrame,
    kernel_parity: pd.DataFrame,
) -> str:
    lines = [
        "# Evaluation Comparison Report",
        "",
        f"- runs: {len(comparison)}",
        "",
    ]
    if comparison.empty:
        return "\n".join(lines)
    columns = [
        column
        for column in (
            "run_id",
            "mean_ic",
            "mean_rankic",
            "mean_final_return",
            "mean_turnover",
            "total_cost",
        )
        if column in comparison.columns
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in comparison[columns].to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    lines.append("")
    lines.append("## Combination Comparison")
    lines.append("")
    lines.append(f"- rows: {len(combination_comparison)}")
    lines.append("")
    lines.append("## Kernel Parity")
    lines.append("")
    lines.append(f"- rows: {len(kernel_parity)}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["write_evaluation_comparison_report"]
