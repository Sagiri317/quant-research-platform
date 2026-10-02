"""Evaluation report content."""

from __future__ import annotations

from numbers import Real

import pandas as pd

from quant_research_platform.contracts.evaluations import (
    EvaluationDiagnostics,
    EvaluationSpec,
)


def render_evaluation_summary(
    *,
    spec: EvaluationSpec,
    evaluation_diagnostics: EvaluationDiagnostics,
    ic_summary: pd.DataFrame,
    group_spread_summary: pd.DataFrame,
    coverage_summary: pd.DataFrame,
) -> str:
    signals = ", ".join(signal.signal_id for signal in spec.signals)
    labels = ", ".join(spec.label_columns)
    time_series = spec.evaluation_axis == "single_asset_time_series"
    lines = [
        "# Evaluation Summary",
        "",
        f"- evaluation_id: `{spec.evaluation_id}`",
        f"- evaluation_axis: `{spec.evaluation_axis}`",
        f"- market_id: `{spec.market_id}`",
        f"- research_dataset_id: `{spec.research_dataset_id}`",
        f"- signals: `{signals}`",
        f"- labels: `{labels}`",
        f"- rows_input: `{evaluation_diagnostics.rows_input:,}`",
        f"- rows_after_filter: `{evaluation_diagnostics.rows_after_filter_mode:,}`",
        f"- timestamp_count: `{evaluation_diagnostics.timestamp_count:,}`",
        f"- min_timestamp: `{evaluation_diagnostics.min_timestamp}`",
        f"- max_timestamp: `{evaluation_diagnostics.max_timestamp}`",
    ]
    if time_series:
        lines.extend(
            [
                "- RankIC: `not_applicable`",
                "- group_spread: `not_applicable`",
            ]
        )
    lines.extend(
        [
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
                coverage_summary,
                (
                    "signal",
                    "label",
                    "mean_n_obs",
                    "min_n_obs",
                    "non_empty_period_count",
                ),
            ),
            "",
            "## Group Spread Summary",
            "",
            (
                "`not_applicable`"
                if time_series
                else _markdown_table(
                    group_spread_summary,
                    ("signal", "label", "top_bottom_label_value_spread_mean"),
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.loc[:, list(columns)].iterrows():
        rows.append(
            "| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |"
        )
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value).replace("|", r"\|")
