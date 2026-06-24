"""Report-owned writers for execution-metrics artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
)
from quant_research_platform.storage.atomic import atomic_write_csv, atomic_write_json


def write_execution_metrics_json(
    path: Path,
    recorder: ExecutionMetricsRecorder,
    *,
    run_id: str | None = None,
) -> Path:
    """Write the recorder payload as a JSON report artifact."""

    return atomic_write_json(path, recorder.payload(run_id=run_id))


def write_execution_metrics_csv(
    path: Path,
    recorder: ExecutionMetricsRecorder,
) -> Path:
    """Write recorder rows as a CSV report artifact."""

    frame = pd.DataFrame([record.to_payload() for record in recorder.records])
    return atomic_write_csv(frame, path, index=False)


def render_stage_profile_report(frame: pd.DataFrame) -> str:
    """Render stage-profile records as report-owned Markdown."""

    lines = [
        "# Stage Profile Report",
        "",
        "This report records stage-level timing and optional RSS observations.",
        "",
        "## Stage Summary",
        "",
    ]
    if frame.empty:
        lines.append(
            "- No stages were recorded; ProfileConfig.enabled is likely false."
        )
    else:
        for row in frame.to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['category']}.{row['stage']}: "
                f"{float(row['seconds']):.6f}s, "
                f"{int(row['n_calls'])} calls, "
                f"peak_rss={format_optional_mb(row.get('peak_rss_mb'))}"
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Profiling records runtime observations only and does not perform "
            "research calculations or access data providers.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional_mb(value: object) -> str:
    """Format an optional RSS value for report text."""

    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.3f} MB"


__all__ = [
    "format_optional_mb",
    "render_stage_profile_report",
    "write_execution_metrics_csv",
    "write_execution_metrics_json",
]
