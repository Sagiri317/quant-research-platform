"""Execution metrics renderers."""

from __future__ import annotations

import pandas as pd


def render_stage_profile_report(frame: pd.DataFrame) -> str:
    """Render stage-profile records as Markdown."""

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


__all__ = ["format_optional_mb", "render_stage_profile_report"]
