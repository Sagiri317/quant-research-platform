"""Local verification run summary renderers."""

from __future__ import annotations

from typing import Any


def render_run_summary_markdown(payload: dict[str, Any]) -> str:
    status = payload["status"]
    lines = [
        "# Local Verification Run Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- overall: `{status['overall']}`",
        f"- validation: `{status['validation']}`",
        f"- smoke: `{status['smoke']}`",
        f"- compare: `{status['compare']}`",
        f"- share_safety: `{status['share_safety']}`",
        f"- full_horizon: `{status['full_horizon']}`",
        f"- validation_matrix: `{status['validation_matrix']}`",
        f"- data_layers: `{len(payload['data']['layers'])}`",
        f"- outputs: `{len(payload['outputs'])}`",
        "",
    ]
    return "\n".join(lines)
