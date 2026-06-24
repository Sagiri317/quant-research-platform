"""Artifact writers for execution metrics."""

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
    """Write the recorder payload as JSON."""

    return atomic_write_json(path, recorder.payload(run_id=run_id))


def write_execution_metrics_csv(
    path: Path,
    recorder: ExecutionMetricsRecorder,
) -> Path:
    """Write recorder rows as a CSV report artifact."""

    frame = pd.DataFrame([record.to_payload() for record in recorder.records])
    return atomic_write_csv(frame, path, index=False)


__all__ = [
    "write_execution_metrics_csv",
    "write_execution_metrics_json",
]
