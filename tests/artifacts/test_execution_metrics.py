from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.execution_metrics import (
    write_execution_metrics_json,
)
from quant_research_platform.runtime.execution_metrics import ExecutionMetricsRecorder


def test_execution_metrics_writer_persists_recorder_payload(tmp_path: Path) -> None:
    recorder = ExecutionMetricsRecorder()

    path = write_execution_metrics_json(tmp_path / "timings.json", recorder)

    assert path.is_file()
