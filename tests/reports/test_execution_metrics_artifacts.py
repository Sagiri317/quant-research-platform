from __future__ import annotations

import json

import pandas as pd

from quant_research_platform.reports.execution_metrics_artifacts import (
    write_execution_metrics_csv,
    write_execution_metrics_json,
)
from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
    ProcessMemorySampler,
)


def test_execution_metrics_artifact_writers_emit_json_and_csv(tmp_path) -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())
    with recorder.record("build_panel") as metric:
        metric.output_rows = 3

    json_path = write_execution_metrics_json(
        tmp_path / "timings.json",
        recorder,
        run_id="run_demo",
    )
    csv_path = write_execution_metrics_csv(tmp_path / "timings.csv", recorder)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    assert payload["run_id"] == "run_demo"
    assert payload["summary"]["step_count"] == 1
    assert frame["step_name"].tolist() == ["build_panel"]
    assert json_path.name == "timings.json"
    assert csv_path.name == "timings.csv"
    assert "rss_start_end_max_mb" in frame.columns
    assert "rss_sampled_max_mb" in frame.columns
    assert "rss_sample_interval_seconds" in frame.columns
    assert "rss_sample_count" in frame.columns
    assert "rss_sampler_unavailable" in frame.columns
