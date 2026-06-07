from __future__ import annotations

import csv
import json

from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
    ProcessMemorySampler,
    write_execution_metrics_csv,
    write_execution_metrics_json,
)


def test_execution_metrics_recorder_writes_timings_json_and_csv(tmp_path) -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())

    with recorder.record("build_processed", logical_layer="processed") as record:
        record.output_rows = 3
        record.file_count = 1
        record.file_size_bytes = 123

    json_path = recorder.write_json(tmp_path / "timings.json", run_id="s8")
    csv_path = recorder.write_csv(tmp_path / "timings.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert payload["run_id"] == "s8"
    assert payload["summary"]["memory_sampler_unavailable"] is True
    assert payload["records"][0]["step_name"] == "build_processed"
    assert payload["records"][0]["duration_seconds"] >= 0
    assert payload["records"][0]["output_rows"] == 3
    assert payload["records"][0]["file_count"] == 1
    assert payload["records"][0]["file_size_bytes"] == 123
    assert payload["records"][0]["rss_start_mb"] is None
    assert payload["records"][0]["rss_end_mb"] is None
    assert payload["records"][0]["rss_peak_mb"] is None
    assert payload["records"][0]["rows_per_second"] is not None
    assert rows[0]["step_name"] == "build_processed"
    assert rows[0]["status"] == "success"


def test_execution_metrics_records_optional_rss_fields(tmp_path) -> None:
    sampler = ProcessMemorySampler.from_callable(lambda: 128.0)
    recorder = ExecutionMetricsRecorder(memory_sampler=sampler)

    with recorder.record("build_daily_panel") as record:
        record.input_rows = 2

    payload = recorder.payload(run_id="s9")

    assert payload["summary"]["memory_sampler_unavailable"] is False
    assert payload["records"][0]["rss_start_mb"] == 128.0
    assert payload["records"][0]["rss_end_mb"] == 128.0
    assert payload["records"][0]["rss_peak_mb"] == 128.0


def test_execution_metrics_writer_functions_keep_timings_output_names(tmp_path) -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())
    with recorder.record("validation_summary"):
        pass

    json_path = write_execution_metrics_json(
        tmp_path / "timings.json",
        recorder.records,
        run_id="s10",
        memory_sampler_unavailable=True,
    )
    csv_path = write_execution_metrics_csv(tmp_path / "timings.csv", recorder.records)

    assert json_path.name == "timings.json"
    assert csv_path.name == "timings.csv"


def test_execution_metrics_records_failure_status_and_error_code() -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())

    try:
        with recorder.record("build_factor_price_volume"):
            raise RuntimeError("synthetic failure")
    except RuntimeError:
        pass

    payload = recorder.payload()
    record = payload["records"][0]

    assert record["step_name"] == "build_factor_price_volume"
    assert record["status"] == "fail"
    assert record["error_code"] == "RuntimeError"
    assert record["duration_seconds"] >= 0


def test_execution_metrics_sampler_exception_degrades_to_null_rss() -> None:
    def broken_sampler() -> float:
        raise RuntimeError("sampler unavailable")

    recorder = ExecutionMetricsRecorder(
        memory_sampler=ProcessMemorySampler.from_callable(broken_sampler)
    )

    with recorder.record("manual_comparison"):
        pass

    payload = recorder.payload()
    record = payload["records"][0]

    assert payload["summary"]["memory_sampler_unavailable"] is True
    assert record["rss_start_mb"] is None
    assert record["rss_end_mb"] is None
    assert record["rss_peak_mb"] is None
