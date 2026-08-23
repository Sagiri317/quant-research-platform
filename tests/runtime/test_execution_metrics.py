from __future__ import annotations

import time

from quant_research_platform.runtime.execution_metrics import ExecutionMetricsRecorder


def test_execution_metrics_records_duration_rows_and_throughput() -> None:
    recorder = ExecutionMetricsRecorder()

    with recorder.record("build_processed", logical_layer="processed") as record:
        record.output_rows = 3
        record.file_count = 1
        record.file_size_bytes = 123
        time.sleep(0.001)

    payload = recorder.payload(run_id="s8")
    runtime_record = recorder.records[0]
    record = payload["records"][0]

    assert payload["run_id"] == "s8"
    assert payload["summary"] == {"step_count": 1}
    assert record["step_name"] == "build_processed"
    assert record["duration_seconds"] >= 0
    assert record["output_rows"] == 3
    assert record["file_count"] == 1
    assert record["file_size_bytes"] == 123
    assert record["rows_per_second"] is not None
    assert runtime_record.status == "success"


def test_execution_metrics_records_failure_status_and_error_code() -> None:
    recorder = ExecutionMetricsRecorder()

    try:
        with recorder.record("build_factor_price_volume"):
            raise RuntimeError("synthetic failure")
    except RuntimeError:
        pass

    record = recorder.payload()["records"][0]
    assert record["status"] == "fail"
    assert record["error_code"] == "RuntimeError"
    assert record["duration_seconds"] >= 0


def test_execution_metrics_marks_missing_rows() -> None:
    recorder = ExecutionMetricsRecorder()
    with recorder.record("manual_comparison"):
        pass

    record = recorder.payload()["records"][0]
    assert record["unavailable_reason"] == "row_counts_unavailable"
