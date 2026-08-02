from __future__ import annotations

import time

import pytest

import quant_research_platform.runtime.execution_metrics as execution_metrics
from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
    ProcessMemorySampler,
)


def test_execution_metrics_recorder_exposes_payload_and_frame() -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())

    with recorder.record("build_processed", logical_layer="processed") as record:
        record.output_rows = 3
        record.file_count = 1
        record.file_size_bytes = 123

    payload = recorder.payload(run_id="s8")
    runtime_record = recorder.records[0]

    assert payload["run_id"] == "s8"
    assert payload["summary"]["memory_sampler_unavailable"] is True
    assert payload["records"][0]["step_name"] == "build_processed"
    assert payload["records"][0]["duration_seconds"] >= 0
    assert payload["records"][0]["output_rows"] == 3
    assert payload["records"][0]["file_count"] == 1
    assert payload["records"][0]["file_size_bytes"] == 123
    assert payload["records"][0]["rss_start_mb"] is None
    assert payload["records"][0]["rss_end_mb"] is None
    assert payload["records"][0]["rss_start_end_max_mb"] is None
    assert payload["records"][0]["rows_per_second"] is not None
    assert runtime_record.step_name == "build_processed"
    assert runtime_record.status == "success"
    assert runtime_record.rss_start_end_max_mb is None
    assert runtime_record.rss_sampled_max_mb is None
    assert runtime_record.rss_sample_interval_seconds is None
    assert runtime_record.rss_sample_count == 0
    assert runtime_record.rss_sampler_unavailable is True


def test_execution_metrics_records_sampled_rss_fields(monkeypatch) -> None:
    values = iter([100.0, 110.0, 120.0, 130.0, 140.0])
    sampler = ProcessMemorySampler.from_callable(lambda: next(values, 140.0))
    monkeypatch.setenv("QRP_RSS_SAMPLE_INTERVAL_SECONDS", "0.001")
    recorder = ExecutionMetricsRecorder(memory_sampler=sampler)

    with recorder.record("build_market_panel") as record:
        record.input_rows = 2
        time.sleep(0.01)

    payload = recorder.payload(run_id="s9")

    record = payload["records"][0]
    assert payload["summary"]["memory_sampler_unavailable"] is False
    assert record["rss_sampled_max_mb"] >= 100.0
    assert record["rss_sample_interval_seconds"] == 0.001
    assert record["rss_sample_count"] >= 2
    assert record["rss_sampler_unavailable"] is False
    runtime_record = recorder.records[0]
    assert runtime_record.rss_sampled_max_mb
    assert runtime_record.rss_sample_interval_seconds == 0.001
    assert runtime_record.rss_sample_count >= 2
    assert runtime_record.rss_sampler_unavailable is False


def test_execution_metrics_keeps_start_end_rss_fields() -> None:
    sampler = ProcessMemorySampler.from_callable(lambda: 128.0)
    recorder = ExecutionMetricsRecorder(memory_sampler=sampler)

    with recorder.record("build_market_panel") as record:
        record.input_rows = 2

    payload = recorder.payload(run_id="s9")

    assert payload["summary"]["memory_sampler_unavailable"] is False
    assert payload["records"][0]["rss_start_mb"] == 128.0
    assert payload["records"][0]["rss_end_mb"] == 128.0
    assert payload["records"][0]["rss_start_end_max_mb"] == 128.0


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
    assert record["rss_start_end_max_mb"] is None
    assert record["rss_sampled_max_mb"] is None
    assert record["rss_sample_interval_seconds"] is None
    assert record["rss_sample_count"] == 0
    assert record["rss_sampler_unavailable"] is True


def test_execution_metrics_sampler_unavailable_serializes_aggregate_fields() -> None:
    recorder = ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())

    with recorder.record("manual_comparison"):
        pass

    record = recorder.payload()["records"][0]

    assert record["rss_sampled_max_mb"] is None
    assert record["rss_sample_interval_seconds"] is None
    assert record["rss_sample_count"] == 0
    assert record["rss_sampler_unavailable"] is True


def test_invalid_rss_sample_interval_env_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("QRP_RSS_SAMPLE_INTERVAL_SECONDS", "not-a-number")

    with pytest.raises(ValueError, match="invalid_rss_sample_interval_seconds"):
        ExecutionMetricsRecorder(memory_sampler=ProcessMemorySampler.disabled())


def test_rss_sampler_thread_unavailable_path_serializes_true(monkeypatch) -> None:
    class BrokenThread:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("thread unavailable")

    monkeypatch.setattr(execution_metrics.threading, "Thread", BrokenThread)
    recorder = ExecutionMetricsRecorder(
        memory_sampler=ProcessMemorySampler.from_callable(lambda: 128.0)
    )

    with recorder.record("manual_comparison"):
        pass

    record = recorder.payload()["records"][0]

    assert record["rss_sampler_unavailable"] is True
    assert record["rss_sample_count"] == 0
