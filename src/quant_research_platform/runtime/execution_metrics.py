"""Execution metrics helpers for verification evidence runs."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_research_platform.storage.atomic import atomic_write_csv, atomic_write_json


@dataclass
class ExecutionStepMetric:
    step_name: str
    logical_layer: str | None = None
    dataset_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    rss_start_mb: float | None = None
    rss_end_mb: float | None = None
    rss_start_end_max_mb: float | None = None
    rss_sampled_max_mb: float | None = None
    rss_sample_interval_seconds: float | None = None
    rss_sample_count: int = 0
    rss_sampler_unavailable: bool = False
    input_rows: int | None = None
    output_rows: int | None = None
    file_count: int | None = None
    file_size_bytes: int | None = None
    partitions: int | None = None
    rows_per_second: float | None = None
    status: str = "success"
    error_code: str | None = None
    unavailable_reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "step_name": self.step_name,
            "logical_layer": self.logical_layer,
            "dataset_id": self.dataset_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "rss_start_mb": self.rss_start_mb,
            "rss_end_mb": self.rss_end_mb,
            "rss_start_end_max_mb": self.rss_start_end_max_mb,
            "rss_sampled_max_mb": self.rss_sampled_max_mb,
            "rss_sample_interval_seconds": self.rss_sample_interval_seconds,
            "rss_sample_count": self.rss_sample_count,
            "rss_sampler_unavailable": self.rss_sampler_unavailable,
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "file_count": self.file_count,
            "file_size_bytes": self.file_size_bytes,
            "partitions": self.partitions,
            "rows_per_second": self.rows_per_second,
            "status": self.status,
            "error_code": self.error_code,
            "unavailable_reason": self.unavailable_reason,
        }


class ProcessMemorySampler:
    def __init__(
        self,
        sampler: Callable[[], float | None] | None,
        *,
        unavailable: bool = False,
    ) -> None:
        self._sampler = sampler
        self.unavailable = unavailable

    @classmethod
    def create(cls) -> ProcessMemorySampler:
        try:
            import psutil  # type: ignore[import-not-found]
        except Exception:
            return cls.disabled()
        process = psutil.Process()
        return cls(lambda: process.memory_info().rss / (1024 * 1024))

    @classmethod
    def disabled(cls) -> ProcessMemorySampler:
        return cls(None, unavailable=True)

    @classmethod
    def from_callable(
        cls,
        sampler: Callable[[], float | None],
    ) -> ProcessMemorySampler:
        return cls(sampler)

    def sample_mb(self) -> float | None:
        if self._sampler is None:
            self.unavailable = True
            return None
        try:
            value = self._sampler()
        except Exception:
            self.unavailable = True
            return None
        if value is None:
            self.unavailable = True
            return None
        return round(float(value), 6)


@dataclass(frozen=True)
class _RssSampleAggregate:
    sampled_max_mb: float | None
    sample_interval_seconds: float | None
    sample_count: int
    sampler_unavailable: bool


class _StepRssSampler:
    def __init__(
        self,
        memory_sampler: ProcessMemorySampler,
        *,
        interval_seconds: float | None,
    ) -> None:
        self._memory_sampler = memory_sampler
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sampled_max_mb: float | None = None
        self._sample_count = 0
        self._thread_unavailable = False

    def start(self) -> None:
        if self._interval_seconds is None:
            return
        try:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception:
            self._thread_unavailable = True

    def stop(self) -> _RssSampleAggregate:
        if self._interval_seconds is None:
            return _RssSampleAggregate(
                sampled_max_mb=None,
                sample_interval_seconds=None,
                sample_count=0,
                sampler_unavailable=True,
            )
        if self._thread_unavailable:
            return _RssSampleAggregate(
                sampled_max_mb=None,
                sample_interval_seconds=self._interval_seconds,
                sample_count=0,
                sampler_unavailable=True,
            )
        self._record_sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds * 2))
        with self._lock:
            return _RssSampleAggregate(
                sampled_max_mb=self._sampled_max_mb,
                sample_interval_seconds=self._interval_seconds,
                sample_count=self._sample_count,
                sampler_unavailable=self._memory_sampler.unavailable,
            )

    def _run(self) -> None:
        self._record_sample()
        while not self._stop.wait(self._interval_seconds):
            self._record_sample()

    def _record_sample(self) -> None:
        value = self._memory_sampler.sample_mb()
        if value is None:
            return
        with self._lock:
            self._sample_count += 1
            if self._sampled_max_mb is None:
                self._sampled_max_mb = value
            else:
                self._sampled_max_mb = max(self._sampled_max_mb, value)


class ExecutionMetricsRecorder:
    def __init__(self, memory_sampler: ProcessMemorySampler | None = None) -> None:
        self.records: list[ExecutionStepMetric] = []
        self.memory_sampler = memory_sampler or ProcessMemorySampler.create()
        self.rss_sample_interval_seconds = _rss_sample_interval_seconds()

    @contextmanager
    def record(
        self,
        step_name: str,
        *,
        logical_layer: str | None = None,
        dataset_id: str | None = None,
    ) -> Iterator[ExecutionStepMetric]:
        started = datetime.now(UTC)
        rss_start = self.memory_sampler.sample_mb()
        rss_sampler = _StepRssSampler(
            self.memory_sampler,
            interval_seconds=(
                None
                if self.memory_sampler.unavailable
                else self.rss_sample_interval_seconds
            ),
        )
        rss_sampler.start()
        item = ExecutionStepMetric(
            step_name=step_name,
            logical_layer=logical_layer,
            dataset_id=dataset_id,
            start_time=_format_time(started),
            rss_start_mb=rss_start,
            rss_start_end_max_mb=rss_start,
            rss_sample_interval_seconds=(
                None
                if self.memory_sampler.unavailable
                else self.rss_sample_interval_seconds
            ),
            rss_sampler_unavailable=self.memory_sampler.unavailable,
        )
        try:
            yield item
        except Exception as exc:
            item.status = "fail"
            item.error_code = exc.__class__.__name__
            raise
        finally:
            ended = datetime.now(UTC)
            rss_end = self.memory_sampler.sample_mb()
            item.end_time = _format_time(ended)
            item.duration_seconds = round(
                max(0.0, (ended - started).total_seconds()),
                6,
            )
            item.rss_end_mb = rss_end
            rss_aggregate = rss_sampler.stop()
            item.rss_sampled_max_mb = rss_aggregate.sampled_max_mb
            item.rss_sample_interval_seconds = (
                rss_aggregate.sample_interval_seconds
            )
            item.rss_sample_count = rss_aggregate.sample_count
            item.rss_sampler_unavailable = rss_aggregate.sampler_unavailable
            rss_values = [
                value
                for value in (item.rss_start_mb, item.rss_end_mb)
                if value is not None
            ]
            item.rss_start_end_max_mb = max(rss_values) if rss_values else None
            rows = item.output_rows if item.output_rows is not None else item.input_rows
            if rows is not None and item.duration_seconds is not None:
                divisor = item.duration_seconds or 1e-9
                item.rows_per_second = round(float(rows) / divisor, 6)
            if item.input_rows is None and item.output_rows is None:
                item.unavailable_reason = "row_counts_unavailable"
            self.records.append(item)

    def payload(self, *, run_id: str | None = None) -> dict[str, object]:
        return _payload(
            self.records,
            run_id=run_id,
            memory_sampler_unavailable=self.memory_sampler.unavailable,
        )

    def write_json(self, path: Path, *, run_id: str | None = None) -> Path:
        return write_execution_metrics_json(
            path,
            self.records,
            run_id=run_id,
            memory_sampler_unavailable=self.memory_sampler.unavailable,
        )

    def write_csv(self, path: Path) -> Path:
        return write_execution_metrics_csv(path, self.records)


def write_execution_metrics_json(
    path: Path,
    records: list[ExecutionStepMetric],
    *,
    run_id: str | None = None,
    memory_sampler_unavailable: bool = False,
) -> Path:
    return atomic_write_json(
        path,
        _payload(
            records,
            run_id=run_id,
            memory_sampler_unavailable=memory_sampler_unavailable,
        ),
    )


def write_execution_metrics_csv(path: Path, records: list[ExecutionStepMetric]) -> Path:
    frame = pd.DataFrame([record.to_payload() for record in records])
    return atomic_write_csv(frame, path, index=False)


def _payload(
    records: list[ExecutionStepMetric],
    *,
    run_id: str | None,
    memory_sampler_unavailable: bool,
) -> dict[str, object]:
    total = _total_duration_seconds(records)
    payload = {
        "total_duration_seconds": round(total, 6),
        "summary": {
            "step_count": len(records),
            "memory_sampler_unavailable": bool(memory_sampler_unavailable),
        },
        "records": [record.to_payload() for record in records],
    }
    if run_id is not None:
        payload["run_id"] = run_id
    return payload


def _total_duration_seconds(records: list[ExecutionStepMetric]) -> float:
    processed_child_steps = {
        "build_market_panel_1d",
        "build_context",
        "build_universe",
        "build_masks",
        "build_factor_price_volume",
        "build_label_forward_returns",
    }
    has_processed_parent = any(
        record.step_name == "processed_generation" for record in records
    )
    return sum(
        record.duration_seconds or 0.0
        for record in records
        if not (has_processed_parent and record.step_name in processed_child_steps)
    )


def _format_time(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rss_sample_interval_seconds() -> float:
    raw = os.environ.get("QRP_RSS_SAMPLE_INTERVAL_SECONDS")
    if raw is None or raw == "":
        return 0.5
    try:
        interval = float(raw)
    except ValueError as exc:
        raise ValueError("invalid_rss_sample_interval_seconds") from exc
    if interval <= 0:
        raise ValueError("invalid_rss_sample_interval_seconds")
    return interval


__all__ = [
    "ExecutionMetricsRecorder",
    "ExecutionStepMetric",
    "ProcessMemorySampler",
    "write_execution_metrics_csv",
    "write_execution_metrics_json",
]
