"""Timing helpers for private evidence runs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quant_research_platform.storage.atomic import atomic_write_csv, atomic_write_json


@dataclass
class TimingRecord:
    step_name: str
    logical_layer: str | None = None
    dataset_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None
    input_rows: int | None = None
    output_rows: int | None = None
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
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "partitions": self.partitions,
            "rows_per_second": self.rows_per_second,
            "status": self.status,
            "error_code": self.error_code,
            "unavailable_reason": self.unavailable_reason,
        }


class TimingRecorder:
    def __init__(self) -> None:
        self.records: list[TimingRecord] = []

    @contextmanager
    def record(
        self,
        step_name: str,
        *,
        logical_layer: str | None = None,
        dataset_id: str | None = None,
    ) -> Iterator[TimingRecord]:
        started = datetime.now(UTC)
        item = TimingRecord(
            step_name=step_name,
            logical_layer=logical_layer,
            dataset_id=dataset_id,
            start_time=_format_time(started),
        )
        try:
            yield item
        except Exception as exc:
            item.status = "fail"
            item.error_code = exc.__class__.__name__
            raise
        finally:
            ended = datetime.now(UTC)
            item.end_time = _format_time(ended)
            item.duration_seconds = max(0.0, (ended - started).total_seconds())
            rows = item.output_rows if item.output_rows is not None else item.input_rows
            if rows is not None and item.duration_seconds is not None:
                divisor = item.duration_seconds or 1e-9
                item.rows_per_second = round(float(rows) / divisor, 6)
            if item.input_rows is None and item.output_rows is None:
                item.unavailable_reason = "row_counts_unavailable"
            self.records.append(item)

    def payload(self) -> dict[str, object]:
        total = sum(record.duration_seconds or 0.0 for record in self.records)
        return {
            "total_duration_seconds": round(total, 6),
            "records": [record.to_payload() for record in self.records],
        }

    def write_json(self, path: Path) -> Path:
        return write_timings_json(path, self.records)

    def write_csv(self, path: Path) -> Path:
        return write_timings_csv(path, self.records)


def write_timings_json(path: Path, records: list[TimingRecord]) -> Path:
    total = sum(record.duration_seconds or 0.0 for record in records)
    payload = {
        "total_duration_seconds": round(total, 6),
        "records": [record.to_payload() for record in records],
    }
    return atomic_write_json(path, payload)


def write_timings_csv(path: Path, records: list[TimingRecord]) -> Path:
    frame = pd.DataFrame([record.to_payload() for record in records])
    return atomic_write_csv(frame, path, index=False)


def _format_time(value: datetime) -> str:
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "TimingRecord",
    "TimingRecorder",
    "write_timings_csv",
    "write_timings_json",
]
