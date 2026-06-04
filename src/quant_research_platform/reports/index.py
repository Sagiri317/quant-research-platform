"""Report run index abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.reports.summaries import ReportRunIndexRecord


@dataclass
class ReportRunIndex:
    _items: dict[str, ReportRunIndexRecord] = field(default_factory=dict)

    def register(self, record: ReportRunIndexRecord) -> None:
        if record.run_id in self._items:
            raise ValueError(f"duplicate_report_run: {record.run_id}")
        self._items[record.run_id] = record

    def get(self, run_id: str) -> ReportRunIndexRecord:
        try:
            return self._items[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown_report_run: {run_id}") from exc

    def list(self) -> tuple[ReportRunIndexRecord, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["ReportRunIndex"]
