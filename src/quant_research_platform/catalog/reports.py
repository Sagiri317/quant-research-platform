"""Report definition catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.reports import ReportDefinition


@dataclass
class ReportCatalog:
    _items: dict[str, ReportDefinition] = field(default_factory=dict)

    def register(self, definition: ReportDefinition) -> None:
        if definition.report_definition_id in self._items:
            raise ValueError(
                f"duplicate_report_definition: {definition.report_definition_id}"
            )
        self._items[definition.report_definition_id] = definition

    def get(self, report_definition_id: str) -> ReportDefinition:
        try:
            return self._items[report_definition_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown_report_definition: {report_definition_id}"
            ) from exc

    def list(self) -> tuple[ReportDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["ReportCatalog"]
