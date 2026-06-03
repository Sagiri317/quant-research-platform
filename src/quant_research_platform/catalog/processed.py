"""Interim and processed data catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.data import InterimDataSpec, ProcessedDataSpec


@dataclass
class InterimDataCatalog:
    _items: dict[str, InterimDataSpec] = field(default_factory=dict)

    def register(self, spec: InterimDataSpec) -> None:
        if spec.interim_data_id in self._items:
            raise ValueError(f"duplicate_interim_data_spec: {spec.interim_data_id}")
        self._items[spec.interim_data_id] = spec

    def get(self, interim_data_id: str) -> InterimDataSpec:
        try:
            return self._items[interim_data_id]
        except KeyError as exc:
            raise KeyError(f"unknown_interim_data_spec: {interim_data_id}") from exc

    def list(self) -> tuple[InterimDataSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


@dataclass
class ProcessedDataCatalog:
    _items: dict[str, ProcessedDataSpec] = field(default_factory=dict)

    def register(self, spec: ProcessedDataSpec) -> None:
        if spec.processed_data_id in self._items:
            raise ValueError(f"duplicate_processed_data_spec: {spec.processed_data_id}")
        self._items[spec.processed_data_id] = spec

    def get(self, processed_data_id: str) -> ProcessedDataSpec:
        try:
            return self._items[processed_data_id]
        except KeyError as exc:
            raise KeyError(f"unknown_processed_data_spec: {processed_data_id}") from exc

    def list(self) -> tuple[ProcessedDataSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["InterimDataCatalog", "ProcessedDataCatalog"]
