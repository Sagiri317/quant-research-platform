"""Interim and processed data catalogs."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.data import InterimDataSpec, ProcessedDataSpec


class InterimDataCatalog(ImmutableCatalog[InterimDataSpec]):
    def __init__(self, items: Iterable[InterimDataSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.data_id in seen:
                raise ValueError(f"duplicate_interim_data_spec: {spec.data_id}")
            seen.add(spec.data_id)
        super().__init__(sorted(values, key=lambda spec: spec.data_id))

    def get(self, data_id: str) -> InterimDataSpec:
        try:
            return next(spec for spec in self._items if spec.data_id == data_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_interim_data_spec: {data_id}") from exc


class ProcessedDataCatalog(ImmutableCatalog[ProcessedDataSpec]):
    def __init__(self, items: Iterable[ProcessedDataSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.data_id in seen:
                raise ValueError(f"duplicate_processed_data_spec: {spec.data_id}")
            seen.add(spec.data_id)
        super().__init__(sorted(values, key=lambda spec: spec.data_id))

    def get(self, data_id: str) -> ProcessedDataSpec:
        try:
            return next(spec for spec in self._items if spec.data_id == data_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_processed_data_spec: {data_id}") from exc
