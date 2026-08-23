"""Research dataset catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.datasets import ResearchDatasetSpec


class ResearchDatasetCatalog(ImmutableCatalog[ResearchDatasetSpec]):
    def __init__(self, items: Iterable[ResearchDatasetSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.data_id in seen:
                raise ValueError(f"duplicate_research_dataset_spec: {spec.data_id}")
            seen.add(spec.data_id)
        super().__init__(sorted(values, key=lambda spec: spec.data_id))

    def get(self, data_id: str) -> ResearchDatasetSpec:
        try:
            return next(spec for spec in self._items if spec.data_id == data_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_research_dataset_spec: {data_id}") from exc


__all__ = ["ResearchDatasetCatalog"]
