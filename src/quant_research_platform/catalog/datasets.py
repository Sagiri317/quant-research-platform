"""Research dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.datasets import ResearchDatasetSpec


@dataclass
class ResearchDatasetCatalog:
    _items: dict[str, ResearchDatasetSpec] = field(default_factory=dict)

    def register(self, spec: ResearchDatasetSpec) -> None:
        if spec.data_id in self._items:
            raise ValueError(f"duplicate_research_dataset_spec: {spec.data_id}")
        self._items[spec.data_id] = spec

    def get(self, data_id: str) -> ResearchDatasetSpec:
        try:
            return self._items[data_id]
        except KeyError as exc:
            raise KeyError(f"unknown_research_dataset_spec: {data_id}") from exc

    def list(self) -> tuple[ResearchDatasetSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["ResearchDatasetCatalog"]
