"""Research dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.datasets import ResearchDatasetSpec


@dataclass
class ResearchDatasetCatalog:
    _items: dict[str, ResearchDatasetSpec] = field(default_factory=dict)

    def register(self, spec: ResearchDatasetSpec) -> None:
        if spec.dataset_id in self._items:
            raise ValueError(f"duplicate_research_dataset_spec: {spec.dataset_id}")
        self._items[spec.dataset_id] = spec

    def get(self, dataset_id: str) -> ResearchDatasetSpec:
        try:
            return self._items[dataset_id]
        except KeyError as exc:
            raise KeyError(f"unknown_research_dataset_spec: {dataset_id}") from exc

    def list(self) -> tuple[ResearchDatasetSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["ResearchDatasetCatalog"]
