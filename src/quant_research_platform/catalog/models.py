"""Model catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.models import ModelSpec


@dataclass
class ModelCatalog:
    _items: dict[str, ModelSpec] = field(default_factory=dict)

    def register(self, spec: ModelSpec) -> None:
        if spec.model_id in self._items:
            raise ValueError(f"duplicate_model_spec: {spec.model_id}")
        self._items[spec.model_id] = spec

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._items[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown_model_spec: {model_id}") from exc

    def list(self) -> tuple[ModelSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["ModelCatalog"]
