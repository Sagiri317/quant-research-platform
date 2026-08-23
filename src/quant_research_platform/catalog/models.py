"""Model catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.models import ModelSpec


class ModelCatalog(ImmutableCatalog[ModelSpec]):
    def __init__(self, items: Iterable[ModelSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.model_id in seen:
                raise ValueError(f"duplicate_model_spec: {spec.model_id}")
            seen.add(spec.model_id)
        super().__init__(sorted(values, key=lambda spec: spec.model_id))

    def get(self, model_id: str) -> ModelSpec:
        try:
            return next(spec for spec in self._items if spec.model_id == model_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_model_spec: {model_id}") from exc
