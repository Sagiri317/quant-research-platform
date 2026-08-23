"""Prediction output catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.predictions import PredictionSpec


class PredictionCatalog(ImmutableCatalog[PredictionSpec]):
    def __init__(self, items: Iterable[PredictionSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.prediction_id in seen:
                raise ValueError(f"duplicate_prediction_spec: {spec.prediction_id}")
            seen.add(spec.prediction_id)
        super().__init__(sorted(values, key=lambda spec: spec.prediction_id))

    def get(self, prediction_id: str) -> PredictionSpec:
        try:
            return next(
                spec for spec in self._items if spec.prediction_id == prediction_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown_prediction_spec: {prediction_id}") from exc
