"""Prediction output catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.predictions import PredictionSpec


@dataclass
class PredictionCatalog:
    _items: dict[str, PredictionSpec] = field(default_factory=dict)

    def register(self, spec: PredictionSpec) -> None:
        if spec.prediction_id in self._items:
            raise ValueError(f"duplicate_prediction_spec: {spec.prediction_id}")
        self._items[spec.prediction_id] = spec

    def get(self, prediction_id: str) -> PredictionSpec:
        try:
            return self._items[prediction_id]
        except KeyError as exc:
            raise KeyError(f"unknown_prediction_spec: {prediction_id}") from exc

    def list(self) -> tuple[PredictionSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["PredictionCatalog"]
