"""Evaluation catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.evaluations import EvaluationSpec


@dataclass
class EvaluationCatalog:
    _items: dict[str, EvaluationSpec] = field(default_factory=dict)

    def register(self, spec: EvaluationSpec) -> None:
        if spec.run_id in self._items:
            raise ValueError(f"duplicate_evaluation_spec: {spec.run_id}")
        self._items[spec.run_id] = spec

    def get(self, run_id: str) -> EvaluationSpec:
        try:
            return self._items[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown_evaluation_spec: {run_id}") from exc

    def list(self) -> tuple[EvaluationSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["EvaluationCatalog"]
