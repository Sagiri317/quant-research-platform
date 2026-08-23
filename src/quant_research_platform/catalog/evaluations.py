"""Evaluation catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.evaluations import EvaluationSpec


class EvaluationCatalog(ImmutableCatalog[EvaluationSpec]):
    def __init__(self, items: Iterable[EvaluationSpec] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for spec in values:
            if spec.run_id in seen:
                raise ValueError(f"duplicate_evaluation_spec: {spec.run_id}")
            seen.add(spec.run_id)
        super().__init__(sorted(values, key=lambda spec: spec.run_id))

    def get(self, run_id: str) -> EvaluationSpec:
        try:
            return next(spec for spec in self._items if spec.run_id == run_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_evaluation_spec: {run_id}") from exc


__all__ = ["EvaluationCatalog"]
