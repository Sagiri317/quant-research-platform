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
            if spec.evaluation_id in seen:
                raise ValueError(
                    f"duplicate_evaluation_spec: {spec.evaluation_id}"
                )
            seen.add(spec.evaluation_id)
        super().__init__(sorted(values, key=lambda spec: spec.evaluation_id))

    def get(self, evaluation_id: str) -> EvaluationSpec:
        try:
            return next(
                spec for spec in self._items if spec.evaluation_id == evaluation_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown_evaluation_spec: {evaluation_id}") from exc
