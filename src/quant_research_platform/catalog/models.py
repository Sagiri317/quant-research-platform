"""Model catalog."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

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

    @classmethod
    def from_configuration(cls, items) -> ModelCatalog:
        if type(items) is not list:
            raise ValueError("research_model_selection_invalid")
        models = []
        for item in items:
            if type(item) is not dict:
                raise ValueError("research_model_configuration_invalid")
            fields = {
                key: value
                for key, value in item.items()
                if key
                not in {
                    "parameter_candidates",
                    "training_asset_ids",
                    "evaluation_asset_ids",
                }
            }
            for key in ("input_factor_columns", "input_feature_columns"):
                if type(fields.get(key)) is not list:
                    raise ValueError("research_model_input_selection_invalid")
                fields[key] = tuple(fields[key])
            try:
                models.append(ModelSpec(**fields))
            except TypeError:
                raise ValueError("research_model_configuration_invalid") from None
            if "parameter_candidates" in item and (
                type(item["parameter_candidates"]) is not list
                or not item["parameter_candidates"]
                or any(
                    type(value) is not dict for value in item["parameter_candidates"]
                )
            ):
                raise ValueError("research_parameter_candidates_invalid")
            for parameters in item.get("parameter_candidates", ()):
                replace(models[-1], hyperparameters=parameters)
        return cls(models)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return next(spec for spec in self._items if spec.model_id == model_id)
        except StopIteration as exc:
            raise KeyError(f"unknown_model_spec: {model_id}") from exc
