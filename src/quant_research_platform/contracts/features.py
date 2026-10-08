"""Named A-share feature recipe contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.validation import _validate_column_tuple

FEATURE_RECIPES = {"a_share_canonical", "a_share_quantile_1_99"}
FEATURE_PREFIXES = ("feature_", "feature_missing_", "feature_observed_")


@dataclass(frozen=True)
class FeatureTransformSpec:
    recipe_id: str
    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.recipe_id) is not str or self.recipe_id not in FEATURE_RECIPES:
            raise ValueError("unsupported_feature_recipe")
        _validate_column_tuple(self.source_columns, "source_columns")

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(
            prefix + column
            for column in self.source_columns
            for prefix in FEATURE_PREFIXES
        )
