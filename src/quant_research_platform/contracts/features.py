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
    clip: str | None = None
    industry_neutralize: bool = True
    size_neutralize: bool = True
    center: bool = True
    scale: bool = True
    impute_missing: bool = True
    size_column: str = "log_total_market_cap"

    def __post_init__(self) -> None:
        if type(self.recipe_id) is not str or self.recipe_id not in FEATURE_RECIPES:
            raise ValueError("unsupported_feature_recipe")
        _validate_column_tuple(self.source_columns, "source_columns")
        if type(self.size_column) is not str or self.size_column not in {
            "log_total_market_cap",
            "log_float_market_cap",
        }:
            raise ValueError("invalid_feature_size_column")
        clip = self.clip
        if clip is None:
            clip = "mad" if self.recipe_id == "a_share_canonical" else "quantile_1_99"
        if type(clip) is not str or clip not in {"none", "mad", "quantile_1_99"}:
            raise ValueError("invalid_feature_clip")
        object.__setattr__(self, "clip", clip)
        for name in (
            "industry_neutralize",
            "size_neutralize",
            "center",
            "scale",
            "impute_missing",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"invalid_feature_{name}")

    @property
    def support_columns(self) -> tuple[str, ...]:
        columns = ["eligible"]
        if self.industry_neutralize:
            columns.extend(("industry_standard_id", "industry_level1_code"))
        if self.size_neutralize and any(
            column != self.size_column for column in self.source_columns
        ):
            columns.append(self.size_column)
        return tuple(columns)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(
            prefix + column
            for column in self.source_columns
            for prefix in FEATURE_PREFIXES
        )
