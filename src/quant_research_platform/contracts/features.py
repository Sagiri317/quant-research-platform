"""Declare ordered feature transformations and named A-share feature recipes."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.validation import _validate_column_tuple

FEATURE_RECIPES = {"a_share_canonical", "a_share_quantile_1_99"}


def inherited_feature_transforms(declared, manifest, columns):
    """Project the recorded transformation of reused, already computed features."""
    inherited = manifest.get("producer_diagnostics", {}).get(
        "reused_feature_transforms", {}
    )
    lineage = {column: inherited[column] for column in columns if column in inherited}
    for transform in declared.get("feature_transforms", []):
        outputs = FeatureTransformSpec(
            **{
                **transform,
                "source_columns": tuple(transform["source_columns"]),
                "output_groups": tuple(transform["output_groups"]),
            }
        ).output_columns
        lineage.update({column: transform for column in outputs if column in columns})
    return lineage


@dataclass(frozen=True)
class FeatureTransformSpec:
    recipe_id: str
    source_columns: tuple[str, ...]
    clip: str | None = None
    industry_neutralize: bool = True
    size_neutralize: bool = True
    center: bool = True
    scale: bool = True
    size_column: str = "log_total_market_cap"
    output_groups: tuple[str, ...] = ("processed",)

    def __post_init__(self) -> None:
        if type(self.recipe_id) is not str or self.recipe_id not in FEATURE_RECIPES:
            raise ValueError("unsupported_feature_recipe")
        _validate_column_tuple(self.source_columns, "source_columns")
        if type(self.size_column) is not str or self.size_column not in {
            "log_total_market_cap",
            "log_float_market_cap",
        }:
            raise ValueError("invalid_feature_size_column")
        if (
            type(self.output_groups) is not tuple
            or not self.output_groups
            or any(type(group) is not str for group in self.output_groups)
            or len(set(self.output_groups)) != len(self.output_groups)
            or any(
                group not in {"processed", "pre_neutral"}
                for group in self.output_groups
            )
        ):
            raise ValueError("invalid_feature_output_groups")
        if len(set(self.output_columns)) != len(self.output_columns):
            raise ValueError("feature_transform_outputs_must_not_contain_duplicates")
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
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"invalid_feature_{name}")

    @property
    def support_columns(self) -> tuple[str, ...]:
        columns = ["eligible"]
        if "processed" in self.output_groups and self.industry_neutralize:
            columns.extend(("industry_standard_id", "industry_level1_code"))
        if (
            "processed" in self.output_groups
            and self.size_neutralize
            and any(column != self.size_column for column in self.source_columns)
        ):
            columns.append(self.size_column)
        return tuple(columns)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(
            "feature_" + ("pre_neutral_" if group == "pre_neutral" else "") + column
            for group in self.output_groups
            for column in self.source_columns
        )
