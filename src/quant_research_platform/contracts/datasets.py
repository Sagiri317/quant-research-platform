"""Declare research datasets assembled from selected factors, features, and labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.risk import FactorRiskModel
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
)


@dataclass(frozen=True, kw_only=True)
class ResearchDatasetSpec(DataSpec):
    column_roles: Mapping[str, ColumnRole]
    market_id: MarketId
    input_data_ids: tuple[str, ...]
    base_input_data_id: str
    factor_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    feature_transforms: tuple[FeatureTransformSpec, ...] = ()
    label_columns: tuple[str, ...]
    filter_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    eligibility_recipe: str | None = None
    minimum_listing_sessions: int = 252
    exclude_st: bool = True
    exclude_pt: bool = True
    target_transform: str = "date_zscore"
    target_label_column: str = "forward_return_20d"
    target_clip: str = "none"
    risk_model: FactorRiskModel | None = None
    key_coverage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(
            self.target_label_column, "target_label_column_required"
        )
        super().__post_init__()
        feature_risks = (
            tuple(
                transform.risk_model
                for transform in self.feature_transforms
                if isinstance(transform, FeatureTransformSpec)
                and transform.risk_model is not None
            )
            if type(self.feature_transforms) is tuple
            else ()
        )
        if self.risk_model is None and feature_risks:
            object.__setattr__(self, "risk_model", feature_risks[0])
        if self.risk_model is None and self.target_transform == "risk_residual":
            object.__setattr__(self, "risk_model", FactorRiskModel())
        if any(model != self.risk_model for model in feature_risks):
            raise ValueError("research_risk_model_conflict")
        if self.risk_model is not None and type(self.risk_model) is not FactorRiskModel:
            raise ValueError("research_risk_model_invalid")
        if type(self.target_transform) is not str or self.target_transform not in {
            "date_zscore",
            "raw",
            "percentile",
            "industry_size_residual",
            "risk_residual",
        }:
            raise ValueError("invalid_target_transform")
        if type(self.target_clip) is not str or self.target_clip not in {
            "none",
            "mad",
            "quantile_1_99",
        }:
            raise ValueError("invalid_target_clip")
        if (
            type(self.minimum_listing_sessions) is not int
            or self.minimum_listing_sessions < 0
            or type(self.exclude_st) is not bool
            or type(self.exclude_pt) is not bool
        ):
            raise ValueError("invalid_a_share_eligibility_parameters")
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        if self.eligibility_recipe is not None and (
            self.market_id != "a_share"
            or type(self.eligibility_recipe) is not str
            or self.eligibility_recipe
            not in {"a_share_all", "a_share_nonfinancial", "a_share_fundamental"}
        ):
            raise ValueError("invalid_research_eligibility_recipe")
        if type(self.input_data_ids) is tuple and not self.input_data_ids:
            raise ValueError(
                f"base_input_data_id_not_in_inputs: {self.base_input_data_id}"
            )
        _validate_column_tuple(self.input_data_ids, "input_data_ids")
        _validate_non_empty_string(
            self.base_input_data_id,
            "base_input_data_id_required",
        )
        if type(self.feature_transforms) is not tuple:
            raise ValueError("feature_transforms_must_be_tuple")
        if any(
            type(transform) is not FeatureTransformSpec
            for transform in self.feature_transforms
        ):
            raise ValueError("feature_transforms_invalid")
        if not isinstance(self.key_coverage, Mapping) or any(
            type(data_id) is not str or not data_id for data_id in self.key_coverage
        ):
            raise ValueError("invalid_research_key_coverage")
        if any(type(policy) is not str for policy in self.key_coverage.values()):
            raise ValueError("invalid_research_key_coverage")
        if self.base_input_data_id not in self.input_data_ids:
            raise ValueError(
                f"base_input_data_id_not_in_inputs: {self.base_input_data_id}"
            )
        invalid_inputs = sorted(set(self.key_coverage).difference(self.input_data_ids))
        if invalid_inputs:
            raise ValueError(
                "unknown_research_key_coverage_input: " + ", ".join(invalid_inputs)
            )
        invalid_policies = sorted(
            f"{data_id}={policy}"
            for data_id, policy in self.key_coverage.items()
            if policy not in {"strict", "sparse"}
        )
        if invalid_policies:
            raise ValueError(
                "invalid_research_key_coverage: " + ", ".join(invalid_policies)
            )
        if self.key_coverage.get(self.base_input_data_id, "strict") != "strict":
            raise ValueError("base_input_key_coverage_must_be_strict")
        semantic_columns = (
            ("factor_columns", (ColumnRole.FACTOR,)),
            ("feature_columns", (ColumnRole.FEATURE,)),
            ("label_columns", (ColumnRole.LABEL,)),
            ("filter_columns", (ColumnRole.MASK, ColumnRole.UNIVERSE)),
            ("context_columns", (ColumnRole.CONTEXT,)),
        )
        for field_name, _ in semantic_columns:
            _validate_column_tuple(
                getattr(self, field_name), field_name, required=False
            )
        factor_feature_overlap = sorted(
            set(self.factor_columns).intersection(self.feature_columns)
        )
        if factor_feature_overlap:
            raise ValueError(
                "research_factor_feature_overlap: " + ", ".join(factor_feature_overlap)
            )
        for field_name, _ in semantic_columns:
            missing = [
                column
                for column in getattr(self, field_name)
                if column not in self.column_roles
            ]
            if missing:
                raise ValueError(
                    f"research_semantic_column_not_required: {field_name}: "
                    + ", ".join(missing)
                )
        _validate_feature_transforms(
            self.feature_transforms,
            feature_columns=self.feature_columns,
        )
        reserved = set(self.label_columns).union(
            self.filter_columns,
            self.context_columns,
        )
        invalid_research_inputs = sorted(
            set(self.factor_columns).union(self.feature_columns).intersection(reserved)
        )
        if invalid_research_inputs:
            raise ValueError(
                "research_input_column_overlap: " + ", ".join(invalid_research_inputs)
            )
        for field_name, allowed_roles in semantic_columns:
            invalid = [
                column
                for column in getattr(self, field_name)
                if self.column_roles[column] not in allowed_roles
            ]
            if invalid:
                raise ValueError(
                    f"research_semantic_column_role_invalid: {field_name}: "
                    + ", ".join(invalid)
                )
        normalized_key_coverage = {
            data_id: self.key_coverage.get(data_id, "strict")
            for data_id in self.input_data_ids
        }
        object.__setattr__(
            self,
            "key_coverage",
            MappingProxyType(normalized_key_coverage),
        )


def _validate_feature_transforms(
    feature_transforms: tuple[FeatureTransformSpec, ...],
    *,
    feature_columns: tuple[str, ...],
) -> None:
    if len(feature_transforms) > 1:
        raise ValueError("feature_recipe_must_be_exclusive")
    transformed_features = tuple(
        column
        for transform in feature_transforms
        for column in transform.output_columns
    )
    undeclared_outputs = sorted(set(transformed_features).difference(feature_columns))
    if undeclared_outputs:
        raise ValueError(
            "feature_transform_output_not_in_feature_columns: "
            + ", ".join(undeclared_outputs)
        )
