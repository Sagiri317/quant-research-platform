"""Research dataset contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId, validate_market_id
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
    target_clip: str = "none"
    key_coverage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.target_transform) is not str or self.target_transform not in {
            "date_zscore",
            "raw",
            "percentile",
            "industry_size_residual",
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
        _validate_column_tuple(self.factor_columns, "factor_columns", required=False)
        _validate_column_tuple(self.feature_columns, "feature_columns", required=False)
        _validate_column_tuple(self.label_columns, "label_columns", required=False)
        _validate_column_tuple(self.filter_columns, "filter_columns", required=False)
        _validate_column_tuple(self.context_columns, "context_columns", required=False)
        if not self.factor_columns and not self.feature_columns:
            raise ValueError("research_factor_or_feature_columns_required")
        factor_feature_overlap = sorted(
            set(self.factor_columns).intersection(self.feature_columns)
        )
        if factor_feature_overlap:
            raise ValueError(
                "research_factor_feature_overlap: " + ", ".join(factor_feature_overlap)
            )
        for field_name, columns in (
            ("factor_columns", self.factor_columns),
            ("feature_columns", self.feature_columns),
            ("label_columns", self.label_columns),
            ("filter_columns", self.filter_columns),
            ("context_columns", self.context_columns),
        ):
            _validate_semantic_column_membership(
                columns,
                field_name=field_name,
                required_columns=self.required_columns,
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
        _validate_semantic_column_roles(
            self.factor_columns,
            field_name="factor_columns",
            column_roles=self.column_roles,
            allowed_roles=(ColumnRole.FACTOR,),
        )
        _validate_semantic_column_roles(
            self.feature_columns,
            field_name="feature_columns",
            column_roles=self.column_roles,
            allowed_roles=(ColumnRole.FEATURE,),
        )
        _validate_semantic_column_roles(
            self.label_columns,
            field_name="label_columns",
            column_roles=self.column_roles,
            allowed_roles=(ColumnRole.LABEL,),
        )
        _validate_semantic_column_roles(
            self.filter_columns,
            field_name="filter_columns",
            column_roles=self.column_roles,
            allowed_roles=(ColumnRole.MASK, ColumnRole.UNIVERSE),
        )
        _validate_semantic_column_roles(
            self.context_columns,
            field_name="context_columns",
            column_roles=self.column_roles,
            allowed_roles=(ColumnRole.CONTEXT,),
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


def _validate_semantic_column_membership(
    columns: tuple[str, ...],
    *,
    field_name: str,
    required_columns: tuple[str, ...],
) -> None:
    required = set(required_columns)
    missing = [column for column in columns if column not in required]
    if missing:
        raise ValueError(
            f"research_semantic_column_not_required: {field_name}: "
            + ", ".join(missing)
        )


def _validate_semantic_column_roles(
    columns: tuple[str, ...],
    *,
    field_name: str,
    column_roles: Mapping[str, ColumnRole],
    allowed_roles: tuple[ColumnRole, ...],
) -> None:
    invalid = [
        column for column in columns if column_roles.get(column) not in allowed_roles
    ]
    if invalid:
        raise ValueError(
            f"research_semantic_column_role_invalid: {field_name}: "
            + ", ".join(invalid)
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
    if len(set(transformed_features)) != len(transformed_features):
        raise ValueError("feature_transform_outputs_must_not_contain_duplicates")
    undeclared_outputs = sorted(set(transformed_features).difference(feature_columns))
    if undeclared_outputs:
        raise ValueError(
            "feature_transform_output_not_in_feature_columns: "
            + ", ".join(undeclared_outputs)
        )
