"""Research dataset contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.validation import _validate_column_tuple


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
    key_coverage: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.column_roles) not in (dict, MappingProxyType) or any(
            type(column) is not str or not column or type(role) is not ColumnRole
            for column, role in self.column_roles.items()
        ):
            raise ValueError("invalid_research_column_roles")
        super().__post_init__()
        object.__setattr__(
            self,
            "column_roles",
            MappingProxyType(self.column_roles),
        )
        if type(self.key_coverage) not in (dict, MappingProxyType) or any(
            type(data_id) is not str for data_id in self.key_coverage
        ):
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
            if type(policy) is not str or policy not in {"strict", "sparse"}
        )
        if invalid_policies:
            raise ValueError(
                "invalid_research_key_coverage: " + ", ".join(invalid_policies)
            )
        if self.key_coverage.get(self.base_input_data_id, "strict") != "strict":
            raise ValueError("base_input_key_coverage_must_be_strict")
        _validate_column_tuple(self.factor_columns, "factor_columns", required=False)
        _validate_column_tuple(self.feature_columns, "feature_columns", required=False)
        _validate_column_tuple(self.label_columns, "label_columns")
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
            factor_columns=self.factor_columns,
            context_columns=self.context_columns,
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
    factor_columns: tuple[str, ...],
    context_columns: tuple[str, ...],
    feature_columns: tuple[str, ...],
) -> None:
    transform_ids = [transform.transform_id for transform in feature_transforms]
    if len(set(transform_ids)) != len(transform_ids):
        raise ValueError("feature_transform_ids_must_not_contain_duplicates")
    source_columns = {
        column
        for transform in feature_transforms
        for column in transform.source_columns
    }
    allowed_sources = set(factor_columns).union(context_columns)
    missing_sources = sorted(source_columns.difference(allowed_sources))
    if missing_sources:
        raise ValueError(
            "feature_transform_source_not_in_factor_or_context_columns: "
            + ", ".join(missing_sources)
        )
    transformed_features = tuple(
        column
        for transform in feature_transforms
        for column in transform.output_columns
    )
    if len(set(transformed_features)) != len(transformed_features):
        raise ValueError("feature_transform_outputs_must_not_contain_duplicates")
    missing_outputs = sorted(set(feature_columns).difference(transformed_features))
    if missing_outputs:
        raise ValueError(
            "feature_columns_without_transform: " + ", ".join(missing_outputs)
        )
    undeclared_outputs = sorted(set(transformed_features).difference(feature_columns))
    if undeclared_outputs:
        raise ValueError(
            "feature_transform_output_not_in_feature_columns: "
            + ", ".join(undeclared_outputs)
        )
