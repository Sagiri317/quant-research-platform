"""Research dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True, kw_only=True)
class ResearchDatasetSpec(DataSpec):
    market_id: MarketId
    input_data_ids: tuple[str, ...]
    base_input_data_id: str
    factor_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    feature_transforms: tuple[FeatureTransformSpec, ...] = ()
    label_columns: tuple[str, ...]
    filter_columns: tuple[str, ...] = ()
    context_columns: tuple[str, ...] = ()
    key_coverage: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
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
                "research_factor_feature_overlap: "
                + ", ".join(factor_feature_overlap)
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
        normalized = {
            data_id: self.key_coverage.get(data_id, "strict")
            for data_id in self.input_data_ids
        }
        object.__setattr__(self, "key_coverage", normalized)


def _validate_column_tuple(
    columns: tuple[str, ...],
    field_name: str,
    *,
    required: bool = True,
) -> None:
    if required and not columns:
        raise ValueError(f"{field_name}_required")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


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

__all__ = ["ResearchDatasetSpec"]
