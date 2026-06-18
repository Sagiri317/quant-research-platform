"""Feature transformation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant_research_platform.contracts.ids import MarketId

FEATURE_TRANSFORMS = {
    "rank",
    "winsorize",
    "zscore",
    "rank_zscore",
    "standardize",
}


@dataclass(frozen=True)
class FeatureTransformSpec:
    transform_id: str
    market_id: MarketId
    source_factor_columns: tuple[str, ...]
    output_feature_columns: tuple[str, ...]
    transform: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_feature_transform_contract_version")
        _validate_column_tuple(self.source_factor_columns, "source_factor_columns")
        _validate_column_tuple(self.output_feature_columns, "output_feature_columns")
        if len(self.source_factor_columns) != len(self.output_feature_columns):
            raise ValueError("feature_transform_source_output_count_mismatch")
        if self.transform not in FEATURE_TRANSFORMS:
            raise ValueError("unsupported_feature_transform")
        overlap = sorted(
            set(self.source_factor_columns).intersection(self.output_feature_columns)
        )
        if overlap:
            raise ValueError(
                "feature_transform_raw_factor_passthrough: " + ", ".join(overlap)
            )
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _validate_column_tuple(columns: tuple[str, ...], field_name: str) -> None:
    if not columns:
        raise ValueError(f"{field_name}_required")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


__all__ = ["FEATURE_TRANSFORMS", "FeatureTransformSpec"]
