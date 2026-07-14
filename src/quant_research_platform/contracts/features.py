"""Feature transformation contracts."""

from __future__ import annotations

from dataclasses import dataclass

FEATURE_TRANSFORMS = {
    "cross_sectional_zscore",
    "cross_sectional_rank",
    "winsorize",
    "winsorized_zscore",
    "winsorized_rank",
    "fillna_zero",
    "fillna_median",
}
FILLNA_POLICIES = {"none", "zero", "median"}


@dataclass(frozen=True)
class FeatureTransformSpec:
    transform_id: str
    source_columns: tuple[str, ...]
    transform: str
    output_columns: tuple[str, ...]
    winsorize_quantiles: tuple[float, float] | None = None
    fillna_policy: str = "none"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_feature_transform_contract_version")
        if not isinstance(self.transform_id, str) or not self.transform_id:
            raise ValueError("feature_transform_id_required")
        _validate_column_tuple(self.source_columns, "source_columns")
        _validate_column_tuple(self.output_columns, "output_columns")
        if len(self.source_columns) != len(self.output_columns):
            raise ValueError("feature_transform_source_output_count_mismatch")
        if (
            not isinstance(self.transform, str)
            or self.transform not in FEATURE_TRANSFORMS
        ):
            raise ValueError("unsupported_feature_transform")
        if (
            not isinstance(self.fillna_policy, str)
            or self.fillna_policy not in FILLNA_POLICIES
        ):
            raise ValueError("unsupported_feature_transform_fillna_policy")
        overlap = sorted(
            set(self.source_columns).intersection(self.output_columns)
        )
        if overlap:
            raise ValueError(
                "feature_transform_raw_factor_passthrough: " + ", ".join(overlap)
            )
        if self.winsorize_quantiles is not None:
            validated_winsorize_quantiles(self.winsorize_quantiles)
        elif self.transform in {
            "winsorize",
            "winsorized_zscore",
            "winsorized_rank",
        }:
            object.__setattr__(self, "winsorize_quantiles", (0.01, 0.99))


def _validate_column_tuple(columns: tuple[str, ...], field_name: str) -> None:
    if not columns:
        raise ValueError(f"{field_name}_required")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


def validated_winsorize_quantiles(value: object) -> tuple[float, float]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int | float)
            for item in value
        )
    ):
        raise TypeError("winsorize_quantiles_must_be_numeric_pair")
    try:
        lower = float(value[0])
        upper = float(value[1])
    except OverflowError as exc:
        raise ValueError("invalid_winsorize_quantiles") from exc
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("invalid_winsorize_quantiles")
    return lower, upper


__all__ = ["FEATURE_TRANSFORMS", "FILLNA_POLICIES", "FeatureTransformSpec"]
