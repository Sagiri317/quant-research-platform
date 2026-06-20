"""Model contracts for non-DL local research."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    reject_filter_input_columns,
    validate_sample_filter_mode,
)


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    model_id: str
    model_algorithm: str
    input_factor_columns: tuple[str, ...] = ()
    input_feature_columns: tuple[str, ...] = ()
    label_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    hyperparameters: Mapping[str, object] = field(default_factory=dict)
    random_seed: int | None = None
    purge_gap_steps: int = 0
    max_train_rows: int | None = None
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_model_contract_version")
        if not isinstance(self.model_algorithm, str) or not self.model_algorithm:
            raise ValueError("model_algorithm_required")
        validate_sample_filter_mode(self.sample_filter_mode, owner="model")
        _validate_column_tuple(
            self.input_factor_columns,
            "input_factor_columns",
            required=False,
        )
        _validate_column_tuple(
            self.input_feature_columns,
            "input_feature_columns",
            required=False,
        )
        if not self.input_factor_columns and not self.input_feature_columns:
            raise ValueError("model_input_columns_required")
        overlap = sorted(
            set(self.input_factor_columns).intersection(self.input_feature_columns)
        )
        if overlap:
            raise ValueError("model_factor_feature_overlap: " + ", ".join(overlap))
        reject_filter_input_columns(self.input_factor_columns)
        reject_filter_input_columns(self.input_feature_columns)
        if self.label_column in self.input_columns:
            raise ValueError("model_label_column_in_inputs")
        if self.random_seed is not None and isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer or None")
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer or None")
        if not isinstance(self.purge_gap_steps, int) or isinstance(
            self.purge_gap_steps, bool
        ):
            raise TypeError("purge_gap_steps must be an integer")
        if self.purge_gap_steps < 0:
            raise ValueError("purge_gap_steps_must_be_nonnegative")
        if self.max_train_rows is not None and (
            not isinstance(self.max_train_rows, int)
            or isinstance(self.max_train_rows, bool)
        ):
            raise TypeError("max_train_rows must be an integer or None")
        if self.max_train_rows is not None and self.max_train_rows <= 0:
            raise ValueError("max_train_rows_must_be_positive")
        object.__setattr__(
            self,
            "hyperparameters",
            MappingProxyType(dict(self.hyperparameters)),
        )

    @property
    def input_columns(self) -> tuple[str, ...]:
        return (*self.input_factor_columns, *self.input_feature_columns)


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


__all__ = ["ModelSpec"]
