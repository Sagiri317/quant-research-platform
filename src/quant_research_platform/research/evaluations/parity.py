"""Parity-or-explained-delta helpers for migration validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class DeltaClassification(StrEnum):
    EXPECTED_DELTA = "expected_delta"
    NUMERIC_TOLERANCE_DELTA = "numeric_tolerance_delta"
    OLD_BUG_FIXED_DELTA = "old_bug_fixed_delta"
    UNEXPLAINED_REGRESSION = "unexplained_regression"


@dataclass(frozen=True)
class ParityDeltaResult:
    classification: DeltaClassification
    stop_required: bool
    max_abs_delta: float
    row_count_delta: int
    missing_keys: int
    extra_keys: int
    reason: str | None = None


def classify_parity_delta(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    tolerance: float,
    expected_delta_reason: str | None = None,
    old_bug_fixed_reason: str | None = None,
) -> ParityDeltaResult:
    """Classify candidate-vs-reference deltas for clean-rewrite migration gates."""

    _validate_columns(reference, candidate, key_columns + value_columns)
    merged = reference[list(key_columns + value_columns)].merge(
        candidate[list(key_columns + value_columns)],
        how="outer",
        on=list(key_columns),
        indicator=True,
        suffixes=("_reference", "_candidate"),
    )
    missing_keys = int((merged["_merge"] == "left_only").sum())
    extra_keys = int((merged["_merge"] == "right_only").sum())
    max_abs_delta = _max_abs_delta(merged, value_columns)
    row_count_delta = int(len(candidate) - len(reference))

    if missing_keys or extra_keys or max_abs_delta > tolerance:
        if expected_delta_reason:
            return ParityDeltaResult(
                classification=DeltaClassification.EXPECTED_DELTA,
                stop_required=False,
                max_abs_delta=max_abs_delta,
                row_count_delta=row_count_delta,
                missing_keys=missing_keys,
                extra_keys=extra_keys,
                reason=expected_delta_reason,
            )
        if old_bug_fixed_reason:
            return ParityDeltaResult(
                classification=DeltaClassification.OLD_BUG_FIXED_DELTA,
                stop_required=False,
                max_abs_delta=max_abs_delta,
                row_count_delta=row_count_delta,
                missing_keys=missing_keys,
                extra_keys=extra_keys,
                reason=old_bug_fixed_reason,
            )
        return ParityDeltaResult(
            classification=DeltaClassification.UNEXPLAINED_REGRESSION,
            stop_required=True,
            max_abs_delta=max_abs_delta,
            row_count_delta=row_count_delta,
            missing_keys=missing_keys,
            extra_keys=extra_keys,
        )

    return ParityDeltaResult(
        classification=DeltaClassification.NUMERIC_TOLERANCE_DELTA,
        stop_required=False,
        max_abs_delta=max_abs_delta,
        row_count_delta=row_count_delta,
        missing_keys=missing_keys,
        extra_keys=extra_keys,
    )


def ensure_no_unexplained_regression(result: ParityDeltaResult) -> None:
    if result.stop_required:
        raise RuntimeError("unexplained_regression")


def _validate_columns(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    missing_reference = sorted(set(columns).difference(reference.columns))
    missing_candidate = sorted(set(columns).difference(candidate.columns))
    if missing_reference or missing_candidate:
        raise ValueError(
            "missing_parity_columns: "
            f"reference={missing_reference}, candidate={missing_candidate}"
        )


def _max_abs_delta(merged: pd.DataFrame, value_columns: tuple[str, ...]) -> float:
    max_abs = 0.0
    both = merged[merged["_merge"] == "both"]
    for column in value_columns:
        reference_column = f"{column}_reference"
        candidate_column = f"{column}_candidate"
        reference = both[reference_column]
        candidate = both[candidate_column]
        if pd.api.types.is_numeric_dtype(reference) and pd.api.types.is_numeric_dtype(
            candidate
        ):
            column_delta = (reference - candidate).abs().max()
            if pd.notna(column_delta):
                max_abs = max(max_abs, float(column_delta))
            continue
        mismatch = reference.fillna("<NA>") != candidate.fillna("<NA>")
        if bool(mismatch.any()):
            max_abs = max(max_abs, 1.0)
    return max_abs


__all__ = [
    "DeltaClassification",
    "ParityDeltaResult",
    "classify_parity_delta",
    "ensure_no_unexplained_regression",
]
