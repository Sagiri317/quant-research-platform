"""Generic persisted-prediction alignment."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.filtering import apply_filter_mode


def _required_prediction_rows(
    chunk: pd.DataFrame,
    *,
    input_columns: tuple[str, ...],
    model_algorithm: str,
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
) -> pd.DataFrame:
    eligible, _metadata = apply_filter_mode(
        chunk, mode=sample_filter_mode, metadata_key="sample_filter_mode"
    )
    eligible = eligible.dropna(
        subset=list(model_input_validity_columns(input_columns, model_algorithm))
    )
    timestamps = strict_timestamp_series(
        eligible["timestamp"], error="invalid_prediction_timestamp_values"
    )
    required_mask = timestamps >= pd.Timestamp(test_start)
    if test_end is not None:
        required_mask &= timestamps <= pd.Timestamp(test_end)
    return eligible.loc[required_mask]


def _iter_prediction_joined_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    prediction_lookup,
    input_columns: tuple[str, ...],
    model_algorithm: str,
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
):
    missing_required = False
    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
            values, _found = prediction_lookup.values_for_frame(
                chunk, mark_matched=False
            )
            required = _required_prediction_rows(
                chunk,
                input_columns=input_columns,
                model_algorithm=model_algorithm,
                sample_filter_mode=sample_filter_mode,
                test_start=test_start,
                test_end=test_end,
            )
            if not required.empty:
                _required_values, required_found = prediction_lookup.values_for_frame(
                    required, mark_matched=True
                )
                missing_required |= 0 in required_found
            output = chunk.copy()
            output[prediction_lookup.prediction_column] = values
            try:
                yield output
            except GeneratorExit:
                return
    unexpected, extra = prediction_lookup.coverage_counts()
    if extra:
        raise RuntimeError("persisted_prediction_match_state_invalid")
    if missing_required and unexpected:
        raise ValueError("misaligned_persisted_prediction_keys")
    if missing_required:
        raise ValueError("missing_required_prediction_keys")
    if unexpected:
        raise ValueError("misaligned_persisted_prediction_keys")


def _iter_common_prediction_chunks(chunks, *, prediction_lookups):
    """Select exact-key finite intersections without altering persisted outputs."""
    with closing_iterator(iter(chunks)) as source:
        for chunk in source:
            common = np.ones(len(chunk), dtype=bool)
            output = chunk.copy()
            for column, lookup in prediction_lookups.items():
                values, found = lookup.values_for_frame(chunk, mark_matched=False)
                common &= np.asarray(found, dtype=bool) & np.isfinite(values)
                output[column] = values
            yielded_value = output.loc[common]
            try:
                yield yielded_value
            except GeneratorExit:
                return
