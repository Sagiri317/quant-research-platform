"""Generic persisted-prediction alignment."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import (
    _filter_working_set_bound,
    apply_filter_mode,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)


def _required_prediction_rows(
    chunk: pd.DataFrame,
    *,
    input_columns: tuple[str, ...],
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
) -> pd.DataFrame:
    eligible, _metadata = apply_filter_mode(
        chunk,
        mode=sample_filter_mode,
        metadata_key="sample_filter_mode",
    )
    eligible = eligible.dropna(subset=list(input_columns))
    timestamps = strict_timestamp_series(
        eligible["timestamp"],
        error="invalid_prediction_timestamp_values",
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
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
):
    missing_required = False
    with closing_iterator(iter(chunks)) as active_chunks:
        for chunk in active_chunks:
            row_count = len(chunk)
            frame_bytes = pandas_frame_payload_bytes(chunk)
            index_bytes = int(chunk.index.memory_usage(deep=True))
            timestamp_bytes = index_bytes + 8 * row_count
            required_mask_bytes = index_bytes + row_count
            required_output_bytes = frame_bytes
            output_frame_bytes = frame_bytes + index_bytes + 8 * row_count
            output_frame_bytes = max(
                output_frame_bytes,
                _filter_working_set_bound(
                    chunk,
                    mode=sample_filter_mode,
                ),
            )
            policy = current_runtime_resource_policy()
            with ExitStack() as chunk_state:
                if policy is not None:
                    chunk_state.enter_context(
                        policy.lease(
                            "research.prediction.alignment",
                            timestamp_bytes
                            + required_mask_bytes
                            + required_output_bytes
                            + output_frame_bytes,
                        )
                    )
                values, _found = chunk_state.enter_context(
                    prediction_lookup.values_for_frame(
                        chunk,
                        mark_matched=False,
                    )
                )
                required = _required_prediction_rows(
                    chunk,
                    input_columns=input_columns,
                    sample_filter_mode=sample_filter_mode,
                    test_start=test_start,
                    test_end=test_end,
                )
                if not required.empty:
                    _required_values, required_found = chunk_state.enter_context(
                        prediction_lookup.values_for_frame(
                            required,
                            mark_matched=True,
                        )
                    )
                    missing_required |= 0 in required_found
                output = chunk.copy()
                output[prediction_lookup.prediction_column] = values
                yield output
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
            policy = current_runtime_resource_policy()
            with ExitStack() as state:
                if policy is not None:
                    state.enter_context(
                        policy.lease(
                            "research.prediction.common_sample",
                            3 * pandas_frame_payload_bytes(chunk)
                            + len(chunk) * (16 * len(prediction_lookups) + 32),
                        )
                    )
                common = np.ones(len(chunk), dtype=bool)
                output = chunk.copy()
                for column, lookup in prediction_lookups.items():
                    values, found = state.enter_context(
                        lookup.values_for_frame(chunk, mark_matched=False)
                    )
                    common &= np.asarray(found, dtype=bool) & np.isfinite(values)
                    output[column] = values
                yield output.loc[common]
