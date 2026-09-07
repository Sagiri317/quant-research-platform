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
from quant_research_platform.runtime.dataframes import _selected_frame_payload_bytes
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
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
    prediction_source,
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
    resource_policy: RuntimeResourcePolicy | None,
):
    if isinstance(prediction_source, pd.DataFrame):
        yield from _iter_prediction_frame_joined_chunks(
            chunks,
            prediction_frame=prediction_source,
            primary_key=primary_key,
            input_columns=input_columns,
            sample_filter_mode=sample_filter_mode,
            test_start=test_start,
            test_end=test_end,
            resource_policy=resource_policy,
        )
        return
    yield from _iter_prediction_lookup_joined_chunks(
        chunks,
        prediction_lookup=prediction_source,
        input_columns=input_columns,
        sample_filter_mode=sample_filter_mode,
        test_start=test_start,
        test_end=test_end,
    )


def _iter_prediction_frame_joined_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    prediction_frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    input_columns: tuple[str, ...],
    sample_filter_mode: str,
    test_start: str,
    test_end: str | None,
    resource_policy: RuntimeResourcePolicy | None,
):
    missing_columns = tuple(
        column for column in primary_key if column not in prediction_frame.columns
    )
    if missing_columns:
        raise ValueError("persisted_prediction_join_key_invalid")
    if prediction_frame.duplicated(list(primary_key)).any():
        raise ValueError("duplicate_persisted_prediction_keys")
    prediction_bytes = pandas_frame_payload_bytes(prediction_frame)
    prediction_rows = len(prediction_frame)
    prediction_key_bytes = _selected_frame_payload_bytes(
        prediction_frame,
        list(primary_key),
    )
    prediction_index_bytes = prediction_key_bytes + prediction_rows * (
        8 * len(primary_key)
    )
    matched_bytes = prediction_rows
    with ExitStack() as prediction_state:
        if resource_policy is not None:
            try:
                prediction_state.enter_context(
                    resource_policy.lease(
                        "research.portfolio.prediction.input",
                        prediction_bytes + prediction_index_bytes + matched_bytes,
                    )
                )
            except MemoryError:
                raise ValueError(
                    "portfolio_prediction_join_working_set_exceeds_task_memory_budget"
                ) from None
        prediction_index = pd.MultiIndex.from_frame(
            prediction_frame.loc[:, list(primary_key)]
        )
        matched = np.zeros(prediction_rows, dtype=bool)
        missing_required = False
        with closing_iterator(iter(chunks)) as active_chunks:
            for chunk in active_chunks:
                chunk_rows = len(chunk)
                chunk_bytes = pandas_frame_payload_bytes(chunk)
                chunk_index_bytes = int(chunk.index.memory_usage(deep=True))
                timestamp_bytes = chunk_index_bytes + 8 * chunk_rows
                required_mask_bytes = chunk_index_bytes + chunk_rows
                required_key_bytes = _selected_frame_payload_bytes(
                    chunk,
                    list(primary_key),
                )
                required_index_bytes = required_key_bytes + chunk_rows * (
                    8 * len(primary_key)
                )
                positions_bytes = chunk_rows * np.dtype(np.intp).itemsize
                join_hash_bytes = chunk_rows * (32 + 16 * len(primary_key))
                joined_output_bytes = chunk_bytes + chunk_index_bytes + 8 * chunk_rows
                joined_output_bytes = max(
                    joined_output_bytes,
                    _filter_working_set_bound(
                        chunk,
                        mode=sample_filter_mode,
                    ),
                )
                chunk_working_bytes = (
                    timestamp_bytes
                    + required_mask_bytes
                    + required_index_bytes
                    + positions_bytes
                    + join_hash_bytes
                    + joined_output_bytes
                )
                with ExitStack() as chunk_state:
                    if resource_policy is not None:
                        chunk_state.enter_context(
                            resource_policy.lease(
                                "research.portfolio.prediction.join",
                                chunk_working_bytes,
                            )
                        )
                    required = _required_prediction_rows(
                        chunk,
                        input_columns=input_columns,
                        sample_filter_mode=sample_filter_mode,
                        test_start=test_start,
                        test_end=test_end,
                    )
                    required_index = pd.MultiIndex.from_frame(
                        required.loc[:, list(primary_key)]
                    )
                    positions = prediction_index.get_indexer(required_index)
                    missing_required |= bool((positions < 0).any())
                    matched[positions[positions >= 0]] = True
                    try:
                        yield chunk.merge(
                            prediction_frame,
                            on=list(primary_key),
                            how="left",
                            validate="one_to_one",
                        )
                    except pd.errors.MergeError as exc:
                        raise ValueError(
                            "persisted_prediction_join_key_invalid"
                        ) from exc
        unexpected = bool((~matched).any())
        if missing_required and unexpected:
            raise ValueError("misaligned_persisted_prediction_keys")
        if missing_required:
            raise ValueError("missing_required_prediction_keys")
        if unexpected:
            raise ValueError("misaligned_persisted_prediction_keys")


def _iter_prediction_lookup_joined_chunks(
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
