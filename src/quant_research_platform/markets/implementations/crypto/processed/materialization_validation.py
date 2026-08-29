"""Crypto processed materialization lifecycle validation."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
    strict_hashable_key_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    canonical_crypto_asset_id,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    filter_frame_by_time_window,
    iter_dataset_chunks,
    iter_dataset_file_batches,
    parquet_filters_for_time_window,
)
from quant_research_platform.storage.datasets.schema import dataset_identity_payload
from quant_research_platform.storage.datasets.writing import DatasetWriteReceipt

_EXPECTED_KEY_NAMESPACE = "crypto_expected"


@dataclass
class OrderedDependencyState:
    """Bounded ordering state shared across dependency chunks."""

    last_key: tuple[str, str, pd.Timestamp] | None = None


@contextmanager
def crypto_asset_session_views(
    session: DatasetSession,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    error: str,
    key_error: str | None = None,
) -> Iterator[tuple[tuple[tuple[str, ...], DatasetSession], ...]]:
    if not columns:
        raise ValueError(error)
    with runtime_resource_policy_scope(session.resource_policy):
        selections: dict[tuple[str, ...], int] = {}
        if session.resource_policy.planning_headroom_bytes <= 0:
            raise ValueError("crypto_asset_session_views_exceed_task_memory_budget")
        current_path: Path | None = None
        current_key: tuple[str, ...] | None = None
        discovery_columns = tuple(
            dict.fromkeys(
                (
                    *columns,
                    *(
                        column
                        for column, _operator, _value in session.parquet_filters or ()
                    ),
                )
            )
        )

        def select_current_path() -> None:
            if current_path is None or current_key is None:
                return
            selection = selections.get(current_key)
            if selection is None:
                selection = session.inventory.new_path_selection()
                selections[current_key] = selection
            session.inventory.select_path(selection, current_path)

        try:
            with closing_iterator(
                iter_dataset_file_batches(
                    session,
                    columns=discovery_columns,
                    target_chunk_bytes=target_chunk_bytes,
                )
            ) as file_batches:
                for path, frame in file_batches:
                    values = tuple(frame[column] for column in columns)
                    for row in zip(*values, strict=True):
                        if any(
                            not isinstance(value, str) or not value for value in row
                        ):
                            raise ValueError(error if key_error is None else key_error)
                        key = tuple(row)
                        if path != current_path:
                            select_current_path()
                            current_path = path
                            current_key = key
                        elif current_key != key:
                            raise ValueError(error)
            select_current_path()
            if not selections:
                raise ValueError(error if key_error is None else key_error)
            views = tuple(
                (
                    key,
                    replace(
                        session,
                        parquet_filters=tuple(session.parquet_filters or ())
                        + tuple(
                            (column, "==", value)
                            for column, value in zip(columns, key, strict=True)
                        ),
                        path_selection=selections[key],
                        owns_inventory=False,
                    ),
                )
                for key in sorted(selections)
            )
            session.verify()
            yield views
        finally:
            primary = sys.exception()
            first_failure: BaseException | None = None
            for selection in selections.values():
                try:
                    session.inventory.release_path_selection(selection)
                except BaseException as exc:
                    if primary is None and first_failure is None:
                        first_failure = exc
                    elif primary is not None:
                        primary.add_note(
                            "crypto asset path-selection cleanup also failed: "
                            f"{type(exc).__name__}"
                        )
                    else:
                        first_failure.add_note(
                            "crypto asset path-selection cleanup also failed: "
                            f"{type(exc).__name__}"
                        )
            if first_failure is not None:
                raise first_failure


def core_time_window_mask(
    frame: pd.DataFrame,
    time_window: TimeWindow,
) -> pd.Series:
    spec = {
        "column_roles": {"timestamp": "timestamp"},
        "primary_key": ("timestamp",),
    }
    work = frame.loc[:, ["timestamp"]].copy()
    work["_qrp_original_position"] = range(len(frame))
    filtered = filter_frame_by_time_window(
        work,
        spec=spec,
        time_window=time_window,
    )
    positions = pd.Series(False, index=range(len(frame)))
    positions.iloc[filtered["_qrp_original_position"].to_numpy(dtype=int)] = True
    return pd.Series(positions.to_numpy(dtype=bool), index=frame.index)


def input_dataset_identities(
    sessions: Mapping[str, DatasetSession],
) -> dict[str, dict[str, str]]:
    if not sessions:
        raise ValueError("missing_crypto_materialization_inputs")
    identities: dict[str, dict[str, str]] = {}
    for data_id, session in sessions.items():
        if not isinstance(data_id, str) or session.spec.get("data_id") != data_id:
            raise ValueError("invalid_crypto_materialization_input_identity")
        identities[data_id] = dataset_identity_payload(
            session.spec,
            session.manifest,
        )
    return dict(sorted(identities.items()))


def validate_materialization_input_identities(
    manifest: Mapping[str, object],
    *,
    expected: Mapping[str, Mapping[str, str]],
    error: str,
) -> None:
    if manifest.get("input_dataset_identities") != dict(expected):
        raise ValueError(error)


def validate_panel_materialization(
    receipt: DatasetWriteReceipt,
    *,
    bar_session: DatasetSession,
    spec: ProcessedDataSpec,
    expected_input_identities: Mapping[str, Mapping[str, str]],
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        receipt.session.resource_policy is not resource_policy
        or bar_session.resource_policy is not resource_policy
    ):
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        validate_materialization_input_identities(
            receipt.session.manifest,
            expected=expected_input_identities,
            error="crypto_panel_input_dataset_identity_mismatch",
        )
        chunk_target_bytes = _validation_chunk_target_bytes(resource_policy)
        bar_view = _key_projection_view(
            bar_session,
            columns=("source_asset_id", "timestamp"),
        )
        with (
            closing_iterator(
                iter_dataset_chunks(bar_view, target_chunk_bytes=chunk_target_bytes)
            ) as bar_chunks,
            closing_iterator(
                _iter_panel_primary_key_chunks(
                    bar_chunks,
                    primary_key=spec.primary_key,
                )
            ) as key_chunks,
        ):
            _validate_receipt_key_coverage(
                key_chunks,
                receipt,
                primary_key=spec.primary_key,
                error="invalid_crypto_panel_key_coverage",
                resource_policy=resource_policy,
            )


def validate_panel_dependent_materialization(
    receipt: DatasetWriteReceipt,
    *,
    panel_session: DatasetSession,
    spec: ProcessedDataSpec,
    time_window: TimeWindow,
    expected_input_identities: Mapping[str, Mapping[str, str]],
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        receipt.session.resource_policy is not resource_policy
        or panel_session.resource_policy is not resource_policy
    ):
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        validate_materialization_input_identities(
            receipt.session.manifest,
            expected=expected_input_identities,
            error="crypto_processed_input_dataset_identity_mismatch",
        )
        chunk_target_bytes = _validation_chunk_target_bytes(resource_policy)
        with closing_iterator(
            _iter_core_window_chunks(
                panel_session,
                time_window=time_window,
                target_chunk_bytes=chunk_target_bytes,
                columns=spec.primary_key,
            )
        ) as expected_chunks:
            _validate_receipt_key_coverage(
                expected_chunks,
                receipt,
                primary_key=spec.primary_key,
                error="invalid_crypto_processed_key_coverage",
                resource_policy=resource_policy,
            )


def _validate_receipt_key_coverage(
    expected: Iterable[pd.DataFrame],
    receipt: DatasetWriteReceipt,
    *,
    primary_key: tuple[str, ...],
    error: str,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if not primary_key:
        raise ValueError(error)
    chunk_target_bytes = _validation_chunk_target_bytes(resource_policy)
    receipt.session.verify()
    index = receipt.exact_primary_key_index
    for frame in expected:
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(error)
        active_policy = current_runtime_resource_policy()
        if active_policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        if active_policy is not resource_policy:
            raise RuntimeError("runtime_resource_policy_conflict")
        with resource_policy.lease(
            "crypto.validation.keys",
            max(1, pandas_frame_payload_bytes(frame)),
        ):
            keys = _normalized_primary_keys(frame, primary_key, error=error)
            if pandas_frame_payload_bytes(keys) > chunk_target_bytes:
                raise ValueError(error)
            try:
                duplicates = index.add_frame(keys, namespace=_EXPECTED_KEY_NAMESPACE)
            except ValueError as exc:
                raise ValueError(error) from exc
            if duplicates:
                raise ValueError(error)
    missing_count, extra_count = index.difference_counts(
        expected_namespace=_EXPECTED_KEY_NAMESPACE,
        actual_namespace="dataset",
    )
    if missing_count or extra_count:
        raise ValueError(error)


def validate_existing_panel_materialization(
    panel_session: DatasetSession,
    *,
    bar_session: DatasetSession,
    spec: ProcessedDataSpec,
    expected_input_identities: Mapping[str, Mapping[str, str]],
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        panel_session.resource_policy is not resource_policy
        or bar_session.resource_policy is not resource_policy
    ):
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        validate_materialization_input_identities(
            panel_session.manifest,
            expected=expected_input_identities,
            error="crypto_panel_input_dataset_identity_mismatch",
        )
        chunk_target_bytes = _validation_chunk_target_bytes(resource_policy)
        bar_view = _key_projection_view(
            bar_session,
            columns=("source_asset_id", "timestamp"),
        )
        panel_view = _key_projection_view(
            panel_session,
            columns=spec.primary_key,
        )
        with (
            closing_iterator(
                iter_dataset_chunks(
                    bar_view,
                    target_chunk_bytes=chunk_target_bytes,
                )
            ) as bar_chunks,
            closing_iterator(
                _iter_panel_primary_key_chunks(
                    bar_chunks,
                    primary_key=spec.primary_key,
                )
            ) as expected_chunks,
            closing_iterator(
                iter_dataset_chunks(
                    panel_view,
                    target_chunk_bytes=chunk_target_bytes,
                )
            ) as actual_chunks,
        ):
            _validate_ordered_key_stream_coverage(
                expected_chunks,
                actual_chunks,
                primary_key=spec.primary_key,
                error="invalid_crypto_panel_key_coverage",
            )


def _iter_core_window_chunks(
    session: DatasetSession,
    *,
    time_window: TimeWindow,
    target_chunk_bytes: int,
    columns: tuple[str, ...] | None = None,
) -> Iterator[pd.DataFrame]:
    filters = tuple(
        item
        for item in session.parquet_filters or ()
        if item[0] != session.timestamp_column
    ) + tuple(
        parquet_filters_for_time_window(
            timestamp_column=session.timestamp_column,
            time_window=time_window,
        )
        or ()
    )
    logical_columns = (
        getattr(session, "logical_columns", ()) if columns is None else columns
    )
    physical_columns = tuple(
        dict.fromkeys(
            (
                *logical_columns,
                *(column for column, _operator, _value in filters),
            )
        )
    )
    view = replace(
        session,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        time_window=time_window,
        parquet_filters=filters or None,
        owns_inventory=False,
    )
    view.verify()
    with closing_iterator(
        iter_dataset_chunks(
            view,
            target_chunk_bytes=target_chunk_bytes,
        )
    ) as frames:
        for frame in frames:
            if not frame.empty:
                yield frame


def _validation_chunk_target_bytes(
    resource_policy: RuntimeResourcePolicy,
) -> int:
    return max(1, resource_policy.planning_headroom_bytes // 8)


def _key_projection_view(
    session: DatasetSession,
    *,
    columns: tuple[str, ...],
) -> DatasetSession:
    filter_columns = tuple(
        column for column, _operator, _value in session.parquet_filters or ()
    )
    return replace(
        session,
        logical_columns=columns,
        physical_columns=tuple(dict.fromkeys((*columns, *filter_columns))),
        owns_inventory=False,
    )


def _validate_ordered_key_stream_coverage(
    expected: Iterable[pd.DataFrame],
    actual: Iterable[pd.DataFrame],
    *,
    primary_key: tuple[str, ...],
    error: str,
) -> None:
    sentinel = object()
    expected_rows = _iter_normalized_primary_key_rows(
        expected,
        primary_key=primary_key,
        error=error,
    )
    actual_rows = _iter_normalized_primary_key_rows(
        actual,
        primary_key=primary_key,
        error=error,
    )
    with (
        closing_iterator(expected_rows),
        closing_iterator(actual_rows),
    ):
        for expected_row, actual_row in zip_longest(
            expected_rows,
            actual_rows,
            fillvalue=sentinel,
        ):
            if expected_row != actual_row:
                raise ValueError(error)


def _iter_normalized_primary_key_rows(
    frames: Iterable[pd.DataFrame],
    *,
    primary_key: tuple[str, ...],
    error: str,
) -> Iterator[tuple[object, ...]]:
    previous: tuple[object, ...] | None = None
    for frame in frames:
        policy = current_runtime_resource_policy()
        if policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        with policy.lease(
            "crypto.validation.keys",
            max(1, pandas_frame_payload_bytes(frame)),
        ):
            keys = _normalized_primary_keys(frame, primary_key, error=error)
            for row in keys.itertuples(index=False, name=None):
                normalized = tuple(row)
                if previous is not None and normalized <= previous:
                    raise ValueError(error)
                previous = normalized
                yield normalized


def validated_ordered_dependency_chunk(
    frame: pd.DataFrame,
    *,
    state: OrderedDependencyState,
    error: str,
) -> pd.DataFrame:
    required = ("market_id", "asset_id", "timestamp")
    if any(column not in frame for column in required):
        raise ValueError(error)
    ordered = frame.copy()
    for column in ("market_id", "asset_id"):
        if (
            not ordered[column]
            .map(lambda value: isinstance(value, str) and bool(value))
            .all()
        ):
            raise ValueError(error)
    ordered["timestamp"] = strict_timestamp_series(
        ordered["timestamp"],
        error=error,
    )
    if not ordered.empty:
        markets = ordered["market_id"].to_numpy(dtype=object, copy=False)
        assets = ordered["asset_id"].to_numpy(dtype=object, copy=False)
        timestamps = ordered["timestamp"].to_numpy(
            dtype="datetime64[ns]",
            copy=False,
        )
        first_key = (str(markets[0]), str(assets[0]), pd.Timestamp(timestamps[0]))
        if state.last_key is not None and first_key <= state.last_key:
            raise ValueError(error)
        same_market = np.equal(markets[1:], markets[:-1])
        same_asset = same_market & np.equal(assets[1:], assets[:-1])
        invalid = (
            np.less(markets[1:], markets[:-1])
            | (same_market & np.less(assets[1:], assets[:-1]))
            | (same_asset & np.less_equal(timestamps[1:], timestamps[:-1]))
        )
        if bool(np.any(invalid)):
            raise ValueError(error)
        state.last_key = (
            str(markets[-1]),
            str(assets[-1]),
            pd.Timestamp(timestamps[-1]),
        )
    return ordered.reset_index(drop=True)


def _normalized_primary_keys(
    frame: pd.DataFrame,
    primary_key: tuple[str, ...],
    *,
    error: str,
) -> pd.DataFrame:
    if any(column not in frame for column in primary_key):
        raise ValueError(error)
    keys = frame.loc[:, list(primary_key)].copy()
    for column in primary_key:
        if column == "timestamp":
            keys[column] = strict_timestamp_series(keys[column], error=error)
        else:
            keys[column] = strict_hashable_key_series(keys[column], error=error)
    return keys


def _panel_primary_keys(
    bars: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
) -> pd.DataFrame:
    required = {"market_id", "asset_id", "timestamp"}
    if set(primary_key) != required or any(
        column not in bars for column in ("source_asset_id", "timestamp")
    ):
        raise ValueError("invalid_crypto_panel_primary_key_contract")
    expected = pd.DataFrame(index=bars.index)
    expected["market_id"] = str(MARKET_ID)
    try:
        expected["asset_id"] = bars["source_asset_id"].map(canonical_crypto_asset_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_crypto_panel_source_asset_id") from exc
    expected["timestamp"] = strict_timestamp_series(
        bars["timestamp"],
        error="invalid_crypto_panel_dependency_timestamp",
    )
    return expected.loc[:, list(primary_key)]


def _iter_panel_primary_key_chunks(
    bars: Iterable[pd.DataFrame],
    *,
    primary_key: tuple[str, ...],
) -> Iterator[pd.DataFrame]:
    for chunk in bars:
        if chunk.empty:
            continue
        policy = current_runtime_resource_policy()
        if policy is None:
            raise RuntimeError("runtime_resource_policy_not_active")
        with policy.lease(
            "crypto.validation.panel-keys",
            max(1, pandas_frame_payload_bytes(chunk)),
        ):
            yield _panel_primary_keys(chunk, primary_key=primary_key)
