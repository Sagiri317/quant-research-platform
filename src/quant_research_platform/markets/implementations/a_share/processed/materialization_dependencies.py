"""A-share processed dependency planning and materialization helpers."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from contextlib import ExitStack

import pandas as pd

from quant_research_platform.catalog.processed import ProcessedDataCatalog
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow, require_date_window
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io as materialization_io,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_planning as materialization_planning,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_validation as materialization_validation,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_factors as materialize_factors,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_labels as materialize_labels,
)
from quant_research_platform.markets.session_windows import (
    require_non_negative_session_offset,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout


def ensure_processed_dependencies(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
    processed_catalog: ProcessedDataCatalog,
    dependency_windows: dict[str, TimeWindow],
    materialize_one,
    cleanup_stack: ExitStack | None = None,
) -> None:
    dependency_window = _processed_dependency_time_window(
        spec,
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        sessions=sessions,
    )
    with runtime_resource_policy_scope(policy):
        for data_id in spec.processed_input_data_ids:
            dependency = processed_catalog.get(data_id)
            required_window = merge_time_windows(
                dependency_window,
                dependency_windows.get(data_id, dependency_window),
            )
            dependency_root = materialization_io.processed_root(layout, dependency)
            if dependency_root.exists():
                if data_id in written:
                    continue
                materialization_planning.require_dataset_coverage(
                    dependency_root,
                    required_window,
                    sessions=sessions,
                    resource_policy=policy,
                )
                continue
            materialize_one(
                dependency,
                layout=layout,
                source_id=source_id,
                time_window=required_window,
                policy=policy,
                sessions=sessions,
                written=written,
                requested=False,
                dependency_windows=dependency_windows,
                cleanup_stack=cleanup_stack,
            )


def planned_processed_dependency_windows(
    targets: tuple[ProcessedDataSpec, ...],
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    sessions: pd.DatetimeIndex,
) -> dict[str, TimeWindow]:
    windows: dict[str, TimeWindow] = {}
    for spec in targets:
        if not spec.processed_input_data_ids:
            continue
        dependency_window = _processed_dependency_time_window(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            sessions=sessions,
        )
        for data_id in spec.processed_input_data_ids:
            windows[data_id] = merge_time_windows(
                windows.get(data_id, dependency_window),
                dependency_window,
            )
    return windows


def required_session_offsets(
    targets: tuple[ProcessedDataSpec, ...],
) -> tuple[int, int]:
    offsets = tuple(_processed_dependency_session_offsets(spec) for spec in targets)
    return (
        max((lookback for lookback, _ in offsets), default=0),
        max((lookahead for _, lookahead in offsets), default=0),
    )


def merge_time_windows(left: TimeWindow, right: TimeWindow) -> TimeWindow:
    require_date_window(left, context="a_share_dependency_window")
    require_date_window(right, context="a_share_dependency_window")
    return TimeWindow(min(left.start, right.start), max(left.end, right.end))


def _processed_dependency_time_window(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    sessions: pd.DatetimeIndex,
) -> TimeWindow:
    if spec.kind not in {ProcessedDataKind.FACTOR, ProcessedDataKind.LABEL}:
        return time_window
    lookback, lookahead = _processed_dependency_session_offsets(spec)
    if lookback <= 0 and lookahead <= 0:
        return time_window

    return _session_offset_time_window(
        sessions=sessions,
        time_window=time_window,
        lookback_sessions=lookback,
        lookahead_sessions=lookahead,
    )


def _processed_dependency_session_offsets(
    spec: ProcessedDataSpec,
) -> tuple[int, int]:
    if spec.kind is ProcessedDataKind.PANEL:
        return 0, 1
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return 0, 1
    if spec.kind is ProcessedDataKind.FACTOR:
        definitions = materialize_factors.select_factor_definitions_for_spec(
            spec,
            factor_definitions.factor_catalog(),
        )
        materialization_validation.validate_processed_input_contract(
            spec,
            contract=materialization_validation.factor_input_contract(definitions),
            purpose="factor",
        )
        lookback = factor_calculations.factor_lookback_bars(definitions)
        return lookback, 0
    if spec.kind is ProcessedDataKind.LABEL:
        definitions = materialize_labels.select_label_definitions_for_spec(
            spec,
            label_definitions.label_catalog(),
        )
        materialization_validation.validate_processed_input_contract(
            spec,
            contract=materialization_validation.label_input_contract(definitions),
            purpose="label",
        )
        lookahead = label_calculations.label_lookahead_bars(definitions)
        return 0, lookahead
    return 0, 0


def _session_offset_time_window(
    *,
    sessions: pd.DatetimeIndex,
    time_window: TimeWindow,
    lookback_sessions: int,
    lookahead_sessions: int,
) -> TimeWindow:
    require_date_window(time_window, context="a_share_dependency_session_offset")
    require_non_negative_session_offset(
        lookback_sessions, error="invalid_a_share_session_offset"
    )
    require_non_negative_session_offset(
        lookahead_sessions, error="invalid_a_share_session_offset"
    )
    if len(sessions) == 0 or (lookback_sessions <= 0 and lookahead_sessions <= 0):
        return time_window
    start = pd.Timestamp(time_window.start).normalize()
    end = pd.Timestamp(time_window.end).normalize()
    first = bisect_left(sessions, start)
    stop = bisect_right(sessions, end)
    if first == stop:
        return time_window
    start_position = max(0, first - lookback_sessions)
    end_position = min(
        len(sessions) - 1,
        stop - 1 + lookahead_sessions,
    )
    return TimeWindow(
        sessions[start_position].date(),
        sessions[end_position].date(),
    )
