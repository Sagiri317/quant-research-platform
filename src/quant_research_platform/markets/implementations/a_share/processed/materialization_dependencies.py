"""A-share processed dependency planning and materialization helpers."""

from __future__ import annotations

from pathlib import Path

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
    materialization_io,
    materialization_planning,
    materialization_validation,
    materialize_factors,
    materialize_labels,
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
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    processed_catalog: ProcessedDataCatalog,
    dependency_windows: dict[str, TimeWindow],
    panel_key_coverages: dict[Path, dict[object, tuple[pd.Timestamp, ...]]],
    materialize_one,
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
                panel_key_coverages=panel_key_coverages,
            )


def planned_processed_dependency_windows(
    targets: tuple[ProcessedDataSpec, ...],
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    sessions: tuple[pd.Timestamp, ...],
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
    sessions: tuple[pd.Timestamp, ...],
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
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
    lookback_sessions: int,
    lookahead_sessions: int,
) -> TimeWindow:
    require_date_window(time_window, context="a_share_dependency_session_offset")
    if not sessions or (lookback_sessions <= 0 and lookahead_sessions <= 0):
        return time_window
    normalized_sessions = tuple(
        pd.Timestamp(session).normalize() for session in sessions
    )
    start = pd.Timestamp(time_window.start).normalize()
    end = pd.Timestamp(time_window.end).normalize()
    core_positions = [
        position
        for position, session in enumerate(normalized_sessions)
        if start <= session <= end
    ]
    if not core_positions:
        return time_window
    start_position = max(0, core_positions[0] - max(0, int(lookback_sessions)))
    end_position = min(
        len(normalized_sessions) - 1,
        core_positions[-1] + max(0, int(lookahead_sessions)),
    )
    return TimeWindow(
        normalized_sessions[start_position].date(),
        normalized_sessions[end_position].date(),
    )

__all__ = [
    "ensure_processed_dependencies",
    "merge_time_windows",
    "planned_processed_dependency_windows",
    "required_session_offsets",
]
