"""A-share processed dataset materialization."""

from __future__ import annotations

from contextlib import ExitStack

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog_module,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_dependencies as materialization_dependencies,
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
    materialize_annotations as materialize_annotations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_factors as materialize_factors,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_labels as materialize_labels,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialize_panel as materialize_panel,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    source_ids as source_ids,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    _inspect_dataset_owned,
    _release_owned_value,
    dataset_inspection_retained_memory_bytes,
)
from quant_research_platform.storage.datasets import DatasetInspection
from quant_research_platform.storage.layout import StorageLayout

processed_catalog = processed_catalog_module.processed_catalog


def materialize_processed_data(
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None = None,
    kinds: frozenset[ProcessedDataKind] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, DatasetInspection]:
    """Materialize registered A-share processed datasets."""

    with runtime_resource_policy_scope(resource_policy) as policy:
        return _materialize_processed_data_with_policy(
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            data_ids=data_ids,
            kinds=kinds,
            resource_policy=policy,
        )


def _materialize_processed_data_with_policy(
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    data_ids: tuple[str, ...] | None,
    kinds: frozenset[ProcessedDataKind] | None,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, DatasetInspection]:
    policy = resource_policy
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = processed_catalog()
    if data_ids is not None:
        targets = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        targets = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not targets:
        raise ValueError("no_processed_targets")
    materialization_validation.validate_processed_dependency_graph(catalog, targets)
    materialization_validation.validate_active_processed_profile_frequencies(
        catalog,
        targets,
    )
    lookback_sessions, lookahead_sessions = (
        materialization_dependencies.required_session_offsets(targets)
    )
    with runtime_resource_policy_scope(policy), ExitStack() as cleanup_stack:
        sessions = materialization_planning.available_trading_sessions(
            layout,
            source_id,
            time_window=time_window,
            lookback_sessions=lookback_sessions,
            lookahead_sessions=lookahead_sessions,
            resource_policy=policy,
        )
        materialization_planning.require_target_sessions(sessions, time_window)
        dependency_windows = (
            materialization_dependencies.planned_processed_dependency_windows(
                targets,
                layout=layout,
                source_id=source_id,
                time_window=time_window,
                sessions=sessions,
            )
        )
        written: dict[str, DatasetInspection] = {}
        for spec in targets:
            target_time_window = materialization_dependencies.merge_time_windows(
                time_window,
                dependency_windows.get(spec.data_id, time_window),
            )
            _materialize_one(
                spec,
                layout=layout,
                source_id=source_id,
                time_window=target_time_window,
                policy=policy,
                sessions=sessions,
                written=written,
                requested=True,
                dependency_windows=dependency_windows,
                cleanup_stack=cleanup_stack,
            )
        return written


def _materialize_one(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: pd.DatetimeIndex,
    written: dict[str, DatasetInspection],
    requested: bool,
    dependency_windows: dict[str, TimeWindow],
    cleanup_stack: ExitStack | None = None,
) -> DatasetInspection:
    root = materialization_io.processed_root(layout, spec)
    if spec.data_id in written:
        return written[spec.data_id]
    retained_written_bytes = sum(
        dataset_inspection_retained_memory_bytes(inspection)
        for inspection in written.values()
    )
    if root.exists():
        if requested:
            raise ValueError("dataset_exists")
        materialization_planning.require_dataset_coverage(
            root,
            time_window,
            sessions=sessions,
            resource_policy=policy,
            retained_payload_bytes=retained_written_bytes,
        )
        owned = _inspect_dataset_owned(
            root,
            time_window=time_window,
            resource_policy=policy,
        )
        inspection = owned.value

        def release_inspection() -> None:
            nonlocal inspection
            inspection = None
            _release_owned_value(
                owned,
                owner="a_share.materializer.inspection.result",
            )

        if cleanup_stack is None:
            _release_owned_value(
                owned,
                owner="a_share.materializer.inspection.result",
            )
        else:
            cleanup_stack.callback(release_inspection)
        return inspection
    catalog = processed_catalog()
    materialization_dependencies.ensure_processed_dependencies(
        spec,
        layout=layout,
        source_id=source_id,
        time_window=time_window,
        policy=policy,
        sessions=sessions,
        written=written,
        processed_catalog=catalog,
        dependency_windows=dependency_windows,
        cleanup_stack=cleanup_stack,
        materialize_one=_materialize_one,
    )
    if spec.kind is ProcessedDataKind.PANEL:
        return materialize_panel.write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            market_bars_data_id=source_ids.market_bars_source_data_id(spec),
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            cleanup_stack=cleanup_stack,
        )
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return materialize_annotations.write_annotations(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            cleanup_stack=cleanup_stack,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return materialize_factors.write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            cleanup_stack=cleanup_stack,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return materialize_labels.write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            cleanup_stack=cleanup_stack,
        )
    raise ValueError(f"unsupported_processed_kind: {spec.kind}")


__all__ = ["materialize_processed_data"]
