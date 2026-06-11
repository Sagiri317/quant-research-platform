"""A-share processed dataset materialization."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import (
    SessionWindow,
    SessionWindowPlanner,
    session_dates,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_A_SHARE_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels.definitions import (
    a_share_forward_return_label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
    panels,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20


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

    policy = resource_policy or RuntimeResourcePolicy.from_environment()
    if (data_ids is None) == (kinds is None):
        raise ValueError("data_ids_xor_kinds_required")
    catalog = a_share_processed_catalog()
    if data_ids is not None:
        targets = tuple(catalog.get(data_id) for data_id in data_ids)
    else:
        assert kinds is not None
        targets = tuple(spec for spec in catalog.list() if spec.kind in kinds)
    if not targets:
        raise ValueError("no_processed_targets")
    sessions = _available_trading_sessions(layout, source_id)
    _require_target_sessions(sessions, time_window)
    written: dict[str, DatasetInspection] = {}
    for spec in targets:
        _materialize_one(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            requested=True,
        )
    return written


def _materialize_one(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
    requested: bool,
) -> DatasetInspection:
    root = _processed_root(layout, spec)
    if spec.data_id in written:
        return written[spec.data_id]
    if root.exists():
        if requested:
            raise ValueError("dataset_exists")
        _require_dataset_coverage(root, time_window, sessions=sessions)
        return inspect_dataset(root, time_window=time_window)
    if spec.kind is ProcessedDataKind.PANEL:
        return _write_panel(
            spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    panel_spec = _catalog_get("daily_panel")
    panel_root = _processed_root(layout, panel_spec)
    if not panel_root.exists():
        _materialize_one(
            panel_spec,
            layout=layout,
            source_id=source_id,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
            requested=False,
        )
    elif panel_spec.data_id not in written:
        _require_dataset_coverage(panel_root, time_window, sessions=sessions)
    if spec.kind is ProcessedDataKind.ANNOTATION:
        return _write_annotation(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        return _write_factors(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    if spec.kind is ProcessedDataKind.LABEL:
        return _write_labels(
            spec,
            layout=layout,
            time_window=time_window,
            policy=policy,
            sessions=sessions,
            written=written,
        )
    raise ValueError(f"unsupported_processed_kind: {spec.kind}")


def _write_panel(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    source_id: SourceId,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    daily_market_root = layout.interim_data_root(source_id, "daily_market")
    stock_master = read_dataset(layout.interim_data_root(source_id, "stock_master"))
    market_cap = read_dataset(layout.interim_data_root(source_id, "market_cap"))
    trading_calendar = read_dataset(
        layout.interim_data_root(source_id, "trading_calendar")
    )
    estimated = _estimate_panel_bytes_per_session(
        sessions,
        time_window=time_window,
        daily_market_root=daily_market_root,
        inspection=inspect_dataset(daily_market_root),
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
        required_columns=spec.required_columns,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=inspect_dataset(daily_market_root),
        estimated_bytes_per_session=estimated,
        lookahead_sessions=1,
        reason="panel_next_session",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            daily_market = read_dataset(daily_market_root, time_window=window.read)
            panel = panels.compute_daily_panel(
                daily_market,
                stock_master=stock_master,
                market_cap=market_cap,
                trading_calendar=trading_calendar,
            )
            output = _filter_output_sessions(panel, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_annotation(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel_root = _processed_root(layout, _catalog_get("daily_panel"))
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=inspect_dataset(panel_root),
        reason="research_assembly",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            panel = read_dataset(panel_root, time_window=window.core)
            outputs = annotations.derive_annotations(panel, data_ids=(spec.data_id,))
            yield outputs[spec.data_id]

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_factors(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel_root = _processed_root(layout, _catalog_get("daily_panel"))
    available = inspect_dataset(panel_root)
    lookback_days = factor_calculations.price_volume_lookback_days()
    estimated = _estimate_factor_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookback_days=lookback_days,
        required_columns=spec.required_columns,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=available,
        estimated_bytes_per_session=estimated,
        warmup_sessions=lookback_days,
        reason="factor_warmup",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            panel = read_dataset(panel_root, time_window=window.read)
            factors = factor_calculations.calculate_price_volume_factors_for_panel(
                panel,
            )
            output = _filter_output_sessions(factors, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_labels(
    spec: ProcessedDataSpec,
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    sessions: tuple[pd.Timestamp, ...],
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    panel_root = _processed_root(layout, _catalog_get("daily_panel"))
    available = inspect_dataset(panel_root)
    lookahead_sessions = max(
        definition.lookahead
        for definition in a_share_forward_return_label_definitions()
    )
    estimated = _estimate_label_bytes_per_session(
        sessions,
        time_window=time_window,
        panel_root=panel_root,
        inspection=available,
        lookahead_sessions=lookahead_sessions,
        required_columns=spec.required_columns,
    )
    windows = _planned_session_windows(
        sessions,
        time_window=time_window,
        policy=policy,
        inspection=available,
        estimated_bytes_per_session=estimated,
        lookahead_sessions=lookahead_sessions,
        reason="label_lookahead",
    )

    def chunks() -> Iterable[pd.DataFrame]:
        for window in windows:
            panel = read_dataset(panel_root, time_window=window.read)
            labels = label_calculations.calculate_forward_return_labels_for_panel(
                panel,
            )
            output = _filter_output_sessions(labels, window.core_sessions)
            yield output.loc[:, list(spec.required_columns)].copy()

    return _write_and_inspect(
        spec,
        chunks(),
        layout=layout,
        time_window=time_window,
        written=written,
    )


def _write_and_inspect(
    spec: ProcessedDataSpec,
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    layout: StorageLayout,
    time_window: TimeWindow,
    written: dict[str, DatasetInspection],
) -> DatasetInspection:
    root = _processed_root(layout, spec)
    if root.exists():
        raise ValueError("dataset_exists")
    output = _iter_sorted_output_chunks(frame, spec)
    write_dataset(root, output, spec=spec)
    inspection = inspect_dataset(root, time_window=time_window)
    written[spec.data_id] = inspection
    return inspection


def _processed_root(layout: StorageLayout, spec: ProcessedDataSpec) -> Path:
    return layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)


def _catalog_get(data_id: str) -> ProcessedDataSpec:
    return a_share_processed_catalog().get(data_id)


def _filter_output_sessions(
    frame: pd.DataFrame,
    sessions: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    session_dates_set = {session.date() for session in sessions}
    dates = output["timestamp"].dt.date
    return output.loc[dates.isin(session_dates_set).fillna(False)].copy()


def _require_dataset_coverage(
    root: Path,
    time_window: TimeWindow,
    *,
    sessions: tuple[pd.Timestamp, ...],
) -> None:
    available = _available_window(inspect_dataset(root))
    target = _target_sessions(sessions, time_window)
    if available.start > target[0].date() or available.end < target[-1].date():
        raise ValueError("insufficient_dataset_coverage")


def _available_window(inspection: DatasetInspection) -> TimeWindow:
    if inspection.date_min is None or inspection.date_max is None:
        raise ValueError("time_window_not_supported")
    return TimeWindow(inspection.date_min, inspection.date_max)


def _iter_sorted_output_chunks(
    frame: pd.DataFrame | Iterable[pd.DataFrame],
    spec: ProcessedDataSpec,
) -> Iterable[pd.DataFrame]:
    frames = [frame] if isinstance(frame, pd.DataFrame) else frame
    for chunk in frames:
        output = chunk.loc[:, list(spec.required_columns)].copy()
        if output.empty:
            continue
        yield output.sort_values(list(spec.primary_key)).reset_index(drop=True)


def _available_trading_sessions(
    layout: StorageLayout,
    source_id: SourceId,
) -> tuple[pd.Timestamp, ...]:
    calendar = read_dataset(layout.interim_data_root(source_id, "trading_calendar"))
    source_market_type = pd.to_numeric(
        calendar["source_market_type"],
        errors="coerce",
    )
    mask = (
        calendar["is_trading_day"].astype(bool)
        & source_market_type.isin(CURRENT_A_SHARE_MARKET_TYPES)
    )
    return session_dates(calendar.loc[mask, "timestamp"])


def _target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> tuple[pd.Timestamp, ...]:
    target = tuple(
        session
        for session in sessions
        if time_window.start <= session.date() <= time_window.end
    )
    if not target:
        raise ValueError("no_trading_sessions_in_time_window")
    return target


def _require_target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> None:
    _ = _target_sessions(sessions, time_window)


def _planned_session_windows(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    policy: RuntimeResourcePolicy,
    inspection: DatasetInspection,
    reason: str,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
    estimated_bytes_per_session: int | None = None,
) -> tuple[SessionWindow, ...]:
    target_sessions = _target_sessions(sessions, time_window)
    estimated = (
        _estimated_bytes_per_session(inspection, len(target_sessions))
        if estimated_bytes_per_session is None
        else estimated_bytes_per_session
    )
    resource_plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=len(target_sessions),
        estimated_bytes_per_session=estimated,
    )
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=resource_plan.target_core_session_count,
        warmup_sessions=warmup_sessions,
        lookahead_sessions=lookahead_sessions,
    ).windows(time_window, reason=reason)


def _estimated_bytes_per_session(
    inspection: DatasetInspection,
    session_count: int,
) -> int:
    size = int(inspection.manifest.get("file_size_bytes", 0) or 0)
    if size <= 0:
        size = max(1, int(inspection.row_count)) * 128
    return max(1, ceil(size / max(1, session_count)))


def _estimate_panel_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    daily_market_root: Path,
    inspection: DatasetInspection,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        lookahead_sessions=1,
        reason="panel_sizing",
    )
    daily_market = read_dataset(daily_market_root, time_window=sample.read)
    panel = panels.compute_daily_panel(
        daily_market,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
    )
    output = _filter_output_sessions(panel, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(daily_market, panel, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _estimate_factor_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookback_days: int,
    required_columns: tuple[str, ...],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        warmup_sessions=lookback_days,
        reason="factor_sizing",
    )
    panel = read_dataset(panel_root, time_window=sample.read)
    factors = factor_calculations.calculate_price_volume_factors_for_panel(panel)
    output = _filter_output_sessions(factors, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(panel, factors, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _estimate_label_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookahead_sessions: int,
    required_columns: tuple[str, ...],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        lookahead_sessions=lookahead_sessions,
        reason="label_sizing",
    )
    panel = read_dataset(panel_root, time_window=sample.read)
    labels = label_calculations.calculate_forward_return_labels_for_panel(panel)
    output = _filter_output_sessions(labels, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(panel, labels, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def _sample_window(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    reason: str,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
) -> SessionWindow:
    target_sessions = _target_sessions(sessions, time_window)
    sample_session_count = min(len(target_sessions), SIZING_SAMPLE_SESSION_CAP)
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=sample_session_count,
        warmup_sessions=warmup_sessions,
        lookahead_sessions=lookahead_sessions,
    ).windows(time_window, reason=reason)[0]


def _measured_peak_bytes_per_session(
    *,
    frames: tuple[pd.DataFrame, ...],
    row_count: int,
    sessions: tuple[pd.Timestamp, ...],
    inspection: DatasetInspection,
    target_session_count: int,
) -> int:
    if row_count <= 0:
        return 1
    sample_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames
    )
    sample_rows_per_session = max(1, row_count) / max(1, len(sessions))
    target_rows_per_session = max(
        sample_rows_per_session,
        int(inspection.row_count) / max(1, target_session_count),
    )
    bytes_per_row = sample_bytes / max(1, row_count)
    return max(1, ceil(bytes_per_row * target_rows_per_session))


__all__ = ["materialize_processed_data"]
