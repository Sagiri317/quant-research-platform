"""A-share processed materialization planning helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow, require_date_window
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.processed import panels
from quant_research_platform.markets.session_windows import (
    SessionWindow,
    SessionWindowPlanner,
    normalize_session_dates,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20

def filter_output_sessions(
    frame: pd.DataFrame,
    sessions: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    session_dates_set = {session.date() for session in sessions}
    dates = output["timestamp"].dt.date
    return output.loc[dates.isin(session_dates_set).fillna(False)].copy()


def require_dataset_coverage(
    root: Path,
    time_window: TimeWindow,
    *,
    sessions: tuple[pd.Timestamp, ...],
) -> None:
    require_date_window(time_window, context="a_share_dataset_coverage")
    target = _target_sessions(sessions, time_window)
    coverage_window = TimeWindow(target[0].date(), target[-1].date())
    available_sessions = _dataset_session_dates(root, coverage_window)
    missing_sessions = [
        session.date()
        for session in target
        if session.date() not in available_sessions
    ]
    if missing_sessions:
        raise ValueError(
            "insufficient_dataset_coverage: "
            f"missing_sessions={len(missing_sessions)} "
            f"first_missing_session={missing_sessions[0]}"
        )


def _dataset_session_dates(
    root: Path,
    time_window: TimeWindow,
) -> set[object]:
    inspection = inspect_dataset(root)
    if "timestamp" not in inspection.columns:
        raise ValueError("time_window_not_supported")
    chunks = read_dataset(
        root,
        columns=["timestamp"],
        time_window=time_window,
        mode=DatasetReadMode.CHUNKED,
    )
    dates: set[object] = set()
    for chunk in chunks:
        timestamp = pd.to_datetime(chunk["timestamp"], errors="coerce")
        dates.update(timestamp.dropna().dt.date)
    return dates


def processed_panel_key_coverage(
    root: Path,
    *,
    cache: dict[Path, dict[object, tuple[pd.Timestamp, ...]]] | None = None,
) -> dict[object, tuple[pd.Timestamp, ...]]:
    cache_key = root.resolve()
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    inspection = inspect_dataset(root)
    missing = {"asset_id", "timestamp"}.difference(inspection.columns)
    if missing:
        raise ValueError(
            "missing_processed_panel_key_columns: " + ", ".join(sorted(missing))
        )
    chunks = read_dataset(
        root,
        columns=["asset_id", "timestamp"],
        mode=DatasetReadMode.CHUNKED,
    )
    key_chunks = (chunks,) if isinstance(chunks, pd.DataFrame) else chunks
    coverage: dict[object, set[pd.Timestamp]] = {}
    for chunk in key_chunks:
        if chunk.empty:
            continue
        keys = chunk.loc[:, ["asset_id", "timestamp"]].copy()
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], errors="coerce")
        keys = keys.dropna(subset=["asset_id", "timestamp"])
        for asset_id, asset_rows in keys.groupby("asset_id", sort=False):
            asset_dates = coverage.setdefault(asset_id, set())
            asset_dates.update(asset_rows["timestamp"].dt.normalize())
    result = {
        asset_id: tuple(sorted(asset_dates))
        for asset_id, asset_dates in coverage.items()
    }
    if cache is not None:
        cache[cache_key] = result
    return result


def available_trading_sessions(
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
        & source_market_type.isin(CURRENT_MARKET_TYPES)
    )
    return normalize_session_dates(calendar.loc[mask, "timestamp"])


def _target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> tuple[pd.Timestamp, ...]:
    require_date_window(time_window, context="a_share_target_sessions")
    target = tuple(
        session
        for session in sessions
        if time_window.start <= session.date() <= time_window.end
    )
    if not target:
        raise ValueError("no_trading_sessions_in_time_window")
    return target


def require_target_sessions(
    sessions: tuple[pd.Timestamp, ...],
    time_window: TimeWindow,
) -> None:
    _ = _target_sessions(sessions, time_window)


def planned_session_windows(
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
    require_date_window(time_window, context="a_share_planned_session_windows")
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


def estimate_panel_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    market_bars_root: Path,
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
    market_bars = read_dataset(market_bars_root, time_window=sample.read)
    panel = panels.compute_market_panel(
        market_bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=trading_calendar,
    )
    output = filter_output_sessions(panel, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(market_bars, panel, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def estimate_factor_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookback_days: int,
    panel_input_columns: tuple[str, ...],
    required_columns: tuple[str, ...],
    definitions: Sequence[FactorDefinition],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        warmup_sessions=lookback_days,
        reason="factor_sizing",
    )
    panel = read_dataset(
        panel_root,
        time_window=sample.read,
        columns=panel_input_columns,
    )
    factors = factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=definitions,
    )
    output = filter_output_sessions(factors, sample.core_sessions)
    selected = output.loc[:, list(required_columns)]
    return _measured_peak_bytes_per_session(
        frames=(panel, factors, selected),
        row_count=len(selected),
        sessions=sample.core_sessions,
        inspection=inspection,
        target_session_count=len(_target_sessions(sessions, time_window)),
    )


def estimate_label_bytes_per_session(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    panel_root: Path,
    inspection: DatasetInspection,
    lookahead_sessions: int,
    panel_input_columns: tuple[str, ...],
    required_columns: tuple[str, ...],
    definitions: Sequence[LabelDefinition],
) -> int:
    sample = _sample_window(
        sessions,
        time_window=time_window,
        lookahead_sessions=lookahead_sessions,
        reason="label_sizing",
    )
    panel = read_dataset(
        panel_root,
        time_window=sample.read,
        columns=panel_input_columns,
    )
    labels = label_calculations.calculate_labels_for_panel(
        panel,
        definitions=definitions,
    )
    output = filter_output_sessions(labels, sample.core_sessions)
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
    require_date_window(time_window, context="a_share_sample_window")
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

__all__ = [
    "available_trading_sessions",
    "estimate_factor_bytes_per_session",
    "estimate_label_bytes_per_session",
    "estimate_panel_bytes_per_session",
    "filter_output_sessions",
    "planned_session_windows",
    "processed_panel_key_coverage",
    "require_dataset_coverage",
    "require_target_sessions",
]
