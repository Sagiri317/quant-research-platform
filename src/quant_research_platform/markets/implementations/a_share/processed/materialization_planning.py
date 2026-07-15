"""A-share processed materialization planning helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import ceil
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow, require_date_window
from quant_research_platform.contracts.validation import strict_hashable_key_series
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    calculations as label_calculations,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io,
    panels,
)
from quant_research_platform.markets.implementations.a_share.validation import (
    strict_values,
)
from quant_research_platform.markets.session_windows import (
    SessionWindow,
    SessionWindowPlanner,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage.datasets import (
    DatasetInspection,
    inspect_dataset,
)
from quant_research_platform.storage.layout import StorageLayout

SIZING_SAMPLE_SESSION_CAP = 20


def filter_output_sessions(
    frame: pd.DataFrame,
    sessions: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = strict_values.strict_timestamp_value_series(
        output["timestamp"],
        error="invalid_output_session_timestamp",
    )
    session_dates_set = {session.date() for session in sessions}
    dates = output["timestamp"].dt.date
    return output.loc[dates.isin(session_dates_set)].copy()


def require_dataset_coverage(
    root: Path,
    time_window: TimeWindow,
    *,
    sessions: tuple[pd.Timestamp, ...],
    resource_policy: RuntimeResourcePolicy,
) -> None:
    require_date_window(time_window, context="a_share_dataset_coverage")
    target = _target_sessions(sessions, time_window)
    coverage_window = TimeWindow(target[0].date(), target[-1].date())
    available_sessions = _dataset_session_dates(
        root,
        coverage_window,
        resource_policy=resource_policy,
    )
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
    *,
    resource_policy: RuntimeResourcePolicy,
) -> set[object]:
    inspection = inspect_dataset(root)
    if "timestamp" not in inspection.columns:
        raise ValueError("time_window_not_supported")
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        root,
        columns=["timestamp"],
        time_window=time_window,
        resource_policy=resource_policy,
        target_chunk_bytes=resource_policy.payload_target_bytes,
    )
    dates: set[object] = set()
    for chunk in chunks:
        timestamp = strict_values.strict_timestamp_value_series(
            chunk["timestamp"],
            error="invalid_dataset_session_timestamp",
        )
        dates.update(timestamp.dt.date)
    return dates


def processed_panel_key_coverage(
    root: Path,
    *,
    cache: dict[Path, dict[object, tuple[pd.Timestamp, ...]]] | None = None,
    resource_policy: RuntimeResourcePolicy,
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
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        root,
        columns=["asset_id", "timestamp"],
        time_window=None,
        resource_policy=resource_policy,
        target_chunk_bytes=resource_policy.payload_target_bytes,
    )
    coverage: dict[object, set[pd.Timestamp]] = {}
    for chunk in chunks:
        if chunk.empty:
            continue
        keys = chunk.loc[:, ["asset_id", "timestamp"]].copy()
        keys["asset_id"] = strict_hashable_key_series(
            keys["asset_id"],
            error="invalid_processed_panel_asset_id",
        )
        keys["timestamp"] = strict_values.strict_timestamp_value_series(
            keys["timestamp"],
            error="invalid_processed_panel_timestamp",
        )
        for asset_id, asset_rows in keys.groupby("asset_id", sort=False):
            asset_dates = coverage.get(asset_id)
            if asset_dates is None:
                asset_dates = set()
                coverage[asset_id] = asset_dates
            new_dates = tuple(
                pd.Timestamp(value)
                for value in asset_rows["timestamp"].dt.normalize().unique()
                if pd.Timestamp(value) not in asset_dates
            )
            asset_dates.update(new_dates)
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
    *,
    time_window: TimeWindow,
    lookback_sessions: int,
    lookahead_sessions: int,
    resource_policy: RuntimeResourcePolicy,
) -> tuple[pd.Timestamp, ...]:
    require_date_window(time_window, context="a_share_trading_sessions")
    if lookback_sessions < 0 or lookahead_sessions < 0:
        raise ValueError("invalid_a_share_session_offset")
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        layout.interim_data_root(source_id, "trading_calendar"),
        resource_policy=resource_policy,
        columns=("source_market_type", "is_trading_day", "timestamp"),
        target_chunk_bytes=resource_policy.payload_target_bytes,
    )
    before: deque[pd.Timestamp] = deque(maxlen=lookback_sessions)
    core: list[pd.Timestamp] = []
    after: list[pd.Timestamp] = []
    current_session: pd.Timestamp | None = None
    current_is_trading = False

    def retain_session(session: pd.Timestamp) -> bool:
        if session.date() < time_window.start:
            if lookback_sessions <= 0:
                return False
            before.append(session)
        elif session.date() <= time_window.end:
            core.append(session)
        elif len(after) < lookahead_sessions:
            after.append(session)
        return session.date() > time_window.end and len(after) >= lookahead_sessions

    def result() -> tuple[pd.Timestamp, ...]:
        return (*before, *core, *after)

    for calendar in chunks:
        source_market_type = strict_values.strict_market_type_series(
            calendar["source_market_type"],
            error="invalid_source_market_type",
        )
        trading = strict_values.strict_trading_bool_series(
            calendar["is_trading_day"],
            error="invalid_is_trading_day",
        )
        timestamps = strict_values.strict_timestamp_value_series(
            calendar["timestamp"],
            error="invalid_trading_calendar_timestamp",
        ).dt.normalize()
        for session, is_trading in zip(
            timestamps,
            trading & source_market_type.isin(CURRENT_MARKET_TYPES),
            strict=True,
        ):
            normalized = pd.Timestamp(session)
            if current_session is not None and normalized < current_session:
                raise ValueError("out_of_order_trading_calendar")
            if current_session is None or normalized == current_session:
                current_session = normalized
                current_is_trading = current_is_trading or bool(is_trading)
                continue
            if current_is_trading and retain_session(current_session):
                return result()
            current_session = normalized
            current_is_trading = bool(is_trading)
    if current_session is not None and current_is_trading:
        retain_session(current_session)
    return result()


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
    if estimated <= 0:
        raise ValueError("invalid_estimated_bytes_per_session")
    target_core_session_count = max(
        1,
        min(
            len(target_sessions),
            policy.payload_target_bytes // estimated,
        ),
    )
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=target_core_session_count,
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
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int = 0,
) -> int:
    samples = _sample_windows(
        sessions,
        time_window=time_window,
        lookahead_sessions=1,
        reason="panel_sizing",
    )
    estimates = []
    for sample in samples:
        market_bars = materialization_io.read_materialized_dataset(
            market_bars_root,
            time_window=sample.read,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_payload_bytes,
        )
        panel = panels.compute_market_panel(
            market_bars,
            stock_master=stock_master,
            market_cap=market_cap,
            trading_calendar=trading_calendar,
        )
        output = filter_output_sessions(panel, sample.core_sessions)
        selected = output.loc[:, list(required_columns)]
        estimates.append(
            _measured_peak_bytes_per_session(
                frames=(market_bars, panel, output),
                output=selected,
                retained_payload_bytes=retained_payload_bytes,
                row_count=len(selected),
                sessions=sample.core_sessions,
                inspection=inspection,
                target_session_count=len(_target_sessions(sessions, time_window)),
            )
        )
    return max(estimates, default=1)


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
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int = 0,
) -> int:
    samples = _sample_windows(
        sessions,
        time_window=time_window,
        warmup_sessions=lookback_days,
        reason="factor_sizing",
    )
    estimates = []
    for sample in samples:
        panel = materialization_io.read_materialized_dataset(
            panel_root,
            time_window=sample.read,
            columns=panel_input_columns,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_payload_bytes,
        )
        factors = factor_calculations.calculate_factors_for_panel(
            panel,
            definitions=definitions,
        )
        output = filter_output_sessions(factors, sample.core_sessions)
        selected = output.loc[:, list(required_columns)]
        estimates.append(
            _measured_peak_bytes_per_session(
                frames=(panel, factors, output),
                output=selected,
                retained_payload_bytes=retained_payload_bytes,
                row_count=len(selected),
                sessions=sample.core_sessions,
                inspection=inspection,
                target_session_count=len(_target_sessions(sessions, time_window)),
            )
        )
    return max(estimates, default=1)


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
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int = 0,
) -> int:
    samples = _sample_windows(
        sessions,
        time_window=time_window,
        lookahead_sessions=lookahead_sessions,
        reason="label_sizing",
    )
    estimates = []
    for sample in samples:
        panel = materialization_io.read_materialized_dataset(
            panel_root,
            time_window=sample.read,
            columns=panel_input_columns,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_payload_bytes,
        )
        labels = label_calculations.calculate_labels_for_panel(
            panel,
            definitions=definitions,
        )
        output = filter_output_sessions(labels, sample.core_sessions)
        selected = output.loc[:, list(required_columns)]
        estimates.append(
            _measured_peak_bytes_per_session(
                frames=(panel, labels, output),
                output=selected,
                retained_payload_bytes=retained_payload_bytes,
                row_count=len(selected),
                sessions=sample.core_sessions,
                inspection=inspection,
                target_session_count=len(_target_sessions(sessions, time_window)),
            )
        )
    return max(estimates, default=1)


def _sample_windows(
    sessions: tuple[pd.Timestamp, ...],
    *,
    time_window: TimeWindow,
    reason: str,
    warmup_sessions: int = 0,
    lookahead_sessions: int = 0,
) -> tuple[SessionWindow, ...]:
    require_date_window(time_window, context="a_share_sample_windows")
    return SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=1,
        warmup_sessions=warmup_sessions,
        lookahead_sessions=lookahead_sessions,
    ).windows(time_window, reason=reason)[:SIZING_SAMPLE_SESSION_CAP]


def _measured_peak_bytes_per_session(
    *,
    frames: tuple[pd.DataFrame, ...],
    output: pd.DataFrame,
    row_count: int,
    sessions: tuple[pd.Timestamp, ...],
    inspection: DatasetInspection,
    target_session_count: int,
    retained_payload_bytes: int = 0,
) -> int:
    if row_count <= 0:
        return 1
    sample_bytes = retained_payload_bytes + sum(
        pandas_frame_payload_bytes(frame) for frame in frames
    ) + 4 * pandas_frame_payload_bytes(output)
    sample_rows_per_session = max(1, row_count) / max(1, len(sessions))
    target_rows_per_session = max(
        sample_rows_per_session,
        int(inspection.row_count) / max(1, target_session_count),
    )
    bytes_per_row = sample_bytes / max(1, row_count)
    return max(1, ceil(bytes_per_row * target_rows_per_session))


def require_known_materialization_payload_bound(
    *,
    frames: Sequence[pd.DataFrame],
    output: pd.DataFrame | None,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    unique_frames = {id(frame): frame for frame in frames}
    if output is not None:
        unique_frames.setdefault(id(output), output)
        if pandas_frame_payload_bytes(output) > resource_policy.payload_target_bytes:
            raise ValueError(
                "a_share_indivisible_market_unit_exceeds_payload_target"
            )
    if (
        sum(pandas_frame_payload_bytes(frame) for frame in unique_frames.values())
        > resource_policy.task_memory_budget_bytes
    ):
        raise ValueError("a_share_known_payloads_exceed_task_memory_budget")

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
