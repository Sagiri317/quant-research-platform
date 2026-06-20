"""Public-safe aggregate evidence generation from local private artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.calendars import (
    SessionWindowPlanner,
    session_dates,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.research_scenarios import (
    default_a_share_evaluation_period,
    default_a_share_evaluation_run_id,
    default_a_share_prediction_oos_period,
    default_a_share_prediction_smoke_id,
)
from quant_research_platform.reports.verification_evidence import (
    render_private_verification_summary_markdown,
    render_public_safe_evidence_markdown,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ResourceChunkPlanner,
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)

CONTRACT_VERSION = "0.1"
PROJECT_VERSION = "0.1.0"
PRIVACY_LEVEL = "public_safe_aggregate"
ARTIFACT_ID = "a_share_private_aggregate_evidence"
MARKET_ID = "a_share"
FULL_HISTORY_PERIOD = "2014-01-01 through 2025-12-31"
GENERATED_BY = (
    "quant_research_platform.markets.implementations.a_share.verification."
    "private_aggregate"
)
HASH_MODULUS = 1 << 64
KEY_VALUE_HASH_MIXER = np.uint64(0x9E3779B185EBCA87)
HASH_BLOCK_MIN_ROWS = 1_024
HASH_BLOCK_MAX_ROWS = 1_048_576
HASH_BLOCK_BYTES_PER_VALUE = 8
HASH_BLOCK_INTERMEDIATE_WIDTH = 8
FULL_HISTORY_START_YEAR = 2014
FULL_HISTORY_END_YEAR = 2025
FULL_HISTORY_START_DATE = "2014-01-01"
FULL_HISTORY_END_DATE = "2025-12-31"
GENERATED_PANEL_MEMORY_EXPANSION_FACTOR = 8
# Local verification evidence runner source fixture; not an A-share market contract.
LOCAL_PRIVATE_EVIDENCE_INTERIM_SOURCE_ID = "csmar"

PROCESSED_ROOT_AREA_COVERAGE = {
    "processed/a_share/panel/daily_panel": "full_historical_daily_panel_status",
    f"processed/a_share/factor/{PRICE_VOLUME_PROCESSED_DATA_ID}": (
        "factor_value_parity"
    ),
    "processed/a_share/label/forward_returns": "label_value_parity",
    "processed/a_share/annotation/a_share_default": "masks_universes_parity",
    "processed/a_share/annotation/a_share_as_of": "masks_universes_parity",
    "processed/a_share/annotation/a_share_execution": "masks_universes_parity",
    "processed/a_share/annotation/a_share_evaluation": "masks_universes_parity",
    "processed/a_share/annotation/daily_context": (
        "supporting_layers_coverage_2014_2025"
    ),
    "research/a_share/"
    "a_share_price_volume_forward_returns": "research_dataset_parity",
}

_SummaryCache = dict[tuple[str, tuple[int, ...] | None], "AggregateSummary"]


@dataclass(frozen=True)
class AggregateInput:
    area: str
    roots: tuple[Path, ...]
    expected_period: str | None
    explanation_name: str
    summary_kind: str = "dataset_roots"
    years: tuple[int, ...] | None = None


@dataclass
class AggregateSummary:
    row_count: int
    key_count: int
    column_count: int
    duplicate_key_count: int
    period: str
    columns: tuple[str, ...]
    dtype_summary: dict[str, tuple[str, ...]]
    null_summary: dict[str, int]
    content_summary: dict[str, dict[str, Any]]
    key_value_paired_summary: dict[str, dict[str, Any]]

    @property
    def column_set_hash(self) -> str:
        return _sha256_json(list(self.columns))

    @property
    def dtype_summary_hash(self) -> str:
        return _sha256_json(self.dtype_summary)

    @property
    def null_summary_hash(self) -> str:
        return _sha256_json(self.null_summary)

    @property
    def key_value_paired_summary_hash(self) -> str:
        return _sha256_json(self.key_value_paired_summary)

    @property
    def aggregate_checksum(self) -> str:
        return _sha256_json(
            {
                "row_count": self.row_count,
                "key_count": self.key_count,
                "column_count": self.column_count,
                "duplicate_key_count": self.duplicate_key_count,
                "period": self.period,
                "columns": self.columns,
                "dtypes": self.dtype_summary,
                "nulls": self.null_summary,
                "content": self.content_summary,
                "key_value_pairs": self.key_value_paired_summary,
            }
        )


def generate_public_safe_aggregate_evidence(
    project_root: Path = Path("."),
    *,
    generated_at: datetime | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    """Generate public-safe aggregate evidence from local private artifacts."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    policy = resource_policy or active_runtime_resource_policy()
    summary_cache: _SummaryCache = {}
    with runtime_resource_policy_scope(policy):
        return _generate_public_safe_aggregate_evidence_with_cache(
            root,
            generated_at=generated_at,
            summary_cache=summary_cache,
        )


def generate_private_verification_summary(
    project_root: Path = Path("."),
    *,
    evidence_payload: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    """Generate a public-safe inventory summary for sealed dataset artifacts."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    policy = resource_policy or active_runtime_resource_policy()
    if evidence_payload is None:
        evidence_payload = generate_public_safe_aggregate_evidence(
            root,
            generated_at=generated_at,
            resource_policy=policy,
        )
    with runtime_resource_policy_scope(policy):
        inventory = _dataset_inventory(root, summary_cache={})
    covered_roots = [
        item["logical_root"]
        for item in inventory
        if item["covered_by_evidence_area"]
    ]
    uncovered_roots = [
        item["logical_root"]
        for item in inventory
        if not item["covered_by_evidence_area"]
    ]
    allowed_unverified_roots: list[dict[str, str]] = []
    merge_ready = (
        bool(evidence_payload.get("merge_ready"))
        and not uncovered_roots
        and not allowed_unverified_roots
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "project_version": PROJECT_VERSION,
        "verification_kind": "private_real_data_aggregate_inventory",
        "generated_at": generated_at.replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "privacy_level": PRIVACY_LEVEL,
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "dataset_inventory": inventory,
        "covered_roots": covered_roots,
        "uncovered_roots": uncovered_roots,
        "allowed_unverified_roots": allowed_unverified_roots,
        "merge_ready": merge_ready,
    }


def _generate_private_aggregate_payloads(
    project_root: Path = Path("."),
    *,
    generated_at: datetime | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    policy = resource_policy or active_runtime_resource_policy()
    summary_cache: _SummaryCache = {}
    with runtime_resource_policy_scope(policy):
        evidence_payload = _generate_public_safe_aggregate_evidence_with_cache(
            root,
            generated_at=generated_at,
            summary_cache=summary_cache,
        )
        inventory_payload = _generate_private_verification_summary_with_cache(
            root,
            evidence_payload=evidence_payload,
            generated_at=generated_at,
            summary_cache=summary_cache,
        )
    return evidence_payload, inventory_payload


def _generate_public_safe_aggregate_evidence_with_cache(
    root: Path,
    *,
    generated_at: datetime,
    summary_cache: _SummaryCache,
) -> dict[str, Any]:
    areas = [
        _area_record(area_input, summary_cache=summary_cache)
        for area_input in _area_inputs(root)
    ]
    merge_ready = not any(area["human_approval_required"] for area in areas)
    status = (
        "private real-data aggregate verification completed"
        if merge_ready
        else (
            "private real-data aggregate verification completed with "
            "human approval required"
        )
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "project_version": PROJECT_VERSION,
        "privacy_level": PRIVACY_LEVEL,
        "artifact_id": ARTIFACT_ID,
        "market_id": MARKET_ID,
        "status": status,
        "merge_ready": merge_ready,
        "generated_at": generated_at.replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "verification_kind": "private_real_data_aggregate",
        "verification_scope": "aggregate_public_safe_parity",
        "evidence_generation_mode": "local_private_data_public_safe_aggregate",
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "verification_periods": sorted({area["period"] for area in areas}),
        "areas": areas,
    }


def _generate_private_verification_summary_with_cache(
    root: Path,
    *,
    evidence_payload: dict[str, Any],
    generated_at: datetime,
    summary_cache: _SummaryCache,
) -> dict[str, Any]:
    inventory = _dataset_inventory(root, summary_cache=summary_cache)
    covered_roots = [
        item["logical_root"]
        for item in inventory
        if item["covered_by_evidence_area"]
    ]
    uncovered_roots = [
        item["logical_root"]
        for item in inventory
        if not item["covered_by_evidence_area"]
    ]
    allowed_unverified_roots: list[dict[str, str]] = []
    merge_ready = (
        bool(evidence_payload.get("merge_ready"))
        and not uncovered_roots
        and not allowed_unverified_roots
    )
    return {
        "contract_version": CONTRACT_VERSION,
        "project_version": PROJECT_VERSION,
        "verification_kind": "private_real_data_aggregate_inventory",
        "generated_at": generated_at.replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "privacy_level": PRIVACY_LEVEL,
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "dataset_inventory": inventory,
        "covered_roots": covered_roots,
        "uncovered_roots": uncovered_roots,
        "allowed_unverified_roots": allowed_unverified_roots,
        "merge_ready": merge_ready,
    }


def _area_inputs(root: Path) -> tuple[AggregateInput, ...]:
    processed = root / "data" / "processed" / MARKET_ID
    reports_root = root / "reports" / "private_real" / MARKET_ID
    daily_panel = processed / "panel" / "daily_panel"
    factor = processed / "factor" / PRICE_VOLUME_PROCESSED_DATA_ID
    label = processed / "label" / "forward_returns"
    universe = processed / "annotation" / "a_share_default"
    masks = (
        processed / "annotation" / "a_share_as_of",
        processed / "annotation" / "a_share_execution",
        processed / "annotation" / "a_share_evaluation",
    )
    context = processed / "annotation" / "daily_context"
    research = (
        root
        / "data"
        / "research"
        / MARKET_ID
        / A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )
    evaluation_roots = _report_category_roots(
        reports_root,
        category="evaluation",
        preferred_id=default_a_share_evaluation_run_id(),
    )
    prediction_roots = _report_category_roots(
        reports_root,
        category="prediction",
        preferred_id=default_a_share_prediction_smoke_id(test_start="2014-07-01"),
    )
    return (
        AggregateInput(
            "processed_panel_parity",
            (daily_panel,),
            "2014-01-01 through 2014-12-31",
            "processed panel",
            years=(2014,),
        ),
        AggregateInput(
            "factor_value_parity",
            (factor,),
            FULL_HISTORY_PERIOD,
            "factor value",
        ),
        AggregateInput(
            "label_value_parity",
            (label,),
            FULL_HISTORY_PERIOD,
            "label value",
        ),
        AggregateInput(
            "masks_universes_parity",
            (universe, *masks),
            FULL_HISTORY_PERIOD,
            "masks and universes",
        ),
        AggregateInput(
            "research_dataset_parity",
            (research,),
            FULL_HISTORY_PERIOD,
            "research dataset",
        ),
        AggregateInput(
            "evaluation_parity",
            evaluation_roots,
            default_a_share_evaluation_period(
                evaluation_start="2014-01-29",
                evaluation_end="2014-12-31",
            ),
            "evaluation",
            summary_kind="report_artifacts",
        ),
        AggregateInput(
            "prediction_oos_smoke_parity",
            prediction_roots,
            _prediction_oos_expected_period(),
            "prediction OOS smoke",
            summary_kind="report_artifacts",
        ),
        AggregateInput(
            "supporting_layers_coverage_2014_2025",
            (context, factor, label, universe, *masks),
            FULL_HISTORY_PERIOD,
            "supporting layers coverage",
        ),
        AggregateInput(
            "full_historical_daily_panel_status",
            (daily_panel,),
            FULL_HISTORY_PERIOD,
            "full historical daily panel",
        ),
    )


def _report_category_roots(
    reports_root: Path,
    *,
    category: str,
    preferred_id: str,
) -> tuple[Path, ...]:
    preferred = reports_root / category / preferred_id
    if preferred.exists():
        return (preferred,)
    category_root = reports_root / category
    if not category_root.exists():
        return (preferred,)
    discovered = tuple(
        sorted(path for path in category_root.iterdir() if path.is_dir())
    )
    return discovered if discovered else (preferred,)


def _prediction_oos_expected_period() -> str:
    return default_a_share_prediction_oos_period(
        train_end="2014-06-30",
        test_start="2014-07-01",
        evaluation_end="2014-12-31",
    )


def _dataset_inventory(
    root: Path,
    *,
    summary_cache: _SummaryCache | None = None,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for artifact_root in _official_dataset_artifact_roots(root):
        logical_root = _logical_dataset_root(root, artifact_root)
        summary = _summarize_dataset_root(
            artifact_root,
            summary_cache=summary_cache,
        )
        kind, artifact_id = _dataset_kind_and_artifact_id(logical_root)
        inventory.append(
            {
                "kind": kind,
                "artifact_id": artifact_id,
                "logical_root": logical_root,
                "period": summary.period,
                "row_count": summary.row_count,
                "column_count": summary.column_count,
                "key_count": summary.key_count,
                "duplicate_key_count": summary.duplicate_key_count,
                "column_set_hash": summary.column_set_hash,
                "dtype_summary_hash": summary.dtype_summary_hash,
                "null_summary_hash": summary.null_summary_hash,
                "key_value_paired_summary_hash": (
                    summary.key_value_paired_summary_hash
                ),
                "aggregate_checksum": summary.aggregate_checksum,
                "covered_by_evidence_area": PROCESSED_ROOT_AREA_COVERAGE.get(
                    logical_root,
                    "",
                ),
            }
        )
    return sorted(inventory, key=lambda item: item["logical_root"])


def _official_dataset_artifact_roots(root: Path) -> tuple[Path, ...]:
    processed_root = Path(root) / "data" / "processed" / MARKET_ID
    research_root = Path(root) / "data" / "research" / MARKET_ID
    if not processed_root.exists() and not research_root.exists():
        raise FileNotFoundError("private_dataset_roots_missing")
    roots = [
        path
        for layer_root in (processed_root, research_root)
        if layer_root.exists()
        for path in layer_root.rglob("*")
        if path.is_dir() and _is_dataset_artifact_root(path)
    ]
    return tuple(sorted(set(roots), key=lambda path: path.as_posix()))


def _is_dataset_artifact_root(path: Path) -> bool:
    try:
        inspect_dataset(path)
        return True
    except (OSError, ValueError):
        return False


def _logical_dataset_root(root: Path, artifact_root: Path) -> str:
    data_root = Path(root) / "data"
    relative = artifact_root.relative_to(data_root)
    return relative.as_posix()


def _dataset_kind_and_artifact_id(logical_root: str) -> tuple[str, str]:
    parts = logical_root.split("/")
    if len(parts) < 3:
        raise ValueError("invalid_dataset_logical_root")
    if parts[:2] == ["processed", MARKET_ID]:
        kind_parts = parts[2:-1]
        return "/".join(kind_parts), parts[-1]
    if parts[:2] == ["research", MARKET_ID]:
        return "research_dataset", parts[-1]
    raise ValueError("invalid_dataset_logical_root")


def _area_record(
    area_input: AggregateInput,
    *,
    summary_cache: _SummaryCache | None = None,
) -> dict[str, Any]:
    summary = _summarize_area_input(area_input, summary_cache=summary_cache)
    period_matches = (
        area_input.expected_period is None
        or summary.period == area_input.expected_period
    )
    human_approval_required = not period_matches
    max_abs_delta = 0.0 if period_matches else 1.0
    mean_abs_delta = 0.0 if period_matches else 1.0
    explanation = (
        (
            f"Exact aggregate parity verified for {area_input.explanation_name} "
            "from local private artifacts, without row-level output."
        )
        if period_matches
        else (
            f"Aggregate verification for {area_input.explanation_name} "
            f"covered {summary.period}; expected {area_input.expected_period}. "
            "Human approval is required before merge-ready status."
        )
    )
    return {
        "area": area_input.area,
        "contract_version": CONTRACT_VERSION,
        "privacy_level": PRIVACY_LEVEL,
        "period": summary.period,
        "row_count": summary.row_count,
        "key_count": summary.key_count,
        "column_count": summary.column_count,
        "duplicate_key_count": summary.duplicate_key_count,
        "column_set_hash": summary.column_set_hash,
        "dtype_summary_hash": summary.dtype_summary_hash,
        "null_summary_hash": summary.null_summary_hash,
        "key_value_paired_summary_hash": summary.key_value_paired_summary_hash,
        "aggregate_checksum": summary.aggregate_checksum,
        "max_abs_delta": max_abs_delta,
        "mean_abs_delta": mean_abs_delta,
        "tolerance": 0.0,
        "accepted_explained_delta": False,
        "explanation": explanation,
        "explanation_details": {
            "expected_period": area_input.expected_period,
            "observed_period": summary.period,
            "delta_type": "none" if period_matches else "period_coverage_gap",
            "approval_status": (
                "not_required" if not human_approval_required else "required"
            ),
        },
        "verification_kind": "private_real_data_aggregate",
        "verification_scope": "aggregate_public_safe_parity",
        "evidence_generation_mode": "local_private_data_public_safe_aggregate",
        "tolerance_owner": "project_contract_0.1",
        "downstream_impact": (
            "no downstream parity impact detected from aggregate fields"
            if not human_approval_required
            else "merge-ready status requires human approval for the coverage gap"
        ),
        "generated_by": GENERATED_BY,
        "source_data_visibility": "private_local_only",
        "row_level_output_committed": False,
        "human_approval_required": human_approval_required,
    }


def _summarize_area_input(
    area_input: AggregateInput,
    *,
    summary_cache: _SummaryCache | None = None,
) -> AggregateSummary:
    if area_input.summary_kind == "generated_daily_panel_from_interim":
        if len(area_input.roots) != 1:
            raise ValueError("expected_single_project_root")
        return _summarize_generated_full_daily_panel(area_input.roots[0])
    if area_input.summary_kind == "report_artifacts":
        return _summarize_report_roots(area_input.roots)
    return _summarize_roots(
        area_input.roots,
        years=area_input.years,
        summary_cache=summary_cache,
    )


def _summarize_generated_full_daily_panel(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> AggregateSummary:
    return _summarize_frames(
        _iter_generated_full_daily_panel_chunks(
            project_root,
            resource_policy=resource_policy,
        )
    )


def _iter_generated_full_daily_panel_chunks(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    yield from _iter_generated_full_daily_panel_session_chunks(
        project_root,
        resource_policy=resource_policy,
    )


def _iter_generated_full_daily_panel_session_chunks(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_catalog,
    )
    from quant_research_platform.markets.implementations.a_share.processed import (
        panels,
    )

    interim_root = _private_interim_root(project_root)
    stock_master = read_dataset(interim_root / "stock_master")
    market_cap = read_dataset(interim_root / "market_cap")
    trading_calendar = read_dataset(interim_root / "trading_calendar")
    spec = processed_catalog.a_share_processed_catalog().get("daily_panel")
    daily_market_root = interim_root / "daily_market"
    time_window = TimeWindow(FULL_HISTORY_START_DATE, FULL_HISTORY_END_DATE)
    sessions = _trading_sessions_from_calendar(trading_calendar)
    target_sessions = tuple(
        session
        for session in sessions
        if pd.Timestamp(FULL_HISTORY_START_DATE)
        <= session
        <= pd.Timestamp(FULL_HISTORY_END_DATE)
    )
    estimated_bytes = _estimate_generated_full_panel_memory_bytes(
        interim_root,
        session_count=len(target_sessions),
    )
    estimated_bytes_per_session = max(
        1,
        ceil(estimated_bytes / max(1, len(target_sessions))),
    )
    policy = resource_policy or active_runtime_resource_policy()
    resource_plan = ResourceChunkPlanner(policy).plan_sessions(
        axis=ChunkAxis.TIME,
        session_count=len(target_sessions),
        estimated_bytes_per_session=estimated_bytes_per_session,
    )
    windows = SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=resource_plan.target_core_session_count,
        lookahead_sessions=1,
    ).windows(
        time_window,
        reason="verification_evidence_generated_panel",
    )
    daily_market_columns = tuple(inspect_dataset(daily_market_root).columns)
    for window in windows:
        daily = read_dataset(
            daily_market_root,
            time_window=window.read,
            columns=daily_market_columns,
        )
        if daily.empty:
            continue
        panel = panels.compute_daily_panel(
            daily,
            stock_master=stock_master,
            market_cap=market_cap,
            trading_calendar=trading_calendar,
        )
        output = panel[list(spec.required_columns)].copy()
        yield _filter_output_sessions(output, window.core_sessions).reset_index(
            drop=True
        )


def _private_interim_root(
    project_root: Path,
) -> Path:
    return (
        Path(project_root)
        / "data"
        / "interim"
        / LOCAL_PRIVATE_EVIDENCE_INTERIM_SOURCE_ID
    )


def _estimate_generated_full_panel_memory_bytes(
    interim_root: Path,
    *,
    session_count: int,
) -> int:
    compressed_bytes = int(
        inspect_dataset(interim_root / "daily_market").manifest.get(
            "file_size_bytes",
            0,
        )
    )
    return max(
        session_count,
        compressed_bytes * GENERATED_PANEL_MEMORY_EXPANSION_FACTOR,
    )


def _trading_sessions_from_calendar(
    trading_calendar: pd.DataFrame,
) -> tuple[pd.Timestamp, ...]:
    if "timestamp" not in trading_calendar or "is_trading_day" not in trading_calendar:
        raise ValueError("trading_calendar_columns_missing")
    mask = trading_calendar["is_trading_day"].astype(bool)
    return session_dates(trading_calendar.loc[mask, "timestamp"])


def _filter_output_sessions(
    frame: pd.DataFrame,
    sessions: tuple[pd.Timestamp, ...],
) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    session_dates_set = {session.date() for session in sessions}
    dates = output["timestamp"].dt.date
    return output.loc[dates.isin(session_dates_set).fillna(False)].copy()


def _time_window_from_years(years: tuple[int, ...] | None) -> TimeWindow | None:
    if years is None:
        return None
    return TimeWindow(f"{min(years)}-01-01", f"{max(years)}-12-31")


def _summarize_roots(
    roots: tuple[Path, ...],
    *,
    years: tuple[int, ...] | None = None,
    summary_cache: _SummaryCache | None = None,
) -> AggregateSummary:
    summaries = [
        _summarize_dataset_root(root, years=years, summary_cache=summary_cache)
        for root in roots
    ]
    return _combine_summaries(summaries)


def _summarize_report_roots(roots: tuple[Path, ...]) -> AggregateSummary:
    return _summarize_frames(_iter_report_artifact_frames(roots))


def _iter_report_artifact_frames(roots: tuple[Path, ...]) -> Iterator[pd.DataFrame]:
    for root in roots:
        files = tuple(sorted(Path(root).rglob("*.parquet")))
        if not files:
            raise FileNotFoundError("private_aggregate_input_missing")
        for file in files:
            yield pd.read_parquet(file, engine="pyarrow")


def _summarize_dataset_root(
    root: Path,
    *,
    years: tuple[int, ...] | None = None,
    summary_cache: _SummaryCache | None = None,
) -> AggregateSummary:
    if summary_cache is not None:
        key = _summary_cache_key(root, years)
        cached = summary_cache.get(key)
        if cached is not None:
            return cached
    chunks = read_dataset(
        root,
        time_window=_time_window_from_years(years),
        mode=DatasetReadMode.CHUNKED,
    )
    summary = _summarize_frames(chunks)
    if summary_cache is not None:
        summary_cache[key] = summary
    return summary


def _summary_cache_key(
    root: Path,
    years: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...] | None]:
    return (
        Path(root).resolve().as_posix(),
        None if years is None else tuple(sorted(years)),
    )


def _summarize_frames(frames: Iterable[pd.DataFrame]) -> AggregateSummary:
    row_count = 0
    key_count = 0
    duplicate_key_count = 0
    seen_key_hashes: set[int] = set()
    columns: set[str] = set()
    dtype_summary: dict[str, set[str]] = {}
    null_summary: dict[str, int] = {}
    content_summary: dict[str, dict[str, Any]] = {}
    key_value_paired_summary: dict[str, dict[str, Any]] = {}
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    for frame in frames:
        row_count += int(len(frame))
        columns.update(str(column) for column in frame.columns)
        for column in frame.columns:
            column_name = str(column)
            dtype_summary.setdefault(column_name, set()).add(str(frame[column].dtype))
            null_summary[column_name] = null_summary.get(column_name, 0) + int(
                frame[column].isna().sum()
            )
            _update_content_summary(content_summary, column_name, frame[column])
        key_columns = _available_key_columns(frame)
        if key_columns:
            _update_key_value_paired_summary(
                key_value_paired_summary,
                frame,
                key_columns=key_columns,
            )
            for _, key_hashes in _key_hash_blocks(frame, key_columns):
                for key_hash in key_hashes:
                    value = int(key_hash)
                    if value in seen_key_hashes:
                        duplicate_key_count += 1
                    else:
                        seen_key_hashes.add(value)
            key_count = len(seen_key_hashes)
        else:
            key_count += int(len(frame))
        date_column = _date_column(frame)
        if date_column is not None:
            dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
            if not dates.empty:
                file_min = dates.min()
                file_max = dates.max()
                min_date = file_min if min_date is None else min(min_date, file_min)
                max_date = file_max if max_date is None else max(max_date, file_max)
    normalized_dtypes = {
        column: tuple(sorted(values))
        for column, values in sorted(dtype_summary.items())
    }
    normalized_content = {
        column: _normalize_content_stats(stats)
        for column, stats in sorted(content_summary.items())
    }
    normalized_key_value_pairs = {
        column: _normalize_content_stats(stats)
        for column, stats in sorted(key_value_paired_summary.items())
    }
    return AggregateSummary(
        row_count=row_count,
        key_count=key_count,
        column_count=len(columns),
        duplicate_key_count=duplicate_key_count,
        period=_format_period(min_date, max_date),
        columns=tuple(sorted(columns)),
        dtype_summary=normalized_dtypes,
        null_summary=dict(sorted(null_summary.items())),
        content_summary=normalized_content,
        key_value_paired_summary=normalized_key_value_pairs,
    )


def _combine_summaries(summaries: list[AggregateSummary]) -> AggregateSummary:
    row_count = sum(summary.row_count for summary in summaries)
    key_count = sum(summary.key_count for summary in summaries)
    duplicate_key_count = sum(summary.duplicate_key_count for summary in summaries)
    columns = tuple(
        sorted({column for summary in summaries for column in summary.columns})
    )
    dtype_summary: dict[str, set[str]] = {}
    null_summary: dict[str, int] = {}
    content_summary: dict[str, dict[str, Any]] = {}
    key_value_paired_summary: dict[str, dict[str, Any]] = {}
    periods = [_parse_period(summary.period) for summary in summaries]
    min_dates = [period[0] for period in periods if period[0] is not None]
    max_dates = [period[1] for period in periods if period[1] is not None]
    for summary in summaries:
        for column, dtypes in summary.dtype_summary.items():
            dtype_summary.setdefault(column, set()).update(dtypes)
        for column, null_count in summary.null_summary.items():
            null_summary[column] = null_summary.get(column, 0) + int(null_count)
        for column, stats in summary.content_summary.items():
            _merge_content_stats(content_summary, column, stats)
        for column, stats in summary.key_value_paired_summary.items():
            _merge_content_stats(key_value_paired_summary, column, stats)
    normalized_dtypes = {
        column: tuple(sorted(values))
        for column, values in sorted(dtype_summary.items())
    }
    normalized_content = {
        column: _normalize_content_stats(stats)
        for column, stats in sorted(content_summary.items())
    }
    normalized_key_value_pairs = {
        column: _normalize_content_stats(stats)
        for column, stats in sorted(key_value_paired_summary.items())
    }
    return AggregateSummary(
        row_count=row_count,
        key_count=key_count,
        column_count=len(columns),
        duplicate_key_count=duplicate_key_count,
        period=_format_period(
            min(min_dates) if min_dates else None,
            max(max_dates) if max_dates else None,
        ),
        columns=columns,
        dtype_summary=normalized_dtypes,
        null_summary=dict(sorted(null_summary.items())),
        content_summary=normalized_content,
        key_value_paired_summary=normalized_key_value_pairs,
    )


def _update_content_summary(
    content_summary: dict[str, dict[str, Any]],
    column: str,
    series: pd.Series,
) -> None:
    stats = content_summary.setdefault(
        column,
        {"hash_sum": 0, "hash_xor": 0, "non_null_count": 0},
    )
    for hashes in _series_hash_blocks(series):
        stats["hash_sum"] = int(
            (int(stats["hash_sum"]) + int(hashes.sum())) % HASH_MODULUS
        )
        stats["hash_xor"] = int(stats["hash_xor"]) ^ int(
            np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
        )
    stats["non_null_count"] = int(stats["non_null_count"]) + int(series.notna().sum())
    numeric_min_max = _finite_numeric_min_max(series)
    if numeric_min_max is None:
        return
    numeric_min, numeric_max = numeric_min_max
    stats["numeric_min"] = min(
        float(stats.get("numeric_min", numeric_min)),
        numeric_min,
    )
    stats["numeric_max"] = max(
        float(stats.get("numeric_max", numeric_max)),
        numeric_max,
    )


def _update_key_value_paired_summary(
    paired_summary: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
) -> None:
    data_columns = [
        str(column) for column in frame.columns if str(column) not in key_columns
    ]
    column_hashes = {
        column_name: np.uint64(
            pd.util.hash_pandas_object(
                pd.Series([column_name], dtype="string"),
                index=False,
            ).iloc[0]
        )
        for column_name in data_columns
    }
    for column in frame.columns:
        column_name = str(column)
        if column_name in key_columns:
            continue
        stats = paired_summary.setdefault(
            column_name,
            {"hash_sum": 0, "hash_xor": 0, "non_null_count": 0},
        )
        stats["non_null_count"] = int(stats["non_null_count"]) + int(
            frame[column].notna().sum()
        )
    for start, key_hashes in _key_hash_blocks(frame, key_columns):
        stop = start + len(key_hashes)
        for column_name in data_columns:
            stats = paired_summary[column_name]
            values = frame[column_name].iloc[start:stop]
            value_hashes = _value_hashes(values)
            hashes = _mix_key_value_hashes(
                key_hashes,
                column_hash=column_hashes[column_name],
                value_hashes=value_hashes,
            )
            stats["hash_sum"] = int(
                (int(stats["hash_sum"]) + int(hashes.sum())) % HASH_MODULUS
            )
            stats["hash_xor"] = int(stats["hash_xor"]) ^ int(
                np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
            )


def _series_hash_blocks(series: pd.Series) -> Iterator[np.ndarray]:
    block_rows = _hash_block_rows(value_width=1)
    for start in range(0, len(series), block_rows):
        block = series.iloc[start : start + block_rows].reset_index(drop=True)
        yield pd.util.hash_pandas_object(block, index=False).to_numpy(dtype="uint64")


def _key_hash_blocks(
    frame: pd.DataFrame,
    key_columns: list[str],
) -> Iterator[tuple[int, np.ndarray]]:
    block_rows = _hash_block_rows(value_width=max(1, len(key_columns)))
    for start in range(0, len(frame), block_rows):
        block = frame.iloc[start : start + block_rows].loc[:, key_columns]
        yield start, pd.util.hash_pandas_object(block, index=False).to_numpy(
            dtype="uint64"
        )


def _hash_block_rows(*, value_width: int) -> int:
    try:
        effective_bytes = active_runtime_resource_policy().effective_chunk_bytes
    except Exception:
        effective_bytes = 16 * 1024 * 1024
    bytes_per_row = max(1, value_width) * HASH_BLOCK_BYTES_PER_VALUE
    bytes_per_row *= HASH_BLOCK_INTERMEDIATE_WIDTH
    rows = max(1, effective_bytes // max(1, bytes_per_row))
    return max(HASH_BLOCK_MIN_ROWS, min(HASH_BLOCK_MAX_ROWS, int(rows)))


def _finite_numeric_min_max(series: pd.Series) -> tuple[float, float] | None:
    values: list[tuple[float, float]] = []
    block_rows = _hash_block_rows(value_width=1)
    for start in range(0, len(series), block_rows):
        block = series.iloc[start : start + block_rows]
        if pd.api.types.is_bool_dtype(block):
            numeric = block.to_numpy(dtype="float64", na_value=np.nan)
        elif pd.api.types.is_numeric_dtype(block):
            numeric = pd.to_numeric(block, errors="coerce").to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        elif pd.api.types.is_datetime64_any_dtype(block):
            datetimes = pd.to_datetime(block, errors="coerce")
            mask = datetimes.notna().to_numpy()
            if not bool(mask.any()):
                continue
            numeric = datetimes.astype("int64").to_numpy(dtype="float64")[mask]
        else:
            return None
        finite = np.isfinite(numeric)
        if bool(finite.any()):
            finite_values = numeric[finite]
            values.append((float(finite_values.min()), float(finite_values.max())))
    if not values:
        return None
    return min(item[0] for item in values), max(item[1] for item in values)


def _value_hashes(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.mask(numeric.eq(0), 0.0)
        return pd.util.hash_pandas_object(
            numeric.reset_index(drop=True),
            index=False,
        ).to_numpy(dtype="uint64")
    return pd.util.hash_pandas_object(
        series.reset_index(drop=True),
        index=False,
    ).to_numpy(dtype="uint64")


def _mix_key_value_hashes(
    key_hashes: np.ndarray,
    *,
    column_hash: np.uint64,
    value_hashes: np.ndarray,
) -> np.ndarray:
    with np.errstate(over="ignore"):
        mixed = key_hashes ^ (
            value_hashes
            + KEY_VALUE_HASH_MIXER
            + (key_hashes << 6)
            + (key_hashes >> 2)
        )
        result = mixed ^ (
            column_hash + KEY_VALUE_HASH_MIXER + (mixed << 6) + (mixed >> 2)
        )
    return result


def _merge_content_stats(
    content_summary: dict[str, dict[str, Any]],
    column: str,
    stats: dict[str, Any],
) -> None:
    target = content_summary.setdefault(
        column,
        {"hash_sum": 0, "hash_xor": 0, "non_null_count": 0},
    )
    target["hash_sum"] = int(
        (int(target["hash_sum"]) + int(stats["hash_sum"])) % HASH_MODULUS
    )
    target["hash_xor"] = int(target["hash_xor"]) ^ int(stats["hash_xor"])
    target["non_null_count"] = int(target["non_null_count"]) + int(
        stats["non_null_count"]
    )
    if "numeric_min" not in stats:
        return
    target["numeric_min"] = min(
        float(target.get("numeric_min", stats["numeric_min"])),
        float(stats["numeric_min"]),
    )
    target["numeric_max"] = max(
        float(target.get("numeric_max", stats["numeric_max"])),
        float(stats["numeric_max"]),
    )


def _normalize_content_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = {
        "hash_sum": int(stats["hash_sum"]),
        "hash_xor": int(stats["hash_xor"]),
        "non_null_count": int(stats["non_null_count"]),
    }
    if "numeric_min" in stats:
        result.update(
            {
                "numeric_min": round(float(stats["numeric_min"]), 10),
                "numeric_max": round(float(stats["numeric_max"]), 10),
            }
        )
    return result


def _available_key_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["market_id", "asset_id", "timestamp"]
    if all(column in frame.columns for column in preferred):
        return preferred
    if "timestamp" in frame.columns:
        secondary = [
            column
            for column in ("timestamp", "signal", "label", "group")
            if column in frame.columns
        ]
        return secondary if secondary else ["timestamp"]
    return []


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ("timestamp", "session_date"):
        if column in frame.columns:
            return column
    return None


def _format_period(
    min_date: pd.Timestamp | None,
    max_date: pd.Timestamp | None,
) -> str:
    if min_date is None or max_date is None:
        return "not_applicable"
    return f"{min_date.date()} through {max_date.date()}"


def _parse_period(period: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if " through " not in period:
        return None, None
    start, end = period.split(" through ", 1)
    return pd.Timestamp(start), pd.Timestamp(end)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "generate_private_verification_summary",
    "generate_public_safe_aggregate_evidence",
    "render_private_verification_summary_markdown",
    "render_public_safe_evidence_markdown",
]
