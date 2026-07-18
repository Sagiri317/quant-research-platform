"""Review-safe aggregate evidence generation from local outputs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.artifacts._output_bundle_validation import (
    validate_output_bundle,
)
from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.ids import (
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materialization_io,
    materialization_planning,
    source_ids,
)
from quant_research_platform.markets.implementations.a_share.research_scenarios import (
    default_evaluation_period,
    default_evaluation_run_id,
    default_prediction_oos_period,
    default_prediction_smoke_id,
)
from quant_research_platform.markets.implementations.a_share.validation import (
    strict_values,
)
from quant_research_platform.markets.session_windows import (
    SessionWindowPlanner,
    normalize_session_dates,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import (
    EXACT_KEY_SQLITE_PAGE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    dataset_file_identity,
)
from quant_research_platform.storage._dataset_reading import (
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.datasets import inspect_dataset

PROJECT_VERSION = "0.1.0"
OUTPUT_SCOPE = "REVIEW_SAFE"
ARTIFACT_ID = "review_safe_aggregate_evidence"
MARKET_ID = "a_share"
FULL_HISTORY_PERIOD = "2014-01-01 through 2025-12-31"
GENERATED_BY = (
    "quant_research_platform.markets.implementations.a_share.verification.aggregates"
)
HASH_MODULUS = 1 << 64
KEY_VALUE_HASH_MIXER = np.uint64(0x9E3779B185EBCA87)
# Performance-only batching cap. It is not a resource measurement or gate.
HASH_BLOCK_ROWS = 1_048_576
FULL_HISTORY_START_YEAR = 2014
FULL_HISTORY_END_YEAR = 2025
FULL_HISTORY_START_DATE = "2014-01-01"
FULL_HISTORY_END_DATE = "2025-12-31"
GENERATED_PANEL_MEMORY_EXPANSION_FACTOR = 8
# Local verification run source fixture; not an A-share market contract.
LOCAL_VERIFICATION_INTERIM_SOURCE = "csmar"

PROCESSED_ROOT_AREA_COVERAGE = {
    f"processed/a_share/panel/{market_panel_data_id()}": (
        f"full_historical_{market_panel_data_id()}_status"
    ),
    f"processed/a_share/factor/{PRICE_VOLUME_PROCESSED_DATA_ID}": (
        "factor_value_parity"
    ),
    "processed/a_share/label/forward_returns": "label_value_parity",
    "processed/a_share/annotation/default": "masks_universes_parity",
    "processed/a_share/annotation/as_of": "masks_universes_parity",
    "processed/a_share/annotation/execution": "masks_universes_parity",
    "processed/a_share/annotation/evaluation": "masks_universes_parity",
    f"processed/a_share/annotation/{market_context_data_id()}": (
        "supporting_layers_coverage_2014_2025"
    ),
    "research/a_share/price_volume_forward_returns": "research_dataset_parity",
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


@dataclass(frozen=True, slots=True)
class _FrozenOutputBundle:
    root: Path
    manifest_bytes: bytes
    declared_paths: tuple[Path, ...]
    member_identities: tuple[tuple[Path, DatasetFileIdentity], ...]

    @property
    def parquet_members(self) -> tuple[tuple[Path, DatasetFileIdentity], ...]:
        return tuple(
            (path, identity)
            for path, identity in self.member_identities
            if path.suffix.casefold() == ".parquet"
        )


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
        return canonical_json_hash(list(self.columns))

    @property
    def dtype_summary_hash(self) -> str:
        return canonical_json_hash(self.dtype_summary)

    @property
    def null_summary_hash(self) -> str:
        return canonical_json_hash(self.null_summary)

    @property
    def key_value_paired_summary_hash(self) -> str:
        return canonical_json_hash(self.key_value_paired_summary)

    @property
    def aggregate_checksum(self) -> str:
        return canonical_json_hash(
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


def build_review_safe_aggregate_evidence_payload(
    project_root: Path = Path("."),
    *,
    generated_at: datetime | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    """Build review-safe aggregate evidence from local outputs."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    policy = resource_policy or active_runtime_resource_policy()
    summary_cache: _SummaryCache = {}
    with runtime_resource_policy_scope(policy):
        return _build_review_safe_aggregate_evidence_payload_with_cache(
            root,
            generated_at=generated_at,
            summary_cache=summary_cache,
            resource_policy=policy,
        )


def build_local_verification_inventory_payload(
    project_root: Path = Path("."),
    *,
    evidence_payload: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> dict[str, Any]:
    """Build a review-safe inventory summary for local dataset roots."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    policy = resource_policy or active_runtime_resource_policy()
    if evidence_payload is None:
        evidence_payload = build_review_safe_aggregate_evidence_payload(
            root,
            generated_at=generated_at,
            resource_policy=policy,
        )
    with runtime_resource_policy_scope(policy):
        inventory = _dataset_inventory(
            root,
            summary_cache={},
            resource_policy=policy,
        )
    covered_roots = [
        item["logical_root"] for item in inventory if item["covered_by_evidence_area"]
    ]
    uncovered_roots = [
        item["logical_root"]
        for item in inventory
        if not item["covered_by_evidence_area"]
    ]
    allowed_unverified_roots: list[dict[str, str]] = []
    handoff_ready = (
        bool(evidence_payload.get("handoff_ready"))
        and not uncovered_roots
        and not allowed_unverified_roots
    )
    return {
        "schema_id": "qrp-a-share-verification-inventory",
        "project_version": PROJECT_VERSION,
        "verification_kind": "local_result_aggregate_inventory",
        "generated_at": generated_at.replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "output_scope": OUTPUT_SCOPE,
        "source_data_visibility": "local_only",
        "row_level_output_committed": False,
        "dataset_inventory": inventory,
        "covered_roots": covered_roots,
        "uncovered_roots": uncovered_roots,
        "allowed_unverified_roots": allowed_unverified_roots,
        "handoff_ready": handoff_ready,
    }


def _build_review_safe_aggregate_evidence_payload_with_cache(
    root: Path,
    *,
    generated_at: datetime,
    summary_cache: _SummaryCache,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, Any]:
    areas = [
        _area_record(
            area_input,
            summary_cache=summary_cache,
            resource_policy=resource_policy,
        )
        for area_input in _area_inputs(root)
    ]
    handoff_ready = not any(area["human_approval_required"] for area in areas)
    status = (
        "local aggregate verification completed"
        if handoff_ready
        else ("local aggregate verification completed with human approval required")
    )
    return {
        "schema_id": "qrp-a-share-aggregate-evidence",
        "project_version": PROJECT_VERSION,
        "output_scope": OUTPUT_SCOPE,
        "artifact_id": ARTIFACT_ID,
        "market_id": MARKET_ID,
        "status": status,
        "handoff_ready": handoff_ready,
        "generated_at": generated_at.replace(microsecond=0)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "verification_kind": "local_result_aggregate",
        "verification_scope": "aggregate_review_safe_parity",
        "evidence_generation_mode": "local_result_review_safe_aggregate",
        "source_data_visibility": "local_only",
        "row_level_output_committed": False,
        "verification_periods": sorted({area["period"] for area in areas}),
        "areas": areas,
    }


def _area_inputs(root: Path) -> tuple[AggregateInput, ...]:
    processed = root / "data" / "processed" / MARKET_ID
    outputs_root = root / "outputs"
    market_panel_root = processed / "panel" / market_panel_data_id()
    factor = processed / "factor" / PRICE_VOLUME_PROCESSED_DATA_ID
    label = processed / "label" / "forward_returns"
    universe = processed / "annotation" / "default"
    masks = (
        processed / "annotation" / "as_of",
        processed / "annotation" / "execution",
        processed / "annotation" / "evaluation",
    )
    context = processed / "annotation" / market_context_data_id()
    research = (
        root / "data" / "research" / MARKET_ID / PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )
    evaluation_bundles = _output_bundles(
        outputs_root,
        preferred_id=default_evaluation_run_id(),
        output_definition_id="evaluation_outputs",
    )
    prediction_bundles = _output_bundles(
        outputs_root,
        preferred_id=default_prediction_smoke_id(test_start="2014-07-01"),
        output_definition_id="prediction_outputs",
    )
    return (
        AggregateInput(
            "processed_panel_parity",
            (market_panel_root,),
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
            evaluation_bundles,
            default_evaluation_period(
                evaluation_start="2014-01-29",
                evaluation_end="2014-12-31",
            ),
            "evaluation",
            summary_kind="outputs",
        ),
        AggregateInput(
            "prediction_oos_smoke_parity",
            prediction_bundles,
            _prediction_oos_expected_period(),
            "prediction OOS smoke",
            summary_kind="outputs",
        ),
        AggregateInput(
            "supporting_layers_coverage_2014_2025",
            (context, factor, label, universe, *masks),
            FULL_HISTORY_PERIOD,
            "supporting layers coverage",
        ),
        AggregateInput(
            f"full_historical_{market_panel_data_id()}_status",
            (market_panel_root,),
            FULL_HISTORY_PERIOD,
            "full historical market panel",
        ),
    )


def _output_bundles(
    outputs_root: Path,
    *,
    preferred_id: str,
    output_definition_id: str,
) -> tuple[Path, ...]:
    preferred = outputs_root / preferred_id
    if preferred.exists():
        payload, _ = validate_output_bundle(
            preferred,
            expected_output_id=preferred.name,
        )
        if payload["output_definition_id"] != output_definition_id:
            raise ValueError("invalid_aggregate_output_manifest")
        return (preferred,)
    discovered = tuple(
        sorted(
            path
            for path in outputs_root.iterdir()
            if path.is_dir()
            and (path / "manifest.json").is_file()
            and _output_manifest_definition_id(path) == output_definition_id
            and path.name != preferred_id
        )
        if outputs_root.exists()
        else ()
    )
    return discovered if discovered else (preferred,)


def _output_manifest_definition_id(output_bundle: Path) -> str | None:
    try:
        payload, _ = validate_output_bundle(
            output_bundle,
            expected_output_id=output_bundle.name,
        )
    except ValueError:
        return None
    definition_id = payload["output_definition_id"]
    return definition_id if type(definition_id) is str else None


def _prediction_oos_expected_period() -> str:
    return default_prediction_oos_period(
        train_end="2014-06-30",
        test_start="2014-07-01",
        evaluation_end="2014-12-31",
    )


def _dataset_inventory(
    root: Path,
    *,
    summary_cache: _SummaryCache | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> list[dict[str, Any]]:
    policy = resource_policy
    inventory: list[dict[str, Any]] = []
    for artifact_root in _official_dataset_artifact_roots(root):
        logical_root = _logical_dataset_root(root, artifact_root)
        summary = _summarize_dataset_root(
            artifact_root,
            summary_cache=summary_cache,
            resource_policy=policy,
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
        raise FileNotFoundError("local_roots_missing")
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
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, Any]:
    policy = resource_policy
    summary = _summarize_area_input(
        area_input,
        summary_cache=summary_cache,
        resource_policy=policy,
    )
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
            "from local outputs, without row-level output."
        )
        if period_matches
        else (
            f"Aggregate verification for {area_input.explanation_name} "
            f"covered {summary.period}; expected {area_input.expected_period}. "
            "Human approval is required before review-ready status."
        )
    )
    return {
        "area": area_input.area,
        "output_scope": OUTPUT_SCOPE,
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
        "verification_kind": "local_result_aggregate",
        "verification_scope": "aggregate_review_safe_parity",
        "evidence_generation_mode": "local_result_review_safe_aggregate",
        "tolerance_owner": "project_contract_0.1",
        "downstream_impact": (
            "no downstream parity impact detected from aggregate fields"
            if not human_approval_required
            else "review-ready status requires human approval for the coverage gap"
        ),
        "generated_by": GENERATED_BY,
        "source_data_visibility": "local_only",
        "row_level_output_committed": False,
        "human_approval_required": human_approval_required,
    }


def _summarize_area_input(
    area_input: AggregateInput,
    *,
    summary_cache: _SummaryCache | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> AggregateSummary:
    policy = resource_policy
    if area_input.summary_kind == "generated_market_panel_from_interim":
        if len(area_input.roots) != 1:
            raise ValueError("expected_single_project_root")
        return _summarize_generated_full_market_panel(
            area_input.roots[0],
            resource_policy=policy,
        )
    if area_input.summary_kind == "outputs":
        return _summarize_output_bundles(
            area_input.roots,
            resource_policy=policy,
        )
    return _summarize_roots(
        area_input.roots,
        years=area_input.years,
        summary_cache=summary_cache,
        resource_policy=policy,
    )


def _summarize_generated_full_market_panel(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> AggregateSummary:
    retained_payload_bytes = [0]
    return _summarize_frames(
        _iter_generated_full_market_panel_chunks(
            project_root,
            resource_policy=resource_policy,
            _retained_payload_bytes=retained_payload_bytes,
        ),
        scratch_parent=project_root,
        resource_policy=resource_policy,
        _retained_payload_bytes=retained_payload_bytes,
    )


def _iter_generated_full_market_panel_chunks(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    _retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    yield from _iter_generated_full_market_panel_session_chunks(
        project_root,
        resource_policy=resource_policy,
        _retained_payload_bytes=_retained_payload_bytes,
    )


def _iter_generated_full_market_panel_session_chunks(
    project_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    _retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_catalog,
    )
    from quant_research_platform.markets.implementations.a_share.processed import panels

    interim_root = _local_interim_root(project_root)
    policy = resource_policy
    stock_master = materialization_io.read_materialized_dataset(
        interim_root / "stock_master",
        resource_policy=policy,
    )
    market_cap = materialization_io.read_materialized_dataset(
        interim_root / "market_cap",
        resource_policy=policy,
        retained_payload_bytes=pandas_frame_payload_bytes(stock_master),
    )
    trading_calendar = materialization_io.read_materialized_dataset(
        interim_root / "trading_calendar",
        resource_policy=policy,
        columns=(
            "calendar_date",
            "source_market_type",
            "is_trading_day",
            "timestamp",
        ),
        retained_payload_bytes=sum(
            pandas_frame_payload_bytes(frame) for frame in (stock_master, market_cap)
        ),
    )
    retained_inputs = (stock_master, market_cap, trading_calendar)
    materialization_planning.require_known_materialization_payload_bound(
        frames=retained_inputs,
        output=None,
        resource_policy=policy,
    )
    spec = processed_catalog.processed_catalog().get(market_panel_data_id())
    market_bars_root = interim_root / _local_market_bars_data_id()
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
    target_core_session_count = max(
        1,
        min(
            len(target_sessions),
            policy.payload_target_bytes // estimated_bytes_per_session,
        ),
    )
    windows = SessionWindowPlanner.from_sessions(
        sessions,
        target_core_session_count=target_core_session_count,
        lookahead_sessions=1,
    ).windows(
        time_window,
        reason="verification_summary_generated_panel",
    )
    market_bars_columns = tuple(inspect_dataset(market_bars_root).columns)
    previous_window_payload_bytes = 0
    previous_yielded_frame: pd.DataFrame | None = None
    for window in windows:
        market_bars = materialization_io.read_materialized_dataset(
            market_bars_root,
            time_window=window.read,
            columns=market_bars_columns,
            resource_policy=policy,
            retained_payload_bytes=sum(
                pandas_frame_payload_bytes(frame) for frame in retained_inputs
            )
            + previous_window_payload_bytes,
        )
        if market_bars.empty:
            del market_bars
            continue
        panel = panels.compute_market_panel(
            market_bars,
            stock_master=stock_master,
            market_cap=market_cap,
            trading_calendar=trading_calendar,
        )
        output = panel[list(spec.required_columns)].copy()
        selected = materialization_planning.filter_output_sessions(
            output,
            window.core_sessions,
        ).reset_index(drop=True)
        materialization_planning.require_known_materialization_payload_bound(
            frames=(
                *retained_inputs,
                market_bars,
                panel,
                output,
                *(
                    (previous_yielded_frame,)
                    if previous_yielded_frame is not None
                    else ()
                ),
            ),
            output=selected,
            resource_policy=policy,
        )
        previous_yielded_frame = selected
        if _retained_payload_bytes is not None:
            _retained_payload_bytes[0] = sum(
                pandas_frame_payload_bytes(frame)
                for frame in (*retained_inputs, market_bars, panel, output)
            )
        previous_window_payload_bytes = pandas_frame_payload_bytes(selected)
        yield selected
        del market_bars, panel, output, selected


def _local_interim_root(
    project_root: Path,
) -> Path:
    return Path(project_root) / "data" / "interim" / LOCAL_VERIFICATION_INTERIM_SOURCE


def _estimate_generated_full_panel_memory_bytes(
    interim_root: Path,
    *,
    session_count: int,
) -> int:
    compressed_bytes = int(
        inspect_dataset(interim_root / _local_market_bars_data_id()).manifest.get(
            "file_size_bytes",
            0,
        )
    )
    return max(
        session_count,
        compressed_bytes * GENERATED_PANEL_MEMORY_EXPANSION_FACTOR,
    )


def _local_market_bars_data_id() -> str:
    return source_ids.MARKET_BARS_SOURCE_DATA_ID


def _trading_sessions_from_calendar(
    trading_calendar: pd.DataFrame,
) -> tuple[pd.Timestamp, ...]:
    if "timestamp" not in trading_calendar or "is_trading_day" not in trading_calendar:
        raise ValueError("trading_calendar_columns_missing")
    mask = strict_values.strict_trading_bool_series(
        trading_calendar["is_trading_day"],
        error="invalid_is_trading_day",
    )
    return normalize_session_dates(trading_calendar.loc[mask, "timestamp"])


def _time_window_from_years(years: tuple[int, ...] | None) -> TimeWindow | None:
    if years is None:
        return None
    return TimeWindow(f"{min(years)}-01-01", f"{max(years)}-12-31")


def _summarize_roots(
    roots: tuple[Path, ...],
    *,
    years: tuple[int, ...] | None = None,
    summary_cache: _SummaryCache | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> AggregateSummary:
    policy = resource_policy
    summaries = [
        _summarize_dataset_root(
            root,
            years=years,
            summary_cache=summary_cache,
            resource_policy=policy,
        )
        for root in roots
    ]
    return _combine_summaries(summaries)


def _summarize_output_bundles(
    roots: tuple[Path, ...],
    *,
    resource_policy: RuntimeResourcePolicy,
) -> AggregateSummary:
    if not roots:
        raise ValueError("aggregate_output_roots_required")
    scratch_parent = Path(roots[0]).parent
    if any(
        Path(root).resolve().anchor != scratch_parent.resolve().anchor for root in roots
    ):
        raise ValueError("aggregate_output_roots_cross_volume")
    policy = resource_policy
    operation_headroom_bytes = (
        policy.task_memory_budget_bytes - EXACT_KEY_SQLITE_PAGE_BYTES
    )
    if operation_headroom_bytes <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    retained_payload_bytes = [0]
    return _summarize_frames(
        _iter_output_frames(
            roots,
            target_chunk_bytes=policy.payload_target_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            _retained_payload_bytes=retained_payload_bytes,
        ),
        scratch_parent=scratch_parent,
        sqlite_cache_bytes=EXACT_KEY_SQLITE_PAGE_BYTES,
        resource_policy=policy,
        _retained_payload_bytes=retained_payload_bytes,
    )


def _iter_output_frames(
    roots: tuple[Path, ...],
    *,
    target_chunk_bytes: int,
    operation_headroom_bytes: int,
    _retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    previous_output_bytes = [0]

    def observe_batch(batch, _frame: pd.DataFrame) -> None:  # noqa: ANN001
        assert _retained_payload_bytes is not None
        _retained_payload_bytes[0] = batch.get_total_buffer_size()

    for root in roots:
        frozen = _freeze_output_bundle(root)
        try:
            for file, identity in frozen.parquet_members:
                yield from iter_declared_parquet_member_batches(
                    file,
                    target_chunk_bytes=target_chunk_bytes,
                    operation_headroom_bytes=operation_headroom_bytes,
                    expected_identity=identity,
                    batch_observer=(
                        observe_batch if _retained_payload_bytes is not None else None
                    ),
                    _previous_output_bytes=previous_output_bytes,
                )
        finally:
            _verify_frozen_output_bundle(frozen)


def _freeze_output_bundle(root: Path) -> _FrozenOutputBundle:
    bundle_root = Path(root)
    _, declared = validate_output_bundle(
        bundle_root,
        expected_output_id=bundle_root.name,
    )
    members = (bundle_root / "manifest.json", *declared)
    frozen = _FrozenOutputBundle(
        root=bundle_root,
        manifest_bytes=(bundle_root / "manifest.json").read_bytes(),
        declared_paths=declared,
        member_identities=tuple(
            (path, dataset_file_identity(path)) for path in members
        ),
    )
    if not frozen.parquet_members:
        raise ValueError("invalid_aggregate_output_files")
    return frozen


def _verify_frozen_output_bundle(frozen: _FrozenOutputBundle) -> None:
    try:
        _, declared = validate_output_bundle(
            frozen.root,
            expected_output_id=frozen.root.name,
        )
        if declared != frozen.declared_paths:
            raise ValueError("aggregate_output_bundle_changed")
        if (frozen.root / "manifest.json").read_bytes() != frozen.manifest_bytes:
            raise ValueError("aggregate_output_bundle_changed")
        for path, identity in frozen.member_identities:
            if dataset_file_identity(path) != identity:
                raise ValueError("aggregate_output_bundle_changed")
    except OSError as exc:
        raise ValueError("aggregate_output_bundle_changed") from exc


def _summarize_dataset_root(
    root: Path,
    *,
    years: tuple[int, ...] | None = None,
    summary_cache: _SummaryCache | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> AggregateSummary:
    if summary_cache is not None:
        key = _summary_cache_key(root, years)
        cached = summary_cache.get(key)
        if cached is not None:
            return cached
    policy = resource_policy
    retained_payload_bytes = [0]
    chunks = materialization_io.iter_dataset_chunks_with_policy(
        root,
        time_window=_time_window_from_years(years),
        resource_policy=policy,
        target_chunk_bytes=policy.payload_target_bytes,
        _retained_payload_bytes=retained_payload_bytes,
    )
    summary = _summarize_frames(
        chunks,
        scratch_parent=root.parent,
        resource_policy=policy,
        _retained_payload_bytes=retained_payload_bytes,
    )
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


def _summarize_frames(
    frames: Iterable[pd.DataFrame],
    *,
    scratch_parent: Path,
    resource_policy: RuntimeResourcePolicy,
    sqlite_cache_bytes: int | None = None,
    _retained_payload_bytes: list[int] | None = None,
) -> AggregateSummary:
    retained_payload_bytes = (
        [0] if _retained_payload_bytes is None else _retained_payload_bytes
    )
    exact_key_cache_bytes = (
        EXACT_KEY_SQLITE_PAGE_BYTES
        if sqlite_cache_bytes is None
        else sqlite_cache_bytes
    )
    row_count = 0
    keyed_row_count = 0
    unkeyed_row_count = 0
    duplicate_key_count = 0
    columns: set[str] = set()
    dtype_summary: dict[str, set[str]] = {}
    null_summary: dict[str, int] = {}
    content_summary: dict[str, dict[str, Any]] = {}
    key_value_paired_summary: dict[str, dict[str, Any]] = {}
    min_date: pd.Timestamp | None = None
    max_date: pd.Timestamp | None = None
    exact_key_columns: tuple[str, ...] | None = None
    with ExitStack() as stack:
        frame_iterator = iter(frames)
        close_frames = getattr(frame_iterator, "close", None)
        if callable(close_frames):
            stack.callback(close_frames)
        key_index: ExactPrimaryKeyIndex | None = None
        for frame in frame_iterator:
            key_columns = tuple(_available_key_columns(frame))
            reducer_headroom_bytes = (
                resource_policy.task_memory_budget_bytes
                - retained_payload_bytes[0]
                - pandas_frame_payload_bytes(frame)
                - exact_key_cache_bytes
            )
            if reducer_headroom_bytes <= 0:
                raise ValueError("aggregate_working_set_exceeds_task_memory_budget")
            if key_columns:
                if exact_key_columns is not None and key_columns != exact_key_columns:
                    raise ValueError("aggregate_key_columns_mismatch")
            if frame.empty:
                columns.update(str(column) for column in frame.columns)
                for column in frame.columns:
                    column_name = str(column)
                    dtype_summary.setdefault(column_name, set()).add(
                        str(frame[column].dtype)
                    )
                if key_columns and exact_key_columns is None:
                    exact_key_columns = key_columns
                continue
            rows_per_block = 1
            previous_block_rows = 0
            previous_block_working_bytes = 0
            frame_metadata_recorded = False
            start = 0
            while start < len(frame):
                while True:
                    stop = min(len(frame), start + rows_per_block)
                    block = frame.iloc[start:stop]
                    candidate_payload_bytes = pandas_frame_payload_bytes(block)
                    candidate_working_bytes = candidate_payload_bytes
                    if previous_block_rows:
                        candidate_working_bytes += (
                            previous_block_working_bytes * len(block)
                            + previous_block_rows
                            - 1
                        ) // previous_block_rows
                    if candidate_working_bytes <= reducer_headroom_bytes:
                        break
                    if len(block) == 1:
                        raise ValueError(
                            "aggregate_working_set_exceeds_task_memory_budget"
                        )
                    rows_per_block = max(
                        1,
                        min(
                            len(block) - 1,
                            len(block)
                            * reducer_headroom_bytes
                            // candidate_working_bytes,
                        ),
                    )
                    del block
                block_null_summary: dict[str, int] = {}
                block_content_summary: dict[str, dict[str, Any]] = {}
                block_paired_summary: dict[str, dict[str, Any]] = {}
                block_working_bytes = 0
                for column in block.columns:
                    column_name = str(column)
                    block_null_summary[column_name] = block_null_summary.get(
                        column_name,
                        0,
                    ) + int(block[column].isna().sum())
                    block_working_bytes = max(
                        block_working_bytes,
                        _update_content_summary(
                            block_content_summary,
                            column_name,
                            block[column],
                        ),
                    )
                key_frame: pd.DataFrame | None = None
                if key_columns:
                    block_working_bytes = max(
                        block_working_bytes,
                        _update_key_value_paired_summary(
                            block_paired_summary,
                            block,
                            key_columns=list(key_columns),
                        ),
                    )
                    key_frame = block.loc[:, list(key_columns)]
                    key_bytes = pandas_frame_payload_bytes(key_frame)
                    block_working_bytes = max(block_working_bytes, key_bytes)
                date_column = _date_column(block)
                block_min_date: pd.Timestamp | None = None
                block_max_date: pd.Timestamp | None = None
                if date_column is not None:
                    dates = pd.to_datetime(
                        block[date_column],
                        errors="coerce",
                    ).dropna()
                    block_working_bytes = max(
                        block_working_bytes,
                        pandas_frame_payload_bytes(dates.to_frame()),
                    )
                    if not dates.empty:
                        block_min_date = dates.min()
                        block_max_date = dates.max()
                    del dates
                block_total_working_bytes = (
                    candidate_payload_bytes + block_working_bytes
                )
                if block_total_working_bytes > reducer_headroom_bytes:
                    if len(block) == 1:
                        raise ValueError(
                            "aggregate_working_set_exceeds_task_memory_budget"
                        )
                    rows_per_block = max(
                        1,
                        min(
                            len(block) - 1,
                            len(block)
                            * reducer_headroom_bytes
                            // block_total_working_bytes,
                        ),
                    )
                    del block, block_null_summary, block_content_summary
                    del block_paired_summary, key_frame
                    continue
                if key_columns:
                    if exact_key_columns is None:
                        exact_key_columns = key_columns
                    if key_index is None:
                        key_index = stack.enter_context(
                            ExactPrimaryKeyIndex(
                                primary_key=key_columns,
                                timestamp_columns=tuple(
                                    column
                                    for column in key_columns
                                    if column == "timestamp"
                                ),
                                scratch_parent=scratch_parent,
                                sqlite_cache_bytes=exact_key_cache_bytes,
                            )
                        )
                    assert key_frame is not None
                    require_free_disk_headroom(
                        scratch_parent,
                        reserve_bytes=key_bytes,
                    )
                    duplicate_key_count += key_index.add_frame(
                        key_frame,
                        namespace="aggregate",
                    )
                    keyed_row_count += int(len(block))
                else:
                    unkeyed_row_count += int(len(block))
                if not frame_metadata_recorded:
                    columns.update(str(column) for column in frame.columns)
                    for column in frame.columns:
                        column_name = str(column)
                        dtype_summary.setdefault(column_name, set()).add(
                            str(frame[column].dtype)
                        )
                    frame_metadata_recorded = True
                row_count += int(len(block))
                for column_name, block_null_count in block_null_summary.items():
                    null_summary[column_name] = (
                        null_summary.get(column_name, 0) + block_null_count
                    )
                for column_name, stats in block_content_summary.items():
                    _merge_content_stats(content_summary, column_name, stats)
                for column_name, stats in block_paired_summary.items():
                    _merge_content_stats(
                        key_value_paired_summary,
                        column_name,
                        stats,
                    )
                if block_min_date is not None:
                    min_date = (
                        block_min_date
                        if min_date is None
                        else min(min_date, block_min_date)
                    )
                    assert block_max_date is not None
                    max_date = (
                        block_max_date
                        if max_date is None
                        else max(max_date, block_max_date)
                    )
                start = stop
                previous_block_rows = len(block)
                previous_block_working_bytes = max(
                    1,
                    block_working_bytes,
                )
                if start < len(frame):
                    rows_per_block = max(
                        1,
                        min(
                            HASH_BLOCK_ROWS,
                            len(frame) - start,
                            len(block)
                            * reducer_headroom_bytes
                            // block_total_working_bytes,
                        ),
                    )
                del block, block_null_summary, block_content_summary
                del block_paired_summary, key_frame
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
        key_count=keyed_row_count - duplicate_key_count + unkeyed_row_count,
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
) -> int:
    stats = content_summary.setdefault(
        column,
        {"hash_sum": 0, "hash_xor": 0, "non_null_count": 0},
    )
    working_bytes = 0
    for hashes in _series_hash_blocks(series):
        working_bytes = max(working_bytes, int(hashes.nbytes))
        stats["hash_sum"] = int(
            (int(stats["hash_sum"]) + int(hashes.sum())) % HASH_MODULUS
        )
        stats["hash_xor"] = int(stats["hash_xor"]) ^ int(
            np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
        )
        del hashes
    stats["non_null_count"] = int(stats["non_null_count"]) + int(series.notna().sum())
    numeric_working_bytes = [0]
    numeric_min_max = _finite_numeric_min_max(
        series,
        _working_bytes=numeric_working_bytes,
    )
    working_bytes = max(working_bytes, numeric_working_bytes[0])
    if numeric_min_max is None:
        return working_bytes
    numeric_min, numeric_max = numeric_min_max
    stats["numeric_min"] = min(
        float(stats.get("numeric_min", numeric_min)),
        numeric_min,
    )
    stats["numeric_max"] = max(
        float(stats.get("numeric_max", numeric_max)),
        numeric_max,
    )
    return working_bytes


def _update_key_value_paired_summary(
    paired_summary: dict[str, dict[str, Any]],
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
) -> int:
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
    working_bytes = 0
    for start, key_hashes in _key_hash_blocks(frame, key_columns):
        stop = start + len(key_hashes)
        for column_name in data_columns:
            stats = paired_summary[column_name]
            values = frame[column_name].iloc[start:stop]
            value_working_bytes = [0]
            value_hashes = _value_hashes(
                values,
                _working_bytes=value_working_bytes,
            )
            mix_working_bytes = [0]
            hashes = _mix_key_value_hashes(
                key_hashes,
                column_hash=column_hashes[column_name],
                value_hashes=value_hashes,
                _working_bytes=mix_working_bytes,
            )
            working_bytes = max(
                working_bytes,
                int(key_hashes.nbytes) + value_working_bytes[0],
                int(key_hashes.nbytes)
                + int(value_hashes.nbytes)
                + mix_working_bytes[0],
            )
            stats["hash_sum"] = int(
                (int(stats["hash_sum"]) + int(hashes.sum())) % HASH_MODULUS
            )
            stats["hash_xor"] = int(stats["hash_xor"]) ^ int(
                np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
            )
            del values, value_hashes, hashes
        del key_hashes
    return working_bytes


def _series_hash_blocks(
    series: pd.Series,
) -> Iterator[np.ndarray]:
    for start in range(0, len(series), HASH_BLOCK_ROWS):
        block = series.iloc[start : start + HASH_BLOCK_ROWS].reset_index(drop=True)
        yield pd.util.hash_pandas_object(block, index=False).to_numpy(dtype="uint64")


def _key_hash_blocks(
    frame: pd.DataFrame,
    key_columns: list[str],
) -> Iterator[tuple[int, np.ndarray]]:
    for start in range(0, len(frame), HASH_BLOCK_ROWS):
        block = frame.iloc[start : start + HASH_BLOCK_ROWS].loc[:, key_columns]
        yield (
            start,
            pd.util.hash_pandas_object(block, index=False).to_numpy(dtype="uint64"),
        )


def _finite_numeric_min_max(
    series: pd.Series,
    *,
    _working_bytes: list[int] | None = None,
) -> tuple[float, float] | None:
    working_bytes = [0] if _working_bytes is None else _working_bytes
    values: list[tuple[float, float]] = []
    for start in range(0, len(series), HASH_BLOCK_ROWS):
        block = series.iloc[start : start + HASH_BLOCK_ROWS]
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
                del datetimes, mask
                continue
            numeric = datetimes.astype("int64").to_numpy(dtype="float64")[mask]
            working_bytes[0] = max(
                working_bytes[0],
                pandas_frame_payload_bytes(datetimes.to_frame())
                + int(mask.nbytes)
                + int(numeric.nbytes),
            )
            del datetimes, mask
        else:
            return None
        finite = np.isfinite(numeric)
        if bool(finite.any()):
            finite_values = numeric[finite]
            working_bytes[0] = max(
                working_bytes[0],
                int(numeric.nbytes) + int(finite.nbytes) + int(finite_values.nbytes),
            )
            values.append((float(finite_values.min()), float(finite_values.max())))
            del finite_values
        else:
            working_bytes[0] = max(
                working_bytes[0],
                int(numeric.nbytes) + int(finite.nbytes),
            )
        del numeric, finite
    if not values:
        return None
    return min(item[0] for item in values), max(item[1] for item in values)


def _value_hashes(
    series: pd.Series,
    *,
    _working_bytes: list[int] | None = None,
) -> np.ndarray:
    working_bytes = [0] if _working_bytes is None else _working_bytes
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        numeric = numeric.mask(numeric.eq(0), 0.0)
        hashes = pd.util.hash_pandas_object(
            numeric.reset_index(drop=True),
            index=False,
        ).to_numpy(dtype="uint64")
        working_bytes[0] = max(
            working_bytes[0],
            pandas_frame_payload_bytes(numeric.to_frame()) + int(hashes.nbytes),
        )
        return hashes
    hashes = pd.util.hash_pandas_object(
        series.reset_index(drop=True),
        index=False,
    ).to_numpy(dtype="uint64")
    working_bytes[0] = max(working_bytes[0], int(hashes.nbytes))
    return hashes


def _mix_key_value_hashes(
    key_hashes: np.ndarray,
    *,
    column_hash: np.uint64,
    value_hashes: np.ndarray,
    _working_bytes: list[int] | None = None,
) -> np.ndarray:
    working_bytes = [0] if _working_bytes is None else _working_bytes
    with np.errstate(over="ignore"):
        mixed = key_hashes ^ (
            value_hashes + KEY_VALUE_HASH_MIXER + (key_hashes << 6) + (key_hashes >> 2)
        )
        result = mixed ^ (
            column_hash + KEY_VALUE_HASH_MIXER + (mixed << 6) + (mixed >> 2)
        )
    working_bytes[0] = max(
        working_bytes[0],
        int(mixed.nbytes) + int(result.nbytes),
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


__all__ = [
    "build_local_verification_inventory_payload",
    "build_review_safe_aggregate_evidence_payload",
]
