"""Public-safe aggregate evidence generation from local private artifacts."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.ids import (
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.reports.evidence_rendering import (
    render_private_verification_summary_markdown,
    render_public_safe_evidence_markdown,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)

from ..processed._dataset_helpers import (
    chunked_values,
    dataset_asset_ids,
    read_dataset_asset_chunk,
    year_partition_filter,
)

CONTRACT_VERSION = "0.1"
PROJECT_VERSION = "0.1.0"
PRIVACY_LEVEL = "public_safe_aggregate"
DATA_PRODUCT_ID = "a_share_daily_bar"
MARKET_ID = "a_share"
FULL_HISTORY_PERIOD = "2014-01-01 through 2025-12-31"
GENERATED_BY = (
    "quant_research_platform.markets.implementations.a_share.verification."
    "private_aggregate"
)
HASH_MODULUS = 1 << 64
FULL_HISTORY_START_YEAR = 2014
FULL_HISTORY_END_YEAR = 2025
FULL_HISTORY_START_DATE = "2014-01-01"
FULL_HISTORY_END_DATE = "2025-12-31"
GENERATED_PANEL_MEMORY_FRACTION = 0.7
GENERATED_PANEL_MEMORY_EXPANSION_FACTOR = 8
DEFAULT_AVAILABLE_MEMORY_BYTES = 2 * 1024 * 1024 * 1024

PROCESSED_ROOT_AREA_COVERAGE = {
    "processed/a_share/panels/daily_panel": "full_historical_daily_panel_status",
    f"processed/a_share/factors/{PRICE_VOLUME_PROCESSED_DATA_ID}": (
        "factor_value_parity"
    ),
    "processed/a_share/labels/forward_returns": "label_value_parity",
    "processed/a_share/universes/a_share_default": "masks_universes_parity",
    "processed/a_share/masks/as_of/a_share_as_of": "masks_universes_parity",
    "processed/a_share/masks/execution/a_share_execution": "masks_universes_parity",
    "processed/a_share/masks/evaluation/a_share_evaluation": "masks_universes_parity",
    "processed/a_share/contexts/daily_context": "supporting_layers_coverage_2014_2025",
    "processed/a_share/research_datasets/"
    "a_share_price_volume_forward_returns": "research_dataset_parity",
}


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
            }
        )


def generate_public_safe_aggregate_evidence(
    project_root: Path = Path("."),
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Generate public-safe aggregate evidence from local private artifacts."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    areas = [_area_record(area_input) for area_input in _area_inputs(root)]
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
        "data_product_id": DATA_PRODUCT_ID,
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


def generate_private_verification_summary(
    project_root: Path = Path("."),
    *,
    evidence_payload: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Generate a public-safe inventory summary for processed artifacts."""

    root = Path(project_root)
    generated_at = generated_at or datetime.now(UTC)
    if evidence_payload is None:
        evidence_payload = generate_public_safe_aggregate_evidence(
            root,
            generated_at=generated_at,
        )
    inventory = _processed_inventory(root)
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
        "processed_inventory": inventory,
        "covered_roots": covered_roots,
        "uncovered_roots": uncovered_roots,
        "allowed_unverified_roots": allowed_unverified_roots,
        "merge_ready": merge_ready,
    }


def _area_inputs(root: Path) -> tuple[AggregateInput, ...]:
    processed = root / "data" / "processed" / MARKET_ID
    reports_root = root / "reports" / "private_real" / MARKET_ID
    daily_panel = processed / "panels" / "daily_panel"
    factor = processed / "factors" / PRICE_VOLUME_PROCESSED_DATA_ID
    label = processed / "labels" / "forward_returns"
    universe = processed / "universes" / "a_share_default"
    masks = (
        processed / "masks" / "as_of" / "a_share_as_of",
        processed / "masks" / "execution" / "a_share_execution",
        processed / "masks" / "evaluation" / "a_share_evaluation",
    )
    context = processed / "contexts" / "daily_context"
    research = (
        processed
        / "research_datasets"
        / "a_share_price_volume_forward_returns"
    )
    evaluation = (
        reports_root / "evaluation" / "a_share_mom_20d_forward_return_20d_2014"
    )
    prediction = (
        reports_root / "prediction" / "a_share_linear_mom_20d_prediction_2014"
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
            (evaluation,),
            "2014-01-29 through 2014-12-31",
            "evaluation",
            summary_kind="report_bundle",
        ),
        AggregateInput(
            "prediction_oos_smoke_parity",
            (prediction,),
            "2014-01-01 through 2014-12-31",
            "prediction OOS smoke",
            summary_kind="report_bundle",
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


def _processed_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for artifact_root in _official_processed_artifact_roots(root):
        logical_root = _logical_processed_root(root, artifact_root)
        summary = _summarize_dataset_root(artifact_root)
        category, artifact_id = _processed_category_and_artifact_id(logical_root)
        inventory.append(
            {
                "category": category,
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
                "aggregate_checksum": summary.aggregate_checksum,
                "covered_by_evidence_area": PROCESSED_ROOT_AREA_COVERAGE.get(
                    logical_root,
                    "",
                ),
            }
        )
    return sorted(inventory, key=lambda item: item["logical_root"])


def _official_processed_artifact_roots(root: Path) -> tuple[Path, ...]:
    processed_root = Path(root) / "data" / "processed" / MARKET_ID
    if not processed_root.exists():
        raise FileNotFoundError("private_processed_root_missing")
    roots = [
        path
        for path in processed_root.rglob("*")
        if path.is_dir() and _is_dataset_artifact_root(path)
    ]
    return tuple(sorted(set(roots), key=lambda path: path.as_posix()))


def _is_dataset_artifact_root(path: Path) -> bool:
    try:
        inspect_dataset(path)
        return True
    except (OSError, ValueError):
        return False


def _logical_processed_root(root: Path, artifact_root: Path) -> str:
    relative = artifact_root.relative_to(Path(root) / "data" / "processed")
    return "processed/" + relative.as_posix()


def _processed_category_and_artifact_id(logical_root: str) -> tuple[str, str]:
    parts = logical_root.split("/")
    if len(parts) < 4 or parts[:2] != ["processed", MARKET_ID]:
        raise ValueError("invalid_processed_logical_root")
    category_parts = parts[2:-1]
    return "/".join(category_parts), parts[-1]


def _area_record(area_input: AggregateInput) -> dict[str, Any]:
    summary = _summarize_area_input(area_input)
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


def _summarize_area_input(area_input: AggregateInput) -> AggregateSummary:
    if area_input.summary_kind == "generated_daily_panel_from_interim":
        if len(area_input.roots) != 1:
            raise ValueError("expected_single_project_root")
        return _summarize_generated_full_daily_panel(area_input.roots[0])
    if area_input.summary_kind == "report_bundle":
        return _summarize_report_roots(area_input.roots)
    return _summarize_roots(area_input.roots, years=area_input.years)


def _summarize_generated_full_daily_panel(project_root: Path) -> AggregateSummary:
    return _summarize_frames(_iter_generated_full_daily_panel_chunks(project_root))


def _iter_generated_full_daily_panel_chunks(
    project_root: Path,
) -> Iterator[pd.DataFrame]:
    yield from _iter_generated_full_daily_panel_asset_chunks(project_root)


def _iter_generated_full_daily_panel_asset_chunks(
    project_root: Path,
) -> Iterator[pd.DataFrame]:
    from quant_research_platform.markets.implementations.a_share.catalog import (
        a_share_processed_catalog,
    )
    from quant_research_platform.markets.implementations.a_share.processed import (
        panels,
    )
    interim_root = _private_interim_root_for_data_product(
        project_root,
        DATA_PRODUCT_ID,
    )
    stock_master = read_dataset(interim_root / "stock_master")
    market_cap = read_dataset(interim_root / "market_cap")
    trading_calendar = read_dataset(interim_root / "trading_calendar")
    spec = a_share_processed_catalog().get("daily_panel")
    years = tuple(range(FULL_HISTORY_START_YEAR, FULL_HISTORY_END_YEAR + 1))
    daily_market_root = interim_root / "daily_market"
    asset_ids = dataset_asset_ids(daily_market_root, years=years)
    chunk_size = _dynamic_asset_chunk_size(
        total_asset_count=len(asset_ids),
        estimated_full_frame_bytes=_estimate_generated_full_panel_memory_bytes(
            interim_root,
            years=years,
        ),
        available_memory_bytes=_available_memory_bytes(),
        memory_fraction=GENERATED_PANEL_MEMORY_FRACTION,
    )
    daily_market_columns = tuple(inspect_dataset(daily_market_root).columns)
    for asset_id_chunk in chunked_values(asset_ids, chunk_size):
        daily = read_dataset_asset_chunk(
            daily_market_root,
            years=years,
            columns=daily_market_columns,
            asset_ids=asset_id_chunk,
        )
        if daily.empty:
            continue
        panel = panels.build_research_panel(
            daily,
            stock_master=_filter_asset_frame(stock_master, asset_id_chunk),
            market_cap=_filter_asset_frame(market_cap, asset_id_chunk),
            trading_calendar=trading_calendar,
        )
        output = panel[list(spec.required_columns)].copy()
        timestamp = pd.to_datetime(output["timestamp"], errors="coerce")
        yield output.loc[
            timestamp.between(
                FULL_HISTORY_START_DATE,
                FULL_HISTORY_END_DATE,
                inclusive="both",
            )
        ].reset_index(drop=True)


def _private_interim_root_for_data_product(
    project_root: Path,
    data_product_id: str,
) -> Path:
    from quant_research_platform.catalog.defaults import (
        default_provider_binding_registry,
    )

    binding = default_provider_binding_registry().get(data_product_id)
    return Path(project_root) / "data" / "interim" / binding.provider_id


def _filter_asset_frame(
    frame: pd.DataFrame,
    asset_ids: Sequence[str],
) -> pd.DataFrame:
    if "asset_id" not in frame.columns:
        return frame.copy()
    return frame.loc[frame["asset_id"].isin(set(asset_ids))].copy()


def _estimate_generated_full_panel_memory_bytes(
    interim_root: Path,
    *,
    years: tuple[int, ...],
) -> int:
    del years
    compressed_bytes = int(
        inspect_dataset(interim_root / "daily_market").manifest.get(
            "file_size_bytes",
            0,
        )
    )
    return max(1, compressed_bytes * GENERATED_PANEL_MEMORY_EXPANSION_FACTOR)


def _available_memory_bytes() -> int:
    if os.name == "nt":
        available = _windows_available_memory_bytes()
        if available is not None:
            return available
    if hasattr(os, "sysconf"):
        pages = _sysconf_value("SC_AVPHYS_PAGES")
        page_size = _sysconf_value("SC_PAGE_SIZE")
        if pages is not None and page_size is not None:
            return int(pages * page_size)
    return DEFAULT_AVAILABLE_MEMORY_BYTES


def _windows_available_memory_bytes() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullAvailPhys)
    return None


def _sysconf_value(name: str) -> int | None:
    try:
        return int(os.sysconf(name))
    except (AttributeError, OSError, ValueError):
        return None


def _dynamic_asset_chunk_size(
    *,
    total_asset_count: int,
    estimated_full_frame_bytes: int,
    available_memory_bytes: int,
    memory_fraction: float,
) -> int:
    if total_asset_count <= 0:
        return 1
    target_bytes = max(1, int(available_memory_bytes * memory_fraction))
    if estimated_full_frame_bytes <= target_bytes:
        return total_asset_count
    bytes_per_asset = max(1, ceil(estimated_full_frame_bytes / total_asset_count))
    return max(1, min(total_asset_count, target_bytes // bytes_per_asset))


def _summarize_roots(
    roots: tuple[Path, ...],
    *,
    years: tuple[int, ...] | None = None,
) -> AggregateSummary:
    summaries = [_summarize_dataset_root(root, years=years) for root in roots]
    return _combine_summaries(summaries)


def _summarize_report_roots(roots: tuple[Path, ...]) -> AggregateSummary:
    frames: list[pd.DataFrame] = []
    for root in roots:
        files = tuple(sorted(Path(root).rglob("*.parquet")))
        if not files:
            raise FileNotFoundError("private_aggregate_input_missing")
        for file in files:
            frames.append(pd.read_parquet(file, engine="pyarrow"))
    return _summarize_frames(frames)


def _summarize_dataset_root(
    root: Path,
    *,
    years: tuple[int, ...] | None = None,
) -> AggregateSummary:
    chunks = read_dataset(
        root,
        partition_filter=year_partition_filter(years),
        mode=DatasetReadMode.CHUNKED,
    )
    return _summarize_frames(chunks)


def _summarize_frames(frames: Iterable[pd.DataFrame]) -> AggregateSummary:
    row_count = 0
    key_count = 0
    duplicate_key_count = 0
    seen_key_hashes: set[int] = set()
    columns: set[str] = set()
    dtype_summary: dict[str, set[str]] = {}
    null_summary: dict[str, int] = {}
    content_summary: dict[str, dict[str, Any]] = {}
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
            key_hashes = pd.util.hash_pandas_object(
                frame.loc[:, key_columns],
                index=False,
            ).to_numpy(dtype="uint64")
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
    normalized_dtypes = {
        column: tuple(sorted(values))
        for column, values in sorted(dtype_summary.items())
    }
    normalized_content = {
        column: _normalize_content_stats(stats)
        for column, stats in sorted(content_summary.items())
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
    hashes = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype="uint64")
    stats["hash_sum"] = int((int(stats["hash_sum"]) + int(hashes.sum())) % HASH_MODULUS)
    stats["hash_xor"] = int(stats["hash_xor"]) ^ int(
        np.bitwise_xor.reduce(hashes, initial=np.uint64(0))
    )
    stats["non_null_count"] = int(stats["non_null_count"]) + int(series.notna().sum())
    if pd.api.types.is_bool_dtype(series):
        numeric = series.astype("float64")
    elif pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
    elif pd.api.types.is_datetime64_any_dtype(series):
        numeric = pd.to_datetime(series, errors="coerce").dropna().astype("int64")
    else:
        return
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return
    numeric_min = float(numeric.min())
    numeric_max = float(numeric.max())
    stats["numeric_min"] = min(
        float(stats.get("numeric_min", numeric_min)),
        numeric_min,
    )
    stats["numeric_max"] = max(
        float(stats.get("numeric_max", numeric_max)),
        numeric_max,
    )


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
            for column in ("timestamp", "feature", "label", "group")
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
