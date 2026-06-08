"""A-share processed layer materialization orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.runtime.execution_metrics import ExecutionMetricsRecorder

from ..factors.build_factors import calculate_price_volume_factor_asset_batches
from ..ids import PRICE_VOLUME_PROCESSED_DATA_ID
from ..labels.build_labels import calculate_forward_returns_asset_batches
from ..masks.build import write_masks_from_panel_dataset
from ..universes.build import write_universe_from_panel_dataset
from .contexts import write_contexts_from_panel_dataset
from .panels import (
    build_panel_asset_batches_from_interim,
)


@dataclass(frozen=True)
class AShareProcessedBuildResult:
    """Aggregate metadata for a processed-layer build run."""

    row_counts: dict[str, int]
    artifact_roots: dict[str, Path]


def build_a_share_processed_layers(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    label_calculation_end_year: int | None = None,
    min_listing_trading_days: int = 120,
    metrics_recorder: ExecutionMetricsRecorder | None = None,
) -> AShareProcessedBuildResult:
    """Build panel, contexts, universes, masks, factors, and labels.

    This orchestrator owns sequencing only. Persisted outputs remain owned by
    spec-driven storage writers called inside each layer builder.
    """

    calculation_end_year = label_calculation_end_year or end_year
    with _record_step(
        metrics_recorder,
        "build_daily_panel",
        logical_layer="processed",
        dataset_id="daily_panel",
    ) as record:
        panel_root, panel_rows = build_panel_asset_batches_from_interim(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
            end_date=end_date,
            calculation_end_year=calculation_end_year,
            min_listing_trading_days=min_listing_trading_days,
        )
        record.output_rows = panel_rows
    with _record_step(
        metrics_recorder,
        "build_context",
        logical_layer="processed",
        dataset_id="daily_context",
    ) as record:
        context_root, context_rows = write_contexts_from_panel_dataset(
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )
        record.input_rows = panel_rows
        record.output_rows = context_rows
    with _record_step(
        metrics_recorder,
        "build_universe",
        logical_layer="processed",
        dataset_id="a_share_default",
    ) as record:
        universe_root, universe_rows = write_universe_from_panel_dataset(
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )
        record.input_rows = panel_rows
        record.output_rows = universe_rows
    with _record_step(
        metrics_recorder,
        "build_masks",
        logical_layer="processed",
    ) as record:
        mask_outputs = write_masks_from_panel_dataset(
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )
        record.input_rows = panel_rows
        record.output_rows = sum(rows for _, rows in mask_outputs.values())
    with _record_step(
        metrics_recorder,
        "build_factor_price_volume",
        logical_layer="processed",
        dataset_id=PRICE_VOLUME_PROCESSED_DATA_ID,
    ) as record:
        factor_root, factor_rows = calculate_price_volume_factor_asset_batches(
            interim_root=interim_root,
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )
        record.input_rows = panel_rows
        record.output_rows = factor_rows
    with _record_step(
        metrics_recorder,
        "build_label_forward_returns",
        logical_layer="processed",
        dataset_id="forward_returns",
    ) as record:
        label_root, label_rows = calculate_forward_returns_asset_batches(
            processed_root=processed_root,
            start_year=start_year,
            end_year=end_year,
        )
        record.input_rows = panel_rows
        record.output_rows = label_rows
    artifact_roots = {
        "daily_panel": panel_root,
        "daily_context": context_root,
        "a_share_default": universe_root,
        PRICE_VOLUME_PROCESSED_DATA_ID: factor_root,
        "forward_returns": label_root,
    }
    row_counts = {
        "daily_panel": panel_rows,
        "daily_context": context_rows,
        "a_share_default": universe_rows,
        PRICE_VOLUME_PROCESSED_DATA_ID: factor_rows,
        "forward_returns": label_rows,
    }
    for mask_kind, (path, rows) in mask_outputs.items():
        artifact_roots[f"mask_{mask_kind}"] = path
        row_counts[f"mask_{mask_kind}"] = rows
    return AShareProcessedBuildResult(
        row_counts=row_counts,
        artifact_roots=artifact_roots,
    )


@contextmanager
def _record_step(
    recorder: ExecutionMetricsRecorder | None,
    step_name: str,
    *,
    logical_layer: str | None = None,
    dataset_id: str | None = None,
):
    if recorder is None:
        yield _NoopMetric()
        return
    with recorder.record(
        step_name,
        logical_layer=logical_layer,
        dataset_id=dataset_id,
    ) as metric:
        yield metric


@dataclass
class _NoopMetric:
    input_rows: int | None = None
    output_rows: int | None = None


__all__ = ["AShareProcessedBuildResult", "build_a_share_processed_layers"]
