"""Atomic Binance Futures HFT research orchestration."""

from __future__ import annotations

import datetime as dt
import math
import os
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from quant_research_platform.artifacts._output_bundle_validation import (
    _read_prediction_output_bundle,
)
from quant_research_platform.artifacts.evidence import write_json_evidence
from quant_research_platform.artifacts.outputs import write_prediction_outputs
from quant_research_platform.contracts.order_book import BookBoundary, BookState
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
    simulate_maker,
    simulate_taker,
    taker_latency_sweep,
)
from quant_research_platform.markets.implementations.crypto.hft import HFT_GRID_MS
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_FACTOR_DATA_ID,
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
    HFT_RESEARCH_DATASET_ID,
    MARKET_ID,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_profiles import (
    build_hft_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MODEL_ALGORITHMS,
    build_hft_model_catalog,
    build_hft_prediction_catalog,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.models.dependencies import (
    preflight_model_dependency,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    _OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    RecorderRun,
    iter_agg_trades,
    iter_reconstructed_depth,
    open_recorder_run,
)
from quant_research_platform.storage._dataset_reading import (
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import crypto_hft_materialization

_LATENCY_SCENARIOS_MS = (0, HFT_GRID_MS, 2 * HFT_GRID_MS)
_EXECUTION_MODEL_ID = "crypto_hft_linear_regression"
_EPOCH = dt.datetime(1970, 1, 1)


def run_crypto_hft_experiment(
    *,
    output_root: Path,
    raw_root: Path,
    start_ms: int,
    end_ms: int,
    taker_config: TakerConfig,
    maker_config: MakerConfig,
    explicit_cap_bytes: int | None = None,
) -> dict[str, Any]:
    """Run one exact recorder run through the standard QRP research chain."""

    if (
        type(start_ms) is not int
        or type(end_ms) is not int
        or start_ms < 0
        or end_ms <= start_ms
        or start_ms % HFT_GRID_MS
        or end_ms % HFT_GRID_MS
    ):
        raise ValueError("crypto_hft_time_window_invalid")
    total_steps = (end_ms - start_ms) // HFT_GRID_MS
    if total_steps < 30:
        raise ValueError("crypto_hft_time_window_too_short")
    final_root = Path(output_root)
    with runtime_resource_policy_scope(
        explicit_cap_bytes=explicit_cap_bytes,
    ) as policy:
        return atomic_write_directory(
            final_root,
            lambda staging: _run_in_staging(
                staging,
                raw_root=Path(raw_root),
                start_ms=start_ms,
                end_ms=end_ms,
                taker_config=taker_config,
                maker_config=maker_config,
                resource_policy=policy,
            ),
        )


def _run_in_staging(
    staging: Path,
    *,
    raw_root: Path,
    start_ms: int,
    end_ms: int,
    taker_config: TakerConfig,
    maker_config: MakerConfig,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, Any]:
    process = psutil.Process()
    rss0 = process.memory_info().rss
    peak_rss = rss0
    peak_threads = process.num_threads()
    peak_descendants = len(process.children(recursive=True))
    peak_reserved = resource_policy.used_memory_bytes
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    io_start = _io_counters(process)
    layout = StorageLayout(staging)
    time_window = TimeWindow(_datetime(start_ms), _datetime(end_ms))

    def sample_resources() -> None:
        nonlocal peak_descendants, peak_reserved, peak_rss, peak_threads
        peak_rss = max(peak_rss, process.memory_info().rss)
        peak_threads = max(peak_threads, process.num_threads())
        peak_descendants = max(
            peak_descendants,
            len(process.children(recursive=True)),
        )
        peak_reserved = max(peak_reserved, resource_policy.used_memory_bytes)

    with open_recorder_run(raw_root, resource_policy) as run:
        inspections, reconstruction = (
            crypto_hft_materialization.materialize_hft_processed_data(
                layout=layout,
                run=run,
                time_window=time_window,
                resource_policy=resource_policy,
            )
        )
        sample_resources()
        research_root = assemble_research_dataset(
            layout=layout,
            dataset_catalog=build_hft_research_dataset_catalog(),
            processed_catalog=hft_processed_catalog(),
            dataset_id=HFT_RESEARCH_DATASET_ID,
            time_window=time_window,
            resource_policy=resource_policy,
        )
        sample_resources()
        split_steps = (end_ms - start_ms) // HFT_GRID_MS * 2 // 3
        split_ms = start_ms + split_steps * HFT_GRID_MS
        model_catalog = build_hft_model_catalog(
            training_row_bound=inspections[HFT_PANEL_DATA_ID].row_count,
        )
        prediction_catalog = build_hft_prediction_catalog(
            train_end=_datetime(split_ms - HFT_GRID_MS).isoformat(),
            test_start=_datetime(split_ms).isoformat(),
            test_end=_datetime(end_ms).isoformat(),
        )
        for algorithm in HFT_MODEL_ALGORITHMS:
            preflight_model_dependency(algorithm)
        bundles: dict[str, tuple[Path, str, Any]] = {}
        model_reports: dict[str, dict[str, Any]] = {}
        for prediction_spec in prediction_catalog.list():
            result = build_prediction_result_from_research_dataset(
                layout=layout,
                dataset_catalog=build_hft_research_dataset_catalog(),
                model_catalog=model_catalog,
                prediction_catalog=prediction_catalog,
                prediction_id=prediction_spec.prediction_id,
                time_window=time_window,
                resource_policy=resource_policy,
            )
            output_id = f"{prediction_spec.prediction_id}_output"
            try:
                bundle = write_prediction_outputs(
                    output_chunks=result.output_chunks,
                    spec=result.spec,
                    output_spec=OutputSpec(
                        output_id=output_id,
                        output_definition_id="prediction_outputs",
                        run_id=prediction_spec.prediction_id,
                        market_id=MARKET_ID,
                        run_type="prediction",
                        scope=OutputScope.LOCAL_ONLY,
                    ),
                    layout=layout,
                    model_metadata=result.model_metadata,
                    prediction_metadata_factory=(result.prediction_metadata_factory),
                    resource_policy=resource_policy,
                )
                bundles[prediction_spec.model_id] = (
                    bundle,
                    output_id,
                    prediction_spec,
                )
                model_reports[prediction_spec.model_id] = _model_report(
                    result.model_metadata
                )
            finally:
                result.close()
            sample_resources()
        coverage = dict(reconstruction.coverage_segments)
        execution_bundle, execution_output_id, execution_prediction_spec = bundles[
            _EXECUTION_MODEL_ID
        ]

        def book_factory() -> Iterator[BookState | BookBoundary]:
            return _book_events(run, end_ms=end_ms)

        def decision_factory() -> Iterator[dict[str, Any]]:
            return _prediction_decisions(
                research_root=research_root,
                prediction_root=execution_bundle,
                output_id=execution_output_id,
                prediction_spec=execution_prediction_spec,
                resource_policy=resource_policy,
            )

        taker = simulate_taker(
            book_factory(),
            decision_factory(),
            prediction_column="prediction_score",
            config=taker_config,
            observed_coverage=coverage,
        )
        maker = simulate_maker(
            book_factory(),
            iter_agg_trades(run, FUTURES_MARKET_TYPE, end_ms=end_ms),
            _maker_quote_schedule(
                research_root=research_root,
                time_window=time_window,
                resource_policy=resource_policy,
            ),
            config=maker_config,
            observed_coverage=coverage,
        )
        latency = taker_latency_sweep(
            book_factory,
            decision_factory,
            prediction_column="prediction_score",
            config=taker_config,
            latency_scenarios_ms=_LATENCY_SCENARIOS_MS,
            observed_coverage=coverage,
        )
        execution_relative = Path("execution") / "linear_regression.json"
        write_json_evidence(
            staging / execution_relative,
            _json_safe(
                {
                    "schema_id": "qrp-crypto-hft-execution-diagnostics",
                    "model_id": _EXECUTION_MODEL_ID,
                    "lineage": _lineage(run.source_identity),
                    "taker": taker,
                    "maker": maker,
                    "latency_sweep": latency,
                }
            ),
        )
        sample_resources()
        wall_seconds = max(0.001, time.perf_counter() - wall_start)
        cpu_seconds = max(0.0, time.process_time() - cpu_start)
        io_end = _io_counters(process)
        raw_record_count = sum(
            int(item["record_count"]) for item in run.manifest["files"]
        )
        model_thread_limit = max(
            1,
            *(
                int(metadata["resolved_hyperparameters"].get("n_jobs", 1))
                for metadata in model_reports.values()
            ),
        )
        report = _json_safe(
            {
                "schema_id": "qrp-crypto-hft-report",
                "status": (
                    "PASS"
                    if run.manifest["market_data_status"] == "PASS"
                    else "PARTIAL"
                ),
                "market_data_status": run.manifest["market_data_status"],
                "research_status": "PASS",
                "model_status": "PASS",
                "execution_status": "PASS",
                "lineage": _lineage(run.source_identity),
                "data_quality": {
                    **asdict(reconstruction),
                    "raw_record_count": raw_record_count,
                    "panel_rows": inspections[HFT_PANEL_DATA_ID].row_count,
                    "factor_rows": inspections[HFT_FACTOR_DATA_ID].row_count,
                    "label_rows": inspections[HFT_LABEL_DATA_ID].row_count,
                },
                "models": {
                    "algorithms": list(HFT_MODEL_ALGORITHMS),
                    "prediction_outputs": {
                        model_id: output_id
                        for model_id, (_path, output_id, _spec) in bundles.items()
                    },
                    "metadata": model_reports,
                },
                "execution": {
                    "taker_model_id": _EXECUTION_MODEL_ID,
                    "maker_policy_id": "public_l2_inventory_maker_baseline",
                    "maker_prediction_used": False,
                    "path": execution_relative.as_posix(),
                    "taker": taker,
                    "maker": maker,
                    "latency_sweep": latency,
                },
                "resource": {
                    "available0_bytes": (
                        resource_policy.captured_available_memory_bytes
                    ),
                    "memory_budget_bytes": resource_policy.memory_budget_bytes,
                    "rss0_bytes": rss0,
                    "sampled_peak_rss_bytes": peak_rss,
                    "sampled_peak_reserved_bytes": peak_reserved,
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "read_bytes": max(0, io_end[0] - io_start[0]),
                    "write_bytes": max(0, io_end[1] - io_start[1]),
                    "raw_records": raw_record_count,
                    "dataset_files": sum(
                        inspection.file_count for inspection in inspections.values()
                    ),
                    "threads": model_thread_limit,
                    "process_threads_peak": peak_threads,
                    "native_thread_env": {
                        key: os.environ.get(key)
                        for key in (
                            "OPENBLAS_NUM_THREADS",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        )
                    },
                    "descendants": peak_descendants,
                },
            }
        )
        write_json_evidence(staging / "report.json", report)
        return report


def _datetime(value: int) -> dt.datetime:
    return _EPOCH + dt.timedelta(milliseconds=value)


def _lineage(source_identity: Mapping[str, str]) -> dict[str, Any]:
    return {
        "source_artifact_identity": dict(source_identity),
        "panel": HFT_PANEL_DATA_ID,
        "factor": HFT_FACTOR_DATA_ID,
        "label": HFT_LABEL_DATA_ID,
        "research_dataset": HFT_RESEARCH_DATASET_ID,
    }


def _model_report(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model_algorithm",
        "input_columns",
        "label_column",
        "random_seed",
        "max_train_rows",
        "purge_gap_steps",
        "rows_input",
        "rows_after_filter_mode",
        "rows_after_input_label_dropna",
        "train_start",
        "train_end",
        "resolved_hyperparameters",
        "dependency_status",
    )
    return {key: metadata[key] for key in keys}


def _book_events(
    run: RecorderRun,
    *,
    end_ms: int,
) -> Iterator[_OwnedValue[BookState] | BookBoundary]:
    events = iter_reconstructed_depth(run, FUTURES_MARKET_TYPE)
    with closing_iterator(events):
        for event in events:
            observed_time = (
                event.value.observed_time_ms
                if isinstance(event, _OwnedValue)
                else event.observed_time_ms
            )
            if observed_time > end_ms:
                if isinstance(event, _OwnedValue):
                    event.lease.close()
                break
            yield event


def _prediction_decisions(
    *,
    research_root: Path,
    prediction_root: Path,
    output_id: str,
    prediction_spec: PredictionSpec,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[dict[str, Any]]:
    prediction_window = TimeWindow(
        prediction_spec.test_start,
        prediction_spec.test_end,
    )
    with _read_prediction_output_bundle(
        prediction_root,
        expected_output_id=output_id,
        time_window=prediction_window,
    ) as (_manifest, lookup):
        with open_dataset_session(
            research_root,
            columns=("timestamp", "asset_id", "segment_id", "book_segment_id"),
            time_window=prediction_window,
            resource_policy=resource_policy,
        ) as session:
            chunks = iter_dataset_chunks(
                session,
                target_chunk_bytes=max(
                    1,
                    resource_policy.available_memory_bytes,
                ),
            )
            with closing_iterator(chunks):
                for frame in chunks:
                    with lookup.values_for_frame(
                        frame,
                        mark_matched=True,
                    ) as (values, found):
                        rows = frame.itertuples(index=False)
                        for row, value, present in zip(
                            rows,
                            values,
                            found,
                            strict=True,
                        ):
                            if not present:
                                continue
                            timestamp = pd.Timestamp(row.timestamp)
                            if timestamp.tzinfo is not None:
                                timestamp = timestamp.tz_convert("UTC").tz_localize(
                                    None
                                )
                            yield {
                                "timestamp_ms": timestamp.value // 1_000_000,
                                "asset_id": str(row.asset_id),
                                "segment_id": str(row.segment_id),
                                "book_segment_id": str(row.book_segment_id),
                                "prediction_score": float(value),
                            }
            session.verify()
        unexpected, extra = lookup.coverage_counts()
        if unexpected or extra:
            raise ValueError("misaligned_persisted_prediction_keys")


def _maker_quote_schedule(
    *,
    research_root: Path,
    time_window: TimeWindow,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[dict[str, Any]]:
    with open_dataset_session(
        research_root,
        columns=(
            "timestamp",
            "asset_id",
            "segment_id",
            "book_segment_id",
            "trade_connection_id",
        ),
        time_window=time_window,
        resource_policy=resource_policy,
    ) as session:
        chunks = iter_dataset_chunks(
            session,
            target_chunk_bytes=max(1, resource_policy.available_memory_bytes),
        )
        with closing_iterator(chunks):
            for frame in chunks:
                for row in frame.itertuples(index=False):
                    timestamp = pd.Timestamp(row.timestamp)
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
                    yield {
                        "timestamp_ms": timestamp.value // 1_000_000,
                        "asset_id": str(row.asset_id),
                        "segment_id": str(row.segment_id),
                        "book_segment_id": str(row.book_segment_id),
                        "trade_connection_id": str(row.trade_connection_id),
                    }
        session.verify()


def _io_counters(process: psutil.Process) -> tuple[int, int]:
    try:
        counters = process.io_counters()
    except (AttributeError, psutil.Error):
        return 0, 0
    return int(counters.read_bytes), int(counters.write_bytes)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
