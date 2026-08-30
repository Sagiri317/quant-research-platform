"""Atomic Binance Futures HFT research orchestration."""

from __future__ import annotations

import datetime as dt
import os
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import psutil

from quant_research_platform.artifacts.crypto_hft import (
    write_crypto_hft_execution_artifact,
)
from quant_research_platform.artifacts.evidence import write_json_evidence
from quant_research_platform.artifacts.outputs import (
    read_prediction_output_bundle,
    write_evaluation_outputs_from_chunks,
    write_prediction_outputs,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.order_book import BookBoundary, BookState
from quant_research_platform.contracts.outputs import OutputScope, OutputSpec
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
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
from quant_research_platform.markets.implementations.crypto.research_datasets import (
    build_hft_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_EVALUATION_ID,
    HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
    HFT_MODEL_ALGORITHMS,
    HFT_PRICE_MODEL_ALGORITHMS,
    build_hft_evaluation_catalog,
    build_hft_model_catalog,
    build_hft_prediction_catalog,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.reports.crypto_hft import (
    build_crypto_hft_report,
)
from quant_research_platform.research.datasets.assembler import (
    assemble_research_dataset,
)
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.evaluations.runner import (
    build_evaluation_result_from_research_dataset,
)
from quant_research_platform.research.models.dependencies import (
    preflight_model_dependency,
)
from quant_research_platform.research.predictions.alignment import (
    _iter_prediction_joined_chunks,
)
from quant_research_platform.research.predictions.runner import (
    build_prediction_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    RecorderRun,
    iter_agg_trades,
    iter_reconstructed_depth,
    open_recorder_run,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import crypto_hft_materialization

_LATENCY_SCENARIOS_MS = (0, HFT_GRID_MS, 2 * HFT_GRID_MS)
_TAKER_MODEL_ID = "crypto_hft_linear_regression"
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
        source_run_id = str(run.manifest["run_id"])
        inspections, reconstruction = (
            crypto_hft_materialization.materialize_hft_processed_data(
                layout=layout,
                run=run,
                time_window=time_window,
                resource_policy=resource_policy,
            )
        )
        sample_resources()
        dataset_catalog = build_hft_research_dataset_catalog()
        research_root = assemble_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            processed_catalog=hft_processed_catalog(),
            dataset_id=HFT_RESEARCH_DATASET_ID,
            time_window=time_window,
            resource_policy=resource_policy,
        )
        sample_resources()
        evaluation_catalog = build_hft_evaluation_catalog()
        evaluation_spec = evaluation_catalog.get(HFT_EVALUATION_ID)
        with build_evaluation_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            evaluation_catalog=evaluation_catalog,
            evaluation_id=HFT_EVALUATION_ID,
            time_window=time_window,
            resource_policy=resource_policy,
        ) as evaluation_result:
            evaluation_bundle = write_evaluation_outputs_from_chunks(
                evaluation_result.output_chunks,
                summary_factory=evaluation_result.summaries,
                spec=evaluation_spec,
                output_spec=OutputSpec(
                    output_id=HFT_EVALUATION_ID,
                    output_definition_id="evaluation_outputs",
                    run_id=source_run_id,
                    market_id=MARKET_ID,
                    run_type=evaluation_spec.run_type,
                    scope=OutputScope.LOCAL_ONLY,
                ),
                layout=layout,
                n_groups=evaluation_spec.n_groups,
                resource_policy=resource_policy,
            )
            evaluation_summaries = evaluation_result.summaries()
            evaluation_diagnostics = evaluation_summaries[0]
            evaluation_ic_summary = evaluation_summaries[1]
            evaluation_valid_ic_count = int(
                evaluation_ic_summary["valid_ic_count"].sum()
            )
        evaluation_status = "PASS" if evaluation_valid_ic_count > 0 else "FAIL"
        evaluation_relative = evaluation_bundle.relative_to(staging)
        sample_resources()
        split_steps = (end_ms - start_ms) // HFT_GRID_MS * 2 // 3
        split_ms = start_ms + split_steps * HFT_GRID_MS
        test_window = TimeWindow(_datetime(split_ms), _datetime(end_ms))
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
        model_metadata_by_id: dict[str, dict[str, Any]] = {}
        prediction_evaluations: dict[str, dict[str, Any]] = {}
        for prediction_spec in prediction_catalog.list():
            result = build_prediction_result_from_research_dataset(
                layout=layout,
                dataset_catalog=dataset_catalog,
                model_catalog=model_catalog,
                prediction_catalog=prediction_catalog,
                prediction_id=prediction_spec.prediction_id,
                time_window=time_window,
                resource_policy=resource_policy,
            )
            output_id = f"{prediction_spec.prediction_id}_output"
            with result:
                bundle = write_prediction_outputs(
                    output_chunks=result.output_chunks,
                    spec=result.spec,
                    output_spec=OutputSpec(
                        output_id=output_id,
                        output_definition_id="prediction_outputs",
                        run_id=source_run_id,
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
                model_metadata_by_id[prediction_spec.model_id] = dict(
                    result.model_metadata
                )
            prediction_evaluations[prediction_spec.model_id] = (
                _prediction_oos_evaluation(
                    research_root=research_root,
                    prediction_root=bundle,
                    output_id=output_id,
                    prediction_spec=prediction_spec,
                    expected_run_id=source_run_id,
                    label_column=model_catalog.get(
                        prediction_spec.model_id
                    ).label_column,
                    resource_policy=resource_policy,
                )
            )
            sample_resources()
        coverage = dict(reconstruction.coverage_segments)

        def book_factory() -> Iterator[BookState | BookBoundary]:
            return _book_events(run, end_ms=end_ms)

        def decisions_for(model_id: str) -> Iterator[dict[str, Any]]:
            prediction_root, output_id, prediction_spec = bundles[model_id]
            return _prediction_decisions(
                research_root=research_root,
                prediction_root=prediction_root,
                output_id=output_id,
                prediction_spec=prediction_spec,
                expected_run_id=source_run_id,
                resource_policy=resource_policy,
            )

        price_model_ids = tuple(
            f"crypto_hft_{algorithm}" for algorithm in HFT_PRICE_MODEL_ALGORITHMS
        )
        taker_models = {
            model_id: simulate_taker(
                book_factory(),
                decisions_for(model_id),
                prediction_column="prediction_score",
                config=taker_config,
                observed_coverage=coverage,
                resource_policy=resource_policy,
            )
            for model_id in price_model_ids
        }
        taker = taker_models[_TAKER_MODEL_ID]
        maker_baseline = simulate_maker(
            book_factory(),
            iter_agg_trades(run, FUTURES_MARKET_TYPE, end_ms=end_ms),
            _maker_quote_schedule(
                research_root=research_root,
                time_window=test_window,
                resource_policy=resource_policy,
            ),
            config=maker_config,
            observed_coverage=coverage,
            trade_coverage=run.connection_coverage(
                market_type=FUTURES_MARKET_TYPE,
                data_kind="agg_trades",
            ),
            resource_policy=resource_policy,
        )

        def maker_prediction_schedule() -> Iterator[dict[str, Any]]:
            for row in decisions_for(HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID):
                if row.pop("prediction_score") >= 0.5:
                    yield row

        maker = simulate_maker(
            book_factory(),
            iter_agg_trades(run, FUTURES_MARKET_TYPE, end_ms=end_ms),
            maker_prediction_schedule(),
            config=maker_config,
            observed_coverage=coverage,
            trade_coverage=run.connection_coverage(
                market_type=FUTURES_MARKET_TYPE,
                data_kind="agg_trades",
            ),
            resource_policy=resource_policy,
        )
        latency = taker_latency_sweep(
            book_factory,
            lambda: decisions_for(_TAKER_MODEL_ID),
            prediction_column="prediction_score",
            config=taker_config,
            latency_scenarios_ms=_LATENCY_SCENARIOS_MS,
            observed_coverage=coverage,
            resource_policy=resource_policy,
        )
        execution_relative = Path("execution") / "model_comparison.json"
        execution_payload = write_crypto_hft_execution_artifact(
            staging / execution_relative,
            taker_model_id=_TAKER_MODEL_ID,
            maker_policy_id="public_l2_sell_pressure_proxy_gated_maker",
            maker_prediction_used=True,
            maker_model_id=HFT_MAKER_SELL_PRESSURE_PROXY_MODEL_ID,
            source_identity=run.source_identity,
            panel_id=HFT_PANEL_DATA_ID,
            factor_id=HFT_FACTOR_DATA_ID,
            label_id=HFT_LABEL_DATA_ID,
            research_dataset_id=HFT_RESEARCH_DATASET_ID,
            evaluation_id=HFT_EVALUATION_ID,
            taker=taker,
            taker_models=taker_models,
            maker=maker,
            maker_baseline=maker_baseline,
            latency_sweep=latency,
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
                for metadata in model_metadata_by_id.values()
            ),
        )
        research_status = (
            "PASS"
            if run.manifest["market_data_status"] == "PASS"
            and reconstruction.invalid_segments == 0
            else "PARTIAL"
        )
        report = build_crypto_hft_report(
            run_id=source_run_id,
            market_data_status=run.manifest["market_data_status"],
            research_status=research_status,
            evaluation_status=evaluation_status,
            evaluation_valid_ic_count=evaluation_valid_ic_count,
            reconstruction=asdict(reconstruction),
            raw_record_count=raw_record_count,
            row_counts={
                "panel": inspections[HFT_PANEL_DATA_ID].row_count,
                "factor": inspections[HFT_FACTOR_DATA_ID].row_count,
                "label": inspections[HFT_LABEL_DATA_ID].row_count,
            },
            algorithms=HFT_MODEL_ALGORITHMS,
            prediction_outputs={
                model_id: output_id
                for model_id, (_path, output_id, _spec) in bundles.items()
            },
            prediction_evaluations=prediction_evaluations,
            model_metadata=model_metadata_by_id,
            evaluation_id=HFT_EVALUATION_ID,
            evaluation_dataset_id=HFT_RESEARCH_DATASET_ID,
            evaluation_axis=evaluation_spec.evaluation_axis,
            evaluation_primary_key=dataset_catalog.get(
                HFT_RESEARCH_DATASET_ID
            ).primary_key,
            evaluation_requested_window={
                "start": time_value_isoformat(
                    time_window.start,
                    preserve_datetime=True,
                ),
                "end": time_value_isoformat(
                    time_window.end,
                    preserve_datetime=True,
                ),
            },
            evaluation_observed_window={
                "start": evaluation_diagnostics.min_timestamp,
                "end": evaluation_diagnostics.max_timestamp,
            },
            evaluation_path=evaluation_relative.as_posix(),
            execution_path=execution_relative.as_posix(),
            execution_payload=execution_payload,
            resource={
                **resource_policy.to_payload(),
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
        )
        write_json_evidence(staging / "report.json", report)
        return report


def _datetime(value: int) -> dt.datetime:
    return _EPOCH + dt.timedelta(milliseconds=value)


def _book_events(
    run: RecorderRun,
    *,
    end_ms: int,
) -> Iterator[OwnedValue[BookState] | BookBoundary]:
    events = iter_reconstructed_depth(run, FUTURES_MARKET_TYPE)
    with closing_iterator(events):
        for event in events:
            observed_time = (
                event.value.observed_time_ns
                if isinstance(event, OwnedValue)
                else event.observed_time_ns
            )
            if observed_time > end_ms * 1_000_000:
                if isinstance(event, OwnedValue):
                    event.release(owner="workflow.crypto_hft.book_event")
                break
            yield event


def _prediction_decisions(
    *,
    research_root: Path,
    prediction_root: Path,
    output_id: str,
    prediction_spec: PredictionSpec,
    expected_run_id: str,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[dict[str, Any]]:
    frames = _prediction_joined_frames(
        research_root=research_root,
        prediction_root=prediction_root,
        output_id=output_id,
        prediction_spec=prediction_spec,
        expected_run_id=expected_run_id,
        additional_columns=(
            "segment_id",
            "book_segment_id",
            "trade_connection_id",
        ),
        resource_policy=resource_policy,
    )
    with closing_iterator(frames):
        for frame in frames:
            for row in frame.itertuples(index=False):
                value = getattr(row, prediction_spec.prediction_column)
                if pd.isna(value):
                    continue
                timestamp = pd.Timestamp(row.timestamp)
                if timestamp.tzinfo is not None:
                    timestamp = timestamp.tz_convert("UTC").tz_localize(None)
                yield {
                    "timestamp_ms": timestamp.value // 1_000_000,
                    "asset_id": str(row.asset_id),
                    "segment_id": str(row.segment_id),
                    "book_segment_id": str(row.book_segment_id),
                    "trade_connection_id": str(row.trade_connection_id),
                    "prediction_score": float(value),
                }


def _prediction_joined_frames(
    *,
    research_root: Path,
    prediction_root: Path,
    output_id: str,
    prediction_spec: PredictionSpec,
    expected_run_id: str,
    additional_columns: tuple[str, ...],
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    prediction_window = TimeWindow(
        prediction_spec.test_start,
        prediction_spec.test_end,
    )
    with read_prediction_output_bundle(
        prediction_root,
        expected_output_id=output_id,
        expected_run_id=expected_run_id,
        time_window=prediction_window,
    ) as (manifest, lookup):
        model_metadata = manifest["metadata"]["model_metadata"]
        input_columns = tuple(model_metadata["input_columns"])
        sample_filter_mode = model_metadata["sample_filter_mode"]
        read_columns = tuple(
            dict.fromkeys(
                (
                    *lookup.primary_key,
                    *additional_columns,
                    *input_columns,
                    *required_filter_columns(sample_filter_mode),
                )
            )
        )
        with open_dataset_session(
            research_root,
            columns=read_columns,
            time_window=prediction_window,
            resource_policy=resource_policy,
        ) as session:
            chunks = iter_dataset_chunks(
                session,
                target_chunk_bytes=max(
                    1,
                    resource_policy.operation_headroom_bytes,
                ),
            )
            joined = _iter_prediction_joined_chunks(
                chunks,
                prediction_source=lookup,
                primary_key=lookup.primary_key,
                input_columns=input_columns,
                sample_filter_mode=sample_filter_mode,
                test_start=prediction_spec.test_start,
                test_end=prediction_spec.test_end,
                resource_policy=resource_policy,
            )
            with closing_iterator(joined):
                yield from joined
            session.verify()


def _prediction_oos_evaluation(
    *,
    research_root: Path,
    prediction_root: Path,
    output_id: str,
    prediction_spec: PredictionSpec,
    expected_run_id: str,
    label_column: str,
    resource_policy: RuntimeResourcePolicy,
) -> dict[str, Any]:
    signal = SignalSpec(
        signal_id=f"{prediction_spec.model_id}_score",
        source_kind="prediction",
        source_column=prediction_spec.prediction_column,
    )
    spec = EvaluationSpec(
        evaluation_id=f"{prediction_spec.prediction_id}_oos_evaluation",
        market_id=MARKET_ID,
        research_dataset_id=prediction_spec.research_dataset_id,
        signals=(signal,),
        label_columns=(label_column,),
        evaluation_axis="single_asset_time_series",
    )
    frames = _prediction_joined_frames(
        research_root=research_root,
        prediction_root=prediction_root,
        output_id=output_id,
        prediction_spec=prediction_spec,
        expected_run_id=expected_run_id,
        additional_columns=(label_column,),
        resource_policy=resource_policy,
    )
    with build_evaluation_result_from_frames(
        frames,
        spec=spec,
        signal_columns=(signal.signal_id,),
        label_columns=(label_column,),
        n_groups=spec.n_groups,
        asset_column="asset_id",
        source_columns_by_kind={"prediction": (prediction_spec.prediction_column,)},
        resource_policy=resource_policy,
    ) as result:
        for _output in result.output_chunks:
            pass
        diagnostics, ic_summary, _spread, _coverage = result.summaries()
    row = ic_summary.iloc[0]
    return {
        "evaluation_id": spec.evaluation_id,
        "evaluation_axis": spec.evaluation_axis,
        "label_column": label_column,
        "rows_input": diagnostics.rows_input,
        "rows_after_filter": diagnostics.rows_after_filter_mode,
        "observed_start": diagnostics.min_timestamp,
        "observed_end": diagnostics.max_timestamp,
        "ic": None if pd.isna(row["mean_ic"]) else float(row["mean_ic"]),
        "valid_ic_count": int(row["valid_ic_count"]),
        "ic_reason": None if pd.isna(row["ic_reason"]) else str(row["ic_reason"]),
    }


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
            target_chunk_bytes=max(1, resource_policy.operation_headroom_bytes),
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
