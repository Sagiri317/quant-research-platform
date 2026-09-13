"""Persisted research inputs to the event-driven perpetual ledger."""

from array import array
from contextlib import ExitStack
from functools import partial
from itertools import chain, groupby, zip_longest
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.order_book import BookBoundary, TradeEvent
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    MakerConfig,
    TakerConfig,
    _MakerOrder,
    replay_maker_calibration,
    simulate_maker,
    simulate_taker,
)
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.io import iter_research_dataset_chunks
from quant_research_platform.research.predictions.alignment import (
    _iter_prediction_joined_chunks,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_EXECUTION_CONTEXT_COLUMNS,
    HFT_MAKER_CALIBRATION_MODEL_IDS,
    HFT_MODEL_HORIZON_MS,
    execution_windows,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._scratch_ownership import handle_preserving_primary
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence


def _window_bounds(windows):
    if not windows:
        raise PortfolioTargetBlocked("crypto_execution_oos_window_missing")
    return tuple(
        (pd.Timestamp(window.start).value, pd.Timestamp(window.end).value)
        for _fold, window in windows
    )


def _panel(layout, upstream_identities):
    panel = read_dataset_manifest_evidence(
        layout.target_root(HFT_PANEL_DATA_ID, Path("__missing_panel__"))
    )
    if (
        panel[COMMON_MANIFEST_METADATA_KEY]["instance_identity"]
        != upstream_identities[HFT_PANEL_DATA_ID]
    ):
        raise ValueError("crypto_execution_panel_identity_mismatch")
    return panel


def _books_until(run, *, reconstruct_depth, symbol, end_ns):
    source = reconstruct_depth(run, symbol=symbol, end_ns=end_ns)
    with closing_iterator(source):
        for event in source:
            value = event.value if isinstance(event, OwnedValue) else event
            if value.observed_time_ns > end_ns:
                if isinstance(event, OwnedValue):
                    event.release(owner="crypto.execution.book")
                break
            yield event


def _prediction_rows(
    *,
    layout,
    dataset,
    windows,
    model_id,
    asset_id,
    protocol_manifest,
    resource_policy,
    read_prediction_bundle,
    output_window,
):
    target = f"{dataset.data_id}_predictions"
    output_id = f"{model_id}_prediction"
    root = layout.target_root(target, Path("__missing_prediction__"))
    with read_prediction_bundle(
        root / "outputs" / output_id,
        expected_output_id=output_id,
        expected_run_id=f"run_{output_id}",
        time_window=output_window,
    ) as (manifest, lookup):
        metadata = manifest["metadata"]["prediction_metadata"]
        inputs = tuple(metadata["input_columns"])
        assets = tuple(
            protocol_manifest["model_scopes"][model_id]["evaluation_asset_ids"]
        )
        columns = tuple(
            dict.fromkeys(
                (
                    *dataset.primary_key,
                    *inputs,
                    *required_filter_columns(metadata["sample_filter_mode"]),
                    *HFT_EXECUTION_CONTEXT_COLUMNS,
                )
            )
        )

        def frames():
            for fold_id, window in windows:
                source = iter_research_dataset_chunks(
                    layout,
                    market_id=dataset.market_id,
                    dataset_id=dataset.data_id,
                    time_window=window,
                    columns=columns,
                )
                with closing_iterator(source):
                    for frame in source:
                        with resource_policy.lease(
                            "crypto.execution.prediction.scope",
                            max(
                                1, pandas_frame_payload_bytes(frame) + len(frame) * 128
                            ),
                        ):
                            scoped = (
                                frame.loc[frame.asset_id.isin(assets)]
                                if assets
                                else frame
                            )
                            scoped = scoped.assign(fold_id=fold_id)
                            yield scoped
                            del scoped

        joined = _iter_prediction_joined_chunks(
            frames(),
            prediction_lookup=lookup,
            input_columns=inputs,
            sample_filter_mode=metadata["sample_filter_mode"],
            test_start=pd.Timestamp(output_window.start).isoformat(),
            test_end=pd.Timestamp(output_window.end).isoformat(),
        )
        with closing_iterator(joined):
            for frame in joined:
                for row in frame.itertuples(index=False):
                    if row.asset_id == asset_id:
                        yield {
                            "timestamp_ms": pd.Timestamp(row.timestamp).value
                            // 1_000_000,
                            "asset_id": asset_id,
                            "book_segment_id": row.book_segment_id,
                            "trade_observation_id": row.trade_observation_id,
                            "segment_id": row.segment_id,
                            "fold_id": row.fold_id,
                            "prediction_score": None
                            if pd.isna(row.prediction_score)
                            else float(row.prediction_score),
                        }


def _funding_check(funding, expected_identity, bounds):
    if funding.identity != expected_identity:
        raise ValueError("crypto_execution_funding_identity_mismatch")
    if not funding.coverage[0] <= bounds[0][0] <= bounds[-1][1] <= funding.coverage[1]:
        raise PortfolioTargetBlocked("binance_funding_coverage_incomplete")


def _maker_config(plan, asset_id, *, calibration=False):
    limits = plan["exposure_limits"][asset_id]
    execution = plan["execution"]
    maker = execution["maker"]
    return MakerConfig(
        order_quantity=min(
            limits["max_order_quantity"], 2 * limits["max_position_quantity"]
        )
        if calibration
        else limits["max_order_quantity"],
        max_position=limits["max_position_quantity"],
        maker_fee_bps=maker["maker_fee_bps"],
        inventory_target=maker["inventory_target"],
        inventory_skew=maker["inventory_skew"],
        markout_horizon_ms=execution["markout_horizon_ms"],
        latency_ms=maker["latency_ms"],
    )


def produce_taker_ledger(
    *,
    layout,
    candidate,
    protocol_manifest,
    dataset_catalog,
    output_window,
    upstream_identities,
    expected_funding_identity,
    raw_root,
    resource_policy,
    on_ledger,
    symbol,
    open_run,
    reconstruct_depth,
    read_funding,
    read_prediction_bundle,
):
    plan = protocol_manifest["research_plan"]
    execution = plan["execution"]
    limits = plan["exposure_limits"][candidate.asset_id]
    config = TakerConfig(
        order_quantity=limits["max_order_quantity"],
        position_size=limits["max_position_quantity"],
        latency_ms=execution["latency_ms"]
        if candidate.latency_ms is None
        else candidate.latency_ms,
        taker_fee_bps=execution["taker_fee_bps"],
        markout_horizon_ms=execution["markout_horizon_ms"],
    )
    windows = execution_windows(plan, output_window)
    bounds = _window_bounds(windows)
    dataset = dataset_catalog.get(protocol_manifest["scenario_id"])
    panel = _panel(layout, upstream_identities)
    coverage = panel["producer_diagnostics"]["reconstruction"][symbol][
        "coverage_segments"
    ]
    with (
        open_run(raw_root, resource_policy) as run,
        read_funding(
            run.funding_root(symbol), symbol=symbol, resource_policy=resource_policy
        ) as funding,
    ):
        if run.source_identity != panel["source_artifact_identity"]:
            raise ValueError("crypto_execution_recorder_identity_mismatch")
        _funding_check(funding, expected_funding_identity, bounds)

        def decisions():
            rows = _prediction_rows(
                layout=layout,
                dataset=dataset,
                windows=windows,
                model_id=candidate.model_id,
                asset_id=candidate.asset_id,
                protocol_manifest=protocol_manifest,
                resource_policy=resource_policy,
                read_prediction_bundle=read_prediction_bundle,
                output_window=output_window,
            )
            with closing_iterator(rows):
                for row in rows:
                    if row["prediction_score"] is not None:
                        yield row

        result = simulate_taker(
            _books_until(
                run,
                reconstruct_depth=reconstruct_depth,
                symbol=symbol,
                end_ns=bounds[-1][1],
            ),
            decisions(),
            prediction_column="prediction_score",
            prediction_horizon_ms=HFT_MODEL_HORIZON_MS,
            config=config,
            observed_coverage=coverage,
            resource_policy=resource_policy,
            capital=PortfolioCapital(
                candidate.candidate_id,
                plan["initial_capital"],
                plan["valuation_currency"],
                "linear_perpetual",
            ),
            asset_id=candidate.asset_id,
            funding_events=funding.events,
            funding_forecasts=funding.forecasts,
            decision_windows_ns=bounds,
            on_ledger=on_ledger,
        )
        if result["selection_status"] == "BLOCKED":
            raise PortfolioTargetBlocked("portfolio_expected_funding_missing")
        if result["net_pnl"] is None:
            raise PortfolioTargetBlocked("portfolio_terminal_valuation_missing")
        return result


def _validation_rows(artifact, *, fold_id, model_id, asset_id):
    rows = artifact.validation_chunks(fold_id=fold_id, model_id=model_id)
    with closing_iterator(rows):
        for frame in rows:
            for row in frame.itertuples(index=False):
                if row.asset_id == asset_id:
                    yield row


def _paired_validation(artifact, fold_id, asset_id):
    streams = [
        _validation_rows(artifact, fold_id=fold_id, model_id=model, asset_id=asset_id)
        for model in HFT_MAKER_CALIBRATION_MODEL_IDS
    ]
    with closing_iterator(streams[0]), closing_iterator(streams[1]):
        price, proxy = (next(stream, None) for stream in streams)
        while price is not None and proxy is not None:
            if price.timestamp < proxy.timestamp:
                price = next(streams[0], None)
                continue
            if proxy.timestamp < price.timestamp:
                proxy = next(streams[1], None)
                continue
            if any(
                getattr(price, column) != getattr(proxy, column)
                for column in HFT_EXECUTION_CONTEXT_COLUMNS
            ):
                raise ValueError("maker_calibration_context_misaligned")
            if all(
                isinstance(getattr(price, column), str) and getattr(price, column)
                for column in HFT_EXECUTION_CONTEXT_COLUMNS
            ):
                yield {
                    "timestamp_ms": pd.Timestamp(price.timestamp).value // 1_000_000,
                    "asset_id": asset_id,
                    "book_segment_id": price.book_segment_id,
                    "trade_observation_id": price.trade_observation_id,
                    "price_prediction": float(price.prediction_score),
                    "price_label": float(price.label),
                    "proxy_prediction": float(proxy.prediction_score),
                    "label_available_at_ns": max(
                        pd.Timestamp(price.label_available_at).value,
                        pd.Timestamp(proxy.label_available_at).value,
                    ),
                }
            price, proxy = (next(stream, None) for stream in streams)
        # Validate unused mature rows too; unmatched feature eligibility is not a quote.
        for stream in streams:
            for _row in stream:
                pass


_CALIBRATION_BATCH_BYTES = 8 * 1024**2


def produce_maker_calibration(
    *,
    on_table,
    layout,
    protocol_manifest,
    time_window,
    upstream_identities,
    raw_root,
    resource_policy,
    open_run,
    reconstruct_depth,
    read_trades,
    read_model_artifact,
    symbol,
    market_type,
):
    plan = protocol_manifest["research_plan"]
    config = _maker_config(plan, "btc_usdt", calibration=True)
    panel = _panel(layout, upstream_identities)
    model_target = f"{protocol_manifest['scenario_id']}_models"
    models = read_model_artifact(
        layout.target_root(model_target, Path("__missing_models__")),
        expected_identity=upstream_identities[model_target],
        trial_layout=layout,
    )
    selected_folds = {fold for fold, _window in execution_windows(plan, time_window)}
    sample_columns = STATISTICS_TABLE_COLUMNS["crypto_maker_samples.parquet"]
    event_columns = STATISTICS_TABLE_COLUMNS["crypto_maker_events.parquet"]
    integers = {
        "sample_id",
        "decision_at",
        "prediction_end",
        "label_available_at",
        "initial_book_event_id",
        "initial_bid_index",
        "initial_ask_index",
        "event_start",
        "event_end",
        "event_id",
        "event_at",
        "level_index",
        "price_index",
    }
    strings = {
        "fold_id",
        "side",
        "terminal_reason",
        "book_segment_id",
        "trade_observation_id",
        "event_kind",
        "context_id",
    }
    columns_by_table = {
        "crypto_maker_samples": sample_columns,
        "crypto_maker_events": event_columns,
    }

    def empty(columns):
        return {
            column: []
            if column in strings
            else array("q" if column in integers else "d")
            for column in columns
        }

    buffers = {name: empty(columns) for name, columns in columns_by_table.items()}
    sizes = dict.fromkeys(buffers, 0)
    written = set()
    lease = resource_policy.lease("crypto.maker.calibration.buffer", 65536)

    def frame(values, columns):
        arrays = {}
        for column in columns:
            value = values[column]
            if isinstance(value, array):
                value = np.frombuffer(
                    value, dtype=np.int64 if column in integers else np.float64
                )
                if column in {
                    "decision_at",
                    "prediction_end",
                    "label_available_at",
                    "event_at",
                }:
                    value = value.view("datetime64[ns]")
            arrays[column] = value
        return pd.DataFrame(arrays, columns=columns, copy=False)

    def flush(name):
        output = frame(buffers[name], columns_by_table[name])
        try:
            on_table(name, output)
        finally:
            output = None
        written.add(name)
        buffers[name] = empty(columns_by_table[name])
        sizes[name] = 0
        lease.shrink_to(65536 + sum(sizes.values()))

    def append(name, values):
        row_bytes = (512 if name == "crypto_maker_events" else 2048) + 4 * sum(
            len(value) for value in values if isinstance(value, str)
        )
        target = max(
            1,
            min(
                _CALIBRATION_BATCH_BYTES,
                (resource_policy.operation_headroom_bytes + sum(sizes.values())) // 8,
            ),
        )
        if sum(sizes.values()) and sum(sizes.values()) + row_bytes > target:
            flush(max(sizes, key=sizes.get))
        lease.grow(row_bytes)
        sizes[name] += row_bytes
        for column, value in zip(columns_by_table[name], values, strict=True):
            buffers[name][column].append(value)

    try:
        with open_run(raw_root, resource_policy) as run:
            if run.source_identity != panel["source_artifact_identity"]:
                raise ValueError("crypto_execution_recorder_identity_mismatch")
            coverage = panel["producer_diagnostics"]["reconstruction"][symbol][
                "coverage_segments"
            ]
            trade_coverage = run.observation_coverage(
                market_type=market_type, data_kind="agg_trades", symbol=symbol
            )
            for fold in plan["folds"]:
                if fold["fold_id"] not in selected_folds:
                    continue
                counts = {side: [0, 0, 0] for side in ("buy", "sell")}

                def observe(
                    *,
                    side,
                    sample_id,
                    decision_ns,
                    prediction_end_ns,
                    available_ns,
                    prediction,
                    observed_return,
                    filled,
                    quote_quantity,
                    terminal_reason,
                    book_segment_id,
                    trade_observation_id,
                    initial_book_event_id,
                    initial_bid_index,
                    initial_ask_index,
                    event_start,
                    event_end,
                    fold_id=fold["fold_id"],
                    counts=counts,
                ):
                    values = (
                        fold_id,
                        side,
                        sample_id,
                        decision_ns,
                        prediction_end_ns,
                        available_ns,
                        prediction,
                        observed_return,
                        observed_return - prediction,
                        filled,
                        quote_quantity,
                        terminal_reason,
                        book_segment_id,
                        trade_observation_id,
                        initial_book_event_id,
                        initial_bid_index,
                        initial_ask_index,
                        event_start,
                        event_end,
                    )
                    append("crypto_maker_samples", values)
                    counts[side][0] += 1
                    counts[side][1] += int(filled == 0)
                    counts[side][2] += int(
                        terminal_reason == "post_only_crossing_rejected"
                    )

                def observe_event(*, event_id, event, fold_id=fold["fold_id"]):
                    if isinstance(event, TradeEvent):
                        kind, context = "trade", event.observation_id
                        rows = (
                            (
                                "buy" if event.buyer_is_maker else "sell",
                                0,
                                event.price_index,
                                event.visible_quantity,
                            ),
                        )
                    elif isinstance(event, BookBoundary):
                        kind, context = "boundary", event.segment_id
                        rows = (("", 0, 0, 0.0),)
                    else:
                        kind, context = "book", event.segment_id
                        rows = (
                            (side, index, tick, quantity)
                            for side, levels in (
                                ("buy", event.bids),
                                ("sell", event.asks),
                            )
                            for index, (tick, quantity) in enumerate(levels)
                        )
                    for side, level, tick, quantity in rows:
                        values = (
                            fold_id,
                            event_id,
                            event.observed_time_ns,
                            kind,
                            context,
                            side,
                            level,
                            tick,
                            quantity,
                        )
                        append("crypto_maker_events", values)

                cutoff = pd.Timestamp(fold["validation"]["end"]).value
                result = replay_maker_calibration(
                    _books_until(
                        run,
                        reconstruct_depth=reconstruct_depth,
                        symbol=symbol,
                        end_ns=cutoff,
                    ),
                    read_trades(run, symbol=symbol, end_ms=cutoff // 1_000_000),
                    _paired_validation(models, fold["fold_id"], "btc_usdt"),
                    config=config,
                    observed_coverage=coverage,
                    trade_coverage=trade_coverage,
                    cutoff_ns=cutoff,
                    resource_policy=resource_policy,
                    on_sample=observe,
                    on_event=observe_event,
                )
                minimum = plan["execution"]["maker"]["min_calibration_samples"]
                summaries = []
                for side, (count, zero, rejected) in counts.items():
                    summaries.append(
                        {
                            "fold_id": fold["fold_id"],
                            "side": side,
                            "sample_count": count,
                            "zero_fill_count": zero,
                            "post_only_rejection_count": rejected,
                            "excluded_decisions": result["excluded_decisions"],
                            "minimum_samples": minimum,
                            "status": "DEFINED" if count >= minimum else "UNDEFINED",
                            "reason": ""
                            if count >= minimum
                            else "insufficient_calibration_samples",
                        }
                    )

                on_table(
                    "crypto_maker_calibration", pd.DataFrame.from_records(summaries)
                )
                for name in buffers:
                    if sizes[name]:
                        flush(name)
        for name in buffers:
            if sizes[name] or name not in written:
                flush(name)
    finally:
        buffers.clear()
        _close_lease_preserving_primary(lease, owner="crypto.maker.calibration")


def produce_maker_ledger(
    *,
    layout,
    candidate,
    protocol_manifest,
    dataset_catalog,
    output_window,
    upstream_identities,
    expected_funding_identity,
    raw_root,
    resource_policy,
    on_ledger,
    symbol,
    market_type,
    open_run,
    reconstruct_depth,
    read_trades,
    read_funding,
    read_prediction_bundle,
    read_statistics_artifact,
):
    plan = protocol_manifest["research_plan"]
    windows = execution_windows(plan, output_window)
    bounds = _window_bounds(windows)
    config = _maker_config(plan, candidate.asset_id)
    panel = _panel(layout, upstream_identities)
    dataset = dataset_catalog.get(protocol_manifest["scenario_id"])
    coverage = panel["producer_diagnostics"]["reconstruction"][symbol][
        "coverage_segments"
    ]
    with handle_preserving_primary(ExitStack()) as resources:
        calibrations = None
        if candidate.calibrated:
            target = f"{protocol_manifest['scenario_id']}_maker_calibration"
            artifact = read_statistics_artifact(
                layout.target_root(target, Path("__missing_calibration__")),
                expected_identity=upstream_identities[target],
                validate_tables=validate_maker_calibration,
            )
            with artifact.table("crypto_maker_calibration.parquet") as summary:
                if not summary.status.eq("DEFINED").all():
                    raise PortfolioTargetBlocked(
                        "maker_calibration_insufficient_samples"
                    )
            calibrations = {}
            for fold_id, _window in windows:
                calibrations[fold_id] = {}
                for side in ("buy", "sell"):
                    calibration = MakerCalibration(
                        partial(_calibration_samples, artifact, fold_id, side),
                        partial(_calibration_events, artifact, fold_id),
                        resource_policy=resource_policy,
                    )
                    resources.callback(calibration.close)
                    calibrations[fold_id][side] = calibration

        def quotes():
            if candidate.calibrated:
                streams = [
                    _prediction_rows(
                        layout=layout,
                        dataset=dataset,
                        windows=windows,
                        model_id=model,
                        asset_id=candidate.asset_id,
                        protocol_manifest=protocol_manifest,
                        resource_policy=resource_policy,
                        read_prediction_bundle=read_prediction_bundle,
                        output_window=output_window,
                    )
                    for model in HFT_MAKER_CALIBRATION_MODEL_IDS
                ]
                with closing_iterator(streams[0]), closing_iterator(streams[1]):
                    for price, proxy in zip_longest(*streams):
                        if (
                            price is None
                            or proxy is None
                            or any(
                                price[key] != proxy[key]
                                for key in (
                                    "timestamp_ms",
                                    "fold_id",
                                    *HFT_EXECUTION_CONTEXT_COLUMNS,
                                )
                            )
                        ):
                            raise ValueError("maker_prediction_keys_misaligned")
                        yield {
                            **{
                                key: value
                                for key, value in price.items()
                                if key not in {"prediction_score", "segment_id"}
                            },
                            "price_prediction": price["prediction_score"],
                            "proxy_prediction": proxy["prediction_score"],
                        }
            else:
                for fold_id, window in windows:
                    chunks = iter_research_dataset_chunks(
                        layout,
                        market_id=dataset.market_id,
                        dataset_id=dataset.data_id,
                        time_window=window,
                        columns=(
                            "timestamp",
                            "asset_id",
                            *HFT_EXECUTION_CONTEXT_COLUMNS,
                        ),
                    )
                    with closing_iterator(chunks):
                        for frame in chunks:
                            for row in frame.itertuples(index=False):
                                if row.asset_id == candidate.asset_id and all(
                                    isinstance(getattr(row, column), str)
                                    and getattr(row, column)
                                    for column in HFT_EXECUTION_CONTEXT_COLUMNS
                                ):
                                    yield {
                                        "timestamp_ms": pd.Timestamp(
                                            row.timestamp
                                        ).value
                                        // 1_000_000,
                                        "asset_id": candidate.asset_id,
                                        "fold_id": fold_id,
                                        **{
                                            column: getattr(row, column)
                                            for column in HFT_EXECUTION_CONTEXT_COLUMNS
                                            if column != "segment_id"
                                        },
                                    }

        with (
            open_run(raw_root, resource_policy) as run,
            read_funding(
                run.funding_root(symbol), symbol=symbol, resource_policy=resource_policy
            ) as funding,
        ):
            if run.source_identity != panel["source_artifact_identity"]:
                raise ValueError("crypto_execution_recorder_identity_mismatch")
            _funding_check(funding, expected_funding_identity, bounds)
            result = simulate_maker(
                _books_until(
                    run,
                    reconstruct_depth=reconstruct_depth,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                ),
                read_trades(run, symbol=symbol, end_ms=bounds[-1][1] // 1_000_000),
                quotes(),
                config=config,
                observed_coverage=coverage,
                trade_coverage=run.observation_coverage(
                    market_type=market_type, data_kind="agg_trades", symbol=symbol
                ),
                resource_policy=resource_policy,
                capital=PortfolioCapital(
                    candidate.candidate_id,
                    plan["initial_capital"],
                    plan["valuation_currency"],
                    "linear_perpetual",
                ),
                asset_id=candidate.asset_id,
                funding_events=funding.events,
                funding_forecasts=funding.forecasts,
                calibrations=calibrations,
                decision_windows_ns=bounds,
                on_ledger=on_ledger,
            )
            if result["selection_status"] == "BLOCKED":
                raise PortfolioTargetBlocked("portfolio_expected_funding_missing")
            if result["net_pnl"] is None:
                raise PortfolioTargetBlocked("portfolio_terminal_valuation_missing")
            if candidate.calibrated:
                artifact._verify()
            return result


def _calibration_rows(artifact, member, filters=()):
    with closing_iterator(artifact._batches(member, filters=list(filters))) as batches:
        for frame in batches:
            yield from frame.itertuples(index=False)


def _calibration_samples(artifact, fold_id, side):
    with closing_iterator(
        _calibration_rows(
            artifact,
            "crypto_maker_samples.parquet",
            (("fold_id", "==", fold_id), ("side", "==", side)),
        )
    ) as rows:
        for row in rows:
            sample = row._asdict()
            sample["decision_at"] = row.decision_at.value
            yield sample


def _calibration_events(artifact, fold_id, start, stop):
    if start == stop:
        return
    policy = current_runtime_resource_policy()
    active = previous = lease = event = bids = asks = None
    with closing_iterator(
        _calibration_rows(
            artifact,
            "crypto_maker_events.parquet",
            (
                ("fold_id", "==", fold_id),
                ("event_id", ">=", start),
                ("event_id", "<", stop),
            ),
        )
    ) as rows:
        try:
            for event_id, group in groupby(rows, key=lambda row: row.event_id):
                first = next(group)
                if first.event_kind == "book":
                    lease = policy.lease(
                        "crypto.maker.calibration.book",
                        65536 + 4 * len(first.context_id),
                    )
                    bids, asks = [], []
                    for row in chain((first,), group):
                        lease.grow(256)
                        (bids if row.side == "buy" else asks).append(
                            (int(row.price_index), float(row.quantity))
                        )
                    event = SimpleNamespace(
                        kind="book",
                        event_id=event_id,
                        event_at=first.event_at.value,
                        segment_id=first.context_id,
                        bids=tuple(bids),
                        asks=tuple(asks),
                    )
                    bids = asks = None
                    previous, active = active, OwnedValue(event, lease)
                    lease = None
                    yield event
                    event = None
                    if previous is not None:
                        retired, previous = previous, None
                        retired.release(owner="crypto.maker.calibration.book")
                else:
                    event = SimpleNamespace(
                        kind=first.event_kind,
                        event_id=event_id,
                        event_at=first.event_at.value,
                        segment_id=first.context_id,
                        observation_id=first.context_id,
                        price_index=int(first.price_index),
                        visible_quantity=float(first.quantity),
                        buyer_is_maker=first.side == "buy",
                    )
                    yield event
                    event = None
                    if first.event_kind == "boundary" and active is not None:
                        retired, active = active, None
                        retired.release(owner="crypto.maker.calibration.book")
        finally:
            event = bids = asks = None
            for owned in (previous, active):
                if owned is not None:
                    owned.release(owner="crypto.maker.calibration.book")
            if lease is not None:
                _close_lease_preserving_primary(
                    lease, owner="crypto.maker.calibration.book"
                )


def _calibration_headers(artifact, fold_id, cutoff):
    expected = 0
    previous_time = None
    with closing_iterator(
        _calibration_rows(
            artifact, "crypto_maker_events.parquet", (("fold_id", "==", fold_id),)
        )
    ) as rows:
        for event_id, group in groupby(rows, key=lambda row: row.event_id):
            first = next(group)
            stamp = first.event_at.value
            if (
                event_id != expected
                or stamp > cutoff
                or previous_time is not None
                and stamp < previous_time
            ):
                raise ValueError("crypto_calibration_event_order_invalid")
            expected += 1
            previous_time = stamp
            counts = {"buy": 0, "sell": 0}
            best, last = {}, {}
            row_count = 0
            for row in chain((first,), group):
                row_count += 1
                if (
                    (row.event_at, row.event_kind, row.context_id)
                    != (first.event_at, first.event_kind, first.context_id)
                    or not isfinite(row.quantity)
                    or row.quantity < 0
                ):
                    raise ValueError("crypto_calibration_event_invalid")
                if row.event_kind == "book":
                    if (
                        row.side not in counts
                        or not row.context_id
                        or row.price_index <= 0
                        or row.quantity <= 0
                    ):
                        raise ValueError("crypto_calibration_event_invalid")
                    if row.level_index != counts[row.side]:
                        raise ValueError("crypto_calibration_book_invalid")
                    if counts[row.side]:
                        if (
                            row.price_index >= last[row.side]
                            if row.side == "buy"
                            else row.price_index <= last[row.side]
                        ):
                            raise ValueError("crypto_calibration_book_invalid")
                    else:
                        best[row.side] = row.price_index
                    last[row.side] = row.price_index
                    counts[row.side] += 1
                elif row.event_kind == "trade":
                    if (
                        row_count != 1
                        or row.side not in counts
                        or not row.context_id
                        or row.price_index <= 0
                        or row.level_index != 0
                    ):
                        raise ValueError("crypto_calibration_event_invalid")
                elif row.event_kind == "boundary":
                    if (
                        row_count != 1
                        or row.side
                        or row.price_index != 0
                        or row.quantity != 0
                        or row.level_index != 0
                    ):
                        raise ValueError("crypto_calibration_event_invalid")
                else:
                    raise ValueError("crypto_calibration_event_invalid")
            if first.event_kind == "book" and (
                not all(counts.values()) or best["buy"] >= best["sell"]
            ):
                raise ValueError("crypto_calibration_book_invalid")
            yield SimpleNamespace(
                event_id=event_id,
                event_at=stamp,
                kind=first.event_kind,
                context=first.context_id,
                row_count=row_count,
            )


def validate_maker_calibration(artifact, envelope):
    protocol = envelope["semantic_metadata"]["target_contract"]["protocol"]
    plan = protocol["research_plan"]
    minimum = plan["execution"]["maker"]["min_calibration_samples"]
    output = envelope["research_window"]
    folds = {
        fold["fold_id"]: fold
        for fold in plan["folds"]
        if pd.Timestamp(fold["development_test"]["start"])
        <= pd.Timestamp(output["end"])
        and pd.Timestamp(fold["development_test"]["end"])
        >= pd.Timestamp(output["start"])
    }
    reference_config = _maker_config(plan, "btc_usdt", calibration=True)
    policy = current_runtime_resource_policy()
    totals = dict.fromkeys(("samples", "events", "calibration"), 0)
    paired_columns = tuple(
        column
        for column in STATISTICS_TABLE_COLUMNS["crypto_maker_samples.parquet"]
        if column not in ("side", "filled_quantity", "terminal_reason")
    )
    for fold_id, fold in folds.items():
        summaries = {}
        with closing_iterator(
            _calibration_rows(
                artifact,
                "crypto_maker_calibration.parquet",
                (("fold_id", "==", fold_id),),
            )
        ) as rows:
            for row in rows:
                if row.side not in ("buy", "sell") or row.side in summaries:
                    raise ValueError("crypto_calibration_keys_invalid")
                summaries[row.side] = row
        if set(summaries) != {"buy", "sell"}:
            raise ValueError("crypto_calibration_keys_invalid")
        totals["calibration"] += 2
        cutoff = pd.Timestamp(fold["validation"]["end"]).value
        beginning = pd.Timestamp(fold["validation"]["start"]).value
        counts = {side: [0, 0, 0] for side in ("buy", "sell")}
        with handle_preserving_primary(ExitStack()) as reads:
            starts = reads.enter_context(
                closing_iterator(_calibration_headers(artifact, fold_id, cutoff))
            )
            ends = reads.enter_context(
                closing_iterator(_calibration_headers(artifact, fold_id, cutoff))
            )
            samples = [
                reads.enter_context(
                    closing_iterator(
                        _calibration_rows(
                            artifact,
                            "crypto_maker_samples.parquet",
                            (("fold_id", "==", fold_id), ("side", "==", side)),
                        )
                    )
                )
                for side in ("buy", "sell")
            ]
            calibrations = {}
            for side in ("buy", "sell"):
                model = MakerCalibration(
                    partial(_calibration_samples, artifact, fold_id, side),
                    partial(_calibration_events, artifact, fold_id),
                    resource_policy=policy,
                )
                reads.callback(model.close)
                calibrations[side] = model
            start_header = next(starts, None)
            end_header = next(ends, None)
            previous_start = previous_end = latest_state = None
            covered = 0
            for sample_id, pair in enumerate(zip_longest(*samples)):
                buy, sell = pair
                if (
                    buy is None
                    or sell is None
                    or buy.sample_id != sample_id
                    or sell.sample_id != sample_id
                    or any(
                        getattr(buy, column) != getattr(sell, column)
                        for column in paired_columns
                    )
                ):
                    raise ValueError("crypto_calibration_paired_path_invalid")
                for row in pair:
                    if (
                        not all(
                            isfinite(getattr(row, column))
                            for column in (
                                "price_prediction",
                                "observed_return",
                                "forecast_error",
                                "filled_quantity",
                                "quote_quantity",
                            )
                        )
                        or row.forecast_error
                        != row.observed_return - row.price_prediction
                        or not 0 <= row.filled_quantity <= row.quote_quantity
                        or row.quote_quantity != reference_config.order_quantity
                        or not row.book_segment_id
                        or not row.trade_observation_id
                    ):
                        raise ValueError("crypto_calibration_response_invalid")
                decision = buy.decision_at.value
                expiry = buy.prediction_end.value
                if not (
                    beginning <= decision <= cutoff
                    and expiry == decision + protocol["label_horizon_ms"] * 1000000
                    and expiry <= buy.label_available_at.value <= cutoff
                ):
                    raise ValueError("crypto_calibration_outcome_not_mature")
                initial, start, stop = (
                    buy.initial_book_event_id,
                    buy.event_start,
                    buy.event_end,
                )
                if not 0 <= initial < start <= stop:
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                while start_header is not None and start_header.event_id < start:
                    previous_start = start_header
                    if start_header.kind in ("book", "boundary"):
                        latest_state = start_header
                    start_header = next(starts, None)
                if (
                    previous_start is None
                    or previous_start.event_id != start - 1
                    or latest_state is None
                    or latest_state.event_id != initial
                    or latest_state.kind != "book"
                    or latest_state.context != buy.book_segment_id
                    or previous_start.event_at > decision
                    or start_header is not None
                    and start_header.event_at <= decision
                ):
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                while end_header is not None and end_header.event_id < stop:
                    previous_end = end_header
                    totals["events"] += end_header.row_count
                    end_header = next(ends, None)
                if (
                    previous_end is None
                    or previous_end.event_id != stop - 1
                    or stop > start
                    and previous_end.event_at >= expiry
                    or end_header is not None
                    and end_header.event_at < expiry
                    or start > covered
                    and (initial != covered or start != covered + 1)
                ):
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                covered = max(covered, stop)
                for row in pair:
                    model = calibrations[row.side]
                    sample = row._asdict()
                    sample["decision_at"] = decision
                    with model._book(initial) as state:
                        if (row.initial_bid_index, row.initial_ask_index) != (
                            state.bids[0][0],
                            state.asks[0][0],
                        ):
                            raise ValueError("crypto_calibration_origin_invalid")
                        with policy.lease(
                            "crypto.maker.calibration.reference",
                            65536 + 512 * (len(state.bids) + len(state.asks)),
                        ):
                            order = _MakerOrder(
                                0,
                                row.side,
                                (state.bids if row.side == "buy" else state.asks)[0][0],
                                row.quote_quantity,
                                0.0,
                                0.0,
                                row.book_segment_id,
                                row.trade_observation_id,
                                "1",
                                decision,
                                expiry,
                                decision + reference_config.latency_ms * 1000000,
                            )
                            remaining = row.quote_quantity
                            with closing_iterator(
                                model._response(
                                    sample,
                                    state=state,
                                    decision_ns=decision,
                                    new_order=order,
                                    active_order=None,
                                    commands=(),
                                    lifetime_ns=protocol["label_horizon_ms"] * 1000000,
                                    replace_quantity=row.quote_quantity,
                                )
                            ) as response:
                                while True:
                                    try:
                                        _elapsed, quantity, _price, _new = next(
                                            response
                                        )
                                    except StopIteration as complete:
                                        reason = complete.value
                                        break
                                    remaining -= quantity
                            if (
                                row.filled_quantity != row.quote_quantity - remaining
                                or row.terminal_reason != reason
                            ):
                                raise ValueError(
                                    "crypto_calibration_reference_response_invalid"
                                )
                            state = None
                    counts[row.side][0] += 1
                    counts[row.side][1] += row.filled_quantity == 0
                    counts[row.side][2] += (
                        row.terminal_reason == "post_only_crossing_rejected"
                    )
                    totals["samples"] += 1
            if end_header is not None:
                raise ValueError("crypto_calibration_event_coordinates_invalid")
        for side, row in summaries.items():
            count, zero, rejected = counts[side]
            if (
                row.minimum_samples != minimum
                or row.sample_count != count
                or row.zero_fill_count != zero
                or row.post_only_rejection_count != rejected
                or row.status != ("DEFINED" if count >= minimum else "UNDEFINED")
                or row.reason
                != ("" if count >= minimum else "insufficient_calibration_samples")
            ):
                raise ValueError("crypto_calibration_summary_invalid")
    if any(
        count != artifact._row_counts[f"crypto_maker_{name}.parquet"]
        for name, count in totals.items()
    ):
        raise ValueError("crypto_calibration_keys_invalid")
    artifact._verify()
