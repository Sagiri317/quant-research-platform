"""Persisted research inputs to the event-driven perpetual ledger."""

from array import array
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

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
    source = reconstruct_depth(run, symbol=symbol)
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
                            "trade_connection_id": row.trade_connection_id,
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
    coverage = {
        key: tuple(value)
        for key, value in panel["producer_diagnostics"]["reconstruction"][symbol][
            "coverage_segments"
        ].items()
    }
    with (
        open_run(raw_root, resource_policy) as run,
        read_funding(
            raw_root, symbol=symbol, resource_policy=resource_policy
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
                    "trade_connection_id": price.trade_connection_id,
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


def produce_maker_calibration(
    *,
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
        "event_start",
        "event_end",
        "event_id",
        "event_at",
        "level_index",
        "price_tick",
    }
    strings = {
        "fold_id",
        "side",
        "terminal_reason",
        "book_segment_id",
        "trade_connection_id",
        "event_kind",
        "context_id",
    }
    samples = {
        column: [] if column in strings else array("q" if column in integers else "d")
        for column in sample_columns
    }
    events = {
        column: [] if column in strings else array("q" if column in integers else "d")
        for column in event_columns
    }
    summaries = []
    lease = resource_policy.lease("crypto.maker.calibration.samples", 65536)
    try:
        with open_run(raw_root, resource_policy) as run:
            if run.source_identity != panel["source_artifact_identity"]:
                raise ValueError("crypto_execution_recorder_identity_mismatch")
            coverage = {
                key: tuple(value)
                for key, value in panel["producer_diagnostics"]["reconstruction"][
                    symbol
                ]["coverage_segments"].items()
            }
            trade_coverage = run.connection_coverage(
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
                    trade_connection_id,
                    initial_book_event_id,
                    event_start,
                    event_end,
                    fold_id=fold["fold_id"],
                    counts=counts,
                ):
                    lease.grow(512)
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
                        trade_connection_id,
                        initial_book_event_id,
                        event_start,
                        event_end,
                    )
                    for column, value in zip(sample_columns, values, strict=True):
                        samples[column].append(value)
                    counts[side][0] += 1
                    counts[side][1] += int(filled == 0)
                    counts[side][2] += int(
                        terminal_reason == "post_only_crossing_rejected"
                    )

                def observe_event(*, event_id, event, fold_id=fold["fold_id"]):
                    if isinstance(event, TradeEvent):
                        kind, context = "trade", event.connection_id
                        rows = (
                            (
                                "buy" if event.buyer_is_maker else "sell",
                                0,
                                event.price_tick,
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
                        lease.grow(512)
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
                        for column, value in zip(event_columns, values, strict=True):
                            events[column].append(value)

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

        def frame(buffers, columns):
            values = {}
            for column in columns:
                value = buffers[column]
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
                values[column] = value
            return pd.DataFrame(values, columns=columns, copy=False)

        tables = {
            "crypto_maker_samples": frame(samples, sample_columns),
            "crypto_maker_events": frame(events, event_columns),
            "crypto_maker_calibration": pd.DataFrame.from_records(summaries),
        }
        samples = events = summaries = None
        lease.shrink_to(
            max(1, sum(pandas_frame_payload_bytes(table) for table in tables.values()))
        )
        lease.transfer("research.statistics.output")
        return OwnedValue(tables, lease)
    except BaseException:
        _close_lease_preserving_primary(lease, owner="crypto.maker.calibration")
        raise


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
    coverage = {
        key: tuple(value)
        for key, value in panel["producer_diagnostics"]["reconstruction"][symbol][
            "coverage_segments"
        ].items()
    }
    with handle_preserving_primary(ExitStack()) as resources:
        calibrations = None
        if candidate.calibrated:
            target = f"{protocol_manifest['scenario_id']}_maker_calibration"
            artifact = read_statistics_artifact(
                layout.target_root(target, Path("__missing_calibration__")),
                expected_identity=upstream_identities[target],
                validate_tables=validate_maker_calibration,
            )
            calibrations = {}
            with ExitStack() as reads:
                samples = reads.enter_context(
                    artifact.table("crypto_maker_samples.parquet")
                )
                events = reads.enter_context(
                    artifact.table("crypto_maker_events.parquet")
                )
                summary = reads.enter_context(
                    artifact.table("crypto_maker_calibration.parquet")
                )
                if not summary.status.eq("DEFINED").all():
                    raise PortfolioTargetBlocked(
                        "maker_calibration_insufficient_samples"
                    )
                state = resources.enter_context(
                    resource_policy.lease(
                        "crypto.maker.calibration.state",
                        max(
                            1,
                            4
                            * (
                                pandas_frame_payload_bytes(samples)
                                + pandas_frame_payload_bytes(events)
                            ),
                        ),
                    )
                )
                retained = 0
                for fold_id, _window in windows:
                    calibrations[fold_id] = {}
                    event_part = events.loc[events.fold_id.eq(fold_id)]
                    event_data = {
                        column: event_part[column].to_numpy(copy=True)
                        for column in (
                            "event_id",
                            "event_at",
                            "event_kind",
                            "context_id",
                            "side",
                            "level_index",
                            "price_tick",
                            "quantity",
                        )
                    }
                    event_data["event_at"] = event_data["event_at"].view(np.int64)
                    offsets = np.concatenate(
                        (
                            np.flatnonzero(
                                np.r_[True, np.diff(event_data["event_id"]) != 0]
                            ),
                            [len(event_part)],
                        )
                    )
                    retained += pandas_frame_payload_bytes(event_part) + offsets.nbytes
                    for side in ("buy", "sell"):
                        part = samples.loc[
                            samples.fold_id.eq(fold_id) & samples.side.eq(side),
                            [
                                "sample_id",
                                "forecast_error",
                                "decision_at",
                                "initial_book_event_id",
                                "event_start",
                                "event_end",
                                "book_segment_id",
                                "trade_connection_id",
                            ],
                        ].sort_values("sample_id")
                        sample_data = {
                            column: part[column].to_numpy(copy=True)
                            for column in part.columns
                        }
                        sample_data["decision_at"] = sample_data["decision_at"].view(
                            np.int64
                        )
                        calibration = MakerCalibration(
                            sample_data,
                            event_data,
                            offsets,
                            resource_policy=resource_policy,
                        )
                        resources.callback(calibration.close)
                        calibrations[fold_id][side] = calibration
                        retained += pandas_frame_payload_bytes(part)
                        del part, sample_data
                    del event_part, event_data, offsets
                state.shrink_to(max(1, retained))
                samples = events = summary = None

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
                raw_root, symbol=symbol, resource_policy=resource_policy
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
                trade_coverage=run.connection_coverage(
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
            return result


def validate_maker_calibration(tables, envelope):
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

    samples = tables["crypto_maker_samples"]
    events = tables["crypto_maker_events"]
    summary = tables["crypto_maker_calibration"]
    keys = ["fold_id", "side", "sample_id"]
    policy = current_runtime_resource_policy()
    with policy.lease(
        "artifacts.crypto.calibration.validation",
        max(
            65536,
            4 * sum(pandas_frame_payload_bytes(frame) for frame in tables.values()),
        ),
    ):
        if (
            set(zip(summary.fold_id, summary.side, strict=True))
            != {(fold, side) for fold in folds for side in ("buy", "sell")}
            or summary.duplicated(["fold_id", "side"]).any()
            or samples.duplicated(keys).any()
            or events.duplicated(["fold_id", "event_id", "side", "level_index"]).any()
            or not samples.fold_id.isin(folds).all()
            or not samples.side.isin(("buy", "sell")).all()
            or not events.fold_id.isin(folds).all()
        ):
            raise ValueError("crypto_calibration_keys_invalid")
        if not samples.empty and (
            not np.isfinite(
                samples[
                    [
                        "price_prediction",
                        "observed_return",
                        "forecast_error",
                        "filled_quantity",
                        "quote_quantity",
                    ]
                ].to_numpy()
            ).all()
            or not (
                samples.forecast_error.to_numpy()
                == samples.observed_return.to_numpy()
                - samples.price_prediction.to_numpy()
            ).all()
            or not (
                (samples.filled_quantity >= 0)
                & (samples.filled_quantity <= samples.quote_quantity)
                & (samples.quote_quantity > 0)
            ).all()
            or samples.book_segment_id.eq("").any()
            or samples.trade_connection_id.eq("").any()
            or not samples.quote_quantity.eq(reference_config.order_quantity).all()
        ):
            raise ValueError("crypto_calibration_response_invalid")
        if (
            not events.event_kind.isin(("book", "trade", "boundary")).all()
            or not np.isfinite(events.quantity.to_numpy()).all()
            or events.quantity.lt(0).any()
        ):
            raise ValueError("crypto_calibration_event_invalid")
        for fold_id, fold in folds.items():
            sample = samples.loc[samples.fold_id.eq(fold_id)]
            if not sample.empty and not (
                sample.decision_at.between(
                    pd.Timestamp(fold["validation"]["start"]),
                    pd.Timestamp(fold["validation"]["end"]),
                ).all()
                and (
                    sample.prediction_end
                    == sample.decision_at
                    + pd.Timedelta(milliseconds=protocol["label_horizon_ms"])
                ).all()
                and (sample.label_available_at >= sample.prediction_end).all()
                and (
                    sample.label_available_at <= pd.Timestamp(fold["validation"]["end"])
                ).all()
            ):
                raise ValueError("crypto_calibration_outcome_not_mature")
            part = events.loc[events.fold_id.eq(fold_id)]
            headers = part.drop_duplicates("event_id")
            count = len(headers)
            if sample.empty:
                if count:
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                continue
            if (
                not np.array_equal(headers.event_id.to_numpy(), np.arange(count))
                or (part.event_id.diff().dropna() < 0).any()
                or (headers.event_at.diff().dropna() < pd.Timedelta(0)).any()
                or (headers.event_at > pd.Timestamp(fold["validation"]["end"])).any()
            ):
                raise ValueError("crypto_calibration_event_order_invalid")
            grouped = part.groupby("event_id", sort=False)
            if (
                not grouped[["event_at", "event_kind", "context_id"]]
                .nunique(dropna=False)
                .eq(1)
                .all()
                .all()
            ):
                raise ValueError("crypto_calibration_event_invalid")
            books = part.loc[part.event_kind.eq("book")]
            trades = part.loc[part.event_kind.eq("trade")]
            boundaries = part.loc[part.event_kind.eq("boundary")]
            if (
                not books.side.isin(("buy", "sell")).all()
                or books.context_id.eq("").any()
                or books.price_tick.le(0).any()
                or books.quantity.le(0).any()
                or not trades.side.isin(("buy", "sell")).all()
                or trades.context_id.eq("").any()
                or trades.price_tick.le(0).any()
                or not trades.level_index.eq(0).all()
                or not boundaries.side.eq("").all()
                or not boundaries.price_tick.eq(0).all()
                or not boundaries.quantity.eq(0).all()
                or not boundaries.level_index.eq(0).all()
                or not grouped.size()
                .loc[headers.loc[~headers.event_kind.eq("book"), "event_id"]]
                .eq(1)
                .all()
            ):
                raise ValueError("crypto_calibration_event_invalid")
            book_groups = books.groupby(["event_id", "side"], sort=False)
            if (
                not np.array_equal(
                    books.level_index.to_numpy(), book_groups.cumcount().to_numpy()
                )
                or not books.groupby("event_id").side.nunique().eq(2).all()
            ):
                raise ValueError("crypto_calibration_book_invalid")
            for side in ("buy", "sell"):
                levels = books.loc[books.side.eq(side)]
                identifiers = levels.event_id.to_numpy()
                ticks = levels.price_tick.to_numpy()
                same = identifiers[1:] == identifiers[:-1]
                invalid = (
                    ticks[1:] >= ticks[:-1]
                    if side == "buy"
                    else ticks[1:] <= ticks[:-1]
                )
                if (same & invalid).any():
                    raise ValueError("crypto_calibration_book_invalid")
            best = books.loc[books.level_index.eq(0)].pivot(
                index="event_id", columns="side", values="price_tick"
            )
            if not (best["buy"] < best["sell"]).all():
                raise ValueError("crypto_calibration_book_invalid")
            initial = sample.initial_book_event_id.to_numpy()
            starts, ends = sample.event_start.to_numpy(), sample.event_end.to_numpy()
            if (
                (initial < 0).any()
                or (initial >= count).any()
                or (initial >= starts).any()
                or (starts < 0).any()
                or (starts > ends).any()
                or (ends > count).any()
            ):
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            kinds = headers.event_kind.to_numpy()
            times = headers.event_at.to_numpy().view(np.int64)
            decisions = sample.decision_at.to_numpy().view(np.int64)
            expiries = sample.prediction_end.to_numpy().view(np.int64)
            states = np.flatnonzero(np.isin(kinds, ("book", "boundary")))
            if (
                not (kinds[initial] == "book").all()
                or not np.array_equal(
                    headers.context_id.to_numpy()[initial],
                    sample.book_segment_id.to_numpy(),
                )
                or (times[initial] > decisions).any()
                or not np.array_equal(
                    states[np.searchsorted(states, starts, side="left") - 1], initial
                )
                or (times[starts - 1] > decisions).any()
                or (times[starts[starts < count]] <= decisions[starts < count]).any()
                or (times[ends[ends > starts] - 1] >= expiries[ends > starts]).any()
                or (times[ends[ends < count]] < expiries[ends < count]).any()
            ):
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            covered = np.zeros(count + 1, dtype=np.int64)
            np.add.at(covered, starts, 1)
            np.add.at(covered, ends, -1)
            used = np.cumsum(covered[:-1]) > 0
            used[initial] = True
            if not used.all():
                raise ValueError("crypto_calibration_event_coordinates_invalid")
            paired_columns = [
                column
                for column in sample.columns
                if column not in ("side", "filled_quantity", "terminal_reason")
            ]
            buy = (
                sample.loc[sample.side.eq("buy"), paired_columns]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            sell = (
                sample.loc[sample.side.eq("sell"), paired_columns]
                .sort_values("sample_id")
                .reset_index(drop=True)
            )
            if not buy.equals(sell):
                raise ValueError("crypto_calibration_paired_path_invalid")
            event_data = {
                column: part[column].to_numpy()
                for column in part.columns
                if column != "fold_id"
            }
            event_data["event_at"] = event_data["event_at"].view(np.int64)
            offsets = np.r_[
                np.flatnonzero(np.r_[True, np.diff(event_data["event_id"]) != 0]),
                len(part),
            ]
            sample_data = {
                column: sample[column].to_numpy() for column in sample.columns
            }
            sample_data["decision_at"] = sample_data["decision_at"].view(np.int64)
            calibration = MakerCalibration(
                sample_data, event_data, offsets, resource_policy=policy
            )
            try:
                for index, row in enumerate(sample.itertuples(index=False)):
                    state = calibration._book(int(row.initial_book_event_id))
                    order = _MakerOrder(
                        0,
                        row.side,
                        (state.bids if row.side == "buy" else state.asks)[0][0],
                        row.quote_quantity,
                        0.0,
                        0.0,
                        row.book_segment_id,
                        row.trade_connection_id,
                        "1",
                        row.decision_at.value,
                        row.prediction_end.value,
                        row.decision_at.value + reference_config.latency_ms * 1000000,
                    )
                    remaining = row.quote_quantity
                    with closing_iterator(
                        calibration._response(
                            index,
                            state=state,
                            decision_ns=row.decision_at.value,
                            new_order=order,
                            active_order=None,
                            commands=(),
                            lifetime_ns=protocol["label_horizon_ms"] * 1000000,
                            replace_quantity=row.quote_quantity,
                        )
                    ) as response:
                        while True:
                            try:
                                _elapsed, quantity, _price, _new = next(response)
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
            finally:
                calibration.close()
        for row in summary.itertuples(index=False):
            sample = samples.loc[
                samples.fold_id.eq(row.fold_id) & samples.side.eq(row.side)
            ]
            fold = folds[row.fold_id]
            if (
                row.minimum_samples != minimum
                or row.sample_count != len(sample)
                or row.zero_fill_count != int(sample.filled_quantity.eq(0).sum())
                or row.post_only_rejection_count
                != int(sample.terminal_reason.eq("post_only_crossing_rejected").sum())
                or row.status != ("DEFINED" if len(sample) >= minimum else "UNDEFINED")
                or row.reason
                != (
                    "" if len(sample) >= minimum else "insufficient_calibration_samples"
                )
                or not np.array_equal(
                    np.sort(sample.sample_id.to_numpy()), np.arange(len(sample))
                )
            ):
                raise ValueError("crypto_calibration_summary_invalid")
