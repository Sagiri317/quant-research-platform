"""Persisted research inputs to the event-driven perpetual ledger."""

from array import array
from contextlib import ExitStack
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    MakerConfig,
    TakerConfig,
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
            prediction_source=lookup,
            primary_key=lookup.primary_key,
            input_columns=inputs,
            sample_filter_mode=metadata["sample_filter_mode"],
            test_start=pd.Timestamp(output_window.start).isoformat(),
            test_end=pd.Timestamp(output_window.end).isoformat(),
            resource_policy=resource_policy,
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
                    "segment_id": price.segment_id,
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
    sample_columns = (
        "fold_id",
        "side",
        "sample_id",
        "decision_at",
        "prediction_end",
        "label_available_at",
        "price_prediction",
        "observed_return",
        "forecast_error",
        "filled_quantity",
        "quote_quantity",
        "terminal_reason",
    )
    fill_columns = (
        "fold_id",
        "side",
        "sample_id",
        "fill_index",
        "elapsed_ns",
        "cumulative_quantity",
    )
    integers = {
        "sample_id",
        "decision_at",
        "prediction_end",
        "label_available_at",
        "fill_index",
        "elapsed_ns",
    }
    strings = {"fold_id", "side", "terminal_reason"}
    samples = {
        column: [] if column in strings else array("q" if column in integers else "d")
        for column in sample_columns
    }
    fills = {
        column: [] if column in strings else array("q" if column in integers else "d")
        for column in fill_columns
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
                    fills: array,
                    fold_id=fold["fold_id"],
                    counts=counts,
                    fill_buffers=fills,
                ):
                    lease.grow(512 + len(fills) * 64)
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
                    )
                    for column, value in zip(sample_columns, values, strict=True):
                        samples[column].append(value)
                    for index in range(0, len(fills), 2):
                        values = (
                            fold_id,
                            side,
                            sample_id,
                            index // 2,
                            int(fills[index]),
                            fills[index + 1],
                        )
                        for column, value in zip(fill_columns, values, strict=True):
                            fill_buffers[column].append(value)
                    counts[side][0] += 1
                    counts[side][1] += int(filled == 0)
                    counts[side][2] += int(
                        terminal_reason == "post_only_crossing_rejected"
                    )

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
                    }:
                        value = value.view("datetime64[ns]")
                values[column] = value
            return pd.DataFrame(values, columns=columns, copy=False)

        tables = {
            "crypto_maker_samples": frame(samples, sample_columns),
            "crypto_maker_fills": frame(fills, fill_columns),
            "crypto_maker_calibration": pd.DataFrame.from_records(summaries),
        }
        samples = fills = summaries = None
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
            )
            calibrations = {}
            with ExitStack() as reads:
                samples = reads.enter_context(
                    artifact.table("crypto_maker_samples.parquet")
                )
                fills = reads.enter_context(
                    artifact.table("crypto_maker_fills.parquet")
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
                        max(1, 256 * (len(samples) + len(fills))),
                    )
                )
                retained = 0
                for fold_id, _window in windows:
                    calibrations[fold_id] = {}
                    for side in ("buy", "sell"):
                        part = samples.loc[
                            samples.fold_id.eq(fold_id) & samples.side.eq(side),
                            ["sample_id", "filled_quantity", "forecast_error"],
                        ].sort_values("sample_id")
                        path = fills.loc[
                            fills.fold_id.eq(fold_id) & fills.side.eq(side),
                            ["sample_id", "elapsed_ns", "cumulative_quantity"],
                        ]
                        quantities = part.filled_quantity.to_numpy(copy=True)
                        errors = part.forecast_error.to_numpy(copy=True)
                        paths = path.to_numpy(dtype=np.float64, copy=True)
                        calibration = MakerCalibration(
                            quantities, errors, paths, resource_policy=resource_policy
                        )
                        resources.callback(calibration.close)
                        calibrations[fold_id][side] = calibration
                        retained += quantities.nbytes + errors.nbytes + paths.nbytes
                        del part, path, quantities, errors, paths
                state.shrink_to(max(1, retained))
                samples = fills = summary = None

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
                                if key != "prediction_score"
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
