"""Persisted research inputs to the event-driven perpetual ledger."""

from array import array
from contextlib import ExitStack
from functools import partial
from itertools import chain, groupby, islice, zip_longest
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    TradeEvent,
    validate_price_unit,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    MakerConfig,
    TakerConfig,
    _MakerOrder,
    _nonnegative_integer,
    _positive,
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
    CryptoMakerCandidate,
    CryptoMakerPlan,
    execution_windows,
    portfolio_candidates,
    validation_windows,
)
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.sources.binance import ids as binance_ids
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
            value = event
            if value.observed_time_ns > end_ns:
                break
            try:
                yield event
            except GeneratorExit:
                return


def _score_in_label_units(score, calibration):
    if pd.isna(score):
        return None
    value = float(score)
    return value if calibration is None else calibration[0] + calibration[1] * value


def _prediction_rows(
    *,
    layout,
    dataset,
    windows,
    model_id,
    asset_id,
    protocol_manifest,
    read_prediction_bundle,
    output_window,
):
    target = f"{dataset.data_id}_predictions"
    output_id = f"{model_id}_prediction"
    root = layout.target_root(target, Path("__missing_prediction__"))
    manifest, lookup = read_prediction_bundle(
        root / "outputs" / output_id,
        expected_output_id=output_id,
        expected_run_id=f"run_{output_id}",
        time_window=output_window,
    )
    metadata = manifest["metadata"]["prediction_metadata"]
    calibration = manifest["metadata"]["model_metadata"]["score_to_label"]
    inputs = tuple(metadata["input_columns"])
    scope = protocol_manifest["model_scopes"][model_id]
    rank_score = scope["model_algorithm"] in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }
    if rank_score != (calibration is not None):
        raise ValueError("crypto_prediction_score_units_invalid")
    assets = tuple(scope["evaluation_asset_ids"])
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
                    scoped = frame.loc[frame.asset_id.isin(assets)] if assets else frame
                    scoped = scoped.assign(fold_id=fold_id)
                    try:
                        yield scoped
                    except GeneratorExit:
                        return
                    del scoped

    joined = _iter_prediction_joined_chunks(
        frames(),
        prediction_lookup=lookup,
        model_algorithm=scope["model_algorithm"],
        input_columns=inputs,
        sample_filter_mode=metadata["sample_filter_mode"],
        test_start=pd.Timestamp(output_window.start).isoformat(),
        test_end=pd.Timestamp(output_window.end).isoformat(),
    )
    with closing_iterator(joined):
        for frame in joined:
            for row in frame.itertuples(index=False):
                if row.asset_id == asset_id:
                    _yield_value = {
                        "timestamp_ms": pd.Timestamp(row.timestamp).value // 1_000_000,
                        "asset_id": asset_id,
                        "book_segment_id": row.book_segment_id,
                        "trade_observation_id": row.trade_observation_id,
                        "segment_id": row.segment_id,
                        "fold_id": row.fold_id,
                        "prediction_score": _score_in_label_units(
                            row.prediction_score, calibration
                        ),
                    }
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return


def _market_coverage_check(coverage, trade_coverage, bounds):
    if not any(
        start <= bounds[0][0] and bounds[-1][1] < end
        for start, end in coverage.values()
    ):
        raise PortfolioTargetBlocked("crypto_execution_book_coverage_incomplete")
    if len(trade_coverage.covering(bounds[0][0], bounds[-1][1])) != 1:
        raise PortfolioTargetBlocked("crypto_execution_trade_coverage_incomplete")


def _funding_check(funding, expected_identity, bounds):
    if funding.identity != expected_identity:
        raise ValueError("crypto_execution_funding_identity_mismatch")
    cursor, final = bounds[0][0], bounds[-1][1]
    for start, end in funding.coverage:
        if end < cursor:
            continue
        if start > cursor:
            break
        if end >= final:
            return
        cursor = end + 1
    raise PortfolioTargetBlocked("binance_funding_coverage_incomplete")


def _maker_config(plan, asset_id):
    limits = plan["exposure_limits"][asset_id]
    execution = plan["execution"]
    maker = execution["maker"]
    return MakerConfig(
        order_quantity=limits["max_order_quantity"],
        max_position=limits["max_position_quantity"],
        maker_fee_bps=maker["maker_fee_bps"],
        inventory_target=maker["inventory_target"],
        inventory_skew=maker["inventory_skew"],
        markout_horizon_ms=execution["markout_horizon_ms"],
        latency_ms=maker["latency_ms"],
    )


class _EpisodeCursor:
    """Own one source lookahead and an explicitly handed-back book or forecast."""

    def __init__(self, source, key, *, sticky=False):
        self.source = iter(source)
        self.key = key
        self.pending = self.carried = None
        self.sticky = sticky
        self.exhausted = False

    def retain(self, value):
        self.carried = None if isinstance(value, BookBoundary) else value

    def _next(self):
        if self.pending is None and not self.exhausted:
            self.pending = next(self.source, None)
            self.exhausted = self.pending is None

    def window(self, start, stop, *, bootstrap=False):
        self._next()
        while self.pending is not None and self.key(self.pending) < start:
            value, self.pending = self.pending, None
            if bootstrap or self.sticky:
                self.retain(value)
            self._next()
        if self.carried is not None:
            value = self.carried
            if not self.sticky:
                self.carried = None
            yield value
        while self.pending is not None and self.key(self.pending) < stop:
            value, self.pending = self.pending, None
            if self.sticky:
                self.retain(value)
            yield value
            self._next()


def _episode_book_time(event):
    value = event
    return value.observed_time_ns


def _qualified_episodes(coverage, trade_coverage, funding, bounds):
    start, final = bounds[0][0], bounds[-1][1]
    if final == np.iinfo(np.int64).max:
        raise ValueError("crypto_execution_window_overflow")
    stop = final + 1
    book_intervals = iter(sorted(coverage.values()))
    trades = trade_coverage.intervals()

    def funding_intervals():
        left, right = funding.coverage[0]
        for begin, end in funding.coverage[1:]:
            if begin == right + 1:
                right = end
            else:
                yield left, right + 1
                left, right = begin, end
        yield left, right + 1

    funded_intervals = funding_intervals()
    bad_settlements = tuple(
        event.timestamp_ns
        for event in funding.events
        if event.rate is None or event.mark_price is None
    )
    with closing_iterator(trades), closing_iterator(funded_intervals):
        book, trade = next(book_intervals, None), next(trades, None)
        funded = next(funded_intervals, None)
        while book is not None and trade is not None and funded is not None:
            edge = min(book[1], trade[1], funded[1])
            left = max(start, book[0], trade[0], funded[0])
            right = min(stop, edge)
            if left < right:
                cuts = (stamp for stamp in bad_settlements if left <= stamp < right)
                for cut in cuts:
                    if left < cut:
                        _yield_value = left, cut, "TRUNCATED", ("funding_coverage_end",)
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
                    left = cut + 1
                if left < right:
                    reasons = []
                    if right == stop:
                        reasons.append("planned_oos_end")
                    if right == book[1]:
                        reasons.append("book_coverage_end")
                    if right == trade[1]:
                        reasons.append("trade_coverage_end")
                    if right == funded[1]:
                        reasons.append("funding_coverage_end")
                    _yield_value = (
                        left,
                        right,
                        "PLANNED_END" if right == stop else "TRUNCATED",
                        tuple(sorted(reasons)),
                    )
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return
            if book[1] == edge:
                book = next(book_intervals, None)
            if trade[1] == edge:
                trade = next(trades, None)
            if funded[1] == edge:
                funded = next(funded_intervals, None)


def _execute_episodes(
    *,
    books,
    trades,
    decisions,
    forecasts,
    funding,
    coverage,
    trade_coverage,
    bounds,
    resources,
    execute,
    on_episode,
):
    streams = {}
    for name, source, key, sticky in (
        ("books", books, _episode_book_time, False),
        ("decisions", decisions, lambda row: row["timestamp_ms"] * 1_000_000, False),
        ("forecasts", forecasts, lambda row: row.available_at_ns, True),
        ("trades", trades, lambda row: row.observed_time_ns, False),
    ):
        if source is None:
            continue
        iterator = resources.enter_context(closing_iterator(iter(source)))
        cursor = _EpisodeCursor(iterator, key, sticky=sticky)
        streams[name] = cursor
    count = 0
    for start, stop, termination, reasons in _qualified_episodes(
        coverage, trade_coverage, funding, bounds
    ):

        def produce(emit, start=start, stop=stop):
            kwargs = {
                "book_events": streams["books"].window(start, stop, bootstrap=True),
                "funding_events": tuple(
                    event
                    for event in funding.events
                    if start <= event.timestamp_ns < stop
                ),
                "funding_forecasts": streams["forecasts"].window(start, stop),
                "observation_window_ns": (start, stop),
                "on_ledger": emit,
                "on_terminal_book": streams["books"].retain,
            }
            if "trades" in streams:
                kwargs["trades"] = streams["trades"].window(start, stop)
                kwargs["quote_schedule"] = streams["decisions"].window(start, stop)
            else:
                kwargs["decisions"] = streams["decisions"].window(start, stop)
            result = execute(**kwargs)
            if result["selection_status"] == "BLOCKED":
                raise PortfolioTargetBlocked("portfolio_expected_funding_missing")
            if result["net_pnl"] is None:
                raise PortfolioTargetBlocked("portfolio_terminal_valuation_missing")
            return result

        on_episode(
            start_ns=start,
            stop_ns=stop,
            termination=termination,
            stop_reasons=reasons,
            produce=produce,
        )
        count += 1
    if not count:
        raise PortfolioTargetBlocked("crypto_execution_qualified_episode_missing")
    return {"episode_count": count}


def _execute_portfolio(
    *,
    execution_mode,
    execute,
    books,
    trades,
    decisions,
    funding,
    expected_funding_identity,
    coverage,
    trade_coverage,
    bounds,
    on_ledger,
    on_episode,
):
    if funding.identity != expected_funding_identity:
        raise ValueError("crypto_execution_funding_identity_mismatch")
    if execution_mode == "independent_segments":
        with closing_iterator(ExitStack()) as resources:
            return _execute_episodes(
                books=books,
                trades=trades,
                decisions=decisions,
                forecasts=funding.forecasts,
                funding=funding,
                coverage=coverage,
                trade_coverage=trade_coverage,
                bounds=bounds,
                resources=resources,
                execute=execute,
                on_episode=on_episode,
            )
    if execution_mode != "continuous":
        raise ValueError("crypto_execution_mode_invalid")
    _funding_check(funding, expected_funding_identity, bounds)
    _market_coverage_check(coverage, trade_coverage, bounds)
    kwargs = {
        "book_events": books,
        "funding_events": funding.events,
        "funding_forecasts": funding.forecasts,
        "on_ledger": on_ledger,
    }
    if trades is None:
        kwargs["decisions"] = decisions
    else:
        kwargs["trades"] = trades
        kwargs["quote_schedule"] = decisions
    result = execute(**kwargs)
    if result["selection_status"] == "BLOCKED":
        raise PortfolioTargetBlocked("portfolio_expected_funding_missing")
    if result["net_pnl"] is None:
        raise PortfolioTargetBlocked("portfolio_terminal_valuation_missing")
    return result


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
    execution_mode,
    on_ledger=None,
    on_episode=None,
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
    windows = execution_windows(
        plan, output_window, access_window=protocol_manifest["access_window"]
    )
    bounds = _window_bounds(windows)
    dataset = dataset_catalog.get(protocol_manifest["scenario_id"])
    panel = _panel(layout, upstream_identities)
    coverage = panel["producer_diagnostics"]["reconstruction"][symbol][
        "coverage_segments"
    ]
    with (
        open_run(raw_root) as run,
        read_funding(
            run.funding_roots(symbol, time_window=output_window), symbol=symbol
        ) as funding,
    ):
        if run.source_identity != panel["source_artifact_identity"]:
            raise ValueError("crypto_execution_recorder_identity_mismatch")
        trade_coverage = run.observation_coverage(
            market_type=binance_ids.FUTURES_MARKET_TYPE,
            data_kind="agg_trades",
            symbol=symbol,
        )

        def decisions():
            rows = _prediction_rows(
                layout=layout,
                dataset=dataset,
                windows=windows,
                model_id=candidate.model_id,
                asset_id=candidate.asset_id,
                protocol_manifest=protocol_manifest,
                read_prediction_bundle=read_prediction_bundle,
                output_window=output_window,
            )
            with closing_iterator(rows):
                for row in rows:
                    if row["prediction_score"] is not None:
                        try:
                            yield row
                        except GeneratorExit:
                            return

        execute = partial(
            simulate_taker,
            prediction_column="prediction_score",
            prediction_horizon_ms=protocol_manifest["model_scopes"][candidate.model_id][
                "label_horizon_ms"
            ],
            config=config,
            observed_coverage=coverage,
            capital=PortfolioCapital(
                candidate.candidate_id,
                plan["initial_capital"],
                plan["valuation_currency"],
                "linear_perpetual",
            ),
            asset_id=candidate.asset_id,
            decision_windows_ns=bounds,
        )
        return _execute_portfolio(
            execution_mode=execution_mode,
            execute=execute,
            books=_books_until(
                run,
                reconstruct_depth=reconstruct_depth,
                symbol=symbol,
                end_ns=bounds[-1][1],
            ),
            trades=None,
            decisions=decisions(),
            funding=funding,
            expected_funding_identity=expected_funding_identity,
            coverage=coverage,
            trade_coverage=trade_coverage,
            bounds=bounds,
            on_ledger=on_ledger,
            on_episode=on_episode,
        )


def _validation_rows(artifact, *, fold_id, model_id, asset_id):
    entry = next(
        entry
        for entry in artifact.models
        if entry["fold_id"] == fold_id and entry["model_id"] == model_id
    )
    metadata = entry["model_metadata"]
    calibration = (
        metadata["rank_parameters"]["label_calibration"]
        if metadata["model_algorithm"]
        in {
            "equal_weight_rank_ensemble",
            "ic_weighted_rank_ensemble",
        }
        else None
    )
    rows = artifact.validation_chunks(fold_id=fold_id, model_id=model_id)
    with closing_iterator(rows):
        for frame in rows:
            for row in frame.itertuples(index=False):
                if row.asset_id == asset_id:
                    _yield_value = row._replace(
                        prediction_score=_score_in_label_units(
                            row.prediction_score, calibration
                        )
                    )
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return


def _paired_validation(artifact, fold_id, asset_id, model_ids):
    streams = [
        _validation_rows(artifact, fold_id=fold_id, model_id=model, asset_id=asset_id)
        for model in model_ids
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
                _yield_value = {
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
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
            price, proxy = (next(stream, None) for stream in streams)
        # Validate unused mature rows too; unmatched feature eligibility is not a quote.
        for stream in streams:
            for _row in stream:
                pass


def produce_maker_calibration(
    *,
    on_table,
    layout,
    protocol_manifest,
    time_window,
    upstream_identities,
    raw_root,
    open_run,
    reconstruct_depth,
    read_trades,
    read_model_artifact,
    symbol,
    asset_id,
    model_ids,
    market_type,
):
    plan = protocol_manifest["research_plan"]
    calibration = plan["execution"]["maker"]
    panel = _panel(layout, upstream_identities)
    model_target = f"{protocol_manifest['scenario_id']}_models"
    models = read_model_artifact(
        layout.target_root(model_target, Path("__missing_models__")),
        expected_identity=upstream_identities[model_target],
        trial_layout=layout,
    )
    selected_folds = {fold for fold, _window in validation_windows(plan, time_window)}
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
    buffer_rows = dict.fromkeys(buffers, 0)
    written = set()

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
        on_table(name, output)
        written.add(name)
        buffers[name] = empty(columns_by_table[name])
        buffer_rows[name] = 0

    def append(name, rows, count):
        rows = iter(rows)
        while count:
            if buffer_rows[name] == DEFAULT_BATCH_ROWS:
                flush(name)
            take = min(count, DEFAULT_BATCH_ROWS - buffer_rows[name])
            batch = tuple(islice(rows, take))
            for column, values in zip(
                columns_by_table[name], zip(*batch, strict=True), strict=True
            ):
                buffers[name][column].extend(values)
            buffer_rows[name] += take
            count -= take

    with open_run(raw_root) as run:
        if run.source_identity != panel["source_artifact_identity"]:
            raise ValueError("crypto_execution_recorder_identity_mismatch")
        price_unit = run.market_metadata(market_type, symbol)["price_unit"]
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
                append("crypto_maker_samples", (values,), 1)
                counts[side][0] += 1
                counts[side][1] += int(filled == 0)
                counts[side][2] += int(terminal_reason == "post_only_crossing_rejected")

            def observe_event(*, event_id, event, fold_id=fold["fold_id"]):
                if isinstance(event, TradeEvent):
                    if event.visible_quantity is None:
                        raise PortfolioTargetBlocked(
                            "hft_trade_visible_quantity_unknown"
                        )
                    kind, context = "trade", event.observation_id
                    rows = (
                        (
                            "buy" if event.buyer_is_maker else "sell",
                            0,
                            event.price_index,
                            event.visible_quantity,
                        ),
                    )
                    count = 1
                elif isinstance(event, BookBoundary):
                    kind, context = "boundary", event.segment_id
                    rows = (("", 0, 0, 0.0),)
                    count = 1
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
                    count = len(event.bids) + len(event.asks)
                append(
                    "crypto_maker_events",
                    (
                        (
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
                        for side, level, tick, quantity in rows
                    ),
                    count,
                )

            cutoff = pd.Timestamp(fold["validation"]["end"]).value
            result = replay_maker_calibration(
                _books_until(
                    run,
                    reconstruct_depth=reconstruct_depth,
                    symbol=symbol,
                    end_ns=cutoff,
                ),
                read_trades(run, symbol=symbol, end_ms=cutoff // 1000000),
                _paired_validation(models, fold["fold_id"], asset_id, model_ids),
                order_quantity=calibration["order_quantity"],
                latency_ms=calibration["latency_ms"],
                observed_coverage=coverage,
                trade_coverage=trade_coverage,
                cutoff_ns=cutoff,
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
                        "price_unit": price_unit,
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

            on_table("crypto_maker_calibration", pd.DataFrame.from_records(summaries))
            for name in buffers:
                if buffer_rows[name]:
                    flush(name)
    for name in buffers:
        if buffer_rows[name] or name not in written:
            flush(name)


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
    execution_mode,
    on_ledger=None,
    on_episode=None,
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
    windows = execution_windows(
        plan, output_window, access_window=protocol_manifest["access_window"]
    )
    bounds = _window_bounds(windows)
    config = _maker_config(plan, candidate.asset_id)
    panel = _panel(layout, upstream_identities)
    dataset = dataset_catalog.get(protocol_manifest["scenario_id"])
    coverage = panel["producer_diagnostics"]["reconstruction"][symbol][
        "coverage_segments"
    ]
    with closing_iterator(ExitStack()) as resources:
        calibrations = None
        if candidate.calibrated:
            target = candidate.calibration_target_id
            artifact = read_statistics_artifact(
                layout.target_root(target, Path("__missing_calibration__")),
                expected_identity=upstream_identities[target],
                validate_tables=validate_maker_calibration,
            )
            with artifact.table("crypto_maker_calibration.parquet") as summary:
                price_unit = summary.price_unit.iloc[0]
                required = {fold_id for fold_id, _window in windows}
                selected = summary.loc[summary.fold_id.isin(required)]
                if (
                    set(zip(selected.fold_id, selected.side, strict=True))
                    != {(fold, side) for fold in required for side in ("buy", "sell")}
                    or not selected.status.eq("DEFINED").all()
                ):
                    raise PortfolioTargetBlocked(
                        "maker_calibration_insufficient_samples"
                    )
                selected = None
            resources.callback(artifact._verify)
            calibrations = {}
            for fold_id, _window in windows:
                calibration = MakerCalibration(
                    partial(_calibration_samples, artifact, fold_id),
                    partial(_calibration_events, artifact, fold_id),
                    price_unit=price_unit,
                )
                calibrations[fold_id] = calibration

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
                        read_prediction_bundle=read_prediction_bundle,
                        output_window=output_window,
                    )
                    for model in candidate.model_ids
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
                        _yield_value = {
                            **{
                                key: value
                                for key, value in price.items()
                                if key not in {"prediction_score", "segment_id"}
                            },
                            "price_prediction": price["prediction_score"],
                            "proxy_prediction": proxy["prediction_score"],
                        }
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
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
                                    _yield_value = {
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
                                    try:
                                        yield _yield_value
                                    except GeneratorExit:
                                        return

        with (
            open_run(raw_root) as run,
            read_funding(
                run.funding_roots(symbol, time_window=output_window), symbol=symbol
            ) as funding,
        ):
            if run.source_identity != panel["source_artifact_identity"]:
                raise ValueError("crypto_execution_recorder_identity_mismatch")
            trade_coverage = run.observation_coverage(
                market_type=market_type, data_kind="agg_trades", symbol=symbol
            )
            execute = partial(
                simulate_maker,
                config=config,
                observed_coverage=coverage,
                trade_coverage=trade_coverage,
                capital=PortfolioCapital(
                    candidate.candidate_id,
                    plan["initial_capital"],
                    plan["valuation_currency"],
                    "linear_perpetual",
                ),
                asset_id=candidate.asset_id,
                calibrations=calibrations,
                decision_windows_ns=bounds,
            )
            result = _execute_portfolio(
                execution_mode=execution_mode,
                execute=execute,
                books=_books_until(
                    run,
                    reconstruct_depth=reconstruct_depth,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                ),
                trades=read_trades(
                    run, symbol=symbol, end_ms=bounds[-1][1] // 1_000_000
                ),
                decisions=quotes(),
                funding=funding,
                expected_funding_identity=expected_funding_identity,
                coverage=coverage,
                trade_coverage=trade_coverage,
                bounds=bounds,
                on_ledger=on_ledger,
                on_episode=on_episode,
            )
            return result


def _calibration_rows(artifact, member, filters=()):
    with closing_iterator(artifact._batches(member, filters=list(filters))) as batches:
        for frame in batches:
            for _yield_value in frame.itertuples(index=False):
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


def _calibration_samples(artifact, fold_id):
    with closing_iterator(
        _calibration_rows(
            artifact,
            "crypto_maker_samples.parquet",
            (("fold_id", "==", fold_id),),
        )
    ) as rows:
        for row in rows:
            sample = row._asdict()
            for column in ("decision_at", "prediction_end", "label_available_at"):
                sample[column] = getattr(row, column).value
            try:
                yield sample
            except GeneratorExit:
                return


def _calibration_events(artifact, fold_id):
    with closing_iterator(
        _calibration_rows(
            artifact, "crypto_maker_events.parquet", (("fold_id", "==", fold_id),)
        )
    ) as rows:
        for event_id, group in groupby(rows, key=lambda row: row.event_id):
            first = next(group)
            if first.event_kind == "book":
                bids, asks = [], []
                for row in chain((first,), group):
                    (bids if row.side == "buy" else asks).append(
                        (int(row.price_index), float(row.quantity))
                    )
                _yield_value = SimpleNamespace(
                    kind="book",
                    event_id=event_id,
                    event_at=first.event_at.value,
                    segment_id=first.context_id,
                    bids=tuple(bids),
                    asks=tuple(asks),
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
            else:
                _yield_value = SimpleNamespace(
                    kind=first.event_kind,
                    event_id=event_id,
                    event_at=first.event_at.value,
                    segment_id=first.context_id,
                    observation_id=first.context_id,
                    price_index=int(first.price_index),
                    visible_quantity=float(first.quantity),
                    buyer_is_maker=first.side == "buy",
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


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
            _yield_value = SimpleNamespace(
                event_id=event_id,
                event_at=stamp,
                kind=first.event_kind,
                context=first.context_id,
                row_count=row_count,
            )
            try:
                yield _yield_value
            except GeneratorExit:
                return


def validate_maker_calibration(artifact, envelope):
    protocol = envelope["semantic_metadata"]["target_contract"]["protocol"]
    plan = protocol["research_plan"]
    minimum = plan["execution"]["maker"]["min_calibration_samples"]
    output = envelope["research_window"]
    folds = {
        fold["fold_id"]: fold
        for fold in plan["folds"]
        if pd.Timestamp(fold["validation"]["start"]) <= pd.Timestamp(output["end"])
        and pd.Timestamp(fold["validation"]["end"]) >= pd.Timestamp(output["start"])
    }
    candidates = portfolio_candidates(protocol["portfolio_candidates"])
    if (
        len(candidates) != 1
        or not isinstance(candidates[0], CryptoMakerCandidate)
        or not candidates[0].calibrated
        or candidates[0].calibration_target_id != envelope["logical_artifact_id"]
    ):
        raise ValueError("crypto_calibration_candidate_invalid")
    reference = plan["execution"]["maker"]
    _positive(reference["order_quantity"], "hft_maker_order_quantity_invalid")
    _nonnegative_integer(reference["latency_ms"], "hft_maker_latency_invalid")
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        price_unit = validate_price_unit(summary.price_unit.iloc[0])
        if not summary.price_unit.eq(price_unit).all():
            raise ValueError("crypto_calibration_price_unit_invalid")
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
        with closing_iterator(ExitStack()) as reads:
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
            model = MakerCalibration(
                partial(_calibration_samples, artifact, fold_id),
                partial(_calibration_events, artifact, fold_id),
                price_unit=price_unit,
            )
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
                        or row.quote_quantity != reference["order_quantity"]
                        or not row.book_segment_id
                        or not row.trade_observation_id
                    ):
                        raise ValueError("crypto_calibration_response_invalid")
                decision = buy.decision_at.value
                expiry = buy.prediction_end.value
                if not (
                    beginning <= decision <= cutoff
                    and expiry == decision + CryptoMakerPlan.quote_horizon_ms * 1000000
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
                    sample = row._asdict()
                    sample["decision_at"] = decision
                    state = next(model.events(initial, initial + 1))
                    if (row.initial_bid_index, row.initial_ask_index) != (
                        state.bids[0][0],
                        state.asks[0][0],
                    ):
                        raise ValueError("crypto_calibration_origin_invalid")
                    order = _MakerOrder(
                        0,
                        row.side,
                        (state.bids if row.side == "buy" else state.asks)[0][0],
                        row.quote_quantity,
                        0.0,
                        0.0,
                        row.book_segment_id,
                        row.trade_observation_id,
                        price_unit,
                        decision,
                        expiry,
                        decision + reference["latency_ms"] * 1000000,
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
                            lifetime_ns=CryptoMakerPlan.quote_horizon_ms * 1000000,
                            replace_quantity=row.quote_quantity,
                            events=model.events(
                                int(sample["event_start"]), int(sample["event_end"])
                            ),
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
