"""Persisted research inputs to the event-driven perpetual ledger."""

import json
from array import array
from collections import deque
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
from functools import partial
from itertools import chain, groupby, islice, zip_longest
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import (
    prediction_model_metadata,
    read_predictions,
)
from quant_research_platform.artifacts.statistics import STATISTICS_TABLE_COLUMNS
from quant_research_platform.contracts.experiments import (
    legal_training_mask,
    training_availability_columns,
)
from quant_research_platform.contracts.filtering import required_filter_columns
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.models import (
    _model_spec,
    validate_model_input_roles,
)
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    TradeEvent,
    validate_price_unit,
)
from quant_research_platform.contracts.portfolios import (
    PORTFOLIO_CASH_ASSET_ID,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    MAKER_QUOTE_HORIZON_MS,
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
from quant_research_platform.markets.implementations.crypto.ids import (
    canonical_crypto_asset_id,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.features.transforms import (
    model_input_validity_columns,
)
from quant_research_platform.research.predictions.outputs import prediction_rows
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_EXECUTION_ASSET_IDS,
    HFT_EXECUTION_CONTEXT_COLUMNS,
    label_horizon_ms,
    research_chronology,
)
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.sources.binance import ids as binance_ids
from quant_research_platform.sources.binance.funding import read_funding_observations
from quant_research_platform.sources.binance.replay import (
    ReconstructionStats,
    iter_agg_trades,
    iter_reconstructed_depth,
)
from quant_research_platform.sources.market import open_market_source
from quant_research_platform.storage.atomic import reject_if_exists
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence


def _window_bounds(windows):
    if not windows:
        raise PortfolioTargetBlocked("crypto_execution_oos_window_missing")
    return tuple(
        (pd.Timestamp(window.start).value, pd.Timestamp(window.end).value)
        for _fold, window in windows
    )


def _books_until(run, *, reconstruct_depth, symbol, end_ns, stats=None):
    source = reconstruct_depth(run, symbol=symbol, end_ns=end_ns, stats=stats)
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


def _window(value):
    value = _strict_config(value, {"start", "end"})
    window = TimeWindow(**value)
    if any(pd.isna(pd.Timestamp(bound)) for bound in (window.start, window.end)):
        raise ValueError("crypto_operation_window_invalid")
    return window


def _prediction_evidence(
    predictions_path,
    *,
    window,
):
    predictions, metadata = read_predictions(
        Path(predictions_path),
        columns=("timestamp", "asset_id", "prediction_score"),
        time_window=window,
    )
    facts = metadata["model_facts"]
    if facts["market_id"] != "crypto":
        raise ValueError("crypto_prediction_market_invalid")
    target = facts["model"]["target_column"]
    unit = (
        "decimal_return"
        if target.startswith("future_mid_return_")
        else "quantity_ratio"
        if target == "maker_bid_sell_pressure_proxy_500ms"
        else None
    )
    if unit is not None and facts["label_unit"] != unit:
        raise ValueError("crypto_prediction_label_unit_invalid")
    cutoff = pd.Timestamp(facts["fit_cutoff"])
    if cutoff >= pd.Timestamp(window.start):
        raise ValueError("crypto_prediction_model_after_decision")
    declared = TimeWindow(
        metadata["prediction_window"]["start"],
        metadata["prediction_window"]["end"] or metadata["effective_prediction_end"],
    )
    if window.start < declared.start or window.end > declared.end:
        raise ValueError("crypto_prediction_window_coverage_invalid")
    return (
        prediction_model_metadata(metadata),
        {**facts, "prediction_asset_ids": metadata.get("asset_ids")},
        predictions,
    )


def _prediction_rows(
    *,
    inputs,
    evidence,
    window,
    fold_id,
    asset_id,
    sample_role="test",
    input_spec=None,
    label_cutoff=None,
):
    metadata, facts, predictions = evidence
    assets = facts.get("prediction_asset_ids")
    if assets is not None and asset_id not in assets:
        raise ValueError("crypto_prediction_asset_outside_scope")
    calibration = metadata.get("rank_parameters", {}).get("label_calibration")
    is_rank = metadata["model_algorithm"] in {
        "equal_weight_rank_ensemble",
        "ic_weighted_rank_ensemble",
    }
    if is_rank != (calibration is not None):
        raise ValueError("crypto_prediction_score_units_invalid")
    target = facts["model"]["target_column"]
    options = _dataset_input_options(inputs)
    if input_spec is not None:
        options.update(
            _dataset_input_options(
                {"path": options["dataset_root"], "spec": input_spec}
            )
        )
    session = open_dataset_session(**options).filtered_view(window)
    availability = session.spec.get("label_availability_columns", {}).get(
        target, training_availability_columns(target)[0]
    )
    columns = tuple(
        dict.fromkeys(
            (
                "timestamp",
                "asset_id",
                *metadata["input_columns"],
                *model_input_validity_columns(
                    tuple(metadata["input_columns"]),
                    metadata["model_algorithm"],
                    session.spec["column_roles"],
                ),
                *required_filter_columns(metadata["sample_filter_mode"]),
                *HFT_EXECUTION_CONTEXT_COLUMNS,
                *(
                    (target, availability, "eligible")
                    if sample_role == "validation"
                    else ()
                ),
            )
        )
    )
    keys = pd.MultiIndex.from_frame(predictions[["timestamp", "asset_id"]])
    matched = np.zeros(len(predictions), dtype=bool)
    scores = predictions.prediction_score.to_numpy()
    business_columns = {
        *metadata["input_columns"],
        *((target,) if sample_role == "validation" else ()),
    }
    if options.get("columns") is not None and business_columns - set(
        options["columns"]
    ):
        raise ValueError("crypto_prediction_required_columns_excluded")
    session = session.projected_view(columns)
    model_spec = _model_spec(facts["model"])
    validate_model_input_roles(
        session.spec,
        model_spec,
        market_id="crypto",
        include_target=sample_role == "validation",
    )
    if sample_role == "validation":
        declared_unit = session.spec.get("column_units", {}).get(target)
        if declared_unit is not None and declared_unit != facts["label_unit"]:
            raise ValueError("crypto_prediction_label_unit_invalid")
    chunks = iter_dataset_chunks(session)
    with closing_iterator(chunks):
        for frame in chunks:
            required = prediction_rows(
                frame, metadata, window, declared_columns=session.spec["column_roles"]
            )
            if assets is not None:
                required = required.loc[required.asset_id.isin(assets)]
            required_keys = pd.MultiIndex.from_frame(
                required[["timestamp", "asset_id"]]
            )
            positions = keys.get_indexer(required_keys)
            if (positions < 0).any():
                raise ValueError("missing_required_prediction_keys")
            if matched[positions].any():
                raise ValueError("duplicate_prediction_input_keys")
            matched[positions] = True
            required = required.assign(
                prediction_score=scores[positions], fold_id=fold_id
            )
            if sample_role == "validation":
                mature = legal_training_mask(
                    required,
                    fitting_cutoff=window.end if label_cutoff is None else label_cutoff,
                    availability_column=availability,
                    target_column=target,
                )
                required = required.loc[mature].assign(
                    label=lambda data: data[target],
                    label_available_at=lambda data: pd.to_datetime(data[availability]),
                )
            for row in required.itertuples(index=False):
                if row.asset_id == asset_id:
                    score = _score_in_label_units(row.prediction_score, calibration)
                    if sample_role == "validation":
                        yield row._replace(prediction_score=score)
                    else:
                        yield {
                            "timestamp_ms": pd.Timestamp(row.timestamp).value
                            // 1_000_000,
                            "asset_id": asset_id,
                            "fold_id": fold_id,
                            **{
                                name: getattr(row, name)
                                for name in HFT_EXECUTION_CONTEXT_COLUMNS
                            },
                            "prediction_score": score,
                        }
    if not matched.all():
        raise ValueError("misaligned_persisted_prediction_keys")


def _execution_coverage(run, *, reconstruct_depth, symbol, end_ns):
    """Use the selected execution source's reconstructed half-open coverage."""
    stats = ReconstructionStats()
    events = reconstruct_depth(
        run, symbol=symbol, end_ns=end_ns, stats=stats, output_levels=0
    )
    with closing_iterator(events):
        for _ in events:
            pass
    return stats.coverage_segments


class _EpisodeCursor:
    """Own source lookahead and causal state for the next declared episode."""

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
            if not self.sticky and not bootstrap:
                self.carried = None
            yield value
        while self.pending is not None and self.key(self.pending) < stop:
            value, self.pending = self.pending, None
            if self.sticky or bootstrap:
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
    books = iter(sorted(coverage.values()))
    trades = (
        trade_coverage.intervals()
        if trade_coverage is not None
        else iter(((start, stop),))
    )
    funding_cuts = set()
    previous_end = None
    for begin, end in funding.coverage:
        if previous_end is not None and begin == previous_end + 1:
            funding_cuts.discard(begin)
        else:
            funding_cuts.add(begin)
        if end < np.iinfo(np.int64).max:
            funding_cuts.add(end + 1)
        previous_end = end
    for event in funding.events:
        if event.rate is None or event.mark_price is None and event.rate != 0:
            funding_cuts.update((event.timestamp_ns, event.timestamp_ns + 1))
    with closing_iterator(trades):
        book, trade = next(books, None), next(trades, None)
        while book is not None and trade is not None:
            edge = min(book[1], trade[1])
            left, right = max(start, book[0], trade[0]), min(stop, edge)
            if left < right:
                reasons = []
                if right == stop:
                    reasons.append("planned_oos_end")
                if right == book[1]:
                    reasons.append("book_coverage_end")
                if trade_coverage is not None and right == trade[1]:
                    reasons.append("trade_coverage_end")
                for cut in sorted(
                    value for value in funding_cuts if left < value < right
                ):
                    yield left, cut, "TRUNCATED", ("funding_coverage_end",)
                    left = cut
                if right in funding_cuts:
                    reasons.append("funding_coverage_end")
                yield (
                    left,
                    right,
                    "PLANNED_END" if right == stop else "TRUNCATED",
                    tuple(sorted(reasons)),
                )
            if book[1] == edge:
                book = next(books, None)
            if trade[1] == edge:
                trade = next(trades, None)


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
                "funding_coverage": funding.coverage,
                "observation_window_ns": (start, stop),
                "on_ledger": emit,
            }
            if "trades" in streams:
                kwargs["trades"] = streams["trades"].window(start, stop)
                kwargs["quote_schedule"] = streams["decisions"].window(start, stop)
            else:
                kwargs["decisions"] = streams["decisions"].window(start, stop)
            return execute(**kwargs)

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
    coverage,
    trade_coverage,
    bounds,
    on_ledger,
    on_episode,
):
    simulate = execute

    def execute(**kwargs):
        emit = kwargs["on_ledger"]
        first = True

        def qualified(batch):
            nonlocal first
            if first:
                batch = replace(
                    batch,
                    diagnostics=(
                        *batch.diagnostics,
                        PortfolioDiagnostic(
                            PORTFOLIO_CASH_ASSET_ID,
                            None,
                            "funding_qualification",
                            text=json.dumps(
                                {"sources": funding.settlement_qualification},
                                sort_keys=True,
                                allow_nan=False,
                            ),
                        ),
                    ),
                )
                first = False
            emit(batch)

        return simulate(
            **{**kwargs, "on_ledger": qualified if emit is not None else None}
        )

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
    kwargs = {
        "book_events": books,
        "funding_events": funding.events,
        "funding_forecasts": funding.forecasts,
        "funding_coverage": funding.coverage,
        "on_ledger": on_ledger,
    }
    if trades is None:
        kwargs["decisions"] = decisions
    else:
        kwargs["trades"] = trades
        kwargs["quote_schedule"] = decisions
    result = execute(**kwargs)
    return result


_BACKTEST_KEYS = {
    "output",
    "inputs",
    "panel",
    "source",
    "asset_id",
    "symbol",
    "prediction_window",
    "execution_mode",
    "capital",
    "execution",
}


def _execution_asset(config):
    if (
        config["asset_id"] not in HFT_EXECUTION_ASSET_IDS
        or canonical_crypto_asset_id(config["symbol"]) != config["asset_id"]
    ):
        raise ValueError("crypto_execution_asset_invalid")


def _backtest_inputs(config):
    _execution_asset(config)
    if "funding_roots" in config:
        roots = config["funding_roots"]
        if (
            type(roots) is not list
            or not roots
            or any(type(root) is not str or not root for root in roots)
            or len(set(roots)) != len(roots)
        ):
            raise ValueError("crypto_funding_roots_invalid")
    reject_if_exists(Path(config["output"]))
    window = _window(config["prediction_window"])
    windows = (("direct", window),)
    capital = PortfolioCapital(
        **_strict_config(
            config["capital"],
            {
                "candidate_id",
                "initial_capital",
                "valuation_currency",
                "valuation_method",
            },
        )
    )
    if (
        capital.valuation_method != "linear_perpetual"
        or capital.valuation_currency != "USDT"
    ):
        raise ValueError("crypto_valuation_method_invalid")
    if config["execution_mode"] not in {"continuous", "independent_segments"}:
        raise ValueError("crypto_execution_mode_invalid")
    if any(
        type(config[key]) is not str or not config[key]
        for key in ("symbol", "asset_id")
    ):
        raise ValueError("crypto_execution_asset_invalid")
    panel = read_dataset_manifest_evidence(Path(config["panel"]))
    return window, windows, capital, panel


def _source_coordinates(run, panel, symbol):
    try:
        declared = panel["source_evidence"]["market_metadata"][symbol]
        actual = run.market_metadata(binance_ids.FUTURES_MARKET_TYPE, symbol)
        if (
            Fraction(validate_price_unit(declared["price_unit"]))
            != Fraction(validate_price_unit(actual["price_unit"]))
            or declared["snapshot_limit"] != actual["snapshot_limit"]
        ):
            raise ValueError("crypto_execution_source_coordinates_mismatch")
    except (KeyError, TypeError) as error:
        raise ValueError("crypto_execution_source_coordinates_missing") from error


def _write_backtest(config, capital, produce):
    from quant_research_platform.artifacts.crypto_hft import (
        write_crypto_episode_artifact,
    )
    from quant_research_platform.artifacts.portfolio_ledger import (
        write_portfolio_ledger_artifact,
    )

    metadata = {
        key: value
        for key, value in config.items()
        if key not in {"output", "capital", "prediction_window"}
    }
    metadata.update(
        window=config["prediction_window"],
        sample_role="prediction",
        execution_style="taker" if "predictions" in config else "maker",
    )
    if config["execution_mode"] == "independent_segments":
        return write_crypto_episode_artifact(
            Path(config["output"]),
            capital=capital,
            metadata=metadata,
            produce=lambda emit: produce(on_episode=emit),
        )
    return write_portfolio_ledger_artifact(
        Path(config["output"]),
        capital=capital,
        metadata=metadata,
        produce=lambda emit: produce(on_ledger=emit),
    )


def backtest_taker(config):
    """Backtest explicit frozen predictions and native market inputs to a new ledger."""
    config = _strict_config(
        config,
        _BACKTEST_KEYS | {"predictions"},
        {"archive_paths", "input_spec", "chronology", "funding_roots"},
    )
    window, windows, capital, panel = _backtest_inputs(config)
    execution = TakerConfig(
        **_strict_config(config["execution"], TakerConfig.__dataclass_fields__)
    )
    evidence = _prediction_evidence(
        config["predictions"],
        window=window,
    )
    _metadata, facts, _predictions = evidence
    target = facts["model"]["target_column"]
    if not target.startswith("future_mid_return_"):
        raise ValueError("crypto_taker_return_target_required")
    horizon = label_horizon_ms(target)
    bounds = _window_bounds(windows)
    symbol = config["symbol"]
    reconstruct = partial(
        iter_reconstructed_depth, market_type=binance_ids.FUTURES_MARKET_TYPE
    )
    if execution.latency_ms >= horizon:
        reconstruct = partial(reconstruct, output_levels=1)

    def produce(*, on_ledger=None, on_episode=None):
        with (
            open_market_source(
                Path(config["source"]),
                archive_paths=config.get("archive_paths"),
                time_window=TimeWindow(
                    "1970-01-01T00:00:00",
                    pd.Timestamp(max(end for _, end in bounds)).isoformat(),
                ),
            ) as run,
            read_funding_observations(
                tuple(config["funding_roots"])
                if "funding_roots" in config
                else run.funding_roots(symbol, time_window=window),
                symbol=symbol,
            ) as funding,
        ):
            _source_coordinates(run, panel, symbol)
            source_stats = ReconstructionStats()
            coverage = (
                _execution_coverage(
                    run,
                    reconstruct_depth=reconstruct,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                )
                if config["execution_mode"] == "independent_segments"
                else source_stats.coverage_segments
            )
            return _execute_portfolio(
                execution_mode=config["execution_mode"],
                execute=partial(
                    simulate_taker,
                    prediction_column="prediction_score",
                    prediction_horizon_ms=horizon,
                    config=execution,
                    observed_coverage=coverage,
                    capital=capital,
                    asset_id=config["asset_id"],
                    decision_windows_ns=bounds,
                ),
                books=_books_until(
                    run,
                    reconstruct_depth=reconstruct,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                    stats=source_stats,
                ),
                trades=None,
                decisions=_prediction_rows(
                    inputs=config["inputs"],
                    input_spec=config.get("input_spec"),
                    evidence=evidence,
                    window=window,
                    fold_id=windows[0][0],
                    asset_id=config["asset_id"],
                ),
                funding=funding,
                coverage=coverage,
                trade_coverage=None,
                bounds=bounds,
                on_ledger=on_ledger,
                on_episode=on_episode,
            )

    return _write_backtest(config, capital, produce)


def _paired_validation(streams, *, asset_id):
    streams = [iter(stream) for stream in streams]
    with closing_iterator(streams[0]), closing_iterator(streams[1]):
        price, proxy = (next(stream, None) for stream in streams)
        while price is not None and proxy is not None:
            if price.timestamp < proxy.timestamp:
                price = next(streams[0], None)
                continue
            if proxy.timestamp < price.timestamp:
                proxy = next(streams[1], None)
                continue
            if all(
                isinstance(getattr(row, column), str) and getattr(row, column)
                for row in (price, proxy)
                for column in HFT_EXECUTION_CONTEXT_COLUMNS
            ):
                if any(
                    getattr(price, column) != getattr(proxy, column)
                    for column in HFT_EXECUTION_CONTEXT_COLUMNS
                ):
                    raise ValueError("maker_calibration_context_misaligned")
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


def _calibration_windows(config):
    """Resolve explicit windows and cutoffs, using chronology only for defaults."""
    chronology = config.get("chronology")
    entries = config.get("models")
    needs_defaults = (
        entries is None
        or type(entries) is list
        and any(
            type(item) is dict and not {"window", "cutoff"} <= item.keys()
            for item in entries
        )
    )
    folds = (
        {fold.fold_id: fold for fold in research_chronology(chronology)[0]}
        if chronology is not None and needs_defaults
        else {}
    )
    if "models" not in config:
        entries = [{"fold_id": fold_id} for fold_id in folds]
    if (
        type(entries) is not list
        or not entries
        or any(type(item) is not dict for item in entries)
        or any(
            type(item.get("fold_id")) is not str or not item["fold_id"]
            for item in entries
        )
        or len({item["fold_id"] for item in entries}) != len(entries)
    ):
        raise ValueError("crypto_calibration_fold_models_invalid")
    windows = {}
    for item in entries:
        fold_id = item["fold_id"]
        validation = folds[fold_id].validation if fold_id in folds else None
        if validation is None and not {"window", "cutoff"} <= item.keys():
            raise ValueError("crypto_calibration_window_cutoff_required")
        window = _window(item["window"]) if "window" in item else validation
        cutoff = item.get("cutoff", validation.end if validation else None)
        cutoff = pd.Timestamp(TimeWindow(cutoff, cutoff).end)
        if pd.isna(cutoff) or cutoff < pd.Timestamp(window.end):
            raise ValueError("crypto_calibration_window_invalid")
        windows[fold_id] = (window, cutoff)
    return windows


def _produce_maker_calibration(*, on_table, config, model_evidence):
    reference = config["reference"]
    panel = read_dataset_manifest_evidence(Path(config["panel"]))
    windows = _calibration_windows(config)
    symbol, asset_id = config["symbol"], config["asset_id"]
    market_type = binance_ids.FUTURES_MARKET_TYPE
    reconstruct_depth = partial(iter_reconstructed_depth, market_type=market_type)
    read_trades = partial(iter_agg_trades, market_type=market_type)
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
        "known_price_bound",
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

    with open_market_source(
        Path(config["source"]),
        archive_paths=config.get("archive_paths"),
        time_window=TimeWindow(
            "1970-01-01T00:00:00",
            max(cutoff for _, cutoff in windows.values()).isoformat(),
        ),
    ) as run:
        _source_coordinates(run, panel, symbol)
        price_unit = run.market_metadata(market_type, symbol)["price_unit"]
        for fold_id, (window, cutoff_at) in windows.items():
            trade_coverage = run.observation_coverage(
                market_type=market_type, data_kind="agg_trades", symbol=symbol
            )
            model_inputs = next(
                item for item in config["models"] if item["fold_id"] == fold_id
            )
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
                fold_id=fold_id,
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

            def observe_event(*, event_id, event, fold_id=fold_id):
                if isinstance(event, TradeEvent):
                    kind, context = "trade", event.observation_id
                    rows = (
                        (
                            "buy" if event.buyer_is_maker else "sell",
                            0,
                            event.price_index,
                            np.nan
                            if event.visible_quantity is None
                            else event.visible_quantity,
                            0,
                        ),
                    )
                    count = 1
                elif isinstance(event, BookBoundary):
                    kind, context = "boundary", event.segment_id
                    rows = (("", 0, 0, 0.0, 0),)
                    count = 1
                else:
                    kind, context = "book", event.segment_id
                    rows = (
                        (
                            side,
                            index,
                            tick,
                            quantity,
                            levels[-1][0] if bound is None else bound,
                        )
                        for side, levels, bound in (
                            (
                                "buy",
                                event.bids,
                                getattr(event, "known_bid_floor_tick", None),
                            ),
                            (
                                "sell",
                                event.asks,
                                getattr(event, "known_ask_ceiling_tick", None),
                            ),
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
                            bound,
                        )
                        for side, level, tick, quantity, bound in rows
                    ),
                    count,
                )

            cutoff = cutoff_at.value
            source_stats = ReconstructionStats()
            result = replay_maker_calibration(
                _books_until(
                    run,
                    reconstruct_depth=reconstruct_depth,
                    symbol=symbol,
                    end_ns=cutoff,
                    stats=source_stats,
                ),
                read_trades(run, symbol=symbol, end_ns=cutoff),
                _paired_validation(
                    [
                        _prediction_rows(
                            inputs=model_inputs["inputs"],
                            input_spec=model_inputs.get("input_spec"),
                            evidence=model_evidence[fold_id][role],
                            window=window,
                            label_cutoff=cutoff_at,
                            fold_id=fold_id,
                            asset_id=asset_id,
                            sample_role="validation",
                        )
                        for role in ("price", "proxy")
                    ],
                    asset_id=asset_id,
                ),
                order_quantity=reference["order_quantity"],
                latency_ms=reference["latency_ms"],
                observed_coverage=source_stats.coverage_segments,
                trade_coverage=trade_coverage,
                cutoff_ns=cutoff,
                on_sample=observe,
                on_event=observe_event,
            )
            minimum = reference["min_calibration_samples"]
            summaries = []
            for side, (count, zero, rejected) in counts.items():
                summaries.append(
                    {
                        "fold_id": fold_id,
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


def calibrate_maker(config):
    """Replay declared mature validation predictions and freeze maker responses."""
    from quant_research_platform.artifacts.statistics import write_statistics_artifact

    config = _strict_config(
        config,
        {
            "output",
            "panel",
            "source",
            "asset_id",
            "symbol",
            "reference",
            "models",
        },
        {"archive_paths", "chronology"},
    )
    _execution_asset(config)
    reject_if_exists(Path(config["output"]))
    windows = _calibration_windows(config)
    reference = _strict_config(
        config["reference"], {"order_quantity", "latency_ms", "min_calibration_samples"}
    )
    _positive(reference["order_quantity"], "hft_maker_order_quantity_invalid")
    _nonnegative_integer(reference["latency_ms"], "hft_maker_latency_invalid")
    if (
        type(reference["min_calibration_samples"]) is not int
        or reference["min_calibration_samples"] < 1
    ):
        raise ValueError("crypto_calibration_minimum_invalid")
    model_ids = {}
    model_evidence = {}
    for item in config["models"]:
        fold_id = item["fold_id"]
        window, _cutoff = windows[fold_id]
        _strict_config(
            item,
            {
                "fold_id",
                "inputs",
                "price_predictions",
                "proxy_predictions",
            },
            {"input_spec", "window", "cutoff"},
        )
        model_ids[fold_id] = []
        model_evidence[fold_id] = {}
        for role, target in (
            ("price", "future_mid_return_500ms"),
            ("proxy", "maker_bid_sell_pressure_proxy_500ms"),
        ):
            evidence = _prediction_evidence(
                item[f"{role}_predictions"],
                window=window,
            )
            model_evidence[fold_id][role] = evidence
            _metadata, facts, _predictions = evidence
            if facts["model"]["target_column"] != target:
                raise ValueError("crypto_maker_prediction_target_invalid")
            model_ids[fold_id].append(facts["model"]["model_id"])
    metadata = {key: value for key, value in config.items() if key != "output"}
    metadata.update(
        window={
            "start": min(
                pd.Timestamp(window.start) for window, _ in windows.values()
            ).isoformat(),
            "end": max(
                pd.Timestamp(window.end) for window, _ in windows.values()
            ).isoformat(),
        },
        sample_role="calibration",
        models=[
            {
                **item,
                "window": {
                    "start": windows[item["fold_id"]][0].start.isoformat(),
                    "end": windows[item["fold_id"]][0].end.isoformat(),
                },
                "cutoff": windows[item["fold_id"]][1].isoformat(),
            }
            for item in config["models"]
        ],
        model_ids=model_ids,
    )
    return write_statistics_artifact(
        Path(config["output"]),
        metadata=metadata,
        table_group="crypto_maker_calibration",
        tables=partial(
            _produce_maker_calibration, config=config, model_evidence=model_evidence
        ),
        validate_tables=validate_maker_calibration,
    )


def backtest_maker(config):
    """Execute a maker baseline or causal saved calibration on the explicit window."""
    from quant_research_platform.artifacts.statistics import read_statistics_artifact

    config = _strict_config(
        config,
        _BACKTEST_KEYS | {"models", "calibration"},
        {"archive_paths", "input_spec", "chronology", "funding_roots"},
    )
    window, windows, capital, panel = _backtest_inputs(config)
    execution = MakerConfig(
        **_strict_config(config["execution"], MakerConfig.__dataclass_fields__)
    )
    calibrated = config["calibration"] is not None
    if type(config["models"]) is not list or len(config["models"]) != (
        2 if calibrated else 0
    ):
        raise ValueError("crypto_maker_model_inputs_invalid")
    bounds = _window_bounds(windows)
    fold_id = windows[0][0]
    calibrations = None
    model_evidence = []
    if calibrated:
        artifact = read_statistics_artifact(
            Path(config["calibration"]), validate_tables=validate_maker_calibration
        )
        calibration_windows = _calibration_windows(artifact.metadata)
        if (
            artifact.metadata["asset_id"] != config["asset_id"]
            or artifact.metadata["symbol"] != config["symbol"]
            or any(
                cutoff >= pd.Timestamp(window.start)
                for _window_value, cutoff in calibration_windows.values()
            )
        ):
            raise ValueError("crypto_calibration_consumer_relationship_invalid")
        fold_id = max(calibration_windows, key=lambda key: calibration_windows[key][1])
        ids = []
        for item, target in zip(
            config["models"],
            ("future_mid_return_500ms", "maker_bid_sell_pressure_proxy_500ms"),
            strict=True,
        ):
            _strict_config(item, {"predictions"})
            evidence = _prediction_evidence(
                item["predictions"],
                window=window,
            )
            model_evidence.append(evidence)
            _metadata, facts, _predictions = evidence
            if facts["model"]["target_column"] != target:
                raise ValueError("crypto_maker_prediction_target_invalid")
            ids.append(facts["model"]["model_id"])
        if artifact.metadata["model_ids"][fold_id] != ids:
            raise ValueError("crypto_calibration_model_selection_mismatch")
        with artifact.table("crypto_maker_calibration.parquet") as summary:
            selected = summary.loc[summary.fold_id.eq(fold_id)]
            if (
                set(selected.side) != {"buy", "sell"}
                or not selected.status.eq("DEFINED").all()
            ):
                raise PortfolioTargetBlocked("maker_calibration_insufficient_samples")
            price_unit = selected.price_unit.iloc[0]
        calibrations = {
            fold_id: MakerCalibration(
                partial(_calibration_samples, artifact, fold_id),
                partial(_calibration_events, artifact, fold_id),
                price_unit=price_unit,
            )
        }

    def quotes():
        if calibrated:
            streams = [
                _prediction_rows(
                    inputs=config["inputs"],
                    input_spec=config.get("input_spec"),
                    evidence=evidence,
                    window=window,
                    fold_id=fold_id,
                    asset_id=config["asset_id"],
                )
                for evidence in model_evidence
            ]
            with closing_iterator(streams[0]), closing_iterator(streams[1]):
                for price, proxy in zip_longest(*streams):
                    if (
                        price is None
                        or proxy is None
                        or any(
                            price[key] != proxy[key]
                            for key in ("timestamp_ms", "fold_id")
                        )
                    ):
                        raise ValueError("maker_prediction_keys_misaligned")
                    if not all(
                        isinstance(row[key], str) and row[key]
                        for row in (price, proxy)
                        for key in HFT_EXECUTION_CONTEXT_COLUMNS
                    ):
                        continue
                    if any(
                        price[key] != proxy[key]
                        for key in HFT_EXECUTION_CONTEXT_COLUMNS
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
            options = _dataset_input_options(config["inputs"])
            if config.get("input_spec") is not None:
                options.update(
                    _dataset_input_options(
                        {"path": options["dataset_root"], "spec": config["input_spec"]}
                    )
                )
            session = open_dataset_session(**options).filtered_view(window)
            columns = ("timestamp", "asset_id", *HFT_EXECUTION_CONTEXT_COLUMNS)
            if set(columns) - set(session.logical_columns):
                raise ValueError("crypto_prediction_required_columns_excluded")
            session = session.projected_view(columns)
            chunks = iter_dataset_chunks(session)
            with closing_iterator(chunks):
                for frame in chunks:
                    for row in frame.itertuples(index=False):
                        if row.asset_id == config["asset_id"] and all(
                            isinstance(getattr(row, col), str) and getattr(row, col)
                            for col in HFT_EXECUTION_CONTEXT_COLUMNS
                        ):
                            yield {
                                "timestamp_ms": pd.Timestamp(row.timestamp).value
                                // 1_000_000,
                                "asset_id": config["asset_id"],
                                "fold_id": fold_id,
                                **{
                                    column: getattr(row, column)
                                    for column in HFT_EXECUTION_CONTEXT_COLUMNS
                                    if column != "segment_id"
                                },
                            }

    def produce(*, on_ledger=None, on_episode=None):
        symbol = config["symbol"]
        market_type = binance_ids.FUTURES_MARKET_TYPE
        with (
            open_market_source(
                Path(config["source"]),
                archive_paths=config.get("archive_paths"),
                time_window=TimeWindow(
                    "1970-01-01T00:00:00",
                    pd.Timestamp(max(end for _, end in bounds)).isoformat(),
                ),
            ) as run,
            read_funding_observations(
                tuple(config["funding_roots"])
                if "funding_roots" in config
                else run.funding_roots(symbol, time_window=window),
                symbol=symbol,
            ) as funding,
        ):
            _source_coordinates(run, panel, symbol)
            reconstruct = partial(iter_reconstructed_depth, market_type=market_type)
            source_stats = ReconstructionStats()
            coverage = (
                _execution_coverage(
                    run,
                    reconstruct_depth=reconstruct,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                )
                if config["execution_mode"] == "independent_segments"
                else source_stats.coverage_segments
            )
            trade_coverage = run.observation_coverage(
                market_type=market_type, data_kind="agg_trades", symbol=symbol
            )
            return _execute_portfolio(
                execution_mode=config["execution_mode"],
                execute=partial(
                    simulate_maker,
                    config=execution,
                    observed_coverage=coverage,
                    trade_coverage=trade_coverage,
                    capital=capital,
                    asset_id=config["asset_id"],
                    calibrations=calibrations,
                    decision_windows_ns=bounds,
                ),
                books=_books_until(
                    run,
                    reconstruct_depth=reconstruct,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                    stats=source_stats,
                ),
                trades=iter_agg_trades(
                    run,
                    market_type=market_type,
                    symbol=symbol,
                    end_ns=bounds[-1][1],
                ),
                decisions=quotes(),
                funding=funding,
                coverage=coverage,
                trade_coverage=trade_coverage,
                bounds=bounds,
                on_ledger=on_ledger,
                on_episode=on_episode,
            )

    return _write_backtest(config, capital, produce)


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
                bounds = {}
                for row in chain((first,), group):
                    (bids if row.side == "buy" else asks).append(
                        (int(row.price_index), float(row.quantity))
                    )
                    bounds[row.side] = int(row.known_price_bound) or None
                _yield_value = SimpleNamespace(
                    kind="book",
                    event_id=event_id,
                    event_at=first.event_at.value,
                    segment_id=first.context_id,
                    bids=tuple(bids),
                    asks=tuple(asks),
                    known_bid_floor_tick=bounds["buy"],
                    known_ask_ceiling_tick=bounds["sell"],
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
                    visible_quantity=None
                    if pd.isna(first.quantity)
                    else float(first.quantity),
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
            best, last, bounds = {}, {}, {}
            row_count = 0
            for row in chain((first,), group):
                row_count += 1
                if (
                    (row.event_at, row.event_kind, row.context_id)
                    != (first.event_at, first.event_kind, first.context_id)
                    or not isfinite(row.quantity)
                    and not (row.event_kind == "trade" and pd.isna(row.quantity))
                    or row.quantity < 0
                    or row.known_price_bound < 0
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
                    if row.known_price_bound and (
                        row.known_price_bound > row.price_index
                        if row.side == "buy"
                        else row.known_price_bound < row.price_index
                    ):
                        raise ValueError("crypto_calibration_book_invalid")
                    if counts[row.side]:
                        if row.known_price_bound != bounds[row.side]:
                            raise ValueError("crypto_calibration_book_invalid")
                        if (
                            row.price_index >= last[row.side]
                            if row.side == "buy"
                            else row.price_index <= last[row.side]
                        ):
                            raise ValueError("crypto_calibration_book_invalid")
                    else:
                        best[row.side] = row.price_index
                        bounds[row.side] = row.known_price_bound
                    last[row.side] = row.price_index
                    counts[row.side] += 1
                elif row.event_kind == "trade":
                    if (
                        row_count != 1
                        or row.side not in counts
                        or not row.context_id
                        or row.price_index <= 0
                        or row.level_index != 0
                        or row.known_price_bound != 0
                    ):
                        raise ValueError("crypto_calibration_event_invalid")
                elif row.event_kind == "boundary":
                    if (
                        row_count != 1
                        or row.side
                        or row.price_index != 0
                        or row.quantity != 0
                        or row.level_index != 0
                        or row.known_price_bound != 0
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


def validate_maker_calibration(artifact, metadata):
    _execution_asset(metadata)
    folds = _calibration_windows(metadata)
    reference = _strict_config(
        metadata["reference"],
        {"order_quantity", "latency_ms", "min_calibration_samples"},
    )
    minimum = reference["min_calibration_samples"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("crypto_calibration_minimum_invalid")
    output = _window(metadata["window"])
    if output != TimeWindow(
        min(window.start for window, _ in folds.values()),
        max(window.end for window, _ in folds.values()),
    ):
        raise ValueError("crypto_calibration_window_invalid")
    if (
        type(metadata.get("model_ids")) is not dict
        or set(metadata["model_ids"]) != set(folds)
        or any(
            type(ids) is not list
            or len(ids) != 2
            or any(type(value) is not str or not value for value in ids)
            or len(set(ids)) != 2
            for ids in metadata["model_ids"].values()
        )
    ):
        raise ValueError("crypto_calibration_model_ids_invalid")
    _positive(reference["order_quantity"], "hft_maker_order_quantity_invalid")
    _nonnegative_integer(reference["latency_ms"], "hft_maker_latency_invalid")
    with artifact.table("crypto_maker_calibration.parquet") as summary:
        if summary.empty:
            raise ValueError("crypto_calibration_keys_invalid")
        price_unit = validate_price_unit(summary.price_unit.iloc[0])
        if not summary.price_unit.eq(price_unit).all():
            raise ValueError("crypto_calibration_price_unit_invalid")
    totals = dict.fromkeys(("samples", "events", "calibration"), 0)
    paired_columns = tuple(
        column
        for column in STATISTICS_TABLE_COLUMNS["crypto_maker_samples.parquet"]
        if column not in ("side", "filled_quantity", "terminal_reason")
    )
    for fold_id, (sample_window, cutoff_at) in folds.items():
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
        cutoff = cutoff_at.value
        beginning = pd.Timestamp(sample_window.start).value
        last_decision = pd.Timestamp(sample_window.end).value
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
            native_events = reads.enter_context(
                closing_iterator(_calibration_events(artifact, fold_id))
            )
            window = deque()
            native_count = 0
            initial_state = None
            start_header = next(starts, None)
            end_header = next(ends, None)
            previous_start = previous_end = latest_state = None
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
                    beginning <= decision <= last_decision
                    and expiry == decision + MAKER_QUOTE_HORIZON_MS * 1000000
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
                ):
                    raise ValueError("crypto_calibration_event_coordinates_invalid")
                while window and window[0].event_id < start:
                    event = window.popleft()
                    if event.kind in ("book", "boundary"):
                        initial_state = event
                while native_count < stop:
                    event = next(native_events)
                    native_count += 1
                    if event.event_id < start:
                        if event.kind in ("book", "boundary"):
                            initial_state = event
                    else:
                        window.append(event)
                for row in pair:
                    sample = row._asdict()
                    sample["decision_at"] = decision
                    state = initial_state
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
                            lifetime_ns=MAKER_QUOTE_HORIZON_MS * 1000000,
                            replace_quantity=row.quote_quantity,
                            history_scale=1,
                            events=iter(window),
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
            while end_header is not None:
                totals["events"] += end_header.row_count
                end_header = next(ends, None)
            for _event in native_events:
                pass
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
    for name, count in totals.items():
        observed = 0
        with closing_iterator(
            artifact._batches(f"crypto_maker_{name}.parquet")
        ) as batches:
            for frame in batches:
                if not frame.fold_id.isin(folds).all():
                    raise ValueError("crypto_calibration_keys_invalid")
                observed += len(frame)
        if count != observed:
            raise ValueError("crypto_calibration_keys_invalid")
