"""A-share processed target producers."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from itertools import chain
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    is_datetime_window,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
)
from quant_research_platform.markets.implementations.a_share.benchmark import (
    iter_benchmark_weight_snapshots,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.execution import (
    EXECUTION_LOOKAHEAD_SESSIONS,
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_VALUE_COLUMNS,
    FLOW_METRIC_IDS,
    _pit_observations,
    _prepare_pit_fundamentals,
    construct_ttm_events,
    financial_events_from_interim,
)
from quant_research_platform.markets.implementations.a_share.processed import panels
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog as a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _prepare_market_panel_inputs,
    compute_market_panel,
    validate_market_bar_calendar,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors import engine as factor_engine
from quant_research_platform.research.factors.expectations.calculations import (
    FORECAST_FRESHNESS_DAYS,
    _prepare_expectations,
)
from quant_research_platform.research.labels import engine as label_engine
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime import resources
from quant_research_platform.storage.datasets.reading import (
    _prepare_in_filter,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.identity import filesystem_exists

# Amortize per-issuer preparation across panel and downstream session blocks.
DEFAULT_BATCH_ROWS = 8 * resources.DEFAULT_BATCH_ROWS


_FULL_HISTORY_SOURCE_IDS = {
    "stock_master",
    "market_cap",
    "trading_no_limit",
    "special_treatment_changes",
    "trading_calendar",
    "index_info",
    "a_share_industry_snapshots",
}


@contextmanager
def produce_a_share_processed_data(
    spec: ProcessedDataSpec,
    *,
    inputs,
    time_window: TimeWindow,
    processed_catalog: ProcessedDataCatalog | None = None,
    source_catalog: InterimDataCatalog,
):
    """Publish chronological blocks while retaining input and output owners."""
    processed = processed_catalog or a_share_processed_catalog()
    source_sessions = {}
    source = {}
    for data_id in spec.source_input_data_ids:
        source_spec = source_catalog.get(data_id)
        root = inputs.get(data_id)
        if root is None:
            if data_id in spec.optional_source_input_data_ids:
                source[data_id] = pd.DataFrame(columns=source_spec.required_columns)
                continue
            continue
        session = open_dataset_session(root, columns=None, time_window=None)
        if (
            data_id in _FULL_HISTORY_SOURCE_IDS
            or data_id in spec.optional_source_input_data_ids
        ):
            source_sessions[data_id] = session
            continue
        input_window = time_window
        if source_spec.source_id == "choice" or data_id == "market_bars_1d":
            input_window = _preceding_input_window(session, time_window)
        elif data_id == "analyst_forecast":
            freshness = json.loads(spec.parameters_json).get(
                "freshness_days", FORECAST_FRESHNESS_DAYS
            )
            input_window = TimeWindow(
                time_window.start - timedelta(days=freshness), time_window.end
            )
        source_sessions[data_id] = session.filtered_view(input_window)
    if spec.data_id == "market_panel_1d":
        missing = {
            "market_bars_1d",
            "stock_master",
            "market_cap",
            "trading_calendar",
            "trading_no_limit",
            "special_treatment_changes",
            "a_share_industry_snapshots",
        }.difference(source_sessions)
        if missing:
            raise ValueError(
                "a_share_panel_source_data_missing: " + ", ".join(sorted(missing))
            )
    dependencies = {}
    factor_definitions = (
        tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if spec.kind is ProcessedDataKind.FACTOR
        else ()
    )
    for data_id in spec.processed_input_data_ids:
        dependency = processed.get(data_id)
        root = inputs[data_id]
        if not filesystem_exists(root):
            raise ValueError(f"missing_processed_dependency: {data_id}")
        columns = None
        if spec.data_id == "fundamental_pit":
            columns = ("market_id", "asset_id", "timestamp")
        elif factor_definitions:
            requested = {
                *factor_engine.factor_panel_input_columns(factor_definitions),
                "has_legal_fundamental_pit",
                "listing_date",
            }
            if any(
                definition.family == "expectations" for definition in factor_definitions
            ):
                requested.add("close")
            if any(
                definition.family == "market_model" for definition in factor_definitions
            ):
                requested.add("market_type")
            columns = tuple(
                column for column in dependency.required_columns if column in requested
            )
        elif spec.kind is ProcessedDataKind.LABEL:
            requested = {
                *label_engine.label_panel_input_columns(a_share.label_catalog().list()),
                "listing_date",
            }
            columns = tuple(
                column for column in dependency.required_columns if column in requested
            )
        session = open_dataset_session(root, columns=columns, time_window=None)
        if spec.kind is ProcessedDataKind.LABEL or spec.data_id == "execution_mask":
            # Actual supplied PANEL sessions own the future execution/label clock.
            # Each block below reads only its required own-market lookahead.
            dependencies[data_id] = session
        else:
            input_window = (
                _preceding_input_window(session, time_window)
                if factor_engine.factor_lookback_bars(factor_definitions)
                else time_window
            )
            dependencies[data_id] = session.filtered_view(input_window)
    for data_id, session in source_sessions.items():
        if data_id not in {"market_bars_1d", "index_weights"}:
            source[data_id] = materialize_dataset_session(session)

    data = _processed_frames(
        spec,
        time_window=time_window,
        source=source,
        source_sessions=source_sessions,
        dependencies=dependencies,
    )
    with closing_iterator(data):
        first = next((frame for frame in data if not frame.empty), None)
        if first is None:
            raise ValueError("processed_output_empty")
        yield (chain((first,), data), {})


def _preceding_input_window(session, window):
    """Keep supplied history through the exact cutoff, without a protocol epoch."""
    minimum = session.manifest["date_min"]
    start = date.fromisoformat(minimum) if minimum is not None else window.start
    if is_datetime_window(window):
        start = pd.Timestamp(start)
    return TimeWindow(min(start, window.start), window.end)


def _processed_frames(spec, *, time_window, source, source_sessions, dependencies):
    if spec.data_id == "benchmark_weights":
        yield from _benchmark_frames(source, source_sessions["index_weights"])
        return
    panel_source = spec.data_id == "market_panel_1d"
    reference = (
        source_sessions["market_bars_1d"]
        if panel_source
        else next(iter(dependencies.values()))
    )
    stocks = (
        panels._canonical_stock_master(source["stock_master"]) if panel_source else None
    )
    (
        sessions,
        asset_types,
        market_history,
        asset_sessions,
        observed_sessions,
        _raw_axis,
    ) = _processed_axis(
        reference,
        panel_source=panel_source,
        calendar=source["trading_calendar"] if panel_source else None,
        stock_master=stocks,
        validation_window=time_window if panel_source else None,
        include_asset_sessions=spec.kind is ProcessedDataKind.LABEL
        or spec.data_id == "execution_mask"
        or spec.kind is ProcessedDataKind.FACTOR,
    )
    lower, upper, closed = time_window_timestamp_interval(time_window)
    if panel_source:
        source["observed_market_sessions"] = observed_sessions
        history_start = pd.Timestamp(
            min(lower, sessions[0]) if sessions else lower
        ).normalize()
        # Candidate dates cannot come from a potentially incomplete calendar.
        # The panel certifies requested O/C; older unknowns only break state carry.
        sessions = tuple(
            pd.date_range(
                history_start, upper, inclusive="both" if closed else "left"
            ).as_unit("ns")
        )
    definitions = (
        tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if spec.kind is ProcessedDataKind.FACTOR
        else ()
    )
    lookback = factor_engine.factor_lookback_bars(definitions)
    lookahead = (
        label_engine.label_lookahead_bars(a_share.label_catalog().list())
        if spec.kind is ProcessedDataKind.LABEL
        else 0
    )
    events = None
    if spec.data_id == "execution_mask":
        lookahead = EXECUTION_LOOKAHEAD_SESSIONS
    if spec.data_id == "fundamental_pit":
        events = _financial_pit_events(
            source, required_sources=spec.source_input_data_ids
        )
    asset_count = max(1, len(asset_types))
    block_sessions = max(1, DEFAULT_BATCH_ROWS // asset_count)
    if any(definition.family == "market_model" for definition in definitions):
        (
            _source_dates,
            _source_assets,
            market_history,
            _asset_sessions,
            _observed_sessions,
            _raw_axis,
        ) = _processed_axis(
            source_sessions["market_bars_1d"],
            panel_source=True,
            calendar=source["trading_calendar"],
            stock_master=panels._canonical_stock_master(source["stock_master"]),
            validation_window=time_window,
        )

        def panel_frames():
            for start in range(
                0,
                bisect_right(sessions, upper)
                if closed
                else bisect_left(sessions, upper),
                block_sessions,
            ):
                core = sessions[start : start + block_sessions]
                yield materialize_dataset_session(
                    reference.filtered_view(TimeWindow(core[0], core[-1]))
                )

        with closing_iterator(panel_frames()) as frames:
            with closing_iterator(
                factor_engine._market_model_frames(
                    frames,
                    trading_calendar=source["trading_calendar"],
                    market_history=market_history,
                )
            ) as calculated:
                for output in calculated:
                    output = output.loc[
                        output.timestamp.ge(lower)
                        & (
                            output.timestamp.le(upper)
                            if closed
                            else output.timestamp.lt(upper)
                        ),
                        list(spec.required_columns),
                    ]
                    if not output.empty:
                        yield output.reset_index(drop=True)
        return
    first = 0 if panel_source else bisect_left(sessions, lower)
    stop = bisect_right(sessions, upper) if closed else bisect_left(sessions, upper)
    history = None
    expectations = None
    financial_calculate = None
    if panel_source and first < stop:
        source = _prepare_panel_source(source, stocks)
    for start in range(first, stop, block_sessions):
        end = min(stop, start + block_sessions)
        core = sessions[start:end]
        read_end = sessions[min(len(sessions) - 1, end - 1 + lookahead)]
        if lookahead:
            # PANEL keys contain every legal own-market session, including
            # unknown quotes and suspensions; closed-market dates have no key.
            read_end = max(
                (
                    axis[
                        min(len(axis) - 1, bisect_right(axis, core[-1]) - 1 + lookahead)
                    ]
                    for axis in asset_sessions.values()
                    if axis[0] <= core[-1] and axis[-1] >= core[0]
                ),
                default=read_end,
            )
        read_start = sessions[max(0, start - lookback)]
        if lookback:
            read_start = min(
                (
                    axis[max(0, bisect_left(axis, core[0]) - lookback)]
                    for axis in asset_sessions.values()
                    if axis[0] <= core[-1] and axis[-1] >= core[0]
                ),
                default=read_start,
            )
        read_window = TimeWindow(read_start, read_end)
        frames = {}
        for data_id, session in dependencies.items():
            view = session.filtered_view(read_window)
            frames[data_id] = materialize_dataset_session(view)
        current_source = dict(source)
        if panel_source:
            view = reference.filtered_view(read_window)
            current_source["market_bars_1d"] = materialize_dataset_session(view)
        if expectations is None and any(
            definition.family == "expectations" for definition in definitions
        ):
            analyst = source.get("analyst_forecast")
            if analyst is None or analyst.empty:
                raise ValueError("BLOCKED_EXTERNAL_DATA: analyst_forecast")
            expectations = _prepare_expectations(
                analyst.rename(columns={"source_asset_id": "asset_id"}),
                trading_sessions=unified_trading_sessions(source["trading_calendar"]),
                selected_columns=tuple(
                    definition.output_column for definition in definitions
                ),
                freshness_days=json.loads(spec.parameters_json).get(
                    "freshness_days", FORECAST_FRESHNESS_DAYS
                ),
            )
        financial_pit = None
        if spec.data_id == "fundamental_pit":
            observations = _pit_observations(frames["market_panel_1d"])
            if financial_calculate is None:
                financial_calculate = _prepare_pit_fundamentals(events)
            financial_pit = financial_calculate(observations)
        output = _compute_processed_frame(
            spec,
            source=current_source,
            dependencies=frames,
            trading_sessions=sessions,
            asset_sessions=asset_sessions,
            panel_sessions=pd.DatetimeIndex(core),
            required_calendar_start=lower if panel_source else None,
            asset_market_types=asset_types if panel_source else None,
            suspension_history=history,
            financial_pit=financial_pit,
            expectations=expectations,
            market_history=market_history,
        )
        output = (
            output.loc[output["timestamp"].between(core[0], core[-1])]
            .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        if panel_source and (not output.empty):
            combined = (
                output
                if history is None
                else pd.concat([history, output], ignore_index=True)
            )
            keys = ["asset_id", "timestamp"]
            ordered = combined.sort_values(keys, kind="mergesort")
            history = (
                pd.concat(
                    [
                        ordered.groupby("asset_id", sort=False).tail(1),
                        ordered.loc[ordered["is_trading"]]
                        .groupby("asset_id", sort=False)
                        .tail(1),
                        ordered.loc[ordered["adj_close"].notna()]
                        .groupby("asset_id", sort=False)
                        .tail(1),
                        ordered.loc[
                            ~ordered["filling_flag"].isin((0, 1))
                            | ordered["is_trading"]
                            & ordered[["close", "adj_close"]].isna().any(axis=1)
                        ]
                        .groupby("asset_id", sort=False)
                        .tail(1),
                    ]
                )
                .drop_duplicates(keys)
                .sort_values(keys)
                .reset_index(drop=True)
            )
            del combined, ordered
        if panel_source:
            output = output.loc[output["timestamp"].ge(lower)].reset_index(drop=True)
        if not output.empty:
            yield output
        del output, frames, current_source


def _benchmark_frames(source, session):
    (
        sessions,
        assets,
        _market_history,
        _asset_sessions,
        _observed_sessions,
        _raw_axis,
    ) = _processed_axis(
        session, panel_source=False, asset_column="constituent_asset_id"
    )
    block_size = max(1, DEFAULT_BATCH_ROWS // max(1, 2 * len(assets)))

    def inputs():
        for start in range(0, len(sessions), block_size):
            window = TimeWindow(
                sessions[start], sessions[min(len(sessions), start + block_size) - 1]
            )
            view = session.filtered_view(window)
            frame = materialize_dataset_session(view)
            yield frame
            del frame

    with closing_iterator(inputs()) as frames:
        with closing_iterator(
            iter_benchmark_weight_snapshots(source["index_info"], frames)
        ) as normalized:
            for _yield_value in normalized:
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


def _processed_axis(
    session,
    *,
    panel_source,
    asset_column=None,
    calendar=None,
    validation_window=None,
    include_asset_sessions=False,
    stock_master=None,
):
    if asset_column is None:
        asset_column = "source_asset_id" if panel_source else "asset_id"
    columns = (
        (asset_column, "timestamp", "market_type", "filling_flag")
        if panel_source
        else (asset_column, "timestamp")
    )
    view = replace(session, logical_columns=columns, physical_columns=columns)
    if panel_source:
        boundaries = (
            stock_master.loc[:, ["asset_id", "listing_date"]]
            .drop_duplicates()
            .rename(columns={"asset_id": asset_column, "listing_date": "_listing_date"})
            .sort_values("_listing_date", kind="mergesort")
        )

    def with_lifecycles(frame):
        return pd.merge_asof(
            frame.sort_values("timestamp", kind="mergesort"),
            boundaries.astype({asset_column: frame[asset_column].dtype}),
            left_on="timestamp",
            right_on="_listing_date",
            by=asset_column,
            direction="backward",
        )

    dates = set()
    assets = {}
    raw_dates, raw_assets = set(), set()
    first_market_events = {}
    observed_market_sessions = {}
    asset_session_masks = {}
    session_positions = {}
    with closing_iterator(iter_dataset_chunks(view, batch_rows=65536)) as chunks:
        for frame in chunks:
            if panel_source:
                raw_dates.update(frame.timestamp.unique())
                raw_assets.update(frame[asset_column].unique())
                frame["market_type"] = strict_integer_series(
                    frame["market_type"], error="invalid_market_type", allow_null=True
                )
                frame = frame.loc[frame["market_type"].isin(CURRENT_MARKET_TYPES)]
                for market, group in frame.loc[
                    frame["filling_flag"].isin((0, 1))
                ].groupby("market_type", sort=False):
                    observed_market_sessions.setdefault(market, set()).update(
                        group.timestamp.unique()
                    )
                observed = frame
                if validation_window is not None:
                    lower, upper, closed = time_window_timestamp_interval(
                        validation_window
                    )
                    observed = frame.loc[
                        frame["timestamp"].ge(lower)
                        & (
                            frame["timestamp"].le(upper)
                            if closed
                            else frame["timestamp"].lt(upper)
                        )
                    ]
                if not observed.empty:
                    validate_market_bar_calendar(observed, calendar)
                del observed
                frame = with_lifecycles(frame)
                first = (
                    frame.groupby(
                        [asset_column, "_listing_date", "market_type"], sort=False
                    )["timestamp"]
                    .min()
                    .reset_index()
                )
                for asset, listing_date, market_type, timestamp in first.itertuples(
                    index=False, name=None
                ):
                    key = (asset, listing_date, market_type)
                    first_market_events[key] = min(
                        timestamp, first_market_events.get(key, timestamp)
                    )
                frame = frame.loc[frame["filling_flag"].ne(2)]
            if include_asset_sessions:
                for asset, group in frame.groupby(asset_column, sort=False):
                    mask = asset_session_masks.get(asset, 0)
                    for timestamp in group.timestamp:
                        position = session_positions.setdefault(
                            timestamp, len(session_positions)
                        )
                        mask |= 1 << position
                    asset_session_masks[asset] = mask
            new_dates = set(pd.to_datetime(frame["timestamp"]).unique()) - dates
            first = frame.drop_duplicates(asset_column)
            new_assets = first.loc[~first[asset_column].isin(assets)]
            dates.update(new_dates)
            for row in new_assets.itertuples(index=False, name=None):
                assets[row[0]] = row[2] if panel_source else 0
            del frame, first, new_assets, new_dates
    sessions = tuple(sorted(pd.Timestamp(value) for value in dates))
    asset_types = pd.Series(assets)
    del dates, assets
    market_history = None
    if panel_source:
        first = pd.DataFrame(
            [
                (asset, timestamp, market_type, listing_date)
                for (
                    asset,
                    listing_date,
                    market_type,
                ), timestamp in first_market_events.items()
            ],
            columns=("asset_id", "timestamp", "market_type", "_listing_date"),
        ).sort_values(["asset_id", "timestamp"], kind="mergesort")
        asset_types = first.drop_duplicates("asset_id").set_index("asset_id")[
            "market_type"
        ]
        migrating = tuple(
            first.loc[first.asset_id.duplicated(keep=False), "asset_id"].unique()
        )
        market_history = first
        if migrating:
            columns = (asset_column, "timestamp", "market_type")
            view = replace(
                session,
                logical_columns=columns,
                physical_columns=columns,
                parquet_filters=(
                    *(session.parquet_filters or ()),
                    (
                        asset_column,
                        "in",
                        _prepare_in_filter(
                            session.arrow_schema.field(asset_column), migrating
                        ),
                    ),
                ),
            )
            history = with_lifecycles(materialize_dataset_session(view)).rename(
                columns={asset_column: "asset_id"}
            )
            history = history.loc[
                history.market_type.isin(CURRENT_MARKET_TYPES)
            ].sort_values(["asset_id", "timestamp"], kind="mergesort")
            changed = history.market_type.ne(
                history.groupby(
                    ["asset_id", "_listing_date"], sort=False
                ).market_type.shift()
            ).fillna(True)
            market_history = pd.concat(
                (first.loc[~first.asset_id.isin(migrating)], history.loc[changed]),
                ignore_index=True,
            )
        market_history = market_history.drop(columns="_listing_date")
    axes = {
        mask: tuple(
            timestamp
            for timestamp in sessions
            if mask & (1 << session_positions[timestamp])
        )
        for mask in set(asset_session_masks.values())
    }
    raw_axis = (
        (
            tuple(sorted(pd.Timestamp(value) for value in raw_dates)),
            frozenset(raw_assets),
        )
        if panel_source
        else (sessions, frozenset(asset_types.index))
    )
    return (
        sessions,
        asset_types,
        market_history,
        {asset: axes[mask] for asset, mask in asset_session_masks.items()},
        {
            market: pd.DatetimeIndex(sorted(dates))
            for market, dates in observed_market_sessions.items()
        },
        raw_axis,
    )


def _prepare_panel_source(source, stocks):
    return _prepare_market_panel_inputs(
        stocks=stocks,
        market_cap=source["market_cap"],
        trading_calendar=source["trading_calendar"],
        observed_market_sessions=source.get("observed_market_sessions"),
        industry_history=source.get("industry_classification"),
        industry_sets=source.get("industry_classification_sets"),
        industry_versions=source.get("industry_classification_versions"),
        industry_changes=source.get("stock_info_changes"),
        industry_snapshots=source["a_share_industry_snapshots"],
        no_limit_history=source.get("trading_no_limit"),
        special_treatment_history=source.get("special_treatment_changes"),
    )


def _compute_processed_frame(
    spec,
    *,
    source,
    dependencies,
    trading_sessions,
    asset_sessions=None,
    panel_sessions=None,
    required_calendar_start=None,
    asset_market_types=None,
    suspension_history=None,
    financial_pit=None,
    expectations=None,
    market_history=None,
):
    if spec.data_id == "market_panel_1d":
        panel = compute_market_panel(
            source["market_bars_1d"],
            **{key: value for key, value in source.items() if key != "market_bars_1d"},
            session_window=panel_sessions,
            required_calendar_start=required_calendar_start,
            asset_market_types=asset_market_types,
            suspension_history=suspension_history,
            market_history=market_history,
        )
        return (
            pd.DataFrame(columns=spec.required_columns)
            if panel.empty
            else panel.loc[:, list(spec.required_columns)]
        )
    if spec.data_id == "execution_mask":
        return derive_execution_mask(dependencies["market_panel_1d"]).loc[
            :, list(spec.required_columns)
        ]
    if spec.data_id == "fundamental_pit":
        return _fundamental_pit_frame(
            spec,
            panel=dependencies.get("market_panel_1d"),
            pit=financial_pit,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        definitions = tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if "expectations" in {definition.family for definition in definitions}:
            panel = dependencies["market_panel_1d"]
            features = expectations(panel.loc[:, ["asset_id", "timestamp", "close"]])
            return (
                panel.loc[:, ["market_id", "asset_id", "timestamp"]]
                .merge(
                    features,
                    on=["asset_id", "timestamp"],
                    how="left",
                    validate="one_to_one",
                )
                .loc[:, list(spec.required_columns)]
            )
        input_frame = dependencies[next(iter(spec.processed_input_data_ids))]
        if any(
            definition.family == "fundamental"
            and "total_market_cap" in definition.input_columns
            for definition in definitions
        ):
            market = next(
                frame
                for frame in dependencies.values()
                if "total_market_cap" in frame.columns
            ).loc[:, ["market_id", "asset_id", "timestamp", "total_market_cap"]]
            input_frame = input_frame.merge(
                market,
                on=["market_id", "asset_id", "timestamp"],
                how="left",
                validate="one_to_one",
            )
        calculated = factor_engine.calculate_factors_for_panel(
            input_frame,
            definitions=definitions,
            history_keys=("asset_id", "listing_date"),
            output_window=None
            if panel_sessions is None
            else TimeWindow(panel_sessions[0], panel_sessions[-1]),
        )
        if "has_legal_fundamental_pit" in spec.required_columns:
            calculated = calculated.merge(
                input_frame.loc[
                    :,
                    ["market_id", "asset_id", "timestamp", "has_legal_fundamental_pit"],
                ],
                on=["market_id", "asset_id", "timestamp"],
                how="left",
                validate="one_to_one",
            )
        return calculated.loc[:, list(spec.required_columns)]
    if spec.kind is ProcessedDataKind.LABEL:
        panel = dependencies["market_panel_1d"].merge(
            dependencies["execution_mask"],
            on=["market_id", "asset_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        definitions = a_share.label_catalog().list()
        pieces = [
            label_engine.calculate_labels_for_panel(
                group,
                definitions=definitions,
                sessions=trading_sessions
                if asset_sessions is None
                else asset_sessions[asset],
            )
            for (asset, _listing_date), group in panel.groupby(
                ["asset_id", "listing_date"], sort=False
            )
        ]
        return pd.concat(pieces, ignore_index=True).loc[:, list(spec.required_columns)]
    raise ValueError(f"unsupported_a_share_processed_kind: {spec.kind}")


def _financial_pit_events(source, *, required_sources):
    required = set(required_sources)
    if not required.issubset(source):
        raise ValueError(
            "BLOCKED_EXTERNAL_DATA: " + ", ".join(sorted(required.difference(source)))
        )
    calendar = source.get("trading_calendar")
    if calendar is None:
        raise ValueError("fundamental_pit_calendar_required")
    sessions = unified_trading_sessions(calendar)
    events = financial_events_from_interim(
        {
            data_id: source[data_id]
            for data_id in FINANCIAL_VALUE_COLUMNS
            if data_id in required
        },
        source["financial_first_disclosure"],
        trading_sessions=sessions,
    )
    ttm_events = construct_ttm_events(events)
    events = ttm_events
    events["value"] = events["value"].where(
        ~events["metric_id"].isin(FLOW_METRIC_IDS), events["ttm_value"]
    )
    return events


def _fundamental_pit_frame(
    spec: ProcessedDataSpec,
    *,
    panel: pd.DataFrame | None,
    pit: pd.DataFrame,
) -> pd.DataFrame:
    if panel is None:
        raise ValueError("fundamental_pit_market_panel_required")
    values = pit.pivot_table(
        index=["asset_id", "timestamp"],
        columns="metric_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    prior_year = (
        pit.loc[:, ["asset_id", "timestamp", "metric_id", "prior_year_value"]]
        .pivot_table(
            index=["asset_id", "timestamp"],
            columns="metric_id",
            values="prior_year_value",
            aggfunc="first",
        )
        .reset_index()
    )
    prior_year = prior_year.rename(
        columns={
            column: f"prior_year_{column}_ttm"
            if column in FLOW_METRIC_IDS
            else f"prior_year_{column}"
            for column in prior_year.columns
            if column not in {"asset_id", "timestamp"}
        }
    )
    for column in ("total_assets", "parent_equity"):
        source_column = f"prior_year_{column}"
        if source_column in prior_year:
            prior_year[f"beginning_{column}"] = prior_year[source_column]
    output = (
        panel.loc[:, ["market_id", "asset_id", "timestamp"]]
        .merge(values, on=["asset_id", "timestamp"], how="left", validate="one_to_one")
        .merge(
            prior_year, on=["asset_id", "timestamp"], how="left", validate="one_to_one"
        )
    )
    pit_value_columns = tuple(
        column for columns in FINANCIAL_VALUE_COLUMNS.values() for column in columns
    )
    for column in (
        *pit_value_columns,
        "beginning_total_assets",
        "beginning_parent_equity",
    ):
        if column not in output:
            output[column] = np.nan
    numeric_columns = {
        column
        for column in spec.required_columns
        if column not in {"market_id", "asset_id", "timestamp"}
    }
    for column in numeric_columns:
        if column in output:
            output[column] = strict_finite_numeric_series(
                output[column],
                error=f"invalid_fundamental_pit_value: {column}",
                allow_null=True,
            ).astype("float64")
    for column in ("total_assets", "parent_equity"):
        output[f"average_{column}"] = [
            mean(pair)
            for pair in zip(output[column], output[f"beginning_{column}"], strict=True)
        ]
    for column in FLOW_METRIC_IDS:
        output[f"{column}_ttm"] = output[column]
    output["has_legal_fundamental_pit"] = (
        output.loc[:, list(pit_value_columns)].notna().any(axis=1)
    )
    for column in spec.required_columns:
        if column not in output:
            output[column] = np.nan
    return output.loc[:, list(spec.required_columns)]
