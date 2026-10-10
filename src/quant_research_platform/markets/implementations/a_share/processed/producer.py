"""A-share processed target producers."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from itertools import chain
from statistics import mean

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.data import ProcessedDataKind, ProcessedDataSpec
from quant_research_platform.contracts.portfolios import (
    _EXECUTION_COLUMNS,
    _portfolio_session_frame,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
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
    _iter_dataset_session_frames,
    _prepare_in_filter,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.identity import filesystem_exists
from quant_research_platform.storage.layout import StorageLayout

# Amortize per-issuer preparation across panel and downstream session blocks.
DEFAULT_BATCH_ROWS = 8 * resources.DEFAULT_BATCH_ROWS


_OPTIONAL_SOURCE_IDS = frozenset(
    {
        "industry_classification",
        "industry_classification_sets",
        "industry_classification_versions",
        "stock_info_changes",
    }
)
_FULL_HISTORY_SOURCE_IDS = _OPTIONAL_SOURCE_IDS | {
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
    layout: StorageLayout,
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
        root = layout.target_root(
            data_id, layout.interim_data_root(source_spec.source_id, data_id)
        )
        if not filesystem_exists(root):
            if data_id in _OPTIONAL_SOURCE_IDS:
                source[data_id] = pd.DataFrame(columns=source_spec.required_columns)
                continue
            continue
        source_sessions[data_id] = open_dataset_session(
            root,
            columns=None,
            time_window=None
            if data_id in _FULL_HISTORY_SOURCE_IDS
            else a_share.source_history_window(
                source_spec,
                time_window,
                freshness_days=json.loads(spec.parameters_json).get(
                    "freshness_days", FORECAST_FRESHNESS_DAYS
                ),
            ),
        )
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
        root = layout.target_root(
            data_id,
            layout.processed_data_root(dependency.market_id, dependency.kind, data_id),
        )
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
        dependencies[data_id] = open_dataset_session(
            root,
            columns=columns,
            time_window=a_share.processed_input_window(spec, time_window),
        )
    for data_id, session in source_sessions.items():
        if data_id not in {"market_bars_1d", "index_weights"}:
            source[data_id] = materialize_dataset_session(session)

    def verify(_receipt):
        for session in (*source_sessions.values(), *dependencies.values()):
            session.verify()

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
            yield (pd.DataFrame(), {}, verify)
            return
        yield (chain((first,), data), {}, verify)


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
    calendar = (
        unified_trading_sessions(source["trading_calendar"]) if panel_source else None
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
        history_start = min(lower, sessions[0]) if sessions else lower
        sessions = tuple(
            calendar[
                (calendar >= history_start)
                & (calendar <= upper if closed else calendar < upper)
            ]
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


_MARKET_FACT_COLUMNS = ("source_asset_id", "timestamp", "market_type", "filling_flag")
_STOCK_FACT_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "listing_date",
    "delisting_date",
    "company_status",
)


@dataclass(frozen=True, slots=True)
class _PortfolioSession:
    timestamp: pd.Timestamp
    panel: pd.DataFrame
    valuation: pd.DataFrame
    closed_assets: frozenset[str]
    execution: pd.DataFrame


def _portfolio_session_evidence(
    layout, panel_root, calendar, *, include_asset_sessions
):
    bars = open_dataset_session(
        layout.target_root("market_bars_1d", None),
        columns=_MARKET_FACT_COLUMNS,
        time_window=None,
    )
    panel = open_dataset_session(panel_root, columns=None, time_window=None)
    asset_sessions = {}
    if include_asset_sessions:
        _, _, _, asset_sessions, _, _raw_axis = _processed_axis(
            panel, panel_source=False, include_asset_sessions=True
        )
    stocks = panels._canonical_stock_master(
        materialize_dataset_session(
            open_dataset_session(
                layout.target_root("stock_master", None),
                columns=_STOCK_FACT_COLUMNS,
                time_window=None,
            )
        )
    )
    _, _, membership, _, _, raw_axis = _processed_axis(
        bars, panel_source=True, calendar=calendar, stock_master=stocks
    )
    return bars, panel, membership, asset_sessions, stocks, raw_axis


def iter_portfolio_sessions(
    *, layout, panel_root, panel_columns, calendar, time_window, execution_root=None
):
    """Project proved closed-market marks without changing materialized inputs."""
    bars, panel, membership, asset_sessions, stocks, raw_axis = (
        _portfolio_session_evidence(
            layout,
            panel_root,
            calendar,
            include_asset_sessions=execution_root is not None,
        )
    )
    sessions = unified_trading_sessions(calendar)
    calendar = calendar.set_index(["calendar_date", "source_market_type"])[
        "is_trading_day"
    ]
    if calendar.index.has_duplicates:
        raise ValueError("duplicate_trading_calendar_key")
    market_events = iter(membership.sort_values("timestamp").groupby("timestamp"))
    pending_membership = next(market_events, None)
    market_types = pd.Series(dtype="Int64")
    unknown_membership: set[str] = set()
    blocked: set[str] = set()
    last = pd.DataFrame(columns=panel_columns).set_index("asset_id", drop=False)
    masks = pd.DataFrame(columns=(*_EXECUTION_COLUMNS, "execution_at")).set_index(
        "asset_id", drop=False
    )

    def ordered_bars():
        dates, assets = raw_axis
        block_size = max(1, DEFAULT_BATCH_ROWS // max(1, len(assets)))
        for start in range(0, len(dates), block_size):
            core = dates[start : start + block_size]
            frame = materialize_dataset_session(
                bars.filtered_view(TimeWindow(core[0], core[-1]))
            ).sort_values("timestamp", kind="mergesort")
            for _, day in frame.groupby("timestamp", sort=False):
                yield day

    with closing_iterator(ExitStack()) as channels:
        bar_frames = channels.enter_context(closing_iterator(ordered_bars()))
        panel_frames = channels.enter_context(
            closing_iterator(_iter_dataset_session_frames(panel, panel_columns))
        )
        execution_frames = (
            channels.enter_context(
                closing_iterator(
                    _iter_dataset_session_frames(
                        open_dataset_session(
                            execution_root, columns=_EXECUTION_COLUMNS, time_window=None
                        ),
                        _EXECUTION_COLUMNS,
                    )
                )
            )
            if execution_root is not None
            else iter(())
        )
        pending_bar = next(bar_frames, None)
        pending_panel = next(panel_frames, None)
        pending_execution = next(execution_frames, None)
        dates = pd.date_range(
            calendar.index.get_level_values("calendar_date").min(),
            pd.Timestamp(time_window.end),
            freq="D",
        ).as_unit("ns")
        for timestamp in dates:
            day_bars = pd.DataFrame(columns=_MARKET_FACT_COLUMNS)
            while (
                pending_bar is not None
                and pd.Timestamp(pending_bar.timestamp.iloc[0]) <= timestamp
            ):
                current = pending_bar
                unknown = ~current.market_type.isin(CURRENT_MARKET_TYPES)
                unknown_membership.update(
                    current.loc[unknown, "source_asset_id"].astype(str)
                )
                unknown_membership.difference_update(
                    current.loc[~unknown, "source_asset_id"].astype(str)
                )
                blocked.update(
                    current.loc[
                        unknown | ~current.filling_flag.isin((0, 1, 2)),
                        "source_asset_id",
                    ].astype(str)
                )
                if pd.Timestamp(current.timestamp.iloc[0]) == timestamp:
                    day_bars = current.set_index("source_asset_id", drop=False)
                pending_bar = next(bar_frames, None)
            while pending_membership is not None and pending_membership[0] <= timestamp:
                event = pending_membership[1].set_index("asset_id").market_type
                market_types = pd.concat(
                    (market_types.drop(event.index, errors="ignore"), event)
                )
                pending_membership = next(market_events, None)
            status = calendar.reindex(
                pd.MultiIndex.from_arrays(
                    ([timestamp] * len(market_types), market_types),
                    names=("calendar_date", "source_market_type"),
                )
            )
            status.index = market_types.index
            blocked.update(status.index[status.isna()].astype(str))
            closed = frozenset(
                str(asset)
                for asset in status.index[status.eq(False).fillna(False)]
                if str(asset) not in unknown_membership
            )
            raw = pd.DataFrame(columns=panel_columns)
            while (
                pending_panel is not None
                and pd.Timestamp(pending_panel.timestamp.iloc[0]) <= timestamp
            ):
                if pd.Timestamp(pending_panel.timestamp.iloc[0]) == timestamp:
                    raw = _portfolio_session_frame(pending_panel, panel_columns)
                pending_panel = next(panel_frames, None)
            if not raw.empty:
                indexed = raw.set_index("asset_id", drop=False)
                if set(indexed.index.astype(str)) & closed:
                    raise ValueError("portfolio_market_panel_calendar_conflict")
                prices = indexed.loc[:, ["adj_open", "adj_close"]]
                valid = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
                flags = day_bars.filling_flag.reindex(indexed.index)
                known = flags.isin((0, 1)) & ~indexed.index.isin(unknown_membership)
                blocked.update(indexed.index[~valid | ~known].astype(str))
                real = valid & known & flags.eq(0)
                blocked.difference_update(indexed.index[real].astype(str))
                previous = indexed.loc[real]
                if not previous.empty:
                    last = (
                        previous.copy()
                        if last.empty
                        else pd.concat(
                            (last.drop(previous.index, errors="ignore"), previous)
                        )
                    )
            blocked.update(
                status.index[
                    status.eq(True).fillna(False) & ~status.index.isin(raw.asset_id)
                ].astype(str)
            )
            execution = pd.DataFrame(columns=_EXECUTION_COLUMNS)
            while (
                pending_execution is not None
                and pd.Timestamp(pending_execution.timestamp.iloc[0]) <= timestamp
            ):
                if pd.Timestamp(pending_execution.timestamp.iloc[0]) == timestamp:
                    execution = _portfolio_session_frame(
                        pending_execution, _EXECUTION_COLUMNS
                    )
                pending_execution = next(execution_frames, None)
            if not raw.empty:
                masks = masks.drop(raw.asset_id, errors="ignore")
            if not execution.empty:
                execution = execution.copy()
                endpoints = []
                for asset in execution.asset_id.astype(str):
                    axis = asset_sessions.get(asset, ())
                    position = bisect_right(axis, timestamp)
                    endpoints.append(axis[position] if position < len(axis) else pd.NaT)
                execution["execution_at"] = endpoints
                masks = pd.concat((masks, execution.set_index("asset_id", drop=False)))
                masks[["can_enter", "can_exit"]] = masks[
                    ["can_enter", "can_exit"]
                ].astype("boolean")
            if timestamp not in sessions or timestamp < pd.Timestamp(time_window.start):
                continue
            carried = last.loc[last.index.isin(closed)].copy()
            if not carried.empty:
                facts = panels._asof_join_stock_master(
                    pd.DataFrame(
                        {
                            "asset_id": carried.index.astype(stocks.asset_id.dtype),
                            "timestamp": timestamp,
                        }
                    ),
                    stocks,
                ).set_index("asset_id")
                alive = (
                    facts.listing_date.notna()
                    & facts.listing_date.le(timestamp)
                    & facts.delisting_date.ge(timestamp)
                    & facts.listing_date.le(carried.timestamp)
                )
                carried = carried.loc[alive.reindex(carried.index).fillna(False)]
                carried["timestamp"] = timestamp
                carried["market_type"] = market_types.reindex(carried.index)
                carried["adj_open"] = carried["adj_close"]
                carried.loc[carried.index.isin(blocked), ["adj_open", "adj_close"]] = (
                    np.nan
                )
            valuation = raw.copy()
            valuation.loc[
                valuation.asset_id.isin(blocked), ["adj_open", "adj_close"]
            ] = np.nan
            if not carried.empty:
                valuation = (
                    carried.reset_index(drop=True)
                    if valuation.empty
                    else pd.concat(
                        (valuation, carried.reset_index(drop=True)), ignore_index=True
                    )
                )
            result = _PortfolioSession(
                timestamp, raw, valuation, closed, masks.reset_index(drop=True).copy()
            )
            try:
                yield result
            except GeneratorExit:
                return
