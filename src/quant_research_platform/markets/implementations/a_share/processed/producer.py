"""A-share processed target producers."""

from __future__ import annotations

import sys
from bisect import bisect_left, bisect_right
from contextlib import contextmanager
from dataclasses import replace
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
    construct_ttm_events,
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog as a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    compute_market_panel,
    validate_market_bar_calendar,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors import engine as factor_engine
from quant_research_platform.research.factors.expectations.calculations import (
    build_expectations_features,
)
from quant_research_platform.research.labels import engine as label_engine
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)
from quant_research_platform.storage.identity import filesystem_exists
from quant_research_platform.storage.layout import StorageLayout

_OPTIONAL_SOURCE_IDS = frozenset(
    {
        "stock_master",
        "market_cap",
        "industry_classification",
        "industry_classification_sets",
        "industry_classification_versions",
        "stock_info_changes",
        "trading_no_limit",
        "special_treatment_changes",
    }
)
_FULL_HISTORY_SOURCE_IDS = _OPTIONAL_SOURCE_IDS | {
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
    source = {}
    source_sessions = {}
    dependencies = {}
    try:
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
                else a_share.source_history_window(source_spec, time_window),
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
                layout.processed_data_root(
                    dependency.market_id, dependency.kind, data_id
                ),
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
                }
                if any(
                    definition.family == "expectations"
                    for definition in factor_definitions
                ):
                    requested.add("close")
                columns = tuple(
                    column
                    for column in dependency.required_columns
                    if column in requested
                )
            elif spec.kind is ProcessedDataKind.LABEL:
                requested = set(
                    label_engine.label_panel_input_columns(
                        a_share.label_catalog().list()
                    )
                )
                columns = tuple(
                    column
                    for column in dependency.required_columns
                    if column in requested
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
    finally:
        primary = sys.exception()
        if primary is not None:
            pass
        source.clear()
        source_sessions.clear()
        dependencies.clear()
        first = data = None


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
    sessions, asset_types = _processed_axis(
        reference,
        panel_source=panel_source,
        calendar=calendar,
        validation_window=time_window if panel_source else None,
    )
    lower, upper, closed = time_window_timestamp_interval(time_window)
    if panel_source:
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
        events = _financial_pit_events(source)
    asset_count = max(1, len(asset_types))
    block_sessions = max(1, DEFAULT_BATCH_ROWS // asset_count)
    first = 0 if panel_source else bisect_left(sessions, lower)
    stop = bisect_right(sessions, upper) if closed else bisect_left(sessions, upper)
    history = None
    for start in range(first, stop, block_sessions):
        end = min(stop, start + block_sessions)
        core = sessions[start:end]
        read_window = TimeWindow(
            sessions[max(0, start - lookback)],
            sessions[min(len(sessions) - 1, end - 1 + lookahead)],
        )
        frames = {}
        for data_id, session in dependencies.items():
            view = session.filtered_view(read_window)
            frames[data_id] = materialize_dataset_session(view)
        current_source = dict(source)
        if panel_source:
            view = reference.filtered_view(read_window)
            current_source["market_bars_1d"] = materialize_dataset_session(view)
        output = _compute_processed_frame(
            spec,
            source=current_source,
            dependencies=frames,
            trading_sessions=sessions,
            panel_sessions=pd.DatetimeIndex(core),
            asset_market_types=asset_types if panel_source else None,
            suspension_history=history,
            financial_events=events,
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
        yield output
        del output, frames, current_source


def _benchmark_frames(source, session):
    sessions, assets = _processed_axis(
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
    session, *, panel_source, asset_column=None, calendar=None, validation_window=None
):
    if asset_column is None:
        asset_column = "source_asset_id" if panel_source else "asset_id"
    columns = (
        (asset_column, "timestamp", "market_type", "filling_flag")
        if panel_source
        else (asset_column, "timestamp")
    )
    view = replace(session, logical_columns=columns, physical_columns=columns)
    dates = set()
    assets = {}
    with closing_iterator(iter_dataset_chunks(view, batch_rows=65536)) as chunks:
        for frame in chunks:
            if panel_source:
                frame["market_type"] = strict_integer_series(
                    frame["market_type"], error="invalid_market_type", allow_null=True
                )
                frame = frame.loc[frame["market_type"].isin(CURRENT_MARKET_TYPES)]
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
                frame = frame.loc[frame["filling_flag"].ne(2)]
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
    return (sessions, asset_types)


def _compute_processed_frame(
    spec,
    *,
    source,
    dependencies,
    trading_sessions,
    panel_sessions=None,
    asset_market_types=None,
    suspension_history=None,
    financial_events=None,
):
    if spec.data_id == "market_panel_1d":
        required = {
            "market_bars_1d",
            "stock_master",
            "market_cap",
            "trading_calendar",
            "trading_no_limit",
            "special_treatment_changes",
            "a_share_industry_snapshots",
        }
        if not required.issubset(source):
            raise ValueError("a_share_panel_source_data_missing")
        panel = compute_market_panel(
            source["market_bars_1d"],
            stock_master=source["stock_master"],
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
            industry_history=source.get("industry_classification"),
            industry_sets=source.get("industry_classification_sets"),
            industry_versions=source.get("industry_classification_versions"),
            industry_changes=source.get("stock_info_changes"),
            industry_snapshots=source["a_share_industry_snapshots"],
            no_limit_history=source.get("trading_no_limit"),
            special_treatment_history=source.get("special_treatment_changes"),
            session_window=panel_sessions,
            asset_market_types=asset_market_types,
            suspension_history=suspension_history,
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
            source=source,
            panel=dependencies.get("market_panel_1d"),
            events=financial_events,
        )
    if spec.kind is ProcessedDataKind.FACTOR:
        definitions = tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
        if "expectations" in {definition.family for definition in definitions}:
            analyst = source.get("analyst_forecast")
            if analyst is None or analyst.empty:
                raise ValueError("BLOCKED_EXTERNAL_DATA: analyst_forecast")
            records = analyst.rename(columns={"source_asset_id": "asset_id"})
            panel = dependencies["market_panel_1d"]
            features = build_expectations_features(
                records,
                panel.loc[:, ["asset_id", "timestamp", "close"]],
                trading_sessions=trading_sessions,
            )
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
        if any(definition.family == "fundamental" for definition in definitions):
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
        return label_engine.calculate_labels_for_panel(
            panel, definitions=definitions, sessions=trading_sessions
        ).loc[:, list(spec.required_columns)]
    raise ValueError(f"unsupported_a_share_processed_kind: {spec.kind}")


def _financial_pit_events(source):
    required = {
        "financial_balance",
        "financial_income",
        "financial_cashflow",
        "financial_first_disclosure",
    }
    if not required.issubset(source):
        raise ValueError("BLOCKED_EXTERNAL_DATA: financial_first_disclosure")
    calendar = source.get("trading_calendar")
    if calendar is None:
        raise ValueError("fundamental_pit_calendar_required")
    sessions = unified_trading_sessions(calendar)
    events = financial_events_from_interim(
        {
            data_id: source[data_id]
            for data_id in (
                "financial_balance",
                "financial_income",
                "financial_cashflow",
            )
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
    source: dict[str, pd.DataFrame],
    panel: pd.DataFrame | None,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if panel is None:
        raise ValueError("fundamental_pit_market_panel_required")
    if events is None:
        events = _financial_pit_events(source)
    pit = materialize_pit_fundamentals(panel.loc[:, ["asset_id", "timestamp"]], events)
    values = pit.pivot_table(
        index=["asset_id", "timestamp"],
        columns="metric_id",
        values="value",
        aggfunc="first",
    ).reset_index()
    beginning = (
        pit.loc[:, ["asset_id", "timestamp", "metric_id", "beginning_value"]]
        .pivot_table(
            index=["asset_id", "timestamp"],
            columns="metric_id",
            values="beginning_value",
            aggfunc="first",
        )
        .reset_index()
    )
    beginning = beginning.rename(
        columns={
            column: f"beginning_{column}"
            for column in beginning.columns
            if column not in {"asset_id", "timestamp"}
        }
    )
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
    output = (
        panel.loc[:, ["market_id", "asset_id", "timestamp"]]
        .merge(values, on=["asset_id", "timestamp"], how="left", validate="one_to_one")
        .merge(
            beginning, on=["asset_id", "timestamp"], how="left", validate="one_to_one"
        )
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
