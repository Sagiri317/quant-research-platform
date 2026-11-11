"""Publish financing proxies from structured disclosures and explicit decisions."""

import json
from contextlib import contextmanager

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    next_trading_sessions,
    unified_trading_sessions,
)
from quant_research_platform.research.factors.financing import (
    BALANCE_CHANGE_WINDOWS,
    FACTOR_UNITS,
    RECIPES,
    REPORT_IDENTITY,
    calculate,
    decision_columns,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def processed_spec():
    return ProcessedDataSpec(
        data_id="financing_factors",
        market_id="a_share",
        kind="factor",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        column_roles={
            "market_id": "key",
            "asset_id": "key",
            "timestamp": "timestamp",
            **dict.fromkeys(RECIPES, "factor"),
        },
        column_units=FACTOR_UNITS,
        processed_input_data_ids=("decision_panel",),
        source_input_data_ids=("public_margin_balances", "trading_calendar"),
        optional_source_input_data_ids=("trading_calendar",),
        parameters_json=json.dumps({"observation_lag_sessions": 1}),
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    source = open_dataset_session(**_dataset_input_options(inputs["decision_panel"]))
    if not {"signal_session", "execution_session", "return_available_at"} <= set(
        source.logical_columns
    ):
        raise ValueError("financing_requires_explicit_decision_panel")
    start_at, end_at, closed = time_window_timestamp_interval(time_window)
    config = json.loads(spec.parameters_json)
    selected = tuple(
        name for name, role in spec.column_roles.items() if role == "factor"
    )
    required = {"market_id", "asset_id", "timestamp", *decision_columns(selected)}
    source = source.projected_view(
        tuple(name for name in source.logical_columns if name in required)
    )
    frame = materialize_dataset_session(source.filtered_view(time_window))
    sessions = None
    session_successors = None
    calendar_needed = config["observation_lag_sessions"] or any(
        name in BALANCE_CHANGE_WINDOWS for name in selected
    )
    if calendar_needed and "trading_calendar" in inputs:
        calendar = materialize_dataset_session(
            open_dataset_session(**_dataset_input_options(inputs["trading_calendar"]))
        )
        sessions = unified_trading_sessions(calendar)
        session_successors = pd.Series(
            next_trading_sessions(calendar, sessions), index=sessions
        )
    start = frame.signal_session.min()
    if sessions is not None and len(frame):
        positions = sessions.get_indexer(frame.signal_session)
        valid = positions >= config["observation_lag_sessions"]
        if valid.any():
            position = positions[valid].min()
            # Only same-observation denominators need earlier decision rows.
            # The calendar, not observed row counts, locates that dependency.
            history_start = sessions[position - config["observation_lag_sessions"]]
            if (
                config["observation_lag_sessions"]
                and any(RECIPES[name][1] for name in selected)
                and history_start < start_at
            ):
                history = materialize_dataset_session(
                    source.filtered_view(
                        TimeWindow(history_start, start_at - pd.Timedelta(1, "ns"))
                    )
                )
                frame = pd.concat([history, frame], ignore_index=True)
            lookback = max(BALANCE_CHANGE_WINDOWS.get(name, 0) for name in selected)
            start = min(
                start,
                sessions[
                    max(0, position - lookback - config["observation_lag_sessions"])
                ],
            )
    fields = tuple(dict.fromkeys(RECIPES[name][0] for name in selected))
    observations = None
    if len(frame):
        reports = open_dataset_session(
            **_dataset_input_options(inputs["public_margin_balances"])
        )
        fields = tuple(
            c for c in (*REPORT_IDENTITY, *fields) if c in reports.logical_columns
        )
        observations = materialize_dataset_session(
            reports.filtered_view(
                TimeWindow(start, frame.signal_session.max())
            ).projected_view(fields)
        )
    calendar_dependents = (
        list(selected)
        if config["observation_lag_sessions"]
        else [name for name in selected if name in BALANCE_CHANGE_WINDOWS]
    )
    metadata = {
        "producer_diagnostics": {
            "observation_match": "exact_declared_lagged_session",
            "observation_lag_sessions": config["observation_lag_sessions"],
            "release_clock": "available_at_UTC",
            "missing_calendar_affects": calendar_dependents if sessions is None else [],
            "unknown_release_rows": int(observations.available_at.isna().sum())
            if observations is not None
            else 0,
        }
    }
    if len(frame):
        result = calculate(
            frame,
            observations,
            factor_columns=selected,
            sessions=sessions,
            session_successors=session_successors,
            **config,
        )
        metadata["producer_diagnostics"]["input_invalid_numeric_cells"] = result.attrs[
            "input_invalid_numeric_cells"
        ]
        metadata["producer_diagnostics"]["missing_context_columns"] = result.attrs[
            "missing_context_columns"
        ]
        result = result.loc[
            result.timestamp.ge(start_at)
            & (result.timestamp.le(end_at) if closed else result.timestamp.lt(end_at))
        ]
        yield (result,), metadata
    else:
        yield (), metadata
