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
    unified_trading_sessions,
)
from quant_research_platform.research.factors.financing import (
    BALANCE_CHANGE_WINDOWS,
    RECIPES,
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
        column_units=dict.fromkeys(RECIPES, "dimensionless"),
        processed_input_data_ids=("decision_panel",),
        source_input_data_ids=("public_margin_balances", "trading_calendar"),
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
    frame = materialize_dataset_session(
        source.filtered_view(
            TimeWindow(
                pd.Timestamp.min, end_at if closed else end_at - pd.Timedelta(1, "ns")
            )
        )
    )
    sessions = None
    if "trading_calendar" in inputs:
        sessions = unified_trading_sessions(
            materialize_dataset_session(
                open_dataset_session(
                    **_dataset_input_options(inputs["trading_calendar"])
                )
            )
        )
    start = frame.signal_session.min()
    if sessions is not None and len(frame):
        position = sessions.get_indexer([start])[0]
        if position >= 0:
            lookback = max(BALANCE_CHANGE_WINDOWS.get(name, 0) for name in selected)
            start = sessions[
                max(0, position - lookback - config["observation_lag_sessions"])
            ]
    fields = tuple(dict.fromkeys(RECIPES[name][0] for name in selected))
    observations = (
        materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(inputs["public_margin_balances"]),
            )
            .filtered_view(TimeWindow(start, frame.signal_session.max()))
            .projected_view(("asset_id", "timestamp", "available_at", *fields))
        )
        if len(frame)
        else None
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
            frame, observations, factor_columns=selected, sessions=sessions, **config
        )
        result = result.loc[
            result.timestamp.ge(start_at)
            & (result.timestamp.le(end_at) if closed else result.timestamp.lt(end_at))
        ]
        yield (result,), metadata
    else:
        yield (), metadata
