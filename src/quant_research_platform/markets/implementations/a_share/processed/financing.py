"""Publish financing proxies from structured disclosures and explicit decisions."""

from contextlib import contextmanager

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.research.factors.financing import RECIPES, calculate
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
        optional_source_input_data_ids=("trading_calendar",),
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    source = open_dataset_session(**_dataset_input_options(inputs["decision_panel"]))
    if not {"signal_session", "execution_session", "return_available_at"} <= set(
        source.logical_columns
    ):
        raise ValueError("financing_requires_explicit_decision_panel")
    frame = materialize_dataset_session(source.filtered_view(time_window))
    sessions = None
    if "trading_calendar" in inputs:
        sessions = unified_trading_sessions(
            materialize_dataset_session(
                open_dataset_session(
                    **_dataset_input_options(inputs["trading_calendar"])
                )
            )
        )
    selected = tuple(
        name for name, role in spec.column_roles.items() if role == "factor"
    )
    start = frame.signal_session.min()
    if sessions is not None and len(frame):
        position = sessions.get_indexer([start])[0]
        if position >= 0:
            start = sessions[max(0, position - 5)]
    observations = (
        materialize_dataset_session(
            open_dataset_session(
                **_dataset_input_options(inputs["public_margin_balances"]),
            ).filtered_view(TimeWindow(start, frame.signal_session.max()))
        )
        if len(frame)
        else None
    )
    metadata = {
        "producer_diagnostics": {
            "observation_match": "exact_signal_session",
            "release_clock": "available_at_UTC",
            "missing_calendar_affects": [
                name for name in selected if "balance_change" in name
            ]
            if sessions is None
            else [],
            "unknown_release_rows": int(observations.available_at.isna().sum())
            if observations is not None
            else 0,
        }
    }
    yield (
        (calculate(frame, observations, factor_columns=selected, sessions=sessions),)
        if len(frame)
        else (),
        metadata,
    )
