"""Align explicit daily decisions and macro releases on canonical UTC instants."""

import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import time

import pandas as pd

from quant_research_platform.contracts.columns import PLATFORM_ASSET_TIME_KEY
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.features import inherited_feature_transforms
from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.models import _target_definition
from quant_research_platform.contracts.time import (
    DECISION_CONTEXT_COLUMNS,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors.macro_liquidity.calculations import (
    calculate_macro_factors,
    macro_states,
    parameters,
)
from quant_research_platform.research.factors.macro_liquidity.definitions import (
    RECIPES,
    factor_unit,
)
from quant_research_platform.runtime.dataframes import iter_complete_time_frames
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)


def processed_specs():
    keys = {
        name: "timestamp" if name == "timestamp" else "key"
        for name in PLATFORM_ASSET_TIME_KEY
    }
    return (
        ProcessedDataSpec(
            data_id="decision_panel",
            market_id="a_share",
            kind="panel",
            primary_key=PLATFORM_ASSET_TIME_KEY,
            partition_columns=("year",),
            frequency="1d",
            grain="asset_timestamp",
            column_roles=keys,
            processed_input_data_ids=("daily_dataset",),
            source_input_data_ids=("trading_calendar",),
        ),
        ProcessedDataSpec(
            data_id="macro_liquidity_factors",
            market_id="a_share",
            kind="factor",
            primary_key=PLATFORM_ASSET_TIME_KEY,
            partition_columns=("year",),
            frequency="1d",
            grain="asset_timestamp",
            column_roles={**keys, **dict.fromkeys(RECIPES, "factor")},
            column_units={name: factor_unit(name) for name in RECIPES},
            processed_input_data_ids=("decision_panel",),
            source_input_data_ids=("macro_observations",),
        ),
    )


def decision_parameters(values):
    if set(values) != {"mode", "local_time", "latency_seconds"} or values[
        "mode"
    ] not in {"close", "preopen"}:
        raise ValueError("decision_clock_parameters_invalid")
    local = time.fromisoformat(values["local_time"])
    latency = values["latency_seconds"]
    if (
        local.tzinfo is not None
        or type(latency) not in (int, float)
        or not 0 <= latency < float("inf")
    ):
        raise ValueError("decision_clock_parameters_invalid")
    offset = pd.Timedelta(
        hours=local.hour,
        minutes=local.minute,
        seconds=local.second,
        microseconds=local.microsecond,
    )
    if values["mode"] == "preopen":
        if offset + pd.Timedelta(seconds=latency) >= pd.Timedelta(hours=9, minutes=25):
            raise ValueError("decision_must_arrive_before_opening_auction")
    elif offset < pd.Timedelta(hours=15, minutes=30):
        raise ValueError("daily_close_decision_precedes_completed_daily_bar")
    return offset


def configured_spec(spec, config):
    values = config.get("parameters", {})
    if spec.data_id == "decision_panel":
        if {"factor_columns", "label_columns", "base_panel"} & config.keys():
            raise ValueError("decision_panel_uses_input_projection")
        decision_parameters(values)
        source = open_dataset_session(
            **_dataset_input_options(config["inputs"]["daily_dataset"])
        )
        declared = source.spec
        if (
            declared.get("market_id") != "a_share"
            or declared["frequency"] != "1d"
            or tuple(declared["primary_key"]) != PLATFORM_ASSET_TIME_KEY
            or set(DECISION_CONTEXT_COLUMNS) & set(source.logical_columns)
        ):
            raise ValueError("decision_panel_requires_daily_session_dataset")
        roles = {
            name: declared["column_roles"][name] for name in source.logical_columns
        }
        roles.update(dict.fromkeys(DECISION_CONTEXT_COLUMNS, "context"))
        return replace(
            spec,
            column_roles=roles,
            column_units={
                name: unit
                for name, unit in declared.get("column_units", {}).items()
                if name in roles
            },
            label_availability_columns={
                name: column
                for name, column in declared.get(
                    "label_availability_columns", {}
                ).items()
                if name in roles and column in roles
            },
            parameters_json=json.dumps(values),
        )
    parameters(values)
    columns = config.get("factor_columns", list(RECIPES))
    if (
        type(columns) is not list
        or not columns
        or len(set(columns)) != len(columns)
        or not set(columns) <= RECIPES.keys()
        or {"label_columns", "base_panel"} & config.keys()
    ):
        raise ValueError("macro_factor_selection_invalid")
    return replace(
        spec,
        column_roles={
            name: role
            for name, role in spec.column_roles.items()
            if name in PLATFORM_ASSET_TIME_KEY or name in columns
        },
        column_units={name: factor_unit(name) for name in columns},
        parameters_json=json.dumps(values),
    )


def decision_panel(frame, sessions, config):
    """Replace session keys with decision instants; retain next-session labels."""
    offset = decision_parameters(config)
    dates = pd.DatetimeIndex(frame.timestamp)
    positions = sessions.get_indexer(dates)
    if (
        not dates.equals(dates.normalize())
        or (positions < 0).any()
        or (positions + 1 >= len(sessions)).any()
    ):
        raise ValueError("decision_panel_calendar_requires_current_and_next_session")
    execution = sessions.take(positions + 1)
    actual_dates = dates if config["mode"] == "close" else execution
    actual = (
        (actual_dates + offset)
        .tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
        .tz_localize(None)
    )
    arrive = actual + pd.Timedelta(seconds=config["latency_seconds"])
    auction = (
        (execution + pd.Timedelta(hours=9, minutes=25))
        .tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
        .tz_localize(None)
    )
    if (arrive >= auction).any():
        raise ValueError("decision_must_arrive_before_opening_auction")
    result = frame.copy()
    result["signal_session"] = dates
    result["execution_session"] = execution
    result["return_available_at"] = (
        (dates + pd.Timedelta(hours=15))
        .tz_localize("Asia/Shanghai")
        .tz_convert("UTC")
        .tz_localize(None)
    )
    result["timestamp"] = actual
    result["order_submit_at"] = arrive
    return result


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    config = json.loads(spec.parameters_json)
    panel_key = (
        "daily_dataset" if spec.data_id == "decision_panel" else "decision_panel"
    )
    session = open_dataset_session(**_dataset_input_options(inputs[panel_key]))
    metadata = {"producer_diagnostics": {}}
    if spec.data_id == "decision_panel":
        inherited = session.manifest.get("producer_diagnostics", {})
        metadata["producer_diagnostics"] = {
            "label_semantics": {
                name: facts
                for name, facts in inherited.get("label_semantics", {}).items()
                if name in session.logical_columns
                or "model_target" in session.logical_columns
            },
            "reused_feature_transforms": inherited_feature_transforms(
                session.spec, session.manifest, session.logical_columns
            ),
            "target_definitions": {
                "model_target": _target_definition(
                    session.spec, "model_target", session.manifest
                )
            }
            if "model_target" in session.logical_columns
            else {},
        }
        calendar = materialize_dataset_session(
            open_dataset_session(**_dataset_input_options(inputs["trading_calendar"]))
        )
        sessions = unified_trading_sessions(calendar)
        # Select original signal sessions; the calendar includes their successors.
        source = session.filtered_view(time_window)
        with closing_iterator(iter_dataset_chunks(source)) as chunks:
            yield (
                (decision_panel(frame, sessions, config) for frame in chunks),
                metadata,
            )
    else:
        validate_input_market(session.spec, spec.market_id)
        observations = materialize_dataset_session(
            open_dataset_session(**_dataset_input_options(inputs["macro_observations"]))
        )
        columns = tuple(
            name for name, role in spec.column_roles.items() if role == "factor"
        )
        declared = set(session.logical_columns)
        if not set(DECISION_CONTEXT_COLUMNS) <= declared:
            raise ValueError("macro_factors_require_explicit_decision_panel")
        beta = any(RECIPES[name][0] == "beta" for name in columns)
        required = set(PLATFORM_ASSET_TIME_KEY) | {"eligible"}
        required.update(
            RECIPES[name][2] for name in columns if RECIPES[name][0] == "interaction"
        )
        if beta:
            required.update(
                (
                    "listing_date",
                    "return_available_at",
                    "period_return",
                    "market_return",
                )
            )
        session = session.projected_view(
            tuple(name for name in session.logical_columns if name in required)
        )
        if not beta:
            session = session.filtered_view(time_window)
        cutoffs = set()
        with closing_iterator(
            iter_dataset_chunks(session.projected_view(("timestamp",)))
        ) as chunks:
            for frame in chunks:
                cutoffs.update(frame.timestamp.unique())
        states = macro_states(
            observations, sorted(cutoffs), columns=columns, config=parameters(config)
        )
        start, end, closed = time_window_timestamp_interval(time_window)
        metadata["producer_diagnostics"]["feature_preserve_amplitude"] = [
            name for name in columns if RECIPES[name][0] != "beta"
        ]

        def frames():
            source = (
                iter((materialize_dataset_session(session),))
                if beta
                else iter_complete_time_frames(
                    iter_dataset_chunks(session), timestamp_column="timestamp"
                )
            )
            with closing_iterator(source) as chunks:
                for panel in chunks:
                    result = calculate_macro_factors(
                        observations,
                        panel,
                        columns=columns,
                        config=config,
                        states=states,
                    )
                    yield result.loc[
                        result.timestamp.ge(start)
                        & (
                            result.timestamp.le(end)
                            if closed
                            else result.timestamp.lt(end)
                        )
                    ]

        yield frames(), metadata
