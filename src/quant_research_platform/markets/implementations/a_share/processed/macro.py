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
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    next_trading_sessions,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors.macro_liquidity.calculations import (
    BETA_INPUT_COLUMNS,
    calculate_macro_factors,
    macro_states,
    parameters,
    required_factors,
)
from quant_research_platform.research.factors.macro_liquidity.definitions import (
    PUBLIC_SERIES,
    RECIPES,
    factor_unit,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    iter_dataset_time_chunks,
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
            source_input_data_ids=(
                "macro_observations",
                "trading_calendar",
                *dict.fromkeys(recipe[0] for recipe in PUBLIC_SERIES.values()),
            ),
            optional_source_input_data_ids=(
                "macro_observations",
                "trading_calendar",
                *dict.fromkeys(recipe[0] for recipe in PUBLIC_SERIES.values()),
            ),
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


def decision_panel(frame, calendar, config):
    """Replace session keys with decision instants; retain next-session labels."""
    offset = decision_parameters(config)
    dates = pd.DatetimeIndex(frame.timestamp)
    if not dates.equals(dates.normalize()):
        raise ValueError("decision_panel_requires_session_dates")
    if "market_type" in frame:
        execution = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        for market, group in frame.groupby("market_type", sort=False):
            execution.loc[group.index] = next_trading_sessions(
                calendar.loc[calendar.source_market_type.eq(market)],
                pd.DatetimeIndex(group.timestamp),
            )
        execution = pd.DatetimeIndex(execution)
    else:
        execution = next_trading_sessions(calendar, dates)
    valid = execution.notna()
    frame, dates, execution = frame.loc[valid], dates[valid], execution[valid]
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
def produce(spec, *, inputs, time_window):
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
        # Select original signal sessions; the calendar includes their successors.
        source = session.filtered_view(time_window)
        with closing_iterator(iter_dataset_chunks(source)) as chunks:
            yield (
                (decision_panel(frame, calendar, config) for frame in chunks),
                metadata,
            )
    else:
        validate_input_market(session.spec, spec.market_id)
        columns = tuple(
            name for name, role in spec.column_roles.items() if role == "factor"
        )
        from quant_research_platform.research.factors.macro_liquidity.public_observations import (  # noqa: E501
            observations_for_series,
        )

        needed = required_factors(columns, parameters(config))
        public_names = {
            RECIPES[name][1]
            for name in needed
            if RECIPES[name][0] == "series" and RECIPES[name][1] in PUBLIC_SERIES
        }
        needs_macro = any(
            RECIPES[name][0] == "series" and RECIPES[name][1] not in PUBLIC_SERIES
            for name in needed
        )
        observations = (
            materialize_dataset_session(
                open_dataset_session(
                    **_dataset_input_options(inputs["macro_observations"])
                ).projected_view(
                    ("series_id", "observation_date", "timestamp", "value")
                )
            )
            if needs_macro and "macro_observations" in inputs
            else observations_for_series(None, ())
        )
        missing_public = []
        public_clocks = {}
        for table in dict.fromkeys(
            PUBLIC_SERIES[name][0] for name in sorted(public_names)
        ):
            if table not in inputs:
                missing_public.append(table)
                continue
            names = [
                name for name in sorted(public_names) if PUBLIC_SERIES[name][0] == table
            ]
            fields = tuple(
                dict.fromkeys(
                    field for name in names for field in PUBLIC_SERIES[name][2]
                )
            )
            public_session = open_dataset_session(
                **_dataset_input_options(inputs[table])
            )
            facts = materialize_dataset_session(
                public_session.projected_view(
                    (
                        "asset_id",
                        "timestamp",
                        "available_at",
                        "availability_basis",
                        *fields,
                    )
                )
            )
            public_clocks[table] = {
                "unknown_availability_rows": int(facts.available_at.isna().sum()),
                "estimated_availability_rows": int(
                    (
                        facts.available_at.notna()
                        & facts.availability_basis.ne("publication_timestamp")
                    ).sum()
                ),
            }
            converted = observations_for_series(facts, names)
            public_clocks[table]["input_invalid_numeric_cells"] = converted.attrs[
                "input_invalid_numeric_cells"
            ]
            observations = pd.concat((observations, converted), ignore_index=True)
        metadata["producer_diagnostics"]["missing_public_sources"] = missing_public
        metadata["producer_diagnostics"]["public_source_clocks"] = public_clocks
        metadata["producer_diagnostics"]["missing_macro_observations"] = (
            needs_macro and "macro_observations" not in inputs
        )
        declared = set(session.logical_columns)
        if (
            not set(DECISION_CONTEXT_COLUMNS).difference({"return_available_at"})
            <= declared
        ):
            raise ValueError("macro_factors_require_explicit_decision_panel")
        beta = any(RECIPES[name][0] == "beta" for name in needed)
        required = set(PLATFORM_ASSET_TIME_KEY) | {"eligible"}
        required.update(
            RECIPES[name][2]
            for name in columns
            if RECIPES[name][0] in {"interaction", "industry_interaction"}
        )
        if any(RECIPES[name][0] == "industry_interaction" for name in columns):
            required.update(("industry_standard_id", "industry_level1_code"))
        if beta:
            required.update((*BETA_INPUT_COLUMNS, "market_return"))
        session = session.projected_view(
            tuple(name for name in session.logical_columns if name in required)
        )
        start, end, closed = time_window_timestamp_interval(time_window)
        session = session.filtered_view(
            TimeWindow(pd.Timestamp.min, end if closed else end - pd.Timedelta(1, "ns"))
            if beta
            else time_window
        )
        calendar = (
            materialize_dataset_session(
                open_dataset_session(
                    **_dataset_input_options(inputs["trading_calendar"])
                )
            )
            if beta and "trading_calendar" in inputs
            else None
        )
        if beta:
            missing = set(BETA_INPUT_COLUMNS) - declared
            if (
                any(
                    RECIPES[name][2] == "partial_beta"
                    for name in columns
                    if RECIPES[name][0] == "beta"
                )
                and "market_return" not in declared
            ):
                missing.add("market_return")
            if calendar is None:
                missing.add("trading_calendar")
            metadata["producer_diagnostics"]["macro_beta_missing_inputs"] = sorted(
                missing
            )
        states = None
        if not beta:
            cutoffs = set()
            with closing_iterator(
                iter_dataset_chunks(session.projected_view(("timestamp",)))
            ) as chunks:
                for frame in chunks:
                    cutoffs.update(frame.timestamp.unique())
            states = macro_states(
                observations,
                sorted(cutoffs),
                columns=columns,
                config=parameters(config),
            )
        metadata["producer_diagnostics"]["feature_preserve_amplitude"] = [
            name for name in columns if RECIPES[name][0] != "beta"
        ]

        def frames():
            source = (
                iter((materialize_dataset_session(session),))
                if beta
                else iter_dataset_time_chunks(session)
            )
            with closing_iterator(source) as chunks:
                for panel in chunks:
                    result = calculate_macro_factors(
                        observations,
                        panel,
                        columns=columns,
                        config=config,
                        states=states,
                        calendar=calendar,
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
