"""Publish intraday research factors from existing panels, without making panels."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    unified_trading_sessions,
)
from quant_research_platform.research.factors.intraday_context import (
    DAILY_FIELDS,
    FACTORS,
    KEY,
    UNITS,
    calculate,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def processed_spec():
    return ProcessedDataSpec(
        data_id="hft_research_factors",
        market_id="a_share",
        kind="factor",
        frequency="tick",
        grain="asset_timestamp",
        primary_key=KEY,
        partition_columns=("year",),
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(FACTORS, "factor"),
        },
        column_units=UNITS,
        processed_input_data_ids=("hft_panel",),
        source_input_data_ids=("hft_trades", "trading_calendar"),
        optional_source_input_data_ids=("hft_trades", "trading_calendar"),
    )


def configured_spec(spec, config):
    selected = config.get("factor_columns", list(FACTORS))
    params = config.get("parameters", {})
    if (
        type(selected) is not list
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(FACTORS)
    ):
        raise ValueError("data_operation_factor_columns_invalid")
    if type(params) is not dict or set(params) - {"linked_assets"}:
        raise ValueError("intraday_factor_parameters_invalid")
    mapping = params.get("linked_assets", {})
    if type(mapping) is not dict or any(
        type(k) is not str or type(v) is not str or not k or not v
        for k, v in mapping.items()
    ):
        raise ValueError("intraday_linked_assets_invalid")
    return replace(
        spec,
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(selected, "factor"),
        },
        column_units={name: UNITS[name] for name in selected},
        processed_input_data_ids=(
            "hft_panel",
            *(
                name
                for name in ("daily_context", "linked_hft_panel")
                if name in config["inputs"]
            ),
        ),
        parameters_json=json.dumps(params),
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    lower, upper, closed = time_window_timestamp_interval(time_window)
    sessions = {
        name: open_dataset_session(**_dataset_input_options(path))
        for name, path in inputs.items()
    }
    if sessions["hft_panel"].spec["data_id"] != "hft_panel":
        raise ValueError("intraday_requires_hft_panel")
    calendar = None
    if "trading_calendar" in sessions:
        calendar = unified_trading_sessions(
            materialize_dataset_session(sessions["trading_calendar"])
        )
    selected = tuple(
        name for name, role in spec.column_roles.items() if role == "factor"
    )
    metadata = {
        "producer_diagnostics": {
            "missing_inputs": sorted(
                {"daily_context", "hft_trades", "linked_hft_panel", "trading_calendar"}
                - inputs.keys()
            ),
            "new_panel_generation": False,
            "peer_cohort": "same_observed_timestamp",
            "profile_bin_log_width": "log(1.002)",
            "profile_value_area_fraction": 0.7,
        }
    }

    def frames():
        for day in pd.date_range(
            pd.Timestamp(lower).normalize(),
            (pd.Timestamp(upper) - pd.Timedelta(0 if closed else 1, "ns")).normalize(),
            freq="D",
        ):
            if calendar is not None:
                position = calendar.searchsorted(day)
                if position == len(calendar) or calendar[position] != day:
                    continue
                first = calendar[max(0, position - 20)]
                trade_first = calendar[max(0, position - 4)]
            else:
                first = trade_first = day
            end = min(
                day + pd.Timedelta(days=1) - pd.Timedelta(1, "ns"),
                pd.Timestamp(upper) - pd.Timedelta(0 if closed else 1, "ns"),
            )
            window = TimeWindow(first, end)
            panel_columns = (
                *KEY,
                "mid",
                "segment_id",
                "bid_px1",
                "ask_px1",
                "bid_vol1",
                "ask_vol1",
                "observed_trade_volume",
                "observed_signed_trade_volume",
                "exchange",
                "instrument_type",
            )
            if all(name.startswith("trade_profile_") for name in selected):
                panel_columns = (*KEY, "mid")
            panel = materialize_dataset_session(
                sessions["hft_panel"]
                .filtered_view(window)
                .projected_view(panel_columns)
            )
            if panel.empty:
                continue
            daily = None
            if "daily_context" in sessions:
                source = sessions["daily_context"]
                if not {
                    "signal_session",
                    "execution_session",
                    "return_available_at",
                } <= set(source.logical_columns):
                    raise ValueError(
                        "intraday_daily_context_requires_explicit_decision_clock"
                    )
                fields = (
                    *KEY,
                    "signal_session",
                    *(name for name in DAILY_FIELDS if name in source.logical_columns),
                )
                daily = materialize_dataset_session(
                    source.filtered_view(TimeWindow(first, end)).projected_view(fields)
                )
                daily["timestamp"] = pd.to_datetime(
                    daily.timestamp, utc=True
                ).dt.tz_localize(None)
            trades = None
            if (
                any(name.startswith("trade_profile_") for name in selected)
                and "hft_trades" in sessions
            ):
                trades = materialize_dataset_session(
                    sessions["hft_trades"]
                    .filtered_view(TimeWindow(trade_first, end))
                    .projected_view(
                        (
                            "asset_id",
                            "timestamp",
                            "source_row",
                            "event_kind",
                            "price",
                            "volume",
                        )
                    )
                )
            linked = None
            if (
                any(name.startswith("lagged_linked_") for name in selected)
                and "linked_hft_panel" in sessions
            ):
                linked = materialize_dataset_session(
                    sessions["linked_hft_panel"]
                    .filtered_view(TimeWindow(day, end))
                    .projected_view(("asset_id", "timestamp", "segment_id", "mid"))
                )
            result = calculate(
                panel,
                columns=selected,
                daily_context=daily,
                trades=trades,
                linked_panel=linked,
                session_days=calendar,
                **json.loads(spec.parameters_json),
            )
            mask = result.timestamp.ge(
                max(day, pd.Timestamp(lower))
            ) & result.timestamp.le(end)
            yield result.loc[mask]

    yield frames(), metadata
