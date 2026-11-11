"""Native publication of explicit cross-market factor hypotheses."""

import json
from contextlib import contextmanager
from dataclasses import replace

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.research.factors import fund_flows, linked_markets
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)

KEY = ("market_id", "asset_id", "timestamp")
LIBRARIES = {"linked_market_factors": linked_markets, "fund_flow_factors": fund_flows}


def processed_spec(data_id):
    library = LIBRARIES[data_id]
    sources = tuple(
        dict.fromkeys(
            source
            for dependencies in library.RECIPES.values()
            for source in dependencies
            if source != "market_panel_1d"
        )
    )
    if data_id == "fund_flow_factors":
        sources = (*sources, "trading_calendar")
    return ProcessedDataSpec(
        data_id=data_id,
        market_id="a_share",
        kind="factor",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=KEY,
        partition_columns=("year",),
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(library.RECIPES, "factor"),
        },
        column_units=library.UNITS,
        processed_input_data_ids=("decision_panel",),
        source_input_data_ids=sources,
        optional_source_input_data_ids=sources,
    )


def configured_spec(spec, config):
    library = LIBRARIES[spec.data_id]
    params = config.get("parameters", {})
    allowed = (
        {"us_assets", "asset_us_tickers"}
        if spec.data_id == "fund_flow_factors"
        else {
            "future_product",
            "pair_assets",
            "industry_etfs",
            "us_activity_asset",
            "us_assets",
            "asset_us_tickers",
            "index_id",
        }
    )
    if type(params) is not dict or set(params) - allowed:
        raise ValueError("linked_market_parameters_invalid")
    if params.get("future_product", "IF") not in {"IF", "IC", "IH", "IM"}:
        raise ValueError("linked_future_product_invalid")
    if type(params.get("index_id", "000300")) is not str or not params.get(
        "index_id", "000300"
    ):
        raise ValueError("linked_index_id_invalid")
    for field in ("pair_assets", "industry_etfs", "us_assets", "asset_us_tickers"):
        mapping = params.get(field, {})
        if type(mapping) is not dict or any(
            type(k) is not str or type(v) is not str or not k or not v
            for k, v in mapping.items()
        ):
            raise ValueError("linked_asset_mapping_invalid")
    if type(params.get("us_activity_asset", "SPY")) is not str or not params.get(
        "us_activity_asset", "SPY"
    ):
        raise ValueError("linked_activity_asset_invalid")
    selected = config.get("factor_columns", list(library.RECIPES))
    if (
        type(selected) is not list
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= library.RECIPES.keys()
    ):
        raise ValueError("data_operation_factor_columns_invalid")
    dependencies = {source for name in selected for source in library.RECIPES[name]}
    sources = tuple(name for name in spec.source_input_data_ids if name in dependencies)
    if spec.data_id == "fund_flow_factors" and "trading_calendar" in config["inputs"]:
        sources = (*sources, "trading_calendar")
    return replace(
        spec,
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(selected, "factor"),
        },
        column_units={name: library.UNITS[name] for name in selected},
        processed_input_data_ids=(
            "decision_panel",
            *(
                ("market_panel_1d",)
                if "market_panel_1d" in config["inputs"]
                and "market_panel_1d" in dependencies
                else ()
            ),
        ),
        source_input_data_ids=sources,
        optional_source_input_data_ids=sources,
        parameters_json=json.dumps(params),
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    start, end, closed = time_window_timestamp_interval(time_window)
    prefix = TimeWindow(
        pd.Timestamp.min, end if closed else end - pd.Timedelta(1, "ns")
    )
    source = open_dataset_session(**_dataset_input_options(inputs["decision_panel"]))
    if not {"signal_session", "execution_session", "return_available_at"} <= set(
        source.logical_columns
    ):
        raise ValueError("linked_markets_require_explicit_decision_panel")
    context = (
        *KEY,
        "signal_session",
        "period_return",
        "market_return",
        "industry_level1_code",
        "industry_standard_id",
        "hedging_beta",
        "implied_variance_20d",
        "implied_variance_60d",
        "volatility_20d",
        "volatility_60d",
        "close",
        "secondary_flow_imbalance",
        "a_float_shares",
        "avg_volume_20d",
        "peak_drawdown_20d",
        "amihud_20d",
        "amihud_60d",
    )
    frame = materialize_dataset_session(
        source.filtered_view(prefix).projected_view(
            tuple(c for c in context if c in source.logical_columns)
        )
    )
    for width in (20, 60):
        iv, rv = f"implied_variance_{width}d", f"volatility_{width}d"
        name = f"implied_minus_past_realized_variance_{width}d"
        if spec.column_roles.get(name) == "factor" and {iv, rv} <= set(frame):
            if (
                source.spec.get("column_units", {}).get(iv) != "annualized_variance"
                or source.spec.get("column_units", {}).get(rv) != "decimal_return"
            ):
                raise ValueError("linked_implied_variance_units_invalid")
    frame["timestamp"] = (
        pd.to_datetime(frame.timestamp, utc=True)
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    sources = {}
    for name, path in inputs.items():
        if name == "decision_panel":
            continue
        session = open_dataset_session(**_dataset_input_options(path)).filtered_view(
            prefix
        )
        if name == "market_panel_1d":
            session = session.projected_view(
                (
                    "asset_id",
                    "timestamp",
                    "adj_close",
                    *(
                        ("signal_session",)
                        if "signal_session" in session.logical_columns
                        else ()
                    ),
                )
            )
        value = materialize_dataset_session(session)
        if name == "market_panel_1d" and "signal_session" in value:
            value["timestamp"] = value.pop("signal_session")
        if "available_at" in value:
            value["available_at"] = (
                pd.to_datetime(value.available_at, utc=True)
                .dt.tz_localize(None)
                .astype("datetime64[ns]")
            )
        sources[name] = value
    selected = tuple(
        name for name, role in spec.column_roles.items() if role == "factor"
    )
    result = LIBRARIES[spec.data_id].calculate(
        frame, sources, columns=selected, parameters=json.loads(spec.parameters_json)
    )
    result = result.astype(dict.fromkeys(selected, "Float64"))
    metadata = {"producer_diagnostics": result.attrs}
    yield (
        (
            result.loc[
                result.timestamp.ge(start)
                & (result.timestamp.le(end) if closed else result.timestamp.lt(end))
            ],
        ),
        metadata,
    )
