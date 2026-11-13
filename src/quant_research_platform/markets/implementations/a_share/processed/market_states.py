"""Align trusted market-wide factors to one explicit benchmark decision axis."""

from contextlib import contextmanager
from dataclasses import replace

import pandas as pd

from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.factors.market_structure import (
    FACTOR_UNITS as STRUCTURE_UNITS,
)
from quant_research_platform.research.factors.market_structure import (
    MARKET_CONTEXT_FACTORS,
)
from quant_research_platform.research.factors.style_structure import (
    FACTOR_UNITS as STYLE_UNITS,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    iter_dataset_chunks,
    materialize_dataset_session,
    open_dataset_session,
)

KEY = ("market_id", "asset_id", "timestamp")
SOURCES = {
    **dict.fromkeys(MARKET_CONTEXT_FACTORS, "market_structure_factors"),
    **dict.fromkeys(STYLE_UNITS, "style_structure_factors"),
}
UNITS = {
    name: (STYLE_UNITS if name in STYLE_UNITS else STRUCTURE_UNITS)[name]
    for name in SOURCES
}


def processed_spec():
    return ProcessedDataSpec(
        data_id="market_state_factors",
        market_id="a_share",
        kind="factor",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=KEY,
        partition_columns=("year",),
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(SOURCES, "factor"),
        },
        column_units=UNITS,
        processed_input_data_ids=(
            "benchmark_daily_panel",
            "market_structure_factors",
            "style_structure_factors",
        ),
    )


def configured_spec(spec, config):
    selected = config.get("factor_columns", list(SOURCES))
    if (
        not isinstance(selected, list)
        or not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= SOURCES.keys()
        or config.get("parameters")
    ):
        raise ValueError("market_state_selection_invalid")
    return replace(
        spec,
        column_roles={
            **dict.fromkeys(KEY, "key"),
            "timestamp": "timestamp",
            **dict.fromkeys(selected, "factor"),
        },
        column_units={name: UNITS[name] for name in selected},
        processed_input_data_ids=(
            "benchmark_daily_panel",
            *dict.fromkeys(SOURCES[name] for name in selected),
        ),
    )


@contextmanager
def produce(spec, *, inputs, time_window):
    base = materialize_dataset_session(
        open_dataset_session(**_dataset_input_options(inputs["benchmark_daily_panel"]))
        .filtered_view(time_window)
        .projected_view((*KEY, "signal_session"))
    )
    selected = tuple(
        name for name, role in spec.column_roles.items() if role == "factor"
    )
    signals = pd.DatetimeIndex(base.signal_session.dropna().unique())
    for source in dict.fromkeys(SOURCES[name] for name in selected):
        fields = tuple(name for name in selected if SOURCES[name] == source)
        session = open_dataset_session(
            **_dataset_input_options(inputs[source])
        ).projected_view(("timestamp", *fields))
        chunks = []
        if not signals.empty:
            session = session.filtered_view(TimeWindow(signals.min(), signals.max()))
            with closing_iterator(iter_dataset_chunks(session)) as frames:
                for frame in frames:
                    chunks.append(
                        frame.loc[frame.timestamp.isin(signals)].drop_duplicates(
                            "timestamp"
                        )
                    )
        values = (
            pd.concat(chunks, ignore_index=True)
            .drop_duplicates("timestamp")
            .rename(columns={"timestamp": "signal_session"})
            if chunks
            else pd.DataFrame(
                {
                    "signal_session": pd.Series(dtype="datetime64[ns]"),
                    **{name: pd.Series(dtype="float64") for name in fields},
                }
            )
        )
        base = base.merge(
            values, on="signal_session", how="left", validate="many_to_one"
        )
    result = base.loc[:, [*KEY, *selected]]
    yield (
        (result,),
        {
            "producer_diagnostics": {
                "state_clock": "signal_session_to_explicit_benchmark_decision",
                "stock_replicas_counted_once": True,
            }
        },
    )
