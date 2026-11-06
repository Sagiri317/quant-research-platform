"""A-share quote panels and explicit high-frequency data preparation."""

import json
from contextlib import contextmanager

import numpy as np
import pandas as pd

from quant_research_platform.contracts.ids import validate_input_market
from quant_research_platform.contracts.portfolios import A_SHARE_INSTRUMENT_TYPES
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    continuous_session,
)
from quant_research_platform.research.factors.event_microstructure import (
    compute_event_factors,
)
from quant_research_platform.research.factors.microstructure import compute_factors
from quant_research_platform.research.labels.quote_returns import compute_labels
from quant_research_platform.research.protocols import a_share_hft as protocol
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)


def _flow(quotes, events, kind):
    """Recorded interval sums after the first event; no feed-completeness claim."""
    total = np.full(len(quotes), np.nan)
    signed = total.copy()
    if events.empty or len(quotes) < 2:
        return total, signed
    times = quotes.timestamp.to_numpy(dtype="datetime64[ns]")
    event_times = events.timestamp.to_numpy(dtype="datetime64[ns]")
    selected = events.event_kind.eq(kind).to_numpy()
    bins = np.searchsorted(times, event_times[selected], side="left")
    selected_events = events.loc[selected]
    within = (bins > 0) & (bins < len(quotes))
    bins = bins[within]
    quantities = selected_events.volume.to_numpy(dtype="float64")[within]
    sides = selected_events.side.map({"buy": 1.0, "sell": -1.0}).to_numpy(
        dtype="float64"
    )[within]
    counts = np.bincount(bins, weights=np.isnan(quantities), minlength=len(quotes))
    unknown_side = np.bincount(bins, weights=np.isnan(sides), minlength=len(quotes))
    amount = np.bincount(bins, weights=np.nan_to_num(quantities), minlength=len(quotes))
    direction = np.bincount(
        bins, weights=np.nan_to_num(quantities * sides), minlength=len(quotes)
    )
    covered = times[:-1] >= event_times.min()
    total[1:] = np.where(covered & (counts[1:] == 0), amount[1:], np.nan)
    signed[1:] = np.where(
        covered & (counts[1:] == 0) & (unknown_side[1:] == 0), direction[1:], np.nan
    )
    return total, signed


def build_panel(quotes, orders, trades, *, max_gap_seconds):
    quotes = (
        quotes.sort_values(["asset_id", "timestamp", "source_row"], kind="stable")
        .drop_duplicates(["asset_id", "timestamp"], keep="last")
        .copy()
    )
    quotes["instrument_type"] = quotes.get("instrument_type", "stock")
    if not quotes.instrument_type.isin(A_SHARE_INSTRUMENT_TYPES).all():
        raise ValueError("hft_instrument_type_invalid")
    quotes["session"] = continuous_session(
        quotes.timestamp, quotes.exchange, quotes.instrument_type
    )
    quotes = quotes.loc[quotes.session.ne("")]
    sources = {
        name: (
            frame,
            frame.groupby(
                [
                    "asset_id",
                    "session_date",
                    continuous_session(
                        frame.timestamp, frame.exchange, frame.get("instrument_type")
                    ),
                ],
                sort=False,
            ).indices,
        )
        for name, frame in (("order", orders), ("trade", trades))
    }
    output = []
    for (asset, day, session), group in quotes.groupby(
        ["asset_id", "session_date", "session"], sort=True
    ):
        work = group.reset_index(drop=True).copy()
        work["market_id"] = "a_share"
        work["last_price"] = work.price
        valid = (
            work.bid_px1.gt(0)
            & work.ask_px1.gt(work.bid_px1)
            & work.bid_vol1.gt(0)
            & work.ask_vol1.gt(0)
        )
        work["eligible"] = valid
        work["mid"] = ((work.bid_px1 + work.ask_px1) / 2).where(valid)
        breaks = (
            work.timestamp.diff().dt.total_seconds().gt(max_gap_seconds)
            | ~valid
            | ~valid.shift(1, fill_value=False)
        )
        work["segment_id"] = (
            f"{asset}:{day:%Y%m%d}:{session}:" + breaks.cumsum().astype(str)
        )
        for kind, source_name in (
            ("trade", "trade"),
            ("order", "order"),
            ("cancel", "order" if work.exchange.iloc[0] == "SSE" else "trade"),
        ):
            source, positions = sources[source_name]
            selected = source.iloc[positions.get((asset, day, session), [])]
            total, signed = _flow(work, selected, kind)
            work[f"observed_{kind}_volume"] = total
            work[f"observed_signed_{kind}_volume"] = signed
            boundary = work.segment_id.ne(work.segment_id.shift())
            work.loc[
                boundary, [f"observed_{kind}_volume", f"observed_signed_{kind}_volume"]
            ] = np.nan
        output.append(work.loc[:, protocol.PANEL_COLUMNS])
    if not output:
        return pd.DataFrame(columns=protocol.PANEL_COLUMNS)
    return pd.concat(output, ignore_index=True).sort_values(
        ["timestamp", "asset_id"], kind="stable"
    )


@contextmanager
def produce(spec, *, inputs, time_window, processed_catalog):
    sessions = {
        name: open_dataset_session(**_dataset_input_options(path))
        for name, path in inputs.items()
    }
    for name, session in sessions.items():
        if session.spec["data_id"] != name:
            raise ValueError("hft_input_identity_mismatch")
        validate_input_market(session.spec, "a_share")
    metadata = {
        "producer_diagnostics": {
            "timestamp_semantics": "vendor_event_time_no_receive_clock",
        },
    }
    definitions = [
        item
        for item in protocol.label_catalog().list()
        if item.output_column in spec.label_availability_columns
    ]
    if definitions:
        metadata["producer_diagnostics"]["label_semantics"] = {
            item.output_column: item.to_manifest() for item in definitions
        }
    lower, upper, closed = time_window_timestamp_interval(time_window)
    hawkes_model = None
    if spec.data_id == "hft_hawkes_factors":
        from quant_research_platform.research.models.hawkes import load_hawkes

        hawkes_model = load_hawkes(json.loads(spec.parameters_json)["model"])
        if pd.Timestamp(lower) <= hawkes_model[1]:
            raise ValueError("hawkes_prediction_not_out_of_sample")

    def frames():
        for day in pd.date_range(
            pd.Timestamp(lower).normalize(), pd.Timestamp(upper).normalize(), freq="D"
        ):
            window = TimeWindow(day.date(), day.date())
            data = {
                name: materialize_dataset_session(session.filtered_view(window))
                for name, session in sessions.items()
            }
            if spec.data_id == "hft_panel":
                frame = build_panel(
                    data["hft_quotes"],
                    data["hft_orders"],
                    data["hft_trades"],
                    **json.loads(spec.parameters_json),
                )
            elif data["hft_panel"].empty:
                continue
            elif spec.data_id == "hft_factors":
                frame = compute_factors(
                    data["hft_panel"],
                    columns=tuple(
                        name
                        for name, role in spec.column_roles.items()
                        if role == "factor"
                    ),
                    **json.loads(spec.parameters_json),
                )
            elif spec.data_id == "hft_event_factors":
                frame = compute_event_factors(
                    data["hft_panel"],
                    data["hft_orders"],
                    data["hft_trades"],
                    columns=tuple(
                        name
                        for name, role in spec.column_roles.items()
                        if role == "factor"
                    ),
                    config=json.loads(spec.parameters_json),
                )
            elif spec.data_id == "hft_hawkes_factors":
                from quant_research_platform.research.factors.hawkes import (
                    compute_hawkes_factors,
                )

                frame = compute_hawkes_factors(
                    data["hft_panel"], data["hft_trades"], hawkes_model
                )
            else:
                frame = compute_labels(data["hft_panel"], definitions)
            if frame.empty:
                continue
            mask = frame.timestamp.ge(pd.Timestamp(lower)) & (
                frame.timestamp.le(pd.Timestamp(upper))
                if closed
                else frame.timestamp.lt(pd.Timestamp(upper))
            )
            yield frame.loc[mask, spec.required_columns].reset_index(drop=True)

    yield frames(), metadata
