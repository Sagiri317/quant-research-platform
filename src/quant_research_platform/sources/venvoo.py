"""Normalize explicitly selected A-share Level-2 Parquet exports."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.contracts.portfolios import A_SHARE_INSTRUMENT_TYPES
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.parquet import DEFAULT_BATCH_ROWS

STREAM_FILES = {
    "hft_quotes": "hft_quotes/data.parquet",
    "hft_orders": "hft_orders/data.parquet",
    "hft_trades": "hft_trades/data.parquet",
}
REPOSITORY = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"


@dataclass(frozen=True)
class Level2ImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...] = ()


def instrument_mapping(instruments):
    if not isinstance(instruments, list) or not instruments:
        raise ValueError("level2_instruments_required")
    mapping = {}
    for item in instruments:
        if (
            type(item) is not dict
            or not {"wind_code", "asset_id", "exchange"} <= item.keys()
            or item.keys() - {"wind_code", "asset_id", "exchange", "instrument_type"}
        ):
            raise ValueError("level2_instrument_fields_invalid")
        if any(type(value) is not str or not value for value in item.values()):
            raise ValueError("level2_instrument_value_invalid")
        if item.get("instrument_type", "stock") not in A_SHARE_INSTRUMENT_TYPES:
            raise ValueError("level2_instrument_type_invalid")
        code = item["wind_code"].upper()
        if item.get("instrument_type", "stock") != "stock":
            # Vendor ETF suffixes do not consistently identify their exchange.
            code = code[:6]
        if code in mapping or item["exchange"] not in {"SSE", "SZSE"}:
            raise ValueError("level2_instrument_identity_invalid")
        mapping[code] = item
    if len({item["asset_id"] for item in mapping.values()}) != len(mapping):
        raise ValueError("level2_instrument_identity_invalid")
    return mapping


def source_columns(stream):
    required = {"wind_code", "date", "time", "price", "volume"}
    if stream == "hft_quotes":
        required |= {
            f"{side}_{kind}{level}"
            for side in ("bid", "ask")
            for kind in ("px", "vol")
            for level in range(1, 11)
        }
    elif stream == "hft_orders":
        required |= {"order_id", "ex_order_id", "order_type", "order_code"}
    elif stream == "hft_trades":
        required |= {
            "trade_id",
            "trade_code",
            "bs_flag",
            "ask_order_id",
            "bid_order_id",
        }
    else:
        raise ValueError("level2_stream_invalid")
    return tuple(sorted(required))


def interim_spec(stream):
    columns = (
        *(
            name
            for name in source_columns(stream)
            if name not in {"wind_code", "date", "time"}
        ),
        "source_row",
        "asset_id",
        "exchange",
        "instrument_type",
        "timestamp",
        "session_date",
        *(("event_kind", "side") if stream != "hft_quotes" else ()),
    )
    roles = dict.fromkeys(columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", source_row="key")
    return InterimDataSpec(
        data_id=stream,
        source_id="venvoo",
        frequency="tick",
        grain="trade_event",
        column_roles=roles,
        primary_key=("timestamp", "asset_id", "source_row"),
        partition_columns=("year",),
        column_units={
            column: "CNY"
            if column == "price" or column.startswith(("bid_px", "ask_px"))
            else "shares"
            for column in columns
            if column in {"price", "volume"}
            or column.startswith(("bid_px", "ask_px", "bid_vol", "ask_vol"))
        },
    )


def normalize(frame, stream, instruments, *, clock_counts=None):
    mapping = instrument_mapping(instruments)
    required = set(source_columns(stream))
    if not required <= set(frame):
        raise ValueError("level2_source_columns_missing")
    return _normalize(frame, stream, mapping, clock_counts=clock_counts)


def _normalize(frame, stream, mapping, *, clock_counts=None):
    required = source_columns(stream)
    codes = frame.wind_code.str.upper()
    etfs = {
        code
        for code, item in mapping.items()
        if item.get("instrument_type", "stock") != "stock"
    }
    codes = codes.mask(codes.str[:6].isin(etfs), codes.str[:6])
    selected = codes.isin(mapping)
    work = frame.loc[selected, sorted(required)].copy()
    work["source_row"] = np.flatnonzero(selected)
    codes = codes.loc[selected]
    work["asset_id"] = codes.map(
        {code: item["asset_id"] for code, item in mapping.items()}
    )
    work["exchange"] = codes.map(
        {code: item["exchange"] for code, item in mapping.items()}
    )
    work["instrument_type"] = codes.map(
        {code: item.get("instrument_type", "stock") for code, item in mapping.items()}
    )
    raw_time = pd.to_numeric(work.pop("time"), errors="raise").astype("float64")
    if (raw_time % 1 != 0).any():
        raise ValueError("level2_time_invalid")
    # Vendor daytime exports mix second-only and millisecond clock encodings.
    seconds_clock = raw_time.between(10_000, 999_999)
    if clock_counts is not None:
        clock_counts["HHMMSS"] += int(seconds_clock.sum())
        clock_counts["HHMMSSmmm"] += int((~seconds_clock).sum())
    raw_time.loc[seconds_clock] *= 1000
    hour, minute, second, millis = (
        raw_time // 10_000_000,
        raw_time // 100_000 % 100,
        raw_time // 1000 % 100,
        raw_time % 1000,
    )
    if (
        raw_time.isna().any()
        or ((hour < 0) | (hour > 23) | (minute > 59) | (second > 59)).any()
    ):
        raise ValueError("level2_time_invalid")
    day = pd.to_datetime(work.pop("date").astype(str), format="%Y%m%d", errors="raise")
    clock = day + pd.to_timedelta(
        hour * 3_600_000 + minute * 60_000 + second * 1000 + millis, unit="ms"
    )
    work["timestamp"] = (
        clock.dt.tz_localize("Asia/Shanghai")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .astype("datetime64[ns]")
    )
    work["session_date"] = day.astype("datetime64[ns]")
    price_columns = [
        column
        for column in work
        if column == "price" or column.startswith(("bid_px", "ask_px"))
    ]
    quantity_columns = [
        column
        for column in work
        if column == "volume" or column.startswith(("bid_vol", "ask_vol"))
    ]
    for column in (*price_columns, *quantity_columns):
        values = pd.to_numeric(work[column], errors="raise")
        valid = np.isfinite(values) & values.ge(0) & values.lt(2**53)
        if column in price_columns:
            valid &= values.gt(0)
        work[column] = values.where(valid) / (10_000 if column in price_columns else 1)
    if stream == "hft_orders":
        work["event_kind"] = np.where(
            work.exchange.eq("SZSE") | work.order_type.isin(("A",)),
            "order",
            np.where(work.order_type.isin(("D",)), "cancel", "unknown"),
        )
        work["side"] = work.order_code.map({"B": "buy", "S": "sell"})
    elif stream == "hft_trades":
        work["event_kind"] = np.where(
            work.exchange.eq("SZSE") & work.trade_code.isin(("C",)),
            "cancel",
            np.where(
                work.exchange.eq("SSE") | work.trade_code.isin(["0", "T"]),
                "trade",
                "unknown",
            ),
        )
        work["side"] = work.bs_flag.map({"B": "buy", "S": "sell"})
        cancellations = work.exchange.eq("SZSE") & work.trade_code.eq("C")
        work.loc[cancellations, "side"] = np.nan
        # Shenzhen cancellations identify the cancelled side by its sole order index.
        for side, own, other in (
            ("buy", "bid_order_id", "ask_order_id"),
            ("sell", "ask_order_id", "bid_order_id"),
        ):
            identified = work[own].gt(0) & work[own].lt(2**64 - 1) & work[other].eq(0)
            work.loc[cancellations & identified, "side"] = side
    return (
        work.drop(columns="wind_code")
        .sort_values(["timestamp", "asset_id", "source_row"], kind="stable")
        .reset_index(drop=True)
    )


def materialize_interim_data(
    *,
    raw_root,
    output_roots,
    data_ids,
    instruments,
    revision,
    start_date,
    end_date,
):
    from quant_research_platform.contracts.time import TimeWindow

    window = TimeWindow(start=start_date, end=end_date)
    mapping = instrument_mapping(instruments)
    results = {}
    for stream in data_ids:
        if stream not in STREAM_FILES:
            raise ValueError("level2_stream_invalid")
        source = ds.parquet_dataset(Path(raw_root) / STREAM_FILES[stream])
        symbols = [
            code
            for code, item in mapping.items()
            if item.get("instrument_type", "stock") == "stock"
        ]
        selected = ds.field("wind_code").isin(
            pa.array(
                symbols + [code.lower() for code in symbols],
                type=source.schema.field("wind_code").type,
            )
        )
        for code in mapping.keys() - set(symbols):
            selected |= (ds.field("wind_code") >= code) & (
                ds.field("wind_code") < str(int(code) + 1).zfill(6)
            )
        scanner = source.scanner(
            columns=list(source_columns(stream)),
            filter=selected
            & (ds.field("date") >= int(window.start.strftime("%Y%m%d")))
            & (ds.field("date") <= int(window.end.strftime("%Y%m%d"))),
            batch_size=DEFAULT_BATCH_ROWS,
        )
        spec = interim_spec(stream)
        clock_counts = {"HHMMSS": 0, "HHMMSSmmm": 0}
        event_counts = dict.fromkeys(("trade", "order", "cancel", "unknown"), 0)

        def frames(scanner, stream, spec, clock_counts, event_counts):
            offsets = defaultdict(int)
            for batch in scanner.to_batches():
                work = batch.to_pandas(types_mapper={pa.uint64(): pd.UInt64Dtype()}.get)
                for day, rows in work.groupby("date", sort=False):
                    frame = _normalize(
                        rows,
                        stream,
                        mapping,
                        clock_counts=clock_counts,
                    )
                    frame["source_row"] += offsets[day]
                    offsets[day] += len(frame)
                    if stream != "hft_quotes":
                        for kind, count in frame.event_kind.value_counts().items():
                            event_counts[kind] += int(count)
                    yield frame.loc[:, spec.required_columns]
            if not offsets:
                empty = pa.Table.from_batches(
                    [], schema=scanner.projected_schema
                ).to_pandas(types_mapper={pa.uint64(): pd.UInt64Dtype()}.get)
                yield _normalize(empty, stream, mapping).loc[:, spec.required_columns]

        output = Path(output_roots[stream])
        write_dataset(
            output,
            frames(scanner, stream, spec, clock_counts, event_counts),
            spec=spec,
            metadata={
                "source_evidence": {
                    "repository": REPOSITORY,
                    "revision": revision,
                    "source_member": STREAM_FILES[stream],
                    "instruments": instruments,
                    "timestamp_semantics": "vendor_event_time_no_receive_clock",
                    "clock_encoding_counts": clock_counts,
                    **(
                        {"event_kind_counts": event_counts}
                        if stream != "hft_quotes"
                        else {}
                    ),
                    **(
                        {
                            "trade_direction_basis": (
                                "vendor_bs_flag;aggressor_unverified"
                            )
                        }
                        if stream == "hft_trades"
                        else {}
                    ),
                },
            },
        )
        results[stream] = output
    return Level2ImportResult(results)
