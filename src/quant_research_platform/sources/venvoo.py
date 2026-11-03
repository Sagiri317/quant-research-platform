"""Normalize explicitly selected A-share Level-2 Parquet exports."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import _hash_file_content

STREAM_FILES = {
    "hft_quotes": "行情.parquet",
    "hft_orders": "逐笔委托.parquet",
    "hft_trades": "逐笔成交.parquet",
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
        if type(item) is not dict or set(item) != {"wind_code", "asset_id", "exchange"}:
            raise ValueError("level2_instrument_fields_invalid")
        if any(type(value) is not str or not value for value in item.values()):
            raise ValueError("level2_instrument_value_invalid")
        code = item["wind_code"].upper()
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


def normalize(frame, stream, instruments):
    mapping = instrument_mapping(instruments)
    required = set(source_columns(stream))
    if not required <= set(frame):
        raise ValueError("level2_source_columns_missing")
    codes = frame.wind_code.str.upper()
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
    raw_time = pd.to_numeric(work.pop("time"), errors="raise")
    hour, minute, second, millis = (
        raw_time // 10_000_000,
        raw_time // 100_000 % 100,
        raw_time // 1000 % 100,
        raw_time % 1000,
    )
    if (
        raw_time.isna().any()
        or (
            (raw_time % 1 != 0)
            | (hour < 0)
            | (hour > 23)
            | (minute > 59)
            | (second > 59)
        ).any()
    ):
        raise ValueError("level2_time_invalid")
    day = pd.to_datetime(work.pop("date").astype(str), format="%Y%m%d", errors="raise")
    clock = day + pd.to_timedelta(
        hour * 3_600_000 + minute * 60_000 + second * 1000 + millis, unit="ms"
    )
    work["timestamp"] = (
        clock.dt.tz_localize("Asia/Shanghai").dt.tz_convert("UTC").dt.tz_localize(None)
    )
    work["session_date"] = day
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
            work.exchange.eq("SSE") & work.order_type.eq("D"), "cancel", "order"
        )
        work["side"] = work.order_code.map({"B": "buy", "S": "sell"})
    elif stream == "hft_trades":
        work["event_kind"] = np.where(
            work.exchange.eq("SZSE") & work.trade_code.eq("C"),
            "cancel",
            np.where(work.trade_code.isin(["0", "T", "", "\x00"]), "trade", "unknown"),
        )
        work["side"] = work.bs_flag.map({"B": "buy", "S": "sell"})
    return (
        work.drop(columns="wind_code")
        .sort_values(["timestamp", "asset_id", "source_row"], kind="stable")
        .reset_index(drop=True)
    )


def materialize_interim_data(
    *, raw_root, output_roots, data_ids, instruments, revision, start_date, end_date
):
    from quant_research_platform.contracts.time import TimeWindow

    window = TimeWindow(start=start_date, end=end_date)
    instrument_mapping(instruments)
    if (
        type(revision) is not str
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("level2_repository_revision_required")
    results = {}
    for stream in data_ids:
        if stream not in STREAM_FILES:
            raise ValueError("level2_stream_invalid")
        files = [
            path
            for path in sorted(Path(raw_root).glob(f"*/{STREAM_FILES[stream]}"))
            if window.start.strftime("%Y%m%d")
            <= path.parent.name
            <= window.end.strftime("%Y%m%d")
        ]
        if not files:
            raise FileNotFoundError("level2_selected_files_missing")
        hashes = {
            path.relative_to(raw_root).as_posix(): _hash_file_content(path)
            for path in files
        }
        symbols = [item["wind_code"].upper() for item in instruments]
        spec = interim_spec(stream)

        def frames(files, stream, symbols, spec):
            for path in files:
                source = ds.dataset(path, format="parquet")
                scanner = source.scanner(
                    columns=list(source_columns(stream)),
                    filter=ds.field("wind_code").isin(
                        symbols + [code.lower() for code in symbols]
                    ),
                    batch_size=DEFAULT_BATCH_ROWS,
                )
                offset = 0
                for batch in scanner.to_batches():
                    if not batch.num_rows:
                        continue
                    frame = normalize(batch.to_pandas(), stream, instruments)
                    if not frame.session_date.eq(pd.Timestamp(path.parent.name)).all():
                        raise ValueError("level2_source_date_mismatch")
                    frame["source_row"] += offset
                    offset += len(frame)
                    yield frame.loc[:, spec.required_columns]
                if offset == 0:
                    empty = pa.Table.from_batches(
                        [], schema=scanner.projected_schema
                    ).to_pandas()
                    yield normalize(empty, stream, instruments).loc[
                        :, spec.required_columns
                    ]

        output = Path(output_roots[stream])
        write_dataset(
            output,
            frames(files, stream, symbols, spec),
            spec=spec,
            metadata={
                "source_evidence": {
                    "repository": REPOSITORY,
                    "revision": revision,
                    "source_hashes": hashes,
                    "instruments": instruments,
                    "timestamp_semantics": "vendor_event_time_no_receive_clock",
                },
            },
        )
        results[stream] = output
    return Level2ImportResult(results)
