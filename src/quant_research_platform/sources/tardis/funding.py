"""Project native forecasts and sourced settlements through one funding owner."""

import hashlib
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.portfolios import FundingForecast
from quant_research_platform.contracts.validation import (
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import require_free_disk_headroom
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
    _manifest,
    _records,
    _settlement_response,
    settlement_endpoint,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
)
from quant_research_platform.sources.binance.validation import (
    strict_iso_timestamp,
    strict_non_negative_integer,
    strict_positive_float,
)
from quant_research_platform.sources.tardis.raw import SLICE_MINUTES, iter_native_slice
from quant_research_platform.sources.tardis.replay import _archive_span
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    filesystem_write_bytes,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.raw_snapshots import write_csv_snapshot


def import_funding_history(
    archive,
    *,
    symbol,
    source_id,
    response_path,
    http_status,
    start_ms,
    end_ms,
    byte_budget,
    request_limit,
):
    """Preserve source bytes and publish forecasts with vendor availability."""
    selected = [
        item
        for item in archive.archives
        if pd.Timestamp(start_ms, unit="ms", tz="UTC").to_pydatetime()
        == _archive_span(item.manifest)[0]
    ]
    if len(selected) != 1:
        raise ValueError("tardis_funding_archive_window_invalid")
    archive = selected[0]
    endpoint = settlement_endpoint(source_id, symbol)
    if (
        symbol not in RESEARCH_SYMBOLS
        or archive.manifest["exchange"] != "binance-futures"
    ):
        raise ValueError("binance_funding_symbol_invalid")
    midnight = strict_iso_timestamp(
        archive.manifest["date"] + "T00:00:00Z", error="tardis_funding_date_invalid"
    ).value
    first = archive.manifest["files"][0]["offset_minutes"]
    stop = archive.manifest["files"][-1]["offset_minutes"] + SLICE_MINUTES
    start_ns = midnight + first * 60 * 10**9
    end_ns = midnight + stop * 60 * 10**9
    if (
        type(http_status) is not int
        or http_status != 200
        or type(start_ms) is not int
        or start_ms * 1_000_000 != start_ns
        or type(end_ms) is not int
        or (end_ms + 1) * 1_000_000 != end_ns
    ):
        raise ValueError("tardis_funding_request_coverage_invalid")
    if type(byte_budget) is not int or byte_budget <= 0:
        raise ValueError("tardis_funding_budget_invalid")
    path = require_unaliased_path(
        Path(response_path), error="tardis_funding_source_invalid"
    )
    identity = dataset_file_identity(path)
    raw = events = metadata = native = None
    try:
        with closing_iterator(open_filesystem_file(path, "rb")) as handle:
            raw = handle.read(identity.size + 1)
        if identity != dataset_file_identity(path) or len(raw) != identity.size:
            raise ValueError("tardis_funding_source_changed")
        if byte_budget <= len(raw):
            raise ValueError("tardis_funding_budget_exceeded")
        events, proof = _settlement_response(
            raw,
            source_id=source_id,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            request_limit=request_limit,
        )
        native = {
            "source_id": source_id,
            "endpoint": endpoint,
            "symbol": symbol,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "http_status": http_status,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            **proof,
        }
        metadata = {
            "schema_id": FUNDING_SCHEMA_ID,
            "symbol": symbol,
            "market_type": FUTURES_MARKET_TYPE,
            "valuation_currency": "USDT",
            "coverage": {
                "start_ns": start_ns,
                "end_ns": (end_ms + 1) * 1_000_000 - 1,
                "status": "COMPLETE",
            },
            "provenance": "tardis_market_manifest:"
            + canonical_json_hash(archive.source_identity),
            "settlement_source": native,
        }

        def rows():
            for settled in events:
                yield (
                    "settlement",
                    settled.timestamp_ns,
                    None,
                    settled.rate,
                    settled.mark_price,
                    None,
                )
            previous_observed = -1
            for member in archive.manifest["files"]:
                records = iter_native_slice(
                    archive.root / member["path"],
                    date=archive.manifest["date"],
                    offset_minutes=member["offset_minutes"],
                    expected_sha256=member["sha256"],
                    exchange=archive.manifest["exchange"],
                    symbols=tuple(archive.manifest["symbols"]),
                    channels=tuple(archive.manifest["requested_channels"]),
                )
                with closing_iterator(records):
                    value = mark = forecast = None
                    try:
                        for _number, timestamp, value in records:
                            if (
                                value is None
                                or value["stream"] != symbol.lower() + "@markPrice@1s"
                            ):
                                value = None
                                continue
                            observed = strict_iso_timestamp(
                                timestamp,
                                error="tardis_funding_observed_time_invalid",
                            ).value
                            if observed <= previous_observed:
                                raise ValueError(
                                    "tardis_funding_forecast_order_invalid"
                                )
                            previous_observed = observed
                            mark = value["data"]
                            until = (
                                strict_non_negative_integer(
                                    mark.get("T"),
                                    error="tardis_funding_forecast_time_invalid",
                                )
                                * 1_000_000
                            )
                            rate = strict_finite_float(
                                mark["r"],
                                error="tardis_funding_forecast_rate_invalid",
                            )
                            price = strict_positive_float(
                                mark.get("p"),
                                error="tardis_funding_forecast_mark_invalid",
                            )
                            if until > observed:
                                forecast = FundingForecast(
                                    observed, until, ((until, -rate * price),)
                                )
                                _yield_value = (
                                    "forecast",
                                    observed,
                                    until,
                                    None,
                                    None,
                                    [list(item) for item in forecast.cashflows],
                                )
                                try:
                                    yield _yield_value
                                except GeneratorExit:
                                    return
                            value = mark = forecast = None
                    finally:
                        value = mark = forecast = None

        def validate(root):
            checked = _manifest(root, symbol)
            for _item in _records(root, checked):
                pass
            with closing_iterator(open_filesystem_file(path, "rb")) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if identity != dataset_file_identity(path) or digest != native["sha256"]:
                raise ValueError("tardis_funding_source_changed")

        def publish(staging):
            require_free_disk_headroom(staging, reserve_bytes=len(raw))
            filesystem_write_bytes(staging / "settlements.json", raw)
            projected = rows()
            with closing_iterator(projected):
                write_csv_snapshot(
                    staging / "observations",
                    columns=FUNDING_COLUMNS,
                    rows=projected,
                    metadata=metadata,
                    max_bytes=byte_budget - len(raw),
                    validate=validate,
                )

        atomic_write_directory(archive.funding_root(symbol).parent, publish)
    finally:
        raw = events = metadata = native = None
