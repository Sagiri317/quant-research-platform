"""Project native vendor forecasts and official settlements into one funding owner."""

import hashlib
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import FundingEvent, FundingForecast
from quant_research_platform.contracts.validation import (
    clear_exception_frames,
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import require_free_disk_headroom
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
    _manifest,
    _records,
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
from quant_research_platform.sources.tardis.raw import iter_native_slice
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    require_unaliased_path,
)
from quant_research_platform.storage.raw_snapshots import write_csv_snapshot


def import_funding_history(
    day, *, symbol, response_path, http_status, start_ms, end_ms, byte_budget
):
    """Preserve the official bytes and publish forecasts with vendor availability."""
    if symbol not in RESEARCH_SYMBOLS:
        raise ValueError("binance_funding_symbol_invalid")
    midnight = strict_iso_timestamp(
        day.manifest["date"] + "T00:00:00Z", error="tardis_funding_date_invalid"
    ).value
    if (
        type(http_status) is not int
        or http_status != 200
        or type(start_ms) is not int
        or start_ms * 1_000_000 != midnight
        or type(end_ms) is not int
        or end_ms != start_ms + 86_400_000 - 1
    ):
        raise ValueError("tardis_funding_request_coverage_invalid")
    if type(byte_budget) is not int or byte_budget <= 0:
        raise ValueError("tardis_funding_budget_invalid")
    path = require_unaliased_path(
        Path(response_path), error="tardis_funding_source_invalid"
    )
    identity = dataset_file_identity(path)
    policy = day.resource_policy
    with policy.lease(
        "tardis.funding.native",
        json_record_working_set_bytes(identity.size) + 64 * 1024,
    ):
        raw = payload = events = event = row = metadata = native = None
        try:
            with path.open("rb") as handle:
                raw = handle.read(identity.size + 1)
            if identity != dataset_file_identity(path) or len(raw) != identity.size:
                raise ValueError("tardis_funding_source_changed")
            if byte_budget <= len(raw):
                raise ValueError("tardis_funding_budget_exceeded")
            payload = parse_qrp_json_object(b'{"records":' + raw + b"}")["records"]
            if type(payload) is not list or len(payload) >= 1000:
                raise ValueError("tardis_funding_response_incomplete")
            events = []
            previous = -1
            for row in payload:
                if type(row) is not dict or row.get("symbol") != symbol:
                    raise ValueError("tardis_funding_settlement_identity_invalid")
                timestamp = strict_non_negative_integer(
                    row.get("fundingTime"),
                    error="tardis_funding_settlement_time_invalid",
                )
                if not start_ms <= timestamp <= end_ms or timestamp <= previous:
                    raise ValueError("tardis_funding_settlement_order_invalid")
                previous = timestamp
                rate = strict_finite_float(
                    row.get("fundingRate"), error="tardis_funding_rate_invalid"
                )
                event = FundingEvent(
                    timestamp * 1_000_000,
                    rate,
                    strict_positive_float(
                        row.get("markPrice"), error="tardis_funding_mark_invalid"
                    ),
                )
                events.append(event)
            payload = row = event = None
            native = {
                "source_id": "binance",
                "endpoint": "https://fapi.binance.com/fapi/v1/fundingRate",
                "symbol": symbol,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "http_status": http_status,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
            metadata = {
                "schema_id": FUNDING_SCHEMA_ID,
                "symbol": symbol,
                "market_type": FUTURES_MARKET_TYPE,
                "valuation_currency": "USDT",
                "coverage": {
                    "start_ns": midnight,
                    "end_ns": (end_ms + 1) * 1_000_000 - 1,
                    "status": "COMPLETE",
                },
                "provenance": "tardis_market_manifest:"
                + canonical_json_hash(day.source_identity),
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
                for member in day.manifest["files"]:
                    records = iter_native_slice(
                        day.root / member["path"],
                        date=day.manifest["date"],
                        offset_minutes=member["offset_minutes"],
                        resource_policy=policy,
                    )
                    with closing_iterator(records):
                        value = mark = forecast = None
                        try:
                            for _number, timestamp, value in records:
                                if (
                                    value is None
                                    or value["stream"]
                                    != symbol.lower() + "@markPrice@1s"
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
                                    float(mark["r"]),
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
                                    yield (
                                        "forecast",
                                        observed,
                                        until,
                                        None,
                                        None,
                                        [list(item) for item in forecast.cashflows],
                                    )
                                value = mark = forecast = None
                        except BaseException as error:
                            clear_exception_frames(error)
                            raise
                        finally:
                            value = mark = forecast = None

            def validate(root):
                checked = _manifest(root, symbol)
                for _item in _records(root, checked):
                    pass
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if (
                    identity != dataset_file_identity(path)
                    or digest != native["sha256"]
                ):
                    raise ValueError("tardis_funding_source_changed")

            def publish(staging):
                require_free_disk_headroom(staging, reserve_bytes=len(raw))
                (staging / "settlements.json").write_bytes(raw)
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

            atomic_write_directory(day.funding_root(symbol).parent, publish)
        except BaseException as error:
            clear_exception_frames(error)
            raise
        finally:
            raw = payload = events = event = row = metadata = native = None
