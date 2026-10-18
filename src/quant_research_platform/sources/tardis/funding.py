"""Project native forecasts and sourced settlements through one funding owner."""

import hashlib
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.portfolios import FundingForecast
from quant_research_platform.contracts.validation import (
    strict_finite_float,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.binance.funding import (
    FUNDING_COLUMNS,
    FUNDING_SCHEMA_ID,
    _manifest,
    _nullable_funding_value,
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
from quant_research_platform.sources.tardis.raw import iter_native_slice
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.identity import (
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
    request_limit,
    output_root=None,
):
    """Publish independent settlement coverage and native forecast observations.

    output_root contains settlements.json and observations/. A single market
    archive supplies the default location; multiple archives require a destination.
    Forecast timestamps retain their native availability, including any prefix.
    """
    archives = archive.archives
    endpoint = settlement_endpoint(source_id, symbol)
    if (
        symbol not in RESEARCH_SYMBOLS
        or not archives
        or any(
            item.manifest["exchange"] != "binance-futures"
            or symbol.lower() not in item.manifest["symbols"]
            for item in archives
        )
    ):
        raise ValueError("binance_funding_symbol_invalid")
    if (
        type(http_status) is not int
        or http_status != 200
        or type(start_ms) is not int
        or type(end_ms) is not int
        or not 0 <= start_ms <= end_ms
    ):
        raise ValueError("tardis_funding_request_coverage_invalid")
    if output_root is None:
        if len(archives) != 1:
            raise ValueError("tardis_funding_output_required")
        output_root = archives[0].funding_root(symbol).parent
    start_ns = start_ms * 1_000_000
    path = require_unaliased_path(
        Path(response_path), error="tardis_funding_source_invalid"
    )
    raw = events = metadata = native = None
    with closing_iterator(open_filesystem_file(path, "rb")) as handle:
        raw = handle.read()
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
        + canonical_json_hash(
            archives[0].source_identity
            if len(archives) == 1
            else archive.source_identity
        ),
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
        previous_projection = None
        for archive, member in (
            (item, member) for item in archives for member in item.manifest["files"]
        ):
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
                value = mark = None
                for _number, timestamp, value in records:
                    if value is None or value["stream"].split("@")[:2] != [
                        symbol.lower(),
                        "markPrice",
                    ]:
                        value = None
                        continue
                    observed = strict_iso_timestamp(
                        timestamp,
                        error="tardis_funding_observed_time_invalid",
                    ).value
                    mark = value["data"]
                    until = _nullable_funding_value(
                        mark.get("T"), strict_non_negative_integer
                    )
                    if until is not None:
                        until *= 1_000_000
                    rate = _nullable_funding_value(mark.get("r"), strict_finite_float)
                    price = _nullable_funding_value(
                        mark.get("p"), strict_positive_float
                    )
                    projection = (until, rate, price)
                    if observed < previous_observed:
                        raise ValueError("tardis_funding_forecast_order_invalid")
                    if observed == previous_observed:
                        # Equal-time observations have no causal winner. Each
                        # disagreeing fact becomes unknown, independently.
                        previous_projection = tuple(
                            old if old == new else None
                            for old, new in zip(
                                previous_projection, projection, strict=True
                            )
                        )
                        continue
                    if previous_projection is not None:
                        yield forecast_row(previous_observed, previous_projection)
                    previous_observed, previous_projection = observed, projection
                    value = mark = None
        if previous_projection is not None:
            yield forecast_row(previous_observed, previous_projection)

    def forecast_row(observed, projection):
        until, rate, price = projection
        if until is None or until <= observed:
            forecast = FundingForecast(observed, observed, ())
        else:
            amount = (
                0.0
                if rate == 0.0 and price is None
                else None
                if rate is None or price is None
                else _nullable_funding_value(-rate * price, strict_finite_float)
            )
            forecast = FundingForecast(observed, until, ((until, amount),))
        return (
            "forecast",
            observed,
            forecast.known_until_ns,
            None,
            None,
            [list(item) for item in forecast.cashflows],
        )

    def validate(root):
        checked, settlements = _manifest(root, symbol)
        for _item in _records(
            root, checked, settlements=settlements, verify_bytes=True
        ):
            pass

    def publish(staging):
        filesystem_write_bytes(staging / "settlements.json", raw)
        projected = rows()
        with closing_iterator(projected):
            write_csv_snapshot(
                staging / "observations",
                columns=FUNDING_COLUMNS,
                rows=projected,
                metadata=metadata,
                validate=validate,
            )

    atomic_write_directory(Path(output_root), publish)
