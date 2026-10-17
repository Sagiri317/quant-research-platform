"""Immutable per-symbol funding observations with separate known forecasts."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from heapq import merge
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
)
from quant_research_platform.sources.binance.validation import (
    strict_iso_timestamp,
    strict_non_negative_integer,
    strict_positive_float,
)
from quant_research_platform.storage.identity import (
    filesystem_is_file,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
)

FUNDING_COLUMNS = (
    "kind",
    "time_ns",
    "known_until_ns",
    "rate",
    "mark_price",
    "cashflows",
)
FUNDING_SCHEMA_ID = "qrp-binance-funding-observations"


@dataclass(frozen=True)
class FundingObservations:
    identity: str
    coverage: tuple[tuple[int, int], ...]
    events: tuple[FundingEvent, ...]
    forecasts: Iterator[FundingForecast]
    settlement_qualification: tuple[dict, ...]


def _nullable_funding_value(value, validate):
    """Mask only a local numeric value in an otherwise identified observation."""
    try:
        return validate(value, error="binance_funding_value_invalid")
    except ValueError:
        return None


def settlement_endpoint(source_id, symbol):
    """Bind supported settlement provenance to its actual supplier endpoint."""
    if symbol not in RESEARCH_SYMBOLS:
        raise ValueError("binance_funding_settlement_source_invalid")
    if source_id == "binance":
        return "https://fapi.binance.com/fapi/v1/fundingRate"
    if source_id == "xoomar":
        asset = symbol.removesuffix("USDT").lower()
        return f"https://xoomar.com/api/markets/funding-rates/{asset}?history=1y"
    raise ValueError("binance_funding_settlement_source_invalid")


def _settlement_response(raw, *, source_id, symbol, start_ms, end_ms, request_limit):
    """Prove one supplier response covers the requested settlement window."""
    if source_id == "binance":
        if type(request_limit) is not int or not 1 <= request_limit <= 1000:
            raise ValueError("binance_funding_request_limit_invalid")
        rows = parse_qrp_json_object(b'{"records":' + raw + b"}")["records"]
        if type(rows) is not list or len(rows) >= request_limit:
            raise PortfolioTargetBlocked("binance_funding_coverage_unproven")
        selected = rows
        completion = "short_page"
    elif source_id == "xoomar":
        if type(request_limit) is not int or request_limit != 5000:
            raise ValueError("binance_funding_request_limit_invalid")
        payload = parse_qrp_json_object(raw)
        data = payload.get("data")
        if type(data) is not dict or type(data.get("history")) is not list:
            raise PortfolioTargetBlocked("binance_funding_coverage_unproven")
        rows = data["history"]
        if not rows or len(rows) > request_limit:
            raise PortfolioTargetBlocked("binance_funding_coverage_unproven")
        selected = []
        previous = None
        asset = symbol.removesuffix("USDT")
        for row in rows:
            if type(row) is not dict:
                raise ValueError("binance_funding_settlement_identity_invalid")
            stamp = strict_iso_timestamp(
                row.get("fundingTime"), error="binance_funding_settlement_time_invalid"
            ).value
            if stamp % 1_000_000 or (previous is not None and stamp > previous):
                raise ValueError("binance_funding_settlement_order_invalid")
            previous = stamp
            if row.get("exchange") == "binance" and row.get("baseAsset") == asset:
                if row.get("slug") != asset.lower():
                    raise ValueError("binance_funding_settlement_identity_invalid")
                selected.append(row)
        # Require actual settlements on both sides; this is not a supplier
        # guarantee that the interval contains every historical settlement.
        if (
            not selected
            or strict_iso_timestamp(
                selected[-1]["fundingTime"],
                error="binance_funding_settlement_time_invalid",
            ).value
            > start_ms * 1_000_000
            or strict_iso_timestamp(
                selected[0]["fundingTime"],
                error="binance_funding_settlement_time_invalid",
            ).value
            < end_ms * 1_000_000
        ):
            raise PortfolioTargetBlocked("binance_funding_coverage_unproven")
        selected = list(reversed(selected))
        completion = "history_brackets_window"
    else:
        raise ValueError("binance_funding_settlement_source_invalid")
    events = []
    previous = -1
    for row in selected:
        if type(row) is not dict:
            raise ValueError("binance_funding_settlement_identity_invalid")
        if source_id == "binance":
            if row.get("symbol") != symbol:
                raise ValueError("binance_funding_settlement_identity_invalid")
            timestamp = strict_non_negative_integer(
                row.get("fundingTime"), error="binance_funding_settlement_time_invalid"
            )
        else:
            timestamp = (
                strict_iso_timestamp(
                    row["fundingTime"], error="binance_funding_settlement_time_invalid"
                ).value
                // 1_000_000
            )
        if timestamp <= previous:
            raise ValueError("binance_funding_settlement_order_invalid")
        previous = timestamp
        if not start_ms <= timestamp <= end_ms:
            if source_id == "binance":
                raise ValueError("binance_funding_settlement_order_invalid")
            continue
        events.append(
            FundingEvent(
                timestamp * 1_000_000,
                _nullable_funding_value(row.get("fundingRate"), strict_finite_float),
                _nullable_funding_value(row.get("markPrice"), strict_positive_float),
            )
        )
    return events, {
        "request_limit": request_limit,
        "response_count": len(rows),
        "settlement_count": len(events),
        "completion": completion,
    }


def _manifest(root, symbol):
    if not all(
        filesystem_is_file(root / name)
        for name in ("response.csv", "response.metadata.json")
    ):
        raise PortfolioTargetBlocked("binance_funding_input_missing")
    manifest = read_manifest(root / "response.metadata.json")
    metadata = manifest.get("metadata")
    if (
        type(metadata) is not dict
        or set(metadata)
        != {
            "schema_id",
            "symbol",
            "market_type",
            "valuation_currency",
            "coverage",
            "provenance",
            "settlement_source",
        }
        or metadata["schema_id"] != FUNDING_SCHEMA_ID
        or metadata["symbol"] != symbol
        or metadata["market_type"] != FUTURES_MARKET_TYPE
        or metadata["valuation_currency"] != "USDT"
        or type(manifest.get("columns")) is not list
        or tuple(manifest["columns"]) != FUNDING_COLUMNS
        or type(manifest.get("byte_count")) is not int
        or manifest["byte_count"] <= 0
    ):
        raise ValueError("binance_funding_metadata_invalid")
    coverage = metadata["coverage"]
    if (
        type(coverage) is not dict
        or set(coverage) != {"start_ns", "end_ns", "status"}
        or type(coverage["start_ns"]) is not int
        or type(coverage["end_ns"]) is not int
        or not 0 <= coverage["start_ns"] <= coverage["end_ns"]
        or type(coverage["status"]) is not str
        or coverage["status"] not in {"COMPLETE", "UNPROVEN"}
    ):
        raise ValueError("binance_funding_coverage_invalid")
    if (
        coverage["status"] != "COMPLETE"
        or type(metadata["provenance"]) is not str
        or not metadata["provenance"].strip()
    ):
        raise PortfolioTargetBlocked("binance_funding_coverage_unproven")
    native = metadata["settlement_source"]
    events = None
    if native is not None:
        if (
            type(native) is not dict
            or set(native)
            != {
                "source_id",
                "endpoint",
                "symbol",
                "start_ms",
                "end_ms",
                "http_status",
                "sha256",
                "byte_count",
                "request_limit",
                "response_count",
                "settlement_count",
                "completion",
            }
            or native["endpoint"] != settlement_endpoint(native["source_id"], symbol)
            or native["symbol"] != symbol
            or type(native["start_ms"]) is not int
            or type(native["end_ms"]) is not int
            or native["start_ms"] * 1_000_000 > coverage["start_ns"]
            or (native["end_ms"] + 1) * 1_000_000 <= coverage["end_ns"]
            or native["http_status"] != 200
            or any(
                type(native[key]) is not int or native[key] < 0
                for key in ("response_count", "settlement_count")
            )
            or type(native["byte_count"]) is not int
            or native["byte_count"] <= 0
            or type(native["sha256"]) is not str
            or len(native["sha256"]) != 64
        ):
            raise ValueError("binance_funding_settlement_source_invalid")
        path = require_unaliased_path(
            root.parent / "settlements.json",
            error="binance_funding_settlement_source_invalid",
        )
        with closing_iterator(open_filesystem_file(path, "rb")) as handle:
            raw = handle.read()
            events, proof = _settlement_response(
                raw,
                source_id=native["source_id"],
                symbol=symbol,
                start_ms=native["start_ms"],
                end_ms=native["end_ms"],
                request_limit=native["request_limit"],
            )
            if any(native[key] != value for key, value in proof.items()):
                raise ValueError("binance_funding_settlement_source_invalid")
    return manifest, events


def _records(root, manifest, *, settlements, verify_bytes=False):
    previous = {"settlement": -1, "forecast": -1}
    settlement_count = 0
    rows = iter_csv_snapshot(
        root, expected_manifest=manifest, verify_bytes=verify_bytes
    )
    with closing_iterator(rows):
        for kind, timestamp, known_until, rate, mark, cashflows in rows:
            if (
                type(kind) is not str
                or kind not in previous
                or type(timestamp) is not int
                or timestamp <= previous[kind]
            ):
                raise ValueError("binance_funding_record_order_invalid")
            previous[kind] = timestamp
            if kind == "settlement":
                coverage = manifest["metadata"]["coverage"]
                if (
                    known_until is not None
                    or cashflows is not None
                    or not coverage["start_ns"] <= timestamp <= coverage["end_ns"]
                ):
                    raise ValueError("binance_funding_settlement_invalid")
                settlement_count += 1
                _yield_value = FundingEvent(timestamp, rate, mark)
                if settlements is not None:
                    if settlement_count > len(settlements):
                        raise ValueError("binance_funding_settlement_count_invalid")
                    if _yield_value != settlements[settlement_count - 1]:
                        raise ValueError(
                            "binance_funding_settlement_projection_invalid"
                        )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
            else:
                if rate is not None or mark is not None or type(cashflows) is not list:
                    raise ValueError("binance_funding_forecast_invalid")
                if any(type(item) is not list or len(item) != 2 for item in cashflows):
                    raise ValueError("binance_funding_forecast_invalid")
                _yield_value = FundingForecast(
                    timestamp, known_until, tuple(tuple(item) for item in cashflows)
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
    native = manifest["metadata"]["settlement_source"]
    if native is not None and settlement_count != native["settlement_count"]:
        raise ValueError("binance_funding_settlement_count_invalid")


@contextmanager
def read_funding_observations(funding_roots, *, symbol):
    """Read ordered immutable sources without inventing coverage between them."""
    if symbol not in RESEARCH_SYMBOLS:
        raise ValueError("binance_funding_symbol_invalid")
    if type(funding_roots) is not tuple:
        raise ValueError("binance_funding_roots_invalid")
    if not funding_roots:
        raise PortfolioTargetBlocked("binance_funding_input_missing")
    sources = []
    coverage = []
    for value in funding_roots:
        root = Path(value)
        manifest, settlements = _manifest(root, symbol)
        interval = manifest["metadata"]["coverage"]
        start, end = interval["start_ns"], interval["end_ns"]
        if coverage and start <= coverage[-1][1]:
            raise ValueError("binance_funding_source_intervals_invalid")
        coverage.append((start, end))
        sources.append(
            (
                root,
                canonical_json_hash(manifest),
                manifest,
                settlements,
            )
        )
        del manifest, interval
    identity = canonical_json_hash({"sources": [item[1] for item in sources]})

    def source_forecasts(source, events):
        root, _expected, manifest, settlements = source
        with closing_iterator(
            _records(root, manifest, settlements=settlements)
        ) as rows:
            for item in rows:
                if type(item) is FundingEvent:
                    if events is not None:
                        events.append(item)
                else:
                    try:
                        yield item
                    except GeneratorExit:
                        return

    def forecasts(event_groups=None):
        # Settlement intervals stay disjoint; their prior PIT observations may
        # overlap. Merge those ordered streams without discarding older knowledge.
        with ExitStack() as readers:
            streams = [
                readers.enter_context(
                    closing_iterator(
                        source_forecasts(
                            source,
                            None if event_groups is None else event_groups[index],
                        )
                    )
                )
                for index, source in enumerate(sources)
            ]
            previous = None
            for item in merge(*streams, key=lambda item: item.available_at_ns):
                if (
                    previous is not None
                    and item.available_at_ns == previous.available_at_ns
                ):
                    if item != previous:
                        raise ValueError("binance_funding_record_conflict")
                    continue
                previous = item
                try:
                    yield item
                except GeneratorExit:
                    return

    event_groups = [[] for _ in sources]
    with closing_iterator(forecasts(event_groups)) as rows:
        for _item in rows:
            pass
    events = tuple(item for group in event_groups for item in group)

    with closing_iterator(forecasts()) as source:
        qualification = []
        for _root, _expected, manifest, *_rest in sources:
            metadata = manifest["metadata"]
            native = metadata["settlement_source"]
            qualification.append(
                {
                    "coverage": metadata["coverage"],
                    "settlement_source": native,
                    "verification": "declared_archive_not_independently_verified"
                    if native is None
                    else "official_response_validated"
                    if native["source_id"] == "binance"
                    else (
                        "supplier_completeness_assumption_not_independently_verified"
                    ),
                }
            )
        yield FundingObservations(
            identity, tuple(coverage), tuple(events), source, tuple(qualification)
        )
