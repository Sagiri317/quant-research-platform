"""Immutable per-symbol funding observations with separate known forecasts."""

import hashlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    json_record_working_set_bytes,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioTargetBlocked,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
)
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    filesystem_is_file,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_manifest
from quant_research_platform.storage.raw_snapshots import (
    MAX_RECORD_BYTES,
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
            }
            or native["endpoint"] != settlement_endpoint(native["source_id"], symbol)
            or native["symbol"] != symbol
            or type(native["start_ms"]) is not int
            or type(native["end_ms"]) is not int
            or native["start_ms"] * 1_000_000 > coverage["start_ns"]
            or (native["end_ms"] + 1) * 1_000_000 <= coverage["end_ns"]
            or native["http_status"] != 200
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
        before = dataset_file_identity(path)
        with open_filesystem_file(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if (
            before != dataset_file_identity(path)
            or before.size != native["byte_count"]
            or digest != native["sha256"]
        ):
            raise ValueError("binance_funding_settlement_source_changed")
    return manifest


def _records(root, manifest):
    previous = {"settlement": -1, "forecast": -1}
    rows = iter_csv_snapshot(
        root, expected_manifest=manifest, max_bytes=manifest["byte_count"]
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
                yield FundingEvent(timestamp, rate, mark)
            else:
                if rate is not None or mark is not None or type(cashflows) is not list:
                    raise ValueError("binance_funding_forecast_invalid")
                if any(type(item) is not list or len(item) != 2 for item in cashflows):
                    raise ValueError("binance_funding_forecast_invalid")
                yield FundingForecast(
                    timestamp, known_until, tuple(tuple(item) for item in cashflows)
                )


@contextmanager
def read_funding_observations(funding_roots, *, symbol, resource_policy=None):
    """Read ordered immutable sources without inventing coverage between them."""
    if symbol not in RESEARCH_SYMBOLS:
        raise ValueError("binance_funding_symbol_invalid")
    if type(funding_roots) is not tuple:
        raise ValueError("binance_funding_roots_invalid")
    if not funding_roots:
        raise PortfolioTargetBlocked("binance_funding_input_missing")
    with (
        runtime_resource_policy_scope(resource_policy) as policy,
        policy.lease(
            "binance.funding.read", json_record_working_set_bytes(MAX_RECORD_BYTES)
        ),
        policy.lease("binance.funding.retained", 1) as retained,
    ):
        sources = []
        coverage = []
        for value in funding_roots:
            retained.grow(4096 + len(str(value)) * 16)
            root = Path(value)
            manifest = _manifest(root, symbol)
            interval = manifest["metadata"]["coverage"]
            start, end = interval["start_ns"], interval["end_ns"]
            if coverage and start <= coverage[-1][1]:
                raise ValueError("binance_funding_source_intervals_invalid")
            coverage.append((start, end))
            sources.append((root, canonical_json_hash(manifest)))
            del manifest, interval
        identity = canonical_json_hash({"sources": [item[1] for item in sources]})

        def records():
            previous = {FundingEvent: -1, FundingForecast: -1}
            for root, expected in sources:
                manifest = _manifest(root, symbol)
                if canonical_json_hash(manifest) != expected:
                    raise ValueError("binance_funding_identity_changed")
                rows = _records(root, manifest)
                with closing_iterator(rows):
                    for item in rows:
                        kind = type(item)
                        stamp = (
                            item.timestamp_ns
                            if kind is FundingEvent
                            else item.available_at_ns
                        )
                        if stamp <= previous[kind]:
                            raise ValueError("binance_funding_record_order_invalid")
                        previous[kind] = stamp
                        yield item
                if canonical_json_hash(_manifest(root, symbol)) != expected:
                    raise ValueError("binance_funding_identity_changed")

        events = []
        with closing_iterator(records()) as rows:
            for item in rows:
                if type(item) is FundingEvent:
                    retained.grow(256)
                    events.append(item)

        def forecasts():
            with closing_iterator(records()) as rows:
                for item in rows:
                    if type(item) is FundingForecast:
                        yield item

        try:
            with closing_iterator(forecasts()) as source:
                yield FundingObservations(
                    identity, tuple(coverage), tuple(events), source
                )
        finally:
            primary = sys.exception()
            try:
                # Revalidate each source even if forecasts were only partly consumed.
                with closing_iterator(records()) as rows:
                    for _item in rows:
                        pass
            except BaseException as error:
                if primary is None or isinstance(primary, GeneratorExit):
                    raise
                primary.add_note(
                    "Funding source verification also failed: " + type(error).__name__
                )


def funding_source_identity(funding_roots, _target_id, *, symbol):
    try:
        with read_funding_observations(funding_roots, symbol=symbol) as observations:
            return observations.identity
    except PortfolioTargetBlocked:
        raise
    except (
        FileNotFoundError,
        NotADirectoryError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError("binance_funding_input_invalid") from error
