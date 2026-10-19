"""Reference-source interim targets for A-share public facts."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
)
from quant_research_platform.contracts.portfolios import (
    A_SHARE_EQUITY_EVENTS_DATA_ID,
    EQUITY_EVENT_COLUMNS,
)
from quant_research_platform.contracts.validation import encode_exact_numeric
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FeeSourceRow,
    load_a_share_fee_source,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    _hash_file_content,
    filesystem_exists,
    filesystem_is_dir,
    filesystem_is_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

SOURCE_ID = SourceId("reference")
A_SHARE_FEE_SCHEDULE_DATA_ID = "a_share_fee_schedule"
A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID = "a_share_industry_snapshots"
_INDUSTRY_MANIFEST_NAME = "metadata.json"
_INDUSTRY_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "observation_date",
    "publication_date",
    "industry_standard_id",
    "industry_category_code",
    "industry_major_code",
    "industry_code",
    "company_name",
    "snapshot_id",
    "source_locator",
)
FEE_SCHEDULE_COLUMNS = (
    "fee_code",
    "effective_date",
    "end_date",
    "market",
    "side",
    "basis",
    "rate",
    "included_in_ledger",
    "included_in",
)
_FEE_SCHEDULE_ROLES = MappingProxyType(
    {
        "fee_code": ColumnRole.KEY,
        "effective_date": ColumnRole.TIMESTAMP,
        "end_date": ColumnRole.KEY,
        "market": ColumnRole.KEY,
        "side": ColumnRole.KEY,
        "basis": ColumnRole.KEY,
        "rate": ColumnRole.CONTEXT,
        "included_in_ledger": ColumnRole.CONTEXT,
        "included_in": ColumnRole.CONTEXT,
    }
)


@dataclass(frozen=True, slots=True)
class ReferenceInterimImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def fee_schedule_spec() -> InterimDataSpec:
    return InterimDataSpec(
        data_id=A_SHARE_FEE_SCHEDULE_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles=_FEE_SCHEDULE_ROLES,
        primary_key=(
            "fee_code",
            "effective_date",
            "end_date",
            "market",
            "side",
            "basis",
        ),
        partition_columns=("year",),
    )


def interim_catalog() -> InterimDataCatalog:
    return InterimDataCatalog(
        (fee_schedule_spec(), _industry_snapshots_spec(), _equity_events_spec())
    )


def materialize_interim_data(
    *,
    raw_root: Path,
    output_roots: Mapping[str, Path],
    data_ids: tuple[str, ...] = (A_SHARE_FEE_SCHEDULE_DATA_ID,),
    start_date: str | None = None,
    end_date: str | None = None,
    metadata: Mapping[str, object] | None = None,
    fee_csv: Path | None = None,
    fee_sidecar: Path | None = None,
) -> ReferenceInterimImportResult:
    """Import independent reference tables, retaining causal state history.

    start_date is unsupported: state established before a research window remains
    necessary. end_date bounds industry/equity knowledge; fee intervals stay intact.
    Each output publishes independently; this is not a multi-output transaction.
    """
    if start_date is not None:
        raise ValueError("reference_start_date_not_consumed")
    if (
        type(data_ids) is not tuple
        or not data_ids
        or any(type(data_id) is not str for data_id in data_ids)
        or len(set(data_ids)) != len(data_ids)
        or not set(data_ids)
        <= {
            A_SHARE_FEE_SCHEDULE_DATA_ID,
            A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
            A_SHARE_EQUITY_EVENTS_DATA_ID,
        }
    ):
        raise ValueError("reference_interim_data_ids_invalid")
    if (fee_csv is None) != (fee_sidecar is None):
        raise ValueError("a_share_fee_source_pair_required")
    if fee_csv is not None and A_SHARE_FEE_SCHEDULE_DATA_ID not in data_ids:
        raise ValueError("reference_fee_files_not_consumed")
    root = require_unaliased_path(raw_root, error="reference_source_root_invalid")
    if filesystem_exists(root) and not filesystem_is_dir(root):
        raise ValueError("reference_source_root_invalid")
    written = {}
    skipped = []
    for data_id in data_ids:
        if data_id != A_SHARE_FEE_SCHEDULE_DATA_ID or fee_csv is None:
            dataset_root = require_unaliased_path(
                root / data_id, error="reference_source_root_invalid"
            )
            if not filesystem_exists(dataset_root):
                skipped.append(data_id)
                continue
            if not filesystem_is_dir(dataset_root):
                raise ValueError("reference_source_root_invalid")
            required = (
                (A_SHARE_FEE_CSV_NAME, A_SHARE_FEE_SIDECAR_NAME)
                if data_id == A_SHARE_FEE_SCHEDULE_DATA_ID
                else ("metadata.json",)
            )
            if not all(filesystem_is_file(dataset_root / name) for name in required):
                raise ValueError(
                    "equity_event_metadata_missing"
                    if data_id == A_SHARE_EQUITY_EVENTS_DATA_ID
                    else "reference_source_required_member_invalid"
                )
        if data_id == A_SHARE_EQUITY_EVENTS_DATA_ID:
            equity_frame, identity = _equity_source(root, end_date=end_date)
        elif data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            source_metadata, members, identity = _industry_source(root)
        else:
            source = load_a_share_fee_source(
                reference_root=root, fee_csv=fee_csv, fee_sidecar=fee_sidecar
            )
            identity = source.source_identity
        if data_id == A_SHARE_EQUITY_EVENTS_DATA_ID:
            frame = equity_frame
            spec = _equity_events_spec()
        elif data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            frame = _industry_frame(source_metadata, members, end_date=end_date)
            if end_date is not None:
                frame = frame.loc[frame["timestamp"].le(pd.Timestamp(end_date))].copy()
            spec = _industry_snapshots_spec()
        else:
            frame = _frame_from_rows(source.rows)
            spec = fee_schedule_spec()

        artifact_root = Path(output_roots[spec.data_id])
        write_dataset(
            artifact_root,
            frame,
            spec=spec,
            metadata={
                **(metadata or {}),
                "producer_diagnostics": {
                    "source_row_counts": frame.attrs.get("source_row_counts", {})
                },
                "source_evidence": {
                    "source_id": str(SOURCE_ID),
                    "content_hash": identity,
                },
            },
        )
        written[spec.data_id] = artifact_root
    return ReferenceInterimImportResult(written, tuple(skipped))


def _frame_from_rows(rows: tuple[FeeSourceRow, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "fee_code": row.fee_code,
                "effective_date": pd.Timestamp(row.effective_date),
                "end_date": pd.Timestamp(row.end_date),
                "market": row.market,
                "side": row.side,
                "basis": row.basis,
                "rate": row.rate,
                "included_in_ledger": row.included_in_ledger,
                "included_in": row.included_in,
            }
            for row in rows
        ],
        columns=FEE_SCHEDULE_COLUMNS,
    )
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame["end_date"] = pd.to_datetime(frame["end_date"])
    frame["rate"] = frame["rate"].astype("float64")
    frame["included_in_ledger"] = frame["included_in_ledger"].astype("bool")
    frame["included_in"] = frame["included_in"].astype("string")
    return frame


def _industry_snapshots_spec() -> InterimDataSpec:
    return InterimDataSpec(
        data_id=A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            column: (
                ColumnRole.TIMESTAMP
                if column == "timestamp"
                else ColumnRole.KEY
                if column in {"source_asset_id", "snapshot_id"}
                else ColumnRole.CONTEXT
            )
            for column in _INDUSTRY_COLUMNS
        },
        primary_key=("source_asset_id", "timestamp", "snapshot_id"),
        partition_columns=("year",),
    )


def _snapshot_date(value):
    if type(value) is not str or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("industry_snapshot_date_invalid")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("industry_snapshot_date_invalid") from exc


def _industry_source(reference_root):
    root = Path(reference_root) / A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    path = root / _INDUSTRY_MANIFEST_NAME
    raw, metadata = read_qrp_json_file(path)
    if (
        set(metadata) != {"schema_id", "files", "snapshots"}
        or metadata["schema_id"] != "a_share_industry_snapshots/0.1"
        or type(metadata["files"]) is not dict
        or not metadata["files"]
        or type(metadata["snapshots"]) is not list
        or not metadata["snapshots"]
    ):
        raise ValueError("industry_snapshot_metadata_invalid")
    candidates = {
        snapshot.get("candidate_file")
        for snapshot in metadata["snapshots"]
        if type(snapshot) is dict and type(snapshot.get("candidate_file")) is str
    }
    members = {}
    for name, expected_hash in metadata["files"].items():
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or name == _INDUSTRY_MANIFEST_NAME
            or type(expected_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError("industry_snapshot_member_invalid")
        member = require_unaliased_path(root / name, error="invalid_dataset_part_files")
        if name in candidates:
            candidate_raw, candidate = read_qrp_json_file(member)
            digest = hashlib.sha256(candidate_raw).hexdigest()
            members[name] = candidate
        else:
            digest = _hash_file_content(member)
            members[name] = None
        if digest != expected_hash:
            raise ValueError("industry_snapshot_member_hash_mismatch")
    seen = set()
    for snapshot in metadata["snapshots"]:
        if type(snapshot) is not dict or set(snapshot) != {
            "snapshot_id",
            "industry_standard_id",
            "observation_date",
            "publication_date",
            "source_url",
            "candidate_file",
            "document_file",
            "page_file",
        }:
            raise ValueError("industry_snapshot_metadata_invalid")
        standard = snapshot["industry_standard_id"]
        if type(standard) is not str or standard not in {"CSRC2012", "CAPCO"}:
            raise ValueError("industry_snapshot_standard_invalid")
        identifier = snapshot["snapshot_id"]
        if (
            type(identifier) is not str
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", identifier) is None
            or identifier in seen
        ):
            raise ValueError("industry_snapshot_id_invalid")
        seen.add(identifier)
        observed = _snapshot_date(snapshot["observation_date"])
        published = _snapshot_date(snapshot["publication_date"])
        if observed > published:
            raise ValueError("industry_snapshot_publication_invalid")
        if type(snapshot["source_url"]) is not str:
            raise ValueError("industry_snapshot_source_invalid")
        url = urlsplit(snapshot["source_url"])
        expected_host = (
            "www.csrc.gov.cn" if standard == "CSRC2012" else "www.capco.org.cn"
        )
        if url.scheme != "https" or url.netloc != expected_host or not url.path:
            raise ValueError("industry_snapshot_source_invalid")
        for key in ("candidate_file", "document_file", "page_file"):
            name = snapshot[key]
            if type(name) is not str or name not in members:
                raise ValueError("industry_snapshot_member_invalid")
        if not snapshot["candidate_file"].endswith(".json") or not snapshot[
            "page_file"
        ].endswith(".html"):
            raise ValueError("industry_snapshot_member_invalid")
    source_identity = canonical_json_hash(
        {
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "files": metadata["files"],
        }
    )
    return metadata, members, source_identity


def _industry_frame(metadata, members, *, end_date=None):
    rows = []
    counts = {"excluded_missing_required_rows": 0, "invalid_industry_rows": 0}
    for snapshot in metadata["snapshots"]:
        dates = (
            pd.Timestamp(snapshot["publication_date"]) + pd.Timedelta(days=1),
            pd.Timestamp(snapshot["observation_date"]),
            pd.Timestamp(snapshot["publication_date"]),
        )
        if end_date is not None and dates[0] > pd.Timestamp(end_date):
            continue
        candidate = members[snapshot["candidate_file"]]
        if (
            set(candidate) != {"source", "source_sha256", "records"}
            or candidate["source"] != snapshot["document_file"]
            or candidate["source_sha256"]
            != metadata["files"][snapshot["document_file"]]
            or type(candidate["records"]) is not list
            or not candidate["records"]
        ):
            raise ValueError("industry_snapshot_candidate_invalid")
        for row in candidate["records"]:
            if type(row) is not dict:
                raise ValueError("industry_snapshot_row_invalid")
            code = row.get("code")
            if type(code) is not str or re.fullmatch(r"\d{6}", code) is None:
                counts["excluded_missing_required_rows"] += 1
                continue
            category, major, industry = (
                row.get(key) for key in ("category", "major", "industry")
            )
            valid = (
                type(category) is str
                and re.fullmatch(r"[A-S]", category) is not None
                and type(industry) is str
                and re.fullmatch(r"\d{2}", industry) is not None
                and type(major) is str
            )
            if valid:
                valid = (
                    re.fullmatch(r"C[A-Z]", major) is not None
                    if snapshot["industry_standard_id"] == "CAPCO" and category == "C"
                    else major == ""
                )
            if not valid:
                counts["invalid_industry_rows"] += 1
                category = major = industry = None
            rows.append(
                (
                    code,
                    *dates,
                    snapshot["industry_standard_id"],
                    category,
                    major,
                    industry,
                    row.get("name") if isinstance(row.get("name"), str) else None,
                    snapshot["snapshot_id"],
                    row.get("source_locator")
                    if isinstance(row.get("source_locator"), str)
                    else None,
                )
            )
        del candidate
    frame = pd.DataFrame(rows, columns=_INDUSTRY_COLUMNS)
    key = ["source_asset_id", "timestamp", "snapshot_id"]
    frame = frame.drop_duplicates()
    conflicting = frame.duplicated(key, keep=False)
    if conflicting.any():
        for column in set(_INDUSTRY_COLUMNS).difference(key):
            different = frame.groupby(key, dropna=False)[column].transform(
                lambda values: values.nunique(dropna=False) > 1
            )
            frame.loc[conflicting & different, column] = None
        frame = frame.drop_duplicates(key)
    for column in ("timestamp", "observation_date", "publication_date"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    frame.attrs["source_row_counts"] = counts
    for column in _INDUSTRY_COLUMNS:
        if column not in {
            "timestamp",
            "observation_date",
            "publication_date",
        }:
            frame[column] = frame[column].astype("string")
    return frame


def _equity_events_spec():
    return InterimDataSpec(
        data_id=A_SHARE_EQUITY_EVENTS_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            column: ColumnRole.TIMESTAMP
            if column == "timestamp"
            else ColumnRole.KEY
            if column in {"source_asset_id", "event_id"}
            else ColumnRole.CONTEXT
            for column in EQUITY_EVENT_COLUMNS
        },
        primary_key=("source_asset_id", "timestamp", "event_id"),
        partition_columns=("year",),
    )


def _equity_source(reference_root, *, end_date=None):
    """Admit official facts stated per actual share and lifecycle."""
    root = Path(reference_root) / A_SHARE_EQUITY_EVENTS_DATA_ID
    path = root / "metadata.json"
    raw, metadata = read_qrp_json_file(path)
    if (
        set(metadata) != {"schema_id", "files", "events"}
        or metadata["schema_id"] != "a_share_equity_events"
        or type(metadata["files"]) is not dict
        or not metadata["files"]
        or type(metadata["events"]) is not list
        or not metadata["events"]
    ):
        raise ValueError("equity_event_metadata_invalid")
    members = {}
    for name, digest in metadata["files"].items():
        if (
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or name == "metadata.json"
            or type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("equity_event_member_invalid")
        member = require_unaliased_path(root / name, error="invalid_dataset_part_files")
        if _hash_file_content(member) != digest:
            raise ValueError("equity_event_member_hash_mismatch")
        members[name] = member
    rows, seen = [], set()
    keys = {
        "event_id",
        "asset_id",
        "listing_date",
        "kind",
        "effective_date",
        "publication_date",
        "observed_at",
        "currency",
        "amount_per_share",
        "remaining_old_units_per_old_unit",
        "successor_asset_id",
        "successor_listing_date",
        "successor_shares_per_old_share",
        "source_url",
        "document_file",
        "source_locator",
    }

    def number(value, *, positive=False):
        if (
            type(value) not in (int, float)
            or (type(value) is float and not math.isfinite(value))
            or value < 0
            or (positive and value == 0)
        ):
            raise ValueError("equity_event_numeric_invalid")
        return Fraction(value)

    for event in metadata["events"]:
        if type(event) is not dict or set(event) != keys:
            raise ValueError("equity_event_row_invalid")
        for key in ("document_file",):
            if type(event[key]) is not str or event[key] not in members:
                raise ValueError("equity_event_member_invalid")
        try:
            effective = pd.Timestamp(_snapshot_date(event["effective_date"]))
            published = pd.Timestamp(_snapshot_date(event["publication_date"]))
            available = published + pd.Timedelta(days=1)
        except (TypeError, ValueError) as exc:
            raise ValueError("equity_event_date_invalid") from exc
        timestamp = max(effective, available)
        if end_date is not None and not timestamp <= pd.Timestamp(end_date):
            continue
        if any(
            type(event[key]) is not str or not event[key].strip()
            for key in (
                "event_id",
                "asset_id",
                "kind",
                "source_url",
                "source_locator",
            )
        ):
            raise ValueError("equity_event_row_invalid")
        if event["event_id"] in seen:
            raise ValueError("equity_event_id_duplicate")
        seen.add(event["event_id"])
        if re.fullmatch(r"[0-9]{6}", event["asset_id"]) is None:
            raise ValueError("equity_event_asset_invalid")
        kind = event["kind"]
        if (
            kind
            not in {"quote", "cash_recovery", "cancel", "conversion", "share_change"}
            or event["currency"] != "CNY"
        ):
            raise ValueError("equity_event_kind_or_currency_invalid")
        url = urlsplit(event["source_url"])
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or not any(
                url.hostname == host or url.hostname.endswith("." + host)
                for host in (
                    "sse.com.cn",
                    "szse.cn",
                    "cninfo.com.cn",
                    "neeq.com.cn",
                    "neeq.cc",
                )
            )
        ):
            raise ValueError("equity_event_source_invalid")
        try:
            listed = pd.Timestamp(_snapshot_date(event["listing_date"]))
            if type(event["observed_at"]) is not str:
                raise ValueError("invalid_observation_timestamp")
            observed = pd.Timestamp(event["observed_at"])
            if observed.tzinfo is None:
                raise ValueError("missing_observation_timezone")
            observed_market_date = (
                observed.tz_convert("Asia/Shanghai").tz_localize(None).normalize()
            )
            observed = observed.tz_convert("UTC").tz_localize(None)
            if listed > effective or observed_market_date < published:
                raise ValueError("invalid_date_order")
        except (TypeError, ValueError) as exc:
            raise ValueError("equity_event_date_invalid") from exc
        remaining = number(event["remaining_old_units_per_old_unit"])
        value = None
        if kind in {"quote", "cash_recovery"}:
            amount = number(event["amount_per_share"], positive=kind == "quote")
            value = amount
        elif event["amount_per_share"] is not None:
            raise ValueError("equity_event_kind_fields_invalid")
        if (kind == "quote" and remaining != 1) or (
            kind in {"cancel", "conversion"} and remaining != 0
        ):
            raise ValueError("equity_event_kind_fields_invalid")
        successor, successor_listing, ratio = None, pd.NaT, None
        successor_keys = (
            "successor_asset_id",
            "successor_listing_date",
            "successor_shares_per_old_share",
        )
        if kind == "conversion":
            successor = event["successor_asset_id"]
            if (
                type(successor) is not str
                or re.fullmatch(r"[0-9]{6}", successor) is None
            ):
                raise ValueError("equity_event_successor_invalid")
            successor_listing = pd.Timestamp(
                _snapshot_date(event["successor_listing_date"])
            )
            if successor_listing > effective or (
                successor == event["asset_id"] and successor_listing == listed
            ):
                raise ValueError("equity_event_successor_invalid")
            shares = number(event["successor_shares_per_old_share"], positive=True)
            ratio = shares
        elif any(event[key] is not None for key in successor_keys):
            raise ValueError("equity_event_kind_fields_invalid")
        evidence = canonical_json_hash(
            {
                "event": event,
                "files": {
                    name: metadata["files"][name] for name in (event["document_file"],)
                },
            }
        )
        rows.append(
            (
                event["asset_id"],
                event["asset_id"],
                listed,
                timestamp,
                event["event_id"],
                kind,
                effective,
                available,
                observed,
                "CNY",
                None if value is None else encode_exact_numeric(value),
                encode_exact_numeric(Fraction(remaining)),
                successor,
                successor_listing,
                None if ratio is None else encode_exact_numeric(ratio),
                evidence,
                event["source_locator"],
            )
        )
    frame = pd.DataFrame(rows, columns=EQUITY_EVENT_COLUMNS)
    dates = {
        "listing_date",
        "timestamp",
        "effective_at",
        "available_at",
        "observed_at",
        "successor_listing_date",
    }
    for column in EQUITY_EVENT_COLUMNS:
        frame[column] = (
            pd.to_datetime(frame[column])
            if column in dates
            else frame[column].astype("string")
        )
    frame = frame.sort_values(
        ["timestamp", "source_asset_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    identity = canonical_json_hash(
        {"metadata_sha256": hashlib.sha256(raw).hexdigest(), "files": metadata["files"]}
    )
    return frame, identity
