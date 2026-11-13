"""Reference-source interim targets for A-share public facts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
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
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FeeSourceRow,
    load_a_share_fee_source,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_is_dir,
    filesystem_is_file,
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
    root = Path(raw_root)
    if filesystem_exists(root) and not filesystem_is_dir(root):
        raise ValueError("reference_source_root_invalid")
    written = {}
    skipped = []
    for data_id in data_ids:
        if data_id != A_SHARE_FEE_SCHEDULE_DATA_ID or fee_csv is None:
            dataset_root = Path(root / data_id)
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
            snapshots, identity = _industry_source(root)
        else:
            source = load_a_share_fee_source(
                reference_root=root, fee_csv=fee_csv, fee_sidecar=fee_sidecar
            )
            identity = source.source_identity
        if data_id == A_SHARE_EQUITY_EVENTS_DATA_ID:
            frame = equity_frame
            spec = _equity_events_spec()
        elif data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            frame = _industry_frame(snapshots, end_date=end_date)
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
    raw, metadata = read_qrp_json_file(root / _INDUSTRY_MANIFEST_NAME)
    if metadata.get("schema_id") != "a_share_industry_snapshots":
        raise ValueError("industry_snapshot_metadata_invalid")
    snapshots = []
    digest = hashlib.sha256(raw)
    with (root / "snapshots.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            if record["status"] == "downloaded":
                snapshots.append(record)
    return snapshots, digest.hexdigest()


def _industry_frame(snapshots, *, end_date=None):
    rows = []
    counts = {"excluded_missing_required_rows": 0, "invalid_industry_rows": 0}
    for snapshot in snapshots:
        dates = (
            pd.Timestamp(snapshot["publication_date"]) + pd.Timedelta(days=1),
            pd.Timestamp(snapshot["observation_date"]),
            pd.Timestamp(snapshot["publication_date"]),
        )
        if end_date is not None and dates[0] > pd.Timestamp(end_date):
            continue
        if dates[1] > dates[2]:
            raise ValueError("industry_snapshot_publication_invalid")
        candidate = snapshot["candidate"]
        for row in candidate["records"]:
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
        not {"schema_id", "events"} <= set(metadata)
        or metadata["schema_id"] != "a_share_equity_events/1"
        or type(metadata["events"]) is not list
        or not metadata["events"]
    ):
        raise ValueError("equity_event_metadata_invalid")
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
        "document_sha256",
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
        return float(value)

    for event in metadata["events"]:
        if type(event) is not dict or not keys <= set(event):
            raise ValueError("equity_event_row_invalid")
        event = {key: event[key] for key in keys}
        if (
            type(event["document_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", event["document_sha256"]) is None
        ):
            raise ValueError("equity_event_document_hash_invalid")
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
        evidence = canonical_json_hash(event)
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
                value,
                remaining,
                successor,
                successor_listing,
                ratio,
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
            else frame[column].astype("float64")
            if column
            in {
                "value_per_old_unit",
                "remaining_old_units_per_old_unit",
                "successor_units_per_old_unit",
            }
            else frame[column].astype("string")
        )
    frame = frame.sort_values(
        ["timestamp", "source_asset_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    identity = hashlib.sha256(raw).hexdigest()
    return frame, identity
