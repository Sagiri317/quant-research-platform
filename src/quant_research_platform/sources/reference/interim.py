"""Reference-source interim targets for A-share public facts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.business_manifest import (
    validate_business_manifest_envelope,
)
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
from quant_research_platform.sources.reference.a_share_fees import (
    A_SHARE_FEE_CSV_NAME,
    A_SHARE_FEE_SIDECAR_NAME,
    FeeSourceRow,
    load_a_share_fee_source,
)
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    _hash_file_content,
    dataset_file_identity,
    filesystem_is_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_bounded_qrp_json

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
    return InterimDataCatalog((fee_schedule_spec(), _industry_snapshots_spec()))


def source_content_identity(raw_root: Path, data_id: str) -> str | None:
    root = Path(raw_root)
    if data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
        if not filesystem_is_file(root / data_id / _INDUSTRY_MANIFEST_NAME):
            return None
        with _industry_source(root) as (_, _, identity):
            return identity
    if data_id != A_SHARE_FEE_SCHEDULE_DATA_ID:
        raise ValueError("unknown_reference_interim_data_id")
    csv_path = root / "a_share_fee_schedule" / A_SHARE_FEE_CSV_NAME
    sidecar_path = root / "a_share_fee_schedule" / A_SHARE_FEE_SIDECAR_NAME
    if not filesystem_is_file(csv_path) or not filesystem_is_file(sidecar_path):
        return None
    return load_a_share_fee_source(reference_root=root).source_identity


def materialize_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    data_ids: tuple[str, ...] = (A_SHARE_FEE_SCHEDULE_DATA_ID,),
    start_date: str | None = None,
    end_date: str | None = None,
    instance_metadata: Mapping[str, object] | None = None,
) -> ReferenceInterimImportResult:
    if type(data_ids) is not tuple or len(data_ids) != 1:
        raise ValueError("reference_interim_data_ids_invalid")
    data_id = data_ids[0]
    if type(data_id) is not str or data_id not in {
        A_SHARE_FEE_SCHEDULE_DATA_ID,
        A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,
    }:
        raise ValueError("reference_interim_data_ids_invalid")
    with ExitStack() as retained:
        root = (
            Path(raw_root)
            if raw_root is not None
            else layout.project_root / "data/raw" / str(SOURCE_ID)
        )
        if data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            metadata, members, identity = retained.enter_context(_industry_source(root))
        else:
            identity = source_content_identity(root, data_id)
        if (
            instance_metadata is not None
            and COMMON_MANIFEST_METADATA_KEY in instance_metadata
        ):
            envelope = validate_business_manifest_envelope(
                instance_metadata[COMMON_MANIFEST_METADATA_KEY]
            )
            if envelope["semantic_metadata"]["raw_content_identity"] != identity:
                raise ValueError("reference_raw_identity_changed")
        if data_id == A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            frame = _industry_frame(metadata, members)
            if end_date is not None:
                frame = frame.loc[frame["timestamp"].le(pd.Timestamp(end_date))].copy()
            spec = _industry_snapshots_spec()
        else:
            source = load_a_share_fee_source(reference_root=root)
            if source.source_identity != identity:
                raise ValueError("reference_raw_identity_changed")
            frame = _frame_from_rows(source.rows)
            spec = fee_schedule_spec()

        def verify_source(_receipt):
            if source_content_identity(root, data_id) != identity:
                raise ValueError("reference_raw_identity_changed")

        artifact_root = layout.target_root(
            spec.data_id,
            layout.interim_data_root(spec.source_id, spec.data_id),
        )
        write_dataset(
            artifact_root,
            frame,
            spec=spec,
            instance_metadata=instance_metadata,
            staged_validator=verify_source,
        )
        return ReferenceInterimImportResult(
            artifact_roots={spec.data_id: artifact_root},
            skipped=(),
        )


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


@contextmanager
def _industry_source(reference_root):
    root = Path(reference_root) / A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
    path = root / _INDUSTRY_MANIFEST_NAME
    expected = dataset_file_identity(path)
    raw, metadata = read_bounded_qrp_json(
        path, expected_identity=expected, max_bytes=128 * 1024
    )
    if (
        set(metadata) != {"schema_id", "files", "snapshots"}
        or metadata["schema_id"] != "a_share_industry_snapshots/0.1"
        or type(metadata["files"]) is not dict
        or not metadata["files"]
        or type(metadata["snapshots"]) is not list
        or not metadata["snapshots"]
    ):
        raise ValueError("industry_snapshot_metadata_invalid")
    members = {}
    for name, expected_hash in metadata["files"].items():
        if (
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or name == _INDUSTRY_MANIFEST_NAME
            or type(expected_hash) is not str
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ValueError("industry_snapshot_member_invalid")
        member = root / name
        identity = dataset_file_identity(member)
        digest = _hash_file_content(
            member,
            identity,
            "reference_raw_identity_changed",
            chunk_size=1024 * 1024,
        )
        if digest != expected_hash:
            raise ValueError("industry_snapshot_member_hash_mismatch")
        members[name] = (member, identity)
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
    yield metadata, members, source_identity
    if dataset_file_identity(path) != expected or any(
        dataset_file_identity(member) != identity
        for member, identity in members.values()
    ):
        raise ValueError("reference_raw_identity_changed")


def _industry_frame(metadata, members):
    rows = []
    for snapshot in metadata["snapshots"]:
        path, expected = members[snapshot["candidate_file"]]
        raw, candidate = read_bounded_qrp_json(path, expected_identity=expected)
        if (
            hashlib.sha256(raw).hexdigest()
            != metadata["files"][snapshot["candidate_file"]]
        ):
            raise ValueError("reference_raw_identity_changed")
        if (
            set(candidate) != {"source", "source_sha256", "records"}
            or candidate["source"] != snapshot["document_file"]
            or candidate["source_sha256"]
            != metadata["files"][snapshot["document_file"]]
            or type(candidate["records"]) is not list
            or not candidate["records"]
        ):
            raise ValueError("industry_snapshot_candidate_invalid")
        seen = set()
        for row in candidate["records"]:
            if type(row) is not dict or set(row) != {
                "code",
                "name",
                "category",
                "major",
                "industry",
                "source_locator",
            }:
                raise ValueError("industry_snapshot_row_invalid")
            if any(
                type(row[key]) is not str
                for key in (
                    "code",
                    "name",
                    "category",
                    "major",
                    "industry",
                    "source_locator",
                )
            ):
                raise ValueError("industry_snapshot_row_invalid")
            code, category, major = row["code"], row["category"], row["major"]
            if (
                re.fullmatch(r"\d{6}", code) is None
                or code in seen
                or not row["name"].strip()
                or re.fullmatch(r"[A-S]", category) is None
                or re.fullmatch(r"\d{2}", row["industry"]) is None
                or re.fullmatch(
                    r"(?:page:[1-9]\d*|table:[1-9]\d*;row:[1-9]\d*)",
                    row["source_locator"],
                )
                is None
            ):
                raise ValueError("industry_snapshot_row_invalid")
            if (
                snapshot["industry_standard_id"] == "CAPCO"
                and category == "C"
                and re.fullmatch(r"C[A-Z]", major) is None
                or (snapshot["industry_standard_id"] != "CAPCO" or category != "C")
                and major != ""
            ):
                raise ValueError("industry_snapshot_major_invalid")
            seen.add(code)
            if len(seen) == 1:
                dates = (
                    pd.Timestamp(snapshot["publication_date"]) + pd.Timedelta(days=1),
                    pd.Timestamp(snapshot["observation_date"]),
                    pd.Timestamp(snapshot["publication_date"]),
                )
            rows.append(
                (
                    code,
                    *dates,
                    snapshot["industry_standard_id"],
                    category,
                    major,
                    row["industry"],
                    row["name"],
                    snapshot["snapshot_id"],
                    row["source_locator"],
                )
            )
        del raw, candidate
    frame = pd.DataFrame(rows, columns=_INDUSTRY_COLUMNS)
    for column in _INDUSTRY_COLUMNS:
        if column not in {
            "timestamp",
            "observation_date",
            "publication_date",
        }:
            frame[column] = frame[column].astype("string")
    return frame
