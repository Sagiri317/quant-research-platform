"""Prepare public observations without claiming unknown historical availability."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.storage.datasets.writing import write_dataset

from .normalization import DATASETS, normalize

PROVENANCE = (
    "source",
    "source_url",
    "source_sha256",
    "retrieved_at",
    "publication_date",
    "published_at",
    "available_at",
    "availability_basis",
)


def interim_catalog():
    return InterimDataCatalog(
        tuple(
            InterimDataSpec(
                data_id=name,
                source_id="public_research",
                frequency="tick",
                grain="asset_timestamp",
                column_roles={
                    column: "timestamp"
                    if column == "timestamp"
                    else "key"
                    if column in {"asset_id", "source_sha256", "source"}
                    else "context"
                    for column in ("asset_id", "timestamp", *fields, *PROVENANCE)
                },
                primary_key=("asset_id", "timestamp", "source", "source_sha256"),
                partition_columns=("year",),
                column_units={k: v for k, v in fields.items()},
            )
            for name, fields in DATASETS.items()
        )
    )


@dataclass(frozen=True)
class PublicImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def materialize_interim_data(
    *,
    raw_root,
    output_roots,
    data_ids,
    start_date=None,
    end_date=None,
    manifests=("manifest.json",),
):
    """Normalize explicitly requested facts, preserving source-file versions.

    timestamp is the observation/report coordinate, NOT the release clock.
    Unknown publication times stay null; a later PIT operation must supply an
    evidence-backed release calendar or an explicit documented assumption.
    """
    raw_root = Path(raw_root).resolve()
    if set(data_ids) != set(output_roots) or set(data_ids) - DATASETS.keys():
        raise ValueError("public_interim_outputs_invalid")
    records, manifest_evidence = [], []
    if not manifests or len(set(manifests)) != len(manifests):
        raise ValueError("public_source_manifests_invalid")
    for name in manifests:
        path = (raw_root / name).resolve()
        if not path.is_relative_to(raw_root):
            raise ValueError("public_source_path_outside_root")
        payload = path.read_bytes()
        manifest = json.loads(payload)
        if manifest["schema_version"] != 1:
            raise ValueError("public_source_manifest_version_invalid")
        records.extend(manifest["records"])
        manifest_evidence.append(
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("public_source_path_duplicate")
    catalog = interim_catalog()
    roots = {}
    skipped = []
    for data_id in data_ids:
        evidence = []
        complete_archives = set()

        def frames(
            data_id=data_id, evidence=evidence, complete_archives=complete_archives
        ):
            for record in records:
                if record["status"] != "downloaded":
                    continue
                path = raw_root / record["path"]
                if not path.resolve().is_relative_to(raw_root):
                    raise ValueError("public_source_path_outside_root")
                # Dispatch before reading: a different source's multi-GB ZIP is
                # not an input to a small margin/ETF import.
                from .normalization import accepts

                if not accepts(data_id, record):
                    continue
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != record["sha256"] or path.stat().st_size != record["bytes"]:
                    raise ValueError("public_source_integrity_mismatch")
                evidence.append(
                    {k: record[k] for k in ("path", "url", "sha256", "retrieved_at")}
                )
                if record["source"] == "cffex_daily" and path.suffix == ".zip":
                    if digest in complete_archives:
                        continue
                    complete_archives.add(digest)
                for frame in normalize(data_id, path, record):
                    if frame.empty:
                        continue
                    frame["source"] = record["source"]
                    if "source_url" not in frame:
                        frame["source_url"] = record["url"]
                    if "source_sha256" not in frame:
                        frame["source_sha256"] = digest
                    frame["retrieved_at"] = pd.Timestamp(record["retrieved_at"])
                    if "publication_date" not in frame:
                        frame["publication_date"] = pd.NaT
                    frame["publication_date"] = pd.to_datetime(
                        frame.publication_date
                    ).astype("datetime64[ns]")
                    if "published_at" not in frame:
                        frame["published_at"] = record.get("historical_published_at")
                    frame["published_at"] = pd.to_datetime(
                        frame.published_at, utc=True
                    ).astype("datetime64[ns, UTC]")
                    frame["retrieved_at"] = pd.to_datetime(
                        frame.retrieved_at, utc=True
                    ).astype("datetime64[ns, UTC]")
                    frame["timestamp"] = pd.to_datetime(frame.timestamp).astype(
                        "datetime64[ns]"
                    )
                    from quant_research_platform.sources.macro.interim import (
                        date_end_available,
                    )

                    zone = (
                        "Asia/Shanghai"
                        if record["source"].startswith(
                            ("chinaamc", "sse_", "szse_", "cffex")
                        )
                        else "America/New_York"
                    )
                    dated_release = date_end_available(frame.publication_date, zone)
                    frame["available_at"] = frame.published_at.fillna(dated_release)
                    frame["availability_basis"] = "unknown_historical_release"
                    frame.loc[frame.publication_date.notna(), "availability_basis"] = (
                        "publication_date_upper_bound"
                    )
                    frame.loc[frame.published_at.notna(), "availability_basis"] = (
                        "publication_timestamp"
                    )
                    if start_date is not None:
                        frame = frame.loc[frame.timestamp.ge(pd.Timestamp(start_date))]
                    if end_date is not None:
                        frame = frame.loc[
                            frame.timestamp.lt(
                                pd.Timestamp(end_date) + pd.Timedelta(days=1)
                            )
                        ]
                    for field, unit in DATASETS[data_id].items():
                        if unit in {"identifier", "text"}:
                            frame[field] = frame[field].astype("str")
                        elif unit == "date":
                            frame[field] = pd.to_datetime(frame[field]).astype(
                                "datetime64[ns]"
                            )
                        else:
                            frame[field] = pd.to_numeric(
                                frame[field], errors="raise"
                            ).astype("Float64")
                    if not frame.empty:
                        yield frame.loc[:, list(catalog.get(data_id).required_columns)]

        iterator = frames()
        first = next(iterator, None)
        if first is None:
            skipped.append(data_id)
            continue
        from itertools import chain

        target = Path(output_roots[data_id])
        write_dataset(
            target,
            chain((first,), iterator),
            spec=catalog.get(data_id),
            metadata={
                "source_evidence": {
                    "manifests": manifest_evidence,
                    "records": evidence,
                },
                "parameters": {
                    "timestamp_semantics": "observation_not_availability",
                    "publication_imputation": "none",
                },
            },
        )
        roots[data_id] = target
    return PublicImportResult(roots, tuple(skipped))
