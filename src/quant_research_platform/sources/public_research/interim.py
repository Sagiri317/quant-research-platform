"""Prepare public observations without claiming unknown historical availability."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.storage.datasets.writing import write_dataset

from .normalization import DATASETS, _numbers, normalize

PROVENANCE = (
    "source",
    "source_url",
    "source_sha256",
    "retrieved_at",
    "publication_date",
    "published_at",
    "available_at",
    "availability_basis",
    "availability_evidence_url",
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
    release_csv=None,
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
    clocks, clock_hash = _release_clocks(release_csv)
    roots = {}
    skipped = []
    for data_id in data_ids:
        evidence = []
        numeric_counts = {}
        complete_archives = set()

        def frames(
            data_id=data_id,
            evidence=evidence,
            complete_archives=complete_archives,
            numeric_counts=numeric_counts,
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
                    if "retrieved_at" not in frame:
                        frame["retrieved_at"] = pd.Timestamp(record["retrieved_at"])
                    if "publication_date" not in frame:
                        frame["publication_date"] = pd.NaT
                    publication_dates = frame.publication_date
                    frame["publication_date"] = pd.to_datetime(
                        frame.publication_date
                    ).astype("datetime64[ns]")
                    if (
                        publication_dates.notna()
                        & publication_dates.astype("str").str.strip().ne("")
                        & frame.publication_date.isna()
                    ).any():
                        raise ValueError("public_release_date_invalid")
                    if "published_at" not in frame:
                        frame["published_at"] = record.get("historical_published_at")
                    frame["published_at"] = _utc_times(frame.published_at)
                    frame["retrieved_at"] = _utc_times(frame.retrieved_at)
                    frame["timestamp"] = pd.to_datetime(frame.timestamp).astype(
                        "datetime64[ns]"
                    )
                    if frame.timestamp.isna().any():
                        raise ValueError("public_observation_date_invalid")
                    if start_date is not None:
                        frame = frame.loc[frame.timestamp.ge(pd.Timestamp(start_date))]
                    if end_date is not None:
                        frame = frame.loc[
                            frame.timestamp.lt(
                                pd.Timestamp(end_date) + pd.Timedelta(days=1)
                            )
                        ]
                    if frame.empty:
                        continue
                    from quant_research_platform.sources.macro.interim import (
                        date_end_available,
                    )

                    zone = _source_timezone(record["source"])
                    dated_release = date_end_available(frame.publication_date, zone)
                    frame["available_at"] = frame.published_at.fillna(dated_release)
                    frame["availability_basis"] = "unknown_historical_release"
                    frame.loc[frame.publication_date.notna(), "availability_basis"] = (
                        "public_nport_filing_date_rule_upper_bound"
                        if data_id == "public_nport_holdings"
                        else "publication_date_upper_bound"
                    )
                    frame.loc[frame.published_at.notna(), "availability_basis"] = (
                        "publication_timestamp"
                    )
                    frame["availability_evidence_url"] = pd.Series(
                        pd.NA, index=frame.index, dtype="str"
                    )
                    if clocks is not None:
                        keys = pd.MultiIndex.from_arrays(
                            [
                                frame.get("_clock_source", frame.source),
                                frame.timestamp.dt.normalize(),
                            ]
                        )
                        clock = clocks.reindex(keys).set_axis(frame.index)
                        actual = frame.published_at.isna() & clock.published_at.notna()
                        frame.loc[actual, "published_at"] = clock.loc[
                            actual, "published_at"
                        ]
                        frame.loc[actual, "available_at"] = clock.loc[
                            actual, "estimated_available_at"
                        ]
                        estimated = (
                            frame.available_at.isna()
                            & clock.estimated_available_at.notna()
                        )
                        applied = actual | estimated
                        frame.loc[estimated, "available_at"] = clock.loc[
                            estimated, "estimated_available_at"
                        ]
                        frame.loc[applied, "availability_basis"] = clock.loc[
                            applied, "availability_basis"
                        ]
                        frame.loc[applied, "availability_evidence_url"] = clock.loc[
                            applied, "evidence_url"
                        ]
                    if frame.available_at.lt(frame.published_at).any():
                        raise ValueError("public_release_clock_precedes_dependency")
                    observed = frame.timestamp.dt.normalize()
                    if data_id == "public_finra_margin":
                        observed += pd.offsets.MonthEnd(0)
                    for name in ("published_at", "available_at"):
                        local_date = (
                            frame[name]
                            .dt.tz_convert(zone)
                            .dt.tz_localize(None)
                            .dt.normalize()
                        )
                        if local_date.lt(observed).any():
                            raise ValueError("public_release_clock_precedes_dependency")
                    if frame.publication_date.lt(observed).any():
                        raise ValueError("public_release_clock_precedes_dependency")
                    numeric_fields = [
                        field
                        for field, unit in DATASETS[data_id].items()
                        if unit not in {"identifier", "text", "date"}
                    ]
                    _numbers(frame, numeric_fields)
                    for field, count in frame.attrs.get(
                        "invalid_numeric_cells", {}
                    ).items():
                        numeric_counts[field] = numeric_counts.get(field, 0) + count
                    for field, unit in DATASETS[data_id].items():
                        if unit in {"identifier", "text"}:
                            frame[field] = frame[field].astype("str")
                        elif unit == "date":
                            frame[field] = pd.to_datetime(frame[field]).astype(
                                "datetime64[ns]"
                            )
                        else:
                            frame[field] = frame[field].astype("Float64")
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
                "producer_diagnostics": {
                    "source_invalid_numeric_cells": numeric_counts
                },
                "source_evidence": {
                    "manifests": manifest_evidence,
                    "records": evidence,
                    **(
                        {"release_clock_sha256": clock_hash}
                        if clock_hash is not None
                        else {}
                    ),
                },
                "parameters": {
                    "timestamp_semantics": "observation_not_availability",
                    "publication_imputation": "explicit_release_csv"
                    if clocks is not None
                    else "none",
                },
            },
        )
        roots[data_id] = target
    return PublicImportResult(roots, tuple(skipped))


def _source_timezone(source):
    return (
        "Asia/Shanghai"
        if source.startswith(
            ("chinaamc", "cn_fund_", "sse_", "szse_", "cffex", "etf_pcf_")
        )
        else "America/New_York"
    )


def _utc_times(values):
    populated = values.dropna().astype("str")
    if not populated.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
        raise ValueError("public_release_clock_timezone_required")
    result = pd.to_datetime(values, utc=True, format="mixed").astype(
        "datetime64[ns, UTC]"
    )
    if (values.notna() & result.isna()).any():
        raise ValueError("public_release_clock_invalid")
    return result


def _release_clocks(path):
    """Read caller-supplied release evidence without deriving a hidden calendar."""
    if path is None:
        return None, None
    payload = Path(path).read_bytes()
    import io

    clocks = pd.read_csv(io.BytesIO(payload), dtype="str")
    required = {
        "source",
        "observation_date",
        "estimated_available_at",
        "availability_basis",
        "evidence_url",
    }
    if not required <= set(clocks) or set(clocks) - required - {"published_at"}:
        raise ValueError("public_release_clock_columns_invalid")
    if clocks[list(required)].isna().any(axis=None):
        raise ValueError("public_release_clock_required_value_missing")
    if "published_at" not in clocks:
        clocks["published_at"] = pd.NA
    for name in ("estimated_available_at", "published_at"):
        clocks[name] = _utc_times(clocks[name])
    clocks["observation_date"] = pd.to_datetime(clocks.observation_date)
    if clocks.observation_date.isna().any():
        raise ValueError("public_observation_date_invalid")
    if clocks.duplicated(["source", "observation_date"]).any():
        raise ValueError("public_release_clock_duplicate")
    return clocks.set_index(["source", "observation_date"]), hashlib.sha256(
        payload
    ).hexdigest()
