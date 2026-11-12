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
    fund_identity_csv=None,
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
        scope = manifest.get("path_base")
        if scope not in (
            None,
            "manifest_directory",
            "public_research_raw_root",
            "data/raw/public_research",
        ):
            raise ValueError("public_source_path_base_invalid")
        directory = path.parent if scope == "manifest_directory" else raw_root
        for record in manifest["records"]:
            source_path = (directory / record["path"]).resolve()
            if not source_path.is_relative_to(raw_root):
                raise ValueError("public_source_path_outside_root")
            records.append(
                {**record, "path": source_path.relative_to(raw_root).as_posix()}
            )
        manifest_evidence.append(
            {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("public_source_path_duplicate")
    catalog = interim_catalog()
    clocks, clock_hash = _release_clocks(release_csv)
    report_dates, cell_dates, correction_evidence = _report_observation_corrections(
        raw_root, records, data_ids
    )
    fund_identities = None
    if fund_identity_csv is not None and "public_fund_nav" in data_ids:
        fund_identities = pd.read_csv(
            fund_identity_csv,
            dtype="str",
            usecols=["fund_code", "clock_source_key", "clock_valid_from"],
        )
        if fund_identities.fund_code.duplicated().any():
            raise ValueError("public_fund_identity_duplicate")
        fund_identities["clock_valid_from"] = pd.to_datetime(
            fund_identities.clock_valid_from
        )
        fund_identities = fund_identities.set_index("fund_code")
    roots = {}
    skipped = []
    for data_id in data_ids:
        evidence = correction_evidence.get(data_id, []).copy()
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
                    {
                        k: record[k]
                        for k in ("path", "url", "sha256", "retrieved_at")
                        if k in record
                    }
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
                        frame["source_url"] = pd.Series(
                            record.get("url"), index=frame.index, dtype="str"
                        )
                    if "source_sha256" not in frame:
                        frame["source_sha256"] = digest
                    if "retrieved_at" not in frame:
                        frame["retrieved_at"] = (
                            pd.Timestamp(record["retrieved_at"])
                            if "retrieved_at" in record
                            else pd.NaT
                        )
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
                    if data_id in correction_evidence:
                        corrected = pd.to_datetime(
                            frame.source_sha256.map(report_dates)
                        )
                        frame["timestamp"] = corrected.fillna(frame.timestamp)
                        if data_id == "public_fund_financial_cells":
                            frame["report_period_end"] = frame.timestamp
                            keys = pd.MultiIndex.from_arrays(
                                [frame.source_sha256, frame.asset_id]
                            )
                            revised = cell_dates.reindex(keys).set_axis(frame.index)
                            frame["value_observation_date"] = revised.fillna(
                                pd.to_datetime(frame.value_observation_date)
                            )
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
                        if data_id
                        in {
                            "public_nport_holdings",
                            "public_nport_funds",
                            "public_security_identifiers",
                        }
                        else "publication_date_upper_bound"
                    )
                    frame.loc[frame.published_at.notna(), "availability_basis"] = (
                        "publication_timestamp"
                    )
                    frame["availability_evidence_url"] = pd.Series(
                        pd.NA, index=frame.index, dtype="str"
                    )
                    if clocks is not None:
                        if data_id == "public_fund_nav" and fund_identities is not None:
                            identity = fund_identities.reindex(
                                frame.fund_code
                            ).set_axis(frame.index)
                            visible = (
                                identity.clock_source_key.notna()
                                & frame.timestamp.ge(identity.clock_valid_from)
                            )
                            frame["_clock_source"] = frame.source.where(
                                ~visible, identity.clock_source_key
                            )
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
                        estimated = clock.estimated_available_at.notna() & (
                            frame.available_at.isna()
                            | clock.estimated_available_at.ge(frame.available_at)
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
                    for name in (
                        ()
                        if data_id
                        in {"public_etf_actions", "public_domestic_etf_actions"}
                        else ("published_at", "available_at")
                    ):
                        local_date = (
                            frame[name]
                            .dt.tz_convert(zone)
                            .dt.tz_localize(None)
                            .dt.normalize()
                        )
                        if local_date.lt(observed).any():
                            raise ValueError("public_release_clock_precedes_dependency")
                    if (
                        data_id
                        not in {"public_etf_actions", "public_domestic_etf_actions"}
                        and frame.publication_date.lt(observed).any()
                    ):
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
                    **(
                        {
                            "fund_identity_sha256": hashlib.sha256(
                                Path(fund_identity_csv).read_bytes()
                            ).hexdigest()
                        }
                        if fund_identity_csv is not None
                        and "public_fund_nav" in data_ids
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


def _report_observation_corrections(raw_root, records, data_ids):
    """Apply explicit structured report evidence, retaining comparative cell dates."""
    dependents = set(data_ids) & {
        "public_fund_holdings",
        "public_fund_holdings_reports",
        "public_fund_financial_cells",
    }
    report_dates, cells, evidence = {}, {}, {}
    for record in records:
        source = record["source"]
        targets = (
            dependents
            if source == "cn_fund_report_observation_metadata_extracted"
            else dependents & {"public_fund_financial_cells"}
            if source == "cn_fund_financial_observation_date_corrections_extracted"
            else set()
        )
        if not targets or record["status"] != "downloaded":
            continue
        path = raw_root / record["path"]
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != record["sha256"]
            or len(payload) != record["bytes"]
        ):
            raise ValueError("public_source_integrity_mismatch")
        for target in targets:
            evidence.setdefault(target, []).append(
                {k: record[k] for k in ("path", "sha256")}
            )
        if source == "cn_fund_report_observation_metadata_extracted":
            for row in json.loads(payload)["records"]:
                key = row["original_report_source_sha256"]
                date = pd.Timestamp(row["actual_report_period_end"])
                if pd.isna(date) or (key in report_dates and report_dates[key] != date):
                    raise ValueError("public_report_observation_correction_invalid")
                report_dates[key] = date
        else:
            frame = pd.read_csv(path, dtype="str")
            for row in frame.itertuples(index=False):
                asset = ":".join(
                    [
                        row.report_version_identity,
                        row.table_kind,
                        row.leaf_table_index,
                        row.source_row_index,
                        row.expanded_column_index,
                    ]
                )
                key = (row.original_report_source_sha256, asset)
                date = pd.Timestamp(row.corrected_value_observation_date)
                if pd.isna(date) or (key in cells and cells[key] != date):
                    raise ValueError("public_report_observation_correction_invalid")
                cells[key] = date
    cell_dates = pd.Series(
        list(cells.values()),
        index=pd.MultiIndex.from_tuples(
            list(cells), names=["source_sha256", "asset_id"]
        ),
        dtype="datetime64[ns]",
    )
    return report_dates, cell_dates, evidence


def _source_timezone(source):
    return (
        "Asia/Shanghai"
        if source.startswith(
            (
                "chinaamc",
                "cn_fund_",
                "sse_",
                "szse_",
                "cffex",
                "etf_pcf_",
                "eastmoney_",
                "csf_",
                "csrc_",
                "public_index_weight_",
            )
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
    import csv
    import io

    required = {
        "source",
        "observation_date",
        "estimated_available_at",
        "availability_basis",
        "evidence_url",
    }
    selected = required | {"published_at"}
    header = next(csv.reader(io.StringIO(payload.decode("utf-8-sig"))), ())
    if (
        any(header.count(column) != 1 for column in required)
        or header.count("published_at") > 1
    ):
        raise ValueError("public_release_clock_columns_invalid")
    clocks = pd.read_csv(
        io.BytesIO(payload), dtype="str", usecols=lambda column: column in selected
    )
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
