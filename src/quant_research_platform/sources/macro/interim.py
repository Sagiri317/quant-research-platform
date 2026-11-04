"""Normalize macro vintages without inventing historical publication timestamps."""

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.sources.macro.catalog import SERIES
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.manifests import read_qrp_json_file

DATA_ID = "macro_observations"
KEY = ("series_id", "observation_date", "timestamp")
COLUMNS = (
    *KEY,
    "value",
    "unit",
    "frequency",
    "source_id",
    "source_series",
    "source_timezone",
    "published_at",
    "vintage_date",
    "realtime_end_date",
    "availability_basis",
    "retrieved_at",
    "source_url",
)


def interim_catalog():
    return InterimDataCatalog(
        (
            InterimDataSpec(
                data_id=DATA_ID,
                source_id="macro",
                frequency="tick",
                grain="market_timestamp",
                column_roles={
                    name: "timestamp"
                    if name == "timestamp"
                    else "key"
                    if name in KEY
                    else "context"
                    for name in COLUMNS
                },
                primary_key=KEY,
                partition_columns=("year",),
                column_units={"value": "series_specific_unit"},
            ),
        )
    )


def date_end_available(dates, timezone):
    """Use the next local midnight, including DST, as a declared date upper bound."""
    return (
        (pd.to_datetime(dates) + pd.Timedelta(days=1))
        .dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        .dt.tz_convert("UTC")
    )


def _frame(name, values, dates, *, record, timezone, vintage=None, end_dates=None):
    series = SERIES[name]
    output = pd.DataFrame({"observation_date": pd.to_datetime(dates), "value": values})
    output["series_id"] = name
    output["unit"] = series.unit
    output["frequency"] = series.frequency
    output["source_id"] = series.source
    output["source_series"] = series.code
    output["source_timezone"] = timezone
    output["published_at"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    output["retrieved_at"] = pd.Timestamp(record["retrieved_at"])
    output["source_url"] = record["url"]
    if vintage is None:
        output["timestamp"] = output["retrieved_at"]
        output["vintage_date"] = pd.NaT
        output["realtime_end_date"] = pd.Series(pd.NA, index=output.index, dtype="str")
        output["availability_basis"] = "retrieval_only"
    else:
        output["vintage_date"] = pd.to_datetime(vintage)
        output["timestamp"] = date_end_available(output.vintage_date, timezone)
        output["realtime_end_date"] = end_dates
        output["availability_basis"] = "archive_date_end"
    return output


def normalize_alfred(payload, name, record):
    """Retain each new or revised value at its archive vintage's conservative cutoff."""
    series = SERIES[name]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = pd.read_csv(
            archive.open("obs._by_real-time_period.csv"),
            na_values=["."],
            dtype={"realtime_end_date": "str"},
        )
    required = {
        "period_start_date",
        series.code,
        "realtime_start_date",
        "realtime_end_date",
    }
    if set(data) != required:
        raise ValueError("alfred_observation_columns_invalid")
    return _frame(
        name,
        data[series.code],
        data.period_start_date,
        record=record,
        timezone="America/Chicago",
        vintage=data.realtime_start_date,
        end_dates=data.realtime_end_date,
    )


def _normalize_response(payload, names, record, *, incomplete=False):
    source = SERIES[names[0]].source
    if source == "alfred":
        return [normalize_alfred(payload, names[0], record)]
    if source == "sf_fed":
        data = pd.read_csv(io.BytesIO(payload))
        # The publisher's chart declares basis points; current PCA weights are ex post.
        return [
            _frame(
                names[0],
                data[SERIES[names[0]].code],
                data.Date,
                record=record,
                timezone="America/New_York",
            )
        ]
    if source == "bis":
        tree = ET.fromstring(payload)
        series = [
            item for item in tree.iter() if item.tag.rsplit("}", 1)[-1] == "Series"
        ]
        if len(series) != 1 or series[0].get("UNIT_MULT") != "6":
            raise ValueError("bis_series_or_unit_invalid")
        data = pd.DataFrame([item.attrib for item in series[0]])
        dates = pd.PeriodIndex(data.TIME_PERIOD, freq="Q").to_timestamp()
        return [
            _frame(
                names[0], data.OBS_VALUE, dates, record=record, timezone="Europe/Zurich"
            )
        ]
    document = json.loads(payload)
    if source == "boj":
        items = document["RESULTSET"]
        if (
            len(items) != 1
            or items[0]["SERIES_CODE"] != SERIES[names[0]].code
            or items[0]["UNIT"] != "100 million yen"
            or document["NEXTPOSITION"] is not None
        ):
            raise ValueError("boj_series_unit_or_completeness_invalid")
        values = items[0]["VALUES"]
        dates = pd.to_datetime(
            pd.Series(values["SURVEY_DATES"]).astype(str), format="%Y%m"
        )
        return [
            _frame(
                names[0], values["VALUES"], dates, record=record, timezone="Asia/Tokyo"
            )
        ]
    if source == "hkma":
        if not document["header"]["success"]:
            if incomplete:
                return []
            raise ValueError("hkma_response_not_successful")
        data = pd.DataFrame(document["result"]["records"])
        date_column, timezone = "end_of_date", "Asia/Hong_Kong"
    else:
        data = pd.DataFrame(document["result"]["data"])
        date_column, timezone = "REPORT_DATE", "Asia/Shanghai"
    if data.empty:
        return []
    return [
        _frame(
            name,
            data[SERIES[name].code],
            data[date_column],
            record=record,
            timezone=timezone,
        )
        for name in names
    ]


def validate_observations(frame):
    """Validate external releases once; preserve explicit missing observations."""
    if not set(COLUMNS) <= set(frame) or not frame.series_id.isin(SERIES).all():
        raise ValueError("macro_observation_schema_invalid")
    frame = frame.loc[:, COLUMNS].copy()
    frame["value"] = pd.to_numeric(frame.value, errors="raise").astype(float)
    if np.isinf(frame.value.to_numpy()).any():
        raise ValueError("macro_value_must_be_finite_or_missing")
    for column in ("timestamp", "published_at", "retrieved_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("observation_date", "vintage_date"):
        frame[column] = pd.to_datetime(frame[column])
    if frame[list(KEY)].isna().any().any():
        raise ValueError("macro_observation_key_missing")
    if not frame.availability_basis.isin(
        ("release_timestamp", "release_date_end", "archive_date_end", "retrieval_only")
    ).all():
        raise ValueError("macro_availability_basis_invalid")
    for name, group in frame.groupby("series_id", sort=False):
        series = SERIES[name]
        if (
            not group.unit.eq(series.unit).all()
            or not group.frequency.eq(series.frequency).all()
        ):
            raise ValueError("macro_series_unit_or_frequency_mismatch")
    for column in (
        "series_id",
        "unit",
        "frequency",
        "source_id",
        "source_series",
        "source_timezone",
        "realtime_end_date",
        "availability_basis",
        "source_url",
    ):
        frame[column] = frame[column].astype("str")
    return frame.sort_values(
        ["timestamp", "series_id", "observation_date"], kind="stable"
    )


def read_release_csv(path):
    """Import timestamped original releases or revisions from an explicit local file.

    Each row supplies a publication timestamp or a release date with its timezone.
    Source identity and units stay explicit, including for private vendor exports.
    """
    data = pd.read_csv(path)
    required = {
        "series_id",
        "observation_date",
        "value",
        "unit",
        "frequency",
        "source_id",
        "source_series",
        "source_timezone",
        "source_url",
        "retrieved_at",
    }
    if not required <= set(data) or ("published_at" in data) == (
        "release_date" in data
    ):
        raise ValueError("macro_release_columns_invalid")
    if "published_at" in data:
        # Explicit offsets prevent machine-local time from becoming economic time.
        if not data.published_at.str.contains(
            r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True
        ).all():
            raise ValueError("macro_release_timezone_required")
        data["published_at"] = pd.to_datetime(data.published_at, utc=True)
        data["timestamp"] = data.published_at
        data["availability_basis"] = "release_timestamp"
        data["vintage_date"] = data.published_at.dt.tz_localize(None).dt.normalize()
    else:
        data["timestamp"] = pd.Series(
            pd.NaT, index=data.index, dtype="datetime64[ns, UTC]"
        )
        for zone, group in data.groupby("source_timezone", sort=False):
            data.loc[group.index, "timestamp"] = date_end_available(
                group.release_date, zone
            )
        data["vintage_date"] = pd.to_datetime(data.release_date)
        data["published_at"] = pd.NaT
        data["availability_basis"] = "release_date_end"
    data["realtime_end_date"] = pd.Series(pd.NA, index=data.index, dtype="str")
    return validate_observations(data)


@dataclass(frozen=True)
class MacroImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def materialize_interim_data(
    *,
    raw_root,
    output_roots,
    data_ids=(DATA_ID,),
    start_date=None,
    end_date=None,
    release_csv=None,
):
    """Prepare source observations only; no factors, fitting, or research are run."""
    if tuple(data_ids) != (DATA_ID,) or set(output_roots) != {DATA_ID}:
        raise ValueError("macro_interim_output_invalid")
    raw_root = Path(raw_root)
    frames, skipped, sources, incomplete = [], [], {}, []
    if release_csv is not None:
        frames.append(read_release_csv(release_csv))
    else:
        manifest = read_qrp_json_file(raw_root / "download.json")[1]
        for key, metadata in manifest["sources"].items():
            records = metadata["responses"]
            if metadata["status"] != "downloaded" and metadata["source"] == "hkma":
                records = [record for record in records if record["status_code"] == 200]
                incomplete.extend(metadata["series_ids"])
            elif metadata["status"] != "downloaded":
                skipped.extend(metadata["series_ids"])
                continue
            chunks = []
            for record in records:
                if record["file"] == "form.html":
                    continue
                member = raw_root / key / record["file"]
                if not member.resolve().is_relative_to(raw_root.resolve()):
                    raise ValueError("macro_source_member_outside_root")
                chunks.extend(
                    _normalize_response(
                        member.read_bytes(),
                        metadata["series_ids"],
                        record,
                        incomplete=metadata["status"] != "downloaded",
                    )
                )
            if chunks:
                data = pd.concat(chunks, ignore_index=True)
                # A vintage spanning adjacent download batches appears in both files.
                economic = [*KEY, "value", "unit", "frequency", "realtime_end_date"]
                distinct = data.drop_duplicates(economic)
                if distinct.duplicated(list(KEY)).any():
                    raise ValueError("macro_conflicting_source_versions")
                frames.append(distinct)
            sources[key] = metadata
    if not frames:
        raise ValueError("macro_no_observations")
    frame = validate_observations(pd.concat(frames, ignore_index=True))
    if start_date is not None:
        frame = frame.loc[frame.observation_date.ge(pd.Timestamp(start_date))]
    if end_date is not None:
        frame = frame.loc[
            frame.observation_date.lt(pd.Timestamp(end_date) + pd.Timedelta(days=1))
        ]
    coverage = []
    for name, group in frame.groupby("series_id", sort=True):
        observed = group.loc[group.value.notna()]
        coverage.append(
            {
                "series_id": name,
                "rows": len(group),
                "nonmissing_rows": len(observed),
                "observation_start": None
                if observed.empty
                else str(observed.observation_date.min().date()),
                "observation_end": None
                if observed.empty
                else str(observed.observation_date.max().date()),
                "first_available_at": str(group.timestamp.min()),
                "availability_basis": sorted(
                    group.availability_basis.unique().tolist()
                ),
            }
        )
    output = Path(output_roots[DATA_ID])
    write_dataset(
        output,
        [frame],
        spec=interim_catalog().get(DATA_ID),
        metadata={
            "source_evidence": sources,
            "producer_diagnostics": {
                "coverage": coverage,
                "skipped": skipped,
                "incomplete": incomplete,
            },
            "parameters": {"publication_date_convention": "next_local_midnight"},
            "inputs": {"raw_root": str(raw_root)},
        },
    )
    return MacroImportResult({DATA_ID: output}, tuple(skipped))
