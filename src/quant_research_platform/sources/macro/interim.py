"""Normalize macro vintages with explicit historical publication assumptions."""

import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.sources.macro.catalog import SERIES
from quant_research_platform.sources.macro.publication import (
    estimate_initial_archive_times,
    estimated_release_at,
)
from quant_research_platform.sources.tabular import read_csv as _read_selected_csv
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


def _frame(
    name,
    values,
    dates,
    *,
    record,
    timezone,
    vintage=None,
    end_dates=None,
    start_date=None,
    end_date=None,
):
    series = SERIES[name]
    output = pd.DataFrame({"observation_date": dates, "value": values})
    if vintage is not None:
        output["vintage_date"] = vintage
        output["realtime_end_date"] = end_dates
    output = _select_observation_window(
        output, start_date=start_date, end_date=end_date
    )
    output["series_id"] = name
    output["unit"] = series.unit
    output["frequency"] = series.frequency
    output["source_id"] = series.source
    output["source_series"] = series.code
    output["source_timezone"] = timezone
    output["published_at"] = pd.Series(
        pd.NaT, index=output.index, dtype="datetime64[ns, UTC]"
    )
    output["retrieved_at"] = (
        pd.NaT if output.empty else pd.Timestamp(record["retrieved_at"])
    )
    output["source_url"] = record["url"]
    if vintage is None:
        output["timestamp"] = estimated_release_at(name, output.observation_date)
        output["vintage_date"] = pd.NaT
        output["realtime_end_date"] = pd.Series(pd.NA, index=output.index, dtype="str")
        output["availability_basis"] = "inferred_snapshot_release"
    else:
        output["vintage_date"] = pd.to_datetime(output.vintage_date)
        output["timestamp"] = date_end_available(output.vintage_date, timezone)
        output["availability_basis"] = "archive_date_end"
    return output


def normalize_alfred(payload, name, record, *, start_date=None, end_date=None):
    """Retain each new or revised value at its archive vintage's conservative cutoff."""
    series = SERIES[name]
    required = {
        "period_start_date",
        series.code,
        "realtime_start_date",
        "realtime_end_date",
    }
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        data = _read_selected_csv(
            archive.open("obs._by_real-time_period.csv"),
            required,
            error="alfred_observation_columns_invalid",
            na_values=["."],
            dtype={"realtime_end_date": "str"},
        )
    return _frame(
        name,
        data[series.code],
        data.period_start_date,
        record=record,
        timezone="America/Chicago",
        vintage=data.realtime_start_date,
        end_dates=data.realtime_end_date,
        start_date=start_date,
        end_date=end_date,
    )


def _normalize_response(
    payload, names, record, *, incomplete=False, start_date=None, end_date=None
):
    source = SERIES[names[0]].source
    if source == "alfred":
        return [
            normalize_alfred(
                payload, names[0], record, start_date=start_date, end_date=end_date
            )
        ]
    frame = partial(_frame, start_date=start_date, end_date=end_date)
    if source == "cboe":
        name = names[0]
        value_column = "CLOSE" if SERIES[name].code == "VIX9D" else SERIES[name].code
        data = _read_selected_csv(
            io.BytesIO(payload), ("DATE", value_column), na_values=["."]
        )
        return [
            frame(
                name,
                data[value_column],
                pd.to_datetime(data.DATE, format="%m/%d/%Y"),
                record=record,
                timezone="America/New_York",
            )
        ]
    if source == "chinabond":
        tenors = {name: name.rsplit("_", 1)[1].replace("y", "年") for name in names}
        data = _read_selected_csv(
            io.BytesIO(payload),
            {"曲线名称", "日期", *tenors.values()},
            error="chinabond_structured_columns_invalid",
            na_values=["--"],
        )
        frames = []
        for name in names:
            curve = (
                "中债中短期票据收益率曲线(AAA)"
                if name == "cn_credit_aaa_3y"
                else "中债国债收益率曲线"
            )
            selected = data.loc[data["曲线名称"].eq(curve)]
            frames.append(
                frame(
                    name,
                    selected[tenors[name]],
                    selected["日期"],
                    record=record,
                    timezone="Asia/Shanghai",
                )
            )
        return frames
    if source == "sf_fed":
        data = _read_selected_csv(io.BytesIO(payload), ("Date", SERIES[names[0]].code))
        # The publisher's chart declares basis points; current PCA weights are ex post.
        return [
            frame(
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
            frame(
                names[0], data.OBS_VALUE, dates, record=record, timezone="Europe/Zurich"
            )
        ]
    document = json.loads(payload)
    if source == "mofcom":
        data = pd.DataFrame(document)
        if data.empty:
            return []
        return [
            frame(
                names[0],
                data[SERIES[names[0]].code],
                pd.to_datetime(data.date.astype(str), format="%Y%m"),
                record=record,
                timezone="Asia/Shanghai",
            )
        ]
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
            frame(
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
    elif source == "cfets_shibor":
        data = pd.DataFrame(document["records"])
        date_column, timezone = "showDateCN", "Asia/Shanghai"
    else:
        data = pd.DataFrame(document["result"]["data"])
        date_column, timezone = "REPORT_DATE", "Asia/Shanghai"
    if data.empty:
        return []
    return [
        frame(
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
        (
            "release_timestamp",
            "release_date_end",
            "archive_date_end",
            "inferred_snapshot_release",
        )
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


def _select_observation_window(frame, *, start_date, end_date):
    """Select observation dependencies before validating their values and clocks."""
    frame = frame.assign(observation_date=pd.to_datetime(frame.observation_date))
    if frame.observation_date.isna().any():
        raise ValueError("macro_observation_key_missing")
    if start_date is not None:
        frame = frame.loc[frame.observation_date.ge(pd.Timestamp(start_date))]
    if end_date is not None:
        frame = frame.loc[
            frame.observation_date.lt(pd.Timestamp(end_date) + pd.Timedelta(days=1))
        ]
    return frame


def read_release_csv(path, *, start_date=None, end_date=None):
    """Import timestamped original releases or revisions from an explicit local file.

    Each file selects a publication timestamp, release date, or assumed clock.
    Source identity and units stay explicit, including for private vendor exports.
    """
    payload = Path(path).read_bytes()
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
    clock_columns = {"published_at", "release_date", "estimated_available_at"}
    data = _read_selected_csv(
        io.BytesIO(payload),
        required,
        optional_columns=clock_columns,
        error="macro_release_columns_invalid",
    )
    clocks = set(data) & clock_columns
    if len(clocks) != 1:
        raise ValueError("macro_release_columns_invalid")
    data = _select_observation_window(data, start_date=start_date, end_date=end_date)
    clock = next(iter(clocks))
    if clock != "release_date":
        # Explicit offsets prevent machine-local time from becoming economic time.
        if not data[clock].str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).all():
            raise ValueError("macro_release_timezone_required")
        data["timestamp"] = pd.to_datetime(data[clock], utc=True)
        actual = clock == "published_at"
        data["published_at"] = data.timestamp if actual else pd.NaT
        data["availability_basis"] = (
            "release_timestamp" if actual else "inferred_snapshot_release"
        )
        data["vintage_date"] = (
            data.timestamp.dt.tz_localize(None).dt.normalize() if actual else pd.NaT
        )
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
    result = validate_observations(data)
    result.attrs["source_evidence"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    return result


def resolve_source_conflicts(frame):
    """Prefer official current API/release values over secondary snapshots.

    Keep revisions within the selected source. Same-key upstream corrections use
    the latest retrieval; source and retrieval facts remain on the retained row.
    """
    priority = frame.source_id.map(
        {
            "eastmoney": 0,
            "hkma_historical_page": 1,
            "hkma": 2,
            "pbc_government_reprint": 2,
        }
    ).fillna(2)
    frame = frame.loc[
        priority.eq(
            priority.groupby([frame.series_id, frame.observation_date]).transform("max")
        )
    ]
    return (
        frame.sort_values("retrieved_at", kind="stable")
        .drop_duplicates(list(KEY), keep="last")
        .sort_values(["timestamp", "series_id", "observation_date"], kind="stable")
        .reset_index(drop=True)
    )


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
        frame = read_release_csv(release_csv, start_date=start_date, end_date=end_date)
        sources["release_csv"] = frame.attrs.pop("source_evidence")
    else:
        manifest = read_qrp_json_file(raw_root / "download.json")[1]
        for key, metadata in manifest["sources"].items():
            records = metadata["responses"]
            if metadata["status"] != "downloaded" and metadata["source"] in {
                "hkma",
                "cfets_shibor",
                "chinabond",
            }:
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
                payload = member.read_bytes()
                if (
                    len(payload) != record["bytes"]
                    or hashlib.sha256(payload).hexdigest() != record["sha256"]
                ):
                    raise ValueError("macro_source_integrity_mismatch")
                chunks.extend(
                    _normalize_response(
                        payload,
                        metadata["series_ids"],
                        record,
                        incomplete=metadata["status"] != "downloaded",
                        start_date=start_date,
                        end_date=end_date,
                    )
                )
            if chunks:
                data = pd.concat(chunks, ignore_index=True)
                frames.append(data)
            sources[key] = metadata
        if not frames:
            raise ValueError("macro_no_observations")
        frame = validate_observations(pd.concat(frames, ignore_index=True))
    frame = estimate_initial_archive_times(resolve_source_conflicts(frame))
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
            "parameters": {
                "publication_date_convention": "next_local_midnight",
                "snapshot_availability": "buffered_historical_release_estimate",
                "initial_archive_enrollment": "estimated_original_release",
            },
            "inputs": {"raw_root": str(raw_root)},
        },
    )
    return MacroImportResult({DATA_ID: output}, tuple(skipped))
