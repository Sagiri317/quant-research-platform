"""Choice disclosure and historical financial snapshots to immutable interim facts."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import islice
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.validation import (
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.choice.fields import (
    FINANCIAL_BASE_COLUMNS,
    FINANCIAL_FIELDS,
    FIRST_DISCLOSURE_COLUMNS,
    field_roles,
)
from quant_research_platform.sources.choice.ids import (
    FIRST_DISCLOSURE_DATA_ID,
    SOURCE_ID,
)
from quant_research_platform.sources.choice.raw_exports import (
    ChoiceRequest,
    read_choice_response,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    filesystem_scandir,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

_DISCLOSURE_INDICATORS = {"STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE"}
_CONFLICT = object()


@dataclass(frozen=True, slots=True)
class ChoiceInterimImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def first_disclosure_spec() -> InterimDataSpec:
    return InterimDataSpec(
        data_id=FIRST_DISCLOSURE_DATA_ID,
        source_id=SOURCE_ID,
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
        column_roles=field_roles(),
        primary_key=FIRST_DISCLOSURE_COLUMNS,
        partition_columns=("year",),
    )


def interim_catalog() -> InterimDataCatalog:
    key = ("source_asset_id", "timestamp", "report_type", "version_identity")
    return InterimDataCatalog(
        (
            first_disclosure_spec(),
            *(
                InterimDataSpec(
                    data_id=data_id,
                    source_id=SOURCE_ID,
                    frequency=DataFrequency.DAILY,
                    grain=DataGrain.ASSET_REFERENCE_TIMESTAMP,
                    column_roles={
                        column: (
                            ColumnRole.TIMESTAMP
                            if column == "timestamp"
                            else ColumnRole.KEY
                            if column in key
                            else ColumnRole.CONTEXT
                        )
                        for column in (*FINANCIAL_BASE_COLUMNS, *fields.values())
                    },
                    primary_key=key,
                    partition_columns=("year",),
                )
                for data_id, fields in FINANCIAL_FIELDS.items()
            ),
        )
    )


def normalize_first_disclosure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept only the canonical adapter boundary once a real export is supplied."""

    if tuple(frame.columns) != FIRST_DISCLOSURE_COLUMNS:
        raise ValueError("choice_first_disclosure_header_not_canonical")
    result = frame.loc[:, FIRST_DISCLOSURE_COLUMNS].copy()
    result["asset_id"] = (
        result["asset_id"]
        .where(result["asset_id"].map(lambda value: isinstance(value, str)))
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )
    for column in ("report_period", "first_disclosure_date"):
        result[column] = strict_timestamp_series(
            result[column],
            error="invalid_disclosure_date",
            allow_null=True,
            errors="coerce",
        ).dt.normalize()
    valid = result.notna().all(axis=1) & result["first_disclosure_date"].ge(
        result["report_period"]
    )
    result = result.loc[valid].drop_duplicates()
    conflicts = result.duplicated(["asset_id", "report_period"], keep=False)
    result.attrs["source_row_counts"] = {
        "excluded_invalid_disclosure_rows": int((~valid).sum()),
        "excluded_conflicting_disclosure_rows": int(conflicts.sum()),
    }
    result = result.loc[~conflicts]
    return result.sort_values(
        list(FIRST_DISCLOSURE_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _raw_requests(raw_root: Path, *, data_ids):
    indicators = {
        field for data_id in data_ids for field in FINANCIAL_FIELDS.get(data_id, {})
    }
    root = require_unaliased_path(Path(raw_root), error="choice_raw_identity_changed")
    try:
        entries = filesystem_scandir(root / "css")
    except NotADirectoryError as error:
        raise ValueError("choice_raw_identity_changed") from error
    except FileNotFoundError:
        return
    with closing_iterator(entries):
        for entry in entries:
            if entry.name == "failed":
                continue
            directory = root / "css" / entry.name
            require_unaliased_path(directory, error="choice_raw_identity_changed")
            path = directory / "response.metadata.json"
            raw_metadata, manifest = read_qrp_json_file(path)
            try:
                payload = manifest["metadata"]["request"]
                if type(payload["arguments"]) is not list:
                    raise ValueError("choice_raw_request_metadata_invalid")
                request = ChoiceRequest(
                    endpoint=payload["endpoint"],
                    arguments=tuple(payload["arguments"]),
                    sdk_version=payload["sdk_version"],
                    cutoff=date.fromisoformat(payload["cutoff"]),
                )
                request.payload()
            except (KeyError, TypeError) as error:
                raise ValueError("choice_raw_request_metadata_invalid") from error
            if request.endpoint != "css":
                raise ValueError("choice_raw_identity_changed")
            options = dict(
                (
                    (name.lower(), value)
                    for part in request.arguments[2].split(",")
                    if part
                    for name, _, value in (part.partition("="),)
                )
            )
            fields = set(request.arguments[1].upper().split(","))
            disclosure = bool(_DISCLOSURE_INDICATORS.intersection(fields))
            financial_fields = bool(set(indicators).intersection(fields))
            snapshot = financial_fields and (options.get("type", "1") in {"1", "3"})
            if "reportdate" not in options or not (disclosure or snapshot):
                continue
            _yield_value = (
                directory,
                request,
                options,
                disclosure,
                snapshot,
                raw_metadata,
                manifest,
            )
            yield _yield_value


def _put(index, namespace, key, value):
    facts = index.setdefault(namespace, {})
    previous = facts.setdefault(key, value)
    if previous != value:
        # A later matching observation must never resolve a competing fact.
        facts[key] = _CONFLICT


def _get(index, namespace, key):
    value = index.get(namespace, {}).get(key)
    if value is _CONFLICT:
        return {"values": None, "conflicted": True} if namespace == "snapshot" else None
    return value


def _dates(value: object) -> tuple[date, ...]:
    if value is None:
        return ()
    values = value if type(value) is list else (value,)
    result = set()
    for item in values:
        if item is None:
            continue
        if type(item) is not str:
            raise ValueError("choice_disclosure_date_invalid")
        for token in item.split(","):
            token = token.strip()
            if not re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", token):
                raise ValueError("choice_disclosure_date_invalid")
            try:
                result.add(date(*map(int, re.split(r"[-/]", token))))
            except ValueError as error:
                raise ValueError("choice_disclosure_date_invalid") from error
    return tuple(sorted(result))


def _values(cells):
    result = []
    for value in cells:
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            value = None
        result.append(None if value is None else float(value))
    return result


def _raw_index(raw_root, data_ids, start, end, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    indicators = {
        field for data_id in data_ids for field in FINANCIAL_FIELDS.get(data_id, {})
    }
    index = {}
    hashes = {data_id: [] for data_id in data_ids}

    with closing_iterator(_raw_requests(raw_root, data_ids=data_ids)) as requests:
        for (
            root,
            request,
            options,
            disclosure,
            snapshot,
            raw_metadata,
            manifest,
        ) in requests:
            period = date.fromisoformat(options["reportdate"])
            if (start is not None and period < start) or (
                end is not None and period > end
            ):
                continue
            period = period.isoformat()
            snapshot_date = date.fromisoformat(options["enddate"])
            snapshot = snapshot and (end is None or snapshot_date <= end)
            if not disclosure and not snapshot:
                continue
            if snapshot and "reporttype" in options:
                raise ValueError("choice_financial_css_reporttype_unsupported")
            content_hashes = []
            response = read_choice_response(
                root, request, manifest=manifest, content_hashes=content_hashes
            )
            fields = set(response["Indicators"])
            targets = {
                data_id
                for data_id in data_ids
                if disclosure
                or snapshot
                and fields.intersection(FINANCIAL_FIELDS.get(data_id, {}))
            }
            source_hash = content_hashes[0]
            for data_id in targets:
                hashes[data_id].extend(
                    (
                        (
                            f"css/{root.name}/response.metadata.json",
                            hashlib.sha256(raw_metadata).hexdigest(),
                        ),
                        (f"css/{root.name}/response.csv", source_hash),
                    )
                )
            if not response["Data"]:
                continue
            if len(response["Dates"]) != 1:
                raise ValueError("choice_financial_snapshot_dates_invalid")
            for code, cells in response["Data"].items():
                if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code):
                    diagnostics["excluded_invalid_asset_rows"] = (
                        diagnostics.get("excluded_invalid_asset_rows", 0) + 1
                    )
                    continue
                values = dict(zip(response["Indicators"], cells, strict=True))
                pair = (code, period)
                coordinate = (
                    *pair,
                    snapshot_date.isoformat(),
                    options.get("type", "1"),
                )
                if disclosure:
                    try:
                        parsed_dates = {
                            field: _dates(values[field])
                            for field in _DISCLOSURE_INDICATORS.intersection(values)
                        }
                    except ValueError:
                        index.setdefault("unresolved_report", {})[pair] = True
                        diagnostics["excluded_invalid_disclosure_rows"] = (
                            diagnostics.get("excluded_invalid_disclosure_rows", 0) + 1
                        )
                        continue
                    for field in _DISCLOSURE_INDICATORS.intersection(values):
                        _put(
                            index,
                            "disclosure_field",
                            (*coordinate, field),
                            {
                                "values": [
                                    day.isoformat() for day in parsed_dates[field]
                                ],
                            },
                        )
                    _put(
                        index,
                        f"disclosure_coordinates:{code}:{period}",
                        coordinate,
                        coordinate,
                    )
                    if "STMTACTDATE" in values:
                        first = [
                            day
                            for day in parsed_dates["STMTACTDATE"]
                            if day <= request.cutoff
                        ]
                        if first:
                            _put(
                                index,
                                "disclosure",
                                pair,
                                [*pair, first[0].isoformat()],
                            )
                    if "ITEMVALUECHANGEDATE" in values:
                        for event in parsed_dates["ITEMVALUECHANGEDATE"]:
                            if event <= request.cutoff:
                                _put(
                                    index,
                                    f"events:{code}:{period}",
                                    (event.isoformat(),),
                                    event.isoformat(),
                                )
                if snapshot:
                    for field in indicators.intersection(values):
                        value = values[field]
                        if value is not None and (
                            type(value) not in (int, float) or not math.isfinite(value)
                        ):
                            diagnostics[f"invalid_{field}_cells"] = (
                                diagnostics.get(f"invalid_{field}_cells", 0) + 1
                            )
                        _put(
                            index,
                            "snapshot",
                            (*coordinate, field),
                            {
                                "values": _values((values[field],))[0],
                            },
                        )
                for data_id in targets:
                    _put(index, f"report:{data_id}", pair, pair)

    for namespace in ("snapshot", "disclosure", "disclosure_field"):
        conflicts = sum(
            value is _CONFLICT for value in index.get(namespace, {}).values()
        )
        if conflicts:
            diagnostics[f"conflicting_{namespace}_coordinates"] = conflicts
    identities = {}
    for data_id, members in hashes.items():
        digest = hashlib.sha256()
        for name, member_hash in sorted(members):
            for value in (name, member_hash):
                encoded = value.encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
        identities[data_id] = digest.hexdigest() if members else None
    return index, identities


def _snapshot(index, pair, day, branch, indicators):
    coordinate = (*pair, day.isoformat(), branch)
    return [_get(index, "snapshot", (*coordinate, field)) for field in indicators]


def _report_versions(index, pair, first, end, indicators):
    initial = _snapshot(index, pair, first, "1", indicators)
    adjusted = _snapshot(index, pair, first, "3", indicators)
    current = [None if field is None else field["values"] for field in initial]
    for position, (original, adjustment) in enumerate(
        zip(initial, adjusted, strict=True)
    ):
        if original is None or adjustment is None:
            continue
        if original.get("conflicted", False) or adjustment.get("conflicted", False):
            current[position] = None
        elif adjustment["values"] is not None:
            value = adjustment["values"]
            current[position] = value if current[position] in (None, value) else None
    yield first, False, current.copy()
    branches = {"1": initial, "3": adjusted}
    events = index.get(f"events:{pair[0]}:{pair[1]}", {})
    for (event,) in sorted(events):
        event = date.fromisoformat(event)
        if event <= first or (end is not None and event > end):
            continue
        changed = {}
        for branch in ("1", "3"):
            after = _snapshot(index, pair, event, branch, indicators)
            before = _snapshot(
                index, pair, event - timedelta(days=1), branch, indicators
            )
            previous = branches[branch]
            for position, (prior_field, after_field, previous_field) in enumerate(
                zip(before, after, previous, strict=True)
            ):
                if prior_field is None:
                    prior_field = previous_field
                if after_field is None or prior_field is None:
                    changed.setdefault(position, set()).add(None)
                    continue
                prior, value = prior_field["values"], after_field["values"]
                if after_field.get("conflicted", False) or prior_field.get(
                    "conflicted", False
                ):
                    changed.setdefault(position, set()).add(None)
                elif previous_field is not None and (
                    previous_field.get("conflicted", False)
                    or prior != previous_field["values"]
                ):
                    changed.setdefault(position, set()).add(None)
                elif value != prior:
                    changed.setdefault(position, set()).add(value)
            branches[branch] = after
        revised = False
        for position, values in changed.items():
            value = next(iter(values)) if len(values) == 1 else None
            revised |= value != current[position]
            current[position] = value
        if revised:
            yield event, True, current.copy()


def _rows(index, data_id, end, diagnostics=None) -> Iterator[tuple[object, ...]]:
    diagnostics = {} if diagnostics is None else diagnostics
    for pair in sorted(index.get(f"report:{data_id}", {})):
        if pair in index.get("unresolved_report", {}):
            diagnostics["excluded_unresolved_report_rows"] = (
                diagnostics.get("excluded_unresolved_report_rows", 0) + 1
            )
            continue
        disclosure = _get(index, "disclosure", pair)
        first = date.fromisoformat(disclosure[2]) if disclosure is not None else None
        if first is not None and end is not None and first > end:
            continue
        coordinates = index.get(f"disclosure_coordinates:{pair[0]}:{pair[1]}", {})
        complete = all(
            _get(index, "disclosure_field", (*coordinate, field)) is not None
            for coordinate in coordinates
            for field in _DISCLOSURE_INDICATORS
        )
        if first is None or not complete:
            diagnostics["excluded_unresolved_disclosure_rows"] = (
                diagnostics.get("excluded_unresolved_disclosure_rows", 0) + 1
            )
            continue
        earliest = min(index.get(f"events:{pair[0]}:{pair[1]}", {}), default=None)
        if earliest is not None and date.fromisoformat(earliest[0]) < first:
            diagnostics["excluded_invalid_chronology_rows"] = (
                diagnostics.get("excluded_invalid_chronology_rows", 0) + 1
            )
            continue
        if data_id == FIRST_DISCLOSURE_DATA_ID:
            _yield_value = (pair[0][:6], pair[1], first.isoformat())
            yield _yield_value
            continue
        for public_date, revised, values in _report_versions(
            index, pair, first, end, tuple(FINANCIAL_FIELDS[data_id])
        ):
            _yield_value = (
                pair[0][:6],
                pair[1],
                pair[1],
                "A",
                revised,
                public_date.isoformat() if revised else None,
                f"correction@{public_date}" if revised else "original",
                *values,
            )
            yield _yield_value


def _frames(rows, spec):
    columns = spec.required_columns
    count = 8192
    frame = pending = None
    emitted = False
    while True:
        pending = list(islice(rows, count))
        if not pending and emitted:
            return
        frame = pd.DataFrame.from_records(pending, columns=columns)
        pending = None
        if spec.data_id == FIRST_DISCLOSURE_DATA_ID:
            frame = normalize_first_disclosure_frame(frame)
        else:
            frame["source_asset_id"] = frame["source_asset_id"].astype("string")
            for name in (
                "timestamp",
                "report_period",
                "correction_disclosure_date",
            ):
                frame[name] = strict_timestamp_series(
                    frame[name],
                    error="choice_financial_date_invalid",
                    allow_null=name == "correction_disclosure_date",
                )
            frame["report_type"] = frame["report_type"].astype("string")
            frame["version_identity"] = frame["version_identity"].astype("string")
            frame["is_corrected"] = frame["is_corrected"].astype("bool")
            for name in FINANCIAL_FIELDS[spec.data_id].values():
                frame[name] = frame[name].astype("float64")
        yield frame
        emitted = True
        frame = None


def materialize_interim_data(
    *,
    raw_root: Path,
    output_roots: Mapping[str, Path],
    data_ids: tuple[str, ...] = (*FINANCIAL_FIELDS, FIRST_DISCLOSURE_DATA_ID),
    start_date: str | None = None,
    end_date: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ChoiceInterimImportResult:
    if (
        type(data_ids) is not tuple
        or not data_ids
        or len(set(data_ids)) != len(data_ids)
    ):
        raise ValueError("choice_interim_data_ids_invalid")
    catalog = interim_catalog()
    specs = tuple(catalog.get(data_id) for data_id in data_ids)
    root = Path(raw_root)
    start = date.fromisoformat(start_date) if start_date is not None else None
    end = date.fromisoformat(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("choice_interim_window_invalid")
    written, skipped = {}, []
    source_counts = {}
    index, identities = _raw_index(root, data_ids, start, end, source_counts)
    for spec in specs:
        if identities[spec.data_id] is None:
            skipped.append(spec.data_id)
            continue
        counts = dict(source_counts)
        rows = _rows(index, spec.data_id, end, counts)
        frames = _frames(rows, spec)
        target = Path(output_roots[spec.data_id])
        with closing_iterator(rows), closing_iterator(frames):
            try:
                write_dataset(
                    target,
                    frames,
                    spec=spec,
                    metadata={
                        **(metadata or {}),
                        "producer_diagnostics": {"source_row_counts": counts},
                        "source_evidence": {
                            "source_id": str(SOURCE_ID),
                            "content_hash": identities[spec.data_id],
                        },
                    },
                )
            except ValueError as error:
                if str(error) != "empty_dataset_schema_missing":
                    raise
                skipped.append(spec.data_id)
                continue
        written[spec.data_id] = target
    return ChoiceInterimImportResult(written, tuple(skipped))
