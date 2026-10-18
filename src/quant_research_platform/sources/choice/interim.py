"""Choice disclosure and historical financial snapshots to immutable interim facts."""

from __future__ import annotations

import hashlib
import json
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
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    filesystem_scandir,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

_DISCLOSURE_INDICATORS = {"STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE"}


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
            try:
                yield _yield_value
            except GeneratorExit:
                return


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()


def _identity_payload_row(index, namespace, key, value):
    key, payload = _encode(key), _encode(value)
    old = index.identity_payload_value(namespace=namespace, payload=key)
    if old is not None:
        previous = json.loads(old)
        equal = (
            previous["values"] == value["values"]
            if namespace in {"snapshot", "disclosure_field"}
            else previous == value
        )
        if not equal:
            # Preserve the coordinate but never choose one competing fact.
            index.add_identity_payload_rows(
                rows=(("conflict:" + namespace, key, b"true"),)
            )
    return namespace, key, payload


def _get(index, namespace, key):
    payload = index.identity_payload_value(namespace=namespace, payload=_encode(key))
    if payload is None:
        return None
    value = json.loads(payload)
    if (
        index.identity_payload_value(
            namespace="conflict:" + namespace, payload=_encode(key)
        )
        is not None
    ):
        return (
            {**value, "values": None, "conflicted": True}
            if namespace == "snapshot"
            else None
        )
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
    index = ExactPrimaryKeyIndex(primary_key=("source_coordinate",))
    hashes = {data_id: [] for data_id in data_ids}

    def rows():
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
                    pair = [code, period]
                    coordinate = [
                        *pair,
                        snapshot_date.isoformat(),
                        options.get("type", "1"),
                    ]
                    if disclosure:
                        try:
                            parsed_dates = {
                                field: _dates(values[field])
                                for field in _DISCLOSURE_INDICATORS.intersection(values)
                            }
                        except ValueError:
                            index.add_identity_payload_rows(
                                rows=(("unresolved_report", _encode(pair), b"true"),)
                            )
                            diagnostics["excluded_invalid_disclosure_rows"] = (
                                diagnostics.get("excluded_invalid_disclosure_rows", 0)
                                + 1
                            )
                            continue
                        for field in _DISCLOSURE_INDICATORS.intersection(values):
                            yield _identity_payload_row(
                                index,
                                "disclosure_field",
                                [*coordinate, field],
                                {
                                    "values": [
                                        day.isoformat() for day in parsed_dates[field]
                                    ],
                                    "request_identity": root.name,
                                },
                            )
                        yield _identity_payload_row(
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
                                yield _identity_payload_row(
                                    index,
                                    "disclosure",
                                    pair,
                                    [*pair, first[0].isoformat()],
                                )
                        if "ITEMVALUECHANGEDATE" in values:
                            for event in parsed_dates["ITEMVALUECHANGEDATE"]:
                                if event <= request.cutoff:
                                    yield _identity_payload_row(
                                        index,
                                        f"events:{code}:{period}",
                                        event.isoformat(),
                                        event.isoformat(),
                                    )
                    if snapshot:
                        for field in indicators.intersection(values):
                            value = values[field]
                            if value is not None and (
                                type(value) not in (int, float)
                                or not math.isfinite(value)
                            ):
                                diagnostics[f"invalid_{field}_cells"] = (
                                    diagnostics.get(f"invalid_{field}_cells", 0) + 1
                                )
                            yield _identity_payload_row(
                                index,
                                "snapshot",
                                [*coordinate, field],
                                {
                                    "values": _values((values[field],))[0],
                                    "request_identity": root.name,
                                },
                            )
                    for data_id in targets:
                        yield _identity_payload_row(
                            index, f"report:{data_id}", pair, pair
                        )

    with closing_iterator(rows()) as input_rows:
        for row in input_rows:
            index.add_identity_payload_rows(rows=(row,))
    for namespace in ("snapshot", "disclosure", "disclosure_field"):
        conflicts = index.identity_payload_count(namespace="conflict:" + namespace)
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
    coordinate = [*pair, day.isoformat(), branch]
    fields = [_get(index, "snapshot", [*coordinate, field]) for field in indicators]
    if all(value is None for value in fields):
        return None
    return {
        "values": [None if value is None else value["values"] for value in fields],
        "conflicted": [
            value is not None and value.get("conflicted", False) for value in fields
        ],
        "request_identities": sorted(
            {value["request_identity"] for value in fields if value is not None}
        ),
    }


def _report_versions(index, pair, first, end, indicators):
    initial = _snapshot(index, pair, first, "1", indicators)
    current = [None] * len(indicators) if initial is None else initial["values"].copy()
    adjusted = _snapshot(index, pair, first, "3", indicators)
    if initial is not None and adjusted is not None:
        for position, value in enumerate(adjusted["values"]):
            if initial["conflicted"][position] or adjusted["conflicted"][position]:
                current[position] = None
            elif value is not None:
                current[position] = (
                    value if current[position] in (None, value) else None
                )
    yield first, False, current.copy()
    branches = {"1": initial, "3": adjusted}
    events = index.identity_payload_values(namespace=f"events:{pair[0]}:{pair[1]}")
    with closing_iterator(events):
        for payload in events:
            event = date.fromisoformat(json.loads(payload))
            if event <= first or (end is not None and event > end):
                continue
            changed = {}
            for branch in ("1", "3"):
                after = _snapshot(index, pair, event, branch, indicators)
                before = _snapshot(
                    index, pair, event - timedelta(days=1), branch, indicators
                )
                previous = branches[branch]
                if before is None:
                    before = previous
                for position in range(len(indicators)):
                    if after is None or before is None:
                        changed.setdefault(position, set()).add(None)
                        continue
                    prior, value = before["values"][position], after["values"][position]
                    if after["conflicted"][position] or before["conflicted"][position]:
                        changed.setdefault(position, set()).add(None)
                    elif previous is not None and prior != previous["values"][position]:
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
    reports = index.identity_payload_values(namespace=f"report:{data_id}")
    with closing_iterator(reports):
        for payload in reports:
            row = json.loads(payload)
            pair = row[:2]
            if (
                index.identity_payload_value(
                    namespace="unresolved_report", payload=_encode(pair)
                )
                is not None
            ):
                diagnostics["excluded_unresolved_report_rows"] = (
                    diagnostics.get("excluded_unresolved_report_rows", 0) + 1
                )
                continue
            disclosure = _get(index, "disclosure", pair)
            first = (
                date.fromisoformat(disclosure[2]) if disclosure is not None else None
            )
            if first is not None and end is not None and first > end:
                continue
            coordinates = index.identity_payload_values(
                namespace=f"disclosure_coordinates:{pair[0]}:{pair[1]}"
            )
            complete = True
            with closing_iterator(coordinates):
                for payload in coordinates:
                    coordinate = json.loads(payload)
                    complete &= all(
                        _get(index, "disclosure_field", [*coordinate, field])
                        is not None
                        for field in _DISCLOSURE_INDICATORS
                    )
            if first is None or not complete:
                diagnostics["excluded_unresolved_disclosure_rows"] = (
                    diagnostics.get("excluded_unresolved_disclosure_rows", 0) + 1
                )
                continue
            with closing_iterator(
                index.identity_payload_values(namespace=f"events:{pair[0]}:{pair[1]}")
            ) as events:
                earliest = next(events, None)
            if (
                earliest is not None
                and date.fromisoformat(json.loads(earliest)) < first
            ):
                diagnostics["excluded_invalid_chronology_rows"] = (
                    diagnostics.get("excluded_invalid_chronology_rows", 0) + 1
                )
                continue
            if data_id == FIRST_DISCLOSURE_DATA_ID:
                _yield_value = (pair[0][:6], pair[1], first.isoformat())
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
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
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


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
