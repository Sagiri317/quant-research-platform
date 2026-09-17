"""Choice disclosure and historical financial snapshots to immutable interim facts."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import groupby, islice
from pathlib import Path

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
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.contracts.validation import (
    strict_string_series,
    strict_timestamp_series,
    validate_financial_disclosure_chronology,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
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
from quant_research_platform.storage.datasets.schema import COMMON_MANIFEST_METADATA_KEY
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    dataset_file_identity,
    require_unaliased_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_manifest

_INDICATORS = tuple(field for fields in FINANCIAL_FIELDS.values() for field in fields)
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
    result["asset_id"] = strict_string_series(
        result["asset_id"], error="invalid_choice_asset_id"
    )
    for column in ("report_period", "first_disclosure_date"):
        result[column] = strict_timestamp_series(
            result[column], error="invalid_choice_disclosure_date"
        ).dt.normalize()
    validate_financial_disclosure_chronology(
        result["report_period"], result["first_disclosure_date"]
    )
    if result.duplicated(["asset_id", "report_period"], keep=False).any():
        raise ValueError("duplicate_choice_first_disclosure_key")
    return result.sort_values(
        list(FIRST_DISCLOSURE_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _raw_requests(raw_root: Path, *, financial: bool):
    root = require_unaliased_path(Path(raw_root), error="choice_raw_identity_changed")
    try:
        entries = os.scandir(root / "css")
    except FileNotFoundError:
        return
    except NotADirectoryError as error:
        raise ValueError("choice_raw_identity_changed") from error
    with closing_iterator(entries):
        for entry in entries:
            directory = root / "css" / entry.name
            if not re.fullmatch(r"[0-9a-f]{64}", directory.name):
                continue
            require_unaliased_path(directory, error="choice_raw_identity_changed")
            manifest = read_manifest(directory / "response.metadata.json")
            try:
                payload = manifest["metadata"]["request"]
                if type(payload["arguments"]) is not list:
                    raise ValueError("choice_raw_request_metadata_invalid")
                request = ChoiceRequest(
                    endpoint=payload["endpoint"],
                    arguments=tuple(payload["arguments"]),
                    sdk_version=payload["sdk_version"],
                    cutoff=date.fromisoformat(payload["cutoff"]),
                    max_response_bytes=manifest["byte_count"],
                )
                request_identity = request.identity()
            except (KeyError, TypeError) as error:
                raise ValueError("choice_raw_request_metadata_invalid") from error
            if request_identity != directory.name or request.endpoint != "css":
                raise ValueError("choice_raw_identity_changed")
            options = dict(
                (name.lower(), value)
                for part in request.arguments[2].split(",")
                if part
                for name, _, value in (part.partition("="),)
            )
            fields = set(request.arguments[1].upper().split(","))
            disclosure = bool(_DISCLOSURE_INDICATORS.intersection(fields))
            snapshot = (
                financial
                and bool(set(_INDICATORS).intersection(fields))
                and options.get("type", "1") in {"1", "3"}
            )
            if "reportdate" not in options or not (disclosure or snapshot):
                continue
            yield directory, request, options, disclosure


@contextmanager
def _source_snapshot(raw_root: Path, data_id: str):
    interim_catalog().get(data_id)
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    members = {}
    with (
        policy.lease("source.choice.identity.members", 1) as retained,
        closing_iterator(
            _raw_requests(raw_root, financial=data_id != FIRST_DISCLOSURE_DATA_ID)
        ) as requests,
    ):
        for root, _request, _options, _disclosure in requests:
            retained.grow(2048)
            for name in ("response.csv", "response.metadata.json"):
                path = root / name
                try:
                    member_identity = dataset_file_identity(path)
                except ValueError as error:
                    raise ValueError("choice_raw_identity_changed") from error
                members[path] = member_identity
        identity = (
            content_hash_for_files(
                (
                    (f"css/{path.parent.name}/{path.name}", path, expected)
                    for path, expected in members.items()
                ),
                error="choice_raw_identity_changed",
                resource_policy=policy,
            )
            if members
            else None
        )
        yield identity, members


def source_content_identity(raw_root: Path, data_id: str) -> str | None:
    with _source_snapshot(raw_root, data_id) as (identity, _members):
        return identity


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
            error = ValueError(f"choice_{namespace}_conflict")
            if namespace in {"snapshot", "disclosure_field"}:
                identities = sorted(
                    (previous["request_identity"], value["request_identity"])
                )
                error.add_note(f"request_identities={identities}")
            raise error
    return namespace, key, payload


def _get(index, namespace, key):
    payload = index.identity_payload_value(namespace=namespace, payload=_encode(key))
    return None if payload is None else json.loads(payload)


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
            raise ValueError("choice_financial_value_invalid")
        result.append(None if value is None else float(value))
    return result


@contextmanager
def _raw_index(raw_root, financial, policy, *, members):
    index = ExactPrimaryKeyIndex(
        primary_key=("source_coordinate",),
    )
    with policy.lease("source.choice.index", index.retained_memory_bytes), index:

        def rows():
            with closing_iterator(
                _raw_requests(raw_root, financial=financial)
            ) as requests:
                for root, request, options, disclosure in requests:
                    period = options["reportdate"]
                    try:
                        expected_metadata = members[root / "response.metadata.json"]
                    except KeyError:
                        raise ValueError("choice_raw_identity_changed") from None
                    with read_choice_response(
                        root,
                        request,
                        resource_policy=policy,
                        expected_metadata_identity=expected_metadata,
                    ) as response:
                        if not response["Data"]:
                            continue
                        if len(response["Dates"]) != 1:
                            raise ValueError("choice_financial_snapshot_dates_invalid")
                        fields = response["Indicators"]
                        for code, cells in response["Data"].items():
                            if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code):
                                raise ValueError("invalid_choice_asset_id")
                            values = dict(zip(fields, cells, strict=True))
                            pair = [code, period]
                            coordinate = [
                                *pair,
                                options["enddate"],
                                options.get("type", "1"),
                            ]
                            if disclosure:
                                for field in _DISCLOSURE_INDICATORS.intersection(
                                    values
                                ):
                                    yield _identity_payload_row(
                                        index,
                                        "disclosure_field",
                                        [*coordinate, field],
                                        {
                                            "values": [
                                                day.isoformat()
                                                for day in _dates(values[field])
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
                                yield _identity_payload_row(index, "report", pair, pair)
                                if "STMTACTDATE" in values:
                                    first = [
                                        day
                                        for day in _dates(values["STMTACTDATE"])
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
                                    for event in _dates(values["ITEMVALUECHANGEDATE"]):
                                        if event <= request.cutoff:
                                            yield _identity_payload_row(
                                                index,
                                                f"events:{code}:{period}",
                                                event.isoformat(),
                                                event.isoformat(),
                                            )
                            if (
                                financial
                                and set(_INDICATORS).intersection(values)
                                and options.get("type", "1") in {"1", "3"}
                            ):
                                for field in set(_INDICATORS).intersection(values):
                                    yield _identity_payload_row(
                                        index,
                                        "snapshot",
                                        [*coordinate, field],
                                        {
                                            "values": _values((values[field],))[0],
                                            "request_identity": root.name,
                                        },
                                    )
                                yield _identity_payload_row(index, "report", pair, pair)

        with closing_iterator(rows()) as input_rows:
            index.add_identity_payload_rows(rows=input_rows)
        yield index


def _snapshot(index, pair, day, branch):
    coordinate = [*pair, day.isoformat(), branch]
    fields = [_get(index, "snapshot", [*coordinate, field]) for field in _INDICATORS]
    if all(value is None for value in fields):
        return None
    missing = [
        field for field, value in zip(_INDICATORS, fields, strict=True) if value is None
    ]
    if missing:
        error = PortfolioTargetBlocked("choice_financial_snapshot_field_missing")
        error.add_note(
            f"fields={missing}; report_period={pair[1]}; EndDate={day}; type={branch}"
        )
        raise error
    return {
        "values": [value["values"] for value in fields],
        "request_identities": sorted({value["request_identity"] for value in fields}),
    }


def _report_versions(index, pair, first):
    initial = _snapshot(index, pair, first, "1")
    if initial is None:
        raise PortfolioTargetBlocked("choice_first_disclosure_snapshot_missing")
    current = initial["values"].copy()
    initial_adjusted = _snapshot(index, pair, first, "3")
    if initial_adjusted is not None:
        for position, value in enumerate(initial_adjusted["values"]):
            if value is None:
                continue
            if current[position] is not None and current[position] != value:
                error = ValueError("choice_financial_version_values_conflict")
                error.add_note(f"public_date={first}; field={_INDICATORS[position]}")
                identities = sorted(
                    {
                        *initial["request_identities"],
                        *initial_adjusted["request_identities"],
                    }
                )
                error.add_note(f"request_identities={identities}")
                raise error
            current[position] = value
    yield first, False, current.copy()
    branches = {"1": initial, "3": initial_adjusted}
    events = index.identity_payload_values(namespace=f"events:{pair[0]}:{pair[1]}")
    with closing_iterator(events):
        for payload in events:
            event = date.fromisoformat(json.loads(payload))
            if event == first:
                continue
            changed: dict[int, set[float | None]] = {}
            requests = []
            for branch in ("1", "3"):
                after = _snapshot(index, pair, event, branch)
                if after is None:
                    raise PortfolioTargetBlocked("choice_revision_snapshot_missing")
                before = _snapshot(index, pair, event - timedelta(days=1), branch)
                if before is None:
                    before = branches[branch]
                if before is None:
                    error = PortfolioTargetBlocked(
                        "choice_revision_predecessor_missing"
                    )
                    error.add_note(
                        f"report_period={pair[1]}; public_date={event}; type={branch}"
                    )
                    raise error
                requests.extend(
                    (*before["request_identities"], *after["request_identities"])
                )
                for position, value in enumerate(after["values"]):
                    prior = before["values"][position]
                    if value != prior:
                        changed.setdefault(position, set()).add(value)
                branches[branch] = after
            revised = False
            for position, values in changed.items():
                if len(values) != 1:
                    error = ValueError("choice_financial_version_values_conflict")
                    error.add_note(
                        f"report_period={pair[1]}; public_date={event}; "
                        f"field={_INDICATORS[position]}"
                    )
                    error.add_note(f"request_identities={sorted(set(requests))}")
                    raise error
                value = next(iter(values))
                revised |= value != current[position]
                current[position] = value
            if revised:
                yield event, True, current.copy()


def _rows(index, data_id, start, end) -> Iterator[tuple[object, ...]]:
    reports = index.identity_payload_values(namespace="report")
    with closing_iterator(reports):
        for payload in reports:
            row = json.loads(payload)
            pair = row[:2]
            period = date.fromisoformat(pair[1])
            if (start is not None and period < start) or (
                end is not None and period > end
            ):
                continue
            coordinates = index.identity_payload_values(
                namespace=f"disclosure_coordinates:{pair[0]}:{pair[1]}"
            )
            with closing_iterator(coordinates):
                for payload in coordinates:
                    coordinate = json.loads(payload)
                    for field in sorted(_DISCLOSURE_INDICATORS):
                        value = _get(index, "disclosure_field", [*coordinate, field])
                        if value is None:
                            error = PortfolioTargetBlocked(
                                "choice_disclosure_field_missing"
                            )
                            error.add_note(
                                f"field={field}; report_period={pair[1]}; "
                                f"EndDate={coordinate[2]}; type={coordinate[3]}"
                            )
                            raise error
            disclosure = _get(index, "disclosure", pair)
            if disclosure is None:
                raise PortfolioTargetBlocked("choice_first_disclosure_missing")
            first = date.fromisoformat(disclosure[2])
            with closing_iterator(
                index.identity_payload_values(namespace=f"events:{pair[0]}:{pair[1]}")
            ) as events:
                earliest = next(events, None)
            if (
                earliest is not None
                and date.fromisoformat(json.loads(earliest)) < first
            ):
                error = PortfolioTargetBlocked(
                    "choice_disclosure_chronology_unresolved"
                )
                error.add_note(
                    f"report_period={pair[1]}; first={first}; "
                    f"event={json.loads(earliest)}"
                )
                raise error
            if data_id == FIRST_DISCLOSURE_DATA_ID:
                yield (pair[0][:6], pair[1], first.isoformat())
                continue
            for public_date, revised, values in _report_versions(index, pair, first):
                yield (
                    pair[0][:6],
                    pair[1],
                    pair[1],
                    "A",
                    revised,
                    public_date.isoformat() if revised else None,
                    f"correction@{public_date}" if revised else "original",
                    *(
                        values[_INDICATORS.index(field)]
                        for field in FINANCIAL_FIELDS[data_id]
                    ),
                )


def _frames(rows, spec, policy):
    columns = spec.required_columns
    row_bound = 512 + 128 * len(columns)
    count = min(1024, max(1, policy.operation_headroom_bytes // (4 * row_bound)))
    bound = 65536 + count * row_bound
    lease = None
    frame = pending = owned = None
    try:
        while True:
            lease = yield bound
            pending = list(islice(rows, count))
            if not pending:
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
            lease.shrink_to(pandas_frame_payload_bytes(frame))
            owned = OwnedValue(frame, lease)
            frame = None
            yield owned
            owned = lease = None
    finally:
        frame = pending = owned = None
        if lease is not None:
            _close_lease_preserving_primary(lease, owner="source.choice.interim")


def materialize_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    data_ids: tuple[str, ...] = (*FINANCIAL_FIELDS, FIRST_DISCLOSURE_DATA_ID),
    start_date: str | None = None,
    end_date: str | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
    instance_metadata: Mapping[str, object] | None = None,
) -> ChoiceInterimImportResult:
    if (
        type(data_ids) is not tuple
        or not data_ids
        or len(set(data_ids)) != len(data_ids)
    ):
        raise ValueError("choice_interim_data_ids_invalid")
    catalog = interim_catalog()
    specs = tuple(catalog.get(data_id) for data_id in data_ids)
    root = Path(raw_root) if raw_root is not None else layout.raw_source_root(SOURCE_ID)
    start = date.fromisoformat(start_date) if start_date is not None else None
    end = date.fromisoformat(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("choice_interim_window_invalid")
    written, skipped = {}, []
    with runtime_resource_policy_scope(resource_policy) as policy:
        for financial, grouped in groupby(
            specs, key=lambda spec: spec.data_id != FIRST_DISCLOSURE_DATA_ID
        ):
            group = tuple(grouped)
            with _source_snapshot(root, group[0].data_id) as (identity, members):
                if (
                    instance_metadata is not None
                    and COMMON_MANIFEST_METADATA_KEY in instance_metadata
                ):
                    envelope = validate_business_manifest_envelope(
                        instance_metadata[COMMON_MANIFEST_METADATA_KEY]
                    )
                    if (
                        envelope["semantic_metadata"]["raw_content_identity"]
                        != identity
                    ):
                        raise ValueError("choice_raw_identity_changed")

                with _raw_index(root, financial, policy, members=members) as index:
                    for spec in group:

                        def verify_source(
                            _receipt, data_id=spec.data_id, expected=identity
                        ):
                            if source_content_identity(root, data_id) != expected:
                                raise ValueError("choice_raw_identity_changed")

                        rows = _rows(index, spec.data_id, start, end)
                        frames = _frames(rows, spec, policy)
                        target = layout.target_root(
                            spec.data_id,
                            layout.interim_data_root(SOURCE_ID, spec.data_id),
                        )
                        with closing_iterator(rows), closing_iterator(frames):
                            try:
                                write_dataset(
                                    target,
                                    frames,
                                    spec=spec,
                                    resource_policy=policy,
                                    instance_metadata=instance_metadata,
                                    staged_validator=verify_source,
                                )
                            except ValueError as error:
                                if str(error) != "empty_dataset_schema_missing":
                                    raise
                                skipped.append(spec.data_id)
                                continue
                        written[spec.data_id] = target
    return ChoiceInterimImportResult(written, tuple(skipped))
