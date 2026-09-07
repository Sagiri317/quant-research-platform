"""Choice disclosure and historical financial snapshots to immutable interim facts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import islice
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
    if result.duplicated(["asset_id", "report_period"], keep=False).any():
        raise ValueError("duplicate_choice_first_disclosure_key")
    return result.sort_values(
        list(FIRST_DISCLOSURE_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)


def _raw_requests(raw_root: Path, *, financial: bool):
    root = require_unaliased_path(Path(raw_root), error="choice_raw_identity_changed")
    for directory in (root / "css").glob("*"):
        if not re.fullmatch(r"[0-9a-f]{64}", directory.name):
            continue
        require_unaliased_path(directory, error="choice_raw_identity_changed")
        manifest = read_manifest(directory / "response.metadata.json")
        payload = manifest["metadata"]["request"]
        request = ChoiceRequest(
            endpoint=payload["endpoint"],
            arguments=tuple(payload["arguments"]),
            sdk_version=payload["sdk_version"],
            cutoff=date.fromisoformat(payload["cutoff"]),
            max_response_bytes=manifest["byte_count"],
        )
        if request.identity() != directory.name or request.endpoint != "css":
            raise ValueError("choice_raw_identity_changed")
        options = dict(
            (name.lower(), value)
            for part in request.arguments[2].split(",")
            if part
            for name, _, value in (part.partition("="),)
        )
        fields = set(request.arguments[1].upper().split(","))
        disclosure = _DISCLOSURE_INDICATORS.issubset(fields)
        snapshot = (
            financial
            and set(_INDICATORS).issubset(fields)
            and options.get("type", "1") in {"1", "3"}
        )
        if "reportdate" not in options or not (disclosure or snapshot):
            continue
        yield directory, request, options, disclosure


def source_content_identity(raw_root: Path, data_id: str) -> str | None:
    interim_catalog().get(data_id)
    policy = current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    members = []
    with policy.lease("source.choice.identity.members", 1) as retained:
        for root, _request, _options, _disclosure in _raw_requests(
            raw_root, financial=data_id != FIRST_DISCLOSURE_DATA_ID
        ):
            retained.grow(2048)
            for name in ("response.csv", "response.metadata.json"):
                path = root / name
                try:
                    member_identity = dataset_file_identity(path)
                except (OSError, ValueError) as error:
                    raise ValueError("choice_raw_identity_changed") from error
                members.append((f"css/{root.name}/{name}", path, member_identity))
        if not members:
            return None
        return content_hash_for_files(
            members, error="choice_raw_identity_changed", resource_policy=policy
        )


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()


def _put(index, namespace, key, value):
    key, payload = _encode(key), _encode(value)
    old = index.identity_payload_value(namespace=namespace, payload=key)
    if old is not None:
        previous = json.loads(old)
        equal = (
            previous["values"] == value["values"]
            if namespace == "snapshot"
            else previous == value
        )
        if not equal:
            error = ValueError(f"choice_{namespace}_conflict")
            if namespace == "snapshot":
                identities = sorted(
                    (previous["request_identity"], value["request_identity"])
                )
                error.add_note(f"request_identities={identities}")
            raise error
    index.add_identity_payloads(namespace=namespace, payloads=((key, payload),))


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
def _raw_index(raw_root, financial, policy):
    index = ExactPrimaryKeyIndex(
        primary_key=("source_coordinate",),
        timestamp_columns=(),
        scratch_parent=raw_root.parent,
    )
    with policy.lease("source.choice.index", index.retained_memory_bytes), index:
        for root, request, options, disclosure in _raw_requests(
            raw_root, financial=financial
        ):
            period = options["reportdate"]
            with read_choice_response(
                root, request, resource_policy=policy
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
                    if disclosure:
                        first = [
                            day
                            for day in _dates(values["STMTACTDATE"])
                            if day <= request.cutoff
                        ]
                        if first:
                            _put(
                                index,
                                "disclosure",
                                pair,
                                [code, period, first[0].isoformat()],
                            )
                            _put(index, "report", pair, pair)
                        for event in _dates(values["ITEMVALUECHANGEDATE"]):
                            if event <= request.cutoff:
                                _put(
                                    index,
                                    f"events:{code}:{period}",
                                    event.isoformat(),
                                    event.isoformat(),
                                )
                    if (
                        financial
                        and set(_INDICATORS).issubset(values)
                        and options.get("type", "1") in {"1", "3"}
                    ):
                        coordinate = [
                            *pair,
                            options["enddate"],
                            options.get("type", "1"),
                        ]
                        snapshot = _values(values[field] for field in _INDICATORS)
                        _put(
                            index,
                            "snapshot",
                            coordinate,
                            {
                                "values": snapshot,
                                "request_identity": root.name,
                            },
                        )
                        _put(index, "report", pair, pair)
        yield index


def _snapshot(index, pair, day, branch):
    return _get(index, "snapshot", [*pair, day.isoformat(), branch])


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
                    (initial["request_identity"], initial_adjusted["request_identity"])
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
                requests.extend((before["request_identity"], after["request_identity"]))
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
    namespace = "disclosure" if data_id == FIRST_DISCLOSURE_DATA_ID else "report"
    reports = index.identity_payload_values(namespace=namespace)
    with closing_iterator(reports):
        for payload in reports:
            row = json.loads(payload)
            pair = row[:2]
            period = date.fromisoformat(pair[1])
            if (start is not None and period < start) or (
                end is not None and period > end
            ):
                continue
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
        for spec in specs:
            identity = source_content_identity(root, spec.data_id)
            if (
                instance_metadata is not None
                and COMMON_MANIFEST_METADATA_KEY in instance_metadata
            ):
                envelope = validate_business_manifest_envelope(
                    instance_metadata[COMMON_MANIFEST_METADATA_KEY]
                )
                if envelope["semantic_metadata"]["raw_content_identity"] != identity:
                    raise ValueError("choice_raw_identity_changed")

            def verify_source(_receipt, data_id=spec.data_id, expected=identity):
                if source_content_identity(root, data_id) != expected:
                    raise ValueError("choice_raw_identity_changed")

            with _raw_index(
                root, spec.data_id != FIRST_DISCLOSURE_DATA_ID, policy
            ) as index:
                rows = _rows(index, spec.data_id, start, end)
                frames = _frames(rows, spec, policy)
                target = layout.target_root(
                    spec.data_id, layout.interim_data_root(SOURCE_ID, spec.data_id)
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
