"""Serial official Choice requests and immutable, reversible raw CSV intake."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_research_platform.contracts.json_codec import render_qrp_json_object
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import evidence_text_unsafe_reason
from quant_research_platform.storage.identity import (
    filesystem_exists,
)
from quant_research_platform.storage.manifests import (
    read_manifest,
    read_qrp_json_file,
)
from quant_research_platform.storage.raw_snapshots import (
    iter_csv_snapshot,
    write_csv_snapshot,
)

_SESSION_LOCK = threading.Lock()
_PREFIX = ("__choice_key", "__choice_date_index")
_RESPONSE_FIELDS = (
    "ErrorCode",
    "ErrorMsg",
    "Codes",
    "Indicators",
    "Dates",
    "RequestID",
    "SerialID",
)
_EVENT_FIELDS = frozenset(
    {
        "STMTACTDATE",
        "ACTUALISSUINGDATE",
        "ITEMVALUECHANGEDATE",
        "NOTICEDATE",
        "RPUPDATE",
        "DELIST_DATE",
    }
)
_OPTIONS = frozenset(
    {
        "reportdate",
        "reporttype",
        "startdate",
        "enddate",
        "type",
        "secucode",
        "indexcode",
        "recvtimeout",
        "ispandas",
    }
)


@dataclass(frozen=True)
class ChoiceRequest:
    endpoint: str
    arguments: tuple[str, str, str]
    sdk_version: str
    cutoff: date

    def identity(self) -> str:
        return hashlib.sha256(
            render_qrp_json_object(self.payload()).encode()
        ).hexdigest()

    def payload(self) -> dict[str, object]:
        if self.endpoint not in {"css", "ctr", "sector"}:
            raise ValueError("choice_readonly_endpoint_required")
        if (
            type(self.arguments) is not tuple
            or len(self.arguments) != 3
            or any(type(value) is not str for value in self.arguments)
            or type(self.cutoff) is not date
            or type(self.sdk_version) is not str
            or not self.sdk_version
        ):
            raise ValueError("choice_request_invalid")
        for text in (*self.arguments, self.sdk_version):
            if evidence_text_unsafe_reason(text) is not None:
                raise ValueError("choice_request_unsafe")
        if not re.fullmatch(r"[A-Za-z0-9_.,-]+", self.arguments[0]):
            raise ValueError("choice_request_identifier_invalid")
        options: dict[str, str] = {}
        for part in self.arguments[2].split(","):
            if not part:
                continue
            key, separator, value = part.partition("=")
            key = key.lower()
            if (
                separator != "="
                or key not in _OPTIONS
                or key in options
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", value)
            ):
                raise ValueError("choice_request_option_invalid")
            options[key] = value
        if options.get("ispandas", "0") != "0":
            raise ValueError("choice_structured_response_required")
        for key in ("reportdate", "startdate", "enddate"):
            if key in options and date.fromisoformat(options[key]) > self.cutoff:
                raise ValueError("choice_request_after_cutoff")
        if self.endpoint == "css" and "enddate" not in options:
            raise ValueError("choice_explicit_enddate_required")
        if self.endpoint == "sector":
            if date.fromisoformat(self.arguments[1]) > self.cutoff:
                raise ValueError("choice_request_after_cutoff")
        elif not re.fullmatch(r"[A-Za-z0-9_,]*", self.arguments[1]):
            raise ValueError("choice_request_indicator_invalid")
        return {
            "endpoint": self.endpoint,
            "arguments": list(self.arguments),
            "sdk_version": self.sdk_version,
            "cutoff": self.cutoff.isoformat(),
            "response_schema": "EmQuantData",
        }


@contextmanager
def choice_session(sdk: Any) -> Iterator[ChoiceRawSession]:
    """Use the caller's locally loaded SDK; never persist login material."""
    if not _SESSION_LOCK.acquire(blocking=False):
        raise RuntimeError("choice_session_busy")
    try:
        started = sdk.start(
            "RecordLoginInfo=0,ForceLogin=0,HTTPTimeout=15",
            logcallback=lambda _message: 1,
        )
        if started.ErrorCode != 0:
            raise RuntimeError(f"choice_login_failed:{started.ErrorCode}")
        session = ChoiceRawSession(sdk)
        primary = None
        try:
            yield session
        except BaseException as body_error:
            primary = body_error
            raise
        finally:
            session.active = False
            try:
                stopped = sdk.stop()
                if stopped.ErrorCode != 0:
                    raise RuntimeError(f"choice_logout_failed:{stopped.ErrorCode}")
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"choice logout also failed: {type(cleanup).__name__}")
    finally:
        _SESSION_LOCK.release()


class ChoiceRawSession:
    def __init__(self, sdk: Any) -> None:
        self.sdk = sdk
        self.active = True
        self._request_lock = threading.Lock()

    def fetch(self, output_root: Path, request: ChoiceRequest) -> Path:
        payload = request.payload()
        identity = request.identity()
        root = Path(output_root)
        if not self.active:
            raise RuntimeError("choice_session_closed")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("choice_request_busy")
        try:
            # SDK has no streaming response: caller sizes each request beforehand.
            if filesystem_exists(root):
                raise FileExistsError("choice_output_already_exists")
            response = response_metadata = metadata = None
            response = getattr(self.sdk, request.endpoint)(*request.arguments)
            response_metadata = {
                name: getattr(response, name) for name in _RESPONSE_FIELDS
            }
            _validate_response_metadata(response_metadata)
            metadata = {
                "request": payload,
                "response": response_metadata,
                "collected_at": datetime.now(UTC).isoformat(),
                "status": "COMPLETE" if response.ErrorCode == 0 else "FAILED",
                "scope": _empty_scope(),
                "units": "provider_native_no_conversion",
                "types": "JSON_cell_types; SDK_null=None; bytes_hex_for_char",
                "data_container": type(response.Data).__name__,
            }
            target = (
                root
                if response.ErrorCode == 0
                else root.parent / "failed" / identity / uuid4().hex
            )

            def observed_rows() -> Iterator[tuple[object, ...]]:
                for row in _response_rows(request.endpoint, response):
                    _observe_scope(
                        metadata["scope"],
                        response.Indicators,
                        row[2:],
                        request.cutoff,
                    )
                    yield row

            write_csv_snapshot(
                target,
                columns=(*_PREFIX, *response.Indicators),
                rows=observed_rows(),
                metadata=metadata,
                validate=lambda staging: deque(
                    _validated_rows(
                        staging,
                        request,
                        metadata,
                        require_complete=False,
                        verify_bytes=True,
                    ),
                    maxlen=0,
                ),
            )
            if response.ErrorCode != 0:
                raise RuntimeError(f"choice_query_failed:{response.ErrorCode}")
            return root

        finally:
            self._request_lock.release()


def read_choice_response(
    root: Path, request: ChoiceRequest, *, manifest=None, verify_bytes: bool = False
) -> dict[str, object]:
    """Restore typed provider observations; byte auditing is explicit and optional."""
    request.payload()
    metadata = response = header = {}
    data = [] if request.endpoint == "sector" else {}
    row = value = None
    if manifest is None:
        manifest = read_qrp_json_file(root / "response.metadata.json")[1]
    metadata = manifest["metadata"]
    with closing_iterator(
        _validated_rows(
            root, request, metadata, manifest=manifest, verify_bytes=verify_bytes
        )
    ) as rows:
        for count, row in enumerate(rows):
            header = metadata["response"]
            dates, width = len(header["Dates"]), len(header["Indicators"])
            if request.endpoint == "ctr":
                data[row[0]] = list(row[2:])
            else:
                if request.endpoint == "css" and row[1] == 0:
                    data[row[0]] = [None] * (width * dates)
                elif request.endpoint == "sector" and count == 0:
                    data = [None] * (len(header["Codes"]) * width * dates)
                for field, value in enumerate(row[2:]):
                    if request.endpoint == "css":
                        data[row[0]][field * dates + row[1]] = value
                    else:
                        data[
                            (count // dates) * width * dates + field * dates + row[1]
                        ] = value
    response = dict(metadata["response"])
    if request.endpoint == "css" and not response["Dates"]:
        data.update((code, []) for code in response["Codes"])
    response["Data"] = data
    return response


def _validate_response_metadata(value: dict[str, object]) -> None:
    if set(value) != set(_RESPONSE_FIELDS):
        raise ValueError("choice_response_fields_invalid")
    for name in ("ErrorCode", "RequestID", "SerialID"):
        if type(value[name]) is not int:
            raise ValueError("choice_response_type_invalid")
    for name in ("Codes", "Indicators", "Dates"):
        items = value[name]
        if type(items) is not list or any(type(item) is not str for item in items):
            raise ValueError("choice_response_type_invalid")
        if any(evidence_text_unsafe_reason(item) is not None for item in items):
            raise ValueError("choice_response_metadata_unsafe")
    if type(value["ErrorMsg"]) is not str:
        raise ValueError("choice_response_type_invalid")
    if evidence_text_unsafe_reason(value["ErrorMsg"]) is not None:
        raise ValueError("choice_response_message_unsafe")
    indicators = value["Indicators"]
    if len(indicators) != len(set(indicators)) or any(
        name in _PREFIX for name in indicators
    ):
        raise ValueError("choice_response_indicators_invalid")


def _response_rows(endpoint: str, response: Any) -> Iterator[tuple[object, ...]]:
    width, dates = len(response.Indicators), len(response.Dates)
    if endpoint == "ctr":
        if type(response.Data) is not dict:
            raise ValueError("choice_ctr_shape_invalid")
        for key, values in response.Data.items():
            if type(key) is not str or type(values) is not list or len(values) != width:
                raise ValueError("choice_ctr_shape_invalid")
            yield (key, None, *values)
    elif endpoint == "css":
        if type(response.Data) is not dict or list(response.Data) != response.Codes:
            raise ValueError("choice_css_shape_invalid")
        for code, values in response.Data.items():
            if type(values) is not list or len(values) != width * dates:
                raise ValueError("choice_css_shape_invalid")
            for index in range(dates):
                yield (
                    code,
                    index,
                    *(values[field * dates + index] for field in range(width)),
                )
    else:
        if (
            type(response.Data) is not list
            or len(response.Data) != len(response.Codes) * width * dates
        ):
            # Official errors leave Data={} before the rank resolver runs.
            if response.ErrorCode != 0 and response.Data == {}:
                return
            raise ValueError("choice_sector_shape_invalid")
        for code_index, code in enumerate(response.Codes):
            for index in range(dates):
                start = code_index * width * dates
                yield (
                    code,
                    index,
                    *(
                        response.Data[start + field * dates + index]
                        for field in range(width)
                    ),
                )


def _empty_scope() -> dict[str, object]:
    return {
        "event_date_count": 0,
        "after_cutoff_count": 0,
        "unparsed_event_cells": 0,
        "raw_preserved": True,
        "research_cutoff_applied": False,
    }


def _observe_scope(
    scope: dict[str, object],
    indicators: list[str],
    values: tuple[object, ...],
    cutoff: date,
) -> None:
    for indicator, value in zip(indicators, values, strict=True):
        if indicator.upper() not in _EVENT_FIELDS:
            continue
        items = value if type(value) is list else [value]
        for item in items:
            if item is None or item == "":
                continue
            tokens = (
                re.findall(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}/\d{1,2}/\d{4}", item)
                if type(item) is str
                else []
            )
            if not tokens:
                scope["unparsed_event_cells"] += 1
            for token in tokens:
                parts = token.replace("-", "/").split("/")
                try:
                    observed = date(
                        *map(
                            int,
                            parts
                            if len(parts[0]) == 4
                            else [parts[2], parts[0], parts[1]],
                        )
                    )
                except ValueError:
                    scope["unparsed_event_cells"] += 1
                    continue
                scope["event_date_count"] += 1
                scope["after_cutoff_count"] += int(observed > cutoff)


def _validated_rows(
    root: Path,
    request: ChoiceRequest,
    metadata: dict[str, object],
    *,
    require_complete: bool = True,
    manifest=None,
    verify_bytes: bool = False,
) -> Iterator[tuple[object, ...]]:
    required = {"request", "response", "data_container", "units", "types"}
    diagnostics = {"collected_at", "status", "scope"}
    if (
        type(metadata) is not dict
        or not required.issubset(metadata)
        or (verify_bytes and set(metadata) != required | diagnostics)
    ):
        raise ValueError("choice_saved_metadata_invalid")
    if metadata.get("request") != request.payload():
        raise ValueError("choice_saved_request_mismatch")
    response = metadata["response"]
    _validate_response_metadata(response)
    if metadata["data_container"] != (
        "list"
        if request.endpoint == "sector" and response["ErrorCode"] == 0
        else "dict"
    ):
        raise ValueError("choice_saved_container_invalid")
    if (
        verify_bytes
        and datetime.fromisoformat(metadata["collected_at"]).utcoffset() is None
    ):
        raise ValueError("choice_saved_collection_time_invalid")
    if (
        metadata["units"] != "provider_native_no_conversion"
        or metadata["types"] != "JSON_cell_types; SDK_null=None; bytes_hex_for_char"
    ):
        raise ValueError("choice_saved_representation_invalid")
    if manifest is None:
        manifest = read_manifest(root / "response.metadata.json")
    if manifest["metadata"] != metadata:
        raise ValueError("raw_csv_metadata_invalid")
    if manifest["columns"] != [*_PREFIX, *response["Indicators"]]:
        raise ValueError("choice_saved_columns_invalid")
    if type(manifest["row_count"]) is not int or manifest["row_count"] < 0:
        raise ValueError("choice_saved_shape_invalid")
    if request.endpoint != "ctr" and manifest["row_count"] != len(
        response["Codes"]
    ) * len(response["Dates"]):
        raise ValueError("choice_saved_shape_invalid")
    scope = _empty_scope()
    count = 0
    with closing_iterator(
        iter_csv_snapshot(root, expected_manifest=manifest, verify_bytes=verify_bytes)
    ) as rows:
        for row in rows:
            if len(row) != len(response["Indicators"]) + 2 or type(row[0]) is not str:
                raise ValueError("choice_saved_shape_invalid")
            if request.endpoint == "ctr":
                if row[1] is not None or row[0] != str(count):
                    raise ValueError("choice_saved_ctr_key_invalid")
            else:
                dates = len(response["Dates"])
                if (
                    not dates
                    or count // dates >= len(response["Codes"])
                    or row[:2] != (response["Codes"][count // dates], count % dates)
                ):
                    raise ValueError("choice_saved_coordinates_invalid")
            _observe_scope(scope, response["Indicators"], row[2:], request.cutoff)
            count += 1
            try:
                yield row
            except GeneratorExit:
                return
    if request.endpoint != "ctr" and count != len(response["Codes"]) * len(
        response["Dates"]
    ):
        raise ValueError("choice_saved_shape_invalid")
    if verify_bytes and (
        metadata["scope"] != scope
        or any(
            type(metadata["scope"].get(key)) is not type(value)
            for key, value in scope.items()
        )
    ):
        raise ValueError("choice_saved_scope_invalid")
    if verify_bytes and metadata["status"] != (
        "COMPLETE" if response["ErrorCode"] == 0 else "FAILED"
    ):
        raise ValueError("choice_saved_status_invalid")
    if require_complete and response["ErrorCode"] != 0:
        raise RuntimeError(f"choice_query_failed:{response['ErrorCode']}")
