"""Read structured Choice snapshots and their original request metadata."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant_research_platform.contracts.json_codec import render_qrp_json_object
from quant_research_platform.policy.lifecycle import closing_iterator

_PREFIX = ("__choice_key", "__choice_date_index")


def iter_choice_snapshots(raw_root, endpoint):
    """Read one interface stream; failed and empty responses retain their records."""
    root = Path(raw_root)
    if not root.exists():
        return
    if (
        json.loads((root / "metadata.json").read_bytes())["schema_id"]
        != "qrp-choice-response-stream"
    ):
        raise ValueError("choice_raw_request_metadata_invalid")
    path = root / endpoint / "responses.jsonl"
    if not path.exists():
        return
    with path.open("rb") as stream:
        for line in stream:
            try:
                snapshot = json.loads(line, object_hook=_decode_bytes)
                manifest = snapshot["manifest"]
                if type(manifest["metadata"]["request"]) is not dict:
                    raise ValueError("choice_raw_request_metadata_invalid")
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("choice_raw_request_metadata_invalid") from error
            yield snapshot


def _decode_bytes(value):
    return bytes.fromhex(value["bytes_hex"]) if set(value) == {"bytes_hex"} else value


def _snapshot_rows(snapshot):
    for row in snapshot["rows"]:
        yield tuple(row)


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
            raise ValueError("choice_response_endpoint_invalid")
        if (
            type(self.arguments) is not tuple
            or len(self.arguments) != 3
            or any(type(value) is not str for value in self.arguments)
            or type(self.cutoff) is not date
            or type(self.sdk_version) is not str
            or not self.sdk_version
        ):
            raise ValueError("choice_request_invalid")
        return {
            "endpoint": self.endpoint,
            "arguments": list(self.arguments),
            "sdk_version": self.sdk_version,
            "cutoff": self.cutoff.isoformat(),
            "response_schema": "EmQuantData",
        }


def read_choice_response(
    snapshot,
    request: ChoiceRequest,
) -> dict[str, object]:
    """Restore the selected structured response and its typed provider values."""
    metadata = response = header = {}
    data = [] if request.endpoint == "sector" else {}
    row = value = None
    manifest = snapshot["manifest"]
    metadata = manifest["metadata"]
    with closing_iterator(
        _validated_rows(
            snapshot,
            request,
            metadata,
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
    if not {"ErrorCode", "Codes", "Indicators", "Dates"} <= set(value):
        raise ValueError("choice_response_fields_invalid")
    for name in ("Codes", "Indicators", "Dates"):
        items = value[name]
        if type(items) is not list or any(type(item) is not str for item in items):
            raise ValueError("choice_response_type_invalid")
    indicators = value["Indicators"]
    if len(indicators) != len(set(indicators)) or any(
        name in _PREFIX for name in indicators
    ):
        raise ValueError("choice_response_indicators_invalid")


def _validated_rows(
    snapshot,
    request: ChoiceRequest,
    metadata: dict[str, object],
) -> Iterator[tuple[object, ...]]:
    required = {"request", "response"}
    if type(metadata) is not dict or not required.issubset(metadata):
        raise ValueError("choice_saved_metadata_invalid")
    if metadata.get("request") != request.payload():
        raise ValueError("choice_saved_request_mismatch")
    response = metadata["response"]
    _validate_response_metadata(response)
    if response["ErrorCode"] != 0:
        raise RuntimeError(f"choice_query_failed:{response['ErrorCode']}")
    manifest = snapshot["manifest"]
    if manifest["columns"] != [*_PREFIX, *response["Indicators"]]:
        raise ValueError("choice_saved_columns_invalid")
    count = 0
    with closing_iterator(_snapshot_rows(snapshot)) as rows:
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
            count += 1
            yield row
    if request.endpoint != "ctr" and count != len(response["Codes"]) * len(
        response["Dates"]
    ):
        raise ValueError("choice_saved_shape_invalid")
