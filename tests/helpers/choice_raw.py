"""Write deterministic structured Choice fixtures without SDK or network access."""

import hashlib
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from quant_research_platform.sources.choice.raw_exports import (
    _PREFIX,
    ChoiceRequest,
)


def response(**changes):
    return SimpleNamespace(
        **(
            {
                "ErrorCode": 0,
                "ErrorMsg": "success",
                "Codes": [],
                "Indicators": [],
                "Dates": [],
                "RequestID": 0,
                "SerialID": 0,
                "Data": {},
            }
            | changes
        )
    )


def write_css(session, layout, *, period, end_date, indicators, data, branch="1"):
    session.response = response(
        Codes=list(data),
        Indicators=list(indicators),
        Dates=["2025-12-31"],
        Data=data,
    )
    request = ChoiceRequest(
        "css",
        (
            ",".join(data),
            ",".join(indicators),
            f"ReportDate={period},EndDate={end_date},type={branch}",
        ),
        "synthetic-official-shape",
        date(2025, 12, 31),
    )
    return session.fetch(
        layout.raw_source_root("choice"),
        request,
    ), request


@contextmanager
def choice_session(response):
    yield _ResponseFixture(response)


class _ResponseFixture:
    def __init__(self, response):
        self.response = response

    def fetch(self, root, request):
        snapshot = choice_snapshot(
            request, self.response, collected_at="2026-01-01T00:00:00+00:00"
        )
        folder = Path(root) / request.endpoint
        folder.mkdir(parents=True, exist_ok=True)
        (Path(root) / "metadata.json").write_text(
            json.dumps({"schema_id": "qrp-choice-response-stream"}), encoding="utf-8"
        )
        with (folder / "responses.jsonl").open("a", encoding="utf-8") as out:
            out.write(
                json.dumps(
                    snapshot,
                    ensure_ascii=False,
                    default=lambda value: {"bytes_hex": value.hex()},
                )
                + "\n"
            )
        return snapshot


def choice_snapshot(request, response, *, collected_at):
    """Build a synthetic structured response with explicit scalar cell encodings."""
    header = {
        name: getattr(response, name)
        for name in (
            "ErrorCode",
            "ErrorMsg",
            "Codes",
            "Indicators",
            "Dates",
            "RequestID",
            "SerialID",
        )
    }
    width, dates = len(header["Indicators"]), len(header["Dates"])
    rows = []
    if header["ErrorCode"] == 0 and request.endpoint == "ctr":
        rows = [(key, None, *cells) for key, cells in response.Data.items()]
    elif header["ErrorCode"] == 0:
        for index, code in enumerate(header["Codes"]):
            values = (
                response.Data[code]
                if request.endpoint == "css"
                else response.Data[index * width * dates : (index + 1) * width * dates]
            )
            rows.extend(
                (
                    code,
                    date_index,
                    *(values[field * dates + date_index] for field in range(width)),
                )
                for date_index in range(dates)
            )
    columns = [*_PREFIX, *header["Indicators"]]
    encoded = [list(row) for row in rows]
    digest = hashlib.sha256(
        json.dumps(
            encoded,
            ensure_ascii=False,
            default=lambda value: {"bytes_hex": value.hex()},
        ).encode()
    )
    snapshot = dict(
        manifest=dict(
            columns=columns,
            sha256=digest.hexdigest(),
            metadata=dict(
                request=request.payload(),
                response=header,
                collected_at=collected_at,
                status="COMPLETE" if header["ErrorCode"] == 0 else "FAILED",
            ),
        ),
        rows=encoded,
    )
    if header["ErrorCode"] != 0:
        snapshot["failed_data"] = (
            [
                [
                    json.dumps(key),
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=lambda item: {"bytes_hex": item.hex()},
                    ),
                ]
                for key, value in response.Data.items()
            ]
            if type(response.Data) is dict
            else json.dumps(response.Data)
        )
    return snapshot
