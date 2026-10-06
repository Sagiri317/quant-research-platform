"""Synthetic official SDK shapes written through the real immutable raw collector."""

from datetime import date
from types import SimpleNamespace

from quant_research_platform.sources.choice.raw_exports import ChoiceRequest


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


class SDK:
    def __init__(self, result):
        self.response = result
        self.calls = 0
        self.stops = 0
        self.stop_error = 0
        self.last_arguments = None

    def start(self, options, *, logcallback):
        assert "ForceLogin=0" in options and "RecordLoginInfo=0" in options
        return response()

    def stop(self):
        self.stops += 1
        return response(ErrorCode=self.stop_error)

    def css(self, *args):
        self.calls += 1
        self.last_arguments = args
        return self.response

    ctr = css
    sector = css


def write_css(session, layout, *, period, end_date, indicators, data, branch="1"):
    session.sdk.response = response(
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
        1024 * 1024,
    )
    return session.fetch(layout, request), request
