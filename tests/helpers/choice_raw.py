"""Write deterministic structured Choice fixtures without SDK or network access."""

from contextlib import contextmanager
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
    """A structured response holder for existing fixture builders; no SDK calls."""

    def __init__(self, result):
        self.response = result


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
    )
    return session.fetch(
        layout.raw_source_root("choice") / request.endpoint / request.identity(),
        request,
    ), request


@contextmanager
def choice_session(sdk):
    yield _ResponseFixture(sdk)


class _ResponseFixture:
    def __init__(self, sdk):
        self.sdk = sdk

    def fetch(self, root, request):
        from quant_research_platform.sources.choice.raw_exports import (
            _PREFIX,
            _RESPONSE_FIELDS,
            _empty_scope,
            _observe_scope,
        )
        from quant_research_platform.storage.raw_snapshots import write_csv_snapshot

        value = self.sdk.response
        width, dates = len(value.Indicators), len(value.Dates)
        rows = []
        if request.endpoint == "ctr":
            rows = [(key, None, *cells) for key, cells in value.Data.items()]
        else:
            for code_index, code in enumerate(value.Codes):
                cells = (
                    value.Data[code]
                    if request.endpoint == "css"
                    else value.Data[
                        code_index * width * dates : (code_index + 1) * width * dates
                    ]
                )
                for index in range(dates):
                    rows.append(
                        (
                            code,
                            index,
                            *(cells[field * dates + index] for field in range(width)),
                        )
                    )
        scope = _empty_scope()
        for row in rows:
            _observe_scope(scope, value.Indicators, row[2:], request.cutoff)
        write_csv_snapshot(
            root,
            columns=(*_PREFIX, *value.Indicators),
            rows=rows,
            metadata={
                "request": request.payload(),
                "response": {name: getattr(value, name) for name in _RESPONSE_FIELDS},
                "collected_at": "2026-01-01T00:00:00+00:00",
                "status": "COMPLETE",
                "scope": scope,
                "units": "provider_native_no_conversion",
                "types": "JSON_cell_types; SDK_null=None; bytes_hex_for_char",
                "data_container": type(value.Data).__name__,
            },
        )
        return root
