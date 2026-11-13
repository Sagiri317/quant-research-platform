from __future__ import annotations

import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant_research_platform.sources.choice.raw_exports import (
    ChoiceRequest,
    read_choice_response,
)
from tests.helpers.choice_raw import choice_session
from tests.helpers.choice_raw import response as _response
from tests.helpers.paths import (
    StorageLayout,
)


def _request(endpoint: str = "ctr") -> ChoiceRequest:
    arguments = {
        "ctr": (
            "FinRepCorrDetailInfo",
            "",
            "SecuCode=600000.SH,ReportDate=2024-12-31,Type=1",
        ),
        "css": (
            "600000.SH",
            "CASH,ITEMVALUECHANGEDATE",
            "ReportDate=2024-12-31,EndDate=2025-12-31,type=3",
        ),
        "sector": ("001004", "2025-12-31", ""),
    }[endpoint]
    return ChoiceRequest(endpoint, arguments, "official-sdk-test", date(2025, 12, 31))


@pytest.mark.parametrize("endpoint", ["ctr", "css", "sector"])
def test_structured_response_shapes_roundtrip(tmp_path, endpoint):
    request = _request(endpoint)
    cells = [1, None, "", 0.0]
    values = _response(
        Codes=["600000.SH"],
        Indicators=["CASH", "ITEMVALUECHANGEDATE"],
        Dates=["2025/1/1", "2025/1/2"],
        Data={"0": cells[:2], "1": cells[2:]}
        if endpoint == "ctr"
        else {"600000.SH": cells}
        if endpoint == "css"
        else cells,
    )
    with choice_session(values) as fixture:
        root = fixture.fetch(tmp_path / "structured", request)
    assert read_choice_response(root, request) == vars(values)


def test_typed_null_empty_nonfinite_and_bytes_are_distinct(tmp_path: Path) -> None:
    values = [None, "", "null", 0, False, float("nan"), float("inf"), b"x"]
    sdk = _response(
        Indicators=[f"F{i}" for i in range(len(values))], Data={"0": values}
    )
    request = _request()
    with choice_session(sdk) as session:
        root = session.fetch(
            StorageLayout(tmp_path).raw_source_root("choice"),
            request,
        )
        response = read_choice_response(root, request)
        restored = response["Data"]["0"]
        assert restored[:5] == values[:5]
        assert [type(value) for value in restored[:5]] == [
            type(value) for value in values[:5]
        ]
        assert math.isnan(restored[5]) and restored[6] == float("inf")
        assert restored[7] == b"x"


def test_long_legal_request_has_stable_identity() -> None:
    request = replace(
        _request("css"),
        arguments=(
            ",".join(f"{code:06d}.SH" for code in range(6600)),
            "CASH",
            "EndDate=2025-12-31",
        ),
    )
    assert len(request.arguments[0]) > 65536
    assert request.payload()["arguments"] == list(request.arguments)
    assert request.identity() == replace(request).identity()
