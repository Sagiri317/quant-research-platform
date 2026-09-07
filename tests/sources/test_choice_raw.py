from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant_research_platform.runtime.resources import (
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.choice.raw_exports import (
    ChoiceRequest,
    choice_session,
    read_choice_response,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_manifest
from tests.helpers.choice_raw import SDK
from tests.helpers.choice_raw import response as _response


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
    return ChoiceRequest(
        endpoint, arguments, "official-sdk-test", date(2025, 12, 31), 1024 * 1024
    )


@pytest.mark.parametrize("endpoint", ["ctr", "css", "sector"])
def test_official_shapes_roundtrip_and_exact_reuse(
    tmp_path: Path, endpoint: str
) -> None:
    request = _request(endpoint)
    if endpoint == "ctr":
        response = _response(
            Indicators=["SECUCODE", "NOTICEDATE", "QKMZ", "HKMZ"],
            Data={
                "0": ["600000.SH", "2026/1/2", None, 0.0],
                "1": ["600000.SH", "2025/12/31", "", '含,逗号\n与引号"'],
            },
        )
    else:
        flat = [1, -0.0, ["2025/1/1", "2026/1/2", None], "2025/3/1"]
        response = _response(
            Codes=["600000.SH"],
            Indicators=["CASH", "ITEMVALUECHANGEDATE"],
            Dates=["2025/1/1", "2025/1/2"],
            Data={"600000.SH": flat} if endpoint == "css" else flat,
        )
    sdk = SDK(response)
    layout = StorageLayout(tmp_path)
    with choice_session(sdk) as session:
        root = session.fetch(layout, request)
        assert root.is_relative_to(tmp_path / "data/raw/choice")
        original = (root / "response.csv").read_bytes()
        assert session.fetch(layout, request) == root
        assert sdk.calls == 1
        with read_choice_response(
            root, request, resource_policy=session.policy
        ) as restored:
            assert restored == vars(response)
            actual = copy.deepcopy(restored)
        assert actual == vars(response)
        assert (root / "response.csv").read_bytes() == original
        assert session.policy.used_memory_bytes == 0
    assert sdk.stops == 1
    metadata = read_manifest(root / "response.metadata.json")
    assert metadata["columns"][2:] == response.Indicators
    assert metadata["metadata"]["scope"]["after_cutoff_count"] == 1
    assert metadata["metadata"]["scope"]["research_cutoff_applied"] is False
    assert metadata["metadata"]["response"]["Codes"] == response.Codes


def test_documented_ctr_report_type_is_forwarded_and_persisted_exactly(tmp_path: Path):
    request = ChoiceRequest(
        "ctr",
        (
            "BalanceStatementSHSZ",
            "REPORTDATE,STR_ISLISTINGREPORT,FIRSTNOTICEDATE",
            "SecuCode=600000.SH,ReportDate=2024-12-31,ReportType=1",
        ),
        "official-sdk-test",
        date(2025, 12, 31),
        1024 * 1024,
    )
    sdk = SDK(
        _response(
            Indicators=["REPORTDATE", "STR_ISLISTINGREPORT", "FIRSTNOTICEDATE"],
            Data={"0": ["2024-12-31", 0, "2025-04-15"]},
        )
    )
    with choice_session(sdk) as session:
        root = session.fetch(StorageLayout(tmp_path), request)
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        assert session.fetch(StorageLayout(tmp_path), request) == root
        assert sdk.calls == 1 and sdk.last_arguments == request.arguments
        with read_choice_response(
            root, request, resource_policy=session.policy
        ) as restored:
            assert restored == vars(sdk.response)
        metadata = read_manifest(root / "response.metadata.json")["metadata"]
        assert metadata["request"]["arguments"] == list(request.arguments)
        assert metadata["scope"]["research_cutoff_applied"] is False
        assert before == {path.name: path.read_bytes() for path in root.iterdir()}
        assert session.policy.used_memory_bytes == 0
    assert sdk.stops == 1


def test_sdk_null_empty_nonfinite_and_bytes_are_distinct(tmp_path: Path) -> None:
    values = [None, "", "null", 0, False, float("nan"), float("inf"), b"x"]
    sdk = SDK(
        _response(Indicators=[f"F{i}" for i in range(len(values))], Data={"0": values})
    )
    request = _request()
    with choice_session(sdk) as session:
        root = session.fetch(StorageLayout(tmp_path), request)
        with read_choice_response(
            root, request, resource_policy=session.policy
        ) as response:
            restored = response["Data"]["0"]
            assert restored[:5] == values[:5]
            assert [type(value) for value in restored[:5]] == [
                type(value) for value in values[:5]
            ]
            assert math.isnan(restored[5]) and restored[6] == float("inf")
            assert restored[7] == b"x"


@pytest.mark.parametrize("corruption", ["value", "request", "column", "count", "scope"])
def test_corrupt_completed_response_refuses_reuse_without_overwriting(
    tmp_path: Path, corruption: str
) -> None:
    sdk = SDK(_response(Indicators=["QKMZ"], Data={"0": [12]}))
    request = _request()
    with choice_session(sdk) as session:
        root = session.fetch(StorageLayout(tmp_path), request)
        sidecar = root / "response.metadata.json"
        metadata = json.loads(sidecar.read_text())
        if corruption == "value":
            csv = root / "response.csv"
            csv.write_bytes(csv.read_bytes().replace(b"12", b"13"))
        elif corruption == "request":
            metadata["metadata"]["request"]["sdk_version"] = "different-sdk"
        elif corruption == "column":
            metadata["columns"][-1] = "DIFFERENT"
        elif corruption == "count":
            metadata["row_count"] = 2
        else:
            metadata["metadata"]["scope"]["after_cutoff_count"] = 1
        if corruption != "value":
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        before = {path.name: path.read_bytes() for path in root.iterdir()}
        with pytest.raises(ValueError):
            session.fetch(StorageLayout(tmp_path), request)
        with pytest.raises(ValueError):
            with read_choice_response(root, request, resource_policy=session.policy):
                pytest.fail("corrupt raw became visible")
        assert sdk.calls == 1
        assert before == {path.name: path.read_bytes() for path in root.iterdir()}


def test_empty_success_and_failed_attempt_then_explicit_success(tmp_path: Path) -> None:
    request = _request()
    sdk = SDK(_response(ErrorCode=10001001, ErrorMsg="transport failed"))
    with choice_session(sdk) as session:
        with pytest.raises(RuntimeError, match="choice_query_failed:10001001"):
            session.fetch(StorageLayout(tmp_path), request)
        failures = list(
            (tmp_path / "data/raw/choice/ctr/failed").rglob("response.metadata.json")
        )
        assert len(failures) == 1
        assert read_manifest(failures[0])["metadata"]["status"] == "FAILED"
        sdk.response = _response()
        root = session.fetch(StorageLayout(tmp_path), request)
        assert "failed" not in root.parts
        assert read_manifest(root / "response.metadata.json")["row_count"] == 0
        assert (
            read_manifest(root / "response.metadata.json")["metadata"]["response"][
                "Indicators"
            ]
            == []
        )
        with read_choice_response(
            root, request, resource_policy=session.policy
        ) as response:
            assert response["Data"] == {}
        session.fetch(StorageLayout(tmp_path), request)
        assert sdk.calls == 2


def test_admission_precedes_query_and_leases_are_released(tmp_path: Path) -> None:
    sdk = SDK(_response())
    with runtime_resource_policy_scope(explicit_cap_bytes=16 * 1024 * 1024) as policy:
        with choice_session(sdk, resource_policy=policy) as session:
            with pytest.raises(MemoryError, match="choice_request_exceeds_memory_plan"):
                session.fetch(StorageLayout(tmp_path), _request())
            assert sdk.calls == 0 and policy.used_memory_bytes == 0
    assert sdk.stops == 1


def test_invalid_shape_or_oversize_preserves_old_good(tmp_path: Path) -> None:
    request = _request()
    sdk = SDK(_response(Indicators=["QKMZ"], Data={"0": [1]}))
    with choice_session(sdk) as session:
        root = session.fetch(StorageLayout(tmp_path), request)
        original = (root / "response.csv").read_bytes()
        for values in ([1, 2], ["x" * (65536 + 1)]):
            sdk.response = _response(Indicators=["QKMZ"], Data={"0": values})
            different = replace(request, sdk_version="other-sdk")
            with pytest.raises(ValueError):
                session.fetch(StorageLayout(tmp_path), different)
            assert (root / "response.csv").read_bytes() == original
            assert session.policy.used_memory_bytes == 0
        assert len(list(root.parent.iterdir())) == 1


def test_atomic_failure_preserves_primary_and_session_exclusivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_research_platform.storage.atomic as atomic

    sdk = SDK(_response(Indicators=["QKMZ"], Data={"0": [1]}))
    sdk.stop_error = 123
    primary = KeyboardInterrupt("test_primary")

    def fail_publish(*args: object) -> None:
        raise primary

    monkeypatch.setattr(atomic, "publish_no_replace", fail_publish)
    with pytest.raises(KeyboardInterrupt) as caught:
        with choice_session(sdk) as session:
            with pytest.raises(RuntimeError, match="choice_session_busy"):
                with choice_session(sdk):
                    pass
            session.fetch(StorageLayout(tmp_path), _request())
    assert caught.value is primary
    assert any("choice logout" in note for note in primary.__notes__)
    assert sdk.stops == 1
    assert not list(tmp_path.rglob("response.csv"))


@pytest.mark.parametrize(
    "changes",
    [
        {"endpoint": "order"},
        {"arguments": ("600000.SH", "CASH", "Password=secret")},
        {"arguments": ("600000.SH", "CASH", "EndDate=2026-01-01")},
        {"arguments": ("600000.SH", "CASH", "")},
        {"arguments": ("600000.SH", "CASH", "EndDate=2025-12-31,ISPANDAS=1")},
    ],
)
def test_request_rejects_unsafe_or_implicit_cutoff(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_request("css"), **changes).payload()
