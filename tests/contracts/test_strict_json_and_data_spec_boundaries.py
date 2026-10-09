from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

import quant_research_platform.contracts.json_codec as json_owner
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain, DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import manifests


@pytest.mark.parametrize("bound", ["not-a-date", pd.NaT])
def test_time_window_invalid_bound_raises_stable_error(bound) -> None:
    with pytest.raises(ValueError, match="invalid_time_window_bound"):
        TimeWindow(bound, "2026-07-06")


def _deep_pretty_payload() -> dict[str, object]:
    value: object = [{} for _ in range(100)]
    for _ in range(30):
        value = [value]
    return {"schema_id": "x", "value": value}


def test_json_snapshot_matches_pretty_renderer():
    payload = _deep_pretty_payload()
    assert json_owner.render_qrp_json_object(
        json_owner.qrp_json_object_snapshot(payload)
    ) == json_owner.render_qrp_json_object(payload)


@pytest.mark.parametrize(
    "writer_name", ("write_manifest", "write_internal_artifact_manifest")
)
def test_manifest_writers_roundtrip_above_former_capacity(writer_name, tmp_path):
    payload = {"schema_id": "qrp-test", "value": "x" * (1024 * 1024 + 1)}
    path = tmp_path / writer_name / "manifest.json"
    getattr(manifests, writer_name)(path, payload)
    assert manifests.read_manifest(path) == payload


@pytest.mark.parametrize(
    "writer_name", ("write_manifest", "write_internal_artifact_manifest")
)
def test_manifest_writers_preserve_pretty_rendering(writer_name, tmp_path):
    payload = _deep_pretty_payload()
    path = tmp_path / (writer_name + ".json")
    getattr(manifests, writer_name)(path, payload)
    assert (
        path.read_text(encoding="utf-8")
        == json_owner.render_qrp_json_object(payload) + "\n"
    )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"schema_id": ""},
        {"schema_id": 1},
        {"schema_id": "qrp-test/0.1"},
    ),
    ids=("missing", "empty", "non-string", "slash"),
)
def test_manifest_writer_rejects_invalid_schema_identity(
    payload: dict[str, object],
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="manifest_json_identity_invalid"):
        manifests.write_manifest(path, payload)

    assert not path.exists()


def test_data_spec_snapshots_caller_roles_as_an_immutable_mapping() -> None:
    roles = {
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    spec = DataSpec(
        data_id="demo",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles=roles,
        primary_key=("asset_id", "timestamp"),
        partition_columns=("year",),
    )
    roles["value"] = ColumnRole.FEATURE

    assert type(spec.column_roles) is MappingProxyType
    assert tuple(spec.column_roles) == ("asset_id", "timestamp")
    assert spec.required_columns == ("asset_id", "timestamp")
    assert replace(spec).column_roles == spec.column_roles
    with pytest.raises(TypeError):
        spec.column_roles["value"] = ColumnRole.FEATURE  # type: ignore[index]


def test_snapshot_accepts_only_exact_frozen_containers():
    from collections import UserDict

    native = {"nested": [{"names": ["a", "b"]}]}
    frozen = MappingProxyType({"nested": (MappingProxyType({"names": ("a", "b")}),)})
    result = json_owner.qrp_json_object_snapshot(frozen)
    assert result == native
    result["nested"][0]["names"].append("c")
    assert frozen["nested"][0]["names"] == ("a", "b")
    for invalid in (UserDict({"x": 1}), {1, 2}, range(3), object()):
        with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
            json_owner.qrp_json_object_snapshot({"value": invalid})
    cycle = []
    cyclic = MappingProxyType({"cycle": (cycle,)})
    cycle.append(cyclic)
    with pytest.raises(ValueError, match="qrp_json_cycle"):
        json_owner.qrp_json_object_snapshot(cyclic)
    deep = "leaf"
    for _ in range(65):
        deep = (deep,)
    snapshot = json_owner.qrp_json_object_snapshot({"deep": deep})
    assert (
        json_owner.parse_qrp_json_object(
            json_owner.render_qrp_json_object(snapshot).encode()
        )
        == snapshot
    )
    assert json_owner.qrp_json_object_snapshot(
        MappingProxyType({"large": ("x" * 10_000,)})
    ) == {"large": ["x" * 10_000]}
    assert json_owner.canonical_json_hash([1]) != json_owner.canonical_json_hash((1,))
