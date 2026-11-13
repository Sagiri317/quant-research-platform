from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pandas as pd
import pytest

import quant_research_platform.contracts.json_codec as json_owner
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain, DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import manifests


@pytest.mark.parametrize("bound", ["not-a-date", "NaT", pd.NaT])
def test_time_window_invalid_bound_raises_stable_error(bound) -> None:
    with pytest.raises(ValueError, match="invalid_time_window_bound"):
        TimeWindow(bound, "2026-07-06")


def test_manifest_writer_preserves_nested_research_parameters(tmp_path):
    payload = {
        "schema_id": "example",
        "parameters": {"columns": ["value"], "window": {"start": "2024-01-01"}},
    }
    path = tmp_path / "manifest.json"
    manifests.write_qrp_json_file(path, payload)
    assert manifests.read_qrp_json_file(path)[1] == payload


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
