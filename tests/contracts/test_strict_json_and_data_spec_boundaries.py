from __future__ import annotations

from collections.abc import Iterator, Mapping
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


def test_bounded_json_exact_pretty_max_boundary_matches_renderer() -> None:
    payload = _deep_pretty_payload()
    exact_size = len(json_owner.render_qrp_json_object(payload).encode("utf-8"))

    assert (
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            max_bytes=exact_size,
        )
        == exact_size
    )
    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            max_bytes=exact_size - 1,
        )


@pytest.mark.parametrize(
    "writer_name",
    ("write_manifest", "write_internal_artifact_manifest"),
)
def test_manifest_writers_match_reader_size_cap(
    writer_name: str,
    tmp_path: Path,
) -> None:
    writer = getattr(manifests, writer_name)
    path = tmp_path / writer_name / "manifest.json"

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        writer(
            path,
            {"schema_id": "qrp-test", "value": "x" * (1024 * 1024)},
        )

    assert not path.exists()


@pytest.mark.parametrize(
    "writer_name",
    ("write_manifest", "write_internal_artifact_manifest"),
)
def test_manifest_writers_share_exact_pretty_render_boundary(
    writer_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _deep_pretty_payload()
    rendered = json_owner.render_qrp_json_object(payload)
    writer = getattr(manifests, writer_name)
    accepted = tmp_path / "accepted" / f"{writer_name}.json"
    rejected = tmp_path / "rejected" / f"{writer_name}.json"

    monkeypatch.setattr(manifests, "MAX_QRP_JSON_BYTES", len(rendered.encode()) + 1)
    writer(accepted, payload)
    assert accepted.read_text(encoding="utf-8") == rendered + "\n"

    monkeypatch.setattr(manifests, "MAX_QRP_JSON_BYTES", len(rendered.encode()))
    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        writer(rejected, payload)
    assert not rejected.exists()


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


def test_data_spec_validates_schema_version_before_snapshotting_roles() -> None:
    class ExplodingRoles(Mapping[str, ColumnRole]):
        def __getitem__(self, key: str) -> ColumnRole:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            raise AssertionError("column_roles_snapshotted_before_schema_validation")

        def __len__(self) -> int:
            return 0

    with pytest.raises(ValueError, match="unsupported_schema_version"):
        DataSpec(
            data_id="demo",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles=ExplodingRoles(),  # type: ignore[arg-type]
            primary_key=(),
            partition_columns=(),
            schema_version="unsupported",
        )


def test_bounded_snapshot_accepts_only_exact_frozen_containers():
    from collections import UserDict

    native = {"nested": [{"names": ["a", "b"]}]}
    frozen = MappingProxyType({"nested": (MappingProxyType({"names": ("a", "b")}),)})
    result = json_owner.bounded_qrp_json_object_snapshot(frozen, max_bytes=1024)
    assert result == native
    result["nested"][0]["names"].append("c")
    assert frozen["nested"][0]["names"] == ("a", "b")
    for invalid in (UserDict({"x": 1}), {1, 2}, range(3), object()):
        with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
            json_owner.bounded_qrp_json_object_snapshot(
                {"value": invalid}, max_bytes=1024
            )
    cycle = []
    cyclic = MappingProxyType({"cycle": (cycle,)})
    cycle.append(cyclic)
    with pytest.raises(ValueError, match="qrp_json_cycle"):
        json_owner.bounded_qrp_json_object_snapshot(cyclic, max_bytes=4096)
    deep = "leaf"
    for _ in range(65):
        deep = (deep,)
    with pytest.raises(ValueError, match="qrp_json_recursion_limit"):
        json_owner.bounded_qrp_json_object_snapshot({"deep": deep}, max_bytes=100_000)
    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_qrp_json_object_snapshot(
            MappingProxyType({"large": ("x" * 10_000,)}), max_bytes=64
        )
    assert json_owner.canonical_json_hash([1]) != json_owner.canonical_json_hash((1,))
