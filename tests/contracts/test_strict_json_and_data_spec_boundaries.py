from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

import quant_research_platform.contracts._json as json_owner
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataFrequency, DataGrain, DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage import manifests


def test_time_window_invalid_string_raises_stable_error() -> None:
    with pytest.raises(ValueError, match="invalid_time_window_bound"):
        TimeWindow("not-a-date", "2026-07-06")


@pytest.mark.parametrize(
    "value",
    (
        "x" * 100_000,
        "量" * 100_000,
        ["x" for _ in range(10_000)],
        {"nested": ["x" for _ in range(10_000)]},
    ),
    ids=("huge-ascii", "huge-non-ascii", "many-small", "nested-many-small"),
)
def test_bounded_json_rejects_before_native_payload_copy(
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def copied_before_bound(_value: object) -> object:
        raise AssertionError("bounded_json_copied_payload_before_admission")

    monkeypatch.setattr(json_owner, "_json_native_object", copied_before_bound)

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "value": value},
            lazy_arrays={},
            max_bytes=64,
        )


def test_bounded_json_snapshot_rechecks_mutation_after_initial_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"schema_id": "x", "value": "safe"}
    original = json_owner._require_bounded_json_preflight

    def mutate_after_admission(value, *, max_bytes):  # noqa: ANN001, ANN202
        size = original(value, max_bytes=max_bytes)
        if value is payload:
            payload["value"] = "x" * 1024
        return size

    monkeypatch.setattr(
        json_owner,
        "_require_bounded_json_preflight",
        mutate_after_admission,
    )

    with pytest.raises(ValueError, match="^qrp_json_render_exceeds_max_bytes$"):
        json_owner.bounded_qrp_json_object_snapshot(payload, max_bytes=64)


def test_bounded_json_counts_exact_escaped_chunks_before_native_payload_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        json_owner,
        "_json_native_object",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("bounded_json_copied_escaped_payload_before_admission")
        ),
    )

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "value": "\0" * 32},
            lazy_arrays={},
            max_bytes=64,
        )


def _deep_pretty_payload() -> dict[str, object]:
    value: object = [{} for _ in range(100)]
    for _ in range(30):
        value = [value]
    return {"schema_id": "x", "value": value}


def test_bounded_json_preflights_exact_pretty_size_before_native_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _deep_pretty_payload()
    exact_size = len(json_owner.render_qrp_json_object(payload).encode("utf-8"))
    monkeypatch.setattr(
        json_owner,
        "_json_native_object",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("bounded_json_copied_pretty_payload_before_admission")
        ),
    )

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            lazy_arrays={},
            max_bytes=exact_size - 1,
        )


def test_bounded_json_exact_pretty_max_boundary_matches_renderer() -> None:
    payload = _deep_pretty_payload()
    exact_size = len(json_owner.render_qrp_json_object(payload).encode("utf-8"))

    assert (
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            lazy_arrays={},
            max_bytes=exact_size,
        )
        == exact_size
    )
    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            lazy_arrays={},
            max_bytes=exact_size - 1,
        )


def test_bounded_json_exact_native_object_never_copies_or_encodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _deep_pretty_payload()
    exact_size = len(json_owner.render_qrp_json_object(payload).encode())
    monkeypatch.setattr(
        json_owner,
        "_json_native_object",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("exact_native_payload_was_copied")
        ),
    )
    monkeypatch.setattr(
        json_owner.json,
        "JSONEncoder",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact_native_payload_was_encoded_twice")
        ),
    )

    assert (
        json_owner.bounded_rendered_qrp_json_object_size(
            payload,
            lazy_arrays={},
            max_bytes=exact_size,
        )
        == exact_size
    )


def test_bounded_json_lazy_mapping_matches_canonical_renderer() -> None:
    expected = {
        "fields_raw": ["A", "B"],
        "fields_zh": {"A": "甲", "B": "乙"},
        "schema_id": "x",
    }

    observed = json_owner.bounded_rendered_qrp_json_object_size(
        {"fields_raw": [], "fields_zh": {}, "schema_id": "x"},
        lazy_arrays={
            ("fields_raw",): iter(("A", "B")),
        },
        lazy_mappings={
            ("fields_zh",): iter((("A", "甲"), ("B", "乙"))),
        },
        max_bytes=1024,
    )

    assert observed == len(json_owner.render_qrp_json_object(expected).encode())


@pytest.mark.parametrize(
    "items",
    (
        (("B", "乙"), ("A", "甲")),
        (("A", "甲"), ("A", "乙")),
    ),
    ids=("out-of-order", "duplicate"),
)
def test_bounded_json_lazy_mapping_requires_strict_canonical_order(
    items: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(ValueError, match="^qrp_json_lazy_mapping_order_invalid$"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"fields_zh": {}, "schema_id": "x"},
            lazy_arrays={},
            lazy_mappings={("fields_zh",): iter(items)},
            max_bytes=1024,
        )


def test_bounded_json_lazy_mapping_rejects_invalid_path() -> None:
    with pytest.raises(ValueError, match="^qrp_json_lazy_mapping_path_invalid$"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"fields_zh": [], "schema_id": "x"},
            lazy_arrays={},
            lazy_mappings={("fields_zh",): iter((("A", "甲"),))},
            max_bytes=1024,
        )


def test_bounded_json_lazy_mapping_closes_source_on_size_failure() -> None:
    events: list[str] = []

    class Source:
        def __iter__(self) -> Source:
            return self

        def __next__(self) -> tuple[str, object]:
            return "A", "x" * 128

        def close(self) -> None:
            events.append("close")

    with pytest.raises(ValueError, match="^qrp_json_render_exceeds_max_bytes$"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"fields_zh": {}, "schema_id": "x"},
            lazy_arrays={},
            lazy_mappings={("fields_zh",): Source()},
            max_bytes=100,
        )

    assert events == ["close"]


def test_bounded_json_preflights_each_lazy_item_before_native_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Source:
        def __iter__(self) -> Source:
            return self

        def __next__(self) -> object:
            monkeypatch.setattr(
                json_owner,
                "_json_native_value",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("bounded_json_copied_lazy_item_before_admission")
                ),
            )
            return "\0" * 32

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "values": []},
            lazy_arrays={("values",): Source()},
            max_bytes=64,
        )


def test_bounded_json_preflights_lazy_item_pretty_size_before_native_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Source:
        def __iter__(self) -> Source:
            return self

        def __next__(self) -> object:
            monkeypatch.setattr(
                json_owner,
                "_json_native_value",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("bounded_json_copied_pretty_item_before_admission")
                ),
            )
            return _deep_pretty_payload()["value"]

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "values": []},
            lazy_arrays={("values",): Source()},
            max_bytes=1024,
        )


def test_bounded_json_close_preserves_size_limit_primary() -> None:
    class Source:
        def __iter__(self) -> Source:
            return self

        def __next__(self) -> object:
            return {"value": "x" * 128}

        def close(self) -> None:
            raise RuntimeError("close_failed")

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "values": []},
            lazy_arrays={("values",): Source()},
            max_bytes=1,
        )


def test_bounded_json_active_primary_attempts_all_lazy_closes_and_notes() -> None:
    events: list[str] = []

    class Source:
        def __init__(self, name: str) -> None:
            self.name = name

        def __iter__(self) -> Source:
            return self

        def __next__(self) -> object:
            return {"value": "x" * 128}

        def close(self) -> None:
            events.append(self.name)
            raise OSError(f"{self.name}_close_failed")

    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes") as raised:
        json_owner.bounded_rendered_qrp_json_object_size(
            {"first": [], "schema_id": "x", "second": []},
            lazy_arrays={
                ("first",): Source("first"),
                ("second",): Source("second"),
            },
            max_bytes=100,
        )

    assert events == ["first", "second"]
    assert raised.value.__notes__ == [
        "lazy JSON value close also failed: OSError",
        "lazy JSON value close also failed: OSError",
    ]


def test_bounded_json_close_failure_propagates_without_primary() -> None:
    class EmptySource:
        def __iter__(self) -> EmptySource:
            return self

        def __next__(self) -> object:
            raise StopIteration

        def close(self) -> None:
            raise RuntimeError("close_failed")

    with pytest.raises(RuntimeError, match="close_failed"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"schema_id": "x", "values": []},
            lazy_arrays={("values",): EmptySource()},
            max_bytes=1024,
        )


def test_bounded_json_attempts_all_lazy_closes_and_propagates_first_failure() -> None:
    events: list[str] = []

    class EmptySource:
        def __init__(self, name: str) -> None:
            self.name = name

        def __iter__(self) -> EmptySource:
            return self

        def __next__(self) -> object:
            raise StopIteration

        def close(self) -> None:
            events.append(self.name)
            raise RuntimeError(f"{self.name}_close_failed")

    with pytest.raises(RuntimeError, match="first_close_failed"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"first": [], "schema_id": "x", "second": []},
            lazy_arrays={
                ("first",): EmptySource("first"),
                ("second",): EmptySource("second"),
            },
            max_bytes=1024,
        )

    assert events == ["first", "second"]


def test_bounded_json_closes_owned_sources_after_later_construction_failure() -> None:
    events: list[str] = []

    class FirstSource:
        def __iter__(self) -> FirstSource:
            return self

        def __next__(self) -> object:
            raise StopIteration

        def close(self) -> None:
            events.append("first")
            raise OSError("close_secondary")

    class InvalidSecondSource:
        def __iter__(self) -> Iterator[object]:
            raise RuntimeError("construction_primary")

    with pytest.raises(RuntimeError, match="construction_primary"):
        json_owner.bounded_rendered_qrp_json_object_size(
            {"first": [], "schema_id": "x", "second": []},
            lazy_arrays={
                ("first",): FirstSource(),
                ("second",): InvalidSecondSource(),
            },
            max_bytes=1024,
        )

    assert events == ["first"]


@pytest.mark.parametrize(
    "writer_name",
    ("write_manifest", "write_internal_artifact_manifest", "replace_qrp_json"),
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
    ("write_manifest", "write_internal_artifact_manifest", "replace_qrp_json"),
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

    monkeypatch.setattr(manifests, "_MAX_QRP_JSON_BYTES", len(rendered.encode()) + 1)
    writer(accepted, payload)
    assert accepted.read_text(encoding="utf-8") == rendered + "\n"

    monkeypatch.setattr(manifests, "_MAX_QRP_JSON_BYTES", len(rendered.encode()))
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
