"""Selection is exact on data time, projected at the scanner, and scoped to members."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    read_dataset,
    reading,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import data_spec_payload


def spec():
    return DataSpec(
        data_id="selected",
        frequency="1m",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "value": "factor",
            "unused": "context",
        },
        column_units={"value": "ratio"},
    )


def frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2019-12-31T23:59:59.999999999",
                    "2020-01-01T00:00:00",
                    "2021-01-01T00:00:00",
                    "2021-01-01T00:00:00.000000001",
                    "2021-01-01T23:59:59.999999999",
                    "2021-01-02T00:00:00",
                ],
                format="ISO8601",
            ),
            "asset_id": ["a", "b", "a", "b", "a", "a"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "unused": [0] * 6,
        }
    )


@pytest.mark.parametrize("native", [False, True])
def test_noncontiguous_years_intersect_window_and_preserve_projection(tmp_path, native):
    source = tmp_path / ("native" if native else "bare.parquet")
    declared = replace(spec(), partition_columns=("year",)) if native else spec()
    values = frame()
    if native:
        write_dataset(source, values, spec=declared)
        options = {}
    else:
        values.to_parquet(source, index=False)
        options = {"spec": declared}
    selected = read_dataset(source, columns=["value"], years=[2019, 2021], **options)
    assert selected["value"].tolist() == [1.0, 3.0, 4.0, 5.0, 6.0]
    assert list(selected) == ["value"]
    selected = read_dataset(
        source,
        columns=["value"],
        years=[2019, 2021],
        time_window=TimeWindow("2020-01-01", "2021-01-01"),
        row_filters=(("asset_id", "==", "a"),),
        **options,
    )
    assert selected["value"].tolist() == [3.0, 5.0]
    empty = read_dataset(source, columns=["value"], years=[], **options)
    assert empty.empty
    assert empty.dtypes.to_dict() == {"value": values["value"].dtype}


@pytest.mark.parametrize("native", [False, True])
def test_whole_date_and_closed_nanosecond_window(tmp_path, native):
    source = tmp_path / ("native" if native else "bare.parquet")
    if native:
        write_dataset(source, frame(), spec=spec())
        options = {}
    else:
        frame().to_parquet(source, index=False)
        options = {"spec": spec()}
    day = read_dataset(
        source, time_window=TimeWindow("2021-01-01", "2021-01-01"), **options
    )
    assert day["value"].tolist() == [3.0, 4.0, 5.0]
    instant = TimeWindow(
        "2021-01-01T08:00:00.000000001+08:00", "2021-01-01T08:00:00.000000001+08:00"
    )
    assert instant.start.nanosecond == 1
    exact = read_dataset(source, time_window=instant, **options)
    assert exact["value"].tolist() == [4.0]


def test_native_explicit_members_skip_discovery_and_project_support_only(
    tmp_path, monkeypatch
):
    root = tmp_path / "native"
    write_dataset(root, frame(), spec=replace(spec(), partition_columns=("year",)))
    selected = "data/year=2021/part-000.parquet"
    (root / "data" / "unselected.parquet").write_bytes(b"not parquet")

    def reject_discovery(*_args):
        raise AssertionError("explicit_members_scanned_unselected_files")

    monkeypatch.setattr(reading.DatasetSessionInventory, "discover", reject_discovery)
    calls = []
    original = reading.iter_dataset_part_batches

    def observe(path, **kwargs):
        calls.append((path.relative_to(root).as_posix(), kwargs["columns"]))
        yield from original(path, **kwargs)

    monkeypatch.setattr(reading, "iter_dataset_part_batches", observe)
    result = read_dataset(
        root,
        files=[selected],
        columns=["value"],
        row_filters=(("asset_id", "in", ("b",)),),
    )
    assert result["value"].tolist() == [4.0]
    assert calls == [(selected, ("value", "asset_id"))]
    assert read_dataset(root, files=[], columns=["value"]).empty


def test_bare_members_preserve_order_and_ignore_unconsumed_schema(tmp_path):
    first = frame().iloc[:1].copy()
    second = frame().iloc[2:3].copy()
    first["unused"] = "text"
    second["unused"] = float("inf")
    first.to_parquet(tmp_path / "z.parquet", index=False)
    second.to_parquet(tmp_path / "a.parquet", index=False)
    (tmp_path / "unselected.parquet").write_bytes(b"corrupt")
    source = {"path": tmp_path, "spec": spec(), "files": ["z.parquet", "a.parquet"]}
    session = reading.open_dataset_session(**reading._dataset_input_options(source))
    selected = reading.materialize_dataset_session(session.projected_view(["value"]))
    assert selected["value"].tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        reading.materialize_dataset_session(session)
    with pytest.raises(ValueError, match="dataset_files_invalid"):
        read_dataset(tmp_path, files=["../outside.parquet"], spec=spec())


def test_consumed_rows_validate_keys_and_inf_without_global_summary_gate(tmp_path):
    root = tmp_path / "native"
    values = frame()
    write_dataset(root, values, spec=spec())
    part = root / "data" / "part-000.parquet"
    table = pq.read_table(part)
    table = table.set_column(
        table.schema.get_field_index("value"),
        "value",
        pa.array([float("inf"), *values["value"].iloc[1:]]),
    )
    pq.write_table(table, part)
    assert read_dataset(root, columns=["asset_id"]).shape == (6, 1)
    assert read_dataset(root, columns=["value"], years=[2021])["value"].tolist() == [
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        read_dataset(root, columns=["value"])
    duplicate = tmp_path / "duplicate.parquet"
    pd.concat([values.iloc[:1], values.iloc[:1]], ignore_index=True).to_parquet(
        duplicate, index=False
    )
    with pytest.raises(ValueError, match="duplicate_dataset_primary_key"):
        read_dataset(duplicate, spec=spec())
    assert len(read_dataset(duplicate, spec=spec(), columns=["value"])) == 2


def test_session_parses_native_metadata_once_and_config_is_pure(tmp_path, monkeypatch):
    root = tmp_path / "native"
    write_dataset(root, frame(), spec=spec())
    reads = []
    original = reading.read_qrp_json_file

    def observe(path):
        reads.append(path.name)
        return original(path)

    monkeypatch.setattr(reading, "read_qrp_json_file", observe)
    options = reading._dataset_input_options(
        {
            "path": root,
            "years": [2019, 2021],
            "time_window": {"start": "2019-01-01", "end": "2021-12-31"},
            "row_filters": [["asset_id", "in", ["a"]]],
        }
    )
    assert reads == []
    session = reading.open_dataset_session(**options)
    assert session.spec["column_units"] == {"value": "ratio"}
    narrowed = session.filtered_view(
        TimeWindow("2021-01-01", "2021-01-01")
    ).projected_view(["value"])
    assert reading.materialize_dataset_session(narrowed)["value"].tolist() == [3.0, 5.0]
    assert reads == ["spec.json", "manifest.json"]


def test_spec_semantics_are_authoritative_immutable_and_ordered(tmp_path):
    roles = {
        "timestamp": "timestamp",
        "asset_id": "key",
        "label": "label",
        "mature_at": "context",
    }
    units = {"label": "fraction"}
    declared = DataSpec(
        data_id="inference",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles=roles,
        column_units=units,
        label_availability_columns={"label": "mature_at"},
    )
    units["label"] = "percent"
    payload = data_spec_payload(declared)
    assert payload["required_columns"] == list(roles)
    assert payload["column_units"] == {"label": "fraction"}
    assert payload["label_availability_columns"] == {"label": "mature_at"}
    assert spec().column_units.get("unused") is None
    with pytest.raises(ValueError, match="invalid_column_units"):
        replace(spec(), column_units={"missing": "fraction"})
    with pytest.raises(ValueError, match="invalid_label_availability_columns"):
        replace(spec(), label_availability_columns={"value": "timestamp"})
    assert declared.column_roles["label"] is ColumnRole.LABEL


def test_bare_projection_does_not_validate_unselected_storage_type(tmp_path):
    path = tmp_path / "interval.parquet"
    table = pa.Table.from_pandas(frame().iloc[:1], preserve_index=False)
    table = table.append_column("unconsumed", pa.array([bytes(16)], type=pa.uuid()))
    pq.write_table(table, path)
    result = read_dataset(path, spec=spec(), columns=["value"])
    assert result["value"].tolist() == [1.0]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("operation,member", [("==", "a"), ("in", ("a",))])
def test_categorical_asset_filter_uses_values_not_dictionary_codes(
    tmp_path, native, operation, member
):
    values = frame()
    values["asset_id"] = pd.Categorical(
        values["asset_id"], categories=["unused", "b", "a"], ordered=True
    )
    source = tmp_path / ("native" if native else "bare.parquet")
    if native:
        write_dataset(source, values, spec=spec())
    else:
        values.to_parquet(source, index=False)
    actual = read_dataset(
        source,
        columns=["value"],
        row_filters=(("asset_id", operation, member),),
        **({} if native else {"spec": spec()}),
    )
    assert actual["value"].tolist() == [1.0, 3.0, 5.0, 6.0]


@pytest.mark.parametrize("location", ["manifest", "part", "bare"])
def test_unconsumed_logical_dtype_does_not_gate_selected_columns(tmp_path, location):
    import json

    root = tmp_path / "native"
    write_dataset(root, frame(), spec=spec())
    part = next((root / "data").rglob("*.parquet"))
    if location == "manifest":
        path = root / "_manifest" / "manifest.json"
        payload = json.loads(path.read_text())
        descriptors = payload["logical_dtypes"]["columns"]
    else:
        table = pq.read_table(part)
        metadata = dict(table.schema.metadata)
        payload = json.loads(metadata[b"qrp.logical-dtypes"])
        descriptors = payload["columns"]
    next(item for item in descriptors if item["name"] == "unused")["dtype"] = (
        "unknown_dtype"
    )
    if location == "manifest":
        path.write_text(json.dumps(payload))
    else:
        metadata[b"qrp.logical-dtypes"] = json.dumps(payload).encode()
        pq.write_table(table.replace_schema_metadata(metadata), part)
    source = part if location == "bare" else root
    options = {"spec": spec()} if location == "bare" else {}
    actual = read_dataset(source, columns=["value"], **options)
    assert actual.value.tolist() == frame().value.tolist()
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        read_dataset(source, columns=["unused"], **options)


def test_partial_primary_key_projection_rejects_nan_without_reading_other_keys(
    tmp_path,
):
    path = tmp_path / "missing-key.parquet"
    declared = DataSpec(
        data_id="observed_keys",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={"timestamp": "timestamp", "asset_id": "key", "value": "factor"},
    )
    table = pa.table(
        {
            "timestamp": pa.array([pd.Timestamp("2021-01-01")]),
            "asset_id": pa.array([float("nan")], type=pa.float64(), from_pandas=False),
            "value": [2.0],
        }
    )
    assert table["asset_id"].null_count == 0
    pq.write_table(table, path)
    assert read_dataset(path, spec=declared, columns=["value"]).value.tolist() == [2.0]
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        read_dataset(path, spec=declared, columns=["asset_id"])


@pytest.mark.parametrize("mode", ["chunked", reading.DatasetReadMode.CHUNKED])
def test_chunked_mode_string_and_enum_keep_lazy_reading(tmp_path, mode):
    from collections.abc import Iterator

    source = tmp_path / "not-opened-yet"
    chunks = read_dataset(source, mode=mode)
    assert isinstance(chunks, Iterator)
    chunks.close()


@pytest.mark.parametrize("selection", [{"files": []}, {"years": [1900]}])
def test_empty_selection_validates_consumed_physical_logical_type(tmp_path, selection):
    import json

    root = tmp_path / "native"
    write_dataset(root, frame(), spec=spec())
    path = root / "_manifest" / "manifest.json"
    payload = json.loads(path.read_text())
    descriptor = next(
        item for item in payload["logical_dtypes"]["columns"] if item["name"] == "value"
    )
    descriptor.update(kind="numpy_integer", dtype="int64")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dataset_logical_dtype_mismatch"):
        read_dataset(root, columns=["value"], **selection)
    assert read_dataset(root, columns=["timestamp"], **selection).empty


@pytest.mark.parametrize("difference", ["unit", "unknown_unit", "availability", "role"])
def test_explicit_comparison_includes_selected_scientific_declarations(
    tmp_path, difference
):
    from quant_research_platform.storage.datasets import (
        compare_datasets,
        fingerprint_dataset,
    )

    declared = DataSpec(
        data_id="declared",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "value": "label",
            "mature_a": "context",
            "mature_b": "context",
        },
        column_units={"value": "decimal_return"},
        label_availability_columns={"value": "mature_a"},
    )
    values = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-01-01"]),
            "asset_id": ["a"],
            "value": [1.0],
            "mature_a": pd.to_datetime(["2021-01-02"]),
            "mature_b": pd.to_datetime(["2021-01-03"]),
        }
    )
    if difference == "unit":
        other = replace(declared, column_units={"value": "percent"})
    elif difference == "unknown_unit":
        other = replace(declared, column_units={})
    elif difference == "availability":
        other = replace(declared, label_availability_columns={"value": "mature_b"})
    else:
        other = replace(
            declared,
            column_roles={**declared.column_roles, "value": "factor"},
            label_availability_columns={},
        )
    left, right = tmp_path / "left", tmp_path / "right"
    write_dataset(left, values, spec=declared)
    write_dataset(right, values, spec=other)
    assert compare_datasets(left, right, columns=["value"]).status == "SCHEMA_MISMATCH"
    assert (
        fingerprint_dataset(left, columns=["value"]).schema_hash
        != fingerprint_dataset(right, columns=["value"]).schema_hash
    )
    assert compare_datasets(left, right, columns=["timestamp"]).status == "MATCH"


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("index_columns", [("timestamp",), ("timestamp", "asset_id")])
def test_bare_pandas_index_fields_remain_selected_columns(
    tmp_path, empty, index_columns
):
    values = frame()
    values["value"] = pd.Series(range(len(values)), dtype="Int64")
    path = tmp_path / "indexed.parquet"
    values.set_index(list(index_columns)).to_parquet(path)
    columns = ["value", "asset_id", "timestamp"]
    result = read_dataset(
        path, spec=spec(), columns=columns, **({"years": []} if empty else {})
    )
    expected = values.loc[:, columns]
    if empty:
        expected = expected.iloc[:0]
    pd.testing.assert_frame_equal(result, expected.reset_index(drop=True))


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_partial_nested_key_rejects_invalid_consumed_component(tmp_path, bad):
    path = tmp_path / "nested-key.parquet"
    declared = DataSpec(
        data_id="nested_keys",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={"timestamp": "timestamp", "asset_id": "key", "value": "factor"},
    )
    table = pa.table(
        {
            "timestamp": pa.array([pd.Timestamp("2021-01-01")]),
            "asset_id": pa.array([{"id": bad}], type=pa.struct([("id", pa.float64())])),
            "value": [2.0],
        }
    )
    pq.write_table(table, path)
    assert read_dataset(path, spec=declared, columns=["value"]).value.tolist() == [2.0]
    with pytest.raises(
        ValueError, match="dataset_primary_key_null|invalid_dataset_primary_key_value"
    ):
        read_dataset(path, spec=declared, columns=["asset_id"])


def test_partial_nested_key_does_not_require_partial_uniqueness(tmp_path):
    path = tmp_path / "nested-key.parquet"
    declared = DataSpec(
        data_id="nested_keys",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={"timestamp": "timestamp", "asset_id": "key"},
    )
    pq.write_table(
        pa.table(
            {
                "timestamp": pd.date_range("2021-01-01", periods=2),
                "asset_id": pa.array([{"id": 1}, {"id": 1}]),
            }
        ),
        path,
    )
    assert len(read_dataset(path, spec=declared, columns=["asset_id"])) == 2


@pytest.mark.parametrize(
    "columns", [1, {"value": True}, "value", [], [""], ["value", "value"]]
)
def test_input_parser_rejects_malformed_column_selection_before_consumer_projection(
    columns,
):
    with pytest.raises(
        ValueError, match="dataset_columns_invalid|dataset_columns_empty"
    ):
        reading._dataset_input_options({"path": "not-opened", "columns": columns})


@pytest.mark.parametrize(
    "selection",
    [{"years": [2021]}, {"time_window": TimeWindow("2021-01-01", "2021-12-31")}],
)
def test_native_temporal_selection_does_not_discover_unconsumed_year_members(
    tmp_path, selection, monkeypatch
):
    from quant_research_platform.storage.datasets import inventory

    root = tmp_path / "native"
    write_dataset(root, frame(), spec=replace(spec(), partition_columns=("year",)))
    ignored = root / "data" / "year=1900"
    ignored.mkdir()
    (ignored / "not-a-part.parquet").write_bytes(b"irrelevant invalid member")
    original = inventory._directory_entries
    visited = []

    def entries(path):
        visited.append(Path(path))
        yield from original(path)

    monkeypatch.setattr(inventory, "_directory_entries", entries)
    result = read_dataset(root, **selection)
    assert set(result.timestamp.dt.year) <= {2021}
    assert ignored not in visited
    with pytest.raises(ValueError, match="invalid_dataset_part_files"):
        read_dataset(root)
