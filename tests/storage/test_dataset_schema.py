from __future__ import annotations

import json
from collections import UserDict
from dataclasses import replace
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.datasets.schema as dataset_schema
import quant_research_platform.storage.datasets.writing as dataset_writing
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.json_codec import canonical_identity_bytes
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_read_text,
    native_filesystem_path,
)
from tests.storage.dataset_test_support import (
    _ARROW_SCALAR_LOGICAL_DTYPE_CASES,
    _ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES,
)
from tests.storage.low_level_fixtures import (
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


@pytest.mark.parametrize(
    "dtype",
    (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "boolean",
        "Float32",
        "Float64",
        "string",
    ),
)
def test_dataset_global_extension_dtype_survives_read_scopes(
    tmp_path: Path, dtype: str
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"extension_{dtype}")
    frame = make_default_dataset_frame()
    if dtype == "boolean":
        values = [True, pd.NA, False]
    elif dtype == "string":
        values = ["x", pd.NA, "z"]
    elif dtype.startswith("Float"):
        values = [1.5, pd.NA, 3.5]
    else:
        values = [1, pd.NA, 3]
    frame["value"] = pd.Series(values, index=frame.index, dtype=dtype)
    write_dataset(root, frame, spec=make_processed_dataset_spec(f"extension_{dtype}"))
    columns = ("asset_id", "timestamp", "value")
    materialized = read_dataset(root, columns=columns)
    chunked = list(read_dataset(root, columns=columns, mode=DatasetReadMode.CHUNKED))
    windowed = read_dataset(
        root,
        columns=columns,
        time_window=TimeWindow(date(2015, 1, 1), date(2015, 12, 31)),
    )
    assert str(materialized["value"].dtype) == dtype
    assert {str(part["value"].dtype) for part in chunked} == {dtype}
    assert str(windowed["value"].dtype) == dtype
    expected = frame.loc[:, list(columns)]
    for observed, selected in (
        (materialized, expected),
        (pd.concat(chunked, ignore_index=True), expected),
        (windowed, expected.loc[expected["timestamp"].dt.year == 2015]),
    ):
        pd.testing.assert_frame_equal(
            observed.sort_values(["asset_id", "timestamp"]).reset_index(drop=True),
            selected.sort_values(["asset_id", "timestamp"]).reset_index(drop=True),
            check_dtype=False,
        )


def test_writer_rejects_missing_persisted_logical_dtype_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_dataset_artifact_root(tmp_path, "missing_logical_dtype")

    def write_without_metadata(table, path):
        filesystem_mkdir(path.parent, parents=True, exist_ok=True)
        pq.write_table(
            table.replace_schema_metadata(None), native_filesystem_path(path)
        )
        return path

    monkeypatch.setattr(dataset_writing, "write_parquet_file", write_without_metadata)
    with pytest.raises(ValueError, match="dataset_logical_dtypes_missing"):
        write_dataset(
            root,
            make_default_dataset_frame(),
            spec=make_processed_dataset_spec("missing_logical_dtype"),
        )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"schema_id":"unknown","columns":[]}\n',
        b'{"schema_id":"qrp-dataset-part-logical-dtypes","columns":[],"x":1}\n',
        b'{"schema_id":"qrp-dataset-part-logical-dtypes","columns":[],"x":1,"x":2}\n',
    ),
)
def test_logical_dtype_metadata_rejects_unknown_and_noncanonical_payloads(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        dataset_writing.parse_logical_dtype_metadata(payload)


@pytest.mark.parametrize(
    "series",
    (
        pd.Series([1], dtype="Int64"),
        pd.Series([True], dtype="boolean"),
        pd.Series(["x"], dtype="string"),
        pd.Series([1], dtype="int64"),
        pd.Series([True], dtype="bool"),
        pd.Series([1.0], dtype="float64"),
        pd.Series(pd.to_datetime(["2026-01-01"])),
        pd.Series(pd.to_timedelta([1], unit="s")),
        pd.Series([object()], dtype="object"),
        pd.Series(pd.Categorical(["x"], categories=["x", "y"])),
    ),
)
def test_logical_dtype_grammar_round_trips_writer_canonical_families(
    series: pd.Series,
) -> None:
    frame = pd.DataFrame({"value": series})
    payload = logical_dtype_payload_from_frame(frame)
    assert dataset_schema.validate_logical_dtype_payload(payload) == payload


@pytest.mark.parametrize(
    "value",
    (
        np.int64(7),
        True,
        1.25,
        "text",
        b"\x00\xff",
        Decimal("1.25"),
        date(2026, 1, 2),
        pd.Timestamp("2026-01-02T03:04:05+00:00"),
    ),
)
def test_logical_category_scalar_codec_round_trips_supported_values(
    value: object,
) -> None:
    payload = dataset_schema._typed_category_value_payload(value)
    assert (
        dataset_schema._typed_category_value_payload(
            dataset_schema._typed_category_value(payload)
        )
        == payload
    )
    assert dataset_schema._typed_category_identity(payload) == canonical_identity_bytes(
        payload
    )


def test_logical_category_scalar_codec_rejects_invalid_values() -> None:
    for value in (float("nan"), Decimal("NaN"), pd.Timestamp("NaT"), object()):
        with pytest.raises(ValueError, match="dataset_categorical_value_invalid"):
            dataset_schema._typed_category_value_payload(value)
    for payload in (
        {"type": "float", "value": float("nan")},
        {"type": "bytes", "value": "0"},
        {"type": "decimal", "value": "not-a-decimal"},
        {"type": "decimal", "value": "NaN"},
        {"type": "date", "value": "not-a-date"},
        {"type": "timestamp", "value": "not-a-timestamp"},
        {"type": "unknown", "value": "value"},
    ):
        with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
            dataset_schema._typed_category_value(payload)


def test_logical_dtype_payload_rejects_invalid_contract_shapes() -> None:
    schema_id = "qrp-dataset-part-logical-dtypes"
    valid = {"name": "value", "kind": "numpy_integer", "dtype": "int64"}

    def payload(*columns: object) -> dict[str, object]:
        return {"schema_id": schema_id, "columns": list(columns)}

    invalid_payloads = (
        None,
        {"schema_id": "unknown", "columns": [valid]},
        {"schema_id": schema_id, "columns": []},
        payload(None),
        payload({"name": "", "kind": "numpy_integer", "dtype": "int64"}),
        payload({"name": "value", "kind": "", "dtype": "int64"}),
        payload({"name": "value", "kind": "categorical", "ordered": False}),
        payload(
            {"name": "value", "kind": "categorical", "ordered": 0, "categories": []}
        ),
        payload(
            {"name": "value", "kind": "categorical", "ordered": False, "categories": ()}
        ),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": False,
                "categories": [
                    {"type": "int", "value": 1},
                    {"type": "int", "value": 1},
                ],
            }
        ),
        payload({**valid, "extra": True}),
        payload({"name": "value", "kind": "numpy_integer", "dtype": ""}),
        payload(
            {
                "name": "value",
                "kind": "categorical",
                "ordered": False,
                "categories": [{"type": "int", "value": 1, "extra": True}],
            }
        ),
        payload({"name": "value", "kind": "numpy_integer", "dtype": "invalid"}),
        payload(valid, valid),
    )
    for invalid in invalid_payloads:
        with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
            dataset_schema.validate_logical_dtype_payload(invalid)


def test_logical_dtype_merge_preserves_only_compatible_semantics() -> None:

    def payload(series: pd.Series, name: str = "value") -> dict[str, object]:
        return logical_dtype_payload_from_frame(pd.DataFrame({name: series}))

    numpy_integer = payload(pd.Series([1], dtype="int64"))
    payload(pd.Series([1], dtype="Int64"))
    payload(pd.Series([True], dtype="bool"))
    payload(pd.Series([True], dtype="boolean"))
    payload(pd.Series([1.0], dtype="float64"))
    payload(pd.Series([1.0], dtype="Float64"))
    ordered_a = payload(
        pd.Series(pd.Categorical(["a"], categories=["a"], ordered=True))
    )
    ordered_b = payload(
        pd.Series(pd.Categorical(["b"], categories=["b"], ordered=True))
    )
    payload(pd.Series(pd.Categorical(["a"], categories=["a"], ordered=True)))
    incompatible = (
        ([], "dataset_logical_dtypes_missing"),
        (
            [numpy_integer, payload(pd.Series([1], dtype="int64"), "other")],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [numpy_integer, payload(pd.Series(["x"], dtype="string"))],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [numpy_integer, payload(pd.Series([1], dtype="int32"))],
            "dataset_logical_dtype_mismatch",
        ),
        (
            [
                ordered_a,
                payload(
                    pd.Series(pd.Categorical(["a"], categories=["a"], ordered=False))
                ),
            ],
            "dataset_logical_dtype_mismatch",
        ),
        ([ordered_a, ordered_b], "dataset_ordered_categorical_mismatch"),
    )
    for payloads, error in incompatible:
        with pytest.raises(ValueError, match=error):
            dataset_schema.merge_logical_dtype_payloads(payloads)


def test_dataset_schema_entrypoints_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="dataset_arrow_schema_required"):
        dataset_schema.physical_schema_from_schema([])
    with pytest.raises(TypeError, match="dataset_frame_required"):
        logical_dtype_payload_from_frame(None)
    with pytest.raises(ValueError, match="dataset_column_not_found"):
        logical_dtype_payload_from_frame(pd.DataFrame({"value": [1]}), columns=("x",))
    logical_dtype_payload_from_frame(pd.DataFrame({"value": [1]}))
    for series in (
        pd.Series(pd.array([b"x"], dtype=pd.ArrowDtype(pa.binary()))),
        pd.Series(pd.period_range("2026-01", periods=1, freq="M")),
    ):
        with pytest.raises(ValueError, match="unsupported_dataset_logical_dtype"):
            logical_dtype_payload_from_frame(pd.DataFrame({"value": series}))


def test_physical_schema_payload_rejects_invalid_contract_shapes() -> None:
    field = {"name": "value", "nullable": False, "type": {"kind": "string"}}
    invalid_types = (
        None,
        {"kind": "integer", "signed": 1, "bit_width": 8},
        {"kind": "fixed_size_binary", "byte_width": 0},
        {"kind": "timestamp", "unit": "ns", "timezone": 1},
        {"kind": "struct", "fields": []},
        {"kind": "map", "keys_sorted": 1, "key_field": field, "item_field": field},
        {"kind": "unknown"},
    )
    invalid_schemas = [
        None,
        [],
        [{}],
        [{**field, "nullable": 0}],
        [field, dict(field)],
        *[[{**field, "type": invalid_type}] for invalid_type in invalid_types],
    ]
    for invalid in invalid_schemas:
        with pytest.raises(ValueError, match="manifest_physical_schema_invalid"):
            dataset_schema.validate_physical_schema_payload(invalid)


def test_dataset_schema_validators_reject_mutated_runtime_contracts() -> None:
    with pytest.raises(TypeError, match="invalid_dataset_instance_metadata"):
        dataset_schema.merge_dataset_instance_metadata({}, [])
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        dataset_schema.merge_dataset_instance_metadata({}, {1: "value"})
    with pytest.raises(ValueError, match="invalid_dataset_identity"):
        dataset_schema.dataset_identity_payload({}, {})
    with pytest.raises(ValueError, match="unsupported_dataset_instance_metadata_key"):
        dataset_schema.validate_dataset_manifest_instance_metadata({"unknown": True})
    with pytest.raises(ValueError, match="dataset_spec_type_invalid"):
        dataset_schema.validate_dataset_spec(object())
    for manifest in (
        {},
        {"primary_key": ["asset_id"], "null_count_summary": {}},
        {"primary_key": ["asset_id"], "null_count_summary": {"asset_id": True}},
    ):
        with pytest.raises(ValueError, match="dataset_manifest_not_consumable"):
            dataset_schema.validate_dataset_primary_key_null_summary(manifest)


@pytest.mark.parametrize(
    ("case", "error"),
    (
        ("grain", "dataset_grain_invalid"),
        ("missing_primary_key", "dataset_primary_key_invalid"),
        ("partition", "unsupported_partitioning"),
        ("key_role", "invalid_primary_key_roles"),
        ("unpartitioned_timestamp", "unpartitioned_data_must_not_use_timestamp_key"),
        ("missing_timestamp", "invalid_timestamp_primary_key"),
    ),
)
def test_dataset_schema_rejects_mutated_spec_invariants(case: str, error: str) -> None:
    spec = make_processed_dataset_spec(f"mutated_{case}")
    if case == "grain":
        object.__setattr__(spec, "grain", "asset_timestamp")
    elif case == "missing_primary_key":
        object.__setattr__(spec, "primary_key", ("missing",))
    elif case == "partition":
        object.__setattr__(spec, "partition_columns", ("month",))
    elif case == "key_role":
        roles = dict(spec.column_roles)
        roles["asset_id"] = ColumnRole.FACTOR
        object.__setattr__(spec, "column_roles", MappingProxyType(roles))
    elif case == "unpartitioned_timestamp":
        object.__setattr__(spec, "partition_columns", ())
    else:
        object.__setattr__(spec, "primary_key", ("market_id", "asset_id"))
    with pytest.raises(ValueError, match=error):
        dataset_schema.validate_dataset_spec(spec)


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "expected_kind", "expected_dtype", "sample"),
    _ARROW_SCALAR_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES],
)
@pytest.mark.parametrize("shape", ("empty", "null", "value"))
def test_arrow_scalar_logical_dtypes_use_canonical_pandas_semantic_family(
    _case_id: str,
    arrow_type: pa.DataType,
    expected_kind: str,
    expected_dtype: str,
    sample: object,
    shape: str,
) -> None:
    values = {"empty": [], "null": [None], "value": [sample]}[shape]
    frame = pd.DataFrame({"value": pd.array(values, dtype=pd.ArrowDtype(arrow_type))})
    payload = logical_dtype_payload_from_frame(frame)
    descriptor = payload["columns"][0]
    overrides = dataset_schema.pandas_dtype_overrides_from_logical_payload(
        payload, columns=("value",)
    )
    assert descriptor == {
        "name": "value",
        "kind": expected_kind,
        "dtype": expected_dtype,
    }
    parsed = overrides["value"]
    assert not isinstance(parsed, str)
    assert parsed == pd.api.types.pandas_dtype(expected_dtype)


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "_expected_kind", "expected_dtype", "sample"),
    _ARROW_SCALAR_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_SCALAR_LOGICAL_DTYPE_CASES],
)
def test_arrow_scalar_dtypes_are_stable_across_storage_read_boundaries(
    tmp_path: Path,
    _case_id: str,
    arrow_type: pa.DataType,
    _expected_kind: str,
    expected_dtype: str,
    sample: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"arrow_{_case_id}")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(
        pd.array([sample, None, sample], dtype=pd.ArrowDtype(arrow_type)),
        index=frame.index,
    )
    write_dataset(
        root,
        (frame.iloc[1:].copy(), frame.iloc[:1].copy()),
        spec=make_processed_dataset_spec(
            f"arrow_{_case_id}", value_role=ColumnRole.CONTEXT
        ),
    )
    materialized = read_dataset(root, columns=("timestamp", "value"))
    chunks = list(
        read_dataset(root, columns=("timestamp", "value"), mode=DatasetReadMode.CHUNKED)
    )
    windowed = read_dataset(
        root,
        columns=("timestamp", "value"),
        time_window=TimeWindow(date(2015, 1, 1), date(2015, 12, 31)),
    )
    assert str(materialized["value"].dtype) == expected_dtype
    assert {str(chunk["value"].dtype) for chunk in chunks} == {expected_dtype}
    assert str(windowed["value"].dtype) == expected_dtype


@pytest.mark.parametrize("arrow_type", (pa.float32(), pa.float64()))
@pytest.mark.parametrize("has_nan", (False, True))
def test_public_arrow_float_validity_survives_reads_rewrite_and_exact_compare(
    tmp_path, arrow_type, has_nan
):
    from quant_research_platform.contracts.data import ProcessedDataKind

    values = pa.array(
        [float("nan") if has_nan else 0.5, None, 1.5],
        type=arrow_type,
        from_pandas=False,
    )
    frame = pd.DataFrame(
        {
            "asset_id": pd.Series(
                [f"synthetic_{index}" for index in range(3)], dtype="string"
            ),
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D").astype(
                "datetime64[ns]"
            ),
            "value": pd.Series(values, dtype=pd.ArrowDtype(arrow_type)),
        }
    )
    original = frame.copy(deep=True)
    expected = pa.array(frame["value"], from_pandas=False)
    expected_nan = pc.fill_null(pc.is_nan(expected), False)
    assert expected.null_count == 1
    assert pc.sum(expected_nan).as_py() == int(has_nan)
    assert pc.sum(pc.fill_null(pc.is_finite(expected), False)).as_py() == 2 - int(
        has_nan
    )
    spec = replace(
        make_processed_dataset_spec(
            "arrow_float_validity",
            kind=ProcessedDataKind.PANEL,
            primary_key=("asset_id", "timestamp"),
        ),
        column_roles={
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.CONTEXT,
        },
    )
    root = make_dataset_artifact_root(
        tmp_path / "source", spec.data_id, kind=ProcessedDataKind.PANEL
    )

    def assert_validity(actual):
        assert actual.type == arrow_type
        assert len(actual) == 3
        assert actual.is_null().equals(expected.is_null())
        assert pc.fill_null(pc.is_nan(actual), False).equals(expected_nan)
        assert pc.filter(actual, pc.is_finite(actual)).equals(
            pc.filter(expected, pc.is_finite(expected))
        )

    write_dataset(root, frame, spec=spec)
    original_manifest = (root / "_manifest" / "manifest.json").read_bytes()
    paths = sorted((root / "data").rglob("*.parquet"))
    persisted = []
    for path in paths:
        with pq.ParquetFile(native_filesystem_path(path)) as parquet:
            persisted.append(parquet.read(columns=["value"]).column(0).combine_chunks())
    assert_validity(pa.concat_arrays(persisted))
    for mode in (DatasetReadMode.CHUNKED, DatasetReadMode.MATERIALIZED):
        if mode is DatasetReadMode.CHUNKED:
            readback = tuple(read_dataset(root, mode=mode))
            parts = readback
        else:
            readback = read_dataset(root, mode=mode)
            parts = (readback,)
        assert {str(part["value"].dtype) for part in parts} == {
            f"Float{arrow_type.bit_width}"
        }
        assert_validity(
            pa.concat_arrays(
                [pa.array(part["value"], from_pandas=False) for part in parts]
            )
        )
        actual_keys = pd.concat(
            [part.loc[:, list(spec.primary_key)] for part in parts], ignore_index=True
        )
        pd.testing.assert_frame_equal(actual_keys, frame.loc[:, list(spec.primary_key)])
        rewritten = make_dataset_artifact_root(
            tmp_path / mode.value, spec.data_id, kind=ProcessedDataKind.PANEL
        )
        write_dataset(rewritten, readback, spec=spec)
        assert compare_datasets(root, rewritten).status == "MATCH"
    assert (root / "_manifest" / "manifest.json").read_bytes() == original_manifest
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("_case_id", "arrow_type", "_expected_kind", "expected_dtype", "sample"),
    _ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES,
    ids=[case[0] for case in _ARROW_TIMESTAMP_LOGICAL_DTYPE_CASES],
)
def test_arrow_datetime_dtype_is_stable_for_filter_empty_direct_session(
    tmp_path: Path,
    _case_id: str,
    arrow_type: pa.DataType,
    _expected_kind: str,
    expected_dtype: str,
    sample: object,
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"arrow_filter_empty_{_case_id}")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(
        pd.array([sample, None, sample], dtype=pd.ArrowDtype(arrow_type)),
        index=frame.index,
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(
            f"arrow_filter_empty_{_case_id}", value_role=ColumnRole.CONTEXT
        ),
    )
    session = dataset_reading.open_dataset_session(
        root,
        columns=("value",),
        time_window=None,
        row_filters=(("asset_id", "==", "missing"),),
    )
    chunks = list(dataset_reading.iter_dataset_chunks(session))
    materialized = dataset_reading.materialize_dataset_session(session)
    assert len(chunks) == 1
    assert chunks[0].empty
    assert materialized.empty
    assert str(chunks[0]["value"].dtype) == expected_dtype
    assert str(materialized["value"].dtype) == expected_dtype


@pytest.mark.parametrize(
    "descriptor",
    (
        {"name": "value", "kind": "numpy_boolean", "dtype": "bool[pyarrow]"},
        {"name": "value", "kind": "numpy_integer", "dtype": "int16[pyarrow]"},
        {"name": "value", "kind": "numpy_float", "dtype": "float[pyarrow]"},
        {"name": "value", "kind": "datetime", "dtype": "timestamp[ms][pyarrow]"},
        {"name": "value", "kind": "string", "dtype": "string[pyarrow]"},
        {"name": "value", "kind": "numpy_integer", "dtype": "int"},
        {"name": "value", "kind": "numpy_float", "dtype": "float"},
        {"name": "value", "kind": "unknown", "dtype": "object"},
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "bytes", "value": "FF"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "decimal", "value": "+1.0"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "decimal", "value": "-0"}],
        },
        {
            "name": "value",
            "kind": "categorical",
            "ordered": False,
            "categories": [{"type": "timestamp", "value": "2026-01-01 00:00:00"}],
        },
    ),
)
def test_logical_dtype_grammar_rejects_noncanonical_siblings(
    descriptor: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="dataset_logical_dtypes_invalid"):
        dataset_schema.validate_logical_dtype_payload(
            {"schema_id": "qrp-dataset-part-logical-dtypes", "columns": [descriptor]}
        )


@pytest.mark.parametrize("reverse", (False, True), ids=("negative-first", "zero-first"))
@pytest.mark.parametrize("empty_sibling", (False, True), ids=("rows", "empty"))
def test_unordered_decimal_zero_category_has_one_canonical_domain(
    reverse: bool, empty_sibling: bool
) -> None:
    negative = pd.DataFrame()
    negative["value"] = pd.Categorical(
        [Decimal("-0.00")], categories=[Decimal("-0.00")], ordered=False
    )
    zero = pd.DataFrame()
    zero["value"] = pd.Categorical(
        [] if empty_sibling else [Decimal("0")],
        categories=[Decimal("0")],
        ordered=False,
    )
    payloads = [
        logical_dtype_payload_from_frame(frame)
        for frame in ((zero, negative) if reverse else (negative, zero))
    ]
    merged = dataset_schema.merge_logical_dtype_payloads(payloads)
    descriptor = merged["columns"][0]
    assert descriptor["categories"] == [{"type": "decimal", "value": "0"}]
    final_dtype = dataset_schema.pandas_dtype_overrides_from_logical_payload(
        merged, columns=("value",)
    )["value"]
    assert isinstance(final_dtype, pd.CategoricalDtype)
    assert list(final_dtype.categories) == [Decimal("0")]
    assert final_dtype.categories.size == 1
    materialized = pd.Series([Decimal("-0.00"), Decimal("0")], dtype=final_dtype)
    assert materialized.cat.categories.size == 1
    assert materialized.tolist() == [Decimal("0"), Decimal("0")]


@pytest.mark.parametrize(
    ("left_domain", "right_domain", "expected_type", "expected_values"),
    (
        (("1.0", "2.00"), ("1.00", "3.00"), pa.decimal128(3, 2), ("1", "2", "3")),
        (
            ("12345678901234567890.0100", "2.00000"),
            ("12345678901234567890.01000", "3.00000"),
            pa.decimal128(25, 5),
            ("12345678901234567890.01", "2", "3"),
        ),
    ),
    ids=("equal-values", "precision-exceeds-context"),
)
def test_unordered_decimal_category_shards_match_whole_and_preserve_physical_type(
    tmp_path: Path, left_domain, right_domain, expected_type, expected_values
) -> None:
    left = make_default_dataset_frame().iloc[:1].copy()
    right = make_default_dataset_frame().iloc[1:].copy()
    left["value"] = pd.Categorical(
        [Decimal(left_domain[0])], categories=[Decimal(item) for item in left_domain]
    )
    right["value"] = pd.Categorical(
        [Decimal(right_domain[0]), None],
        categories=[Decimal(item) for item in right_domain],
    )
    assert pa.Array.from_pandas(left["value"].cat.categories).type == expected_type
    assert pa.Array.from_pandas(right["value"].cat.categories).type == expected_type
    whole = pd.concat(
        (
            left.assign(value=left["value"].astype(object)),
            right.assign(value=right["value"].astype(object)),
        ),
        ignore_index=True,
    )
    whole["value"] = pd.Categorical(
        whole["value"],
        categories=[
            Decimal(right_domain[0]),
            Decimal(left_domain[1]),
            Decimal(right_domain[1]),
        ],
    )
    observed = []
    metadata = []
    with localcontext() as context:
        context.prec = 2
        for name, frames in (
            ("whole", whole),
            ("forward", (left, right)),
            ("reverse", (right, left)),
        ):
            root = make_dataset_artifact_root(tmp_path, name)
            write_dataset(
                root,
                frames,
                spec=make_processed_dataset_spec(name, value_role=ColumnRole.CONTEXT),
            )
            inspection = inspect_dataset(root)
            assert inspection.row_count == 3
            session = dataset_reading.open_dataset_session(
                root, columns=None, time_window=None
            )
            assert all(
                pq.read_schema(native_filesystem_path(path)).field("value").type
                == expected_type
                for path in session.all_paths
            )
            materialized = read_dataset(root)
            chunked = pd.concat(
                list(read_dataset(root, mode=DatasetReadMode.CHUNKED)),
                ignore_index=True,
            )
            selected = materialized.sort_values(["asset_id", "timestamp"]).reset_index(
                drop=True
            )
            pd.testing.assert_frame_equal(
                selected,
                chunked.sort_values(["asset_id", "timestamp"]).reset_index(drop=True),
            )
            assert isinstance(selected["value"].dtype, pd.CategoricalDtype)
            assert not selected["value"].cat.ordered
            assert set(selected["value"].cat.categories) == {
                Decimal(item) for item in expected_values
            }
            assert selected["value"].isna().sum() == 1
            assert (
                selected.loc[selected["value"].notna(), "value"].tolist()
                == [Decimal(expected_values[0])] * 2
            )
            descriptor = next(
                item
                for item in inspection.manifest["logical_dtypes"]["columns"]
                if item["name"] == "value"
            )
            assert {item["value"] for item in descriptor["categories"]} == set(
                expected_values
            )
            assert {item["type"] for item in descriptor["categories"]} == {"decimal"}
            metadata.append(
                (
                    inspection.manifest["logical_dtypes"],
                    inspection.manifest["logical_dtype_hash"],
                    inspection.manifest["physical_schema"],
                )
            )
            observed.append(selected)
    assert metadata[0] == metadata[1] == metadata[2]
    for frame in observed[1:]:
        pd.testing.assert_frame_equal(observed[0], frame)


@pytest.mark.parametrize(
    "domain",
    (("1.00", "2.00"), ("2.00", "1.00"), ("1.00", "3.00"), ("1.0", "2.00")),
    ids=(
        "same-order",
        "reversed-order",
        "different-domain",
        "different-representation",
    ),
)
def test_ordered_decimal_category_shards_preserve_order_and_reject_domain_drift(
    tmp_path: Path, domain
) -> None:
    root = make_dataset_artifact_root(tmp_path, "ordered_decimal")
    left = make_default_dataset_frame().iloc[:1].copy()
    right = make_default_dataset_frame().iloc[1:2].copy()
    left["value"] = pd.Categorical(
        [Decimal("1.00")], categories=[Decimal("1.00"), Decimal("2.00")], ordered=True
    )
    right["value"] = pd.Categorical(
        [Decimal("1.00")], categories=[Decimal(item) for item in domain], ordered=True
    )
    assert pa.Array.from_pandas(left["value"].cat.categories).type == pa.decimal128(
        3, 2
    )
    assert pa.Array.from_pandas(right["value"].cat.categories).type == pa.decimal128(
        3, 2
    )
    spec = make_processed_dataset_spec("ordered_decimal", value_role=ColumnRole.CONTEXT)
    if domain != ("1.00", "2.00"):
        with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
            write_dataset(root, (left, right), spec=spec)
        assert not filesystem_exists(root)
    else:
        write_dataset(root, (left, right), spec=spec)
        for frame in (
            read_dataset(root),
            *read_dataset(root, mode=DatasetReadMode.CHUNKED),
        ):
            assert frame["value"].cat.ordered
            assert [str(value) for value in frame["value"].cat.categories] == [
                "1.00",
                "2.00",
            ]
            assert [str(value) for value in frame["value"]] == ["1.00"] * len(frame)
        assert inspect_dataset(root).row_count == 2


@pytest.mark.parametrize("ordered", (False, True))
def test_decimal_category_metadata_admits_existing_exact_representation(
    ordered: bool,
) -> None:
    frame = pd.DataFrame(
        {
            "value": pd.Categorical(
                [Decimal("1.00")], categories=[Decimal("1.00")], ordered=ordered
            )
        }
    )
    payload = logical_dtype_payload_from_frame(frame)
    if ordered:
        assert payload["columns"][0]["categories"] == [
            {"type": "decimal", "value": "1.00"}
        ]
    payload["columns"][0]["categories"] = [{"type": "decimal", "value": "1.00"}]
    assert dataset_schema.validate_logical_dtype_payload(payload) == payload
    parsed = dataset_schema.parse_logical_dtype_metadata(
        dataset_schema.logical_dtype_metadata_bytes(payload)
    )
    merged = dataset_schema.merge_logical_dtype_payloads([parsed])
    assert merged == payload
    dtype = dataset_schema.pandas_dtype_overrides_from_logical_payload(
        merged, columns=("value",)
    )["value"]
    assert dtype.ordered is ordered
    assert dtype.categories[0].as_tuple().exponent == -2


@pytest.mark.parametrize("ordered", (False, True))
def test_decimal_category_reader_accepts_existing_exact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ordered: bool
) -> None:
    root = make_dataset_artifact_root(tmp_path, "existing_decimal_metadata")
    frame = make_default_dataset_frame()
    frame["value"] = pd.Categorical(
        [Decimal("1.00"), None, Decimal("1.00")],
        categories=[Decimal("1.00"), Decimal("2.00")],
        ordered=ordered,
    )
    current_frame_payload = dataset_writing.logical_dtype_payload_from_frame

    def exact_frame_payload(current, *, columns=None):
        payload = current_frame_payload(current, columns=columns)
        for descriptor in payload["columns"]:
            if descriptor["kind"] == "categorical":
                descriptor["categories"] = [
                    dataset_schema._typed_category_value_payload(value)
                    for value in current[descriptor["name"]].cat.categories
                ]
        return payload

    with monkeypatch.context() as patch:
        patch.setattr(
            dataset_writing, "logical_dtype_payload_from_frame", exact_frame_payload
        )
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(
                "existing_decimal_metadata", value_role=ColumnRole.CONTEXT
            ),
        )
    for observed in (
        read_dataset(root),
        *read_dataset(root, mode=DatasetReadMode.CHUNKED),
    ):
        assert observed["value"].cat.ordered is ordered
        assert [str(value) for value in observed["value"].cat.categories] == [
            "1.00",
            "2.00",
        ]
    checked = inspect_dataset(root)
    assert checked.row_count == 3
    descriptor = next(
        item
        for item in checked.manifest["logical_dtypes"]["columns"]
        if item["name"] == "value"
    )
    assert descriptor["categories"] == [
        {"type": "decimal", "value": "1.00"},
        {"type": "decimal", "value": "2.00"},
    ]


def test_decimal_category_identity_keeps_scalar_types_distinct() -> None:
    identities = [
        dataset_schema._typed_category_identity(
            dataset_schema._typed_category_value_payload(value)
        )
        for value in (Decimal("1.00"), 1, 1.0, True, "1")
    ]
    assert len(set(identities)) == len(identities)
    assert canonical_identity_bytes(Decimal("1.0")) != canonical_identity_bytes(
        Decimal("1.00")
    )


@pytest.mark.parametrize(
    "domain",
    ((1, 3), (1.0, 3.0), (Decimal("1.000"), Decimal("3.000"))),
    ids=("integer", "float", "decimal-scale"),
)
def test_decimal_category_shards_reject_physical_type_drift(
    tmp_path: Path, domain
) -> None:
    root = make_dataset_artifact_root(tmp_path, "decimal_type_drift")
    left = make_default_dataset_frame().iloc[:1].copy()
    right = make_default_dataset_frame().iloc[1:2].copy()
    left["value"] = pd.Categorical(
        [Decimal("1.0")], categories=[Decimal("1.0"), Decimal("2.00")]
    )
    right["value"] = pd.Categorical([domain[0]], categories=domain)
    with pytest.raises(ValueError, match="dataset_schema_mismatch"):
        write_dataset(
            root,
            (left, right),
            spec=make_processed_dataset_spec(
                "decimal_type_drift", value_role=ColumnRole.CONTEXT
            ),
        )
    assert not filesystem_exists(root)


def test_empty_unordered_category_domain_unions_with_nonempty_domain(
    tmp_path: Path,
) -> None:
    manifests = []
    for name, reverse in (("empty_first", False), ("empty_last", True)):
        root = make_dataset_artifact_root(tmp_path, name)
        empty = make_default_dataset_frame().iloc[:0].copy()
        empty["asset_id"] = pd.Categorical([], categories=["empty-only"], ordered=False)
        nonempty = make_default_dataset_frame().iloc[:1].copy()
        nonempty["asset_id"] = pd.Categorical(
            ["live-only"], categories=["live-only"], ordered=False
        )
        chunks = (nonempty, empty) if reverse else (empty, nonempty)
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
        manifest = json.loads(
            filesystem_read_text(root / "_manifest" / "manifest.json", encoding="utf-8")
        )
        session = dataset_reading.open_dataset_session(
            root, columns=None, time_window=None
        )
        assert manifest["row_count"] == 1
        assert manifest["file_count"] == len(session.all_paths) == 2
        assert manifest["partition_count"] == 1
        expected_partition = f"year={nonempty['timestamp'].iloc[0].year}"
        assert manifest["partition_row_counts"] == {expected_partition: 1}
        assert [path.name for path in session.all_paths] == [
            "part-000.parquet",
            "part-001.parquet",
        ]
        assert len({path.parent for path in session.all_paths}) == 1
        assert tuple(
            session.metadata_for(path).row_count for path in session.all_paths
        ) == (1, 0)
        assert session.logical_dtypes == manifest["logical_dtypes"]
        assert read_dataset(root)["asset_id"].astype("string").tolist() == ["live-only"]
        manifests.append(manifest)
    domains = []
    for manifest in manifests:
        descriptor = next(
            item
            for item in manifest["logical_dtypes"]["columns"]
            if item["name"] == "asset_id"
        )
        domains.append(descriptor["categories"])
    assert domains[0] == domains[1]
    assert {item["value"] for item in domains[0]} == {"empty-only", "live-only"}


@pytest.mark.parametrize("empty_first", (False, True))
def test_empty_ordered_category_domain_mismatch_is_not_ignored(
    tmp_path: Path, empty_first: bool
) -> None:
    name = f"empty_ordered_mismatch_{empty_first}"
    root = make_dataset_artifact_root(tmp_path, name)
    empty = make_default_dataset_frame().iloc[:0].copy()
    empty["asset_id"] = pd.Categorical([], categories=["a", "b"], ordered=True)
    nonempty = make_default_dataset_frame().iloc[:1].copy()
    nonempty["asset_id"] = pd.Categorical(["a"], categories=["a", "c"], ordered=True)
    chunks = (empty, nonempty) if empty_first else (nonempty, empty)
    with pytest.raises(ValueError, match="dataset_ordered_categorical_mismatch"):
        write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
    assert not filesystem_exists(root)


@pytest.mark.parametrize("empty_chunks", (False, True))
def test_writer_retains_one_incremental_logical_dtype_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty_chunks: bool
) -> None:
    name = f"incremental_dtype_plan_{empty_chunks}"
    root = make_dataset_artifact_root(tmp_path, name)
    base = make_default_dataset_frame().iloc[:1].copy()
    chunks = []
    for index in range(32):
        current = base.iloc[:0].copy() if empty_chunks else base.copy()
        if not empty_chunks:
            current["asset_id"] = f"asset-{index:03d}"
            current["timestamp"] = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        chunks.append(current)
    retained_column_counts: list[int] = []
    original = dataset_writing._WriteStats.observe_logical_dtype_payload

    def record_plan(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        assert self._logical_dtype_plan is not None
        retained_column_counts.append(len(self._logical_dtype_plan["columns"]))
        return result

    monkeypatch.setattr(
        dataset_writing._WriteStats, "observe_logical_dtype_payload", record_plan
    )
    write_dataset(root, chunks, spec=make_processed_dataset_spec(name))
    assert retained_column_counts
    assert set(retained_column_counts) == {len(base.columns)}


def test_unordered_category_domain_uses_live_exact_index_namespace(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "category_domain")
    frame = make_default_dataset_frame().iloc[:1].copy()
    categories = [f"asset-{index:05d}" for index in range(2000)]
    frame["asset_id"] = pd.Categorical(
        [categories[-1]], categories=categories, ordered=False
    )
    observed_counts: list[int] = []

    def validate(receipt: dataset_writing.DatasetWriteReceipt) -> None:
        observed_counts.append(
            len(
                list(
                    receipt.exact_primary_key_index.identity_payload_values(
                        namespace="logical-category:asset_id"
                    )
                )
            )
        )

    dataset_writing.write_dataset(
        root,
        (frame,),
        spec=make_processed_dataset_spec("category_domain"),
        staged_validator=validate,
    )
    assert observed_counts == [len(categories)]
    assert len(read_dataset(root)["asset_id"].cat.categories) == len(categories)


def test_instance_metadata_preserves_typed_mapping_and_error_order() -> None:
    identities = UserDict(
        {
            "input": UserDict(
                {"data_id": "input", "spec_hash": "a" * 64, "manifest_hash": "b" * 64}
            )
        }
    )
    archive = UserDict(
        {"schema_id": "archive", "artifact_id": "first", "manifest_hash": "c" * 64}
    )
    metadata = UserDict(
        {"input_dataset_identities": identities, "source_artifact_identity": archive}
    )
    expected = dataset_schema.merge_dataset_instance_metadata({}, metadata)
    actual = dataset_schema.merge_dataset_instance_metadata({}, metadata)
    assert actual == expected
    actual["input_dataset_identities"]["input"]["spec_hash"] = "d" * 64
    assert identities["input"]["spec_hash"] == "a" * 64
    with pytest.raises(ValueError, match="invalid_input_dataset_identities_metadata"):
        dataset_schema.merge_dataset_instance_metadata(
            {},
            {
                "input_dataset_identities": UserDict({"input": {"data_id": "wrong"}}),
                "producer_diagnostics": object(),
            },
        )


@pytest.mark.parametrize(
    ("old", "new", "accepted"),
    (
        (pa.list_(pa.null()), pa.list_(pa.int64()), True),
        (
            pa.list_(pa.struct([pa.field("leaf", pa.null())])),
            pa.list_(pa.struct([pa.field("leaf", pa.int64())])),
            True,
        ),
        (
            pa.list_(pa.field("item", pa.null())),
            pa.list_(pa.field("element", pa.int64())),
            False,
        ),
        (
            pa.struct([pa.field("leaf", pa.null(), nullable=True)]),
            pa.struct([pa.field("leaf", pa.int64(), nullable=False)]),
            False,
        ),
        (
            pa.struct([pa.field("leaf", pa.null())]),
            pa.struct([pa.field("leaf", pa.int64()), pa.field("added", pa.int64())]),
            False,
        ),
        (
            pa.struct([pa.field("leaf", pa.null()), pa.field("known", pa.int64())]),
            pa.struct([pa.field("leaf", pa.int64()), pa.field("known", pa.float64())]),
            False,
        ),
        (pa.list_(pa.null(), 2), pa.list_(pa.int64(), 3), False),
        (pa.list_(pa.null()), pa.large_list(pa.int64()), False),
    ),
    ids=(
        "list-leaf",
        "deep-leaf",
        "field-name",
        "nullable",
        "field-count",
        "known-type",
        "fixed-length",
        "container-kind",
    ),
)
def test_null_resolution_preserves_arrow_structure(old, new, accepted):
    assert dataset_schema.arrow_type_has_null(old)
    assert dataset_schema.arrow_type_resolves_null(old, new) is accepted
    assert not dataset_schema.arrow_type_has_null(new)


@pytest.mark.parametrize("value_type", (pa.null(), pa.int64()))
def test_null_resolution_does_not_alias_list_item_and_element(value_type):
    old = pa.list_(pa.field("item", value_type))
    new = pa.list_(pa.field("element", value_type))
    assert old == new
    assert not dataset_schema.arrow_type_resolves_null(old, new)


def test_null_resolution_preserves_map_key_order_contract():
    old = pa.map_(pa.string(), pa.null(), keys_sorted=False)
    changed = pa.map_(pa.string(), pa.int64(), keys_sorted=True)
    resolved = pa.map_(pa.string(), pa.int64(), keys_sorted=False)
    assert dataset_schema.arrow_type_has_null(old)
    assert not dataset_schema.arrow_type_resolves_null(old, changed)
    assert dataset_schema.arrow_type_resolves_null(old, resolved)
