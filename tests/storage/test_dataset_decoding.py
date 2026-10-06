from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import quant_research_platform.storage.datasets.reading as dataset_reading
import quant_research_platform.storage.parquet as dataset_physical
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    compare_datasets,
    fingerprint_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.reading import _prepare_in_filter
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_payload_from_frame,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_iterdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_stat,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from tests.storage.low_level_fixtures import (
    fixture_dataset_content_hash,
    make_dataset_artifact_root,
    make_default_dataset_frame,
    make_processed_dataset_spec,
)


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


@pytest.mark.parametrize("filtered", (False, True))
def test_plain_reader_streams_normal_files(tmp_path, filtered):
    path = tmp_path / "plain.parquet"
    table = pa.table({"value": np.arange(4096, dtype=np.int64)})
    pq.write_table(
        table,
        native_filesystem_path(path),
        use_dictionary=False,
        compression=None,
        row_group_size=2048,
    )
    expected = np.arange(1024 if filtered else 0, 4096, dtype=np.int64)
    delivered = 0
    with closing_iterator(
        dataset_physical.iter_dataset_part_batches(
            path,
            expected_identity=None,
            parquet_filters=[("value", ">=", 1024)] if filtered else None,
            batch_rows=1,
        )
    ) as batches:
        for batch in batches:
            assert batch.num_rows <= 10
            np.testing.assert_array_equal(
                batch.column(0).to_numpy(),
                expected[delivered : delivered + batch.num_rows],
            )
            delivered += batch.num_rows
    assert delivered == len(expected)


def test_categorical_converter_rejects_nonnull_value_outside_declared_domain() -> None:
    batch = pa.record_batch([pa.array(["known", "unknown", None])], names=["value"])
    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        tuple(
            dataset_reading._arrow_batch_to_bounded_pandas(
                batch,
                dtype_overrides={"value": pd.CategoricalDtype(categories=["known"])},
                batch_rows=8,
            )
        )


def test_declared_member_preserves_output_across_parent_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "part.parquet"
    batches = (
        pa.record_batch([pa.array(["x" * 4096])], names=["value"]),
        pa.record_batch([pa.array(["y" * 4096])], names=["value"]),
    )
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches)
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda *_args, **_kwargs: iter(batches),
    )
    converted = dataset_reading.iter_declared_parquet_member_batches(
        path, expected_identity=SimpleNamespace()
    )
    output = tuple(converted)
    for observed, expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


def test_dataset_file_batches_preserve_output_across_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = (tmp_path / "part-0.parquet", tmp_path / "part-1.parquet")
    batches = {
        paths[0]: pa.record_batch([pa.array([1], type=pa.int64())], names=["value"]),
        paths[1]: pa.record_batch([pa.array([2], type=pa.int64())], names=["value"]),
    }
    expected = tuple(batch.to_pandas(use_threads=False) for batch in batches.values())
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_part_batches",
        lambda path, **_kwargs: iter((batches[Path(path)],)),
    )
    context = SimpleNamespace(
        logical_dtypes=logical_dtype_payload_from_frame(expected[0]),
        parquet_filters=None,
        read_paths=paths,
        metadata_for=lambda _path: SimpleNamespace(identity=SimpleNamespace()),
        verify=lambda: None,
    )
    converted = dataset_reading.iter_dataset_file_batches(context, columns=("value",))
    output = tuple(converted)
    assert tuple((path for path, _ in output)) == paths
    for (_, observed), expected_frame in zip(output, expected, strict=True):
        pd.testing.assert_frame_equal(observed, expected_frame)


def test_declared_variable_width_member_decodes_multirow_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "declared-member.parquet"
    row_count = 5000
    pq.write_table(
        pa.table({"value": [f"value-{index:05d}" for index in range(row_count)]}),
        native_filesystem_path(path),
    )
    conversions = 0
    original = dataset_reading._arrow_slice_to_pandas

    def record_conversion(batch: pa.RecordBatch, **kwargs) -> pd.DataFrame:
        nonlocal conversions
        conversions += 1
        return original(batch, **kwargs)

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", record_conversion)
    frames = tuple(
        dataset_reading.iter_declared_parquet_member_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            batch_rows=128,
        )
    )
    assert sum(len(frame) for frame in frames) == row_count
    assert conversions < row_count // 10


@pytest.mark.parametrize(
    "kind",
    [
        "string",
        "binary",
        "large_string",
        "large_binary",
        "category",
        "struct",
        "list",
        "large_list",
        "fixed_list",
        "map",
    ],
)
def test_dictionary_decode_preserves_supported_logical_types_and_keys(tmp_path, kind):
    payload = "x" * (32 * 1024)
    values = [None if index % 23 == 0 else payload for index in range(256)]
    if kind in ("string", "large_string"):
        array = pa.array(
            values, type=pa.string() if kind == "string" else pa.large_string()
        )
    elif kind in ("binary", "large_binary"):
        array = pa.array(
            [None if value is None else value.encode() for value in values],
            type=pa.binary() if kind == "binary" else pa.large_binary(),
        )
    elif kind == "category":
        array = pa.array(
            pd.Categorical(values, categories=[payload, "unused"], ordered=True)
        )
    elif kind == "struct":
        array = pa.array(
            [None if value is None else {"text": value} for value in values],
            type=pa.struct([("text", pa.string())]),
        )
    elif kind == "map":
        array = pa.array(
            [None if value is None else [("key", value)] for value in values],
            type=pa.map_(pa.string(), pa.string()),
        )
    else:
        data_type = (
            pa.list_(pa.string(), 2)
            if kind == "fixed_list"
            else pa.large_list(pa.string())
            if kind == "large_list"
            else pa.list_(pa.string())
        )
        array = pa.array([[value, value] for value in values], type=data_type)
    table = pa.table(
        {
            "id": pa.array(np.arange(256, dtype=np.int64) + 9007199254740992),
            "time": pa.array(np.arange(256, dtype=np.int64), type=pa.timestamp("ns")),
            "value": array,
        }
    )
    path = tmp_path / "typed-dictionary.parquet"
    pq.write_table(
        table, native_filesystem_path(path), use_dictionary=True, row_group_size=128
    )
    native = pq.read_table(native_filesystem_path(path))
    before = dataset_physical.dataset_file_identity(path)
    delivered = 0
    with closing_iterator(
        dataset_physical.iter_dataset_part_batches(
            path, expected_identity=before, batch_rows=128
        )
    ) as batches:
        for batch in batches:
            pd.testing.assert_frame_equal(
                batch.to_pandas(), native.slice(delivered, batch.num_rows).to_pandas()
            )
            assert batch.schema.remove_metadata() == native.schema.remove_metadata()
            delivered += batch.num_rows
    assert delivered == 256
    assert dataset_physical.dataset_file_identity(path) == before


@pytest.mark.parametrize(
    "groups,filters,expected_groups",
    [
        ((0,), None, (0,)),
        ((1,), None, (1,)),
        (None, None, (0, 1)),
        ((1, 0), None, (1, 0)),
        (None, [("value", "<", "x")], (0,)),
        (None, [("value", ">=", "x")], (1,)),
    ],
)
def test_mixed_row_group_encoding_only_changes_selected_native_layout(
    monkeypatch, groups, filters, expected_groups
):
    path = Path(__file__).parent / "fixtures" / "mixed_dictionary_row_groups.parquet"
    constructor = pq.ParquetFile
    with constructor(path) as parquet:
        assert parquet.metadata.row_group(0).column(0).encodings == ("PLAIN",)
        assert parquet.metadata.row_group(1).column(0).encodings == (
            "PLAIN_DICTIONARY",
        )
    layouts = []
    constructors = []
    pulls = []

    class ObservedParquet:
        def __init__(self, *args, **kwargs):
            constructors.append(True)
            self.inner = constructor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            layouts.append(
                pa.types.is_dictionary(self.inner.schema_arrow.field("value").type)
            )
            for batch in self.inner.iter_batches(**kwargs):
                pulls.append(batch.num_rows)
                yield batch

    monkeypatch.setattr(pq, "ParquetFile", ObservedParquet)
    delivered = 0
    expected = (
        f"{index:08d}" * 16 if group == 0 else "x" * 128
        for group in expected_groups
        for index in range(2048)
    )
    with closing_iterator(
        dataset_physical.iter_dataset_part_batches(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            row_groups=groups,
            parquet_filters=filters,
            batch_rows=257,
        )
    ) as batches:
        for batch in batches:
            assert batch.column(0).to_pylist() == [
                next(expected) for _index in range(batch.num_rows)
            ]
            delivered += batch.num_rows
    assert delivered == 2048 * len(expected_groups)
    assert len(constructors) == 1
    assert max(pulls) <= 257


@pytest.mark.parametrize("primary_failure", [False, True])
def test_dictionary_reader_closes_same_handle_and_native_iterator(
    tmp_path, monkeypatch, primary_failure
):
    path = tmp_path / "cancelled-dictionary.parquet"
    pq.write_table(
        pa.table({"value": ["x" * (32 * 1024)] * 256}), native_filesystem_path(path)
    )
    original_bytes = filesystem_read_bytes(path)
    open_file = dataset_physical.open_filesystem_file
    constructor = pq.ParquetFile
    handles = []
    iterator_closed = []
    reader_closed = []
    primary = KeyboardInterrupt("dictionary_consumer_primary")

    def observed_open(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        handles.append(handle)
        return handle

    class ObservedParquet:
        def __init__(self, *args, **kwargs):
            self.inner = constructor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            source = self.inner.iter_batches(**kwargs)
            try:
                yield from source
            finally:
                source.close()
                iterator_closed.append(True)

        def close(self):
            self.inner.close()
            reader_closed.append(True)
            if primary_failure:
                raise RuntimeError("dictionary_close_secondary")

    monkeypatch.setattr(dataset_physical, "open_filesystem_file", observed_open)
    monkeypatch.setattr(pq, "ParquetFile", ObservedParquet)
    batches = dataset_physical.iter_dataset_part_batches(
        path, expected_identity=None, batch_rows=128
    )
    first = next(batches)
    assert first.num_rows > 0
    first = None
    if primary_failure:
        with pytest.raises(KeyboardInterrupt) as raised:
            batches.throw(primary)
        assert raised.value is primary
        assert any(
            "iterator close also failed: RuntimeError" in note
            for note in primary.__notes__
        )
    else:
        batches.close()
    assert len(handles) == 1 and handles[0].closed
    assert iterator_closed == [True]
    assert reader_closed == [True]
    assert filesystem_read_bytes(path) == original_bytes


@pytest.mark.parametrize(
    "values",
    (
        tuple("x" * 32 for _ in range(512)),
        tuple([*("x" * 32 for _ in range(511)), "y" * 8000]),
    ),
    ids=("uniform", "skewed"),
)
def test_arrow_batch_converter_observes_each_row_once(
    values: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    row_ids = tuple(range(len(values)))
    batch = pa.record_batch(
        [pa.array(row_ids), pa.array(values)], names=["row_id", "value"]
    )
    conversions = 0
    persisted_rows: list[int] = []
    logical_rows: list[int] = []
    original = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch, **kwargs) -> pd.DataFrame:
        nonlocal conversions
        conversions += 1
        return original(arrow_slice, **kwargs)

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", record_conversion)
    output = tuple(
        dataset_reading._arrow_batch_to_bounded_pandas(
            batch,
            dtype_overrides={"value": "string"},
            persisted_frame_observer=lambda frame: persisted_rows.extend(
                frame["row_id"].tolist()
            ),
            batch_observer=lambda _batch, frame: logical_rows.extend(
                frame["row_id"].tolist()
            ),
        )
    )
    assert conversions == len(output) == 1
    assert persisted_rows == list(row_ids)
    assert logical_rows == list(row_ids)
    assert tuple(row for frame in output for row in frame["row_id"]) == row_ids
    assert all(str(frame["value"].dtype) == "string" for frame in output)


def test_chunk_iterator_verifies_session_on_close_empty_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    error_verifications = 0
    root = make_dataset_artifact_root(tmp_path, "iterator_final_verify")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("iterator_final_verify"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    calls = 0
    original_verify = dataset_reading.DatasetSession.verify

    def record_verify(current: dataset_reading.DatasetSession) -> None:
        nonlocal calls, error_verifications
        if current.root == root:
            calls += 1
        original_verify(current)
        if (
            isinstance(sys.exception(), RuntimeError)
            and str(sys.exception()) == "injected_read_failure"
        ):
            error_verifications += 1
            raise ValueError("injected_final_identity_failure")

    monkeypatch.setattr(dataset_reading.DatasetSession, "verify", record_verify)
    iterator = dataset_reading.iter_dataset_chunks(session, batch_rows=2048)
    next(iterator)
    before_close = calls
    iterator.close()
    assert calls > before_close
    empty_session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=TimeWindow("2099-01-01", "2099-01-02")
    )
    before_empty = calls
    assert (
        len(list(dataset_reading.iter_dataset_chunks(empty_session, batch_rows=2048)))
        == 1
    )
    assert calls > before_empty

    def fail_batches(*_args, **_kwargs):
        raise RuntimeError("injected_read_failure")
        yield

    monkeypatch.setattr(
        dataset_reading, "iter_declared_parquet_member_batches", fail_batches
    )
    error_session = dataset_reading.open_dataset_session(
        root, columns=None, time_window=None
    )
    before_error = calls
    with pytest.raises(RuntimeError, match="injected_read_failure") as caught:
        list(dataset_reading.iter_dataset_chunks(error_session, batch_rows=2048))
    assert calls > before_error
    assert error_verifications >= 1
    assert "dataset verification also failed: ValueError" in caught.value.__notes__


@pytest.mark.parametrize(
    "arrays",
    [
        [pa.nulls(1) for _ in range(500)],
        [pa.array([b"x"], type=pa.binary(1)) for _ in range(500)],
        [pa.array([Decimal("1.23")], type=pa.decimal128(10, 2)) for _ in range(500)],
        [pa.array([date(2026, 1, 1)], type=pa.date32()) for _ in range(500)],
    ],
    ids=["null", "fixed-binary", "decimal", "date"],
)
def test_wide_single_row_preserves_complete_value(
    arrays: list[pa.Array], monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = pa.record_batch(
        arrays, names=[f"column_{index}" for index in range(len(arrays))]
    )
    calls = {"count": 0}
    convert = dataset_reading._arrow_slice_to_pandas

    def record_conversion(arrow_slice: pa.RecordBatch, **kwargs) -> pd.DataFrame:
        calls["count"] += 1
        return convert(arrow_slice, **kwargs)

    monkeypatch.setattr(dataset_reading, "_arrow_slice_to_pandas", record_conversion)
    chunks = tuple(dataset_reading._arrow_batch_to_bounded_pandas(batch, batch_rows=8))
    assert calls["count"] == 1
    assert len(chunks) == 1
    pd.testing.assert_frame_equal(chunks[0], batch.to_pandas(use_threads=False))


def test_materialized_read_preserves_values(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = pd.DataFrame({"value": ["payload"]})
    monkeypatch.setattr(
        dataset_reading,
        "iter_dataset_chunks",
        lambda *_args, **_kwargs: iter((expected,)),
    )
    context = SimpleNamespace(
        logical_columns=tuple(expected.columns),
        logical_dtypes=logical_dtype_payload_from_frame(expected),
    )
    observed = dataset_reading.materialize_dataset_session(context)
    pd.testing.assert_frame_equal(observed, expected)


def test_chunked_and_materialized_reads_share_nullable_numeric_dtype(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "nullable_numeric_dtype")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["value"] = pd.Series([1], dtype="Int64")
    second["value"] = pd.Series([pd.NA], dtype="Int64")
    write_dataset(
        root,
        (first, second),
        spec=make_processed_dataset_spec("nullable_numeric_dtype"),
    )
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)
    assert {str(chunk["value"].dtype) for chunk in chunks} == {
        str(materialized["value"].dtype)
    }


@pytest.mark.parametrize(
    "reverse", (False, True), ids=("native-first", "nullable-first")
)
@pytest.mark.parametrize(
    ("native_dtype", "nullable_dtype"), (("float32", "Float32"), ("float64", "Float64"))
)
def test_nullable_float_plan_is_chunk_order_independent(
    tmp_path: Path, reverse: bool, native_dtype: str, nullable_dtype: str
) -> None:
    root = make_dataset_artifact_root(
        tmp_path, f"nullable_{nullable_dtype.lower()}_{reverse}"
    )
    base = make_default_dataset_frame().iloc[:2].copy()
    native = base.iloc[:1].copy()
    nullable = base.iloc[1:].copy()
    native["value"] = pd.Series([1.5], index=native.index, dtype=native_dtype)
    nullable["value"] = pd.Series([pd.NA], index=nullable.index, dtype=nullable_dtype)
    chunks = (nullable, native) if reverse else (native, nullable)
    write_dataset(
        root,
        chunks,
        spec=make_processed_dataset_spec(
            f"nullable_{nullable_dtype.lower()}_{reverse}"
        ),
    )
    materialized = read_dataset(root)
    chunked = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    assert str(materialized["value"].dtype) == nullable_dtype
    assert {str(chunk["value"].dtype) for chunk in chunked} == {nullable_dtype}


def test_nullable_uint64_read_and_fingerprint_preserve_large_integer_identity(
    tmp_path: Path,
) -> None:
    left_root = make_dataset_artifact_root(tmp_path / "left", "nullable_uint64")
    right_root = make_dataset_artifact_root(tmp_path / "right", "nullable_uint64")
    spec = make_processed_dataset_spec("nullable_uint64")
    left = make_default_dataset_frame().iloc[:2].copy()
    right = left.copy()
    left["value"] = pd.array([2**63, pd.NA], dtype="UInt64")
    right["value"] = pd.array([2**63 + 1, pd.NA], dtype="UInt64")
    write_dataset(left_root, left, spec=spec)
    write_dataset(right_root, right, spec=spec)
    left_read = read_dataset(left_root)
    right_read = read_dataset(right_root)
    comparison = compare_datasets(left_root, right_root)
    assert str(left_read["value"].dtype) == "UInt64"
    assert str(right_read["value"].dtype) == "UInt64"
    assert int(left_read["value"].dropna().iloc[0]) == 2**63
    assert int(right_read["value"].dropna().iloc[0]) == 2**63 + 1
    assert comparison.status == "CONTENT_MISMATCH"


def test_nullable_boolean_preserves_extension_dtype_across_read_modes(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "nullable_boolean")
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["value"] = pd.array([True, pd.NA], dtype="boolean")
    write_dataset(root, frame, spec=make_processed_dataset_spec("nullable_boolean"))
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)
    assert {str(chunk["value"].dtype) for chunk in chunks} == {"boolean"}
    assert str(materialized["value"].dtype) == "boolean"


@pytest.mark.parametrize(
    "mode", [DatasetReadMode.CHUNKED, DatasetReadMode.MATERIALIZED]
)
def test_read_dataset_rejects_empty_column_projection(
    tmp_path: Path, mode: DatasetReadMode
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"empty_projection_{mode.value}")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec(f"empty_projection_{mode.value}"),
    )
    with pytest.raises(ValueError, match="dataset_columns_empty"):
        result = read_dataset(root, columns=[], mode=mode)
        if mode is DatasetReadMode.CHUNKED:
            list(result)


def test_materialized_read_unions_categorical_dictionaries_across_parts(
    tmp_path: Path,
) -> None:
    root = make_dataset_artifact_root(tmp_path, "categorical_parts")
    first = make_default_dataset_frame().iloc[:1].copy()
    second = make_default_dataset_frame().iloc[1:2].copy()
    first["asset_id"] = first["asset_id"].astype("category")
    second["asset_id"] = second["asset_id"].astype("category")
    write_dataset(
        root, (first, second), spec=make_processed_dataset_spec("categorical_parts")
    )
    chunks = list(read_dataset(root, mode=DatasetReadMode.CHUNKED))
    materialized = read_dataset(root)
    assert {str(chunk["asset_id"].dtype) for chunk in chunks} == {"category"}
    assert str(materialized["asset_id"].dtype) == "category"
    assert set(materialized["asset_id"].cat.categories) == set(
        pd.concat((first["asset_id"].astype(str), second["asset_id"].astype(str)))
    )
    assert all(
        chunk["asset_id"].dtype == materialized["asset_id"].dtype for chunk in chunks
    )


@pytest.mark.parametrize(
    "mode", (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED)
)
def test_reads_reject_nonnull_category_outside_declared_domain(
    tmp_path: Path, mode: DatasetReadMode
) -> None:
    root = make_dataset_artifact_root(tmp_path, f"forged_read_domain_{mode.value}")
    frame = make_default_dataset_frame().iloc[:1].copy()
    known = str(frame["asset_id"].iloc[0])
    frame["asset_id"] = pd.Categorical([known], categories=[known])
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(f"forged_read_domain_{mode.value}"),
    )
    part = next(filesystem_glob(root / "data", "**/part-*.parquet"))
    original = pq.ParquetFile(native_filesystem_path(part)).read()
    asset_index = original.schema.get_field_index("asset_id")
    asset_type = original.schema.field(asset_index).type
    forged_asset = pa.DictionaryArray.from_arrays(
        pa.array([0], type=asset_type.index_type),
        pa.array(["unknown"], type=asset_type.value_type),
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                forged_asset if index == asset_index else original.column(index)
                for index in range(original.num_columns)
            ],
            schema=original.schema.remove_metadata(),
        ).replace_schema_metadata(original.schema.metadata),
        native_filesystem_path(part),
        compression="zstd",
    )
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["file_size_bytes"] = sum(
        filesystem_stat(path).st_size
        for path in filesystem_glob(root / "data", "**/part-*.parquet")
    )
    manifest["content_hash"] = fixture_dataset_content_hash(root)
    filesystem_write_text(manifest_path, json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_categorical_domain_mismatch"):
        result = read_dataset(root, mode=mode)
        if mode is DatasetReadMode.CHUNKED:
            tuple(result)


@pytest.mark.parametrize("ordered", [False, True])
def test_fingerprint_preserves_content_identity_across_unused_categories(
    tmp_path: Path, ordered: bool
) -> None:
    left_root = make_dataset_artifact_root(tmp_path / "left", "category_identity")
    right_root = make_dataset_artifact_root(tmp_path / "right", "category_identity")
    left = make_default_dataset_frame().iloc[:1].copy()
    right = left.copy()
    value = str(left["asset_id"].iloc[0])
    left["asset_id"] = pd.Categorical(
        left["asset_id"], categories=[value, "unused_left"], ordered=ordered
    )
    right["asset_id"] = pd.Categorical(
        right["asset_id"], categories=[value, "unused_right"], ordered=ordered
    )
    spec = make_processed_dataset_spec("category_identity")
    write_dataset(left_root, left, spec=spec)
    write_dataset(right_root, right, spec=spec)
    comparison = compare_datasets(left_root, right_root)
    assert comparison.status == ("SCHEMA_MISMATCH" if ordered else "MATCH")


@pytest.mark.parametrize(
    ("column", "left_dtype", "right_dtype"),
    [
        ("value", "int64", "Int64"),
        ("value", "float64", "Float64"),
        ("value", "bool", "boolean"),
        ("asset_id", "object", "string"),
    ],
)
def test_comparison_projects_logical_dtypes_and_always_includes_primary_key(
    tmp_path: Path, column: str, left_dtype: str, right_dtype: str
) -> None:
    roots = [
        make_dataset_artifact_root(tmp_path / side, "dtypes")
        for side in ("left", "right")
    ]
    spec = make_processed_dataset_spec("dtypes")
    for root, dtype in zip(roots, (left_dtype, right_dtype), strict=True):
        frame = make_default_dataset_frame(values=[1, 0, 1]).astype({column: dtype})
        write_dataset(root, frame, spec=spec)
        assert str(read_dataset(root)[column].dtype) == dtype
    assert compare_datasets(*roots).status == "SCHEMA_MISMATCH"
    projected = compare_datasets(
        *roots, columns=("value",) if column == "asset_id" else ("asset_id",)
    )
    assert projected.status == ("SCHEMA_MISMATCH" if column == "asset_id" else "MATCH")


@pytest.mark.parametrize("mode", ("materialized", "chunked", "writer"))
@pytest.mark.parametrize("timestamp", [False, True])
def test_physical_values_cannot_be_reinterpreted_by_forged_logical_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, timestamp: bool
) -> None:
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.storage.datasets import writing
    from quant_research_platform.storage.datasets.schema import (
        logical_dtype_metadata_bytes,
    )

    spec = make_processed_dataset_spec(
        "logical_forgery",
        value_role=ColumnRole.CONTEXT if timestamp else ColumnRole.FACTOR,
    )
    frame = make_default_dataset_frame().iloc[:1].copy().assign(value=1.5)
    if timestamp:
        frame["value"] = pd.Series(
            [pd.Timestamp("2020-01-01") + pd.Timedelta(1, unit="ns")],
            dtype="datetime64[ns]",
        )
    original_root = make_dataset_artifact_root(tmp_path / "original", spec.data_id)
    root = make_dataset_artifact_root(tmp_path / "candidate", spec.data_id)
    write_dataset(original_root, frame, spec=spec)
    original_bytes = _file_bytes(original_root)

    def forge(table):
        metadata = dict(table.schema.metadata)
        logical = json.loads(metadata[b"qrp.logical-dtypes"])
        for descriptor in logical["columns"]:
            if descriptor["name"] == "value":
                descriptor.update(
                    kind="datetime" if timestamp else "numpy_integer",
                    dtype="datetime64[s]" if timestamp else "int64",
                )
        metadata[b"qrp.logical-dtypes"] = logical_dtype_metadata_bytes(logical)
        return (table.replace_schema_metadata(metadata), logical)

    if mode == "writer":
        original_write = writing.write_parquet_file

        def forged_write(table, path):
            forged, _logical = forge(table)
            return original_write(forged, path)

        monkeypatch.setattr(writing, "write_parquet_file", forged_write)
        with pytest.raises(ValueError, match="dataset_logical_dtype_mismatch"):
            write_dataset(root, frame, spec=spec)
        assert not filesystem_exists(root)
        assert not tuple(filesystem_iterdir(root.parent))
    else:
        write_dataset(root, frame, spec=spec)
        part = next(filesystem_glob(root / "data", "**/*.parquet"))
        with pq.ParquetFile(native_filesystem_path(part)) as parquet:
            forged, logical = forge(parquet.read())
        pq.write_table(forged, native_filesystem_path(part), compression="zstd")
        manifest_path = root / "_manifest" / "manifest.json"
        manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
        manifest["logical_dtypes"] = logical
        manifest["logical_dtype_hash"] = canonical_json_hash(logical)
        manifest["file_size_bytes"] = filesystem_stat(part).st_size
        manifest["content_hash"] = fixture_dataset_content_hash(root)
        filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="dataset_logical_dtype_mismatch"):
            result = dataset_reading.read_dataset(root, mode=DatasetReadMode(mode))
            if mode == "chunked":
                tuple(result)
    assert _file_bytes(original_root) == original_bytes


@pytest.mark.parametrize(
    "mode", (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED)
)
def test_timestamp_seconds_survive_native_parquet_unit_widening(
    tmp_path: Path, mode
) -> None:
    from quant_research_platform.contracts.columns import ColumnRole

    root = make_dataset_artifact_root(tmp_path, "seconds")
    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["timestamp"] = frame.timestamp.astype("datetime64[ns]")
    frame["value"] = pd.Series(
        [pd.Timestamp("2020-01-01"), pd.NaT], dtype="datetime64[s]"
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("seconds", value_role=ColumnRole.CONTEXT),
    )
    result = read_dataset(root, mode=mode)
    if mode is DatasetReadMode.CHUNKED:
        result = pd.concat(tuple(result), ignore_index=True)
    keys = ["market_id", "asset_id", "timestamp"]
    pd.testing.assert_frame_equal(
        result.sort_values(keys).reset_index(drop=True),
        frame.sort_values(keys).reset_index(drop=True),
        check_exact=True,
    )


@pytest.mark.parametrize("dtype", ("Int64", "UInt64"))
def test_nullable_integer_values_and_numeric_keys_remain_exact_with_nulls(
    tmp_path: Path, dtype: str
) -> None:
    base = 2**53 + 1 if dtype == "Int64" else 2**63 + 1
    root = make_dataset_artifact_root(tmp_path, "nullable_precision")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": pd.Series([base, base + 1], dtype=dtype),
            "timestamp": pd.Series(
                pd.to_datetime(["2020-01-01"] * 2), dtype="datetime64[ns]"
            ),
            "value": pd.Series([base, pd.NA], dtype=dtype),
        }
    )
    write_dataset(root, frame, spec=make_processed_dataset_spec("nullable_precision"))
    assert len(tuple(filesystem_glob(root / "data", "**/*.parquet"))) == 1
    pd.testing.assert_frame_equal(read_dataset(root), frame, check_exact=True)
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    for target in (1, 1024**2):
        loaded = pd.concat(
            tuple(dataset_reading.iter_dataset_chunks(session, batch_rows=target)),
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(loaded, frame, check_exact=True)


@pytest.mark.parametrize(
    "mode", (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED)
)
def test_nonnullable_boolean_rejects_physical_null_values(
    tmp_path: Path, mode: DatasetReadMode
) -> None:
    root = make_dataset_artifact_root(tmp_path, "boolean_null")
    frame = make_default_dataset_frame().iloc[:1].assign(value=True)
    write_dataset(root, frame, spec=make_processed_dataset_spec("boolean_null"))
    part = next(filesystem_glob(root / "data", "**/*.parquet"))
    with pq.ParquetFile(native_filesystem_path(part)) as parquet:
        table = parquet.read()
    table = table.set_column(
        table.schema.get_field_index("value"),
        table.schema.field("value"),
        pa.array([None], type=pa.bool_()),
    )
    pq.write_table(table, native_filesystem_path(part), compression="zstd")
    manifest_path = root / "_manifest" / "manifest.json"
    manifest = json.loads(filesystem_read_text(manifest_path, encoding="utf-8"))
    manifest["file_size_bytes"] = filesystem_stat(part).st_size
    manifest["content_hash"] = fixture_dataset_content_hash(root)
    filesystem_write_text(manifest_path, json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_nonnullable_value_missing"):
        result = read_dataset(root, mode=mode)
        if mode is DatasetReadMode.CHUNKED:
            tuple(result)


def test_fingerprint_content_hash_preserves_registered_formula(tmp_path: Path) -> None:
    root = make_dataset_artifact_root(tmp_path, "fingerprint_formula")
    frame = make_default_dataset_frame()
    spec = make_processed_dataset_spec("fingerprint_formula")
    write_dataset(root, frame, spec=spec)
    fingerprint = fingerprint_dataset(root)
    hash_columns = tuple(dict.fromkeys((*spec.primary_key, *spec.required_columns)))
    reference_readback = pd.concat(
        tuple(
            pd.read_parquet(native_filesystem_path(path))
            for path in sorted(filesystem_glob(root / "data", "**/part-*.parquet"))
        ),
        ignore_index=True,
    )
    hashes = pd.util.hash_pandas_object(
        reference_readback.loc[:, list(hash_columns)], index=False
    ).to_numpy("uint64")
    expected = canonical_json_hash(
        {
            "row_hash_sum": int(hashes.sum(dtype=np.uint64)),
            "row_hash_xor": int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0))),
            "row_count": len(frame),
            "columns": list(spec.required_columns),
            "hash_columns": list(hash_columns),
        }
    )
    assert fingerprint.content_hash == expected


@pytest.mark.parametrize(
    ("use_dictionary", "failure"),
    (
        (False, "native_close"),
        (True, "native_close"),
        (False, "identity"),
        (True, "identity"),
    ),
)
@pytest.mark.parametrize("reader_kind", ("physical", "declared", "prediction"))
@pytest.mark.parametrize("consumer_primary", (False, True))
def test_parquet_early_close_propagates_cleanup_and_identity_failures(
    tmp_path, monkeypatch, use_dictionary, failure, consumer_primary, reader_kind
):
    original = tmp_path / "original.parquet"
    path = tmp_path / "read.parquet"
    if reader_kind == "prediction":
        from quant_research_platform.artifacts.outputs.prediction import (
            _merge_family_predictions,
        )
        from quant_research_platform.artifacts.outputs.prediction_validation import (
            validate_family_prediction_frame,
        )
        from quant_research_platform.contracts.predictions import (
            A_SHARE_PREDICTION_PRIMARY_KEY,
            RAW_PREDICTION_COLUMN,
            prediction_output_columns,
        )

        columns = prediction_output_columns(
            "a_share_oos", RAW_PREDICTION_COLUMN, A_SHARE_PREDICTION_PRIMARY_KEY
        )
        metadata = {
            "quarterly_models": [
                {
                    "quarter_id": "2020Q1",
                    "family_id": family,
                    "model_id": f"{family}_2020Q1",
                    "prediction_window": {"start": "2020-01-02", "end": "2020-03-31"},
                }
                for family in ("linear_regression", "ridge")
            ]
        }
        prediction_paths = []
        prediction_originals = {}
        for model in metadata["quarterly_models"]:
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2020-01-02", "2020-01-03", "2020-01-06"]
                    ),
                    "asset_id": ["000001.SZ"] * 3,
                    "family_id": model["family_id"],
                    "quarter_id": model["quarter_id"],
                    "model_id": model["model_id"],
                    "raw_prediction": [0.1, 0.2, 0.3],
                    "alpha_score": [0.01, 0.02, 0.03],
                },
                columns=columns,
            )
            validate_family_prediction_frame(frame, metadata)
            assert not frame.duplicated(list(A_SHARE_PREDICTION_PRIMARY_KEY)).any()
            member = path if not prediction_paths else tmp_path / "other.parquet"
            pq.write_table(
                pa.Table.from_pandas(frame, preserve_index=False),
                native_filesystem_path(member),
                use_dictionary=use_dictionary,
            )
            prediction_paths.append((model["quarter_id"], model["family_id"], member))
            prediction_originals[member] = filesystem_read_bytes(member)
        filesystem_write_bytes(original, prediction_originals[path])
    else:
        pq.write_table(
            pa.table(
                {
                    "value": ["x" * (32 * 1024)] * 256,
                    "timestamp": pd.date_range("2020-01-01", periods=256),
                    "asset_id": ["asset"] * 256,
                }
            ),
            native_filesystem_path(original),
            use_dictionary=use_dictionary,
            row_group_size=1,
        )
        filesystem_write_bytes(path, filesystem_read_bytes(original))
    original_bytes = filesystem_read_bytes(original)
    open_file = dataset_physical.open_filesystem_file
    constructor = pq.ParquetFile
    handles, reader_closed, iterator_closed = ([], [], [])

    def observed_open(*args, **kwargs):
        handle = open_file(*args, **kwargs)
        handles.append(handle)
        return handle

    class ObservedParquet:
        def __init__(self, *args, **kwargs):
            self.inner = constructor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def iter_batches(self, **kwargs):
            source = self.inner.iter_batches(**kwargs)
            try:
                yield from source
            finally:
                source.close()
                iterator_closed.append(True)

        def close(self):
            self.inner.close()
            reader_closed.append(True)
            if failure == "native_close":
                raise OSError("parquet_native_close")

    monkeypatch.setattr(dataset_physical, "open_filesystem_file", observed_open)
    monkeypatch.setattr(pq, "ParquetFile", ObservedParquet)
    primary = KeyboardInterrupt("parquet_consumer_primary")
    if reader_kind == "prediction":
        batches = _merge_family_predictions(prediction_paths, columns)
    else:
        reader = (
            dataset_physical.iter_dataset_part_batches
            if reader_kind == "physical"
            else dataset_reading.iter_declared_parquet_member_batches
        )
        batches = reader(
            path,
            expected_identity=dataset_physical.dataset_file_identity(path),
            batch_rows=128,
        )
    assert len(next(batches)) > 0
    readers = 2 if reader_kind == "prediction" else 1
    assert len(handles) == readers and all(not handle.closed for handle in handles)
    assert not reader_closed and (not iterator_closed)
    if use_dictionary:
        pass
    if failure == "identity":
        before = filesystem_stat(path)
        with open_filesystem_file(path, "ab") as handle:
            handle.write(b"x")
        assert filesystem_stat(path).st_size == before.st_size + 1
    error = (
        KeyboardInterrupt
        if consumer_primary
        else ValueError
        if failure == "identity"
        else OSError
    )
    with pytest.raises(error) as caught:
        if consumer_primary:
            with closing_iterator(batches):
                raise primary
        else:
            batches.close()
    if consumer_primary:
        assert caught.value is primary
        secondary = "ValueError" if failure == "identity" else "OSError"
        assert any(secondary in note for note in getattr(primary, "__notes__", ()))
    elif failure == "identity":
        assert str(caught.value) == "dataset_file_identity_changed"
    assert len(handles) == readers and all(handle.closed for handle in handles)
    assert reader_closed == [True] * readers
    assert iterator_closed == [True] * readers
    assert filesystem_read_bytes(original) == original_bytes
    if failure != "identity":
        assert filesystem_read_bytes(path) == original_bytes
    if reader_kind == "prediction":
        for member, payload in prediction_originals.items():
            if member != path or failure != "identity":
                assert filesystem_read_bytes(member) == payload


@pytest.mark.parametrize("observer_kind", ("persisted", "batch"))
def test_declared_reader_preserves_observer_generator_exit(
    tmp_path, monkeypatch, observer_kind
):
    path = tmp_path / "observer.parquet"
    pq.write_table(pa.table({"value": [1, 2, 3]}), native_filesystem_path(path))
    original = filesystem_read_bytes(path)
    constructor = pq.ParquetFile
    closed = []

    class ObservedParquet:
        def __init__(self, *args, **kwargs):
            self.inner = constructor(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def close(self):
            self.inner.close()
            closed.append(True)
            raise OSError("observer_native_close")

    primary = GeneratorExit("observer_primary")

    def observer(*args):
        raise primary

    monkeypatch.setattr(pq, "ParquetFile", ObservedParquet)
    frames = dataset_reading.iter_declared_parquet_member_batches(
        path,
        expected_identity=dataset_physical.dataset_file_identity(path),
        persisted_frame_observer=observer if observer_kind == "persisted" else None,
        batch_observer=observer if observer_kind == "batch" else None,
        batch_rows=128,
    )
    with pytest.raises(GeneratorExit) as caught:
        next(frames)
    assert caught.value is primary
    assert any("OSError" in note for note in getattr(primary, "__notes__", ()))
    assert closed == [True]
    assert filesystem_read_bytes(path) == original


@pytest.mark.parametrize("unit", ("ms", "us", "ns"))
@pytest.mark.parametrize("offset_ns", (0, 1, 1000, 1000000))
def test_timestamp_equality_keeps_predicate_precision(tmp_path, unit, offset_ns):
    from quant_research_platform.contracts.columns import ColumnRole

    root = make_dataset_artifact_root(tmp_path, "timestamp_equality")
    frame = make_default_dataset_frame().iloc[:2].copy()
    origin = pd.Timestamp("2020-01-01").as_unit("ns")
    frame["value"] = pd.Series(
        [origin, origin + pd.Timedelta(milliseconds=1)], dtype=f"datetime64[{unit}]"
    )
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(
            "timestamp_equality", value_role=ColumnRole.CONTEXT
        ),
    )
    predicate = origin + pd.Timedelta(offset_ns, unit="ns")
    session = dataset_reading.open_dataset_session(
        root,
        columns=("value",),
        time_window=None,
        row_filters=(("value", "==", predicate),),
    )
    assert session.arrow_schema.field("value").type == pa.timestamp(unit)
    rows = 0
    with closing_iterator(
        dataset_reading.iter_dataset_chunks(session, batch_rows=2048)
    ) as chunks:
        for result in chunks:
            rows += len(result)
            assert (result["value"] == predicate).all()
    assert rows == (1 if offset_ns in (0, 1000000) else 0)


@pytest.mark.parametrize("dtype", ["object", "float64"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_numeric_infinity_is_rejected_independently_of_logical_dtype(
    tmp_path, dtype, value
):
    from quant_research_platform.contracts.columns import ColumnRole

    frame = make_default_dataset_frame().iloc[:1].copy()
    frame["value"] = pd.Series([value], index=frame.index, dtype=dtype)
    root = make_dataset_artifact_root(tmp_path, "nonfinite")
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        write_dataset(
            root,
            frame,
            spec=make_processed_dataset_spec(
                "nonfinite", value_role=ColumnRole.CONTEXT
            ),
        )
    assert not root.exists()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("values", [[2**53 + 1, None], [3, None], [2**53 + 1, 5]])
def test_public_nested_integer_null_readback_is_exact(tmp_path, nested, values):
    from quant_research_platform.contracts.columns import ColumnRole

    frame = make_default_dataset_frame().iloc[:1].copy()
    frame["value"] = pd.Series(
        [{"items": values} if nested else values], index=frame.index, dtype=object
    )
    root = make_dataset_artifact_root(tmp_path, "nested_exact")
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec("nested_exact", value_role=ColumnRole.CONTEXT),
    )
    physical = (
        pq.read_table(next((root / "data").rglob("*.parquet")))
        .column("value")[0]
        .as_py()
    )
    expected = physical["items"] if nested else physical
    assert expected == values
    for mode in (DatasetReadMode.CHUNKED, DatasetReadMode.MATERIALIZED):
        result = read_dataset(root, mode=mode)
        output = (
            pd.concat(list(result), ignore_index=True)
            if mode is DatasetReadMode.CHUNKED
            else result
        )
        observed = output["value"].iloc[0]
        observed = observed["items"] if nested else observed
        assert int(observed[0]) == values[0]
        assert (
            pd.isna(observed[1]) if values[1] is None else int(observed[1]) == values[1]
        )


@pytest.mark.parametrize("zone", ["UTC", "Asia/Shanghai", "America/New_York"])
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_materialized_timezone_context_preserves_epoch_unit_and_nat(
    tmp_path, zone, unit
):
    from quant_research_platform.contracts.columns import ColumnRole

    frame = make_default_dataset_frame()
    frame["value"] = pd.Series(
        pd.to_datetime(["2024-01-15", "2024-07-15", None])
        .tz_localize(zone)
        .as_unit(unit)
    )
    root = make_dataset_artifact_root(tmp_path, "timezone_context")
    write_dataset(
        root,
        frame,
        spec=make_processed_dataset_spec(
            "timezone_context", value_role=ColumnRole.CONTEXT
        ),
    )
    expected = frame.sort_values(["market_id", "asset_id", "timestamp"])[
        "value"
    ].reset_index(drop=True)
    for mode in (DatasetReadMode.MATERIALIZED, DatasetReadMode.CHUNKED):
        result = read_dataset(root, mode=mode)
        result = (
            pd.concat(list(result), ignore_index=True)
            if mode is DatasetReadMode.CHUNKED
            else result
        )
        result = result.sort_values(["market_id", "asset_id", "timestamp"])
        assert result["value"].dtype == expected.dtype
        np.testing.assert_array_equal(result["value"].array.asi8, expected.array.asi8)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("parts", [1, 2])
def test_categorical_single_and_multiple_parts_share_the_same_normal_form(
    tmp_path, ordered, parts
):
    from quant_research_platform.contracts.columns import ColumnRole

    frame = make_default_dataset_frame().iloc[:2].copy()
    frame["value"] = pd.Categorical(
        ["a", "b"], categories=["b", "a", "unused"], ordered=ordered
    )
    roots = [
        make_dataset_artifact_root(tmp_path / str(index), "categories")
        for index in range(2)
    ]
    for index, root in enumerate(roots):
        current = frame.copy()
        if index and (not ordered):
            current["value"] = current["value"].cat.reorder_categories(
                ["a", "b", "unused"]
            )
        data = current if parts == 1 else (current.iloc[:1], current.iloc[1:])
        write_dataset(
            root,
            data,
            spec=make_processed_dataset_spec(
                "categories", value_role=ColumnRole.CONTEXT
            ),
        )
        actual = read_dataset(root)
        assert actual["value"].cat.ordered is ordered
        if ordered:
            assert list(actual["value"].cat.categories) == ["b", "a", "unused"]
    assert compare_datasets(*roots).matches


@pytest.mark.parametrize("kind", ["timestamp", "duration", "time", "float"])
def test_membership_does_not_round_queries_into_matches(kind):
    from datetime import time

    if kind == "timestamp":
        dtype, stored = (pa.timestamp("ms"), pd.Timestamp("2024-01-01"))
        query = stored + pd.Timedelta(1, "ns")
    elif kind == "duration":
        dtype, stored = (pa.duration("ms"), pd.Timedelta(1, "s"))
        query = stored + pd.Timedelta(1, "ns")
    elif kind == "time":
        dtype, stored, query = (pa.time32("s"), time(1, 2, 3), time(1, 2, 3, 1))
    else:
        dtype, stored, query = (pa.float32(), 1.0, float(np.nextafter(1.0, 2.0)))
    batch = pa.record_batch([pa.array([stored], type=dtype)], names=["value"])
    for values, expected in [
        ((query,), 0),
        ((stored,), 1),
        ((query, stored, stored), 1),
    ]:
        prepared = _prepare_in_filter(pa.field("value", dtype), values)
        result = dataset_physical._filter_arrow_batch(
            batch, [("value", "in", prepared)]
        )
        assert result.num_rows == expected


def test_comparison_rechecks_dtype_structure_after_schema_hash_collision(
    tmp_path, monkeypatch
):
    from quant_research_platform.storage.datasets import fingerprinting

    roots = [
        make_dataset_artifact_root(tmp_path / str(index), "dtype_exact")
        for index in range(2)
    ]
    for root, dtype in zip(roots, ("int64", "Int64"), strict=True):
        frame = make_default_dataset_frame().iloc[:2].copy()
        frame["value"] = frame["value"].astype(dtype)
        write_dataset(root, frame, spec=make_processed_dataset_spec("dtype_exact"))
    assert compare_datasets(*roots).status == "SCHEMA_MISMATCH"
    monkeypatch.setattr(fingerprinting, "canonical_json_hash", lambda _value: "a" * 64)
    assert compare_datasets(*roots).status == "SCHEMA_MISMATCH"


def test_explicit_generator_exit_keeps_primary_during_final_verification(
    tmp_path, monkeypatch
):
    root = make_dataset_artifact_root(tmp_path, "generator_exit")
    write_dataset(
        root,
        make_default_dataset_frame(),
        spec=make_processed_dataset_spec("generator_exit"),
    )
    session = dataset_reading.open_dataset_session(root, columns=None, time_window=None)
    original_verify = dataset_reading.DatasetSession.verify
    primary = GeneratorExit("explicit_primary")

    def verify(current):
        import sys

        if sys.exception() is primary:
            raise ValueError("secondary_identity_failure")
        original_verify(current)

    monkeypatch.setattr(dataset_reading.DatasetSession, "verify", verify)
    with pytest.raises(GeneratorExit) as caught:

        def observe(_batch, _frame):
            raise primary

        list(
            dataset_reading.iter_dataset_chunks(
                session, batch_observer=observe, batch_rows=2048
            )
        )
    assert caught.value is primary
    assert "dataset verification also failed: ValueError" in primary.__notes__
