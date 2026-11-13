"""Save producer-built dataset batches and publish a new immutable table."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.observation import DatasetSummaryObserver
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
    partition_part_path,
    relative_partition_label,
    timestamp_column_from_spec_payload,
    timestamp_primary_key_column_from_spec,
)
from quant_research_platform.storage.datasets.reading import iter_frame_chunks
from quant_research_platform.storage.datasets.schema import (
    data_spec_payload,
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
    merge_dataset_metadata,
    merge_logical_dtype_payloads,
    object_string_columns,
    physical_schema_columns,
    physical_schema_from_schema,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    filesystem_stat,
    native_filesystem_path,
)
from quant_research_platform.storage.manifests import write_qrp_json_file
from quant_research_platform.storage.parquet import (
    DEFAULT_BATCH_ROWS,
    parquet_table_writer,
)


@dataclass
class _WriteStats:
    spec: DataSpec
    data_root: Path
    file_count: int = 0
    file_size_bytes: int = 0
    partition_row_counts: dict[str, int] = field(default_factory=dict)
    _schema: list[dict[str, Any]] | None = None
    _arrow_schema: pa.Schema | None = None
    _logical_dtype_plan: dict[str, Any] | None = None
    _frame_dtype_bytes: dict[bytes, bytes] = field(default_factory=dict)
    _summary_observer: DatasetSummaryObserver = field(init=False)

    def __post_init__(self) -> None:
        self._summary_observer = DatasetSummaryObserver(
            primary_key=tuple(self.spec.primary_key),
            timestamp_column=timestamp_primary_key_column_from_spec(self.spec),
            column_roles={
                column: role.value for column, role in self.spec.column_roles.items()
            },
        )

    def prepare_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        output = frame.copy(deep=False)
        for column in self.spec.primary_key:
            if self.spec.column_roles.get(column) is ColumnRole.TIMESTAMP:
                output[column] = (
                    pd.to_datetime(output[column], utc=True)
                    .dt.tz_localize(None)
                    .astype("datetime64[ns]")
                )
        logical_dtypes = logical_dtype_payload_from_frame(output)
        return (output, logical_dtypes)

    def table_for_frame(
        self,
        frame: pd.DataFrame,
        *,
        object_string_columns: tuple[str, ...],
    ) -> pa.Table:
        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError("unsupported_dataset_logical_dtype") from exc
        for name in frame:
            dtype = frame[name].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                values_type = pa.Array.from_pandas(dtype.categories).type
                if not (
                    pa.types.is_string(values_type)
                    or pa.types.is_large_string(values_type)
                ):
                    position = table.schema.get_field_index(name)
                    table = table.set_column(
                        position, name, table.column(position).cast(values_type)
                    )
        physical_schema = _canonical_storage_schema(table.schema.remove_metadata())
        if self._arrow_schema is None:
            for name in object_string_columns:
                field = physical_schema.field(name)
                if pa.types.is_null(field.type):
                    physical_schema = physical_schema.set(
                        physical_schema.get_field_index(name),
                        field.with_type(pa.string()),
                    )
            self._arrow_schema = physical_schema
        elif physical_schema != self._arrow_schema:
            try:
                unified = pa.unify_schemas(
                    [self._arrow_schema, physical_schema], promote_options="default"
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
                raise ValueError("dataset_schema_mismatch") from error
            if any(
                not (pa.types.is_null(old.type) or old.type == new.type)
                for source in (self._arrow_schema, physical_schema)
                for old, new in zip(source, unified, strict=True)
            ):
                raise ValueError("dataset_schema_mismatch")
            self._arrow_schema = unified
            physical_schema = self._arrow_schema
        return table.cast(physical_schema, safe=True)

    def observe_file(self, path: Path, table: pa.Table) -> None:
        """Record the table just written and the closed file size."""
        if self._schema is None:
            physical_schema = physical_schema_from_schema(
                table.schema.remove_metadata()
            )
            self._schema = physical_schema
        partition = relative_partition_label(self.data_root, path)
        self.file_count += 1
        self.file_size_bytes += filesystem_stat(path).st_size
        self.partition_row_counts[partition] = (
            self.partition_row_counts.get(partition, 0) + table.num_rows
        )

    def dtype_metadata(self, payload: dict[str, Any]) -> bytes:
        identity = canonical_identity_bytes(payload)
        if identity not in self._frame_dtype_bytes:
            self.observe_logical_dtype_payload(payload)
            self._frame_dtype_bytes[identity] = logical_dtype_metadata_bytes(payload)
        return self._frame_dtype_bytes[identity]

    def observe_logical_dtype_payload(self, logical_payload: Mapping[str, Any]) -> None:
        self._logical_dtype_plan = (
            merge_logical_dtype_payloads([logical_payload])
            if self._logical_dtype_plan is None
            else merge_logical_dtype_payloads(
                [self._logical_dtype_plan, logical_payload]
            )
        )

    def payload(self) -> dict[str, Any]:
        return {
            "row_count": self._summary_observer.row_count,
            "file_count": self.file_count,
            "file_size_bytes": self.file_size_bytes,
            "columns": list(physical_schema_columns(self._schema)),
            "physical_schema": self._schema,
            "logical_dtypes": self._logical_dtype_plan,
            "partition_row_counts": dict(sorted(self.partition_row_counts.items())),
            "partition_count": len(self.partition_row_counts),
            "partitions": sorted(self.partition_row_counts),
            **self._summary_observer.summary_payload(),
        }


def write_dataset(
    dataset_root: Path,
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    spec: DataSpec,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetInspection:
    """Publish rows whose source or computation owner established their key domain."""
    spec_payload = data_spec_payload(spec)
    final_root = Path(dataset_root)
    input_chunks = iter((data,)) if isinstance(data, pd.DataFrame) else iter(data)
    with closing_iterator(ExitStack()) as input_owner:
        close = getattr(input_chunks, "close", None)
        if close is not None:
            input_owner.callback(close)
        chunks = input_chunks

        def writer(temp_root: Path) -> DatasetInspection:
            accumulator = _write_contents(
                chunks,
                temp_root / "data",
                spec=spec,
                timestamp_column=timestamp_column_from_spec_payload(spec_payload),
                batch_rows=DEFAULT_BATCH_ROWS,
            )
            write_qrp_json_file(temp_root / "spec.json", spec_payload)
            payload = accumulator.payload()
            payload["schema_id"] = "qrp-dataset-manifest"
            manifest = merge_dataset_metadata(payload, metadata)
            write_qrp_json_file(temp_root / "_manifest" / "manifest.json", manifest)
            result = DatasetInspection(
                root=final_root,
                data_root=final_root / "data",
                spec=spec_payload,
                columns=tuple(manifest["columns"]),
                file_count=manifest["file_count"],
                row_count=manifest["row_count"],
                partitions=tuple(manifest["partitions"]),
                partition_row_counts=manifest["partition_row_counts"],
                date_min=manifest["date_min"],
                date_max=manifest["date_max"],
                manifest=manifest,
            )
            input_owner.close()
            return result

        try:
            return atomic_write_directory(final_root, writer)
        except OutputAlreadyExistsError as exc:
            raise ValueError("dataset_exists") from exc


def _write_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
    timestamp_column: str | None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> _WriteStats:
    accumulator = _WriteStats(spec=spec, data_root=root)
    part_indices: dict[str, int] = {}
    pending_parts: list[Path] = []

    def observe_pending() -> None:
        schema = accumulator._arrow_schema
        assert schema is not None
        if any(pa.types.is_null(field.type) for field in schema):
            return
        for pending in pending_parts:
            with closing_iterator(
                pq.ParquetFile(native_filesystem_path(pending))
            ) as parquet:
                table = parquet.read()
            resolved = table.cast(
                schema.with_metadata(table.schema.metadata), safe=True
            )
            write_parquet_file(resolved, pending, timestamp_column=timestamp_column)
            accumulator.observe_file(pending, resolved)
        pending_parts.clear()

    saw_empty_schema = False
    for chunk in chunks:
        declared = pd.DataFrame(chunk, copy=False).loc[:, list(spec.required_columns)]
        string_columns = object_string_columns(declared)
        for bounded_frame in (
            (declared,)
            if declared.empty
            else iter_frame_chunks(declared, batch_rows=batch_rows)
        ):
            frame, logical_dtypes = accumulator.prepare_frame(bounded_frame)
            if frame.empty:
                accumulator.table_for_frame(frame, object_string_columns=string_columns)
                accumulator.dtype_metadata(logical_dtypes)
                saw_empty_schema = True
                continue
            for partition, partition_frame in partition_frames(
                frame,
                timestamp_column=timestamp_column,
                partition_columns=spec.partition_columns,
            ):
                table = accumulator.table_for_frame(
                    partition_frame,
                    object_string_columns=string_columns,
                )
                persisted = table.replace_schema_metadata(
                    {b"qrp.logical-dtypes": accumulator.dtype_metadata(logical_dtypes)}
                )
                accumulator._summary_observer.observe_frame(partition_frame)
                part_index = part_indices.get(partition, 0)
                part_indices[partition] = part_index + 1
                path = partition_part_path(root, partition, part_index)
                write_parquet_file(persisted, path, timestamp_column=timestamp_column)
                assert accumulator._arrow_schema is not None
                if any(
                    pa.types.is_null(field.type) for field in accumulator._arrow_schema
                ):
                    pending_parts.append(path)
                else:
                    observe_pending()
                    accumulator.observe_file(path, persisted)
    if pending_parts:
        observe_pending()
    if pending_parts:
        raise ValueError("unsupported_dataset_arrow_type")
    row_count = accumulator._summary_observer.row_count
    if row_count == 0 and (not saw_empty_schema):
        raise ValueError("empty_dataset_schema_missing")
    if row_count == 0:
        logical_dtypes = accumulator._logical_dtype_plan
        path = partition_part_path(root, "", 0)
        assert accumulator._arrow_schema is not None
        table = pa.Table.from_batches([], schema=accumulator._arrow_schema)
        table = table.replace_schema_metadata(
            {b"qrp.logical-dtypes": logical_dtype_metadata_bytes(logical_dtypes)}
        )
        write_parquet_file(table, path, timestamp_column=timestamp_column)
        accumulator.observe_file(path, table)
    return accumulator


def _canonical_storage_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                field.name,
                pa.dictionary(
                    pa.int32(),
                    pa.string()
                    if pa.types.is_large_string(field.type.value_type)
                    else field.type.value_type,
                    ordered=field.type.ordered,
                )
                if pa.types.is_dictionary(field.type)
                else pa.timestamp("ms", tz=field.type.tz)
                if pa.types.is_timestamp(field.type) and field.type.unit == "s"
                else field.type,
                nullable=field.nullable,
            )
            for field in schema
        ]
    )


def write_parquet_file(
    table: pa.Table, path: Path, *, timestamp_column: str | None = None
) -> Path:
    filesystem_mkdir(path.parent, parents=True, exist_ok=True)
    with parquet_table_writer(
        native_filesystem_path(path),
        compression="zstd",
        statistics=(timestamp_column,) if timestamp_column is not None else (),
    ) as write:
        write(table)
    return path
