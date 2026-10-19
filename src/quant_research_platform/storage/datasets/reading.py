"""Dataset reading implementation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import (
    PredictionWindow,
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.datasets.inventory import (
    DatasetSessionInventory,
)
from quant_research_platform.storage.datasets.paths import (
    parse_year_partition_label,
    relative_partition_label,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage.datasets.schema import (
    arrow_schema_from_payload,
    data_spec_payload,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_dataset_manifest_payload,
    validate_logical_dtype_payload,
    validate_persisted_logical_schema,
)
from quant_research_platform.storage.datasets.types import DatasetReadMode
from quant_research_platform.storage.identity import (
    filesystem_is_dir,
    filesystem_is_file,
    native_filesystem_path,
    require_artifact_root,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file
from quant_research_platform.storage.parquet import (
    DatasetPartMetadata,
    PreparedMembershipValues,
    _physical_filter_scalar,
    inspect_dataset_part,
    iter_dataset_part_batches,
)


@dataclass(frozen=True, slots=True)
class DatasetSession:
    root: Path
    data_root: Path
    _spec: dict[str, Any]
    _manifest: dict[str, Any]
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema_hash: str
    arrow_schema: pa.Schema
    _logical_dtypes: dict[str, Any]
    native: bool
    parquet_filters: tuple[tuple[str, str, Any], ...] | None
    all_paths: tuple[Path, ...]
    read_paths: tuple[Path, ...]
    _expected_partition_rows: tuple[tuple[str, int], ...] | None = None
    _allow_infinite_columns: tuple[str, ...] = ()

    @property
    def spec(self) -> dict[str, Any]:
        return deepcopy(self._spec)

    @property
    def manifest(self) -> dict[str, Any]:
        return deepcopy(self._manifest)

    @property
    def schema(self) -> list[dict[str, Any]]:
        if self.native:
            return list(self.manifest["physical_schema"])
        return physical_schema_from_schema(
            pa.schema(
                [self.arrow_schema.field(column) for column in self.physical_columns]
            )
        )

    @property
    def logical_dtypes(self) -> dict[str, Any]:
        return deepcopy(self._logical_dtypes)

    def projected_view(self, columns) -> DatasetSession:
        """Select output columns, retaining only filter support in the scanner."""
        requested = resolve_columns_from_schema(
            tuple(self.arrow_schema.names), spec=self._spec, columns=columns
        )
        physical = tuple(
            dict.fromkeys(
                (
                    *requested,
                    *(column for column, _, _ in self.parquet_filters or ()),
                )
            )
        )
        return replace(self, logical_columns=requested, physical_columns=physical)

    def filtered_view(
        self, time_window: TimeWindow | PredictionWindow
    ) -> DatasetSession:
        spec = self.spec
        return replace(
            self,
            read_paths=prune_paths_for_time_window(
                self.data_root, self.read_paths, spec=spec, time_window=time_window
            )
            if self.native
            else self.read_paths,
            _expected_partition_rows=_select_partition_row_counts(
                self._expected_partition_rows,
                _partition_years_for_selection(None, time_window),
            )
            if self.native and tuple(spec["partition_columns"]) == ("year",)
            else self._expected_partition_rows,
            time_window=time_window,
            parquet_filters=(
                *(self.parquet_filters or ()),
                *(
                    parquet_filters_for_time_window(
                        timestamp_column=self.timestamp_column, time_window=time_window
                    )
                    or ()
                ),
            ),
            physical_columns=tuple(
                dict.fromkeys(
                    (
                        *self.physical_columns,
                        *projection_columns_for_window(
                            self.logical_columns, spec=spec, time_window=time_window
                        ),
                    )
                )
            ),
        )


def read_dataset(
    dataset_root: Path,
    *,
    files=None,
    columns=None,
    years=None,
    time_window=None,
    row_filters=(),
    mode=DatasetReadMode.MATERIALIZED,
    spec: DataSpec | None = None,
):
    """Read one logical table in member order, without sorting rows."""
    try:
        mode = DatasetReadMode(mode)
    except (TypeError, ValueError):
        raise ValueError("unsupported_dataset_read_mode") from None

    def activate():
        context = open_dataset_session(
            dataset_root,
            files=files,
            columns=columns,
            years=years,
            time_window=time_window,
            row_filters=row_filters,
            spec=spec,
        )
        yield from iter_dataset_chunks(context, batch_rows=DEFAULT_BATCH_ROWS)

    chunks = activate()
    if mode is DatasetReadMode.CHUNKED:
        return chunks
    with closing_iterator(chunks) as owned:
        frames = list(owned)
    return (
        pd.concat(frames, ignore_index=True)
        if len(frames) > 1
        else frames[0].reset_index(drop=True)
    )


def _selected_paths(root: Path, files) -> tuple[Path, ...]:
    if type(files) not in (list, tuple) or any(
        type(member) is not str or not member for member in files
    ):
        raise ValueError("dataset_files_invalid")
    paths = []
    for member in files:
        relative = Path(member)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.casefold() != ".parquet"
        ):
            raise ValueError("dataset_files_invalid")
        path = root / relative
        require_unaliased_path(path, error="invalid_dataset_part_files")
        if not filesystem_is_file(path) or path in paths:
            raise ValueError("invalid_dataset_part_files")
        paths.append(path)
    return tuple(paths)


def _validated_years(years) -> tuple[int, ...] | None:
    if years is None:
        return None
    if type(years) not in (list, tuple) or any(
        type(year) is not int or not 1 <= year <= 9999 for year in years
    ):
        raise ValueError("dataset_years_invalid")
    return tuple(sorted(set(years)))


def _selection_filters(schema, *, timestamp, time_window, years, row_filters):
    filters = (
        parquet_filters_for_time_window(
            timestamp_column=timestamp, time_window=time_window
        )
        or []
    )
    if years is not None:
        if timestamp is None:
            raise ValueError("time_window_not_supported")
        filters.append((timestamp, "year_in", years))
    for column, operator, value in row_filters:
        if column not in schema.names:
            raise ValueError("dataset_filter_column_not_found")
        if operator not in {"==", "in"}:
            raise ValueError("unsupported_dataset_physical_filter")
        field = schema.field(column)
        if pa.types.is_dictionary(field.type):
            field = pa.field(column, field.type.value_type)
        if operator == "in":
            if type(value) is not tuple or not value:
                raise ValueError("invalid_dataset_physical_filter_value")
            value = _prepare_in_filter(field, value)
        else:
            value = _validated_physical_filter_value(field, value)
        filters.append((column, operator, value))
    return filters


def _dataset_input_options(value, *, time_window=None, discovery_windows=None):
    """Intersect explicit inputs with the consumer domain before discovering files.

    Disjoint consumers still read each window separately; discovery_windows only
    narrows their shared inventory, without filling the intervening years.
    """
    if isinstance(value, (str, Path)):
        value = {"path": value}
    if (
        not isinstance(value, Mapping)
        or set(value)
        - {"path", "spec", "files", "columns", "years", "time_window", "row_filters"}
        or "path" not in value
    ):
        raise ValueError("dataset_input_invalid")
    options = dict(value)
    path = options.pop("path")
    if not isinstance(path, (str, Path)):
        raise ValueError("dataset_input_invalid")
    options["dataset_root"] = Path(path)
    if options.get("columns") is not None:
        options["columns"] = _validated_columns(options["columns"])
    declared = options.get("spec")
    if declared is not None and not isinstance(declared, DataSpec):
        if not isinstance(declared, Mapping):
            raise ValueError("parquet_semantic_spec_required")
        values = dict(declared)
        for name in ("primary_key", "partition_columns"):
            if name in values:
                values[name] = tuple(values[name])
        options["spec"] = DataSpec(**values)
    window = options.get("time_window")
    if window is not None and not isinstance(window, TimeWindow):
        if not isinstance(window, Mapping):
            raise ValueError("dataset_time_window_invalid")
        options["time_window"] = TimeWindow(**window)
    if "row_filters" in options:
        filters = options["row_filters"]
        if type(filters) not in (tuple, list) or any(
            type(item) not in (tuple, list) or len(item) != 3 for item in filters
        ):
            raise ValueError("dataset_row_filters_invalid")
        for column, operator, items in filters:
            if type(column) is not str or not column:
                raise ValueError("dataset_row_filters_invalid")
            if (
                options.get("spec") is not None
                and column not in options["spec"].required_columns
            ):
                raise ValueError("dataset_filter_column_not_found")
            if type(operator) is not str or operator not in {"==", "in"}:
                raise ValueError("unsupported_dataset_physical_filter")
            if operator == "in" and (type(items) not in (tuple, list) or not items):
                raise ValueError("invalid_dataset_physical_filter_value")
        options["row_filters"] = tuple(
            (
                column,
                operator,
                tuple(items) if operator == "in" and type(items) is list else items,
            )
            for column, operator, items in filters
        )
    if time_window is not None or discovery_windows is not None:
        years = _validated_years(options.get("years"))

        def interval(window):
            if window.end is None:
                return pd.Timestamp(window.start), None
            lower, upper, closed = time_window_timestamp_interval(window)
            return pd.Timestamp(lower), pd.Timestamp(upper) - (
                pd.Timedelta(0) if closed else pd.Timedelta(1, "ns")
            )

        limits = [
            interval(window)
            for window in (options.get("time_window"), time_window)
            if window is not None
        ]
        if time_window is not None:
            start = max(lower for lower, _ in limits)
            ends = [upper for _, upper in limits if upper is not None]
            end = min(ends) if ends else None
            if end is not None and start > end:
                options["years"] = ()
                return options
            options["time_window"] = (
                PredictionWindow(start, None) if end is None else TimeWindow(start, end)
            )
        if discovery_windows is not None:
            selected_years = set()
            for window in discovery_windows:
                bounds = [*limits, interval(window)]
                start = max(lower for lower, _ in bounds)
                ends = [upper for _, upper in bounds if upper is not None]
                end = min(ends) if ends else None
                if end is None or start <= end:
                    selected_years.update(
                        range(start.year, (end.year if end is not None else 9999) + 1)
                    )
            options["years"] = tuple(
                sorted(
                    selected_years
                    if years is None
                    else selected_years.intersection(years)
                )
            )
    return options


def _bare_dataset_session(
    path, *, columns, time_window, files, years, row_filters, spec
):
    if spec is None:
        raise ValueError("parquet_semantic_spec_required")
    is_file = Path(native_filesystem_path(path)).is_file()
    if is_file:
        if files is not None:
            raise ValueError("dataset_files_require_directory")
        require_unaliased_path(path, error="invalid_dataset_part_files")
        paths = (path,)
    else:
        if files is None:
            raise ValueError("parquet_files_required")
        paths = _selected_paths(path, files)
        if not paths:
            raise ValueError("parquet_files_required")
    payload = data_spec_payload(spec)
    metadata = inspect_dataset_part(paths[0], require_logical_dtypes=False)
    schema = metadata.schema.with_metadata(
        {b"pandas": metadata.pandas_metadata} if metadata.pandas_metadata else None
    )
    logical = (
        parse_logical_dtype_metadata(
            metadata.logical_dtype_payload, selected_columns=()
        )
        if metadata.logical_dtype_payload
        else {}
    )
    requested = resolve_columns_from_schema(
        tuple(schema.names), spec=payload, columns=columns
    )
    timestamp = timestamp_column_from_spec_payload(payload)
    filters = _selection_filters(
        schema,
        timestamp=timestamp,
        time_window=time_window,
        years=years,
        row_filters=row_filters,
    )
    return DatasetSession(
        root=path,
        data_root=path.parent if is_file else path,
        _spec=payload,
        _manifest={},
        logical_columns=requested,
        physical_columns=tuple(
            dict.fromkeys((*requested, *(column for column, _, _ in filters)))
        ),
        timestamp_column=timestamp,
        time_window=time_window,
        schema_hash="",
        arrow_schema=schema,
        _logical_dtypes=logical,
        native=False,
        parquet_filters=tuple(filters) or None,
        all_paths=paths,
        read_paths=paths,
    )


def _project_logical_dtypes(payload, columns):
    by_name = {item["name"]: item for item in payload["columns"]}
    return {
        "schema_id": payload["schema_id"],
        "columns": [by_name[column] for column in columns],
    }


def materialize_dataset_session(
    context: DatasetSession, *, batch_rows: int = DEFAULT_BATCH_ROWS
) -> pd.DataFrame:
    """Materialize the validated stream once, preserving declared logical dtypes."""
    with closing_iterator(
        iter_dataset_chunks(context, batch_rows=batch_rows)
    ) as chunks:
        frames = list(chunks)
    return (
        pd.concat(frames, ignore_index=True)
        if len(frames) > 1
        else frames[0].reset_index(drop=True)
    )


def open_dataset_session(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
    files: tuple[str, ...] | list[str] | None = None,
    years: tuple[int, ...] | list[int] | None = None,
    spec: DataSpec | None = None,
) -> DatasetSession:
    years = _validated_years(years)
    path = Path(dataset_root)
    native_path = Path(native_filesystem_path(path))
    if not native_path.exists():
        raise FileNotFoundError(path)
    if native_path.is_file() or (
        spec is not None and not (native_path / "spec.json").is_file()
    ):
        return _bare_dataset_session(
            path,
            columns=columns,
            time_window=time_window,
            files=files,
            years=years,
            row_filters=row_filters,
            spec=spec,
        )
    if spec is not None:
        raise ValueError("dataset_spec_already_declared")
    root = require_artifact_root(Path(dataset_root))
    spec_path, data_root, manifest_path = (
        root / "spec.json",
        root / "data",
        root / "_manifest" / "manifest.json",
    )
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(data_root, error="invalid_dataset_part_files")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    if not filesystem_is_file(spec_path):
        raise ValueError("dataset_spec_missing")
    if not filesystem_is_dir(data_root):
        raise ValueError("dataset_data_dir_missing")
    if not filesystem_is_file(manifest_path):
        raise ValueError("dataset_manifest_missing")
    _, spec_payload = read_qrp_json_file(spec_path)
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    _, manifest = read_qrp_json_file(manifest_path)
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload, logical_columns=())
    inventory = DatasetSessionInventory(data_root=data_root)
    expected_partition_rows = None
    if files is None:
        partition_years = _partition_years_for_selection(
            years,
            time_window
            if tuple(spec_payload["partition_columns"]) == ("year",)
            else None,
        )
        expected_partition_rows = _select_partition_row_counts(
            tuple(manifest["partition_row_counts"].items()), partition_years
        )
        inventory.discover(spec_payload, years=partition_years)
    else:
        for path in _selected_paths(root, files):
            try:
                inventory.add(path)
            except ValueError as exc:
                raise ValueError("invalid_dataset_part_files") from exc
    session = _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_payload=spec_payload,
        manifest=manifest,
        inventory=inventory,
        columns=columns,
        time_window=time_window,
        row_filters=row_filters,
        years=_validated_years(years),
    )
    return replace(session, _expected_partition_rows=expected_partition_rows)


def _partition_years_for_selection(years, time_window):
    selected = None if years is None else set(years)
    if time_window is not None:
        start, end = time_window_date_bounds(time_window)
        window_years = set(range(int(start.year), (end or date.max).year + 1))
        selected = window_years if selected is None else selected & window_years
    return selected


def _select_partition_row_counts(partition_rows, years):
    if partition_rows is None:
        return None
    return tuple(
        (partition, rows)
        for partition, rows in partition_rows
        if years is None
        or (
            years
            and (
                partition == "unpartitioned"
                or parse_year_partition_label(partition) in years
            )
        )
    )


def _build_dataset_session(
    *,
    root: Path,
    data_root: Path,
    spec_payload: dict[str, Any],
    manifest: dict[str, Any],
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...],
    years: tuple[int, ...] | None,
) -> DatasetSession:
    all_paths = inventory.paths
    schema = manifest["physical_schema"]
    arrow_schema = arrow_schema_from_payload(schema)
    schema_columns = physical_schema_columns(schema)
    schema_hash = manifest["schema_hash"]
    logical_dtypes = manifest["logical_dtypes"]
    logical_columns = resolve_columns_from_schema(
        schema_columns, spec=spec_payload, columns=columns
    )
    timestamp_column = timestamp_column_from_spec_payload(spec_payload)
    physical_columns = projection_columns_for_window(
        logical_columns, spec=spec_payload, time_window=time_window
    )
    physical_columns = tuple(
        dict.fromkeys(
            (
                *physical_columns,
                *((timestamp_column,) if years is not None else ()),
                *(column for column, _, _ in row_filters),
            )
        )
    )
    physical_filters = _selection_filters(
        arrow_schema,
        timestamp=timestamp_column,
        time_window=time_window,
        years=years,
        row_filters=row_filters,
    )
    selected_paths = prune_paths_for_time_window(
        data_root, all_paths, spec=spec_payload, time_window=time_window
    )
    if years == ():
        selected_paths = ()
    if years is not None and tuple(spec_payload["partition_columns"]) == ("year",):
        selected_paths = tuple(
            path
            for path in selected_paths
            if path.parent == data_root
            or parse_year_partition_label(path.parent.name) in years
        )
    session = DatasetSession(
        all_paths=all_paths,
        read_paths=selected_paths,
        root=root,
        data_root=data_root,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema_hash=schema_hash,
        arrow_schema=arrow_schema,
        _spec=deepcopy(spec_payload),
        _manifest=deepcopy(manifest),
        _logical_dtypes=deepcopy(logical_dtypes),
        native=True,
        parquet_filters=tuple(
            (
                (column, operator, _freeze_filter_value(value))
                for column, operator, value in physical_filters
            )
        )
        or None,
    )
    return session


def iter_dataset_chunks(
    context: DatasetSession,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    rows_per_batch = _validated_batch_rows(batch_rows)
    key_index = None
    keys = None

    def observe_consumed(batch, frame):
        nonlocal key_index, keys
        if keys is None:
            keys = tuple(context.spec["primary_key"])
        for column in keys:
            if column not in batch.schema.names:
                continue
            values = batch.column(column)
            if pa.types.is_dictionary(values.type):
                values = pc.dictionary_decode(values)
            if values.null_count or (
                pa.types.is_floating(values.type) and pc.any(pc.is_nan(values)).as_py()
            ):
                raise ValueError("dataset_primary_key_null")
        if not all(column in batch.schema.names for column in keys):
            from quant_research_platform.storage.exact_keys import (
                _append_arrow_scalar_payload,
            )

            for column in keys:
                if column not in batch.schema.names:
                    continue
                values = batch.column(column)
                value_type = values.type
                if pa.types.is_dictionary(value_type):
                    values = pc.dictionary_decode(values)
                    value_type = values.type
                if pa.types.is_nested(value_type):
                    for value in values:
                        _append_arrow_scalar_payload(bytearray(), value, value_type)
        if all(column in batch.schema.names for column in keys):
            if key_index is None:
                from quant_research_platform.storage.exact_keys import (
                    ExactPrimaryKeyIndex,
                )

                key_index = ExactPrimaryKeyIndex(primary_key=keys)
            if key_index.add_table(pa.Table.from_batches([batch])):
                raise ValueError("duplicate_dataset_primary_key")
        for field, column in zip(batch.schema, batch.columns, strict=True):
            if (
                (
                    field.name not in context._allow_infinite_columns
                    or field.name in keys
                )
                and pa.types.is_floating(column.type)
                and pc.any(pc.is_inf(column)).as_py()
            ):
                raise ValueError("dataset_infinite_value")
        if batch_observer is not None:
            batch_observer(batch, frame)

    yielded = False
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=context.physical_columns,
            batch_rows=rows_per_batch,
            batch_observer=observe_consumed,
        )
    ) as file_batches:
        for _, frame in file_batches:
            filtered = frame.loc[:, list(context.logical_columns)]
            if filtered.empty:
                continue
            yielded = True
            try:
                yield filtered
            except GeneratorExit:
                return
    if not yielded:
        yield _empty_frame(context, context.logical_columns)


def iter_declared_parquet_member_batches(
    path: Path,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    dtype_overrides: Mapping[str, object] | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    """Traverse one frozen member through the sole logical Pandas converter."""
    rows_per_batch = _validated_batch_rows(batch_rows)
    with closing_iterator(
        iter_dataset_part_batches(
            Path(path),
            batch_rows=rows_per_batch,
            columns=columns,
            parquet_filters=parquet_filters,
            metadata_observer=metadata_observer,
        )
    ) as batches:
        for batch in batches:
            # Observer failures remain outside the consumer-close boundary.
            for frame in _arrow_batch_to_bounded_pandas(
                batch,
                batch_rows=rows_per_batch,
                dtype_overrides=dtype_overrides,
                persisted_frame_observer=persisted_frame_observer,
                batch_observer=batch_observer,
            ):
                try:
                    yield frame
                except GeneratorExit:
                    return


def iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    batch_rows: int = DEFAULT_BATCH_ROWS,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""
    rows_per_batch = _validated_batch_rows(batch_rows)
    logical = context.logical_dtypes
    declared_logical = _project_logical_dtypes(logical, columns) if logical else {}
    if declared_logical:
        validate_logical_dtype_payload(declared_logical)
    dtype_overrides = (
        pandas_dtype_overrides_from_logical_payload(declared_logical, columns=columns)
        if declared_logical
        else {}
    )
    expected = pa.schema([context.arrow_schema.field(column) for column in columns])
    expected_rows = dict(context._expected_partition_rows or ())
    observed_rows = dict.fromkeys(expected_rows, 0)
    present_partitions = {
        relative_partition_label(context.data_root, path) for path in context.read_paths
    }
    if any(
        rows and partition not in present_partitions
        for partition, rows in expected_rows.items()
    ):
        raise ValueError("dataset_selected_partition_incomplete")

    def validate_metadata(metadata):
        try:
            selected = pa.schema([metadata.schema.field(column) for column in columns])
        except KeyError as exc:
            raise ValueError("dataset_column_not_found") from exc
        if selected != expected:
            raise ValueError("dataset_schema_mismatch")
        if not metadata.logical_dtype_payload and not context.native:
            if declared_logical:
                raise ValueError("dataset_logical_dtypes_missing")
            return
        current = _project_logical_dtypes(
            parse_logical_dtype_metadata(
                metadata.logical_dtype_payload, selected_columns=columns
            ),
            columns,
        )
        validate_persisted_logical_schema(selected, current)
        if not declared_logical:
            raise ValueError("dataset_logical_dtypes_mismatch")
        merged = merge_logical_dtype_payloads([declared_logical, current])
        if merged != declared_logical:
            raise ValueError("manifest_logical_dtypes_mismatch")
        if context._expected_partition_rows is not None:
            if partition not in expected_rows and metadata.row_count:
                raise ValueError("dataset_selected_partition_incomplete")
            if partition in observed_rows:
                observed_rows[partition] += metadata.row_count

    for path in context.read_paths:
        partition = relative_partition_label(context.data_root, path)
        with closing_iterator(
            iter_declared_parquet_member_batches(
                path,
                columns=columns,
                parquet_filters=context.parquet_filters,
                batch_rows=rows_per_batch,
                dtype_overrides=dtype_overrides,
                metadata_observer=validate_metadata,
                batch_observer=batch_observer,
            )
        ) as member_batches:
            for frame in member_batches:
                try:
                    yield path, frame
                except GeneratorExit:
                    return
    if observed_rows != expected_rows:
        raise ValueError("dataset_selected_partition_incomplete")


def dataset_primary_key_schema(
    context: DatasetSession, primary_key: tuple[str, ...]
) -> pa.Schema:
    try:
        return pa.schema([context.arrow_schema.field(column) for column in primary_key])
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc


def _validated_batch_rows(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_batch_rows")
    return value


def _freeze_filter_value(value: Any) -> Any:
    if isinstance(value, PreparedMembershipValues):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_filter_value(item) for item in value)
    if isinstance(
        value,
        (
            str,
            bytes,
            bool,
            int,
            float,
            Decimal,
            date,
            datetime,
            time,
            timedelta,
            pd.Timestamp,
            pd.Timedelta,
        ),
    ):
        return value
    raise ValueError("invalid_dataset_physical_filter_value")


def _filter_value_is_representable(field: pa.Field, value: Any) -> bool:
    source = _physical_filter_scalar(field, value)
    try:
        converted = source.cast(field.type, safe=True)
    except pa.ArrowInvalid:
        return False
    return bool(pc.equal(source, converted).as_py())


def _prepare_in_filter(
    field: pa.Field, values: tuple[Any, ...]
) -> PreparedMembershipValues:
    for value in values:
        _validated_physical_filter_value(field, value)
    try:
        selected = pa.array(
            [value for value in values if _filter_value_is_representable(field, value)],
            type=field.type,
        )
        unique = pc.unique(selected)
        return PreparedMembershipValues(pc.take(unique, pc.sort_indices(unique)))
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
        raise ValueError("invalid_dataset_physical_filter_value") from None


def _validated_physical_filter_value(field: pa.Field, value: Any) -> Any:
    frozen = _freeze_filter_value(value)
    data_type = field.type
    if frozen is pd.NaT or isinstance(frozen, np.generic):
        raise ValueError("invalid_dataset_physical_filter_value")
    if isinstance(frozen, (float, Decimal)) and (not math.isfinite(frozen)):
        raise ValueError("invalid_dataset_physical_filter_value")
    if pa.types.is_boolean(data_type):
        valid = type(frozen) is bool
    elif pa.types.is_integer(data_type):
        valid = type(frozen) is int
    elif pa.types.is_floating(data_type):
        valid = type(frozen) is float
    elif pa.types.is_decimal(data_type):
        valid = type(frozen) is Decimal
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        valid = type(frozen) is str
        if valid:
            try:
                frozen.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("invalid_dataset_physical_filter_value") from None
    elif (
        pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        valid = type(frozen) is bytes
    elif pa.types.is_timestamp(data_type):
        valid = type(frozen) in (pd.Timestamp, datetime)
    elif pa.types.is_date(data_type):
        valid = type(frozen) is date
    elif pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        valid = type(frozen) is time
    elif pa.types.is_duration(data_type):
        valid = type(frozen) in (pd.Timedelta, timedelta)
    else:
        valid = False
    if not valid:
        raise ValueError("invalid_dataset_physical_filter_value")
    try:
        pa.scalar(frozen, type=data_type)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        OverflowError,
    ):
        raise ValueError("invalid_dataset_physical_filter_value") from None
    return frozen


def _empty_frame(context: DatasetSession, columns: tuple[str, ...]) -> pd.DataFrame:
    logical = context.logical_dtypes
    selected_logical = _project_logical_dtypes(logical, columns) if logical else {}
    if selected_logical:
        validate_logical_dtype_payload(selected_logical)
    selected = pa.schema(
        [context.arrow_schema.field(column) for column in columns],
        metadata=context.arrow_schema.metadata,
    )
    if selected_logical:
        validate_persisted_logical_schema(selected, selected_logical)
    overrides = (
        pandas_dtype_overrides_from_logical_payload(selected_logical, columns=columns)
        if selected_logical
        else {}
    )
    frame = _arrow_slice_to_pandas(
        pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in selected], schema=selected
        ),
        dtype_overrides=overrides,
    )
    return frame.astype(overrides) if overrides else frame


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    dtype_overrides: Mapping[str, object] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    target = _validated_batch_rows(batch_rows)
    rows = target
    overrides = dict(dtype_overrides or {})
    categorical = {
        name: dtype
        for name, value in overrides.items()
        if isinstance((dtype := pd.api.types.pandas_dtype(value)), pd.CategoricalDtype)
    }
    ordinary = {
        name: value for name, value in overrides.items() if name not in categorical
    }
    for start in range(0, batch.num_rows, rows):
        arrow_slice = batch.slice(start, rows)
        frame = _arrow_slice_to_pandas(arrow_slice, dtype_overrides=overrides)
        if persisted_frame_observer is not None:
            persisted_frame_observer(frame)
        output = frame.astype(ordinary) if ordinary else frame
        for name, dtype in categorical.items():
            codes = dtype.categories.get_indexer(frame[name])
            if np.any((codes == -1) & frame[name].notna().to_numpy()):
                raise ValueError("dataset_categorical_domain_mismatch")
            output[name] = pd.Categorical.from_codes(codes, dtype=dtype, validate=False)
        if batch_observer is not None:
            batch_observer(arrow_slice, output)
        yield output


def _arrow_slice_to_pandas(
    arrow_slice: pa.RecordBatch, *, dtype_overrides: Mapping[str, object] | None = None
) -> pd.DataFrame:
    metadata = arrow_slice.schema.metadata
    if metadata and b"pandas" in metadata:
        pandas_metadata = json.loads(metadata[b"pandas"])
        if pandas_metadata.get("index_columns"):
            # Dataset keys remain columns; retain pandas extension dtype metadata.
            pandas_metadata["index_columns"] = []
            arrow_slice = arrow_slice.replace_schema_metadata(
                {**metadata, b"pandas": json.dumps(pandas_metadata).encode("utf-8")}
            )
    native_types = set()
    nullable_float_columns = []
    nullable_discrete_columns = {}
    for name, dtype in (dtype_overrides or {}).items():
        dtype = pd.api.types.pandas_dtype(dtype)
        data_type = arrow_slice.schema.field(name).type
        if pa.types.is_timestamp(data_type) and pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            expected = pa.Array.from_pandas(pd.Series([], dtype=dtype)).type
            if data_type != expected:
                try:
                    values = arrow_slice.column(name).cast(expected, safe=True)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    raise ValueError("dataset_logical_dtype_mismatch") from None
                arrow_slice = arrow_slice.set_column(
                    arrow_slice.schema.get_field_index(name), name, values
                )
        if pa.types.is_integer(data_type) or pa.types.is_boolean(data_type):
            native_types.add(data_type)
            if (
                isinstance(dtype, pd.api.extensions.ExtensionDtype)
                and dtype.kind in "biu"
            ):
                nullable_discrete_columns[name] = dtype
        if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
            native_types.add(data_type)
            nullable_float_columns.append(name)
        if (
            isinstance(dtype, np.dtype)
            and dtype.kind in "biu"
            and arrow_slice.column(name).null_count
        ):
            raise ValueError("dataset_nonnullable_value_missing")
    frame = arrow_slice.to_pandas(
        use_threads=True,
        integer_object_nulls=True,
        types_mapper=(
            lambda data_type: (
                pd.ArrowDtype(data_type) if data_type in native_types else None
            )
        )
        if native_types
        else None,
    )
    for name, dtype in nullable_discrete_columns.items():
        # Convert from Arrow directly: astype can round nullable integers via float.
        frame[name] = dtype.__from_arrow__(arrow_slice.column(name))
    for name in nullable_float_columns:
        values = arrow_slice.column(name)
        frame[name] = pd.arrays.FloatingArray(
            values.to_numpy(zero_copy_only=False),
            values.is_null().to_numpy(zero_copy_only=False),
            copy=False,
        )
    return frame


def iter_frame_chunks(
    frame: pd.DataFrame, *, batch_rows: int = DEFAULT_BATCH_ROWS
) -> Iterator[pd.DataFrame]:
    rows = _validated_batch_rows(batch_rows)
    for start in range(0, len(frame), rows):
        yield frame.iloc[start : start + rows]


def prune_paths_for_time_window(
    data_root: Path,
    paths: tuple[Path, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[Path, ...]:
    if time_window is None:
        return paths
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns != ("year",):
        return paths
    years = _partition_years_for_selection(None, time_window)
    return tuple(
        path
        for path in paths
        if path.parent == Path(data_root)
        or parse_year_partition_label(relative_partition_label(data_root, path))
        in years
    )


def parquet_filters_for_time_window(
    *, timestamp_column: str | None, time_window: TimeWindow | None
) -> list[tuple[str, str, Any]] | None:
    if time_window is None:
        return None
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    if time_window.end is None:
        return [(timestamp_column, ">=", pd.Timestamp(time_window.start))]
    start, end, end_inclusive = time_window_timestamp_interval(time_window)
    upper_operator = "<=" if end_inclusive else "<"
    return [
        (timestamp_column, ">=", pd.Timestamp(start)),
        (timestamp_column, upper_operator, pd.Timestamp(end)),
    ]


def _validated_columns(columns) -> tuple[str, ...]:
    if type(columns) not in (tuple, list):
        raise ValueError("dataset_columns_invalid")
    if not columns:
        raise ValueError("dataset_columns_empty")
    if any(type(column) is not str or not column for column in columns) or len(
        columns
    ) != len(set(columns)):
        raise ValueError("dataset_columns_invalid")
    return tuple(columns)


def resolve_columns_from_schema(
    schema_columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    requested = (
        tuple(spec["required_columns"])
        if columns is None
        else _validated_columns(columns)
    )
    missing = [column for column in requested if column not in schema_columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def projection_columns_for_window(
    columns: tuple[str, ...], *, spec: Mapping[str, Any], time_window: TimeWindow | None
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))


def _iter_dataset_session_frames(session, columns):
    from quant_research_platform.runtime.dataframes import iter_complete_session_frames

    with closing_iterator(
        iter_dataset_chunks(session.projected_view(columns))
    ) as batches:
        source = batches
        with closing_iterator(
            iter_complete_session_frames(source, timestamp_column="timestamp")
        ) as frames:
            for frame in frames:
                if not frame.empty:
                    try:
                        yield frame
                    except GeneratorExit:
                        return
