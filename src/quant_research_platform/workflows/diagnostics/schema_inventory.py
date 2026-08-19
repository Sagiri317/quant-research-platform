"""Workflow-owned schema inventory payload builder."""

from __future__ import annotations

import sys
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from quant_research_platform.contracts._json import (
    bounded_rendered_qrp_json_object_size,
    canonical_json_hash,
)
from quant_research_platform.runtime.logical_paths import RUN_RESULT_DATASET_LAYERS
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_inspection import (
    dataset_inspection_retained_memory_bytes,
    inspect_dataset_session,
)
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    closing_dataset_iterator,
    iter_dataset_chunks,
    open_dataset_session,
)

_INVENTORY_KEYS = frozenset({"schema_id", "datasets"})
_INVENTORY_ITEM_KEYS = frozenset(
    {
        "artifact_id",
        "layer",
        "kind",
        "logical_relative_root",
        "columns",
        "dtypes",
        "primary_key",
        "partition_columns",
        "file_count",
        "row_count",
        "row_count_by_year",
        "min_timestamp",
        "max_timestamp",
        "duplicate_primary_key_count",
        "null_key_counts",
        "selected_important_column_null_counts",
        "schema_hash",
        "warnings",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


@contextmanager
def open_schema_inventory_payload(
    project_root: Path,
    *,
    important_null_count_columns: Collection[str] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[dict[str, Any]]:
    """Retain sanitized schema metadata through its one consumer."""

    with runtime_resource_policy_scope(resource_policy) as policy:
        root = Path(project_root)
        declared_important_columns = (
            () if important_null_count_columns is None else important_null_count_columns
        )
        if any(
            type(column) is not str
            or not column
            for column in declared_important_columns
        ):
            raise ValueError("schema_inventory_important_columns_invalid")
        important_columns = frozenset(declared_important_columns)
        data_root = root / "data"
        for candidate in (
            root,
            data_root,
            *(data_root / layer for layer in sorted(RUN_RESULT_DATASET_LAYERS)),
        ):
            require_unaliased_path(candidate, error="schema_inventory_dataset_invalid")
        with ExitStack() as retained:
            items: list[dict[str, Any]] = []
            retained_item_bytes = 0
            for artifact_root in _iter_dataset_artifact_roots(root):
                try:
                    item, item_bytes = _inventory_item(
                        root,
                        artifact_root,
                        important_null_count_columns=important_columns,
                        resource_policy=policy,
                        retained_items=retained,
                    )
                except MemoryError:
                    raise ValueError(
                        "schema_inventory_payload_exceeds_task_memory_budget"
                    ) from None
                items.append(item)
                retained_item_bytes += item_bytes
            if not items:
                raise ValueError("schema_inventory_no_datasets_found")
            render_bound = 128 + retained_item_bytes + 2 * len(items)
            try:
                retained.enter_context(
                    policy.reserve(
                        "schema.inventory.consumer_overlap",
                        render_bound + len(items) * sys.getsizeof(None),
                    )
                )
            except MemoryError:
                raise ValueError(
                    "schema_inventory_payload_exceeds_task_memory_budget"
                ) from None
            payload = {
                "schema_id": "qrp-schema-inventory",
                "datasets": sorted(
                    items,
                    key=lambda item: item["logical_relative_root"],
                ),
            }
            try:
                bounded_rendered_qrp_json_object_size(
                    payload,
                    lazy_arrays={},
                    max_bytes=render_bound,
                )
            except (MemoryError, ValueError):
                raise ValueError(
                    "schema_inventory_payload_exceeds_task_memory_budget"
                ) from None
            yield payload


def summarize_schema_inventory_payload(
    value: object,
) -> dict[str, object]:
    """Validate and summarize one complete schema inventory without coercion."""

    payload = validate_schema_inventory_payload(value)
    datasets = payload["datasets"]
    return {
        "schema_id": "qrp-schema-inventory-summary",
        "dataset_count": len(datasets),
        "total_row_count": sum(item["row_count"] for item in datasets),
        "schema_hash": canonical_json_hash(
            [
                {
                    "artifact_id": item["artifact_id"],
                    "layer": item["layer"],
                    "kind": item["kind"],
                    "logical_relative_root": item["logical_relative_root"],
                    "schema_hash": item["schema_hash"],
                    "row_count": item["row_count"],
                }
                for item in datasets
            ]
        ),
    }


def validate_schema_inventory_payload(
    value: object,
) -> dict[str, list[dict[str, Any]] | str]:
    """Validate the exact Workflow-owned schema-inventory grammar."""

    if type(value) is not dict or set(value) != _INVENTORY_KEYS:
        raise ValueError("schema_inventory_fields_invalid")
    if value["schema_id"] != "qrp-schema-inventory":
        raise ValueError("schema_inventory_schema_id_invalid")
    datasets = value["datasets"]
    if type(datasets) is not list or not datasets:
        raise ValueError("schema_inventory_datasets_required")
    validated: list[dict[str, Any]] = []
    logical_roots: list[str] = []
    identities: set[tuple[str, str, str, str]] = set()
    for item in datasets:
        validated_item = _validate_schema_inventory_item(item)
        logical_root = validated_item["logical_relative_root"]
        identity = _logical_identifiers(logical_root)
        if logical_root in logical_roots or identity in identities:
            raise ValueError("schema_inventory_dataset_duplicate")
        logical_roots.append(logical_root)
        identities.add(identity)
        validated.append(validated_item)
    if logical_roots != sorted(logical_roots):
        raise ValueError("schema_inventory_dataset_order_invalid")
    return {"schema_id": "qrp-schema-inventory", "datasets": validated}


def _validate_schema_inventory_item(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _INVENTORY_ITEM_KEYS:
        raise ValueError("schema_inventory_dataset_invalid")
    for key in ("artifact_id", "layer", "kind", "logical_relative_root"):
        if type(value[key]) is not str or not value[key]:
            raise ValueError("schema_inventory_dataset_invalid")
    if value["layer"] not in RUN_RESULT_DATASET_LAYERS:
        raise ValueError("schema_inventory_dataset_invalid")
    logical_root = value["logical_relative_root"]
    logical_path = PurePosixPath(logical_root)
    if (
        logical_path.is_absolute()
        or logical_path.as_posix() != logical_root
        or not logical_path.parts
        or any(part in {"", ".", ".."} for part in logical_path.parts)
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    try:
        path_layer, _, path_kind, path_artifact_id = _logical_identifiers(logical_root)
    except ValueError:
        raise ValueError("schema_inventory_dataset_invalid") from None
    if (
        value["layer"],
        value["kind"],
        value["artifact_id"],
    ) != (path_layer, path_kind, path_artifact_id):
        raise ValueError("schema_inventory_dataset_invalid")

    columns = _exact_string_list(value["columns"], allow_empty=False)
    primary_key = _exact_string_list(value["primary_key"], allow_empty=True)
    partition_columns = _exact_string_list(value["partition_columns"], allow_empty=True)
    warnings = _exact_string_list(value["warnings"], allow_empty=True)
    if warnings != sorted(warnings):
        raise ValueError("schema_inventory_dataset_invalid")
    dtypes = _exact_string_mapping(value["dtypes"], expected_keys=set(columns))
    row_count_by_year = _exact_count_mapping(value["row_count_by_year"])
    null_key_counts = _exact_count_mapping(
        value["null_key_counts"], expected_keys=set(primary_key)
    )
    important_null_counts = _exact_count_mapping(
        value["selected_important_column_null_counts"]
    )
    if not set(important_null_counts).issubset(columns):
        raise ValueError("schema_inventory_dataset_invalid")
    file_count = _exact_nonnegative_int(value["file_count"])
    row_count = _exact_nonnegative_int(value["row_count"])
    duplicate_count = _exact_nonnegative_int(value["duplicate_primary_key_count"])
    if not set(primary_key).issubset(columns):
        raise ValueError("schema_inventory_dataset_invalid")
    if any(count > row_count for count in null_key_counts.values()) or any(
        count > row_count for count in important_null_counts.values()
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    if duplicate_count > row_count or (row_count > 0 and file_count == 0):
        raise ValueError("schema_inventory_dataset_invalid")
    date_min, date_max = _exact_date_range(
        value["min_timestamp"], value["max_timestamp"]
    )
    schema_hash = value["schema_hash"]
    if (
        type(schema_hash) is not str
        or len(schema_hash) != 64
        or any(character not in _LOWER_HEX for character in schema_hash)
        or schema_hash
        != _schema_hash(
            columns=tuple(columns),
            dtypes=dtypes,
            primary_key=tuple(primary_key),
            partition_columns=tuple(partition_columns),
        )
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    years = list(row_count_by_year)
    if (
        years != sorted(years)
        or any(
            len(year) != 4 or not year.isdecimal() or row_count_by_year[year] == 0
            for year in years
        )
        or sum(row_count_by_year.values()) > row_count
        or (
            "timestamp" in primary_key
            and null_key_counts["timestamp"] == 0
            and sum(row_count_by_year.values()) != row_count
        )
        or bool(years) != (date_min is not None)
        or (years and (years[0] != date_min[:4] or years[-1] != date_max[:4]))
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    return value


def _exact_date_range(
    minimum: object,
    maximum: object,
) -> tuple[str | None, str | None]:
    if minimum is None and maximum is None:
        return None, None
    if type(minimum) is not str or type(maximum) is not str:
        raise ValueError("schema_inventory_dataset_invalid")
    try:
        parsed_minimum = date.fromisoformat(minimum)
        parsed_maximum = date.fromisoformat(maximum)
    except ValueError:
        raise ValueError("schema_inventory_dataset_invalid") from None
    if (
        parsed_minimum.isoformat() != minimum
        or parsed_maximum.isoformat() != maximum
        or parsed_minimum > parsed_maximum
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    return minimum, maximum


def _exact_string_list(value: object, *, allow_empty: bool) -> list[str]:
    if (
        type(value) is not list
        or (not allow_empty and not value)
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    return value


def _exact_string_mapping(
    value: object,
    *,
    expected_keys: set[str],
) -> dict[str, str]:
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or any(
            type(key) is not str or not key or type(item) is not str or not item
            for key, item in value.items()
        )
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    return value


def _exact_count_mapping(
    value: object,
    *,
    expected_keys: set[str] | None = None,
) -> dict[str, int]:
    if type(value) is not dict or (
        expected_keys is not None and set(value) != expected_keys
    ):
        raise ValueError("schema_inventory_dataset_invalid")
    for key, item in value.items():
        if type(key) is not str or not key:
            raise ValueError("schema_inventory_dataset_invalid")
        _exact_nonnegative_int(item)
    return value


def _exact_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("schema_inventory_dataset_invalid")
    return value


def _iter_dataset_artifact_roots(project_root: Path) -> Iterable[Path]:
    data_root = project_root / "data"
    require_unaliased_path(project_root, error="schema_inventory_dataset_invalid")
    require_unaliased_path(data_root, error="schema_inventory_dataset_invalid")
    allowed_roots = tuple(
        data_root / layer for layer in sorted(RUN_RESULT_DATASET_LAYERS)
    )
    for layer_root in allowed_roots:
        require_unaliased_path(layer_root, error="schema_inventory_dataset_invalid")
        if not layer_root.exists():
            continue

        def fail_walk(error: OSError) -> None:
            raise ValueError("schema_inventory_dataset_invalid") from error

        for parent, dirnames, filenames in layer_root.walk(
            follow_symlinks=False,
            on_error=fail_walk,
        ):
            require_unaliased_path(parent, error="schema_inventory_dataset_invalid")
            dirnames.sort()
            for name in dirnames:
                require_unaliased_path(
                    parent / name,
                    error="schema_inventory_dataset_invalid",
                )
            for name in sorted(filenames):
                if name != "spec.json":
                    continue
                spec_path = parent / name
                require_unaliased_path(
                    spec_path, error="schema_inventory_dataset_invalid"
                )
                yield parent


def _inventory_item(
    project_root: Path,
    artifact_root: Path,
    *,
    important_null_count_columns: frozenset[str],
    resource_policy: RuntimeResourcePolicy,
    retained_items: ExitStack,
) -> tuple[dict[str, Any], int]:
    with open_dataset_session(
        artifact_root,
        columns=None,
        time_window=None,
        require_consumable=False,
        resource_policy=resource_policy,
    ) as session:
        return _inventory_item_from_session(
            project_root,
            artifact_root,
            session=session,
            important_null_count_columns=important_null_count_columns,
            resource_policy=resource_policy,
            retained_items=retained_items,
        )


def _inventory_item_from_session(
    project_root: Path,
    artifact_root: Path,
    *,
    session: DatasetSession,
    important_null_count_columns: frozenset[str],
    resource_policy: RuntimeResourcePolicy,
    retained_items: ExitStack,
    retained_payload_bytes: int = 0,
) -> tuple[dict[str, Any], int]:
    inspection = inspect_dataset_session(
        session,
        resource_policy=resource_policy,
        retained_payload_bytes=retained_payload_bytes,
    )
    logical_root = artifact_root.relative_to(project_root).as_posix()
    spec_payload = inspection.spec
    warnings: list[str] = []
    columns = inspection.columns
    primary_key = tuple(spec_payload.get("primary_key", ()))
    partition_columns = tuple(spec_payload.get("partition_columns", ()))
    if not primary_key:
        primary_key = _infer_primary_key(columns)
        warnings.append(
            "primary_key_inferred" if primary_key else "primary_key_missing"
        )
    build_bound = _inventory_item_upper_bound_bytes(
        logical_root=logical_root,
        columns=columns,
        primary_key=primary_key,
        partition_columns=partition_columns,
        important_columns=important_null_count_columns.intersection(columns),
        date_min=inspection.date_min,
        date_max=inspection.date_max,
    )
    try:
        retained_items.enter_context(
            resource_policy.reserve("schema.inventory.items", build_bound)
        )
        stats = _aggregate_dataset_stats(
            session,
            columns,
            primary_key,
            important_null_count_columns=important_null_count_columns,
            resource_policy=resource_policy,
            exact_duplicate_count=int(
                inspection.manifest["duplicate_primary_key_count"]
            ),
            retained_payload_bytes=(
                retained_payload_bytes
                + dataset_inspection_retained_memory_bytes(inspection)
            ),
        )
        session.verify()
        layer, _, kind, artifact_id = _logical_identifiers(logical_root)
        item = {
            "artifact_id": artifact_id,
            "layer": layer,
            "kind": kind,
            "logical_relative_root": logical_root,
            "columns": list(columns),
            "dtypes": stats["dtypes"],
            "primary_key": list(primary_key),
            "partition_columns": list(partition_columns),
            "file_count": inspection.file_count,
            "row_count": inspection.row_count,
            "row_count_by_year": stats["row_count_by_year"],
            "min_timestamp": inspection.date_min,
            "max_timestamp": inspection.date_max,
            "duplicate_primary_key_count": stats["duplicate_primary_key_count"],
            "null_key_counts": stats["null_key_counts"],
            "selected_important_column_null_counts": stats[
                "selected_important_column_null_counts"
            ],
            "schema_hash": _schema_hash(
                columns=columns,
                dtypes=stats["dtypes"],
                primary_key=primary_key,
                partition_columns=partition_columns,
            ),
            "warnings": sorted(set(warnings)),
        }
        try:
            bounded_rendered_qrp_json_object_size(
                item,
                lazy_arrays={},
                max_bytes=build_bound,
            )
        except (MemoryError, ValueError):
            raise ValueError(
                "schema_inventory_payload_exceeds_task_memory_budget"
            ) from None
        return item, build_bound
    except MemoryError:
        raise ValueError(
            "schema_inventory_payload_exceeds_task_memory_budget"
        ) from None


def _inventory_item_upper_bound_bytes(
    *,
    logical_root: str,
    columns: tuple[str, ...],
    primary_key: tuple[str, ...],
    partition_columns: tuple[str, ...],
    important_columns: frozenset[str],
    date_min: str | None,
    date_max: str | None,
) -> int:
    year_count = 0
    if date_min is not None and date_max is not None:
        year_count = date.fromisoformat(date_max).year - date.fromisoformat(
            date_min
        ).year + 1
    text_bytes = len(logical_root.encode("utf-8")) + sum(
        len(value.encode("utf-8"))
        for value in (*columns, *primary_key, *partition_columns, *important_columns)
    )
    item_entries = 2 * len(columns) + 2 * len(primary_key) + len(
        important_columns
    ) + year_count
    return 16 * 1024 + 4 * text_bytes + 512 * item_entries


def _aggregate_dataset_stats(
    session: DatasetSession,
    columns: tuple[str, ...],
    primary_key: tuple[str, ...],
    *,
    important_null_count_columns: frozenset[str],
    exact_duplicate_count: int,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int,
) -> dict[str, Any]:
    dtypes: dict[str, str] = {}
    row_count = 0
    row_count_by_year: dict[str, int] = {}
    null_key_counts = {column: 0 for column in primary_key}
    important_columns = tuple(
        column for column in columns if column in important_null_count_columns
    )
    important_null_counts = {column: 0 for column in important_columns}
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None
    duplicate_primary_key_count = exact_duplicate_count
    chunks = iter_dataset_chunks(
        session,
        target_chunk_bytes=resource_policy.remaining_memory_bytes,
        operation_headroom_bytes=(
            resource_policy.task_memory_budget_bytes - retained_payload_bytes
        ),
    )
    with closing_dataset_iterator(chunks):
        for frame in chunks:
            row_count += int(len(frame))
            for column in frame.columns:
                dtypes.setdefault(str(column), str(frame[column].dtype))
            if "timestamp" in frame.columns:
                timestamps = pd.to_datetime(
                    frame["timestamp"], errors="coerce"
                ).dropna()
                if not timestamps.empty:
                    year_counts = timestamps.dt.year.astype(str).value_counts()
                    for year, count in year_counts.items():
                        row_count_by_year[year] = row_count_by_year.get(year, 0) + int(
                            count
                        )
                    chunk_min = timestamps.min()
                    chunk_max = timestamps.max()
                    min_timestamp = (
                        chunk_min
                        if min_timestamp is None
                        else min(min_timestamp, chunk_min)
                    )
                    max_timestamp = (
                        chunk_max
                        if max_timestamp is None
                        else max(max_timestamp, chunk_max)
                    )
            for column in primary_key:
                null_key_counts[column] += int(frame[column].isna().sum())
            for column in important_columns:
                important_null_counts[column] += int(frame[column].isna().sum())

    return {
        "dtypes": {column: dtypes.get(column, "") for column in columns},
        "row_count": row_count,
        "row_count_by_year": dict(sorted(row_count_by_year.items())),
        "min_timestamp": _format_timestamp(min_timestamp),
        "max_timestamp": _format_timestamp(max_timestamp),
        "null_key_counts": dict(sorted(null_key_counts.items())),
        "duplicate_primary_key_count": duplicate_primary_key_count,
        "selected_important_column_null_counts": dict(
            sorted(important_null_counts.items())
        ),
    }


def _logical_identifiers(logical_root: str) -> tuple[str, str, str, str]:
    parts = logical_root.split("/")
    if len(parts) < 2 or parts[0] != "data" or any(not part for part in parts):
        raise ValueError("invalid_inventory_root")
    layer = parts[1]
    if layer == "interim" and len(parts) == 4:
        return layer, parts[2], "interim", parts[3]
    if layer == "processed" and len(parts) == 5:
        return layer, parts[2], parts[3], parts[4]
    if layer == "research" and len(parts) == 4:
        return layer, parts[2], "research_dataset", parts[3]
    raise ValueError("invalid_inventory_root")


def _infer_primary_key(columns: tuple[str, ...]) -> tuple[str, ...]:
    canonical_key = ("market_id", "asset_id", "timestamp")
    source_asset_timestamp_key = ("source_asset_id", "timestamp")
    source_asset_key = ("source_asset_id",)
    calendar_key = ("timestamp",)
    for candidate in (
        canonical_key,
        source_asset_timestamp_key,
        source_asset_key,
        calendar_key,
    ):
        if all(column in columns for column in candidate):
            return candidate
    return ()


def _format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.date().isoformat()


def _schema_hash(
    *,
    columns: tuple[str, ...],
    dtypes: dict[str, str],
    primary_key: tuple[str, ...],
    partition_columns: tuple[str, ...],
) -> str:
    return canonical_json_hash(
        {
            "columns": list(columns),
            "dtypes": dtypes,
            "primary_key": list(primary_key),
            "partition_columns": list(partition_columns),
        },
    )


__all__ = [
    "open_schema_inventory_payload",
    "summarize_schema_inventory_payload",
    "validate_schema_inventory_payload",
]
