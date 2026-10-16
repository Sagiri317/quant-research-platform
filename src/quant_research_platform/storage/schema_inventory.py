"""Read-only inventory for the legacy project data/{interim,processed,research} layout.

This diagnostic scans that layout only; it is not a dataset admission interface.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    render_qrp_json_object,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import (
    atomic_write_chunks,
    is_atomic_directory_staging_name,
)
from quant_research_platform.storage.datasets.inspection import (
    _inspection_runtime_metadata,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import physical_schema_columns
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_walk,
    require_unaliased_path,
)

_RUN_RESULT_DATASET_LAYERS = frozenset({"interim", "processed", "research"})


@contextmanager
def iter_schema_inventory_items(
    project_root: Path, *, important_null_count_columns: Collection[str] | None = None
) -> Iterator[Iterator[dict[str, Any]]]:
    """Yield one validated inventory item at a time."""

    root = Path(project_root)
    declared_important_columns = (
        () if important_null_count_columns is None else important_null_count_columns
    )
    if any(
        type(column) is not str or not column for column in declared_important_columns
    ):
        raise ValueError("schema_inventory_important_columns_invalid")
    important_columns = frozenset(declared_important_columns)
    data_root = root / "data"
    for candidate in (
        root,
        data_root,
        *(data_root / layer for layer in sorted(_RUN_RESULT_DATASET_LAYERS)),
    ):
        require_unaliased_path(candidate, error="schema_inventory_dataset_invalid")

    def generate() -> Iterator[dict[str, Any]]:
        for artifact_root in _iter_dataset_artifact_roots(root):
            yield _inventory_item(
                root,
                artifact_root,
                important_null_count_columns=important_columns,
            )

    with closing_iterator(generate()) as items:
        yield items


def write_schema_inventory(
    project_root: Path,
    output: Path,
    *,
    important_null_count_columns: Collection[str] | None = None,
) -> Path:
    """Stream a deterministic schema inventory through atomic publication."""

    def chunks() -> Iterator[str]:
        found = False
        yield '{\n  "datasets": ['
        with iter_schema_inventory_items(
            project_root, important_null_count_columns=important_null_count_columns
        ) as items:
            for item in items:
                rendered = render_qrp_json_object(item)
                prefix = "\n" if not found else ",\n"
                found = True
                chunk = prefix + "\n".join(
                    "  " + line for line in rendered.splitlines()
                )
                try:
                    yield chunk
                except GeneratorExit:
                    return
        if not found:
            raise ValueError("schema_inventory_no_datasets_found")
        yield '\n  ],\n  "schema_id": "qrp-schema-inventory"\n}\n'

    return atomic_write_chunks(output, chunks(), encoding="utf-8")


def _iter_dataset_artifact_roots(project_root: Path) -> Iterable[Path]:
    data_root = project_root / "data"
    require_unaliased_path(project_root, error="schema_inventory_dataset_invalid")
    require_unaliased_path(data_root, error="schema_inventory_dataset_invalid")
    allowed_roots = tuple(
        data_root / layer for layer in sorted(_RUN_RESULT_DATASET_LAYERS)
    )
    for layer_root in allowed_roots:
        require_unaliased_path(layer_root, error="schema_inventory_dataset_invalid")
        if not filesystem_exists(layer_root):
            continue

        def fail_walk(error: OSError) -> None:
            raise ValueError("schema_inventory_dataset_invalid") from error

        roots = []
        for parent, dirnames, filenames in filesystem_walk(
            layer_root,
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
            dirnames[:] = [
                name for name in dirnames if not is_atomic_directory_staging_name(name)
            ]
            for name in sorted(filenames):
                if name != "spec.json":
                    continue
                spec_path = parent / name
                require_unaliased_path(
                    spec_path, error="schema_inventory_dataset_invalid"
                )
                roots.append(parent)
        yield from sorted(
            roots, key=lambda path: path.relative_to(project_root).as_posix()
        )


def _inventory_item(
    project_root: Path,
    artifact_root: Path,
    *,
    important_null_count_columns: frozenset[str],
) -> dict[str, Any]:
    session = open_dataset_session(artifact_root, columns=None, time_window=None)
    return _inventory_item_from_session(
        project_root,
        artifact_root,
        session=session,
        important_null_count_columns=important_null_count_columns,
    )


def _inventory_item_from_session(
    project_root: Path,
    artifact_root: Path,
    *,
    session: DatasetSession,
    important_null_count_columns: frozenset[str],
) -> dict[str, Any]:
    columns = physical_schema_columns(session.schema)
    spec_payload = session.spec
    primary_key = tuple(spec_payload["primary_key"])
    partition_columns = tuple(spec_payload["partition_columns"])
    important_columns = tuple(
        column for column in columns if column in important_null_count_columns
    )
    scanned_null_counts = {
        column: 0
        for column in important_columns
        if column not in primary_key
        and spec_payload["column_roles"][column] not in {"factor", "label"}
    }
    row_count_by_year: dict[str, int] = {}
    timestamp_null_count = 0

    def observe(frame):
        nonlocal timestamp_null_count
        if session.timestamp_column is not None:
            timestamps = strict_timestamp_series(
                frame[session.timestamp_column],
                error="schema_inventory_invalid_timestamp",
                allow_null=True,
            ).dropna()
            timestamp_null_count += len(frame) - len(timestamps)
            for year, count in timestamps.dt.year.value_counts().items():
                year = str(year)
                row_count_by_year[year] = row_count_by_year.get(year, 0) + int(count)
        for column in scanned_null_counts:
            scanned_null_counts[column] += int(frame[column].isna().sum())

    inspection = _inspection_runtime_metadata(session, frame_observer=observe)
    if timestamp_null_count:
        raise ValueError("schema_inventory_invalid_timestamp")
    manifest = session.manifest
    verified_null_counts = {
        **manifest["null_count_summary"],
        **{
            column: manifest[f"{role}_nan_profile"].get(column, 0)
            for column, role in spec_payload["column_roles"].items()
            if role in {"factor", "label"}
        },
        **scanned_null_counts,
    }
    dtypes = {
        descriptor["name"]: (
            "category" if descriptor["kind"] == "categorical" else descriptor["dtype"]
        )
        for descriptor in session.logical_dtypes["columns"]
    }
    dtypes = {column: dtypes[column] for column in columns}
    logical_root = artifact_root.relative_to(project_root).as_posix()
    layer, _, kind, artifact_id = _logical_identifiers(logical_root)
    return {
        "artifact_id": artifact_id,
        "layer": layer,
        "kind": kind,
        "logical_relative_root": logical_root,
        "columns": list(columns),
        "dtypes": dtypes,
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "file_count": inspection["file_count"],
        "row_count": inspection["row_count"],
        "row_count_by_year": dict(sorted(row_count_by_year.items())),
        "min_timestamp": inspection["date_min"],
        "max_timestamp": inspection["date_max"],
        "duplicate_primary_key_count": int(manifest["duplicate_primary_key_count"]),
        "null_key_counts": dict(sorted(manifest["null_count_summary"].items())),
        "selected_important_column_null_counts": {
            column: verified_null_counts[column] for column in sorted(important_columns)
        },
        "schema_hash": _schema_hash(
            columns=columns,
            dtypes=dtypes,
            primary_key=primary_key,
            partition_columns=partition_columns,
        ),
        "warnings": [],
    }


def _logical_identifiers(logical_root: str) -> tuple[str, str, str, str]:
    parts = tuple(logical_root.split("/"))
    if len(parts) < 2 or parts[0] != "data" or any(not part for part in parts):
        raise ValueError("invalid_inventory_root")
    layer = parts[1]
    if layer in {"interim", "research"} and len(parts) >= 4:
        kind = "interim" if layer == "interim" else "research_dataset"
        artifact_id = parts[3]
    elif layer == "processed" and len(parts) >= 5:
        kind = parts[3]
        artifact_id = parts[4]
    else:
        raise ValueError("invalid_inventory_root")
    return layer, parts[2], kind, artifact_id


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
