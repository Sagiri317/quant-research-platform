"""Read-only inventory under project data/{interim,processed,research}.

This diagnostic scans that layout only; it is not a dataset admission interface.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    render_qrp_json_object,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.atomic import (
    atomic_write_chunks,
    is_atomic_directory_staging_name,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_chunks,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import physical_schema_columns
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_walk,
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
                yield chunk
        if not found:
            raise ValueError("schema_inventory_no_datasets_found")
        yield '\n  ],\n  "schema_id": "qrp-schema-inventory"\n}\n'

    return atomic_write_chunks(output, chunks(), encoding="utf-8")


def _iter_dataset_artifact_roots(project_root: Path) -> Iterable[Path]:
    data_root = project_root / "data"
    allowed_roots = tuple(
        data_root / layer for layer in sorted(_RUN_RESULT_DATASET_LAYERS)
    )
    for layer_root in allowed_roots:
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
            dirnames.sort()
            dirnames[:] = [
                name for name in dirnames if not is_atomic_directory_staging_name(name)
            ]
            for name in sorted(filenames):
                if name != "spec.json":
                    continue
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
    manifest = session.manifest
    row_count_by_year = {
        partition.removeprefix("year="): count
        for partition, count in manifest["partition_row_counts"].items()
        if partition.startswith("year=") and count
    }
    time_column = session.timestamp_column if not partition_columns else None
    scan_columns = tuple(
        dict.fromkeys(
            (
                *((time_column,) if time_column is not None else ()),
                *scanned_null_counts,
            )
        )
    )
    if scan_columns:
        with closing_iterator(
            iter_dataset_chunks(session.projected_view(scan_columns))
        ) as frames:
            for frame in frames:
                if time_column is not None:
                    for year, count in (
                        frame[time_column].dt.year.value_counts().items()
                    ):
                        year = str(year)
                        row_count_by_year[year] = row_count_by_year.get(year, 0) + int(
                            count
                        )
                for column in scanned_null_counts:
                    scanned_null_counts[column] += int(frame[column].isna().sum())
    null_counts = {
        **dict.fromkeys(primary_key, 0),
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
    layer = artifact_root.relative_to(project_root / "data").parts[0]
    kind = (
        spec_payload["kind"]
        if layer == "processed"
        else "research_dataset"
        if layer == "research"
        else "interim"
    )
    return {
        "artifact_id": spec_payload["data_id"],
        "layer": layer,
        "kind": kind,
        "logical_relative_root": logical_root,
        "columns": list(columns),
        "dtypes": dtypes,
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "file_count": manifest["file_count"],
        "row_count": manifest["row_count"],
        "row_count_by_year": dict(sorted(row_count_by_year.items())),
        "min_timestamp": manifest["date_min"],
        "max_timestamp": manifest["date_max"],
        "selected_important_column_null_counts": {
            column: null_counts[column] for column in sorted(important_columns)
        },
    }
