"""Dataset manifest runtime validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def manifest_has_runtime_metadata(manifest: Mapping[str, Any]) -> bool:
    return all(
        key in manifest
        for key in (
            "columns",
            "physical_schema",
            "schema_hash",
            "partition_row_counts",
            "file_count",
            "row_count",
        )
    )
