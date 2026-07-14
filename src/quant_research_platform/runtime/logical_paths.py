"""Logical path helpers for generated QRP artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

DEFAULT_GENERATED_MARKERS: tuple[str, ...] = (
    "data",
    "outputs",
    "logs",
    "handoff_evidence",
)
RUN_RESULT_LOGICAL_PATH_MARKERS: tuple[str, ...] = (
    "data",
    "outputs",
    "logs",
    "handoff_evidence",
)
RUN_RESULT_DATASET_LAYERS = frozenset({"interim", "processed", "research"})


def logical_path(
    path: str | Path,
    *,
    run_root: str | Path | None = None,
    markers: Iterable[str] = DEFAULT_GENERATED_MARKERS,
) -> str:
    """Return a sanitized logical path relative to a generated root when possible."""

    item = Path(path)
    if run_root is not None:
        try:
            return item.relative_to(Path(run_root)).as_posix()
        except ValueError:
            pass
    parts = _posix_parts(item)
    marker_set = {str(marker) for marker in markers}
    marker_index = next(
        (
            index
            for index in range(len(parts) - 1, -1, -1)
            if parts[index] in marker_set
        ),
        None,
    )
    if marker_index is not None:
        return PurePosixPath(*parts[marker_index:]).as_posix()
    if item.is_absolute():
        return item.name
    return item.as_posix()


def dataset_layer_from_path(path: str | Path) -> str | None:
    """Infer a dataset logical layer from the last ``data/<layer>`` root."""

    parts = _posix_parts(Path(path))
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "data":
            layer = parts[index + 1]
            if layer in {"raw", "interim", "processed", "research"}:
                return layer
    return None


def _posix_parts(path: Path) -> tuple[str, ...]:
    text = path.as_posix()
    if ":" in text:
        text = text.split(":", 1)[1]
    return tuple(part for part in PurePosixPath(text).parts if part not in {"", "/"})


__all__ = [
    "DEFAULT_GENERATED_MARKERS",
    "RUN_RESULT_DATASET_LAYERS",
    "RUN_RESULT_LOGICAL_PATH_MARKERS",
    "dataset_layer_from_path",
    "logical_path",
]
