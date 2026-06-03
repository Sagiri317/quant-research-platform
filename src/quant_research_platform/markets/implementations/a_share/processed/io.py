"""A-share processed storage helpers."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.storage.layout import StorageLayout


def layout_from_processed_root(processed_root: Path) -> StorageLayout:
    """Resolve ``data/processed/a_share`` back to the project storage layout."""

    root = Path(processed_root)
    parts = root.parts
    if len(parts) < 3 or parts[-3:] != ("data", "processed", "a_share"):
        raise ValueError("processed_root_must_be_data_processed_a_share")
    project_root = Path(*parts[:-3]) if len(parts) > 3 else Path(".")
    return StorageLayout(project_root=project_root)


__all__ = ["layout_from_processed_root"]
