"""Compatibility entry point for the CSMAR interim importer.

The implementation lives in ``sources.csmar.importer`` and writes only through
spec/catalog-driven storage APIs.
"""

from __future__ import annotations

from quant_research_platform.sources.csmar.importer import (
    DEFAULT_INTERIM_DATASETS,
    InterimImportResult,
    build_interim_frame,
    import_interim_data,
    main,
)

__all__ = [
    "DEFAULT_INTERIM_DATASETS",
    "InterimImportResult",
    "build_interim_frame",
    "import_interim_data",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
