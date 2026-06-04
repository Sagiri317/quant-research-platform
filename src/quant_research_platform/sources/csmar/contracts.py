"""Thin compatibility facade for internal provider contracts.

Implementation ownership lives in ``raw_schemas.py``, ``field_mappings.py``,
and ``checksums.py``. This module is internal-only and is not imported by
API/CLI public data-product surfaces.
"""

from quant_research_platform.sources.csmar.checksums import (
    build_raw_export_metadata,
    raw_export_sha256,
)
from quant_research_platform.sources.csmar.field_mappings import (
    map_csmar_columns,
    missing_raw_column_names,
    missing_raw_columns,
    normalize_stock_code,
    normalize_trdnew_capchg,
    normalize_trdnew_daily_market,
)
from quant_research_platform.sources.csmar.raw_schemas import (
    DATASET_CONTRACTS,
    FILENAME_RE,
    NATIVE_TABLE_DATASETS,
    RAW_TABLE_CONTRACTS,
    DatasetContract,
    RawExportInspection,
    RawFileMetadata,
    RawTableContract,
    assert_required_columns,
    default_metadata_path,
    duplicate_primary_keys,
    infer_dataset_from_native_export,
    inspect_raw_export_file,
    missing_required_columns,
    negative_liquidity_rows,
    ohlc_violations,
    parse_raw_filename,
    read_export_header,
)

__all__ = [
    "DATASET_CONTRACTS",
    "FILENAME_RE",
    "NATIVE_TABLE_DATASETS",
    "RAW_TABLE_CONTRACTS",
    "DatasetContract",
    "RawExportInspection",
    "RawFileMetadata",
    "RawTableContract",
    "assert_required_columns",
    "build_raw_export_metadata",
    "default_metadata_path",
    "duplicate_primary_keys",
    "infer_dataset_from_native_export",
    "inspect_raw_export_file",
    "map_csmar_columns",
    "missing_raw_column_names",
    "missing_raw_columns",
    "missing_required_columns",
    "negative_liquidity_rows",
    "normalize_stock_code",
    "normalize_trdnew_capchg",
    "normalize_trdnew_daily_market",
    "ohlc_violations",
    "parse_raw_filename",
    "raw_export_sha256",
    "read_export_header",
]
