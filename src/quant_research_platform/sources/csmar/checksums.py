"""Internal provider raw-export checksum and metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from quant_research_platform.sources.csmar.raw_schemas import (
    RAW_TABLE_CONTRACTS,
    parse_raw_filename,
    read_export_header,
)
from quant_research_platform.storage.checksums import sha256_file


def _format_metadata_date(value: str) -> str:
    """Format compact ``YYYYMMDD`` metadata dates."""

    if value == "full":
        return value
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def _count_csv_rows(path: Path) -> int:
    """Count CSV-like export rows without materializing the full table."""

    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def _file_sha256(path: Path) -> str:
    """Return the raw export file SHA-256 digest."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_export_metadata(
    path: str | Path,
    *,
    filters: Mapping[str, object] | None = None,
    notes: str | None = None,
) -> dict[str, object]:
    """Build the approved private metadata payload for one raw export."""

    export_path = Path(path)
    filename_metadata = parse_raw_filename(export_path)
    dataset = filename_metadata.dataset
    contract = RAW_TABLE_CONTRACTS[dataset]
    columns = read_export_header(export_path)
    row_count = _count_csv_rows(export_path) if export_path.suffix == ".csv" else None
    payload: dict[str, object] = {
        "source": "CSMAR",
        "account_type": "school_authorized_web_export",
        "exported_by": "user",
        "export_date": _format_metadata_date(filename_metadata.export_date),
        "dataset": dataset,
        "table_code": contract.table_code,
        "table_name_zh": contract.table_name_zh,
        "frequency": "daily"
        if dataset in {"daily_market", "adjustment", "index_market"}
        else "event_or_static",
        "market_scope": filename_metadata.market_scope,
        "start_date": _format_metadata_date(filename_metadata.start_date),
        "end_date": _format_metadata_date(filename_metadata.end_date),
        "part": filename_metadata.part,
        "filters": dict(filters)
        if filters is not None
        else {
            "share_class": "A-share",
            "boards": ["main", "chinext", "star"],
            "bjse": "excluded_in_first_version",
        },
        "fields_zh": {
            column: contract.chinese_meanings.get(column, "") for column in columns
        },
        "fields_raw": list(columns),
        "row_count": row_count,
        "file_sha256": _file_sha256(export_path),
        "license_boundary": ("private research only; do not commit raw vendor data"),
        "notes": notes
        or "No credentials, cookies, tokens, or school account information stored.",
    }
    return payload


def raw_export_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest for one raw export file."""

    return sha256_file(Path(path))


__all__ = ["build_raw_export_metadata", "raw_export_sha256"]
