"""CSMAR raw-to-interim parquet importer.

The importer owns orchestration only. It obtains ``InterimDataSpec`` objects
from the CSMAR catalog and persists parquet through ``storage.parquet``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.sources.csmar.catalog import csmar_interim_catalog
from quant_research_platform.sources.csmar.columns import (
    apply_interim_column_rules,
    load_column_overrides,
    render_column_pruning_report,
)
from quant_research_platform.sources.csmar.mappers import (
    filter_by_date_range,
    find_dataset_files,
    iter_private_export_chunks,
    normalize_private_export_frame,
    with_benchmark_index_fields,
)
from quant_research_platform.sources.csmar.raw_schemas import parse_raw_filename
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.parquet import write_interim_data

DEFAULT_INTERIM_DATASETS: tuple[str, ...] = (
    "daily_market",
    "benchmark_index_market",
    "stock_master",
    "adjustment",
    "market_cap",
    "index_info",
)
TIME_RANGE_FILTER_DATASETS = {"daily_market", "index_market", "benchmark_index_market"}
END_ONLY_FILTER_DATASETS = {"market_cap"}
START_END_FILTER_DATASETS = {"adjustment"}


@dataclass(frozen=True)
class InterimImportResult:
    """Summary of a CSMAR interim import run."""

    written_data_roots: dict[str, Path]
    skipped: tuple[str, ...]
    pruning_report_text: str


def build_csmar_interim_frame(
    *,
    interim_data_id: str,
    input_files: list[Path],
    start_date: str | None,
    end_date: str | None,
    overrides: dict[str, dict[str, object]] | None = None,
    chunksize: int = 500_000,
    pruning_reports: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Build one canonical interim frame without persisting output."""

    raw_dataset = _raw_dataset_for_interim(interim_data_id)
    frames: list[pd.DataFrame] = []
    for input_file in sorted(input_files):
        for chunk in iter_private_export_chunks(
            input_file,
            dataset=raw_dataset,
            chunksize=chunksize,
        ):
            normalized = normalize_private_export_frame(
                chunk,
                raw_dataset,
                fallback_timestamp=_fallback_timestamp_for_file(
                    input_file,
                    raw_dataset,
                ),
            )
            normalized = _filter_normalized_frame(
                normalized,
                interim_data_id=interim_data_id,
                start_date=start_date,
                end_date=end_date,
            )
            if interim_data_id == "benchmark_index_market":
                normalized = with_benchmark_index_fields(normalized)
            if normalized.empty:
                continue
            pruned, report = apply_interim_column_rules(
                normalized,
                interim_data_id,
                overrides=overrides,
            )
            if pruning_reports is not None:
                pruning_reports.append(report)
            frames.append(pruned)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def import_interim_data(
    *,
    layout: StorageLayout,
    raw_root: Path | None = None,
    interim_data_ids: tuple[str, ...] = DEFAULT_INTERIM_DATASETS,
    start_date: str | None = "2014-01-01",
    end_date: str | None = "2025-12-31",
    override_path: Path | None = None,
    chunksize: int = 500_000,
) -> InterimImportResult:
    """Build and persist CSMAR interim parquet datasets through catalog specs."""

    catalog = csmar_interim_catalog()
    source_raw_root = raw_root or layout.raw_csmar_root
    overrides = load_column_overrides(override_path) if override_path else None
    written: dict[str, Path] = {}
    skipped: list[str] = []
    pruning_reports: list[dict[str, object]] = []

    for interim_data_id in interim_data_ids:
        catalog.get(interim_data_id)
        input_files = find_dataset_files(
            source_raw_root,
            _raw_dataset_for_interim(interim_data_id),
        )
        if not input_files:
            skipped.append(interim_data_id)
            continue
        frame = build_csmar_interim_frame(
            interim_data_id=interim_data_id,
            input_files=input_files,
            start_date=start_date,
            end_date=end_date,
            overrides=overrides,
            chunksize=chunksize,
            pruning_reports=pruning_reports,
        )
        if frame.empty:
            skipped.append(interim_data_id)
            continue
        written[interim_data_id] = write_interim_data(
            frame,
            layout=layout,
            catalog=catalog,
            interim_data_id=interim_data_id,
        )

    return InterimImportResult(
        written_data_roots=written,
        skipped=tuple(skipped),
        pruning_report_text=render_column_pruning_report(pruning_reports),
    )


def _raw_dataset_for_interim(interim_data_id: str) -> str:
    if interim_data_id == "benchmark_index_market":
        return "index_market"
    return interim_data_id


def _fallback_timestamp_for_file(
    input_file: Path,
    raw_dataset: str,
) -> pd.Timestamp | None:
    if raw_dataset != "index_info":
        return None
    try:
        metadata = parse_raw_filename(input_file)
    except ValueError:
        return None
    return pd.Timestamp(metadata.export_date)


def _filter_normalized_frame(
    frame: pd.DataFrame,
    *,
    interim_data_id: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if interim_data_id in TIME_RANGE_FILTER_DATASETS:
        return filter_by_date_range(
            frame,
            dataset=_raw_dataset_for_interim(interim_data_id),
            start_date=start_date,
            end_date=end_date,
        )
    if interim_data_id in START_END_FILTER_DATASETS:
        return filter_by_date_range(
            frame,
            dataset=interim_data_id,
            start_date=start_date,
            end_date=end_date,
        )
    if interim_data_id in END_ONLY_FILTER_DATASETS:
        return filter_by_date_range(
            frame,
            dataset=interim_data_id,
            start_date=None,
            end_date=end_date,
        )
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local private CSMAR interim parquet datasets."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-root")
    parser.add_argument("--override-path")
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--dataset", action="append", dest="datasets")
    args = parser.parse_args(argv)

    layout = StorageLayout(project_root=Path(args.project_root))
    raw_root = Path(args.raw_root) if args.raw_root else None
    if raw_root is not None and not raw_root.exists():
        print("ERROR: raw root does not exist")
        return 2
    if raw_root is None and not layout.raw_csmar_root.exists():
        print("ERROR: raw root does not exist")
        return 2

    result = import_interim_data(
        layout=layout,
        raw_root=raw_root,
        interim_data_ids=tuple(args.datasets or DEFAULT_INTERIM_DATASETS),
        start_date=args.start_date,
        end_date=args.end_date,
        override_path=Path(args.override_path) if args.override_path else None,
        chunksize=args.chunksize,
    )
    print(f"written_interim_datasets={len(result.written_data_roots)}")
    print(f"skipped_interim_datasets={len(result.skipped)}")
    return 0


__all__ = [
    "DEFAULT_INTERIM_DATASETS",
    "InterimImportResult",
    "build_csmar_interim_frame",
    "import_interim_data",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
