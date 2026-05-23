"""估算完整私有 CSMAR processed Parquet 生成是否安全。"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .storage import (  # noqa: E402
    GIB,
    collect_interim_daily_rows,
    collect_observed_processed_bytes,
    collect_observed_processed_rows,
    evaluate_space_gate,
    format_gib,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate whether full CSMAR processed Parquet generation is safe."
    )
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
    parser.add_argument("--interim-root", default="data/interim_private/csmar")
    parser.add_argument("--observed-start-year", type=int, default=2014)
    parser.add_argument("--observed-end-year", type=int, default=2025)
    parser.add_argument("--target-start-year", type=int, default=2014)
    parser.add_argument("--target-end-year", type=int, default=2025)
    parser.add_argument(
        "--legacy-processed-gb",
        type=float,
        default=15.6,
        help="Old processed size baseline in GiB.",
    )
    parser.add_argument("--max-legacy-ratio", type=float, default=0.8)
    parser.add_argument("--max-free-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)

    processed_root = Path(args.processed_root)
    interim_root = Path(args.interim_root)
    target_years = range(args.target_start_year, args.target_end_year + 1)

    observed_bytes = collect_observed_processed_bytes(
        processed_root,
        start_year=args.observed_start_year,
        end_year=args.observed_end_year,
    )
    observed_rows = collect_observed_processed_rows(
        processed_root,
        start_year=args.observed_start_year,
        end_year=args.observed_end_year,
    )
    target_rows = collect_interim_daily_rows(interim_root, target_years)
    free_bytes = shutil.disk_usage(processed_root.parent).free
    result = evaluate_space_gate(
        observed_bytes=observed_bytes,
        observed_rows=observed_rows,
        target_rows=target_rows,
        free_bytes=free_bytes,
        legacy_processed_bytes=int(args.legacy_processed_gb * GIB),
        max_legacy_ratio=args.max_legacy_ratio,
        max_free_ratio=args.max_free_ratio,
    )

    print("CSMAR processed space gate")
    print(f"observed bytes: {format_gib(result.observed_bytes)}")
    print(f"observed rows: {result.observed_rows}")
    print(f"target rows: {result.target_rows}")
    print(f"estimated full processed: {format_gib(result.estimated_bytes)}")
    print(f"free disk: {format_gib(result.free_bytes)}")
    print(f"legacy processed baseline: {format_gib(result.legacy_processed_bytes)}")
    print(f"allowed: {result.allowed}")
    for reason in result.reasons:
        print(f"BLOCK: {reason}")
    return 0 if result.allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
