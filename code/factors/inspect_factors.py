"""Inspect the full-sample slim factor Parquet file."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from validation.common import has_blocking_issues  # noqa: E402
from validation.factor_tables import check_factor_table  # noqa: E402


def _print_report_summary(report) -> None:
    print(
        f"{report.file}: rows={report.row_count} cols={report.column_count} "
        f"dates={report.min_date}..{report.max_date} issues={len(report.issues)}"
    )
    print(
        f"  factors={report.stats.get('factor_count')} "
        f"labels={report.stats.get('label_count')}"
    )
    for issue in report.issues:
        print(
            f"  {issue.severity}: {issue.check} count={issue.count} "
            f"{issue.message}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the full-sample slim factor Parquet file."
    )
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args(argv)

    processed_root = Path(args.processed_root)
    path = (
        processed_root
        / "price_volume_factors_full"
        / f"price_volume_factors_{args.start_year}_{args.end_year}.parquet"
    )
    if not path.exists():
        print(f"ERROR: missing factor file: {path}")
        return 2
    frame = pd.read_parquet(path, engine="pyarrow")
    report = check_factor_table(
        frame,
        expected_year=None,
        file=path.relative_to(processed_root).as_posix(),
    )
    _print_report_summary(report)
    if has_blocking_issues(report.issues):
        print("BLOCKING factor issues found.")
        return 1
    years = sorted(pd.to_datetime(frame["trade_date"]).dt.year.unique().tolist())
    expected_years = list(range(args.start_year, args.end_year + 1))
    if years != expected_years:
        print(
            "BLOCKING factor issues found: "
            f"expected years {expected_years}, found {years}."
        )
        return 1
    print("No blocking factor issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
