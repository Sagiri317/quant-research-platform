"""Inspect private CSMAR interim tables without writing reports."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research_platform.markets.a_share.validation.common import (
    has_blocking_issues,  # noqa: E402
)
from quant_research_platform.markets.a_share.validation.internal_tables import (  # noqa: E402
    check_benchmark_index_market,
    check_daily_market,
)


def _read_interim_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"stock_code": str, "index_code": str},
        encoding="utf-8-sig",
        low_memory=False,
    )


def _print_report_summary(report) -> None:
    print(
        f"{report.dataset} {report.file}: rows={report.row_count} "
        f"cols={report.column_count} dates={report.min_date}..{report.max_date} "
        f"issues={len(report.issues)}"
    )
    for issue in report.issues:
        print(
            f"  {issue.severity}: {issue.check} count={issue.count} "
            f"{issue.message}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect CSMAR interim tables.")
    parser.add_argument("--interim-root", default="data/interim_private/csmar")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args(argv)

    interim_root = Path(args.interim_root)
    if not interim_root.exists():
        print(f"ERROR: interim root does not exist: {interim_root}")
        return 2

    reports = []
    for year in range(args.start_year, args.end_year + 1):
        daily_file = interim_root / "daily_market_by_year" / f"daily_market_{year}.csv"
        benchmark_file = (
            interim_root
            / "benchmark_index_market_by_year"
            / f"benchmark_index_market_{year}.csv"
        )
        if not daily_file.exists():
            print(f"ERROR: missing daily market file: {daily_file}")
            return 2
        if not benchmark_file.exists():
            print(f"ERROR: missing benchmark index file: {benchmark_file}")
            return 2
        reports.append(
            check_daily_market(
                _read_interim_csv(daily_file),
                expected_year=year,
                file=daily_file.relative_to(interim_root).as_posix(),
            )
        )
        reports.append(
            check_benchmark_index_market(
                _read_interim_csv(benchmark_file),
                expected_year=year,
                file=benchmark_file.relative_to(interim_root).as_posix(),
            )
        )

    for report in reports:
        _print_report_summary(report)
    all_issues = [issue for report in reports for issue in report.issues]
    if has_blocking_issues(all_issues):
        print("BLOCKING interim issues found.")
        return 1
    print("No blocking interim issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
