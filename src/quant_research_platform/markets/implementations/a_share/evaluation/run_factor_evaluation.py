"""Run evaluation artifacts from private CSMAR processed factor Parquet files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluation_adapter import (  # noqa: E402
    DATE_COLUMN,
    DEFAULT_RETURN_SOURCE_COLUMN,
    parse_columns,
    run_processed_factor_evaluation,
    schema_columns,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate private CSMAR slim factor Parquet through evaluation layer."
        )
    )
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
    parser.add_argument(
        "--input-file",
        default="",
        help="Backend/local CLI-only Parquet input override; not exposed by API/UI.",
    )
    parser.add_argument(
        "--report-root",
        default=None,
        help=(
            "Output root. Defaults to "
            "reports/private_real/factor_evaluation/full_<start-year>_<end-year>."
        ),
    )
    parser.add_argument("--start-year", type=int, default=2024)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--labels", nargs="+", default=["forward_return_20d"])
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--n-groups", type=int, default=5)
    parser.add_argument("--return-source-column", default=DEFAULT_RETURN_SOURCE_COLUMN)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--portfolio-modes",
        nargs="+",
        choices=("long_short", "long_only"),
        default=["long_short"],
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=("dataframe", "tensor"),
        default=["dataframe"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--segment-columns", default="")
    parser.add_argument("--risk-column", default="")
    parser.add_argument("--neutralization-method", default="none")
    parser.add_argument("--neutralization-group-columns", default="")
    parser.add_argument("--neutralization-exposure-columns", default="")
    args = parser.parse_args(argv)

    report_root = (
        Path(args.report_root)
        if args.report_root
        else Path("reports/private_real/factor_evaluation")
        / f"full_{args.start_year}_{args.end_year}"
    )
    input_file = Path(args.input_file) if args.input_file else None
    requested_labels = tuple(args.labels)
    requested_features = tuple(args.features)
    if args.dry_run:
        try:
            path = input_file
            if path is None:
                from quant_research_platform.runtime.paths import (
                    factor_full_path_covering_range,
                )

                path = factor_full_path_covering_range(
                    Path(args.processed_root),
                    args.start_year,
                    args.end_year,
                )
            columns = set(schema_columns(path))
            required = {
                "trade_date",
                "stock_code",
                *requested_features,
                *requested_labels,
                args.return_source_column,
                *parse_columns(args.segment_columns),
                args.risk_column,
                *parse_columns(args.neutralization_group_columns),
                *parse_columns(args.neutralization_exposure_columns),
                "eligible",
                "can_enter",
                "can_exit",
            }
            required.discard("")
            required.discard(None)
            missing = sorted(column for column in required if column not in columns)
            if missing:
                print(f"ERROR: missing required columns: {missing}")
                return 2
            coverage_start = args.start_date or f"{args.start_year}-01-01"
            coverage_end = args.end_date or f"{args.end_year}-12-31"
            rows_in_range = _count_trade_date_rows(
                path,
                start_date=coverage_start,
                end_date=coverage_end,
            )
            if rows_in_range <= 0:
                print(
                    "ERROR: no rows in requested date coverage: "
                    f"start_date={coverage_start} end_date={coverage_end}"
                )
                return 2
            combination_count = (
                len(requested_features)
                * len(requested_labels)
                * len(args.kernels)
                * len(args.portfolio_modes)
            )
            print(
                "Dry run passed: "
                f"input_file={path} "
                f"date_coverage={coverage_start}..{coverage_end} "
                f"rows_in_range={rows_in_range} "
                f"combinations={combination_count} "
                f"output_root={report_root}"
            )
            return 0
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2
    _, manifest_path, run_records = run_processed_factor_evaluation(
        processed_root=Path(args.processed_root),
        report_root=report_root,
        input_file=input_file,
        start_year=args.start_year,
        end_year=args.end_year,
        requested_labels=requested_labels,
        requested_factors=requested_features,
        all_factors=False,
        n_groups=args.n_groups,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        return_source_column=args.return_source_column,
        cost_bps=args.cost_bps,
        portfolio_modes=tuple(args.portfolio_modes),
        kernels=tuple(args.kernels),
        segment_columns=parse_columns(args.segment_columns),
        risk_column=args.risk_column or None,
        neutralization_method=args.neutralization_method,
        neutralization_group_columns=parse_columns(args.neutralization_group_columns),
        neutralization_exposure_columns=parse_columns(
            args.neutralization_exposure_columns
        ),
    )
    manifest_path = Path(manifest_path)
    for run in run_records:
        print(
            f"evaluated years={run['start_year']}-{run['end_year']} "
            f"labels={len(run['labels'])} features={len(run['features'])} "
            f"output={run['output_dir']}"
        )
    print(f"wrote {manifest_path.parent / 'input_quality' / 'label_availability.csv'}")
    print(f"wrote {manifest_path}")
    return 0


def _count_trade_date_rows(
    path: Path,
    *,
    start_date: str,
    end_date: str,
) -> int:
    """Count matching Parquet rows without materializing the full date column."""

    try:
        import pyarrow.dataset as ds

        trade_date = ds.field(DATE_COLUMN)
        date_filter = (
            (trade_date >= pd.Timestamp(start_date).to_pydatetime())
            & (trade_date <= pd.Timestamp(end_date).to_pydatetime())
        )
        return int(ds.dataset(path, format="parquet").count_rows(filter=date_filter))
    except ImportError:
        trade_dates = pd.to_datetime(
            pd.read_parquet(
                path,
                columns=[DATE_COLUMN],
                engine="pyarrow",
                filters=[
                    (DATE_COLUMN, ">=", pd.Timestamp(start_date).to_pydatetime()),
                    (DATE_COLUMN, "<=", pd.Timestamp(end_date).to_pydatetime()),
                ],
            )[DATE_COLUMN]
        )
        return int(trade_dates.between(start_date, end_date).sum())


if __name__ == "__main__":
    raise SystemExit(main())
