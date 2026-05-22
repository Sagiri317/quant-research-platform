"""Run core evaluation on private CSMAR processed slim factor Parquet files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

from quant_research_platform.markets.a_share.evaluation.core_bridge import (  # noqa: E402
    parse_columns,
    run_processed_factor_evaluation,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate private CSMAR slim factor Parquet through core."
    )
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
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
    parser.add_argument("--labels", default="forward_return_20d")
    parser.add_argument("--factors", default="")
    parser.add_argument("--factor-set", default="price_volume")
    parser.add_argument("--all-factors", action="store_true")
    parser.add_argument("--n-groups", type=int, default=5)
    parser.add_argument("--pnl-return-column", default="forward_return_1d")
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--portfolio-mode",
        choices=("long_short", "long_only"),
        default="long_short",
    )
    parser.add_argument(
        "--engine",
        choices=("dataframe", "tensor"),
        default="dataframe",
    )
    parser.add_argument("--segment-columns", default="")
    parser.add_argument("--risk-column", default="")
    parser.add_argument("--neutralization-method", default="none")
    parser.add_argument("--neutralization-group-columns", default="")
    parser.add_argument("--neutralization-exposure-columns", default="")
    args = parser.parse_args(argv)

    if args.factor_set != "price_volume":
        print(f"ERROR: unsupported factor set: {args.factor_set}")
        return 2
    report_root = (
        Path(args.report_root)
        if args.report_root
        else Path("reports/private_real/factor_evaluation")
        / f"full_{args.start_year}_{args.end_year}"
    )
    _, manifest_path, core_runs = run_processed_factor_evaluation(
        processed_root=Path(args.processed_root),
        report_root=report_root,
        start_year=args.start_year,
        end_year=args.end_year,
        requested_labels=parse_columns(args.labels),
        requested_factors=parse_columns(args.factors),
        all_factors=args.all_factors,
        n_groups=args.n_groups,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        pnl_return_column=args.pnl_return_column,
        cost_bps=args.cost_bps,
        portfolio_mode=args.portfolio_mode,
        engine=args.engine,
        segment_columns=parse_columns(args.segment_columns),
        risk_column=args.risk_column or None,
        neutralization_method=args.neutralization_method,
        neutralization_group_columns=parse_columns(args.neutralization_group_columns),
        neutralization_exposure_columns=parse_columns(
            args.neutralization_exposure_columns
        ),
    )
    for run in core_runs:
        print(
            f"evaluated years={run['start_year']}-{run['end_year']} "
            f"label={run['label']} features={len(run['features'])} "
            f"output={run['output_dir']}"
        )
    print(f"wrote {report_root / 'label_availability.csv'}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
