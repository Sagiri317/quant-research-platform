"""Run core evaluation on private CSMAR processed slim factor Parquet files."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
CORE_SRC_ROOT = PROJECT_ROOT.parent / "alpha-research-core" / "src"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
# Prefer the sibling source checkout during local research so A-share can test
# core changes before alpha-research-core is packaged or installed.
if CORE_SRC_ROOT.exists() and str(CORE_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_SRC_ROOT))

from evaluation.core_bridge import (  # noqa: E402
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
        default="reports/private_real/factor_evaluation/full_2024_2025",
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
    parser.add_argument("--engine", default="dataframe")
    parser.add_argument("--segment-columns", default="")
    parser.add_argument("--risk-column", default="")
    parser.add_argument("--neutralization-method", default="none")
    parser.add_argument("--neutralization-group-columns", default="")
    parser.add_argument("--neutralization-exposure-columns", default="")
    args = parser.parse_args(argv)

    if args.factor_set != "price_volume":
        print(f"ERROR: unsupported factor set: {args.factor_set}")
        return 2
    _, manifest_path, core_runs = run_processed_factor_evaluation(
        processed_root=Path(args.processed_root),
        report_root=Path(args.report_root),
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
    report_root = Path(args.report_root)
    print(f"wrote {report_root / 'label_availability.csv'}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
