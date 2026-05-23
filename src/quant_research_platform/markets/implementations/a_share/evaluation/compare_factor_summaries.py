"""对比 full-sample 与 diagnostic yearly factor summary outputs。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("year", "label", "feature")
EXCLUDED_COLUMNS = {*KEY_COLUMNS, "run_id"}
FOCUS_METRICS = (
    "mean_ic",
    "mean_rankic",
    "top_bottom_spread_mean",
    "mean_net_portfolio_return",
    "sharpe_like",
    "max_drawdown",
    "mean_turnover",
)


def _numeric_metric_columns(yearly: pd.DataFrame, full: pd.DataFrame) -> list[str]:
    shared = [column for column in yearly.columns if column in full.columns]
    metrics = []
    for column in shared:
        if column in EXCLUDED_COLUMNS:
            continue
        yearly_is_numeric = pd.api.types.is_numeric_dtype(yearly[column])
        full_is_numeric = pd.api.types.is_numeric_dtype(full[column])
        if yearly_is_numeric and full_is_numeric:
            metrics.append(column)
    return metrics


def build_comparison_frame(yearly: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """按 year、label、feature、metric 返回 long-form full-minus-yearly deltas。"""

    metrics = _numeric_metric_columns(yearly, full)
    merged = yearly.loc[:, [*KEY_COLUMNS, *metrics]].merge(
        full.loc[:, [*KEY_COLUMNS, *metrics]],
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_yearly", "_full"),
        indicator=True,
        validate="one_to_one",
    )
    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for metric in metrics:
            yearly_value = row.get(f"{metric}_yearly")
            full_value = row.get(f"{metric}_full")
            delta = full_value - yearly_value
            records.append(
                {
                    "year": row["year"],
                    "label": row["label"],
                    "feature": row["feature"],
                    "metric": metric,
                    "yearly_value": yearly_value,
                    "full_value": full_value,
                    "delta_full_minus_yearly": delta,
                    "abs_delta": abs(delta) if pd.notna(delta) else pd.NA,
                    "match_status": row["_merge"],
                }
            )
    return pd.DataFrame.from_records(records)


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [_format_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown_report(comparison: pd.DataFrame) -> str:
    """构造紧凑的 Markdown comparison summary。"""

    matched = comparison["match_status"].eq("both").all()
    lines = [
        "# factor summary 指标对比",
        "",
        f"- rows: {len(comparison)}",
        f"- matched keys only: {matched}",
        "- delta 定义：full-summary 指标 - yearly-summary 指标。",
        "",
    ]
    focus = comparison[comparison["metric"].isin(FOCUS_METRICS)].copy()
    for metric in FOCUS_METRICS:
        metric_frame = focus[focus["metric"].eq(metric)].copy()
        if metric_frame.empty:
            continue
        max_abs = metric_frame["abs_delta"].max(skipna=True)
        mean_abs = metric_frame["abs_delta"].mean(skipna=True)
        lines.extend(
            [
                f"## {metric}",
                "",
                f"- mean_abs_delta: {_format_value(mean_abs)}",
                f"- max_abs_delta: {_format_value(max_abs)}",
                "",
            ]
        )
        top = metric_frame.sort_values("abs_delta", ascending=False).head(5)
        lines.append(
            _markdown_table(
                top,
                (
                    "year",
                    "feature",
                    "yearly_value",
                    "full_value",
                    "delta_full_minus_yearly",
                ),
            )
        )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare full-sample and diagnostic yearly factor summaries."
    )
    parser.add_argument(
        "--yearly-summary",
        default=(
            "reports/private_real/factor_evaluation/comparisons/"
            "full_vs_yearly_current_2024_2025/yearly_current_factor_summary.csv"
        ),
    )
    parser.add_argument(
        "--full-summary",
        default="reports/private_real/factor_evaluation/full_2024_2025/factor_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "reports/private_real/factor_evaluation/comparisons/"
            "full_vs_yearly_current_2024_2025"
        ),
    )
    args = parser.parse_args(argv)

    yearly = pd.read_csv(args.yearly_summary)
    full = pd.read_csv(args.full_summary)
    comparison = build_comparison_frame(yearly, full)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metric_delta_long.csv"
    md_path = output_dir / "comparison_summary.md"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(build_markdown_report(comparison), encoding="utf-8-sig")
    print(f"wrote {csv_path} rows={len(comparison)}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
