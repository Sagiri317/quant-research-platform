"""基于 factor evaluation runs 构造轻量 summaries。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_COLUMNS = (
    "year",
    "label",
    "feature",
    "kernel",
    "mean_ic",
    "icir",
    "positive_ic_rate",
    "mean_rankic",
    "rankicir",
    "positive_rankic_rate",
    "top_bottom_spread_mean",
    "top_bottom_spread_hit_rate",
    "group_monotonicity_score",
    "portfolio_mode",
    "mean_net_portfolio_return",
    "sharpe_like",
    "max_drawdown",
    "mean_turnover",
    "label_non_null_rate",
    "eligible_rows",
    "can_enter_rows",
    "can_exit_rows",
    "n_rows_evaluated",
    "run_id",
)

PERFORMANCE_COLUMNS = (
    "combination_id",
    "feature",
    "label",
    "kernel",
    "year",
    "portfolio_mode",
    "mean_net_portfolio_return",
    "sharpe_like",
    "max_drawdown",
    "mean_turnover",
)

MARKDOWN_COLUMNS = (
    "feature",
    "mean_ic",
    "mean_rankic",
    "positive_rankic_rate",
    "top_bottom_spread_mean",
    "portfolio_mode",
    "mean_net_portfolio_return",
    "sharpe_like",
)


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _to_markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + " | ".join(_format_markdown_value(row[column]) for column in headers)
            + " |"
        )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_schema_06_run_dirs(report_root: Path) -> list[Path]:
    if (report_root / "manifest.json").is_file():
        return [report_root]
    if not report_root.exists():
        raise FileNotFoundError(f"missing evaluation report root: {report_root}")
    return [
        child
        for child in sorted(report_root.iterdir())
        if child.is_dir() and (child / "manifest.json").is_file()
    ]


def _read_metric_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / "metrics" / filename
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation metric file: {path}")
    return pd.read_csv(path)


def _read_breakdown_metric_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / "breakdowns" / "year" / filename
    if not path.exists():
        raise FileNotFoundError(f"missing evaluation breakdown metric file: {path}")
    return pd.read_csv(path)


def _has_year_breakdown(run_dir: Path) -> bool:
    return (
        run_dir / "breakdowns" / "year" / "ic_summary.csv"
    ).exists()


def _load_label_availability(report_root: Path) -> pd.DataFrame:
    path = report_root / "input_quality" / "label_availability.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "year",
                "label",
                "label_non_null_rate",
                "eligible_rows",
                "can_enter_rows",
                "can_exit_rows",
            ]
        )
    frame = pd.read_csv(path)
    renamed = frame.rename(
        columns={
            "non_null_rate": "label_non_null_rate",
            "coverage_rate": "label_non_null_rate",
        }
    )
    for column in (
        "year",
        "label",
        "label_non_null_rate",
        "eligible_rows",
        "can_enter_rows",
        "can_exit_rows",
    ):
        if column not in renamed.columns:
            renamed[column] = pd.NA
    return renamed[
        [
            "year",
            "label",
            "label_non_null_rate",
            "eligible_rows",
            "can_enter_rows",
            "can_exit_rows",
        ]
    ]


def _load_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _normalize_performance_summary(
    frame: pd.DataFrame,
    *,
    portfolio_mode: str,
) -> pd.DataFrame:
    """规范化当前 performance summary columns。"""

    normalized = frame.copy()
    if "portfolio_mode" not in normalized.columns:
        normalized["portfolio_mode"] = portfolio_mode
    else:
        normalized["portfolio_mode"] = normalized["portfolio_mode"].fillna(
            portfolio_mode
        )
    return normalized


def _metric_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    candidate_keys = [
        "combination_id",
        "feature",
        "label",
        "kernel",
        "portfolio_mode",
        "year",
    ]
    keys = [
        key for key in candidate_keys if key in left.columns and key in right.columns
    ]
    if "feature" not in keys:
        keys.append("feature")
    return keys


def _merge_metric_frame(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = _metric_merge_keys(left, right)
    return left.merge(right, on=keys, how="left", validate="one_to_one")


def _build_one_run_summary(
    run_dir: Path,
) -> pd.DataFrame:
    run_summary = _load_run_summary(run_dir)
    config_path = run_dir / "config.json"
    config = _read_json(config_path) if config_path.exists() else {}
    combinations = pd.read_csv(run_dir / "dimensions" / "combinations.csv")
    label_availability = _load_label_availability(run_dir)
    portfolio_mode = _portfolio_mode(config, combinations)
    has_year_breakdown = _has_year_breakdown(run_dir)
    if has_year_breakdown:
        ic_summary = _read_breakdown_metric_csv(run_dir, "ic_summary.csv")
        spread_summary = _read_breakdown_metric_csv(
            run_dir,
            "group_spread_summary.csv",
        )
        performance_summary = _read_breakdown_metric_csv(
            run_dir,
            "performance_summary.csv",
        )
    else:
        ic_summary = _read_metric_csv(run_dir, "ic_summary.csv")
        spread_summary = _read_metric_csv(run_dir, "group_spread_summary.csv")
        performance_summary = _read_metric_csv(run_dir, "performance_summary.csv")
    performance_summary = _normalize_performance_summary(
        performance_summary,
        portfolio_mode=portfolio_mode,
    )
    performance_columns = [
        column
        for column in PERFORMANCE_COLUMNS
        if column in performance_summary.columns
    ]
    performance_summary = performance_summary[performance_columns]

    merged = _merge_metric_frame(ic_summary, spread_summary)
    merged = _merge_metric_frame(merged, performance_summary)
    merged["run_id"] = run_summary.get("evaluation_id", run_dir.name)
    merged["n_rows_evaluated"] = run_summary.get("n_rows_evaluated", pd.NA)
    return _merge_label_availability(merged, label_availability)


def _portfolio_mode(config: dict[str, Any], combinations: pd.DataFrame) -> str:
    if "portfolio_mode" in combinations.columns:
        modes = sorted(
            str(value)
            for value in combinations["portfolio_mode"].dropna().unique()
        )
        if len(modes) == 1:
            return modes[0]
        if len(modes) > 1:
            return "multiple"
    weight_configs = config.get("weight_configs", [])
    modes = sorted(
        str(item["portfolio_mode"])
        for item in weight_configs
        if isinstance(item, dict) and item.get("portfolio_mode")
    )
    if len(set(modes)) == 1:
        return modes[0]
    return "multiple" if modes else "long_short"


def _merge_label_availability(
    metrics: pd.DataFrame,
    label_availability: pd.DataFrame,
) -> pd.DataFrame:
    if label_availability.empty:
        return metrics
    if (
        "year" in metrics.columns
        and "year" in label_availability.columns
        and label_availability["year"].notna().any()
    ):
        return metrics.merge(
            label_availability,
            on=["year", "label"],
            how="left",
            validate="many_to_one",
        )
    return metrics.merge(
        label_availability.drop(columns=["year"], errors="ignore"),
        on=["label"],
        how="left",
        validate="many_to_one",
    )


def build_factor_summary_frame(report_root: Path) -> pd.DataFrame:
    """从 evaluation layer outputs 返回每个 year、label 和 feature 的一行汇总。"""

    frames = [
        _build_one_run_summary(run_dir)
        for run_dir in _discover_schema_06_run_dirs(report_root)
    ]
    if not frames:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = pd.concat(frames, ignore_index=True)
    for column in SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary.loc[:, SUMMARY_COLUMNS].sort_values(
        ["year", "label", "feature"],
        kind="stable",
    )


def write_factor_summary_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_factor_summary_markdown(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 因子评估汇总",
        "",
        f"- rows: {len(frame)}",
        "- years: "
        + ", ".join(str(year) for year in sorted(frame["year"].dropna().unique())),
        "- labels: "
        + ", ".join(str(label) for label in sorted(frame["label"].dropna().unique())),
        "",
    ]
    if frame.empty:
        lines.append("暂无可汇总的 evaluation run。")
    else:
        for (year, label), group in frame.groupby(["year", "label"], sort=True):
            ranked = group.sort_values(
                ["mean_rankic", "mean_ic"],
                ascending=False,
                na_position="last",
            )
            lines.extend(
                [
                    f"## {year} / {label}",
                    "",
                    _to_markdown_table(ranked.loc[:, MARKDOWN_COLUMNS]),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build factor_summary.csv and factor_summary.md from evaluation runs."
        )
    )
    parser.add_argument(
        "--report-root",
        default="reports/private_real/factor_evaluation/full_2024_2025",
        help="Schema 0.6 evaluation-set run directory or directory of run directories.",
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args(argv)

    report_root = Path(args.report_root)
    output_csv = (
        Path(args.output_csv) if args.output_csv else report_root / "factor_summary.csv"
    )
    output_md = (
        Path(args.output_md) if args.output_md else report_root / "factor_summary.md"
    )
    summary = build_factor_summary_frame(report_root)
    write_factor_summary_csv(output_csv, summary)
    write_factor_summary_markdown(output_md, summary)
    print(f"wrote {output_csv} rows={len(summary)}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
