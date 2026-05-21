"""Build lightweight summaries from core factor evaluation runs."""

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
    "weight_id",
    "cost_id",
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


def _resolve_run_dir(output_dir: str, report_root: Path) -> Path:
    core_runs_root = (report_root / "core_runs").resolve()
    path = Path(output_dir)
    candidates = [path] if path.is_absolute() else [
        report_root / path,
        core_runs_root / path.name,
    ]
    valid_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if core_runs_root not in resolved.parents:
            continue
        valid_candidates.append(resolved)
        if resolved.exists():
            return resolved
    if valid_candidates:
        return valid_candidates[0]
    raise ValueError(
        "core run output_dir must resolve under report_root/core_runs: "
        f"{output_dir}"
    )


def _read_metric_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / "metrics" / filename
    if not path.exists():
        raise FileNotFoundError(f"missing core metric file: {path}")
    return pd.read_csv(path)


def _read_breakdown_metric_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / "metrics_by_year" / filename
    if not path.exists():
        raise FileNotFoundError(f"missing core breakdown metric file: {path}")
    return pd.read_csv(path)


def _has_year_breakdown(run_dir: Path) -> bool:
    return (
        run_dir / "metrics_by_year" / "ic_summary_by_year.csv"
    ).exists()


def _load_label_availability(report_root: Path) -> pd.DataFrame:
    path = report_root / "label_availability.csv"
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
    return frame.rename(columns={"non_null_rate": "label_non_null_rate"})[
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
    run: dict[str, Any],
) -> pd.DataFrame:
    """Normalize current performance summary columns."""

    normalized = frame.copy()
    portfolio_mode = str(run.get("portfolio_mode") or "long_short")
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
        "weight_id",
        "cost_id",
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
    run: dict[str, Any],
    report_root: Path,
    label_availability: pd.DataFrame,
) -> pd.DataFrame:
    run_dir = _resolve_run_dir(str(run["output_dir"]), report_root)
    has_year_breakdown = _has_year_breakdown(run_dir)
    if has_year_breakdown:
        ic_summary = _read_breakdown_metric_csv(run_dir, "ic_summary_by_year.csv")
        spread_summary = _read_breakdown_metric_csv(
            run_dir,
            "group_spread_summary_by_year.csv",
        )
        performance_summary = _read_breakdown_metric_csv(
            run_dir,
            "performance_summary_by_year.csv",
        )
    else:
        ic_summary = _read_metric_csv(run_dir, "ic_summary.csv")
        spread_summary = _read_metric_csv(run_dir, "group_spread_summary.csv")
        performance_summary = _read_metric_csv(run_dir, "performance_summary.csv")
    performance_summary = _normalize_performance_summary(performance_summary, run)
    performance_columns = [
        column
        for column in PERFORMANCE_COLUMNS
        if column in performance_summary.columns
    ]
    performance_summary = performance_summary[performance_columns]

    merged = _merge_metric_frame(ic_summary, spread_summary)
    merged = _merge_metric_frame(merged, performance_summary)
    label = str(run["label"])
    run_summary = _load_run_summary(run_dir)
    merged["run_id"] = run_summary.get("evaluation_id", run_dir.name)
    merged["n_rows_evaluated"] = run_summary.get("n_rows_evaluated", pd.NA)
    if "label" not in merged.columns:
        merged["label"] = label
    if "year" in merged.columns:
        return merged.merge(
            label_availability,
            on=["year", "label"],
            how="left",
            validate="many_to_one",
        )
    if "year" in run:
        merged.insert(0, "year", int(run["year"]))
        return merged.merge(
            label_availability,
            on=["year", "label"],
            how="left",
            validate="many_to_one",
        )
    years = list(range(int(run["start_year"]), int(run["end_year"]) + 1))
    expanded = pd.concat(
        [merged.assign(year=year) for year in years],
        ignore_index=True,
    )
    return expanded.merge(
        label_availability,
        on=["year", "label"],
        how="left",
        validate="many_to_one",
    )


def build_factor_summary_frame(report_root: Path) -> pd.DataFrame:
    """Return one row per year, label, and feature from core evaluation outputs."""

    manifest_path = report_root / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    core_runs = manifest.get("core_runs", [])
    label_availability = _load_label_availability(report_root)
    frames = [
        _build_one_run_summary(run, report_root, label_availability)
        for run in core_runs
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
        lines.append("暂无可汇总的 core run。")
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
        description="Build factor_summary.csv and factor_summary.md from core runs."
    )
    parser.add_argument(
        "--report-root",
        default="reports/private_real/factor_evaluation/full_2024_2025",
        help="Directory containing evaluation_manifest.json and core_runs/.",
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
