"""Phase 1 本地验收运行器和低内存对比器。

用途：
- rename-baseline：把旧 acceptance run 目录重命名为 baseline。
- run：重新生成 2024 acceptance runs。
- compare：只比较 summary/metadata 等小文件，避免读取大组合持仓表导致内存压力。
- factory-smoke：跑一个小规模 Factor Factory smoke，验证 Phase 1 链路。

安全边界：脚本只重命名 baseline，不删除 run 目录；对比时不读取完整 portfolio 表。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FACTOR_EVALUATION_ROOT = Path("reports/private_real/factor_evaluation")
COMPARISON_ROOT = Path("reports/private_real/phase1_acceptance_comparisons")
FACTORY_SMOKE_ROOT = Path("reports/private_real/ff")
MODULE = (
    "quant_research_platform.markets.implementations.a_share.evaluation."
    "run_factor_evaluation"
)
RUN_ID = "csmar_2024_small"
FEATURES = ("mom_20d", "vol_20d", "avg_amount_20d")
LABELS = ("forward_return_5d", "forward_return_20d")
SUMMARY_FILES = (
    "metrics/ic_summary.csv",
    "metrics/group_spread_summary.csv",
    "metrics/performance_summary.csv",
    "metrics_by_year/ic_summary_by_year.csv",
    "metrics_by_year/group_spread_summary_by_year.csv",
    "metrics_by_year/performance_summary_by_year.csv",
)
HEAVY_FILES = (
    "metrics/ic.csv",
    "metrics/grouped_returns.csv",
    "portfolio/positions.csv",
    "portfolio/pnl.csv",
    "portfolio/returns.csv",
    "portfolio/turnover.csv",
    "diagnostics/weight_diagnostics.csv",
    "diagnostics/neutralization_diagnostics.csv",
)
KEY_COLUMNS = (
    "combination_id",
    "feature",
    "label",
    "weight_id",
    "cost_id",
    "year",
)
NUMERIC_TOLERANCE = 1e-9

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass(frozen=True)
class AcceptanceRun:
    name: str
    engine: str
    portfolio_mode: str

    @property
    def path(self) -> Path:
        return FACTOR_EVALUATION_ROOT / self.name


RUNS = (
    AcceptanceRun("acceptance_2024_dataframe_long_only", "dataframe", "long_only"),
    AcceptanceRun(
        "acceptance_2024_dataframe_long_short",
        "dataframe",
        "long_short",
    ),
    AcceptanceRun("acceptance_2024_tensor_long_only", "tensor", "long_only"),
    AcceptanceRun("acceptance_2024_tensor_long_short", "tensor", "long_short"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run and compare Phase 1 A-share acceptance artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan-baseline")
    plan_parser.add_argument("--stamp", default=_default_stamp())

    rename_parser = subparsers.add_parser("rename-baseline")
    rename_parser.add_argument("--stamp", default=_default_stamp())

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--only", choices=[run.name for run in RUNS], default="")

    smoke_parser = subparsers.add_parser("factory-smoke")
    smoke_parser.add_argument("--stamp", default=_default_stamp())

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--stamp", required=True)
    compare_parser.add_argument("--output-dir", default="")

    args = parser.parse_args(argv)
    if args.command == "plan-baseline":
        print(json.dumps(baseline_plan(args.stamp), indent=2, ensure_ascii=False))
        return 0
    if args.command == "rename-baseline":
        rename_baselines(args.stamp)
        return 0
    if args.command == "run":
        run_acceptance(args.only)
        return 0
    if args.command == "factory-smoke":
        run_factory_smoke(args.stamp)
        return 0
    if args.command == "compare":
        output_dir = Path(args.output_dir) if args.output_dir else None
        compare_acceptance(args.stamp, output_dir=output_dir)
        return 0
    raise AssertionError(args.command)


def baseline_plan(stamp: str) -> list[dict[str, str]]:
    return [
        {
            "source": str(run.path),
            "destination": str(_baseline_path(run, stamp)),
        }
        for run in RUNS
        if (PROJECT_ROOT / run.path).exists()
    ]


def rename_baselines(stamp: str) -> None:
    plan = baseline_plan(stamp)
    if len(plan) != len(RUNS):
        missing = [
            str(run.path)
            for run in RUNS
            if not (PROJECT_ROOT / run.path).exists()
        ]
        raise FileNotFoundError(f"missing baseline directories: {missing}")
    for item in plan:
        source = PROJECT_ROOT / item["source"]
        destination = PROJECT_ROOT / item["destination"]
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        print(f"renamed {source} -> {destination}")


def run_acceptance(only: str = "") -> None:
    selected = [run for run in RUNS if not only or run.name == only]
    for run in selected:
        command = _run_command(run)
        print("running " + " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=_child_env())


def compare_acceptance(
    stamp: str,
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output = output_dir or COMPARISON_ROOT / stamp
    output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for run in RUNS:
        baseline = PROJECT_ROOT / _baseline_path(run, stamp)
        current = PROJECT_ROOT / run.path
        comparison = compare_run_tree(baseline, current)
        comparison["run_set"] = run.name
        comparison["baseline"] = baseline.relative_to(PROJECT_ROOT).as_posix()
        comparison["current"] = current.relative_to(PROJECT_ROOT).as_posix()
        comparisons.append(comparison)
    report = {
        "stamp": stamp,
        "numeric_tolerance": NUMERIC_TOLERANCE,
        "comparisons": comparisons,
    }
    json_path = output / "phase1_acceptance_comparison.json"
    md_path = output / "phase1_acceptance_comparison.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"wrote {json_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote {md_path.relative_to(PROJECT_ROOT)}")
    return report


def run_factory_smoke(stamp: str) -> Path:
    """跑两个固定候选因子的轻量 smoke，验证 Factor Factory 到 evaluation 的主路径。"""
    from quant_research_platform.pipelines.a_share.config import PipelineConfig
    from quant_research_platform.research.factor_factory.batch_runner import (
        evaluate_existing_factor_candidates,
    )
    from quant_research_platform.research.factor_factory.factor_registry import (
        FactorTemplateRegistry,
    )

    output_dir = PROJECT_ROOT / FACTORY_SMOKE_ROOT / stamp
    if output_dir.exists():
        raise FileExistsError(output_dir)
    registry = FactorTemplateRegistry.from_directory(
        PROJECT_ROOT / "config/a_share/factor_templates"
    )
    selected_ids = {
        "momentum_60d__window_20@1",
        "reversal_20d__window_20@1",
    }
    candidates = tuple(
        candidate
        for candidate in registry.candidates(
            sample_start="2024-01-01",
            sample_end="2024-12-31",
            label_columns=("forward_return_20d",),
        )
        if candidate.candidate_id in selected_ids
    )
    if len(candidates) != len(selected_ids):
        found = sorted(candidate.candidate_id for candidate in candidates)
        raise RuntimeError(f"missing expected factor candidates, found={found}")
    base_config = PipelineConfig(
        market="a_share",
        run_id="f",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        portfolio_mode="long_only",
        cost_bps=0.0,
        pnl_return_column="forward_return_1d",
        label_columns=("forward_return_20d",),
        engine="dataframe",
        output_root=output_dir,
    )
    records = evaluate_existing_factor_candidates(
        candidates,
        base_config=base_config,
        output_dir=output_dir,
    )
    summary = {
        "stamp": stamp,
        "output_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "candidate_count": len(records),
        "success_count": sum(record.status == "success" for record in records),
        "failed_count": sum(record.status != "success" for record in records),
        "factory_run": (
            FACTORY_SMOKE_ROOT
            / stamp
            / "factor_factory"
            / "f_factory"
        ).as_posix(),
        "candidate_ids": [record.candidate_id for record in records],
        "run_refs": [str(record.to_row().get("run_ref", "")) for record in records],
    }
    summary_path = output_dir / "phase1_factory_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path.relative_to(PROJECT_ROOT)}")
    if summary["failed_count"]:
        raise RuntimeError(
            "Factor Factory smoke failed: "
            f"{summary['failed_count']} of {summary['candidate_count']} candidates "
            f"failed; see {summary_path.relative_to(PROJECT_ROOT)}"
        )
    return summary_path


def compare_run_tree(baseline: Path, current: Path) -> dict[str, Any]:
    """比较旧 baseline 与新 run 的小型汇总产物，保留大文件大小作为参考。"""
    old_runs = _run_dirs(baseline)
    new_runs = _run_dirs(current)
    summary = {
        "status": "pass",
        "old_run_count": len(old_runs),
        "new_run_count": len(new_runs),
        "old_summary_rows": {},
        "new_summary_rows": {},
        "differences": [],
        "heavy_file_sizes": {
            "baseline": _heavy_file_sizes(old_runs),
            "current": _heavy_file_sizes(new_runs),
        },
    }
    for relative_path in SUMMARY_FILES:
        old_rows = _read_summary_rows(old_runs, relative_path)
        new_rows = _read_summary_rows(new_runs, relative_path)
        summary["old_summary_rows"][relative_path] = len(old_rows)
        summary["new_summary_rows"][relative_path] = len(new_rows)
        differences = _compare_rows(old_rows, new_rows, relative_path)
        summary["differences"].extend(differences)
    _compare_metadata(old_runs, new_runs, summary["differences"])
    if summary["differences"]:
        summary["status"] = "review"
    return summary


def _run_command(run: AcceptanceRun) -> list[str]:
    return [
        sys.executable,
        "-m",
        MODULE,
        "--processed-root",
        "data/processed_private/csmar",
        "--report-root",
        run.path.as_posix(),
        "--start-year",
        "2024",
        "--end-year",
        "2024",
        "--start-date",
        "2024-01-01",
        "--end-date",
        "2024-12-31",
        "--labels",
        ",".join(LABELS),
        "--factors",
        ",".join(FEATURES),
        "--n-groups",
        "5",
        "--pnl-return-column",
        "forward_return_1d",
        "--cost-bps",
        "0",
        "--portfolio-mode",
        run.portfolio_mode,
        "--engine",
        run.engine,
    ]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    paths = [str(PROJECT_ROOT / "src")]
    if pythonpath:
        paths.append(pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTHONUTF8", "1")
    return env


def _baseline_path(run: AcceptanceRun, stamp: str) -> Path:
    return run.path.with_name(f"{run.path.name}_baseline_{stamp}")


def _run_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in ("evaluation_runs", "core_runs"):
        run_root = root / child
        if run_root.exists():
            candidates.extend(
                path
                for path in sorted(run_root.iterdir())
                if path.is_dir() and (path / "summary.json").exists()
            )
    return candidates


def _read_summary_rows(
    run_dirs: list[Path],
    relative_path: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run_dir in run_dirs:
        path = run_dir / relative_path
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_run_dir"] = run_dir.name
                rows.append(row)
    return rows


def _compare_rows(
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    relative_path: str,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    old_by_key = {_row_key(row): row for row in old_rows}
    new_by_key = {_row_key(row): row for row in new_rows}
    old_keys = set(old_by_key)
    new_keys = set(new_by_key)
    for key in sorted(old_keys - new_keys):
        differences.append({"file": relative_path, "key": key, "issue": "missing_new"})
    for key in sorted(new_keys - old_keys):
        differences.append({"file": relative_path, "key": key, "issue": "extra_new"})
    for key in sorted(old_keys & new_keys):
        old_row = old_by_key[key]
        new_row = new_by_key[key]
        for column, old_value in old_row.items():
            if column.startswith("_") or column not in new_row:
                continue
            diff = _cell_difference(old_value, new_row[column])
            if diff is not None:
                differences.append(
                    {
                        "file": relative_path,
                        "key": key,
                        "column": column,
                        "issue": diff,
                    }
                )
    return differences


def _compare_metadata(
    old_runs: list[Path],
    new_runs: list[Path],
    differences: list[dict[str, Any]],
) -> None:
    old_summary = _merged_summary(old_runs)
    new_summary = _merged_summary(new_runs)
    for field in (
        "artifact_schema_version",
        "artifact_type",
        "data_stage",
        "uses_external_data",
        "n_groups",
    ):
        if old_summary.get(field) != new_summary.get(field):
            differences.append(
                {
                    "file": "summary.json",
                    "field": field,
                    "issue": "metadata_changed",
                    "old": old_summary.get(field),
                    "new": new_summary.get(field),
                }
            )
    old_schema = old_summary.get("schema", {})
    new_schema = new_summary.get("schema", {})
    for field in ("feature_columns", "label_columns", "pnl_return_column"):
        if old_schema.get(field) != new_schema.get(field):
            differences.append(
                {
                    "file": "summary.json",
                    "field": f"schema.{field}",
                    "issue": "metadata_changed",
                    "old": old_schema.get(field),
                    "new": new_schema.get(field),
                }
            )


def _merged_summary(run_dirs: list[Path]) -> dict[str, Any]:
    summaries = [
        json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    if not summaries:
        return {}
    merged = dict(summaries[0])
    schemas = [summary.get("schema", {}) for summary in summaries]
    merged["schema"] = dict(schemas[0])
    labels = sorted(
        {
            label
            for schema in schemas
            for label in schema.get("label_columns", [])
        }
    )
    features = sorted(
        {
            feature
            for schema in schemas
            for feature in schema.get("feature_columns", [])
        }
    )
    merged["schema"]["label_columns"] = labels
    merged["schema"]["feature_columns"] = features
    return merged


def _row_key(row: dict[str, str]) -> str:
    parts = []
    for column in KEY_COLUMNS:
        value = row.get(column)
        if value not in (None, ""):
            parts.append(f"{column}={value}")
    if parts:
        return "|".join(parts)
    return row.get("combination_id") or row.get("_run_dir") or json.dumps(row)


def _cell_difference(old_value: str, new_value: str) -> str | None:
    old_number = _parse_float(old_value)
    new_number = _parse_float(new_value)
    if old_number is not None and new_number is not None:
        if math.isclose(
            old_number,
            new_number,
            rel_tol=NUMERIC_TOLERANCE,
            abs_tol=NUMERIC_TOLERANCE,
        ):
            return None
        return "numeric_changed"
    if old_value != new_value:
        return "value_changed"
    return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _heavy_file_sizes(run_dirs: list[Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for run_dir in run_dirs:
        for relative_path in HEAVY_FILES:
            path = run_dir / relative_path
            if path.exists():
                key = f"{run_dir.name}/{relative_path}"
                sizes[key] = path.stat().st_size
    return sizes


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 1 Acceptance Comparison",
        "",
        f"- baseline stamp: `{report['stamp']}`",
        f"- numeric tolerance: `{report['numeric_tolerance']}`",
        "",
        "| run set | status | old runs | new runs | differences |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for comparison in report["comparisons"]:
        row_template = (
            "| {run_set} | {status} | {old_run_count} | "
            "{new_run_count} | {diffs} |"
        )
        lines.append(
            row_template.format(
                run_set=comparison["run_set"],
                status=comparison["status"],
                old_run_count=comparison["old_run_count"],
                new_run_count=comparison["new_run_count"],
                diffs=len(comparison["differences"]),
            )
        )
    lines.append("")
    for comparison in report["comparisons"]:
        if not comparison["differences"]:
            continue
        lines.append(f"## {comparison['run_set']}")
        for difference in comparison["differences"][:50]:
            lines.append(f"- `{difference}`")
        if len(comparison["differences"]) > 50:
            lines.append(f"- ... {len(comparison['differences']) - 50} more")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _default_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
