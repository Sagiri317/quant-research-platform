"""Compare private verification evidence runs using public-safe aggregates."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GOAL_ROOT = Path(".run/goal/qrp-architecture-closure-review-fix")
KEY_COLUMNS = ("market_id", "asset_id", "timestamp")
FLOAT_ATOL = 1e-10
FLOAT_RTOL = 1e-10
UINT64_MOD = 1 << 64

RETIRED_RESEARCH_DATA_ID = "a_share" + "_price_volume_forward_returns"
RETIRED_CONTEXT_DATA_ID = "daily" + "_context"
RETIRED_MARKET_DATA_ID = "daily" + "_market"
RETIRED_PANEL_DATA_ID = "daily" + "_panel"
RETIRED_PRIVATE_REPORT_ROOT = "private" + "_real"
RETIRED_RETURN_COLUMN = "return" + "_1d"
RETIRED_VALID_BAR_COLUMN = "valid" + "_daily" + "_bar"
RETIRED_NEXT_BAR_PREFIX = "next" + "_day" + "_*"

DATA_ID_RENAME_MAP = {
    "a_share_as_of": "as_of",
    "a_share_default": "default",
    "a_share_evaluation": "evaluation",
    "a_share_execution": "execution",
    RETIRED_RESEARCH_DATA_ID: "price_volume_forward_returns",
    RETIRED_CONTEXT_DATA_ID: "market_context_1d",
    RETIRED_MARKET_DATA_ID: "market_bars_1d",
    RETIRED_PANEL_DATA_ID: "market_panel_1d",
    RETIRED_PRIVATE_REPORT_ROOT: "private_research",
}


@dataclass
class SeriesStats:
    count: int = 0
    null_count: int = 0
    mean_value: float = 0.0
    m2_value: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, values: pd.Series) -> None:
        numeric = pd.to_numeric(values, errors="coerce")
        self.null_count += int(numeric.isna().sum())
        valid = numeric.dropna()
        if valid.empty:
            return
        batch_count = int(valid.shape[0])
        batch_mean = float(valid.mean())
        batch_m2 = float(((valid - batch_mean) * (valid - batch_mean)).sum())
        self._combine(batch_count=batch_count, batch_mean=batch_mean, batch_m2=batch_m2)
        current_min = float(valid.min())
        current_max = float(valid.max())
        self.min_value = (
            current_min if self.min_value is None else min(self.min_value, current_min)
        )
        self.max_value = (
            current_max if self.max_value is None else max(self.max_value, current_max)
        )

    def _combine(
        self,
        *,
        batch_count: int,
        batch_mean: float,
        batch_m2: float,
    ) -> None:
        if self.count == 0:
            self.count = batch_count
            self.mean_value = batch_mean
            self.m2_value = batch_m2
            return
        total_count = self.count + batch_count
        delta = batch_mean - self.mean_value
        self.m2_value = (
            self.m2_value
            + batch_m2
            + delta * delta * self.count * batch_count / total_count
        )
        self.mean_value = (
            self.mean_value * self.count + batch_mean * batch_count
        ) / total_count
        self.count = total_count

    def payload(self) -> dict[str, Any]:
        mean = self.mean_value if self.count else None
        std = None
        if self.count > 1:
            variance = self.m2_value / (self.count - 1)
            std = math.sqrt(max(variance, 0.0))
        return {
            "count": self.count,
            "null_count": self.null_count,
            "mean": mean,
            "std": std,
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class NumericDiff:
    mismatch_count: int = 0
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0

    def update(self, left: pd.Series, right: pd.Series) -> None:
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        left_na = left_num.isna()
        right_na = right_num.isna()
        both_na = left_na & right_na
        both_valid = ~left_na & ~right_na
        abs_diff = (left_num - right_num).abs()
        tolerance = FLOAT_ATOL + FLOAT_RTOL * right_num.abs()
        equal = both_na | (both_valid & abs_diff.le(tolerance))
        mismatches = ~equal
        self.mismatch_count += int(mismatches.sum())
        finite_abs = abs_diff[mismatches & np.isfinite(abs_diff)]
        if not finite_abs.empty:
            self.max_abs_diff = max(self.max_abs_diff, float(finite_abs.max()))
        denominator = right_num.abs().clip(lower=FLOAT_ATOL)
        rel_diff = abs_diff / denominator
        finite_rel = rel_diff[mismatches & np.isfinite(rel_diff)]
        if not finite_rel.empty:
            self.max_rel_diff = max(self.max_rel_diff, float(finite_rel.max()))

    def payload(self) -> dict[str, Any]:
        return {
            "mismatch_count": self.mismatch_count,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
        }


@dataclass
class TypedValueDiff:
    mismatch_count: int = 0
    left_hash_sum: int = 0
    left_hash_xor: int = 0
    right_hash_sum: int = 0
    right_hash_xor: int = 0

    def update(self, left: pd.Series, right: pd.Series) -> None:
        left_values = _normalized_value_series(left)
        right_values = _normalized_value_series(right)
        self.mismatch_count += int(left_values.ne(right_values).sum())
        left_hashes = pd.util.hash_pandas_object(
            left_values,
            index=False,
        ).to_numpy(dtype=np.uint64)
        right_hashes = pd.util.hash_pandas_object(
            right_values,
            index=False,
        ).to_numpy(dtype=np.uint64)
        self.left_hash_sum = (
            self.left_hash_sum + int(left_hashes.sum(dtype=np.uint64))
        ) % UINT64_MOD
        self.right_hash_sum = (
            self.right_hash_sum + int(right_hashes.sum(dtype=np.uint64))
        ) % UINT64_MOD
        if left_hashes.size:
            self.left_hash_xor ^= int(np.bitwise_xor.reduce(left_hashes))
        if right_hashes.size:
            self.right_hash_xor ^= int(np.bitwise_xor.reduce(right_hashes))

    def payload(self) -> dict[str, Any]:
        left_hash = _hash_payload(self.left_hash_sum, self.left_hash_xor)
        right_hash = _hash_payload(self.right_hash_sum, self.right_hash_xor)
        return {
            "mismatch_count": self.mismatch_count,
            "hash_match": left_hash == right_hash,
            "hash": {"baseline": left_hash, "final": right_hash},
        }


def main() -> None:
    args = _parse_args()
    goal_root = Path(args.goal_root)
    goal_root.mkdir(parents=True, exist_ok=True)
    try:
        runs = run_pairs_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _write_json(goal_root / "compare-rename-map.json", _rename_map_payload())
    comparisons = {}
    for label, paths in runs.items():
        comparisons[label] = compare_run_safely(
            label,
            paths["baseline"],
            paths["final"],
        )
    currentness = _currentness_payload(tuple(path["final"] for path in runs.values()))
    statuses = {label: item["status"] for label, item in comparisons.items()}
    overall_status = "PASS"
    if any(status in {"FAIL", "BLOCKED"} for status in statuses.values()):
        overall_status = "FAIL"
    elif any(status == "PARTIAL" for status in statuses.values()):
        overall_status = "PARTIAL"

    payload = {
        "schema_version": "qrp-baseline-final-compare/0.2",
        "baseline_commit": args.baseline_commit,
        "current_git": currentness,
        "overall_status": overall_status,
        "comparisons": comparisons,
        "rename_map_path": _safe_relative(goal_root / "compare-rename-map.json"),
        "privacy": "public_safe_aggregate_only_no_row_level_values",
    }
    _write_json(goal_root / "baseline-final-compare.json", payload)
    (goal_root / "baseline-final-compare.md").write_text(
        _markdown(payload),
        encoding="utf-8",
    )


def compare_run_safely(
    label: str,
    baseline_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    try:
        return compare_run(label, baseline_root, final_root)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "BLOCKED",
            "failures": ["compare_input_error"],
            "partials": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "baseline_run_label": baseline_root.name,
            "final_run_label": final_root.name,
            "baseline_root": _safe_relative(baseline_root),
            "final_root": _safe_relative(final_root),
        }


def compare_run(label: str, baseline_root: Path, final_root: Path) -> dict[str, Any]:
    baseline_summary = _read_json(baseline_root / "summary.json")
    final_summary = _read_json(final_root / "summary.json")
    baseline_config = _read_json(baseline_root / "config" / "run_config.json")
    final_config = _read_json(final_root / "config" / "run_config.json")
    research_compare = compare_research_dataset(baseline_root, final_root)
    prediction_summary = compare_prediction_outputs(baseline_root, final_root)
    evaluation_summary = compare_evaluation_outputs(baseline_root, final_root)
    portfolio_summary = compare_portfolio_outputs(baseline_root, final_root)
    validation_summary = compare_validation_summaries(baseline_summary, final_summary)
    runtime_summary = compare_runtime(baseline_summary, final_summary)

    failures: list[str] = []
    partials: list[str] = []
    for name, item in (
        ("research_dataset", research_compare),
        ("prediction_summary", prediction_summary),
        ("evaluation_summary", evaluation_summary),
        ("portfolio_summary", portfolio_summary),
        ("validation_summary", validation_summary),
    ):
        if item["status"] == "FAIL":
            failures.append(name)
        elif item["status"] == "PARTIAL":
            partials.append(name)
    if baseline_summary["status"] != "PASS":
        failures.append("baseline_status_not_pass")
    if final_summary["status"] != "PASS":
        failures.append("final_status_not_pass")
    if _window(baseline_config) != _window(final_config):
        failures.append("window_mismatch")

    status = "PASS"
    if failures:
        status = "FAIL"
    elif partials:
        status = "PARTIAL"
    return {
        "status": status,
        "failures": failures,
        "partials": partials,
        "baseline_run_label": baseline_root.name,
        "final_run_label": final_root.name,
        "window_compare": {
            "baseline": _window(baseline_config),
            "final": _window(final_config),
        },
        "validation_status": {
            "baseline": baseline_summary["validation_summary"]["status"],
            "final": final_summary["validation_summary"]["status"],
        },
        "research_dataset": research_compare,
        "prediction_summary": prediction_summary,
        "evaluation_summary": evaluation_summary,
        "portfolio_summary": portfolio_summary,
        "validation_summary": validation_summary,
        "runtime_summary": runtime_summary,
    }


def run_pairs_from_args(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    if args.pair and args.pairs_file:
        raise ValueError("pair_arguments_are_mutually_exclusive")
    if args.pairs_file:
        return run_pairs_from_file(Path(args.pairs_file))
    if not args.pair:
        raise ValueError("missing_pair: provide --pair or --pairs-file")
    runs: dict[str, dict[str, Path]] = {}
    for raw_pair in args.pair:
        if "=" not in raw_pair or "::" not in raw_pair:
            raise ValueError(
                "invalid_pair_format: expected label=baseline_root::final_root"
            )
        label, roots = raw_pair.split("=", 1)
        baseline_root, final_root = roots.split("::", 1)
        if not label:
            raise ValueError("invalid_pair_format: missing_label")
        if label in runs:
            raise ValueError(f"duplicate_pair_label: {label}")
        runs[label] = {
            "baseline": Path(baseline_root),
            "final": Path(final_root),
        }
    return runs


def run_pairs_from_file(path: Path) -> dict[str, dict[str, Path]]:
    payload = _read_json(path)
    raw_pairs = payload.get("pairs", payload)
    if not isinstance(raw_pairs, list):
        raise ValueError("invalid_pairs_file: expected list or {pairs: list}")
    runs: dict[str, dict[str, Path]] = {}
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("invalid_pairs_file: pair_must_be_object")
        try:
            label = str(raw_pair["label"])
            baseline_root = Path(str(raw_pair["baseline"]))
            final_root = Path(str(raw_pair["final"]))
        except KeyError as exc:
            raise ValueError("invalid_pairs_file: missing_required_key") from exc
        if not label:
            raise ValueError("invalid_pairs_file: missing_label")
        if label in runs:
            raise ValueError(f"duplicate_pair_label: {label}")
        runs[label] = {
            "baseline": baseline_root,
            "final": final_root,
        }
    if not runs:
        raise ValueError("invalid_pairs_file: empty_pairs")
    return runs


def compare_research_dataset(baseline_root: Path, final_root: Path) -> dict[str, Any]:
    baseline_dataset = find_research_dataset_root(baseline_root)
    final_dataset = find_research_dataset_root(final_root)
    baseline_files = partitioned_files_by_logical_partition(baseline_dataset)
    final_files = partitioned_files_by_logical_partition(final_dataset)
    baseline_rel = set(baseline_files)
    final_rel = set(final_files)
    missing = sorted(baseline_rel - final_rel)
    extra = sorted(final_rel - baseline_rel)
    common = sorted(baseline_rel & final_rel)

    left = empty_dataset_accumulator(baseline_dataset)
    right = empty_dataset_accumulator(final_dataset)
    numeric_diffs: dict[str, NumericDiff] = defaultdict(NumericDiff)
    typed_value_diffs: dict[str, TypedValueDiff] = defaultdict(TypedValueDiff)
    file_key_mismatches: list[str] = []
    for rel_path in common:
        left_frame = _read_partition_files(baseline_files[rel_path])
        right_frame = _read_partition_files(final_files[rel_path])
        update_dataset_accumulator(left, left_frame)
        update_dataset_accumulator(right, right_frame)
        left_columns = set(left_frame.columns)
        right_columns = set(right_frame.columns)
        common_columns = sorted(left_columns & right_columns)
        left_common = left_frame.loc[:, common_columns].reset_index(drop=True)
        right_common = right_frame.loc[:, common_columns].reset_index(drop=True)
        if _keys_equal(left_common, right_common):
            left_ordered = left_common
            right_ordered = right_common
        else:
            left_ordered = _sort_for_compare(left_common)
            right_ordered = _sort_for_compare(right_common)
        if not _keys_equal(left_ordered, right_ordered):
            file_key_mismatches.append(rel_path)
            continue
        numeric_columns = set(common_numeric_columns(left_ordered, right_ordered))
        for column in numeric_columns:
            if column in KEY_COLUMNS:
                continue
            numeric_diffs[column].update(left_ordered[column], right_ordered[column])
        for column in common_columns:
            if column in KEY_COLUMNS or column in numeric_columns:
                continue
            typed_value_diffs[column].update(
                left_ordered[column],
                right_ordered[column],
            )

    update_key_hash_status(left)
    update_key_hash_status(right)
    schema_match = sorted(left["columns"]) == sorted(right["columns"])
    nan_profile_match = dict(left["nan_profile"]) == dict(right["nan_profile"])
    key_hash_match = left["key_hash"] == right["key_hash"]
    failures = []
    if missing:
        failures.append("missing_final_partitions")
    if extra:
        failures.append("extra_final_partitions")
    if not schema_match:
        failures.append("schema_columns_after_rename_mismatch")
    if file_key_mismatches:
        failures.append("partition_key_order_or_membership_mismatch")
    if not key_hash_match:
        failures.append("key_bound_hash_mismatch")
    if not nan_profile_match:
        failures.append("nan_profile_mismatch")
    if any(diff.mismatch_count for diff in numeric_diffs.values()):
        failures.append("numeric_value_mismatch")
    if any(diff.mismatch_count for diff in typed_value_diffs.values()):
        failures.append("typed_value_mismatch")
    typed_value_payload = {
        column: diff.payload() for column, diff in sorted(typed_value_diffs.items())
    }

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "baseline_dataset": _safe_relative(baseline_dataset),
        "final_dataset": _safe_relative(final_dataset),
        "row_count": {"baseline": left["row_count"], "final": right["row_count"]},
        "row_count_match": left["row_count"] == right["row_count"],
        "asset_count": {
            "baseline": len(left["asset_ids"]),
            "final": len(right["asset_ids"]),
        },
        "asset_count_match": len(left["asset_ids"]) == len(right["asset_ids"]),
        "timestamp_count": {
            "baseline": len(left["timestamps"]),
            "final": len(right["timestamps"]),
        },
        "timestamp_count_match": len(left["timestamps"]) == len(right["timestamps"]),
        "schema_columns_after_rename_match": schema_match,
        "key_bound_hash_match": key_hash_match,
        "key_bound_hash": {"baseline": left["key_hash"], "final": right["key_hash"]},
        "nan_profile_match": nan_profile_match,
        "numeric_diffs": {
            column: diff.payload() for column, diff in sorted(numeric_diffs.items())
        },
        "typed_value_hash_match": all(
            item["hash_match"] for item in typed_value_payload.values()
        ),
        "typed_value_diffs": typed_value_payload,
        "max_numeric_abs_diff": max(
            (diff.max_abs_diff for diff in numeric_diffs.values()),
            default=0.0,
        ),
        "max_numeric_rel_diff": max(
            (diff.max_rel_diff for diff in numeric_diffs.values()),
            default=0.0,
        ),
        "partition_counts": {
            "baseline": len(baseline_files),
            "final": len(final_files),
            "compared": len(common),
        },
        "missing_final_partitions": missing,
        "extra_final_partitions": extra,
        "file_key_mismatches": file_key_mismatches,
    }


def compare_prediction_outputs(baseline_root: Path, final_root: Path) -> dict[str, Any]:
    baseline_files = find_report_files(
        baseline_root,
        "prediction",
        "prediction.parquet",
    )
    final_files = find_report_files(final_root, "prediction", "prediction.parquet")
    baseline_stats = aggregate_column_stats(baseline_files, ("prediction_score",))
    final_stats = aggregate_column_stats(final_files, ("prediction_score",))
    aggregate = compare_required_stats(
        baseline_stats,
        final_stats,
        required_columns=("prediction_score",),
        extra_payload={
            "file_count": {"baseline": len(baseline_files), "final": len(final_files)}
        },
    )
    baseline_by_id = report_files_by_artifact_id(
        baseline_root,
        "prediction",
        "prediction.parquet",
    )
    final_by_id = report_files_by_artifact_id(
        final_root,
        "prediction",
        "prediction.parquet",
    )
    missing_artifacts = sorted(set(baseline_by_id) - set(final_by_id))
    extra_artifacts = sorted(set(final_by_id) - set(baseline_by_id))
    content_compare = compare_report_artifact_content(
        baseline_by_id,
        final_by_id,
        columns=("timestamp", "asset_id", "prediction_score"),
    )
    artifact_diffs = {}
    artifact_failures: list[str] = []
    artifact_partials: list[str] = []
    for artifact_id in sorted(set(baseline_by_id) & set(final_by_id)):
        artifact_compare = compare_required_stats(
            aggregate_column_stats(
                [baseline_by_id[artifact_id]],
                ("prediction_score",),
            ),
            aggregate_column_stats(
                [final_by_id[artifact_id]],
                ("prediction_score",),
            ),
            required_columns=("prediction_score",),
        )
        if artifact_compare["status"] != "PASS":
            artifact_diffs[artifact_id] = artifact_compare
            if artifact_compare["status"] == "FAIL":
                artifact_failures.append(artifact_id)
            else:
                artifact_partials.append(artifact_id)
    failures = list(aggregate["failures"])
    if missing_artifacts:
        failures.append("missing_prediction_artifacts")
    if extra_artifacts:
        failures.append("extra_prediction_artifacts")
    if content_compare["status"] == "FAIL":
        failures.append("keyed_content_mismatch")
    failures.extend(sorted(artifact_failures))
    status = aggregate["status"]
    if failures:
        status = "FAIL"
    elif status != "PARTIAL" and (
        artifact_partials or content_compare["status"] == "PARTIAL"
    ):
        status = "PARTIAL"
    return {
        **aggregate,
        "status": status,
        "failures": failures,
        "partials": artifact_partials,
        "missing_prediction_artifacts": missing_artifacts,
        "extra_prediction_artifacts": extra_artifacts,
        "artifact_diffs": artifact_diffs,
        "artifact_content_diffs": content_compare["artifact_diffs"],
    }


def compare_evaluation_outputs(baseline_root: Path, final_root: Path) -> dict[str, Any]:
    sections = {
        "metrics": (
            "metrics.parquet",
            ("ic", "rankic", "n_obs"),
            ("timestamp", "signal", "label", "ic", "rankic", "n_obs"),
        ),
        "ic_summary": (
            "ic_summary.parquet",
            ("mean_ic", "mean_rankic", "valid_ic_count", "valid_rankic_count"),
            (
                "signal",
                "label",
                "mean_ic",
                "mean_rankic",
                "valid_ic_count",
                "valid_rankic_count",
            ),
        ),
    }
    return compare_report_sections(baseline_root, final_root, "evaluation", sections)


def compare_portfolio_outputs(baseline_root: Path, final_root: Path) -> dict[str, Any]:
    sections = {
        "performance_summary": (
            "performance_summary.parquet",
            (
                "final_cumulative_net_return",
                "max_drawdown",
                "mean_turnover",
                "total_cost",
            ),
            (
                "signal",
                "final_cumulative_net_return",
                "max_drawdown",
                "mean_turnover",
                "total_cost",
            ),
        ),
        "turnover": (
            "turnover.parquet",
            ("turnover",),
            ("timestamp", "signal", "turnover"),
        ),
        "cost_breakdown": (
            "cost_breakdown.parquet",
            ("total_cost",),
            ("timestamp", "signal", "total_cost"),
        ),
    }
    return compare_report_sections(baseline_root, final_root, "portfolio", sections)


def compare_report_sections(
    baseline_root: Path,
    final_root: Path,
    category: str,
    sections: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    failures: list[str] = []
    partials: list[str] = []
    for section, (filename, required_columns, content_columns) in sections.items():
        baseline_files = find_report_files(baseline_root, category, filename)
        final_files = find_report_files(final_root, category, filename)
        baseline_stats = aggregate_column_stats(baseline_files, required_columns)
        final_stats = aggregate_column_stats(final_files, required_columns)
        comparison = compare_required_stats(
            baseline_stats,
            final_stats,
            required_columns=required_columns,
            extra_payload={
                "file_count": {
                    "baseline": len(baseline_files),
                    "final": len(final_files),
                }
            },
        )
        content_compare = compare_report_artifact_content(
            report_files_by_artifact_id_from_paths(baseline_files),
            report_files_by_artifact_id_from_paths(final_files),
            columns=content_columns,
        )
        section_payload = {
            **comparison,
            "artifact_content_diffs": content_compare["artifact_diffs"],
        }
        if content_compare["status"] == "FAIL":
            section_payload["status"] = "FAIL"
            section_payload["failures"] = [
                *section_payload["failures"],
                "keyed_content_mismatch",
            ]
        elif (
            content_compare["status"] == "PARTIAL"
            and section_payload["status"] == "PASS"
        ):
            section_payload["status"] = "PARTIAL"
            section_payload["partials"] = [
                *section_payload["partials"],
                "keyed_content_partial",
            ]
        payload[section] = section_payload
        if section_payload["status"] == "FAIL":
            failures.append(section)
        elif section_payload["status"] == "PARTIAL":
            partials.append(section)
    status = "PASS"
    if failures:
        status = "FAIL"
    elif partials:
        status = "PARTIAL"
    return {"status": status, "failures": failures, "partials": partials, **payload}


def compare_required_stats(
    baseline_stats: dict[str, dict[str, Any]],
    final_stats: dict[str, dict[str, Any]],
    *,
    required_columns: tuple[str, ...],
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_required = [
        column
        for column in required_columns
        if column not in baseline_stats or column not in final_stats
    ]
    diffs = {
        column: stat_diff(baseline_stats[column], final_stats[column])
        for column in required_columns
        if column in baseline_stats and column in final_stats
    }
    failures = [
        column
        for column, diff in diffs.items()
        if any(
            abs(diff[field]) > FLOAT_ATOL
            for field in (
                "count_diff",
                "mean_diff",
                "std_diff",
                "min_diff",
                "max_diff",
            )
            if diff[field] is not None
        )
    ]
    status = "PASS"
    if failures:
        status = "FAIL"
    elif missing_required:
        status = "PARTIAL"
    return {
        "status": status,
        "failures": failures,
        "missing_required_fields": missing_required,
        "required_field_diffs": diffs,
        **(extra_payload or {}),
    }


def compare_validation_summaries(
    baseline_summary: dict[str, Any],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    date_range_mismatches = date_range_mismatches_after_rename(
        baseline_summary,
        final_summary,
    )
    factor_nan_profile_match = (
        baseline_summary["validation_summary"]["factor_nan_profile"]
        == final_summary["validation_summary"]["factor_nan_profile"]
    )
    label_nan_profile_match = (
        baseline_summary["validation_summary"]["label_nan_profile"]
        == final_summary["validation_summary"]["label_nan_profile"]
    )
    failures = []
    if date_range_mismatches:
        failures.append("date_range_mismatch_after_rename_map")
    if not factor_nan_profile_match:
        failures.append("factor_nan_profile_mismatch")
    if not label_nan_profile_match:
        failures.append("label_nan_profile_mismatch")
    if baseline_summary["manifest_status"] != final_summary["manifest_status"]:
        failures.append("manifest_status_mismatch")
    if (
        baseline_summary["validation_summary"]["private_safety_scan"]["status"]
        != "PASS"
    ):
        failures.append("baseline_private_safety_not_pass")
    if final_summary["validation_summary"]["private_safety_scan"]["status"] != "PASS":
        failures.append("final_private_safety_not_pass")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "date_range_mismatches_after_rename_map": date_range_mismatches,
        "factor_nan_profile_match": factor_nan_profile_match,
        "label_nan_profile_match": label_nan_profile_match,
        "manifest_status": {
            "baseline": baseline_summary["manifest_status"],
            "final": final_summary["manifest_status"],
        },
        "private_safety_status": {
            "baseline": baseline_summary["validation_summary"][
                "private_safety_scan"
            ]["status"],
            "final": final_summary["validation_summary"]["private_safety_scan"][
                "status"
            ],
        },
    }


def compare_runtime(
    baseline_summary: dict[str, Any],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_seconds = float(baseline_summary["timings"]["total_duration_seconds"])
    final_seconds = float(final_summary["timings"]["total_duration_seconds"])
    regression_pct = (
        ((final_seconds - baseline_seconds) / baseline_seconds) * 100.0
        if baseline_seconds
        else None
    )
    return {
        "baseline_seconds": baseline_seconds,
        "final_seconds": final_seconds,
        "regression_pct": regression_pct,
    }


def aggregate_column_stats(
    files: list[Path],
    columns: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    stats = {column: SeriesStats() for column in columns}
    for path in files:
        available_columns = set(pq.ParquetFile(path).schema_arrow.names)
        selected_columns = [
            column for column in columns if column in available_columns
        ]
        if not selected_columns:
            continue
        frame = pd.read_parquet(path, columns=selected_columns)
        for column in columns:
            if column in frame.columns:
                stats[column].update(frame[column])
    return {
        column: stat.payload()
        for column, stat in stats.items()
        if stat.count or stat.null_count
    }


def stat_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    def diff(field: str) -> float | None:
        left_value = left[field]
        right_value = right[field]
        if left_value is None or right_value is None:
            return None if left_value == right_value else float("nan")
        return float(right_value - left_value)

    return {
        "count": {"baseline": left["count"], "final": right["count"]},
        "count_diff": int(right["count"] - left["count"]),
        "null_count_diff": int(right["null_count"] - left["null_count"]),
        "mean_diff": diff("mean"),
        "std_diff": diff("std"),
        "min_diff": diff("min"),
        "max_diff": diff("max"),
    }


def find_research_dataset_root(run_root: Path) -> Path:
    roots = {
        path.parents[2]
        for path in (run_root / "data" / "research").rglob("*.parquet")
        if len(path.parents) >= 3
    }
    if len(roots) != 1:
        raise ValueError(f"unexpected_research_dataset_roots: {run_root}: {len(roots)}")
    return next(iter(roots))


def partitioned_files_by_logical_partition(dataset_root: Path) -> dict[str, list[Path]]:
    data_root = dataset_root / "data"
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(data_root.rglob("*.parquet")):
        relative = path.relative_to(data_root)
        logical_partition = relative.parent.as_posix()
        grouped[logical_partition].append(path)
    return dict(grouped)


def _read_partition_files(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def empty_dataset_accumulator(dataset_root: Path) -> dict[str, Any]:
    return {
        "dataset_root": dataset_root,
        "row_count": 0,
        "asset_ids": set(),
        "timestamps": set(),
        "columns": set(),
        "nan_profile": defaultdict(int),
        "key_hash_sum": 0,
        "key_hash_xor": 0,
        "key_hash": None,
    }


def update_dataset_accumulator(target: dict[str, Any], frame: pd.DataFrame) -> None:
    target["row_count"] += int(frame.shape[0])
    target["columns"].update(frame.columns)
    if "asset_id" in frame.columns:
        target["asset_ids"].update(frame["asset_id"].astype("string").dropna())
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        target["timestamps"].update(timestamps.dt.strftime("%Y-%m-%d").dropna())
    for column, value in frame.isna().sum().items():
        target["nan_profile"][column] += int(value)
    keys = _normalized_key_frame(frame)
    hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    target["key_hash_sum"] = (
        target["key_hash_sum"] + int(hashes.sum(dtype=np.uint64))
    ) % UINT64_MOD
    if hashes.size:
        target["key_hash_xor"] ^= int(np.bitwise_xor.reduce(hashes))


def update_key_hash_status(target: dict[str, Any]) -> None:
    target["key_hash"] = (
        f"count={target['row_count']};"
        f"sum64={target['key_hash_sum']:016x};"
        f"xor64={target['key_hash_xor']:016x}"
    )


def _hash_payload(hash_sum: int, hash_xor: int) -> str:
    return f"sum64={hash_sum:016x};xor64={hash_xor:016x}"


def common_numeric_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    columns = []
    for column in left.columns:
        if column not in right.columns:
            continue
        if pd.api.types.is_bool_dtype(left[column]) or pd.api.types.is_bool_dtype(
            right[column]
        ):
            continue
        left_numeric = pd.api.types.is_numeric_dtype(left[column])
        right_numeric = pd.api.types.is_numeric_dtype(right[column])
        if left_numeric and right_numeric:
            columns.append(column)
    return columns


def _sort_for_compare(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def _keys_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape[0] != right.shape[0]:
        return False
    return _normalized_key_frame(left).equals(_normalized_key_frame(right))


def _normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": frame["market_id"].astype("string").fillna("<NA>"),
            "asset_id": frame["asset_id"].astype("string").fillna("<NA>"),
            "timestamp": pd.to_datetime(frame["timestamp"], errors="coerce"),
        }
    )


def _normalized_value_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        nullable = series.astype("boolean")
        values = pd.Series(
            np.full(len(nullable), -1, dtype=np.int8),
            index=series.index,
        )
        values.loc[nullable.fillna(False).to_numpy(dtype=bool)] = 1
        values.loc[nullable.eq(False).fillna(False).to_numpy(dtype=bool)] = 0
        return values
    if pd.api.types.is_datetime64_any_dtype(series):
        values = pd.to_datetime(series, errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )
        return values.astype("string").fillna("<NA>")
    return series.astype("string").fillna("<NA>")


def find_report_files(run_root: Path, category: str, filename: str) -> list[Path]:
    reports_root = run_root / "reports"
    return sorted(
        path
        for path in reports_root.rglob(filename)
        if category in path.relative_to(reports_root).parts
    )


def report_files_by_artifact_id(
    run_root: Path,
    category: str,
    filename: str,
) -> dict[str, Path]:
    return report_files_by_artifact_id_from_paths(
        find_report_files(run_root, category, filename)
    )


def report_files_by_artifact_id_from_paths(paths: list[Path]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in paths:
        artifact_id = path.parent.name
        if artifact_id.startswith("a_share_"):
            artifact_id = artifact_id[len("a_share_") :]
        if artifact_id in files:
            raise ValueError(f"report_artifact_id_collision: {artifact_id}")
        files[artifact_id] = path
    return files


def compare_report_artifact_content(
    baseline_by_id: dict[str, Path],
    final_by_id: dict[str, Path],
    *,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    artifact_diffs: dict[str, Any] = {}
    partial_artifacts: list[str] = []
    for artifact_id in sorted(set(baseline_by_id) & set(final_by_id)):
        baseline_fingerprint = report_file_content_fingerprint(
            baseline_by_id[artifact_id],
            columns=columns,
        )
        final_fingerprint = report_file_content_fingerprint(
            final_by_id[artifact_id],
            columns=columns,
        )
        if (
            baseline_fingerprint["missing_columns"]
            or final_fingerprint["missing_columns"]
        ):
            partial_artifacts.append(artifact_id)
            continue
        match = baseline_fingerprint == final_fingerprint
        if not match:
            artifact_diffs[artifact_id] = {
                "match": False,
                "baseline": baseline_fingerprint,
                "final": final_fingerprint,
            }
    status = "PASS"
    if artifact_diffs:
        status = "FAIL"
    elif partial_artifacts:
        status = "PARTIAL"
    return {
        "status": status,
        "artifact_diffs": artifact_diffs,
        "partial_artifacts": partial_artifacts,
    }


def report_file_content_fingerprint(
    path: Path,
    *,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    available_columns = set(pq.ParquetFile(path).schema_arrow.names)
    missing_columns = [column for column in columns if column not in available_columns]
    selected_columns = [column for column in columns if column in available_columns]
    if not selected_columns:
        return {
            "row_count": 0,
            "missing_columns": missing_columns,
            "content_hash": _hash_payload(0, 0),
        }
    frame = pd.read_parquet(path, columns=selected_columns)
    hash_frame = normalized_report_hash_frame(frame, tuple(selected_columns))
    hashes = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(
        dtype=np.uint64
    )
    hash_sum = int(hashes.sum(dtype=np.uint64)) % UINT64_MOD
    hash_xor = int(np.bitwise_xor.reduce(hashes)) if hashes.size else 0
    return {
        "row_count": int(frame.shape[0]),
        "missing_columns": missing_columns,
        "content_hash": _hash_payload(hash_sum, hash_xor),
    }


def normalized_report_hash_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    normalized = frame.loc[:, list(columns)].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(normalized[column]):
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce",
            ).round(10)
    return normalized


def date_range_mismatches_after_rename(
    baseline_summary: dict[str, Any],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_ranges = {
        DATA_ID_RENAME_MAP.get(item["data_id"], item["data_id"]): {
            "date_min": item["date_min"],
            "date_max": item["date_max"],
            "partition_count": item["partition_count"],
        }
        for item in baseline_summary["validation_summary"]["date_ranges"]
    }
    final_ranges = {
        item["data_id"]: {
            "date_min": item["date_min"],
            "date_max": item["date_max"],
            "partition_count": item["partition_count"],
        }
        for item in final_summary["validation_summary"]["date_ranges"]
    }
    mismatches = {}
    for data_id, baseline_value in sorted(baseline_ranges.items()):
        final_value = final_ranges.get(data_id)
        if final_value != baseline_value:
            mismatches[data_id] = {
                "baseline": baseline_value,
                "final": final_value,
            }
    for data_id in sorted(set(final_ranges) - set(baseline_ranges)):
        mismatches[data_id] = {"baseline": None, "final": final_ranges[data_id]}
    return mismatches


def _window(config: dict[str, Any]) -> dict[str, Any]:
    requested = config["requested_windows"]
    return {
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "evaluation_start_date": requested.get("evaluation_start_date"),
        "evaluation_end_date": requested.get("evaluation_end_date"),
        "prediction_train_end": requested["prediction_train_end"],
        "prediction_test_start": requested["prediction_test_start"],
        "prediction_test_end": requested.get("prediction_test_end"),
    }


def _currentness_payload(final_roots: tuple[Path, ...]) -> dict[str, Any]:
    head = _git_output("git", "rev-parse", "HEAD")
    status = _git_output("git", "status", "--short")
    return {
        "head": head,
        "dirty": bool(status.strip()),
        "final_run_labels": [path.name for path in final_roots],
        "currentness_note": (
            "DIRTY_WORKTREE_REQUIRES_CURRENT_FINAL_SMOKE"
            if status.strip()
            else "clean_worktree"
        ),
    }


def _git_output(*command: str) -> str:
    try:
        return subprocess.check_output(command, text=True, encoding="utf-8").strip()
    except Exception:
        return "UNKNOWN"


def _rename_map_payload() -> dict[str, Any]:
    return {
        "schema_version": "qrp-compare-rename-map/0.2",
        "data_ids": DATA_ID_RENAME_MAP,
        "report_paths": {
            f"reports/{RETIRED_PRIVATE_REPORT_ROOT}": "reports/private_research",
            "a_share_<run_id>": "<run_id>",
        },
        "columns": {
            "frequency": "rebalance_frequency",
            RETIRED_RETURN_COLUMN: "period_return",
            RETIRED_VALID_BAR_COLUMN: "valid_bar",
            RETIRED_NEXT_BAR_PREFIX: "next_bar_*",
        },
        "privacy": "rename metadata only; no row-level data",
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# QRP baseline-final compare",
        "",
        f"- overall_status: {payload['overall_status']}",
        f"- current_head: {payload['current_git']['head']}",
        f"- dirty_worktree: {payload['current_git']['dirty']}",
        f"- currentness_note: {payload['current_git']['currentness_note']}",
        "- privacy: public-safe aggregate summary only; no row-level values emitted.",
        "",
    ]
    for label, comparison in payload["comparisons"].items():
        if comparison["status"] == "BLOCKED":
            lines.extend(
                [
                    f"## {label}",
                    "",
                    "- status: BLOCKED",
                    f"- baseline_run_label: {comparison['baseline_run_label']}",
                    f"- final_run_label: {comparison['final_run_label']}",
                    f"- failures: {comparison['failures']}",
                    f"- error_type: {comparison['error_type']}",
                    "",
                ]
            )
            continue
        research = comparison["research_dataset"]
        prediction = comparison["prediction_summary"]
        runtime = comparison["runtime_summary"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- status: {comparison['status']}",
                f"- baseline_run_label: {comparison['baseline_run_label']}",
                f"- final_run_label: {comparison['final_run_label']}",
                f"- row_count_match: {research['row_count_match']} "
                f"({research['row_count']})",
                f"- asset_count_match: {research['asset_count_match']} "
                f"({research['asset_count']})",
                f"- timestamp_count_match: {research['timestamp_count_match']} "
                f"({research['timestamp_count']})",
                f"- schema_columns_after_rename_match: "
                f"{research['schema_columns_after_rename_match']}",
                f"- key_bound_hash_match: {research['key_bound_hash_match']}",
                f"- nan_profile_match: {research['nan_profile_match']}",
                f"- max_numeric_abs_diff: {research['max_numeric_abs_diff']}",
                f"- max_numeric_rel_diff: {research['max_numeric_rel_diff']}",
                f"- prediction_status: {prediction['status']}",
                "- prediction_score_diffs: "
                f"{prediction['required_field_diffs'].get('prediction_score')}",
                "- prediction_artifact_failures: "
                f"{sorted(prediction.get('artifact_diffs', {}))}",
                f"- evaluation_status: {comparison['evaluation_summary']['status']}",
                f"- portfolio_status: {comparison['portfolio_summary']['status']}",
                "- runtime_seconds: "
                f"baseline={runtime['baseline_seconds']}, "
                f"final={runtime['final_seconds']}, "
                f"regression_pct={runtime['regression_pct']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_relative(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-root", default=str(GOAL_ROOT))
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument(
        "--pairs-file",
        help=(
            "JSON list or {pairs: list} of objects with label, baseline, and final "
            "run roots. Mutually exclusive with --pair."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        help=(
            "Repeatable compare pair in label=baseline_root::final_root form. "
            "Use :: so Windows drive letters are not ambiguous."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
