"""Compare private verification evidence runs using public-safe aggregates."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GOAL_ROOT = Path(".run/goal/verification-evidence-compare")
KEY_COLUMNS = ("market_id", "asset_id", "timestamp")
FLOAT_ATOL = 1e-10
FLOAT_RTOL = 1e-10
UINT64_MOD = 1 << 64
PARQUET_BATCH_SIZE = 65_536
MAX_IN_MEMORY_RESEARCH_PARTITION_ROWS = 1_500_000
EXPECTED_FAILURE_STATUS = "FAIL_EXPECTED"
EXPECTED_FAILURE_REASON = "stale_rank_ensemble_prediction_baseline"
RANK_ENSEMBLE_PREDICTION_ARTIFACTS = (
    "equal_weight_rank_ensemble",
    "ic_weighted_rank_ensemble",
)
OLD_BASELINE_GROUP = "old_baseline"
REFRESHED_PREDICTION_BASELINE_GROUP = "refreshed_prediction_baseline"
DIRECT_BASELINE_FINAL_GROUP = "baseline_final"
BASELINE_FINAL_COMPARE_GROUPS = (
    OLD_BASELINE_GROUP,
    REFRESHED_PREDICTION_BASELINE_GROUP,
    DIRECT_BASELINE_FINAL_GROUP,
)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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
        run_metadata = run_pair_metadata_from_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    comparisons: dict[str, dict[str, Any]] = {}
    for label, paths in runs.items():
        comparison = compare_run_safely(
            label,
            paths["baseline"],
            paths["final"],
        )
        comparison = add_public_summary_fields(comparison)
        comparisons[label] = apply_pair_expectation(comparison, paths)
    currentness = _currentness_payload(tuple(path["final"] for path in runs.values()))
    statuses = {label: item["status"] for label, item in comparisons.items()}
    actual_statuses = {
        label: item.get("actual_status", item["status"])
        for label, item in comparisons.items()
    }
    baseline_commit = str(args.baseline_commit)
    baseline_commit_full = _resolve_commit_full(baseline_commit)
    final_head = str(currentness["head"])
    try:
        smoke_evidence = final_smoke_evidence_from_metadata(run_metadata)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    contract_failures = compare_contract_failures(comparisons)
    contract_failures.extend(
        compare_metadata_contract_failures(
            baseline_commit_full=baseline_commit_full,
            final_head=final_head,
            smoke_evidence=smoke_evidence,
        )
    )
    raw_overall_status = _overall_status(actual_statuses.values())
    overall_status = _overall_status(statuses.values(), contract_failures)
    grouped = comparisons_by_group(comparisons)
    comparison_groups = {
        group: {"status": _group_status(statuses[label] for label in group_comparisons)}
        for group, group_comparisons in grouped.items()
        if group_comparisons
    }

    payload = {
        "schema_version": "qrp-baseline-final-compare-summary/0.1",
        "baseline_commit": baseline_commit[:8],
        "baseline_commit_full": baseline_commit_full,
        "final_head": final_head,
        "current_git": currentness,
        "status": {
            "raw": raw_overall_status,
            "policy_adjusted": overall_status,
            "validation_decision": (
                "PASS" if overall_status == "PASS" else overall_status
            ),
        },
        "contract_failures": contract_failures,
        "final_smoke_commit": smoke_evidence["final_smoke_commit"],
        "git_status_before_smoke": smoke_evidence["git_status_before_smoke"],
        "git_status_after_smoke": smoke_evidence["git_status_after_smoke"],
        "comparison_groups": comparison_groups,
        "status_counts": {
            status: sum(1 for item in statuses.values() if item == status)
            for status in sorted(set(statuses.values()))
        },
        "comparisons": comparisons,
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
            "label": label,
            "failures": ["compare_input_error"],
            "partials": [],
            "error_type": type(exc).__name__,
            "error": "compare_input_error",
            "baseline_run_label": baseline_root.name,
            "final_run_label": final_root.name,
            "baseline_root": _safe_relative(baseline_root),
            "final_root": _safe_relative(final_root),
        }


def apply_pair_expectation(
    comparison: dict[str, Any],
    pair: dict[str, Any],
) -> dict[str, Any]:
    expected_status = str(pair.get("expected_status") or "PASS").strip().upper()
    comparison_group = pair_comparison_group(pair)
    if comparison_group:
        comparison["comparison_group"] = comparison_group
    if expected_status == "PASS":
        comparison["expected_status"] = "PASS"
        return comparison
    if expected_status != EXPECTED_FAILURE_STATUS:
        comparison["expected_status"] = expected_status
        comparison["status"] = "FAIL"
        comparison.setdefault("failures", []).append("unsupported_expected_status")
        return comparison

    reason = str(pair.get("expected_failure_reason") or "").strip()
    expected_artifacts = sorted(
        str(item)
        for item in pair.get("expected_failed_artifacts", ())
        if str(item).strip()
    )
    comparison["expected_status"] = EXPECTED_FAILURE_STATUS
    comparison["expected_failure"] = True
    comparison["reason_code"] = reason
    comparison["expected_failed_artifacts"] = expected_artifacts
    failed_output_ids = prediction_failed_artifacts(comparison)
    failed_artifact_families = artifact_families(
        failed_output_ids,
        expected_artifacts,
    )
    comparison["failed_output_ids"] = failed_output_ids
    comparison["failed_artifacts"] = failed_artifact_families
    policy_failures = expected_failure_policy_failures(
        comparison,
        comparison_group=comparison_group,
        expected_artifacts=expected_artifacts,
        reason=reason,
    )
    if not policy_failures and expected_failure_matches(comparison, expected_artifacts):
        comparison["actual_status"] = "FAIL"
        comparison["status"] = EXPECTED_FAILURE_STATUS
        comparison["expected_failure_match"] = True
        return comparison
    comparison["expected_failure_match"] = False
    comparison["expected_failure_policy_failures"] = policy_failures
    comparison["status"] = "FAIL"
    comparison.setdefault("failures", []).append("expected_failure_policy_mismatch")
    return comparison


def expected_failure_policy_failures(
    comparison: dict[str, Any],
    *,
    comparison_group: str,
    expected_artifacts: list[str],
    reason: str,
) -> list[str]:
    failures: list[str] = []
    if comparison_group != OLD_BASELINE_GROUP:
        failures.append("expected_failure_group_not_old_baseline")
    if reason != EXPECTED_FAILURE_REASON:
        failures.append("expected_failure_reason_mismatch")
    if expected_artifacts != sorted(RANK_ENSEMBLE_PREDICTION_ARTIFACTS):
        failures.append("expected_failed_artifacts_mismatch")
    return failures


def expected_failure_matches(
    comparison: dict[str, Any],
    expected_artifacts: list[str],
) -> bool:
    if comparison.get("status") != "FAIL":
        return False
    if comparison.get("failures") != ["prediction_summary"]:
        return False
    for key in (
        "research_dataset",
        "evaluation_summary",
        "portfolio_summary",
        "validation_summary",
    ):
        if comparison.get(key, {}).get("status") != "PASS":
            return False
    prediction = comparison.get("prediction_summary", {})
    if prediction.get("missing_prediction_outputs"):
        return False
    if prediction.get("extra_prediction_outputs"):
        return False
    return sorted(comparison.get("failed_artifacts", ())) == expected_artifacts


def compare_contract_failures(comparisons: dict[str, dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    grouped = comparisons_by_group(comparisons)
    direct = grouped[DIRECT_BASELINE_FINAL_GROUP]
    if direct:
        for label, comparison in direct.items():
            if comparison.get("status") != "PASS":
                failures.append(f"direct_baseline_status_not_pass: {label}")
        return failures
    for group in BASELINE_FINAL_COMPARE_GROUPS:
        if group == DIRECT_BASELINE_FINAL_GROUP:
            continue
        if not grouped[group]:
            failures.append(f"missing_required_comparison_group: {group}")
    for label, comparison in grouped[OLD_BASELINE_GROUP].items():
        if comparison.get("status") != EXPECTED_FAILURE_STATUS:
            failures.append(f"old_baseline_status_not_fail_expected: {label}")
        if comparison.get("reason_code") != EXPECTED_FAILURE_REASON:
            failures.append(f"old_baseline_reason_mismatch: {label}")
        if comparison.get("failed_artifacts") != sorted(
            RANK_ENSEMBLE_PREDICTION_ARTIFACTS
        ):
            failures.append(f"old_baseline_failed_artifacts_mismatch: {label}")
    for label, comparison in grouped[REFRESHED_PREDICTION_BASELINE_GROUP].items():
        if comparison.get("status") != "PASS":
            failures.append(f"refreshed_baseline_status_not_pass: {label}")
            continue
        prediction = comparison.get("prediction_summary", {})
        missing_artifacts = sorted(
            artifact
            for artifact in RANK_ENSEMBLE_PREDICTION_ARTIFACTS
            if not artifact_family_present(prediction.get("output_ids", ()), artifact)
        )
        failures.extend(
            f"refreshed_baseline_missing_rank_ensemble_artifact: {label}:{artifact}"
            for artifact in missing_artifacts
        )
        if prediction.get("failed_artifacts"):
            failures.append(f"refreshed_baseline_failed_artifacts_present: {label}")
    return failures


def compare_metadata_contract_failures(
    *,
    baseline_commit_full: str,
    final_head: str,
    smoke_evidence: dict[str, str],
) -> list[str]:
    failures = []
    if not is_full_sha(baseline_commit_full):
        failures.append("baseline_commit_full_not_full_sha")
    if not is_full_sha(final_head):
        failures.append("final_head_not_full_sha")
    final_smoke_commit = smoke_evidence["final_smoke_commit"]
    if not is_full_sha(final_smoke_commit):
        failures.append("final_smoke_commit_not_full_sha")
    elif is_full_sha(final_head) and final_smoke_commit != final_head:
        failures.append("final_smoke_commit_not_current_head")
    if smoke_evidence["git_status_before_smoke"] != "clean":
        failures.append("git_status_before_smoke_not_clean")
    if smoke_evidence["git_status_after_smoke"] != "clean_or_only_ignored_run_outputs":
        failures.append("git_status_after_smoke_not_clean")
    return failures


def is_full_sha(value: object) -> bool:
    return isinstance(value, str) and bool(FULL_SHA_RE.fullmatch(value))


def pair_comparison_group(pair: dict[str, Any]) -> str:
    return str(pair.get("comparison_group") or "").strip()


def comparisons_by_group(
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {
        group: {} for group in BASELINE_FINAL_COMPARE_GROUPS
    }
    for label, comparison in comparisons.items():
        group = comparison.get("comparison_group")
        if group in grouped:
            grouped[str(group)][label] = comparison
    return grouped


def artifact_family_present(output_ids: Iterable[Any], expected_family: str) -> bool:
    return any(
        artifact_matches_family(str(output_id), expected_family)
        for output_id in output_ids
    )


def artifact_families(
    output_ids: Iterable[Any],
    expected_families: Iterable[str],
) -> list[str]:
    families: set[str] = set()
    expected = sorted(str(item) for item in expected_families)
    for output_id in output_ids:
        value = str(output_id)
        for family in expected:
            if artifact_matches_family(value, family):
                families.add(family)
    return sorted(families)


def artifact_matches_family(output_id: str, expected_family: str) -> bool:
    return bool(
        re.fullmatch(
            rf"{re.escape(expected_family)}_[A-Za-z0-9_]+_[0-9]+d_prediction_[0-9]{{4}}",
            output_id,
        )
    )


def prediction_failed_artifacts(comparison: dict[str, Any]) -> list[str]:
    prediction = comparison.get("prediction_summary", {})
    artifacts = set(prediction.get("artifact_diffs", {}))
    artifacts.update(prediction.get("artifact_content_diffs", {}))
    artifacts.update(prediction.get("missing_prediction_outputs", ()))
    artifacts.update(prediction.get("extra_prediction_outputs", ()))
    return sorted(str(artifact) for artifact in artifacts)


def add_public_summary_fields(comparison: dict[str, Any]) -> dict[str, Any]:
    if comparison.get("status") == "BLOCKED":
        return comparison
    comparison["research_dataset_summary"] = research_dataset_summary(
        comparison["research_dataset"]
    )
    return comparison


def research_dataset_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": payload["row_count"],
        "asset_count": payload["asset_count"],
        "timestamp_count": payload["timestamp_count"],
        "schema_columns_after_rename_match": payload[
            "schema_columns_after_rename_match"
        ],
        "key_bound_hash_match": payload["key_bound_hash_match"],
        "numeric_max_abs_diff": payload["max_numeric_abs_diff"],
        "numeric_max_rel_diff": payload["max_numeric_rel_diff"],
        "nan_profile_match": payload["nan_profile_match"],
    }


def _overall_status(
    statuses: Iterable[str],
    contract_failures: list[str] | None = None,
) -> str:
    if contract_failures:
        return "FAIL"
    status_set = set(statuses)
    if status_set & {"FAIL", "BLOCKED"}:
        return "FAIL"
    if "PARTIAL" in status_set:
        return "PARTIAL"
    return "PASS"


def _group_status(statuses: Iterable[str]) -> str:
    status_list = list(statuses)
    if not status_list:
        return "NOT_RUN"
    status_set = set(status_list)
    if status_set == {EXPECTED_FAILURE_STATUS}:
        return EXPECTED_FAILURE_STATUS
    if status_set <= {"PASS"}:
        return "PASS"
    if status_set <= {"PASS", EXPECTED_FAILURE_STATUS}:
        return "PASS"
    if "BLOCKED" in status_set:
        return "BLOCKED"
    if "FAIL" in status_set:
        return "FAIL"
    if "PARTIAL" in status_set:
        return "PARTIAL"
    return "FAIL"


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
        "label": label,
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


def run_pairs_from_args(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.pairs_file:
        raise ValueError("missing_pairs_file")
    return run_pairs_from_file(Path(args.pairs_file))


def run_pair_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pairs_file:
        raise ValueError("missing_pairs_file")
    return run_pair_metadata_from_file(Path(args.pairs_file))


def run_pair_metadata_from_file(path: Path) -> dict[str, Any]:
    payload = read_pairs_payload(path)
    if "metadata" not in payload:
        raise ValueError("invalid_pairs_file: metadata_required")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("invalid_pairs_file: metadata_must_be_object")
    return metadata


def final_smoke_evidence_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    raw = metadata.get("final_smoke")
    if not isinstance(raw, dict):
        raise ValueError("invalid_pairs_file: final_smoke_metadata_required")
    return {
        "final_smoke_commit": str(raw.get("final_smoke_commit") or "NOT_PROVIDED"),
        "git_status_before_smoke": str(
            raw.get("git_status_before_smoke") or "NOT_PROVIDED"
        ),
        "git_status_after_smoke": str(
            raw.get("git_status_after_smoke") or "NOT_PROVIDED"
        ),
    }


def run_pairs_from_file(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_pairs_payload(path)
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("invalid_pairs_file: pairs_must_be_list")
    runs: dict[str, dict[str, Any]] = {}
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError("invalid_pairs_file: pair_must_be_object")
        unexpected = sorted(set(raw_pair) - pair_file_pair_keys())
        if unexpected:
            raise ValueError(
                "invalid_pairs_file: pair_unapproved_key: " + ",".join(unexpected)
            )
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
        expected_failed_artifacts = raw_pair.get("expected_failed_artifacts", ())
        if isinstance(expected_failed_artifacts, str):
            expected_failed_artifacts = [expected_failed_artifacts]
        runs[label] = {
            "baseline": baseline_root,
            "final": final_root,
            "expected_status": raw_pair.get("expected_status", "PASS"),
            "expected_failure_reason": raw_pair.get("expected_failure_reason", ""),
            "expected_failed_artifacts": list(expected_failed_artifacts),
            "comparison_group": raw_pair.get("comparison_group", ""),
        }
    if not runs:
        raise ValueError("invalid_pairs_file: empty_pairs")
    return runs


def read_pairs_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("invalid_pairs_file: payload_must_be_object")
    unexpected = sorted(set(payload) - {"metadata", "pairs"})
    if unexpected:
        raise ValueError("invalid_pairs_file: unapproved_key: " + ",".join(unexpected))
    return payload


def pair_file_pair_keys() -> set[str]:
    return {
        "label",
        "baseline",
        "final",
        "expected_status",
        "expected_failure_reason",
        "expected_failed_artifacts",
        "comparison_group",
    }


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
    memory_guard_partitions: list[dict[str, Any]] = []
    for rel_path in common:
        left_rows = parquet_row_count(baseline_files[rel_path])
        right_rows = parquet_row_count(final_files[rel_path])
        if max(left_rows, right_rows) > MAX_IN_MEMORY_RESEARCH_PARTITION_ROWS:
            memory_guard_partitions.append(
                {
                    "partition": rel_path,
                    "baseline_rows": left_rows,
                    "final_rows": right_rows,
                    "max_in_memory_rows": MAX_IN_MEMORY_RESEARCH_PARTITION_ROWS,
                }
            )
            continue
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
    status = "PASS"
    if failures:
        status = "FAIL"
    elif memory_guard_partitions:
        status = "PARTIAL"

    return {
        "status": status,
        "failures": failures,
        "partial_partitions": memory_guard_partitions,
        "baseline_dataset": _safe_relative(baseline_dataset),
        "final_dataset": _safe_relative(final_dataset),
        "row_count": {
            "baseline": left["row_count"],
            "final": right["row_count"],
            "match": left["row_count"] == right["row_count"],
        },
        "asset_count": {
            "baseline": len(left["asset_ids"]),
            "final": len(right["asset_ids"]),
            "match": len(left["asset_ids"]) == len(right["asset_ids"]),
        },
        "timestamp_count": {
            "baseline": len(left["timestamps"]),
            "final": len(right["timestamps"]),
            "match": len(left["timestamps"]) == len(right["timestamps"]),
        },
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
    baseline_files = find_run_output_files(
        baseline_root,
        "prediction",
        "prediction.parquet",
    )
    final_files = find_run_output_files(final_root, "prediction", "prediction.parquet")
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
    baseline_by_id = run_output_files_by_output_id(
        baseline_root,
        "prediction",
        "prediction.parquet",
    )
    final_by_id = run_output_files_by_output_id(
        final_root,
        "prediction",
        "prediction.parquet",
    )
    missing_artifacts = sorted(set(baseline_by_id) - set(final_by_id))
    extra_artifacts = sorted(set(final_by_id) - set(baseline_by_id))
    content_compare = compare_run_output_content(
        baseline_by_id,
        final_by_id,
        columns=("timestamp", "asset_id", "prediction_score"),
    )
    artifact_diffs = {}
    artifact_failures: list[str] = []
    artifact_partials: list[str] = []
    for output_id in sorted(set(baseline_by_id) & set(final_by_id)):
        artifact_compare = compare_required_stats(
            aggregate_column_stats(
                [baseline_by_id[output_id]],
                ("prediction_score",),
            ),
            aggregate_column_stats(
                [final_by_id[output_id]],
                ("prediction_score",),
            ),
            required_columns=("prediction_score",),
        )
        if artifact_compare["status"] != "PASS":
            artifact_diffs[output_id] = artifact_compare
            if artifact_compare["status"] == "FAIL":
                artifact_failures.append(output_id)
            else:
                artifact_partials.append(output_id)
    failures = list(aggregate["failures"])
    if missing_artifacts:
        failures.append("missing_prediction_outputs")
    if extra_artifacts:
        failures.append("extra_prediction_outputs")
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
    prediction_score_diff = aggregate["required_field_diffs"].get(
        "prediction_score",
        {},
    )
    failed_artifacts = sorted(
        {
            *artifact_failures,
            *content_compare["artifact_diffs"],
            *missing_artifacts,
            *extra_artifacts,
        }
    )
    return {
        **aggregate,
        "status": status,
        "failures": failures,
        "partials": artifact_partials,
        "output_ids": sorted(set(baseline_by_id) | set(final_by_id)),
        "artifact_count": {
            "baseline": len(baseline_by_id),
            "final": len(final_by_id),
            "match": len(baseline_by_id) == len(final_by_id),
        },
        "row_count": _count_summary_from_stat_diff(prediction_score_diff),
        "prediction_mean_abs_diff_max": _abs_stat_diff(
            prediction_score_diff,
            "mean_diff",
        ),
        "prediction_std_abs_diff_max": _abs_stat_diff(
            prediction_score_diff,
            "std_diff",
        ),
        "prediction_min_abs_diff_max": _abs_stat_diff(
            prediction_score_diff,
            "min_diff",
        ),
        "prediction_max_abs_diff_max": _abs_stat_diff(
            prediction_score_diff,
            "max_diff",
        ),
        "failed_artifacts": failed_artifacts,
        "missing_prediction_outputs": missing_artifacts,
        "extra_prediction_outputs": extra_artifacts,
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
    payload = compare_run_output_sections(
        baseline_root,
        final_root,
        "evaluation",
        sections,
    )
    payload.update(
        {
            "artifact_count": _section_artifact_count(payload, "metrics"),
            "ic_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("metrics", "ic"),
                ("ic_summary", "mean_ic"),
            ),
            "rank_ic_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("metrics", "rankic"),
                ("ic_summary", "mean_rankic"),
            ),
            "count_diff_max": _max_required_field_abs_diff(
                payload,
                ("metrics", "n_obs"),
                ("ic_summary", "valid_ic_count"),
                ("ic_summary", "valid_rankic_count"),
            ),
            "coverage_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("metrics", "n_obs"),
                ("ic_summary", "valid_ic_count"),
                ("ic_summary", "valid_rankic_count"),
            ),
        }
    )
    return payload


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
    payload = compare_run_output_sections(
        baseline_root,
        final_root,
        "portfolio",
        sections,
    )
    payload.update(
        {
            "artifact_count": _section_artifact_count(
                payload,
                "performance_summary",
            ),
            "net_return_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("performance_summary", "final_cumulative_net_return"),
            ),
            "max_drawdown_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("performance_summary", "max_drawdown"),
            ),
            "turnover_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("performance_summary", "mean_turnover"),
                ("turnover", "turnover"),
            ),
            "cost_abs_diff_max": _max_required_field_abs_diff(
                payload,
                ("performance_summary", "total_cost"),
                ("cost_breakdown", "total_cost"),
            ),
        }
    )
    return payload


def compare_run_output_sections(
    baseline_root: Path,
    final_root: Path,
    category: str,
    sections: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    failures: list[str] = []
    partials: list[str] = []
    for section, (filename, required_columns, content_columns) in sections.items():
        baseline_files = find_run_output_files(baseline_root, category, filename)
        final_files = find_run_output_files(final_root, category, filename)
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
        content_compare = compare_run_output_content(
            run_output_files_by_output_id_from_paths(baseline_files),
            run_output_files_by_output_id_from_paths(final_files),
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


def _section_artifact_count(
    payload: dict[str, Any],
    section: str,
) -> dict[str, Any]:
    file_count = payload.get(section, {}).get("file_count", {})
    baseline = int(file_count.get("baseline", 0))
    final = int(file_count.get("final", 0))
    return {"baseline": baseline, "final": final, "match": baseline == final}


def _count_summary_from_stat_diff(diff: dict[str, Any]) -> dict[str, Any]:
    count = diff.get("count", {})
    baseline = int(count.get("baseline", 0) or 0)
    final = int(count.get("final", 0) or 0)
    return {"baseline": baseline, "final": final, "match": baseline == final}


def _max_required_field_abs_diff(
    payload: dict[str, Any],
    *fields: tuple[str, str],
) -> float:
    values: list[float] = []
    for section, field in fields:
        diff = payload.get(section, {}).get("required_field_diffs", {}).get(field, {})
        values.extend(
            _abs_stat_diff(diff, key)
            for key in ("count_diff", "mean_diff", "std_diff", "min_diff", "max_diff")
        )
    return max(values, default=0.0)


def _abs_stat_diff(diff: dict[str, Any], key: str) -> float:
    value = diff.get(key)
    if value is None:
        return 0.0
    try:
        result = abs(float(value))
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(result) else result


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
    range_mismatches = date_range_mismatches(
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
    if range_mismatches:
        failures.append("date_range_mismatch")
    if not factor_nan_profile_match:
        failures.append("factor_nan_profile_mismatch")
    if not label_nan_profile_match:
        failures.append("label_nan_profile_mismatch")
    if baseline_summary["manifest_status"] != final_summary["manifest_status"]:
        failures.append("manifest_status_mismatch")
    if baseline_summary["validation_summary"]["share_safety_scan"]["status"] != "PASS":
        failures.append("baseline_share_safety_not_pass")
    if final_summary["validation_summary"]["share_safety_scan"]["status"] != "PASS":
        failures.append("final_share_safety_not_pass")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "date_range_mismatches": range_mismatches,
        "factor_nan_profile_match": factor_nan_profile_match,
        "label_nan_profile_match": label_nan_profile_match,
        "manifest_status": {
            "baseline": baseline_summary["manifest_status"],
            "final": final_summary["manifest_status"],
        },
        "share_safety_status": {
            "baseline": baseline_summary["validation_summary"]["share_safety_scan"][
                "status"
            ],
            "final": final_summary["validation_summary"]["share_safety_scan"]["status"],
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
        selected_columns = [column for column in columns if column in available_columns]
        if not selected_columns:
            continue
        for frame in parquet_batches(path, columns=tuple(selected_columns)):
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


def parquet_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
) -> Iterable[pd.DataFrame]:
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=list(columns),
    ):
        yield batch.to_pandas()


def parquet_row_count(paths: list[Path]) -> int:
    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in paths)


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


def find_run_output_files(run_root: Path, category: str, filename: str) -> list[Path]:
    run_outputs_root = run_root / "run_outputs" / "local"
    return sorted(
        path
        for path in run_outputs_root.rglob(filename)
        if category in path.relative_to(run_outputs_root).parts
    )


def run_output_files_by_output_id(
    run_root: Path,
    category: str,
    filename: str,
) -> dict[str, Path]:
    return run_output_files_by_output_id_from_paths(
        find_run_output_files(run_root, category, filename)
    )


def run_output_files_by_output_id_from_paths(paths: list[Path]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in paths:
        output_id = path.parent.name
        if output_id.startswith("a_share_"):
            output_id = output_id[len("a_share_") :]
        if output_id in files:
            raise ValueError(f"run_output_id_collision: {output_id}")
        files[output_id] = path
    return files


def compare_run_output_content(
    baseline_by_id: dict[str, Path],
    final_by_id: dict[str, Path],
    *,
    columns: tuple[str, ...],
) -> dict[str, Any]:
    artifact_diffs: dict[str, Any] = {}
    partial_artifacts: list[str] = []
    for output_id in sorted(set(baseline_by_id) & set(final_by_id)):
        baseline_fingerprint = run_output_file_content_fingerprint(
            baseline_by_id[output_id],
            columns=columns,
        )
        final_fingerprint = run_output_file_content_fingerprint(
            final_by_id[output_id],
            columns=columns,
        )
        if (
            baseline_fingerprint["missing_columns"]
            or final_fingerprint["missing_columns"]
        ):
            partial_artifacts.append(output_id)
            continue
        match = baseline_fingerprint == final_fingerprint
        if not match:
            artifact_diffs[output_id] = {
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


def run_output_file_content_fingerprint(
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
    row_count = 0
    hash_sum = 0
    hash_xor = 0
    for frame in parquet_batches(path, columns=tuple(selected_columns)):
        row_count += int(frame.shape[0])
        hash_frame = normalized_run_output_hash_frame(frame, tuple(selected_columns))
        hashes = pd.util.hash_pandas_object(hash_frame, index=False).to_numpy(
            dtype=np.uint64
        )
        hash_sum = (hash_sum + int(hashes.sum(dtype=np.uint64))) % UINT64_MOD
        if hashes.size:
            hash_xor ^= int(np.bitwise_xor.reduce(hashes))
    return {
        "row_count": row_count,
        "missing_columns": missing_columns,
        "content_hash": _hash_payload(hash_sum, hash_xor),
    }


def normalized_run_output_hash_frame(
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


def date_range_mismatches(
    baseline_summary: dict[str, Any],
    final_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_ranges = {
        item["data_id"]: {
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
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
        ).strip()
    except Exception:
        return "UNKNOWN"


def _resolve_commit_full(commit: str) -> str:
    resolved = _git_output("git", "rev-parse", commit)
    if resolved != "UNKNOWN" and len(resolved) == 40:
        return resolved
    return "UNKNOWN"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# QRP baseline-final compare",
        "",
        f"- overall_status: {payload['status']['policy_adjusted']}",
        f"- current_head: {payload['current_git']['head']}",
        f"- dirty_worktree: {payload['current_git']['dirty']}",
        f"- currentness_note: {payload['current_git']['currentness_note']}",
        "- privacy: public-safe aggregate summary only; no row-level values emitted.",
        "",
    ]
    for label, comparison in payload["comparisons"].items():
        if comparison["status"] == "BLOCKED" or "research_dataset" not in comparison:
            lines.extend(
                [
                    f"## {label}",
                    "",
                    f"- status: {comparison['status']}",
                    f"- baseline_run_label: {comparison['baseline_run_label']}",
                    f"- final_run_label: {comparison['final_run_label']}",
                    f"- failures: {comparison['failures']}",
                    f"- error_type: {comparison.get('error_type', 'NOT_RECORDED')}",
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
                f"- row_count: {research['row_count']}",
                f"- asset_count: {research['asset_count']}",
                f"- timestamp_count: {research['timestamp_count']}",
                f"- schema_columns_after_rename_match: "
                f"{research['schema_columns_after_rename_match']}",
                f"- key_bound_hash_match: {research['key_bound_hash_match']}",
                f"- nan_profile_match: {research['nan_profile_match']}",
                f"- max_numeric_abs_diff: {research['max_numeric_abs_diff']}",
                f"- max_numeric_rel_diff: {research['max_numeric_rel_diff']}",
                f"- prediction_status: {prediction['status']}",
                "- prediction_score_diffs: "
                f"{prediction['required_field_diffs'].get('prediction_score')}",
                "- prediction_output_failures: "
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
        return f"<external-path>/{path.name}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal-root", default=str(GOAL_ROOT))
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument(
        "--pairs-file",
        help=(
            "JSON object with pairs list containing label, baseline, and final "
            "run roots."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
