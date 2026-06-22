"""Build a public-safe QRP handoff package from one task JSON."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))

from handoff_safety import scan_handoff_zip  # noqa: E402

PACKAGE_SCHEMA_VERSION = "handoff-sidecar/0.1"
VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
GENERATION_EVIDENCE = "handoff_evidence/generation.json"
SOURCE_TREE_SCAN_EVIDENCE = "handoff_evidence/source-tree-scan.json"
VERIFICATION_MATRIX_EVIDENCE = "handoff_evidence/verification-matrix.json"
BUILDER_OWNED_HANDOFF_EVIDENCE = {
    GENERATION_EVIDENCE,
    SOURCE_TREE_SCAN_EVIDENCE,
    VERIFICATION_MATRIX_EVIDENCE,
}
RESERVED_HANDOFF_EVIDENCE_STEMS = {
    "diff.patch",
    "external_evidence_json_path",
    "final_sha",
    "pr_body",
    "pr_body.md",
    "review_threads",
    "review_threads.json",
    "review_evidence",
    "review_bundle",
    "review_package",
    "review_sidecar",
    "review_zip_path",
    "validation_table",
}
TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS = RESERVED_HANDOFF_EVIDENCE_STEMS | {
    "generation",
    "source-tree-scan",
    "task_metadata",
    "validation_evidence",
    "verification-matrix",
}
FORBIDDEN_PACKAGE_ENTRY_PARTS = {
    "diff.patch",
    "pr_body.md",
    "review_threads.json",
}
HANDOFF_TOOL_SOURCE_PATHS = [
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_safety.py",
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py",
    ".agents/skills/quant-research-platform-maintenance/templates/handoff/task_metadata.template.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/task_metadata.schema.json",
    ".agents/skills/quant-research-platform-maintenance/schemas/handoff/sidecar.schema.json",
]
SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_safety.py"
    ): {
        "forbidden_diff_patch": {'"diff.patch",'},
        "forbidden_pr_body": {'"pr_body.md",'},
        "forbidden_review_threads": {'"review_threads",'},
        "forbidden_private_real": {'"reports/private_real",'},
        "forbidden_private_data": {'"reports/private_data",'},
        "forbidden_review_evidence": {'"review_evidence",'},
        "forbidden_review_bundle": {'"review_bundle",'},
        "forbidden_review_package": {'"review_package",'},
        "forbidden_review_zip_path": {'"review_zip_path",'},
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "verify_handoff_package.py"
    ): {
        "forbidden_final_head_full": {'"final_head_full",'},
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "package_safety_check.py"
    ): {
        "forbidden_diff_patch": {'"diff.patch",'},
        "forbidden_pr_body": {'"pr_body.md",'},
        "forbidden_review_threads": {'"review_threads",'},
        "forbidden_private_real": {
            'for path_name in ("private_research", "private_real", "private_data")',
            '"reports/private_real",',
        },
        "forbidden_private_data": {
            'for path_name in ("private_research", "private_real", "private_data")',
            '"reports/private_data",',
        },
        "forbidden_review_evidence": {'"review_evidence",'},
        "forbidden_review_bundle": {'"review_bundle",'},
        "forbidden_review_package": {'"review_package",'},
        "forbidden_review_zip_path": {'"review_zip_path",'},
    },
    Path("src/quant_research_platform/runtime/privacy.py"): {
        "forbidden_private_real": {'"private_real",'},
        "forbidden_private_data": {'"private_data",'},
    },
    Path("tests/architecture/test_retired_surface_absent.py"): {
        "retired_a_share_market_definition": {
            '"a_share_market_definition",',
        },
        "retired_a_share_field_catalog": {
            '"a_share_field_catalog",',
        },
        "retired_a_share_processed_catalog": {
            '"a_share_processed_catalog",',
        },
        "retired_a_share_factor_catalog": {
            '"a_share_factor_catalog",',
        },
        "retired_a_share_label_catalog": {
            '"a_share_label_catalog",',
        },
        "retired_build_a_share": {
            '"build_a_share_research_dataset_catalog",',
            '"build_a_share_research_dataset_specs",',
            '"build_a_share_model_catalog",',
            '"build_a_share_model_specs",',
            '"build_a_share_prediction_catalog",',
            '"build_a_share_prediction_specs",',
            '"build_a_share_portfolio_catalog",',
            '"build_a_share_portfolio_specs",',
        },
        "retired_default_a_share": {
            '"default_a_share_prediction_smoke_id",',
            '"default_a_share_prediction_oos_period",',
        },
        "retired_workflow_module": {
            'importlib.util.find_spec("quant_research_platform.contracts.workflow_run")',
        },
        "retired_model_algorithm_collection": {
            'assert not hasattr(contracts_models, "MODEL_ALGORITHMS")',
        },
        "retired_csmar_source_alias": {
            'assert not hasattr(source_ids, "CSMAR_SOURCE_ID")',
        },
        "retired_a_share_market_alias": {
            'assert not hasattr(market_ids, "A_SHARE_MARKET_ID")',
        },
        "forbidden_render_pr_body": {
            '".agents/skills/quant-research-platform-maintenance/scripts/render_pr_body_from_validation_evidence.py"',
            '"render_pr_body_from_validation_evidence.py"',
        },
    },
    Path("tests/agents/handoff/test_handoff_tools.py"): {
        "forbidden_private_data": {
            '"reports/private_data/a_share/run/summary.json",',
        },
        "forbidden_private_real": {
            '"reports/private_real/a_share/run/summary.json",',
            '"retired path reports/private_real/a_share/run\\n",',
        },
    },
    Path("tests/agents/handoff/test_package_safety_check.py"): {
        "forbidden_private_data": {
            '"reports/private_data/a_share/run/summary.json",',
        },
        "forbidden_private_real": {
            '"reports/private_real/a_share/run/summary.json",',
        },
    },
}
SOURCE_TREE_ACTIVE_TEST_LITERAL_PATHS = {
    Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"),
}
SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    ): {
        "active_market_bars_data_id",
        "active_market_panel_data_id",
        "active_market_context_data_id",
        "active_frequency_slug",
    },
    Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"): {
        "active_market_bars_data_id",
        "active_market_panel_data_id",
        "active_market_context_data_id",
        "active_frequency_slug",
    },
}
SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    ): {
        "retired_a_share_market_definition",
        "retired_a_share_field_catalog",
        "retired_a_share_processed_catalog",
        "retired_a_share_factor_catalog",
        "retired_a_share_label_catalog",
        "retired_build_a_share",
        "retired_default_a_share",
        "retired_csmar_source_alias",
        "retired_a_share_market_alias",
        "retired_workflow_module",
        "retired_model_algorithm_collection",
        "retired_execution_mode",
        "retired_daily_panel",
        "retired_daily_market",
        "retired_daily_context",
        "retired_a_share_daily_bar",
        "retired_a_share_daily_bar_interim",
        "retired_daily_backtest",
        "retired_daily_rebalance",
        "retired_daily_open",
        "retired_backtest_spec",
        "retired_dry_run_flag",
        "retired_model_family",
        "retired_prediction_bundle",
        "forbidden_private_real",
        "forbidden_private_data",
        "forbidden_review_evidence",
        "forbidden_review_bundle",
        "forbidden_review_package",
        "forbidden_review_zip",
        "forbidden_review_sidecar",
        "forbidden_review_zip_path",
        "forbidden_external_evidence_json_path",
        "forbidden_final_sha",
        "forbidden_final_head_full",
        "forbidden_validation_table",
        "forbidden_review_threads",
        "forbidden_review_threads_json",
        "forbidden_task_review_threads",
        "forbidden_verify_review_thread_closure",
        "forbidden_diff_patch",
        "forbidden_pr_body",
        "forbidden_goal_status",
        "forbidden_render_pr_body",
    },
    Path("docs/architecture/open-issues-register.md"): {
        "retired_backtest_spec",
    },
    Path("tests/architecture/test_retired_surface_absent.py"): {
        "retired_csmar_source_alias",
        "retired_a_share_market_alias",
        "retired_model_algorithm_collection",
        "forbidden_render_pr_body",
    },
}
SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    ): {
        "RESERVED_HANDOFF_EVIDENCE_STEMS",
        "FORBIDDEN_PACKAGE_ENTRY_PARTS",
        "SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST",
        "SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST",
        "SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST",
        "SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST",
        "TASK_CONTROL_KEYS",
        "TASK_METADATA_KEYS",
        "TASK_INPUT_KEYS",
        "FORBIDDEN_TASK_KEYS",
        "FORBIDDEN_VALIDATION_KEYS",
        "FORBIDDEN_HANDOFF_EVIDENCE_KEYS",
        "source_tree_forbidden_patterns",
        "source_tree_active_test_literal_patterns",
        "source_tree_sensitive_negative_patterns",
        "source_tree_public_text_unsafe_patterns",
        "sanitize_handoff_text",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_safety.py"
    ): {
        "FORBIDDEN_PATH_SEGMENTS",
        "PRIVATE_RUN_PATH_RE",
        "unsafe_text_reason",
        "is_scanner_definition_line",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "self_test.py"
    ): {"main", "_zip_and_sidecar_with_leak"},
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "verify_handoff_package.py"
    ): {
        "contains_local_absolute_path_text",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "package_safety_check.py"
    ): {
        "BLOCKED_PATH_PARTS",
        "_skip_pattern_line",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "verify_governance.py"
    ): {
        "LOCAL_PATH_PATTERN",
        "SECRET_PATTERN",
        "_skip_regex_definition_line",
        "_is_allowed_local_path_fixture",
        "_is_allowed_secret_fixture",
        "check_text_safety",
        "check_markdown_paths",
    },
}
SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST = {
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "build_handoff_package.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "verify_governance.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/"
        "package_safety_check.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_safety.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "verify_handoff_package.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "self_test.py"
    ): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/agents/handoff/test_handoff_tools.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/agents/handoff/test_package_safety_check.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
    },
    Path("tests/runtime/test_privacy_01.py"): {
        "sensitive_local_private_path",
        "sensitive_secret",
        "sensitive_provider_marker",
        "sensitive_stock_code_sample",
    },
}
TASK_CONTROL_KEYS = {
    "validation_evidence",
    "handoff_evidence",
}
TASK_METADATA_KEYS = {
    "schema_version",
    "mode",
    "slug",
    "branch",
    "base",
    "final_head",
    "pr_url",
    "task_title",
    "task_id",
    "generated_at_utc",
    "handoff_package_path",
    "handoff_sidecar_path",
    "scope_summary",
    "fixed_blockers",
    "open_issues",
    "not_run",
    "blocked",
    "commit_list",
    "source_tree_scan",
}
TASK_INPUT_KEYS = TASK_METADATA_KEYS | TASK_CONTROL_KEYS
FORBIDDEN_TASK_KEYS = {
    "diff.patch",
    "external_evidence_json_path",
    "final_head_full",
    "final_sha",
    "goal_status",
    "pr_body",
    "review_bundle",
    "review_evidence",
    "review_package",
    "review_sidecar",
    "review_threads",
    "review_threads_json",
    "review_threads_path",
    "review_zip",
    "review_zip_path",
    "sidecar_logical_path",
    "validation_table",
    "zip_logical_path",
}
FORBIDDEN_VALIDATION_KEYS = {
    "external_evidence_json_path",
    "final_head_full",
    "final_sha",
    "goal_status",
    "review_threads",
    "review_zip_path",
    "validation_table",
}
FORBIDDEN_HANDOFF_EVIDENCE_KEYS = FORBIDDEN_TASK_KEYS | {"pr_url"}
VALIDATION_ROW_KEYS = {"name", "status", "evidence"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a public-safe QRP handoff package from one task JSON."
    )
    parser.add_argument("--task", required=True, help="Public-safe task input JSON.")
    parser.add_argument("--out-dir", default=".run/handoffs")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--mode", choices=("evidence", "pr_closure"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = load_json_object(Path(args.task), "task")
    result = build_handoff_package(
        repo_root=repo_root,
        out_dir=out_dir,
        task=task,
        mode=args.mode,
        verify=args.verify,
    )
    print(f"zip={result['zip_path']}")
    print(f"json={result['json_path']}")
    print(f"zip_sha256={result['zip_sha256']}")
    print(f"handoff_package_verify_status={result['handoff_verify_status']}")
    print(f"handoff_safety_scan_status={result['handoff_safety_status']}")
    return 0 if result["ok"] else 1


def build_handoff_package(
    *,
    repo_root: Path,
    out_dir: Path,
    task: dict[str, object],
    mode: str | None = None,
    verify: bool,
) -> dict[str, object]:
    slug = required_string(task, "slug")
    branch = required_string(task, "branch")
    base = required_string(task, "base")
    package_mode = handoff_mode(task, mode)
    validate_task_contract(task, package_mode)
    if package_mode == "pr_closure" and not str(task.get("pr_url") or "").strip():
        raise SystemExit("task_pr_url_required_for_pr_closure")
    final_head = required_string(task, "final_head")
    zip_path = out_dir / f"{slug}.zip"
    json_path = out_dir / f"{slug}.json"
    handoff_package_path = handoff_package_path_value(task, zip_path)
    handoff_sidecar_path = handoff_sidecar_path_value(task, json_path)
    status_overrides: dict[str, str] = {}
    last_failures: list[str] = []
    max_attempts = 3 if verify else 1
    for _attempt in range(max_attempts):
        validation_evidence = enrich_validation_evidence(
            validation_with_status_overrides(
                task_validation_evidence(task),
                status_overrides,
            ),
            handoff_package_path=handoff_package_path,
            handoff_sidecar_path=handoff_sidecar_path,
        )
        build_once(
            repo_root=repo_root,
            zip_path=zip_path,
            json_path=json_path,
            task=task,
            validation_evidence=validation_evidence,
            branch=branch,
            base=base,
            final_head=final_head,
            mode=package_mode,
            handoff_package_path=handoff_package_path,
            handoff_sidecar_path=handoff_sidecar_path,
        )
        safety_status = (
            "PASS" if scan_handoff_zip(zip_path, json_path=json_path).ok else "FAIL"
        )
        if not verify:
            return {
                "ok": safety_status == "PASS",
                "zip_path": zip_path.as_posix(),
                "json_path": json_path.as_posix(),
                "zip_sha256": sha256_file(zip_path),
                "handoff_verify_status": "NOT_RUN",
                "handoff_safety_status": safety_status,
            }
        from verify_handoff_package import verify_handoff_package

        last_failures = verify_handoff_package(zip_path, json_path)
        verify_status = "PASS" if not last_failures else "FAIL"
        desired = {
            "handoff_verify": verify_status,
            "handoff_safety_scan": safety_status,
            "package_safety_check_zip": package_safety_status(zip_path),
            "package_safety_check_sidecar": package_safety_status(json_path),
        }
        if status_overrides == desired:
            return {
                "ok": verify_status == "PASS" and safety_status == "PASS",
                "zip_path": zip_path.as_posix(),
                "json_path": json_path.as_posix(),
                "zip_sha256": sha256_file(zip_path),
                "handoff_verify_status": verify_status,
                "handoff_safety_status": safety_status,
                "verify_failures": last_failures,
            }
        status_overrides = desired
    return {
        "ok": False,
        "zip_path": zip_path.as_posix(),
        "json_path": json_path.as_posix(),
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else "",
        "handoff_verify_status": "FAIL",
        "handoff_safety_status": status_overrides.get("handoff_safety_scan", "FAIL"),
        "verify_failures": last_failures,
    }


def build_once(
    *,
    repo_root: Path,
    zip_path: Path,
    json_path: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    handoff_package_path: str,
    handoff_sidecar_path: str,
) -> None:
    assert_branch_head(repo_root, branch, final_head)
    with tempfile.TemporaryDirectory(prefix="qrp-handoff-stage-") as temp:
        stage_root = Path(temp)
        write_bundle(
            repo_root,
            stage_root / "final-branch.bundle",
            branch,
            final_head,
        )
        write_text(
            stage_root / "task_metadata.json",
            stable_json(
                task_metadata(
                    {
                        **task,
                        "mode": mode,
                        "handoff_package_path": handoff_package_path,
                        "handoff_sidecar_path": handoff_sidecar_path,
                    }
                )
            ),
        )
        write_text(
            stage_root / "validation_evidence.json",
            stable_json(validation_evidence),
        )
        write_text(
            stage_root / "git_status.txt",
            sanitize_handoff_text(git_status_text(repo_root)),
        )
        stage_handoff_evidence(
            stage_root,
            task,
            validation_evidence,
            bundle_path=stage_root / "final-branch.bundle",
            branch=branch,
            base=base,
            final_head=final_head,
            mode=mode,
            zip_filename=zip_path.name,
        )
        manifest = {
            "schema_version": "handoff-artifact-manifest/0.1",
            "tool_source_final_head": final_head,
            "tool_source_repo_paths": HANDOFF_TOOL_SOURCE_PATHS,
            "entries": artifact_entries(stage_root, staged_paths(stage_root)),
        }
        write_text(stage_root / "artifact_manifest.json", stable_json(manifest))
        paths = tuple(
            sorted(
                path.relative_to(stage_root).as_posix()
                for path in stage_root.rglob("*")
                if path.is_file()
            )
        )
        write_zip(zip_path, stage_root, paths)
    sidecar = sidecar_payload(
        zip_path=zip_path,
        json_path=json_path,
        task=task,
        validation_evidence=validation_evidence,
        branch=branch,
        base=base,
        final_head=final_head,
        handoff_package_path=handoff_package_path,
        handoff_sidecar_path=handoff_sidecar_path,
    )
    write_text(json_path, stable_json(sidecar))


def task_metadata(task: dict[str, object]) -> dict[str, object]:
    mode = handoff_mode(task)
    validate_task_contract(task, mode)
    payload = {key: value for key, value in task.items() if key in TASK_METADATA_KEYS}
    if "schema_version" not in payload:
        payload["schema_version"] = "handoff-task-metadata/0.1"
    mode = handoff_mode(payload)
    payload["mode"] = mode
    if mode == "pr_closure" and not str(payload.get("pr_url") or "").strip():
        raise SystemExit("task_pr_url_required_for_pr_closure")
    return json_sanitize(payload)


def validate_task_contract(task: dict[str, object], mode: str) -> None:
    unknown = sorted(set(task) - TASK_INPUT_KEYS - FORBIDDEN_TASK_KEYS)
    if unknown:
        raise SystemExit("task_metadata_unapproved_field: " + ",".join(unknown))
    forbidden = sorted(set(task) & FORBIDDEN_TASK_KEYS)
    if forbidden:
        raise SystemExit("task_metadata_forbidden_field: " + ",".join(forbidden))
    if mode == "evidence" and "pr_url" in task:
        raise SystemExit("task_metadata_evidence_forbidden_field: pr_url")
    if mode == "pr_closure" and not str(task.get("pr_url") or "").strip():
        raise SystemExit("task_pr_url_required_for_pr_closure")


def handoff_mode(task: dict[str, object], explicit_mode: str | None = None) -> str:
    if explicit_mode is not None:
        raw = explicit_mode
    elif "mode" in task:
        raw = task["mode"]
    else:
        raise SystemExit("task_metadata_schema_missing: mode")
    mode = str(raw).strip()
    if mode not in {"evidence", "pr_closure"}:
        raise SystemExit(f"unsupported_handoff_mode: {mode}")
    return mode


def handoff_package_path_value(task: dict[str, object], zip_path: Path) -> str:
    value = task.get("handoff_package_path")
    if value is None:
        value = f".run/handoffs/{zip_path.name}"
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("task_handoff_package_path_invalid")
    return value.replace("\\", "/")


def handoff_sidecar_path_value(task: dict[str, object], json_path: Path) -> str:
    value = task.get("handoff_sidecar_path")
    if value is None:
        value = f".run/handoffs/{json_path.name}"
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("task_handoff_sidecar_path_invalid")
    return value.replace("\\", "/")


def task_validation_evidence(task: dict[str, object]) -> dict[str, object]:
    payload = task.get("validation_evidence")
    if isinstance(payload, dict):
        result = json_sanitize(payload)
        validate_validation_evidence_contract(result)
        return result
    if payload is not None:
        raise SystemExit("validation_evidence_must_be_object")
    return {
        "schema_version": "handoff-validation-evidence/0.1",
        "checks": [
            {
                "name": "validation_evidence",
                "status": "NOT_RUN",
                "evidence": "task validation_evidence was missing",
            }
        ],
    }


def stage_handoff_evidence(
    stage_root: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
    *,
    bundle_path: Path,
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    zip_filename: str,
) -> None:
    handoff_evidence = task.get("handoff_evidence")
    if handoff_evidence is not None and not isinstance(handoff_evidence, Mapping):
        raise SystemExit("handoff_evidence_must_be_object")
    if isinstance(handoff_evidence, Mapping):
        for raw_name, payload in sorted(handoff_evidence.items()):
            name = normalized_handoff_evidence_path(str(raw_name))
            if is_task_provided_handoff_evidence_forbidden_path(name):
                raise SystemExit(f"handoff_evidence_forbidden_path: {name}")
            if not isinstance(payload, Mapping):
                raise SystemExit(f"handoff_evidence_entry_must_be_object: {name}")
            failures = handoff_evidence_forbidden_key_failures(name, payload)
            if failures:
                raise SystemExit(";".join(failures))
            write_text(stage_root / name, stable_json(json_sanitize(payload)))
    write_text(
        stage_root / VERIFICATION_MATRIX_EVIDENCE,
        stable_json(verification_matrix_payload(validation_evidence)),
    )
    write_text(
        stage_root / GENERATION_EVIDENCE,
        stable_json(
            generation_payload(
                task=task,
                validation_evidence=validation_evidence,
                branch=branch,
                base=base,
                final_head=final_head,
                mode=mode,
                zip_filename=zip_filename,
                bundle_path=bundle_path,
            )
        ),
    )
    write_text(
        stage_root / SOURCE_TREE_SCAN_EVIDENCE,
        stable_json(
            source_tree_scan_payload(
                bundle_path=bundle_path,
                branch=branch,
                final_head=final_head,
            )
        ),
    )


def generation_payload(
    *,
    task: dict[str, object],
    validation_evidence: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    mode: str,
    zip_filename: str,
    bundle_path: Path,
) -> dict[str, object]:
    return {
        "schema_version": "handoff-generation/0.1",
        "mode": mode,
        "base": base,
        "final_head": final_head,
        "branch": branch,
        "zip_filename": zip_filename,
        "zip_sha256_source": "sidecar",
        "bundle_sha256": sha256_file(bundle_path),
        "compare_evidence_path": compare_evidence_path(task),
        "source_tree_scan_path": SOURCE_TREE_SCAN_EVIDENCE,
        "validation_status": validation_status(validation_evidence),
    }


def compare_evidence_path(task: dict[str, object]) -> str | None:
    handoff_evidence = task.get("handoff_evidence")
    if not isinstance(handoff_evidence, dict):
        return None
    for raw_name in handoff_evidence:
        name = normalize_zip_path(str(raw_name))
        if not name.startswith("handoff_evidence/"):
            name = "handoff_evidence/" + name
        if not name.endswith(".json"):
            name += ".json"
        if name == "handoff_evidence/baseline-final-compare.json":
            return name
    return None


def validation_status(validation_evidence: dict[str, object]) -> str:
    checks = validation_evidence.get("checks")
    if not isinstance(checks, list):
        return "BLOCKED"
    statuses = {
        str(row.get("status") or "").upper()
        for row in checks
        if isinstance(row, dict)
    }
    if statuses & {"FAIL"}:
        return "FAIL"
    if statuses & {"BLOCKED"}:
        return "BLOCKED"
    if statuses & {"NOT_RUN"}:
        return "NOT_RUN"
    if statuses == {"PASS"}:
        return "PASS"
    return "NOT_RUN"


def source_tree_scan_payload(
    *,
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="handoff-source-tree-scan-") as temp:
        clone_dir = Path(temp) / "repo"
        subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle_path), str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        return source_tree_scan_from_clone(clone_dir, final_head=final_head)


def source_tree_scan_from_clone(
    clone_dir: Path,
    *,
    final_head: str,
) -> dict[str, object]:
    scanned_head = git_text(clone_dir, "rev-parse", "HEAD").strip()
    scan_groups = source_tree_scan_groups(clone_dir)
    findings = [
        finding
        for group in scan_groups.values()
        for finding in group["findings"]
    ]
    failures = []
    if scanned_head != final_head:
        failures.append("source_tree_scan_head_mismatch")
    failures.extend(finding["rule"] for finding in findings)
    return {
        "schema_version": "handoff-source-tree-scan/0.1",
        "status": "PASS" if not failures else "FAIL",
        "scanned_head": scanned_head,
        "expected_head": final_head,
        "scanned_tree": "final-branch.bundle clone",
        "forbidden_rule_count": len(source_tree_forbidden_patterns()),
        "finding_count": len(findings),
        "findings": findings[:50],
        "failures": failures[:50],
        "scan_groups": {
            group: {
                **payload,
                "findings": payload["findings"][:50],
            }
            for group, payload in scan_groups.items()
        },
    }


def source_tree_findings(clone_dir: Path) -> list[dict[str, object]]:
    return source_tree_scan_groups(clone_dir)["retired_terms"]["findings"]


def source_tree_scan_groups(clone_dir: Path) -> dict[str, dict[str, object]]:
    retired_findings = source_tree_retired_term_findings(clone_dir)
    active_test_literal_findings = source_tree_active_test_literal_findings(clone_dir)
    sensitive_negative_literal_findings = (
        source_tree_sensitive_negative_literal_findings(clone_dir)
    )
    return {
        "retired_terms": {
            "status": "PASS" if not retired_findings else "FAIL",
            "finding_count": len(retired_findings),
            "findings": retired_findings,
        },
        "active_test_literals": {
            "status": "PASS" if not active_test_literal_findings else "FAIL",
            "finding_count": len(active_test_literal_findings),
            "findings": active_test_literal_findings,
        },
        "sensitive_negative_literals": {
            "status": "PASS" if not sensitive_negative_literal_findings else "FAIL",
            "finding_count": len(sensitive_negative_literal_findings),
            "findings": sensitive_negative_literal_findings,
        },
    }


def source_tree_retired_term_findings(clone_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_forbidden_patterns().items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="retired_terms",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
        findings.extend(
            source_tree_split_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
            )
        )
    return findings


def source_tree_split_string_findings(
    relative: Path,
    path: Path,
    lines: list[str],
    skip_lines: set[int],
    rule_definition_lines: set[int],
) -> list[dict[str, object]]:
    return source_tree_synthesized_string_findings(
        relative,
        path,
        lines,
        skip_lines,
        rule_definition_lines,
        scan_group="retired_terms",
        patterns=source_tree_forbidden_patterns(),
    )


def source_tree_synthesized_string_findings(
    relative: Path,
    path: Path,
    lines: list[str],
    skip_lines: set[int],
    rule_definition_lines: set[int],
    *,
    scan_group: str,
    patterns: dict[str, re.Pattern[str]],
) -> list[dict[str, object]]:
    if path.suffix != ".py":
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings: list[dict[str, object]] = []
    for line_number, value, scan_mode in source_tree_python_synthesized_strings(
        tree,
        source,
    ):
        if line_number in skip_lines:
            continue
        for rule, pattern in patterns.items():
            if not pattern.search(value):
                continue
            line_text = lines[line_number - 1].strip() if line_number else ""
            if source_tree_path_allowed(
                relative,
                scan_group=scan_group,
                rule=rule,
                line_text=line_text,
                line_number=line_number,
                rule_definition_lines=rule_definition_lines,
            ):
                continue
            findings.append(
                {
                    "path": relative.as_posix(),
                    "line": line_number,
                    "rule": rule,
                    "scan_mode": scan_mode,
                    "synthesized_value": value,
                }
            )
    return findings


def source_tree_python_synthesized_strings(
    tree: ast.AST,
    source: str,
) -> list[tuple[int, str, str]]:
    values: list[tuple[int, str, str]] = []
    seen: set[tuple[int, str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        value = source_tree_static_string_value(node)
        if value is None:
            continue
        item = (getattr(node, "lineno", 0), value, "ast_static_string")
        if item not in seen:
            seen.add(item)
            values.append(item)
    for item in source_tree_token_synthesized_strings(source):
        if item not in seen:
            seen.add(item)
            values.append(item)
    return values


def source_tree_token_synthesized_strings(source: str) -> list[tuple[int, str, str]]:
    try:
        tokens = [
            token
            for token in tokenize.generate_tokens(io.StringIO(source).readline)
            if token.type not in {tokenize.NL, tokenize.NEWLINE}
        ]
    except tokenize.TokenError:
        return []
    values: list[tuple[int, str, str]] = []
    index = 0
    while index < len(tokens):
        lexical_item = tokens[index]
        if lexical_item.type != tokenize.STRING:
            index += 1
            continue
        pieces: list[str] = []
        line_number = lexical_item.start[0]
        cursor = index
        expect_string = True
        saw_operator_or_adjacent = False
        while cursor < len(tokens):
            current = tokens[cursor]
            if expect_string:
                if current.type != tokenize.STRING:
                    break
                try:
                    pieces.append(str(ast.literal_eval(current.string)))
                except (SyntaxError, ValueError):
                    break
                expect_string = False
                cursor += 1
                continue
            if current.type == tokenize.OP and current.string == "+":
                saw_operator_or_adjacent = True
                expect_string = True
                cursor += 1
                continue
            if current.type == tokenize.STRING:
                saw_operator_or_adjacent = True
                try:
                    pieces.append(str(ast.literal_eval(current.string)))
                except (SyntaxError, ValueError):
                    break
                cursor += 1
                continue
            break
        if len(pieces) >= 2 and saw_operator_or_adjacent and not expect_string:
            values.append((line_number, "".join(pieces), "token_static_string"))
            index = cursor
            continue
        index += 1
    return values


def source_tree_static_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = source_tree_static_string_value(node.left)
        right = source_tree_static_string_value(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def source_tree_governance_rule_definition_lines(
    relative: Path,
    path: Path,
) -> set[int]:
    block_names = SOURCE_TREE_GOVERNANCE_RULE_BLOCK_NAMES.get(relative)
    if not block_names or path.suffix != ".py":
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.iter_child_nodes(tree):
        name = source_tree_definition_name(node)
        if name not in block_names:
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start)
        if start and end:
            lines.update(range(start, end + 1))
    return lines


def source_tree_definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def source_tree_allowlist_definition_lines(
    relative: Path,
    lines: list[str],
) -> set[int]:
    del relative, lines
    return set()


def source_tree_active_test_literal_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if (
            relative.parts[:1] != ("tests",)
            and relative
            != Path(
                ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
                "build_handoff_package.py"
            )
        ):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_active_test_literal_patterns(
            clone_dir
        ).items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="active_test_literals",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
        findings.extend(
            source_tree_synthesized_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
                scan_group="active_test_literals",
                patterns=source_tree_active_test_literal_patterns(clone_dir),
            )
        )
    return findings


def source_tree_active_test_literal_path_allowed(relative: Path) -> bool:
    return bool(SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST.get(relative))


def source_tree_active_test_literal_patterns(
    repo_root: Path | None = None,
) -> dict[str, re.Pattern[str]]:
    ensure_runtime_src_path(repo_root)
    from quant_research_platform.contracts.data import (
        DataFrequency,
        data_frequency_slug,
    )
    from quant_research_platform.markets.implementations.a_share.ids import (
        market_context_data_id,
        market_panel_data_id,
    )

    market_bars_data_id = discover_market_bars_data_id_helper(repo_root)
    frequency = DataFrequency.DAILY
    daily_slug = re.escape(data_frequency_slug(frequency))
    return {
        "active_market_bars_data_id": re.compile(
            rf"\b{re.escape(market_bars_data_id(frequency))}\b"
        ),
        "active_market_panel_data_id": re.compile(
            rf"\b{re.escape(market_panel_data_id(frequency))}\b"
        ),
        "active_market_context_data_id": re.compile(
            rf"\b{re.escape(market_context_data_id(frequency))}\b"
        ),
        "active_frequency_slug": re.compile(
            rf"(?:[\"']{daily_slug}[\"']|"
            rf"(?<![A-Za-z0-9_]){daily_slug}(?![A-Za-z0-9_]))"
        ),
    }


def discover_market_bars_data_id_helper(repo_root: Path | None) -> object:
    src_root = ensure_runtime_src_path(repo_root)
    source_root = src_root / "quant_research_platform" / "sources"
    for path in sorted(source_root.rglob("ids.py")) if source_root.exists() else ():
        if "market_bars_data_id" not in path.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            continue
        module_name = ".".join(path.relative_to(src_root).with_suffix("").parts)
        module = importlib.import_module(module_name)
        helper = getattr(module, "market_bars_data_id", None)
        if callable(helper):
            return helper
    raise RuntimeError("active market bars data id helper unavailable")


def ensure_runtime_src_path(repo_root: Path | None = None) -> Path:
    search_roots = []
    if repo_root is not None:
        search_roots.append(repo_root)
    search_roots.extend([Path.cwd(), *SCRIPT_DIR.parents])
    for parent in search_roots:
        src_root = parent / "src"
        if (src_root / "quant_research_platform").is_dir():
            if str(src_root) not in sys.path:
                sys.path.insert(0, str(src_root))
            return src_root
    raise RuntimeError(
        "active literal source helpers unavailable: "
        "expected src/quant_research_platform under repo root"
    )


def source_tree_sensitive_negative_literal_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if relative.parts[:1] not in {(".agents",), ("tests",)}:
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        skip_lines = source_tree_allowlist_definition_lines(relative, lines)
        rule_definition_lines = source_tree_governance_rule_definition_lines(
            relative,
            path,
        )
        for rule, pattern in source_tree_sensitive_negative_patterns().items():
            for line_number, line in enumerate(lines, start=1):
                if line_number in skip_lines:
                    continue
                if not pattern.search(line):
                    continue
                if source_tree_path_allowed(
                    relative,
                    scan_group="sensitive_negative_literals",
                    rule=rule,
                    line_text=line.strip(),
                    line_number=line_number,
                    rule_definition_lines=rule_definition_lines,
                ):
                    continue
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "line": line_number,
                        "rule": rule,
                    }
                )
        findings.extend(
            source_tree_synthesized_string_findings(
                relative,
                path,
                lines,
                skip_lines,
                rule_definition_lines,
                scan_group="sensitive_negative_literals",
                patterns=source_tree_sensitive_negative_patterns(),
            )
        )
    return findings


def source_tree_tracked_text_files(clone_dir: Path) -> tuple[Path, ...]:
    raw = git_text(clone_dir, "ls-files")
    result: list[Path] = []
    for line in raw.splitlines():
        relative = Path(line)
        text_suffixes = {".py", ".md", ".json", ".toml", ".yaml", ".yml"}
        if relative.suffix.lower() not in text_suffixes:
            continue
        result.append(relative)
    return tuple(result)


def source_tree_path_allowed(
    relative: Path,
    *,
    scan_group: str,
    rule: str | None = None,
    line_text: str | None = None,
    line_number: int | None = None,
    rule_definition_lines: set[int] | None = None,
) -> bool:
    in_governance_rule_definition = (
        line_number is not None
        and rule_definition_lines is not None
        and line_number in rule_definition_lines
    )
    if scan_group == "retired_terms":
        allowed_by_rule = SOURCE_TREE_RETIRED_OCCURRENCE_ALLOWLIST.get(relative, {})
        if rule is None or line_text is None:
            return False
        if line_text in allowed_by_rule.get(rule, set()):
            return True
        allowed_rules = SOURCE_TREE_RETIRED_RULE_DEFINITION_ALLOWLIST.get(
            relative,
            set(),
        )
        return rule in allowed_rules and in_governance_rule_definition
    if scan_group == "active_test_literals":
        allowed_rules = SOURCE_TREE_ACTIVE_TEST_LITERAL_ALLOWLIST.get(relative, set())
        if rule not in allowed_rules:
            return False
        if relative in SOURCE_TREE_ACTIVE_TEST_LITERAL_PATHS:
            return True
        return in_governance_rule_definition
    if scan_group == "sensitive_negative_literals":
        allowed = SOURCE_TREE_SENSITIVE_NEGATIVE_LITERAL_ALLOWLIST.get(
            relative,
            set(),
        )
        if isinstance(allowed, set):
            if rule not in allowed:
                return False
            if relative.parts[:1] == ("tests",):
                return True
            return in_governance_rule_definition
        if rule is None or line_text is None:
            return False
        return any(term in line_text for term in allowed.get(rule, set()))
    raise ValueError(f"unsupported_source_tree_scan_group: {scan_group}")


def source_tree_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    return {
        "retired_a_share_market_definition": re.compile(
            r"\ba_share_market_definition\b"
        ),
        "retired_a_share_field_catalog": re.compile(r"\ba_share_field_catalog\b"),
        "retired_a_share_processed_catalog": re.compile(
            r"\ba_share_processed_catalog\b"
        ),
        "retired_a_share_factor_catalog": re.compile(r"\ba_share_factor_catalog\b"),
        "retired_a_share_label_catalog": re.compile(r"\ba_share_label_catalog\b"),
        "retired_build_a_share": re.compile(r"\bbuild_a_share_[A-Za-z0-9_]+\b"),
        "retired_default_a_share": re.compile(
            r"\bdefault_a_share_[A-Za-z0-9_]+\b"
        ),
        "retired_csmar_source_alias": re.compile(r"\bCSMAR_SOURCE_ID\b"),
        "retired_a_share_market_alias": re.compile(r"\bA_SHARE_MARKET_ID\b"),
        "retired_workflow_module": re.compile(r"\bcontracts\.workflow_run\b"),
        "retired_model_algorithm_collection": re.compile(r"\bMODEL_ALGORITHMS\b"),
        "retired_execution_mode": re.compile(r"\bExecutionMode\b"),
        "retired_daily_panel": re.compile(r"\bdaily_panel\b"),
        "retired_daily_market": re.compile(r"\bdaily_market\b"),
        "retired_daily_context": re.compile(r"\bdaily_context\b"),
        "retired_a_share_daily_bar": re.compile(r"\ba_share_daily_bar\b"),
        "retired_a_share_daily_bar_interim": re.compile(
            r"\ba_share_daily_bar_interim\b"
        ),
        "retired_daily_backtest": re.compile(r"\bdaily_backtest\b"),
        "retired_daily_rebalance": re.compile(r"\bdaily_rebalance\b"),
        "retired_daily_open": re.compile(r"\bdaily_open\b"),
        "retired_backtest_spec": re.compile(r"\bBacktestSpec\b"),
        "retired_dry_run_flag": re.compile(r"--dry-run\b"),
        "retired_model_family": re.compile(r"\bmodel_family\b"),
        "retired_prediction_bundle": re.compile(
            r"\b(?:[A-Za-z0-9_]+_)?prediction_bundle(?:_[A-Za-z0-9_]+)?\b"
        ),
        "forbidden_private_real": re.compile(r"\bprivate_real\b", re.IGNORECASE),
        "forbidden_private_data": re.compile(r"\bprivate_data\b", re.IGNORECASE),
        "forbidden_review_evidence": re.compile(r"\breview_evidence\b"),
        "forbidden_review_bundle": re.compile(r"\breview_bundle\b"),
        "forbidden_review_package": re.compile(r"\breview_package\b"),
        "forbidden_review_zip": re.compile(r"\breview_zip\b"),
        "forbidden_review_sidecar": re.compile(r"\breview_sidecar\b"),
        "forbidden_review_zip_path": re.compile(r"\breview_zip_path\b"),
        "forbidden_external_evidence_json_path": re.compile(
            r"\bexternal_evidence_json_path\b"
        ),
        "forbidden_final_sha": re.compile(r"\bfinal_sha\b"),
        "forbidden_final_head_full": re.compile(r"\bfinal_head_full\b"),
        "forbidden_validation_table": re.compile(r"\bvalidation_table\b"),
        "forbidden_review_threads": re.compile(r"\breview_threads\b"),
        "forbidden_review_threads_json": re.compile(r"\breview_threads\.json\b"),
        "forbidden_task_review_threads": re.compile(r"\btask_review_threads\b"),
        "forbidden_verify_review_thread_closure": re.compile(
            r"\bverify_review_thread_closure\b"
        ),
        "forbidden_diff_patch": re.compile(r"\bdiff\.patch\b"),
        "forbidden_pr_body": re.compile(r"\bpr_body(?:\.md)?\b"),
        "forbidden_goal_status": re.compile(r"\bgoal_status\b"),
        "forbidden_render_pr_body": re.compile(
            r"\brender_pr_body(?:_from_validation_evidence)?(?:\.py)?\b"
        ),
    }


def source_tree_sensitive_negative_patterns() -> dict[str, re.Pattern[str]]:
    return {
        "sensitive_local_private_path": re.compile(
            r"(?i)(?:[A-Za-z]:[\\/][^\n]*(?:"
            r"data[\\/](?:raw|interim|processed|research)"
            r"|reports[\\/]private_research"
            r")"
            r"|/(?:Users|home|workspace|workspaces|mnt/data)[^\n]*(?:"
            r"data/(?:raw|interim|processed|research)"
            r"|reports/private_research"
            r")"
            r"|[A-Za-z]:[\\/][^\n]*(?:private|Users)"
            r"|/(?:Users|home)[^\n]*(?:private|data)"
            r"|\\\\[^\n]*(?:server|share|private))"
        ),
        "sensitive_secret": re.compile(
            r"(?i)(?:api[_-]?key|secret|token|password|cookie)"
            r"\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{8,}"
        ),
        "sensitive_provider_marker": re.compile(
            r"(?i)\b(?:vendor_marker|vendor_payload|raw_vendor|provider_marker"
            r"|provider_name=raw-provider|source_name=raw-source"
            r"|data_provider=raw-provider|vendor_name=raw-provider)\b"
        ),
        "sensitive_stock_code_sample": re.compile(
            r"\b[03468]\d{5}\.(?:SZ|SH|BJ)\b"
        ),
    }


def is_reserved_handoff_evidence_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    for part in parts:
        if part in RESERVED_HANDOFF_EVIDENCE_STEMS:
            return True
        if part.endswith(".json") and part[:-5] in RESERVED_HANDOFF_EVIDENCE_STEMS:
            return True
    return False


def is_task_provided_handoff_evidence_forbidden_path(name: str) -> bool:
    if name in BUILDER_OWNED_HANDOFF_EVIDENCE:
        return True
    parts = PurePosixPath(name).parts
    for part in parts:
        if part in TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS:
            return True
        if (
            part.endswith(".json")
            and part[:-5] in TASK_PROVIDED_HANDOFF_EVIDENCE_RESERVED_STEMS
        ):
            return True
    return False


def normalized_handoff_evidence_path(raw_name: str) -> str:
    name = normalize_zip_path(raw_name)
    if not name.startswith("handoff_evidence/"):
        name = "handoff_evidence/" + name
    if not name.endswith(".json"):
        name += ".json"
    return name


def handoff_evidence_forbidden_key_failures(
    evidence_path: str,
    payload: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    collect_handoff_evidence_forbidden_key_failures(
        evidence_path,
        payload,
        failures,
    )
    return failures


def collect_handoff_evidence_forbidden_key_failures(
    evidence_path: str,
    payload: object,
    failures: list[str],
    *,
    key_path: tuple[str, ...] = (),
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_text = str(key)
            current_path = (*key_path, key_text)
            if key_text in FORBIDDEN_HANDOFF_EVIDENCE_KEYS:
                dotted = ".".join(current_path)
                failures.append(
                    f"handoff_evidence_forbidden_field: {evidence_path}: {dotted}"
                )
            collect_handoff_evidence_forbidden_key_failures(
                evidence_path,
                value,
                failures,
                key_path=current_path,
            )
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            collect_handoff_evidence_forbidden_key_failures(
                evidence_path,
                item,
                failures,
                key_path=(*key_path, str(index)),
            )


def sidecar_payload(
    *,
    zip_path: Path,
    json_path: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
    handoff_package_path: str | None = None,
    handoff_sidecar_path: str | None = None,
) -> dict[str, object]:
    status_map = validation_status_map(validation_evidence)
    safety = scan_handoff_zip(zip_path)
    zip_filename = zip_path.name
    package_logical = (
        handoff_package_path
        if handoff_package_path is not None
        else handoff_package_path_value(task, zip_path)
    )
    sidecar_logical = (
        handoff_sidecar_path
        if handoff_sidecar_path is not None
        else handoff_sidecar_path_value(task, json_path)
    )
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "zip_filename": zip_filename,
        "handoff_package_path": package_logical,
        "handoff_sidecar_path": sidecar_logical,
        "zip_sha256": sha256_file(zip_path),
        "artifact_manifest_sha256": zip_member_sha256(
            zip_path,
            "artifact_manifest.json",
        ),
        "branch": branch,
        "base": base,
        "final_head": final_head,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "status_summary": status_summary_from_validation(
            status_map,
            safety_status="PASS" if safety.ok else "FAIL",
        ),
    }


def status_summary_from_validation(
    status_map: dict[str, str],
    *,
    safety_status: str,
) -> dict[str, object]:
    statuses = dict(status_map)
    statuses["handoff_safety_scan"] = safety_status
    return {
        "validation": aggregate_status(statuses.values()),
        "handoff_verify": statuses.get("handoff_verify", "NOT_RUN"),
        "handoff_safety": safety_status,
        "blocked_count": sum(1 for status in statuses.values() if status == "BLOCKED"),
        "fail_count": sum(1 for status in statuses.values() if status == "FAIL"),
    }


def validation_with_status_overrides(
    validation_evidence: dict[str, object],
    overrides: dict[str, str],
) -> dict[str, object]:
    payload = json.loads(stable_json(validation_evidence))
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    validate_validation_evidence_contract(payload)
    for check_name, status in overrides.items():
        set_check_status(rows, check_name, status)
    return payload


def enrich_validation_evidence(
    validation_evidence: dict[str, object],
    *,
    handoff_package_path: str,
    handoff_sidecar_path: str,
) -> dict[str, object]:
    payload = json.loads(stable_json(validation_evidence))
    validate_validation_evidence_contract(payload)
    payload["handoff_package_path"] = handoff_package_path
    payload["handoff_sidecar_path"] = handoff_sidecar_path
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    for name in (
        "package_safety_check_zip",
        "package_safety_check_sidecar",
    ):
        if name not in validation_status_map(payload):
            set_check_status(rows, name, "NOT_RUN")
    return payload


def validate_validation_evidence_contract(payload: dict[str, object]) -> None:
    forbidden = sorted(set(payload) & FORBIDDEN_VALIDATION_KEYS)
    if forbidden:
        raise SystemExit("validation_evidence_forbidden_field: " + ",".join(forbidden))
    rows = payload.get("checks")
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"validation_evidence_check_not_object: {index}")
        if "check" in row:
            raise SystemExit(f"validation_evidence_row_forbidden_field: {index}:check")
        extra = sorted(set(row) - VALIDATION_ROW_KEYS)
        if extra:
            raise SystemExit(
                f"validation_evidence_row_unapproved_field: {index}:"
                + ",".join(extra)
            )


def set_check_status(
    rows: list[object],
    check_name: str,
    status: str,
) -> None:
    details_key = "evidence"
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if name == check_name:
            row["status"] = status
            row[details_key] = validation_status_evidence(check_name, status)
            return
    rows.append(
        {
            "name": check_name,
            "status": status,
            details_key: validation_status_evidence(check_name, status),
        }
    )


def validation_status_evidence(check_name: str, status: str) -> str:
    if check_name == "package_safety_check_zip":
        return f"package_safety_check.py completed for final zip with {status}"
    if check_name == "package_safety_check_sidecar":
        return f"package_safety_check.py completed for sidecar JSON with {status}"
    if check_name == "handoff_verify":
        return f"verify_handoff_package.py completed with {status}"
    if check_name == "handoff_safety_scan":
        return f"handoff_safety.py completed with {status}"
    return f"{check_name}={status}"


def verification_matrix_payload(
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    status_map = validation_status_map(validation_evidence)
    checks = []
    for name in ("handoff_verify", "handoff_safety_scan"):
        checks.append(
            {
                "name": name,
                "required": True,
                "deterministic": True,
                "status": status_map.get(name, "NOT_RUN"),
                "evidence": validation_status_evidence(
                    name,
                    status_map.get(name, "NOT_RUN"),
                ),
            }
        )
    return {
        "schema_version": "handoff-verification-matrix/0.1",
        "checks": checks,
    }


def validation_status_map(payload: dict[str, object]) -> dict[str, str]:
    rows = payload["checks"]
    if not isinstance(rows, list):
        raise SystemExit("validation_evidence_checks_must_be_list")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        status = str(row.get("status") or "").strip().upper()
        if isinstance(name, str) and status in VALID_STATUSES:
            result[name] = status
    return result


def aggregate_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if not values:
        return "NOT_RUN"
    for status in ("FAIL", "BLOCKED", "NOT_RUN"):
        if status in values:
            return status
    return "PASS"


def artifact_entries(root: Path, paths: Iterable[str]) -> list[dict[str, object]]:
    return [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(paths)
    ]


def staged_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        )
    )


def write_zip(zip_path: Path, root: Path, paths: Iterable[str]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative in sorted(paths):
            archive.write(root / relative, relative)


def write_bundle(
    repo_root: Path,
    bundle_path: Path,
    branch: str,
    final_head: str,
) -> None:
    run_git(repo_root, "bundle", "create", str(bundle_path), branch)
    with tempfile.TemporaryDirectory(prefix="handoff-bundle-check-") as temp:
        clone_dir = Path(temp) / "clone"
        subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle_path), str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        cloned_head = git_text(clone_dir, "rev-parse", "HEAD").strip()
        if cloned_head != final_head:
            raise SystemExit(f"bundle_head_mismatch: {cloned_head} != {final_head}")
        subject = git_text(clone_dir, "show", "-s", "--format=%s", "HEAD").strip()
        if "synthetic" in subject.lower():
            raise SystemExit("bundle_head_synthetic_marker")


def package_safety_status(path: Path) -> str:
    script = SCRIPT_PARENT / "package_safety_check.py"
    result = subprocess.run(
        [sys.executable, str(script), str(path)],
        capture_output=True,
        text=True,
    )
    return "PASS" if result.returncode == 0 else "FAIL"


def assert_branch_head(repo_root: Path, branch: str, final_head: str) -> None:
    actual = git_text(repo_root, "rev-parse", branch).strip()
    if actual != final_head:
        raise SystemExit(f"branch_head_mismatch: {actual} != {final_head}")


def git_status_text(repo: Path) -> str:
    status = git_text(repo, "status", "--short").strip()
    return (status if status else "clean") + "\n"


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True)


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def load_json_object(path: Path, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{label}_must_be_json_object")
    return payload


def required_string(payload: dict[str, object], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise SystemExit(f"task_{key}_required")
    return value


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def stable_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def json_sanitize(payload: object) -> object:
    return json.loads(sanitize_handoff_text(stable_json(payload)))


def normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sanitize_handoff_text(text: str) -> str:
    result = normalize_lf(text)
    result = re.sub(
        r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s`'\"),\]}]*",
        "<LOCAL_ABSOLUTE_PATH>",
        result,
    )
    result = re.sub(
        r"(?i)/(?:Users|home)/[^\s`'\"),\]}]+",
        "<LOCAL_ABSOLUTE_PATH>",
        result,
    )
    result = re.sub(
        r"(?i)/mnt/data[^\s`'\"),\]}]*",
        "<LOCAL_ABSOLUTE_PATH>",
        result,
    )
    result = re.sub(r"(?i)quant-workspace", "<LOCAL_WORKSPACE>", result)
    result = re.sub(r"(?i)site-packages", "<ENV_PATH_TEXT>", result)
    result = re.sub(r"(?i)conda env path", "<ENV_PATH_TEXT>", result)
    result = re.sub(r"\\\\Users\\\\", "WINDOWS_USER_PATH_MARKER", result)
    result = re.sub(
        r"(?i)\b(?:vendor_marker|vendor_payload|raw_vendor|provider_marker)\b",
        "<VENDOR_MARKER_TEXT>",
        result,
    )
    result = re.sub(
        r"Traceback \(most recent call last\):",
        "<TRACEBACK_MARKER_TEXT>",
        result,
    )
    result = re.sub(
        r"(?i)\b(?:runtime_payload|private_runtime_payload|"
        r"exact_private_runtime_payload)\b",
        "<PRIVATE_RUNTIME_PAYLOAD_TEXT>",
        result,
    )
    result = re.sub(
        r'"\\\\Users\\\\[^"\n]+',
        '"<LOCAL_ABSOLUTE_PATH>',
        result,
    )
    result = re.sub(
        r"(?<![A-Za-z0-9.])(?:[03468]\d{5})(?:\.(?:SZ|SH|BJ))?(?![A-Za-z0-9])",
        "<STOCK_CODE_SAMPLE>",
        result,
        flags=re.I,
    )
    result = re.sub(
        r"(?i)\b(?:token|secret|credential|password|api[_-]?key)\s*[:=]\s*"
        r"[A-Za-z0-9._~+/=-]{8,}",
        "<CREDENTIAL_LIKE_TEXT>",
        result,
    )
    return result


def normalize_zip_path(value: str) -> str:
    result = value.replace("\\", "/").lstrip("/")
    if not result or ".." in Path(result).parts:
        raise SystemExit(f"invalid_zip_path: {value}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_member_sha256(zip_path: Path, name: str) -> str:
    with ZipFile(zip_path) as archive:
        return hashlib.sha256(archive.read(name)).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
