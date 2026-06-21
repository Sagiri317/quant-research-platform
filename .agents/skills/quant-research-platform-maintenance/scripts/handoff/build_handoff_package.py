"""Build a public-safe QRP review/handoff package from one task JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
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
    "diff" + ".patch",
    "pr_body" + ".md",
    "review" + "_evidence",
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
TASK_CONTROL_KEYS = {
    "validation_evidence",
    "review_threads",
    "review" + "_evidence",
    "handoff_evidence",
}


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
    final_head = str(task.get("final_head") or task.get("final_sha") or "").strip()
    if not final_head:
        raise SystemExit("task_final_head_required")
    zip_path = out_dir / f"{slug}.zip"
    json_path = out_dir / f"{slug}.json"
    status_overrides: dict[str, str] = {}
    last_failures: list[str] = []
    max_attempts = 3 if verify else 1
    for _attempt in range(max_attempts):
        validation_evidence = enrich_validation_evidence(
            validation_with_status_overrides(
                task_validation_evidence(task),
                status_overrides,
                command_context={
                    "zip_logical_path": str(
                        task.get("zip_logical_path")
                        or f".run/handoffs/{zip_path.name}"
                    ).replace("\\", "/"),
                    "json_logical_path": str(
                        task.get("sidecar_logical_path")
                        or f".run/handoffs/{json_path.name}"
                    ).replace("\\", "/"),
                    "base": base,
                    "final_head": final_head,
                },
            ),
            zip_logical_path=str(
                task.get("zip_logical_path") or f".run/handoffs/{zip_path.name}"
            ).replace("\\", "/"),
            json_logical_path=str(
                task.get("sidecar_logical_path") or f".run/handoffs/{json_path.name}"
            ).replace("\\", "/"),
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
            stable_json(task_metadata({**task, "mode": mode})),
        )
        write_text(
            stage_root / "validation_evidence.json",
            stable_json(validation_evidence),
        )
        write_text(
            stage_root / "git_status.txt",
            sanitize_handoff_text(git_status_text(repo_root)),
        )
        if mode == "pr_closure":
            review_threads = task_review_threads(task)
            write_text(stage_root / "review_threads.json", stable_json(review_threads))
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
        task=task,
        validation_evidence=validation_evidence,
        branch=branch,
        base=base,
        final_head=final_head,
    )
    write_text(json_path, stable_json(sidecar))


def task_metadata(task: dict[str, object]) -> dict[str, object]:
    payload = {
        key: value
        for key, value in task.items()
        if key not in TASK_CONTROL_KEYS and not key.endswith("_path")
    }
    payload.pop("goal_status", None)
    payload.setdefault("schema_version", "handoff-task-metadata/0.1")
    return json_sanitize(payload)


def handoff_mode(task: dict[str, object], explicit_mode: str | None = None) -> str:
    raw = explicit_mode or task.get("mode") or "evidence"
    mode = str(raw).strip()
    if mode not in {"evidence", "pr_closure"}:
        raise SystemExit(f"unsupported_handoff_mode: {mode}")
    return mode


def task_validation_evidence(task: dict[str, object]) -> dict[str, object]:
    payload = task.get("validation_evidence")
    if isinstance(payload, dict):
        return json_sanitize(payload)
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


def task_review_threads(task: dict[str, object]) -> dict[str, object]:
    payload = task.get("review_threads")
    if isinstance(payload, dict):
        result = json_sanitize(payload)
        result.setdefault("status", "NOT_RUN")
        status = str(result.get("status") or "").strip().upper()
        if status == "BLOCKED":
            result["counts_available"] = False
            result["total_count"] = None
            result["unresolved_count"] = None
            result.setdefault(
                "evidence",
                (
                    "Live GitHub review-thread access was unavailable; counts are "
                    "unknown. No PASS is claimed for live thread state."
                ),
            )
        else:
            result.setdefault("counts_available", status == "PASS")
            if result.get("counts_available"):
                result.setdefault("total_count", 0)
                result.setdefault("unresolved_count", 0)
            else:
                result.setdefault("total_count", None)
                result.setdefault("unresolved_count", None)
        result.setdefault("threads", [])
        return result
    return {
        "schema_version": "handoff-review-threads/0.1",
        "status": "NOT_RUN",
        "counts_available": False,
        "total_count": None,
        "unresolved_count": None,
        "threads": [],
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
    if isinstance(handoff_evidence, dict):
        for raw_name, payload in sorted(handoff_evidence.items()):
            name = normalize_zip_path(str(raw_name))
            if not name.startswith("handoff_evidence/"):
                name = "handoff_evidence/" + name
            if not name.endswith(".json"):
                name += ".json"
            if name in BUILDER_OWNED_HANDOFF_EVIDENCE:
                continue
            if is_reserved_handoff_evidence_path(name):
                continue
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
        "zip_sha256": "external_sidecar",
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
    return {
        "retired_terms": {
            "finding_count": len(retired_findings),
            "findings": retired_findings,
        },
        "active_test_literals": {
            "finding_count": len(active_test_literal_findings),
            "findings": active_test_literal_findings,
        },
    }


def source_tree_retired_term_findings(clone_dir: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if source_tree_path_allowed(relative):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for rule, pattern in source_tree_forbidden_patterns().items():
            if pattern.search(text):
                findings.append({"path": relative.as_posix(), "rule": rule})
    return findings


def source_tree_active_test_literal_findings(
    clone_dir: Path,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for relative in source_tree_tracked_text_files(clone_dir):
        if not relative.parts or relative.parts[0] != "tests":
            continue
        if source_tree_active_test_literal_path_allowed(relative):
            continue
        path = clone_dir / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in source_tree_active_test_literal_tokens():
            if token in text:
                findings.append(
                    {
                        "path": relative.as_posix(),
                        "rule": "active_test_literal",
                        "token": token,
                    }
                )
    return findings


def source_tree_active_test_literal_path_allowed(relative: Path) -> bool:
    return relative in {
        Path("tests/architecture/test_frequency_vocabulary_vs_active_support.py"),
        Path("tests/architecture/test_retired_surface_absent.py"),
    }


def source_tree_active_test_literal_tokens() -> tuple[str, ...]:
    return (
        "market_bars_1d",
        "market_panel_1d",
        "market_context_1d",
        '"1d"',
        "'1d'",
        '"1" + "d"',
        "'1' + 'd'",
    )


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


def source_tree_path_allowed(relative: Path) -> bool:
    allowed = {
        Path("tests/agents/handoff/test_build_handoff_package_integration.py"),
        Path("tests/agents/handoff/test_handoff_tools.py"),
        Path("tests/agents/handoff/test_package_safety_check.py"),
        Path("tests/architecture/test_handoff_package_contract.py"),
        Path("tests/architecture/test_retired_surface_absent.py"),
        Path("tests/interfaces/test_api_cli_01_contract.py"),
        Path("tests/reports/test_reports_01.py"),
        Path("tests/runtime/test_logical_paths.py"),
        Path("tests/runtime/test_privacy_01.py"),
        Path("tests/storage/test_manifest_helper_edges.py"),
        Path("tests/storage/test_storage_01.py"),
        Path("docs/architecture/open-issues-register.md"),
        *(Path(path) for path in HANDOFF_TOOL_SOURCE_PATHS),
    }
    return relative in allowed


def source_tree_forbidden_patterns() -> dict[str, re.Pattern[str]]:
    review_evidence = "review" + "_evidence"
    diff_patch = "diff" + r"\.patch"
    pr_body = "pr_body"
    render_pr_body = "render" + "_pr_body"
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
        "forbidden_private_real": re.compile(r"\bprivate_real\b", re.IGNORECASE),
        "forbidden_private_data": re.compile(r"\bprivate_data\b", re.IGNORECASE),
        "forbidden_review_evidence": re.compile(
            rf"\b{re.escape(review_evidence)}\b"
        ),
        "forbidden_diff_patch": re.compile(rf"\b{diff_patch}\b"),
        "forbidden_pr_body": re.compile(rf"\b{pr_body}(?:\.md)?\b"),
        "forbidden_render_pr_body": re.compile(
            rf"\b{render_pr_body}(?:_from_validation_evidence)?(?:\.py)?\b"
        ),
        "forbidden_local_private_path": re.compile(
            r"(?i)(?:[A-Za-z]:[\\/][^\n]*(?:"
            r"data[\\/](?:raw|interim|processed|research)"
            r"|reports[\\/]private_research"
            r")"
            r"|/(?:Users|home|workspace|workspaces|mnt/data)[^\n]*(?:"
            r"data/(?:raw|interim|processed|research)"
            r"|reports/private_research"
            r"))"
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


def sidecar_payload(
    *,
    zip_path: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
    branch: str,
    base: str,
    final_head: str,
) -> dict[str, object]:
    status_map = validation_status_map(validation_evidence)
    safety = scan_handoff_zip(zip_path)
    zip_filename = zip_path.name
    logical = str(
        task.get("zip_logical_path")
        or f".run/handoffs/{zip_filename}"
    ).replace("\\", "/")
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "zip_filename": zip_filename,
        "zip_logical_path": logical,
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
    *,
    command_context: dict[str, str] | None = None,
) -> dict[str, object]:
    payload = json.loads(stable_json(validation_evidence))
    rows = payload.get("checks")
    if not isinstance(rows, list):
        rows = payload.get("validation_table")
    if not isinstance(rows, list):
        rows = []
        payload["checks"] = rows
    for check_name, status in overrides.items():
        set_check_status(rows, check_name, status, command_context=command_context)
    return payload


def enrich_validation_evidence(
    validation_evidence: dict[str, object],
    *,
    zip_logical_path: str,
    json_logical_path: str,
) -> dict[str, object]:
    payload = json.loads(stable_json(validation_evidence))
    payload["review_zip_path"] = zip_logical_path
    payload["external_evidence_json_path"] = json_logical_path
    rows = payload.get("checks")
    if not isinstance(rows, list):
        rows = []
        payload["checks"] = rows
    for name in (
        "package_safety_check_zip",
        "package_safety_check_sidecar",
    ):
        if name not in validation_status_map(payload):
            set_check_status(rows, name, "NOT_RUN")
    return payload


def set_check_status(
    rows: list[object],
    check_name: str,
    status: str,
    *,
    command_context: dict[str, str] | None = None,
) -> None:
    details_key = "evidence"
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name", row.get("check"))
        if name == check_name:
            row["status"] = status
            row[details_key] = validation_status_evidence(check_name, status)
            row.setdefault(
                "command",
                validation_status_command(check_name, command_context=command_context),
            )
            return
    rows.append(
        {
            "name": check_name,
            "status": status,
            "command": validation_status_command(
                check_name,
                command_context=command_context,
            ),
            details_key: validation_status_evidence(check_name, status),
        }
    )


def validation_status_command(
    check_name: str,
    *,
    command_context: dict[str, str] | None = None,
) -> str:
    context = command_context or {}
    zip_path = context.get("zip_logical_path", ".run/handoffs/handoff.zip")
    json_path = context.get("json_logical_path", ".run/handoffs/handoff.json")
    if check_name == "package_safety_check_zip":
        return (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"package_safety_check.py {zip_path}"
        )
    if check_name == "package_safety_check_sidecar":
        return (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"package_safety_check.py {json_path}"
        )
    if check_name == "handoff_verify":
        return (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"handoff/verify_handoff_package.py --zip {zip_path} --json {json_path}"
        )
    if check_name == "handoff_safety_scan":
        return (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"handoff/handoff_safety.py {zip_path} --json {json_path}"
        )
    return check_name


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
    rows = payload.get("checks", payload.get("validation_table", []))
    if not isinstance(rows, list):
        return {}
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name", row.get("check"))
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
