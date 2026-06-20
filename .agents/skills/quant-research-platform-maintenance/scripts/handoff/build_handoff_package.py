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
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))

from handoff_safety import scan_handoff_zip  # noqa: E402
from render_pr_body_from_validation_evidence import render_pr_body  # noqa: E402

PACKAGE_SCHEMA_VERSION = "handoff-sidecar/0.1"
VALID_STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
REQUIRED_REVIEW_EVIDENCE = "review_evidence/handoff_generation.json"
VERIFICATION_MATRIX_EVIDENCE = "review_evidence/verification-matrix.json"
TASK_CONTROL_KEYS = {
    "validation_evidence",
    "review_threads",
    "review_evidence",
    "pr_body",
    "pr_body_markdown",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a public-safe QRP review package from one task JSON."
    )
    parser.add_argument("--task", required=True, help="Public-safe task input JSON.")
    parser.add_argument("--out-dir", default=".run/reviews")
    parser.add_argument("--repo-root", default=".")
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
    verify: bool,
) -> dict[str, object]:
    slug = required_string(task, "slug")
    branch = required_string(task, "branch")
    base = required_string(task, "base")
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
                        or f".run/reviews/{zip_path.name}"
                    ).replace("\\", "/"),
                    "json_logical_path": str(
                        task.get("sidecar_logical_path")
                        or f".run/reviews/{json_path.name}"
                    ).replace("\\", "/"),
                    "base": base,
                    "final_head": final_head,
                },
            ),
            zip_logical_path=str(
                task.get("zip_logical_path") or f".run/reviews/{zip_path.name}"
            ).replace("\\", "/"),
            json_logical_path=str(
                task.get("sidecar_logical_path") or f".run/reviews/{json_path.name}"
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
            "raw_diff_exactness": (
                "FAIL" if "diff_patch_mismatch" in last_failures else "PASS"
            ),
            "handoff_verify_includes_git_apply_check": (
                "FAIL" if "diff_patch_apply_check_failed" in last_failures else "PASS"
            ),
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
        write_text(stage_root / "task_metadata.json", stable_json(task_metadata(task)))
        write_text(
            stage_root / "validation_evidence.json",
            stable_json(validation_evidence),
        )
        write_text(
            stage_root / "git_status.txt",
            sanitize_handoff_text(git_status_text(repo_root)),
        )
        diff = git_text(
            repo_root,
            "diff",
            "--full-index",
            "--binary",
            f"{base}..{final_head}",
        )
        write_text(stage_root / "diff.patch", normalize_lf(diff))
        review_threads = task_review_threads(task)
        write_text(stage_root / "review_threads.json", stable_json(review_threads))
        pr_body = task_pr_body(
            task,
            validation_evidence,
            review_threads,
            external_evidence=sidecar_preview(
                zip_path=zip_path,
                json_path=json_path,
                task=task,
                validation_evidence=validation_evidence,
            ),
        )
        write_text(stage_root / "pr_body.md", sanitize_handoff_text(pr_body))
        stage_review_evidence(stage_root, task, validation_evidence)
        manifest = {
            "schema_version": "handoff-artifact-manifest/0.1",
            "tool_source_final_head": final_head,
            "tool_source_repo_paths": [
                ".agents/skills/quant-research-platform-maintenance/scripts/handoff/build_handoff_package.py",
                ".agents/skills/quant-research-platform-maintenance/scripts/handoff/verify_handoff_package.py",
                ".agents/skills/quant-research-platform-maintenance/scripts/handoff/handoff_safety.py",
                ".agents/skills/quant-research-platform-maintenance/scripts/handoff/self_test.py",
                ".agents/skills/quant-research-platform-maintenance/templates/handoff/task_metadata.template.json",
                ".agents/skills/quant-research-platform-maintenance/schemas/handoff/task_metadata.schema.json",
                ".agents/skills/quant-research-platform-maintenance/schemas/handoff/sidecar.schema.json",
            ],
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


def task_pr_body(
    task: dict[str, object],
    validation_evidence: dict[str, object],
    review_threads: dict[str, object],
    *,
    external_evidence: dict[str, object] | None = None,
) -> str:
    raw = task.get("pr_body_markdown", task.get("pr_body"))
    if isinstance(raw, str) and raw.strip():
        return raw
    return render_pr_body(
        task_metadata(task),
        validation_evidence,
        review_threads=review_threads,
        external_evidence=external_evidence,
    )


def stage_review_evidence(
    stage_root: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
) -> None:
    review_evidence = task.get("review_evidence")
    wrote = False
    if isinstance(review_evidence, dict):
        for raw_name, payload in sorted(review_evidence.items()):
            name = normalize_zip_path(str(raw_name))
            if not name.startswith("review_evidence/"):
                name = "review_evidence/" + name
            if not name.endswith(".json"):
                name += ".json"
            write_text(stage_root / name, stable_json(json_sanitize(payload)))
            wrote = True
    if not (stage_root / VERIFICATION_MATRIX_EVIDENCE).exists():
        write_text(
            stage_root / VERIFICATION_MATRIX_EVIDENCE,
            stable_json(verification_matrix_payload(validation_evidence)),
        )
    if not wrote:
        payload = {
            "schema_version": "handoff-review-evidence/0.1",
            "status": aggregate_status(
                validation_status_map(validation_evidence).values()
            ),
            "evidence": (
                "Generated from task validation_evidence; no extra evidence file "
                "was supplied."
            ),
        }
        write_text(stage_root / REQUIRED_REVIEW_EVIDENCE, stable_json(payload))


def sidecar_preview(
    *,
    zip_path: Path,
    json_path: Path,
    task: dict[str, object],
    validation_evidence: dict[str, object],
) -> dict[str, object]:
    status_map = validation_status_map(validation_evidence)
    return {
        "zip_filename": zip_path.name,
        "zip_logical_path": str(
            task.get("zip_logical_path") or f".run/reviews/{zip_path.name}"
        ).replace("\\", "/"),
        "sidecar_logical_path": str(
            task.get("sidecar_logical_path") or f".run/reviews/{json_path.name}"
        ).replace("\\", "/"),
        "status_summary": status_summary_from_validation(
            status_map,
            safety_status=status_map.get("handoff_safety_scan", "NOT_RUN"),
        ),
    }


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
        or f".run/reviews/{zip_filename}"
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
        "raw_diff_exactness",
        "handoff_verify_includes_git_apply_check",
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
    zip_path = context.get("zip_logical_path", ".run/reviews/handoff.zip")
    json_path = context.get("json_logical_path", ".run/reviews/handoff.json")
    base = context.get("base", "BASE")
    final_head = context.get("final_head", "HEAD")
    if check_name == "raw_diff_exactness":
        return f"git diff --full-index --binary {base}..{final_head}"
    if check_name == "handoff_verify_includes_git_apply_check":
        return (
            "python .agents/skills/quant-research-platform-maintenance/scripts/"
            f"handoff/verify_handoff_package.py --zip {zip_path} --json {json_path}"
        )
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
    if check_name == "raw_diff_exactness":
        return f"raw git diff --full-index --binary base..HEAD exactness check {status}"
    if check_name == "handoff_verify_includes_git_apply_check":
        return (
            "verify_handoff_package.py completed its internal "
            f"git apply --check step with {status}"
        )
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
    for name in ("raw_diff_exactness", "handoff_verify_includes_git_apply_check"):
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
        r"(?<![A-Za-z0-9.])(?:[036]\d{5}|688\d{3})(?:\.(?:SZ|SH|BJ))?(?![A-Za-z0-9])",
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
