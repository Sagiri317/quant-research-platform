"""Build a share-safe QRP handoff package from one task JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_PARENT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PARENT))
if str(SCRIPT_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_SRC))

from governance_contract import (  # noqa: E402
    GENERATION_EVIDENCE,
    HANDOFF_TOOL_SOURCE_PATHS,
    MANDATORY_VALIDATION_CHECKS,
    SOURCE_TREE_SCAN_EVIDENCE,
    TASK_INPUT_KEYS,
    TASK_METADATA_KEYS,
    TASK_REJECTED_KEYS,
    VERIFICATION_MATRIX_EVIDENCE,
    advisory_review_consistency_failures,
    aggregate_mandatory_validation_status,
    enrich_validation_evidence,
    handoff_evidence_blocked_key_failures,
    is_task_provided_handoff_evidence_blocked_path,
    normalized_handoff_evidence_path,
    normalized_handoff_markdown_evidence_path,
    source_tree_scan_payload,
    validate_validation_evidence_contract,
    validation_status_map,
    validation_with_status_overrides,
    verification_matrix_payload,
)

from quant_research_platform.policy.check_status import (  # noqa: E402
    aggregate_validation_matrix_status,
)
from quant_research_platform.policy.share_safety import (  # noqa: E402
    qrp_json_identity_reason,
)

PACKAGE_SCHEMA_ID = "qrp-handoff-sidecar"
ADVISORY_REVIEW_CHECKS = {"review_subagent"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a share-safe QRP handoff package from one task JSON."
    )
    parser.add_argument("--task", required=True, help="Share-safe task input JSON.")
    parser.add_argument("--out-dir", default=".run/handoffs")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--mode", choices=("evidence", "pr_closure"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_dir = resolve_output_dir(repo_root, args.out_dir)
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


def resolve_output_dir(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


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
            "PASS" if handoff_zip_scan(zip_path, json_path=json_path).ok else "FAIL"
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
            share_safe_text_clean(git_status_text(repo_root)),
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
            "schema_id": "qrp-handoff-artifact-manifest",
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
    if "schema_id" not in payload:
        payload["schema_id"] = "qrp-handoff-task-metadata"
    mode = handoff_mode(payload)
    payload["mode"] = mode
    if mode == "pr_closure" and not str(payload.get("pr_url") or "").strip():
        raise SystemExit("task_pr_url_required_for_pr_closure")
    return share_safe_json_clean(payload)


def validate_task_contract(task: dict[str, object], mode: str) -> None:
    unknown = sorted(set(task) - TASK_INPUT_KEYS - TASK_REJECTED_KEYS)
    if unknown:
        raise SystemExit("task_metadata_unapproved_field: " + ",".join(unknown))
    blocked = sorted(set(task) & TASK_REJECTED_KEYS)
    if blocked:
        raise SystemExit("task_metadata_forbidden_field: " + ",".join(blocked))
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
    if "validation_evidence" not in task:
        return missing_validation_evidence_payload()
    payload = task["validation_evidence"]
    if not isinstance(payload, dict):
        raise SystemExit("validation_evidence_must_be_object")
    if "checks" not in payload:
        raise SystemExit("validation_evidence_checks_missing")
    result = share_safe_json_clean(payload)
    validate_validation_evidence_contract(result, require_mandatory=False)
    return result


def missing_validation_evidence_payload() -> dict[str, object]:
    return {
        "schema_id": "qrp-handoff-validation-evidence",
        "checks": [
            {
                "name": name,
                "status": "NOT_RUN",
                "evidence": (
                    "task validation_evidence missing; "
                    "generated mandatory NOT_RUN row"
                ),
            }
            for name in sorted(MANDATORY_VALIDATION_CHECKS)
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
    evidence_paths: list[str] = []
    if handoff_evidence is not None and not isinstance(handoff_evidence, Mapping):
        raise SystemExit("handoff_evidence_must_be_object")
    if isinstance(handoff_evidence, Mapping):
        for raw_name, payload in sorted(handoff_evidence.items()):
            name = normalized_handoff_evidence_path(str(raw_name))
            evidence_paths.append(name)
            if is_task_provided_handoff_evidence_blocked_path(name):
                raise SystemExit(f"handoff_evidence_forbidden_path: {name}")
            if not isinstance(payload, Mapping):
                raise SystemExit(f"handoff_evidence_entry_must_be_object: {name}")
            identity_reason = qrp_json_identity_reason(payload)
            if identity_reason is not None:
                raise SystemExit(
                    f"handoff_evidence_json_identity_invalid: {name}: {identity_reason}"
                )
            failures = handoff_evidence_blocked_key_failures(name, payload)
            if failures:
                raise SystemExit(";".join(failures))
            write_text(stage_root / name, stable_json(share_safe_json_clean(payload)))
    markdown_evidence = task.get("handoff_markdown_evidence")
    if markdown_evidence is not None and not isinstance(markdown_evidence, Mapping):
        raise SystemExit("handoff_markdown_evidence_must_be_object")
    if isinstance(markdown_evidence, Mapping):
        for raw_name, payload in sorted(markdown_evidence.items()):
            name = normalized_handoff_markdown_evidence_path(str(raw_name))
            evidence_paths.append(name)
            if is_task_provided_handoff_evidence_blocked_path(name):
                raise SystemExit(f"handoff_evidence_forbidden_path: {name}")
            if not isinstance(payload, str):
                raise SystemExit(
                    f"handoff_markdown_evidence_entry_must_be_string: {name}"
                )
            write_text(stage_root / name, share_safe_text_clean(payload))
    review_failures = advisory_review_consistency_failures(
        evidence_paths,
        advisory_reviews_from_validation(validation_status_map(validation_evidence)),
    )
    if review_failures:
        raise SystemExit(";".join(review_failures))
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
        "schema_id": "qrp-handoff-generation",
        "mode": mode,
        "base": base,
        "final_head": final_head,
        "branch": branch,
        "zip_filename": zip_filename,
        "zip_sha256_source": "sidecar",
        "bundle_sha256": sha256_file(bundle_path),
        "source_tree_scan_path": SOURCE_TREE_SCAN_EVIDENCE,
        "validation_status": validation_status(validation_evidence),
    }


def validation_status(validation_evidence: dict[str, object]) -> str:
    return aggregate_mandatory_validation_status(validation_evidence)


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
    safety = handoff_zip_scan(zip_path)
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
        "schema_id": PACKAGE_SCHEMA_ID,
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
        "advisory_reviews": advisory_reviews_from_validation(status_map),
    }


def handoff_zip_scan(zip_path: Path, *, json_path: Path | None = None):
    from verify_handoff_package import scan_handoff_zip

    return scan_handoff_zip(zip_path, json_path=json_path)


def status_summary_from_validation(
    status_map: dict[str, str],
    *,
    safety_status: str,
) -> dict[str, object]:
    statuses = {
        name: status
        for name, status in status_map.items()
        if name in MANDATORY_VALIDATION_CHECKS
    }
    statuses["handoff_safety_scan"] = safety_status
    return {
        "validation": aggregate_validation_matrix_status(statuses.values()),
        "handoff_verify": statuses.get("handoff_verify", "NOT_RUN"),
        "handoff_safety": safety_status,
        "blocked_count": sum(1 for status in statuses.values() if status == "BLOCKED"),
        "fail_count": sum(1 for status in statuses.values() if status == "FAIL"),
    }


def advisory_reviews_from_validation(status_map: dict[str, str]) -> dict[str, str]:
    return {
        name: status
        for name, status in sorted(status_map.items())
        if name in ADVISORY_REVIEW_CHECKS
    }


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


def share_safe_json_clean(payload: object) -> object:
    return payload


def normalize_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def share_safe_text_clean(text: str) -> str:
    return normalize_lf(text)


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
