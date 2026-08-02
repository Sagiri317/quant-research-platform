"""Verify a replayable QRP handoff zip with one external sidecar JSON."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo
from zlib import error as ZlibError

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_PARENT = SCRIPT_DIR.parent
SCRIPT_REPO_ROOT = SCRIPT_DIR.parents[4]
SCRIPT_REPO_SRC = SCRIPT_REPO_ROOT / "src"
for candidate in (SCRIPT_DIR, SCRIPT_PARENT, SCRIPT_REPO_SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from governance_contract import (  # noqa: E402
    PACKAGE_FORBIDDEN_PATH_PARTS,
    source_tree_scan_groups,
)
from governance_contract import (  # noqa: E402
    package_member_name_failures as shared_package_member_name_failures,
)
from handoff_contract import (  # noqa: E402
    ARTIFACT_MANIFEST_ENTRY,
    CHECK_RECEIPTS_ENTRY,
    FINAL_BRANCH_BUNDLE_ENTRY,
    HARD_SCAN_RECEIPT_ENTRY,
    PACKAGE_SMOKE_RECEIPT_ENTRY,
    REQUIRED_HANDOFF_ENTRIES,
    TASK_METADATA_ENTRY,
    VALIDATION_SUMMARY_ENTRY,
    build_validation_summary_from_receipts,
    mandatory_validation_failures,
    posix_member_name_failures,
    validate_artifact_manifest_contract,
    validate_handoff_package_layout,
    validate_sidecar_contract,
    validate_task_metadata_contract,
    validate_validation_summary_against_receipts,
    validate_validation_summary_contract,
)
from receipt_contract import (  # noqa: E402
    compare_check_receipts,
    compare_hard_scan_receipt,
    compare_package_smoke_receipt,
    validate_receipt_payload,
)
from replay_evidence import (  # noqa: E402
    ReplayEnvironmentBlocked,
    build_machine_receipts,
)

from quant_research_platform.contracts._json import (  # noqa: E402
    parse_qrp_json_object,
)
from quant_research_platform.policy.share_safety import (  # noqa: E402
    evidence_json_value_unsafe_reason,
    unsafe_for_share_reason,
)

TEXT_LIKE_SUFFIXES = {
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
BUNDLE_TEXT_SIZE_CAP_BYTES = 2 * 1024 * 1024
ZIP_MEMBER_SIZE_CAP_BYTES = 256 * 1024 * 1024
ZIP_TOTAL_SIZE_CAP_BYTES = 512 * 1024 * 1024
ZIP_COMPRESSION_RATIO_CAP = 200.0


@dataclass(frozen=True)
class HandoffVerificationResult:
    failures: tuple[str, ...]
    share_safety_failures: tuple[str, ...]
    zip_sha256: str

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def handoff_package_verify_status(self) -> str:
        return "PASS" if not self.failures else "FAIL"

    @property
    def share_safety_status(self) -> str:
        return "PASS" if not self.share_safety_failures else "FAIL"


@dataclass(frozen=True)
class HandoffPackagePayload:
    zip_path: Path
    json_path: Path | None
    zip_sha256: str
    member_names: tuple[str, ...]
    duplicate_member_names: tuple[str, ...]
    raw_bundle: bytes | None
    sidecar_payload: Mapping[str, object] | None
    task_metadata: Mapping[str, object] | None
    validation_summary: Mapping[str, object] | None
    receipts: Mapping[str, Mapping[str, object]]
    load_failures: tuple[str, ...]
    share_safety_failures: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return (
            self.task_metadata is not None
            and self.validation_summary is not None
            and set(self.receipts)
            == {
                PACKAGE_SMOKE_RECEIPT_ENTRY,
                HARD_SCAN_RECEIPT_ENTRY,
                CHECK_RECEIPTS_ENTRY,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    result = verify_handoff_package_result(Path(args.zip), Path(args.json))
    print(f"share_safety_status={result.share_safety_status}")
    if result.failures:
        print("handoff_package_verify_status=FAIL")
        for failure in result.failures:
            print(failure)
        print(f"zip_sha256={result.zip_sha256}")
        return 1
    print("handoff_package_verify_status=PASS")
    print(f"zip_sha256={result.zip_sha256}")
    return 0


def verify_handoff_package_result(
    zip_path: Path, json_path: Path
) -> HandoffVerificationResult:
    payload = load_handoff_package_payload(zip_path, json_path)
    return validate_handoff_package_payload(
        payload,
        replay_bundle=True,
        expected_receipts=None,
    )


def validate_handoff_package_payload(
    package_payload: HandoffPackagePayload,
    *,
    replay_bundle: bool,
    expected_receipts: Mapping[str, Mapping[str, object]] | None = None,
) -> HandoffVerificationResult:
    failures: list[str] = []
    share_safety_failures = list(package_payload.share_safety_failures)
    failures.extend(package_payload.load_failures)
    if (
        package_payload.sidecar_payload is not None
        and package_payload.json_path is not None
    ):
        failures.extend(
            verify_sidecar(
                package_payload.sidecar_payload,
                package_payload.zip_path,
                package_payload.json_path,
                zip_sha256=package_payload.zip_sha256,
            )
        )
    if package_payload.task_metadata is not None:
        failures.extend(validate_task_metadata_contract(package_payload.task_metadata))
    if package_payload.validation_summary is not None:
        failures.extend(
            validate_validation_summary_contract(package_payload.validation_summary)
        )
        failures.extend(
            validate_validation_summary_against_receipts(
                package_payload.validation_summary,
                package_payload.receipts,
            )
        )
        failures.extend(mandatory_validation_failures(package_payload.validation_summary))
    for entry, payload in package_payload.receipts.items():
        failures.extend(validate_receipt_payload(Path(entry).name, payload))
    if expected_receipts is not None and package_payload.complete:
        failures.extend(
            compare_replayed_handoff_payloads(package_payload, expected_receipts)
        )
    if (
        replay_bundle
        and not failures
        and not share_safety_failures
        and package_payload.complete
        and package_payload.task_metadata is not None
        and package_payload.raw_bundle is not None
    ):
        bundle_failures, bundle_share_safety_failures = verify_bundle_and_replay_once(
            package_payload.raw_bundle,
            package_payload.task_metadata,
            package_payload,
        )
        failures.extend(bundle_failures)
        share_safety_failures.extend(bundle_share_safety_failures)
    failures.extend(f"share_safety_{failure}" for failure in share_safety_failures)
    return HandoffVerificationResult(
        failures=tuple(failures),
        share_safety_failures=tuple(share_safety_failures),
        zip_sha256=package_payload.zip_sha256,
    )


def load_handoff_package_payload(
    zip_path: Path,
    json_path: Path | None,
) -> HandoffPackagePayload:
    failures: list[str] = []
    share_safety_failures: list[str] = []
    names: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    raw_bundle: bytes | None = None
    member_bytes: dict[str, bytes] = {}
    task_metadata: dict[str, object] | None = None
    validation_summary: dict[str, object] | None = None
    receipts: dict[str, dict[str, object]] = {}
    sidecar_payload: dict[str, object] | None = None
    try:
        zip_digest = sha256_file(zip_path)
    except OSError as exc:
        return HandoffPackagePayload(
            zip_path=zip_path,
            json_path=json_path,
            zip_sha256="",
            member_names=(),
            duplicate_member_names=(),
            raw_bundle=None,
            sidecar_payload=None,
            task_metadata=None,
            validation_summary=None,
            receipts={},
            load_failures=(f"zip_read_failed: {exc}",),
            share_safety_failures=(),
        )
    try:
        with ZipFile(zip_path) as archive:
            members = archive.infolist()
            required_member_count = len(REQUIRED_HANDOFF_ENTRIES)
            if len(members) != required_member_count:
                failures.append(
                    "zip_member_count_invalid: "
                    f"expected {required_member_count}, observed {len(members)}"
                )
            bounded_members = members[: required_member_count + 1]
            names = tuple(member.filename for member in bounded_members)
            name_set = set(names)
            duplicates = tuple(
                sorted(name for name, count in Counter(names).items() if count > 1)
            )
            failures.extend(f"duplicate_zip_member: {name}" for name in duplicates)
            for name in names:
                failures.extend(posix_member_name_failures(name))
                share_safety_failures.extend(
                    f"{failure}: {name}" for failure in member_name_failures(name)
                )
                suffix = Path(name).suffix.lower()
                if suffix in {".bundle", ".zip"} and name != FINAL_BRANCH_BUNDLE_ENTRY:
                    share_safety_failures.append(
                        f"nested package/bundle not allowed by default: {name}"
                    )
            if json_path is not None and json_path.name in name_set:
                failures.append("external_sidecar_in_zip")
            failures.extend(validate_handoff_package_layout(names))
            declared_total_size = sum(member.file_size for member in bounded_members)
            total_size_exceeded = declared_total_size > ZIP_TOTAL_SIZE_CAP_BYTES
            if total_size_exceeded:
                failures.append("zip_total_size_limit_exceeded")
            for member in bounded_members:
                name = member.filename
                if member.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
                    failures.append(f"zip_member_compression_unsupported: {name}")
                if member.file_size > ZIP_MEMBER_SIZE_CAP_BYTES:
                    failures.append(f"zip_member_size_limit_exceeded: {name}")
                compression_ratio = member.file_size / max(member.compress_size, 1)
                if compression_ratio > ZIP_COMPRESSION_RATIO_CAP:
                    failures.append(
                        f"zip_member_compression_ratio_exceeded: {name}"
                    )
            metadata_preflight_passed = not failures and not share_safety_failures
            if metadata_preflight_passed:
                for member in bounded_members:
                    name = member.filename
                    raw, read_failure = _read_zip_member_streaming(archive, member)
                    if read_failure is not None:
                        failures.append(read_failure)
                        continue
                    member_bytes[name] = raw
                    suffix = Path(name).suffix.lower()
                    if name == FINAL_BRANCH_BUNDLE_ENTRY:
                        raw_bundle = raw
                    if suffix in TEXT_LIKE_SUFFIXES:
                        share_safety_failures.extend(text_blob_failures(raw, name=name))
    except (BadZipFile, OSError) as exc:
        failures.append(f"zip_read_failed: {exc}")
    if member_bytes:
        failures.extend(verify_manifest_hashes_from_bytes(member_bytes, names))
        for entry, target in (
            (TASK_METADATA_ENTRY, "task_metadata"),
            (VALIDATION_SUMMARY_ENTRY, "validation_summary"),
        ):
            try:
                payload = parse_qrp_json_object(member_bytes[entry])
            except KeyError:
                failures.append(f"json_member_missing: {entry}")
                continue
            except ValueError:
                failures.append(f"json_member_invalid: {entry}")
                continue
            if target == "task_metadata":
                task_metadata = payload
            else:
                validation_summary = payload
        for entry in (
            PACKAGE_SMOKE_RECEIPT_ENTRY,
            HARD_SCAN_RECEIPT_ENTRY,
            CHECK_RECEIPTS_ENTRY,
        ):
            try:
                payload = parse_qrp_json_object(member_bytes[entry])
            except KeyError:
                failures.append(f"json_member_missing: {entry}")
                continue
            except ValueError:
                failures.append(f"json_member_invalid: {entry}")
                continue
            receipts[entry] = payload
    if json_path is not None:
        try:
            raw_sidecar = json_path.read_bytes()
            share_safety_failures.extend(
                f"sidecar_{failure}"
                for failure in text_blob_failures(raw_sidecar, name="sidecar.json")
            )
            sidecar = parse_qrp_json_object(raw_sidecar)
        except (OSError, ValueError) as exc:
            failures.append(f"sidecar_json_invalid: {exc}")
        else:
            sidecar_payload = sidecar
    return HandoffPackagePayload(
        zip_path=zip_path,
        json_path=json_path,
        zip_sha256=zip_digest,
        member_names=names,
        duplicate_member_names=duplicates,
        raw_bundle=raw_bundle,
        sidecar_payload=sidecar_payload,
        task_metadata=task_metadata,
        validation_summary=validation_summary,
        receipts=receipts,
        load_failures=tuple(failures),
        share_safety_failures=tuple(share_safety_failures),
    )


def verify_manifest_hashes_from_bytes(
    member_bytes: Mapping[str, bytes],
    names: Iterable[str],
) -> list[str]:
    failures: list[str] = []
    try:
        raw = member_bytes[ARTIFACT_MANIFEST_ENTRY]
        payload = parse_qrp_json_object(raw)
    except (KeyError, ValueError):
        return ["artifact_manifest_invalid"]
    failures.extend(validate_artifact_manifest_contract(payload))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return [*failures, "artifact_manifest_entries_invalid"]
    actual_names = sorted(name for name in names if name != ARTIFACT_MANIFEST_ENTRY)
    manifest_names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            failures.append(f"artifact_manifest_entry_invalid: {index}")
            continue
        path = entry.get("path")
        if not isinstance(path, str):
            failures.append(f"artifact_manifest_path_invalid: {index}")
            continue
        manifest_names.append(path)
        try:
            raw_member = member_bytes[path]
        except KeyError:
            failures.append(f"artifact_manifest_entry_missing_member: {path}")
            continue
        if entry.get("size") != len(raw_member):
            failures.append(f"artifact_manifest_size_mismatch: {path}")
        if entry.get("sha256") != hashlib.sha256(raw_member).hexdigest():
            failures.append(f"artifact_manifest_sha256_mismatch: {path}")
    if sorted(manifest_names) != actual_names:
        failures.append("artifact_manifest_member_set_mismatch")
    return failures


def verify_sidecar(
    sidecar: dict[str, object],
    zip_path: Path,
    json_path: Path,
    *,
    zip_sha256: str | None = None,
) -> list[str]:
    failures = validate_sidecar_contract(sidecar)
    if sidecar.get("zip_filename") != zip_path.name:
        failures.append("sidecar_zip_filename_mismatch")
    actual_sha256 = zip_sha256 if zip_sha256 is not None else sha256_file(zip_path)
    if sidecar.get("zip_sha256") != actual_sha256:
        failures.append("sidecar_zip_sha256_mismatch")
    if json_path.name == sidecar.get("zip_filename"):
        failures.append("sidecar_filename_conflicts_with_zip")
    return failures


def compare_replayed_handoff_payloads(
    package_payload: HandoffPackagePayload,
    replayed: Mapping[str, Mapping[str, object]],
) -> list[str]:
    failures: list[str] = []
    failures.extend(
        compare_check_receipts(
            package_payload.receipts[CHECK_RECEIPTS_ENTRY],
            replayed[CHECK_RECEIPTS_ENTRY],
        )
    )
    failures.extend(
        compare_package_smoke_receipt(
            package_payload.receipts[PACKAGE_SMOKE_RECEIPT_ENTRY],
            replayed[PACKAGE_SMOKE_RECEIPT_ENTRY],
        )
    )
    failures.extend(
        compare_hard_scan_receipt(
            package_payload.receipts[HARD_SCAN_RECEIPT_ENTRY],
            replayed[HARD_SCAN_RECEIPT_ENTRY],
        )
    )
    derived_summary = build_validation_summary_from_receipts(replayed)
    if package_payload.validation_summary != derived_summary:
        failures.append("validation_summary_replay_mismatch")
    return failures


def verify_bundle_and_replay_once(
    raw_bundle: bytes,
    task: Mapping[str, object],
    package_payload: HandoffPackagePayload,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    share_safety_failures: list[str] = []
    try:
        branch = str(task["branch"])
        final_head = str(task["final_head"])
        base = str(task["base"])
    except KeyError:
        return ["final_branch_bundle_or_task_context_missing"], []
    with tempfile.TemporaryDirectory(prefix="handoff-verify-") as temp:
        root = Path(temp)
        bundle = root / FINAL_BRANCH_BUNDLE_ENTRY
        bundle.write_bytes(raw_bundle)
        verify_repo = root / "verify-repo"
        subprocess.run(["git", "init", "-q", str(verify_repo)], check=True)
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=verify_repo,
            capture_output=True,
            text=True,
        )
        if verify.returncode != 0:
            return ["bundle_verify_failed"], []
        clone_dir = root / "clone"
        clone = subprocess.run(
            ["git", "clone", "--branch", branch, str(bundle), str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if clone.returncode != 0:
            return ["bundle_clone_failed"], []
        branch_in_clone = git_text(
            clone_dir, "rev-parse", "--abbrev-ref", "HEAD"
        ).strip()
        if branch_in_clone != branch:
            failures.append("bundle_clone_branch_mismatch")
        head = git_text(clone_dir, "rev-parse", "HEAD").strip()
        if head != final_head:
            failures.append("bundle_clone_head_mismatch")
            return failures, share_safety_failures
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", base, final_head],
            cwd=clone_dir,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            failures.append("bundle_base_not_ancestor")
        if git_text(clone_dir, "status", "--porcelain").strip():
            failures.append("bundle_clone_worktree_not_clean")
        scan_groups = source_tree_scan_groups(clone_dir)
        share_safety_failures.extend(
            scan_clone_tree_for_share_safety(
                clone_dir,
                source_tree_scan_groups_payload=scan_groups,
            )
        )
        if failures:
            return failures, share_safety_failures
        try:
            replayed = build_machine_receipts(
                clone_dir,
                final_head,
                source_tree_scan_groups_payload=scan_groups,
            )
        except ReplayEnvironmentBlocked as exc:
            failures.extend(
                f"replay_environment_blocked:{failure}" for failure in exc.failures
            )
            return failures, share_safety_failures
    failures.extend(compare_replayed_handoff_payloads(package_payload, replayed))
    return failures, share_safety_failures


def scan_clone_tree_for_share_safety(
    clone_dir: Path,
    *,
    name_prefix: str = "bundle",
    source_tree_scan_groups_payload: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    failures: list[str] = []
    files = git_text(clone_dir, "ls-files").splitlines()
    for relative in files:
        failures.extend(
            f"{name_prefix}_{failure}" for failure in member_name_failures(relative)
        )
    scan_groups = source_tree_scan_groups_payload
    if scan_groups is None:
        scan_groups = source_tree_scan_groups(clone_dir)
    for group_name, group_payload in sorted(scan_groups.items()):
        findings = list(group_payload.get("findings") or [])
        failures.extend(
            (
                f"{name_prefix}_source_tree_scan:{group_name}:"
                f"{finding.get('path')}:{finding.get('line', 0)}:"
                f"{finding.get('rule')}"
            )
            for finding in findings
            if isinstance(finding, Mapping)
        )
    return failures


def text_blob_failures(raw: bytes, *, name: str) -> list[str]:
    if len(raw) > BUNDLE_TEXT_SIZE_CAP_BYTES:
        return [f"text scan exceeded maximum size: {name}"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return [f"non_utf8_text_artifact: {name}"]
    reason = unsafe_text_reason(text, name=name)
    return [f"{reason}_text: {name}"] if reason is not None else []


def _read_zip_member_streaming(
    archive: ZipFile,
    member: ZipInfo,
) -> tuple[bytes, str | None]:
    raw = bytearray()
    try:
        with archive.open(member, "r") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                raw.extend(chunk)
                if len(raw) > ZIP_MEMBER_SIZE_CAP_BYTES:
                    return b"", f"zip_member_size_limit_exceeded: {member.filename}"
    except (
        BadZipFile,
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        ZlibError,
    ) as exc:
        return (
            b"",
            f"zip_member_read_failed: {member.filename}: {type(exc).__name__}",
        )
    if len(raw) != member.file_size:
        return b"", f"zip_member_size_mismatch: {member.filename}"
    return bytes(raw), None


def unsafe_text_reason(text: str, *, name: str = "") -> str | None:
    shared_reason = unsafe_for_share_reason(text)
    if shared_reason in {"local_only_path", "local_path"}:
        return "local_absolute_path"
    if shared_reason == "credential":
        return "credential"
    if shared_reason == "provider_source_marker":
        return "vendor_marker"
    if shared_reason == "stock_code_sample":
        return "stock_sample"
    if shared_reason == "traceback":
        return "raw_traceback"
    if shared_reason == "row_sample":
        return "row_sample"
    if shared_reason == "raw_export_filename":
        return "vendor_filename"
    if shared_reason == "source_local_key":
        return "source_local_key"
    if shared_reason == "runtime_payload":
        return "runtime_payload"
    if shared_reason == "local_run_path":
        return "local_run_path"
    if name.lower().endswith(".json"):
        try:
            payload = parse_qrp_json_object(text.encode("utf-8", errors="strict"))
        except (UnicodeEncodeError, ValueError):
            json_reason = "invalid_json"
        else:
            json_reason = evidence_json_value_unsafe_reason(payload)
        if json_reason is not None:
            return (
                "local_absolute_path" if json_reason == "absolute_path" else json_reason
            )
    return None


def member_name_failures(name: str) -> list[str]:
    failures = list(shared_package_member_name_failures(name))
    parts = Path(name.replace("\\", "/")).parts
    for part in PACKAGE_FORBIDDEN_PATH_PARTS:
        if part in parts or name.replace("\\", "/").startswith(f"{part}/"):
            failures.append(f"forbidden_path_segment: {part}")
    return failures


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
