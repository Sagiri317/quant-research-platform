from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

TEXT_LIKE_SUFFIXES = {
    ".log",
    ".txt",
    ".md",
    ".json",
    ".patch",
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".js",
    ".xml",
}
FORBIDDEN_DATA_SUFFIXES = {
    ".parquet",
    ".csv",
    ".pickle",
    ".pkl",
    ".feather",
    ".arrow",
}
AMBIGUOUS_COMPLETION_WORDS = ("mostly", "partial", "partially")


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(
            "usage: check_handoff_artifacts.py <review_root> <run_handoff_zip> "
            "<run_handoff_manifest_json> <reviews_handoff_zip> "
            "<reviews_handoff_manifest_json>",
            file=sys.stderr,
        )
        return 2

    review_root = Path(argv[1])
    run_zip = Path(argv[2])
    run_manifest = Path(argv[3])
    reviews_zip = Path(argv[4])
    reviews_manifest = Path(argv[5])

    failures: list[str] = []
    failures.extend(_existing_files(review_root, run_zip, run_manifest))
    failures.extend(_existing_files(review_root, reviews_zip, reviews_manifest))
    if failures:
        _emit(failures)
        return 1

    run_zip_sha = _sha256(run_zip)
    reviews_zip_sha = _sha256(reviews_zip)
    if run_zip.read_bytes() != reviews_zip.read_bytes():
        failures.append("zip_copies_not_byte_identical")
    if run_zip_sha != reviews_zip_sha:
        failures.append("zip_sha256_mismatch")

    run_manifest_sha = _sha256(run_manifest)
    reviews_manifest_sha = _sha256(reviews_manifest)
    if run_manifest.read_bytes() != reviews_manifest.read_bytes():
        failures.append("manifest_copies_not_byte_identical")
    if run_manifest_sha != reviews_manifest_sha:
        failures.append("manifest_sha256_mismatch")

    manifest = _read_json(run_manifest, failures)
    if manifest:
        failures.extend(
            _validate_external_exports(
                manifest=manifest,
                run_zip=run_zip,
                run_zip_sha=run_zip_sha,
                run_manifest=run_manifest,
                reviews_zip=reviews_zip,
                reviews_zip_sha=reviews_zip_sha,
                reviews_manifest=reviews_manifest,
            )
        )

    zip_result = _validate_zip(run_zip, run_manifest, reviews_manifest)
    failures.extend(zip_result.failures)
    failures.extend(
        _validate_artifact_safety_report(
            review_root,
            run_manifest_sha=run_manifest_sha,
            reviews_manifest_sha=reviews_manifest_sha,
        )
    )
    failures.extend(_validate_prompt_checklist(review_root))

    _emit(
        failures,
        text_like_count=zip_result.text_like_count,
        run_manifest_sha=run_manifest_sha,
        reviews_manifest_sha=reviews_manifest_sha,
    )
    return 1 if failures else 0


class ZipValidationResult:
    def __init__(self, *, failures: list[str], text_like_count: int) -> None:
        self.failures = failures
        self.text_like_count = text_like_count


def _existing_files(review_root: Path, *paths: Path) -> list[str]:
    failures = []
    if not review_root.is_dir():
        failures.append(f"review_root_missing: {review_root.as_posix()}")
    for path in paths:
        if not path.is_file():
            failures.append(f"handoff_artifact_missing: {path.as_posix()}")
    return failures


def _validate_external_exports(
    *,
    manifest: dict[str, object],
    run_zip: Path,
    run_zip_sha: str,
    run_manifest: Path,
    reviews_zip: Path,
    reviews_zip_sha: str,
    reviews_manifest: Path,
) -> list[str]:
    failures: list[str] = []
    exports = manifest.get("external_review_exports")
    if not isinstance(exports, dict):
        return ["external_review_exports_missing"]

    expected = {
        "run_handoff_zip": (run_zip, run_zip_sha, run_zip.stat().st_size, True),
        "reviews_handoff_zip": (
            reviews_zip,
            reviews_zip_sha,
            reviews_zip.stat().st_size,
            True,
        ),
        "run_handoff_manifest": (
            run_manifest,
            None,
            run_manifest.stat().st_size,
            False,
        ),
        "reviews_handoff_manifest": (
            reviews_manifest,
            None,
            reviews_manifest.stat().st_size,
            False,
        ),
    }
    for key, (path, sha, size, require_sha) in expected.items():
        item = exports.get(key)
        if not isinstance(item, dict):
            failures.append(f"external_export_missing: {key}")
            continue
        if item.get("path") != path.as_posix():
            failures.append(f"external_export_path_mismatch: {key}")
        if item.get("size_bytes") != size:
            failures.append(f"external_export_size_mismatch: {key}")
        if require_sha and item.get("sha256") != sha:
            failures.append(f"external_export_sha_mismatch: {key}")
        if not require_sha and "sha256" in item:
            failures.append(f"manifest_self_hash_forbidden: {key}")

    if exports.get("handoff_manifest_copies_byte_identical") is not True:
        failures.append("manifest_identity_flag_missing")
    return failures


def _validate_zip(
    zip_path: Path,
    run_manifest: Path,
    reviews_manifest: Path,
) -> ZipValidationResult:
    failures: list[str] = []
    text_like_count = 0
    external_manifest_names = {
        "handoff-manifest.json",
        run_manifest.name,
        reviews_manifest.name,
    }
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            parts = Path(name).parts
            if name in external_manifest_names:
                failures.append(f"external_manifest_inside_zip: {name}")
            if "\\" in name:
                failures.append(f"windows_backslash_zip_path: {name}")
            if suffix in FORBIDDEN_DATA_SUFFIXES:
                failures.append(f"forbidden_data_extension: {name}")
            if any(part.lower() == "raw" for part in parts):
                failures.append(f"raw_vendor_data_entry: {name}")
            if suffix in TEXT_LIKE_SUFFIXES:
                text_like_count += 1
                try:
                    archive.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    failures.append(f"non_utf8_text_like_entry: {name}")
    return ZipValidationResult(failures=failures, text_like_count=text_like_count)


def _validate_artifact_safety_report(
    review_root: Path,
    *,
    run_manifest_sha: str,
    reviews_manifest_sha: str,
) -> list[str]:
    report = review_root / "artifact_safety_report.md"
    if not report.is_file():
        return ["artifact_safety_report_missing"]
    text = report.read_text(encoding="utf-8")
    failures = []
    for label, value in (
        ("run_handoff_manifest_sha256", run_manifest_sha),
        ("reviews_handoff_manifest_sha256", reviews_manifest_sha),
    ):
        if label not in text or value not in text:
            failures.append(f"artifact_safety_manifest_sha_missing: {label}")
    return failures


def _validate_prompt_checklist(review_root: Path) -> list[str]:
    checklist = review_root / "prompt_completion_checklist.md"
    if not checklist.is_file():
        return ["prompt_completion_checklist_missing"]
    text = checklist.read_text(encoding="utf-8").lower()
    return [
        f"ambiguous_completion_word: {word}"
        for word in AMBIGUOUS_COMPLETION_WORDS
        if word in text
    ]


def _read_json(path: Path, failures: list[str]) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"manifest_json_invalid: {exc}")
        return {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit(
    failures: list[str],
    *,
    text_like_count: int | None = None,
    run_manifest_sha: str | None = None,
    reviews_manifest_sha: str | None = None,
) -> None:
    if failures:
        print("handoff_artifacts_check=FAIL")
        for failure in failures:
            print(failure)
        return
    print("handoff_artifacts_check=PASS")
    if text_like_count is not None:
        print(f"text_like_utf8_checked_count={text_like_count}")
    if run_manifest_sha is not None:
        print(f"run_handoff_manifest_sha256={run_manifest_sha}")
    if reviews_manifest_sha is not None:
        print(f"reviews_handoff_manifest_sha256={reviews_manifest_sha}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
