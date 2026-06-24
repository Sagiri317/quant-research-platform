#!/usr/bin/env python3
"""QRP public-safe package scanner.

Scan handoff packages for private paths, blocked file types,
secret-like text, local absolute paths, and stock-code samples.
Uses runtime privacy as the single public/private text-safety source.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from governance_contract import (  # noqa: E402
    PACKAGE_FORBIDDEN_PATH_PARTS,
    is_scanner_definition_line,
    package_member_name_failures,
)
from quant_research_platform.runtime.privacy import public_text_unsafe_reason  # noqa: E402

PACKAGE_EXTENSIONS = {".zip", ".bundle"}
TEXT_EXTENSIONS = {".md", ".patch", ".txt", ".json", ".yaml", ".yml", ".toml", ".py"}
MAX_TEXT_SCAN_BYTES = 2_000_000
BLOCKED_PATH_PARTS = PACKAGE_FORBIDDEN_PATH_PARTS


@dataclass
class SafetyResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    entries_scanned: int = 0
    text_entries_scanned: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def normalize_entry(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def check_entry_path(
    result: SafetyResult,
    name: str,
    allow_nested_packages: bool,
) -> None:
    norm = normalize_entry(name)
    if not norm or norm.endswith("/"):
        return
    lower = norm.lower()
    suffix = PurePosixPath(lower).suffix
    for failure in package_member_name_failures(name):
        if failure == "absolute_zip_path":
            result.errors.append(f"absolute path entry: {name}")
        elif failure == "path_traversal_zip_path":
            result.errors.append(f"path traversal entry: {name}")
        elif failure == "windows_backslash_zip_path":
            result.errors.append(f"Windows backslash entry: {name}")
        elif failure == "forbidden_suffix":
            result.errors.append(f"blocked extension: {norm}")
        elif failure == "forbidden_path_segment":
            result.errors.append(f"blocked private/sensitive path: {norm}")
    if (
        suffix in PACKAGE_EXTENSIONS
        and not allow_nested_packages
        and norm != "final-branch.bundle"
    ):
        result.errors.append(f"nested package/bundle not allowed by default: {norm}")


def _is_scanner_source(name: str) -> bool:
    norm = normalize_entry(name)
    return norm.startswith("run_tools/") or norm.endswith(
        (
            "build_handoff_package.py",
            "handoff_safety.py",
            "self_test.py",
            "verify_handoff_package.py",
            "verify_governance.py",
        )
    )


def _skip_pattern_line(line: str, name: str) -> bool:
    if not _is_scanner_source(name):
        return False
    return is_scanner_definition_line(name, line)


def check_text(result: SafetyResult, name: str, data: bytes) -> None:
    if len(data) > MAX_TEXT_SCAN_BYTES:
        result.errors.append(f"text scan exceeded maximum size: {name}")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        result.warnings.append(f"utf-8 decode skipped: {name}")
        return
    result.text_entries_scanned += 1
    for line_number, line in enumerate(text.splitlines(), start=1):
        if _skip_pattern_line(line, name):
            continue
        shared_reason = public_text_unsafe_reason(line)
        if shared_reason == "local_path":
            result.errors.append(f"local absolute path in {name}:{line_number}")
        if shared_reason == "private_marker":
            result.errors.append(f"private marker in {name}:{line_number}")
        if shared_reason == "private_path":
            result.errors.append(f"private path in {name}:{line_number}")
        if shared_reason == "row_sample":
            result.errors.append(f"row sample in {name}:{line_number}")
        if shared_reason == "raw_export_filename":
            result.errors.append(f"raw export filename in {name}:{line_number}")
        if shared_reason == "source_local_key":
            result.errors.append(f"source-local key in {name}:{line_number}")
        if shared_reason == "private_runtime_payload":
            result.errors.append(f"private runtime payload in {name}:{line_number}")
        if shared_reason == "private_run_path":
            result.errors.append(f"private run path in {name}:{line_number}")
        if shared_reason == "credential":
            result.errors.append(f"credential text in {name}:{line_number}")
        if shared_reason == "stock_code_sample":
            result.warnings.append(
                f"possible stock-code sample in {name}:{line_number}"
            )

def scan_zip(path: Path, allow_nested_packages: bool) -> SafetyResult:
    result = SafetyResult()
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            result.entries_scanned += 1
            name = info.filename
            check_entry_path(result, name, allow_nested_packages)
            suffix = PurePosixPath(name.lower()).suffix
            if suffix in TEXT_EXTENSIONS and not name.endswith("/"):
                if info.file_size > MAX_TEXT_SCAN_BYTES:
                    result.errors.append(f"text scan exceeded maximum size: {name}")
                    continue
                check_text(result, name, zf.read(info))
    return result


def scan_dir(path: Path, allow_nested_packages: bool) -> SafetyResult:
    result = SafetyResult()
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        result.entries_scanned += 1
        rel = p.relative_to(path).as_posix()
        check_entry_path(result, rel, allow_nested_packages)
        if p.suffix.lower() in TEXT_EXTENSIONS:
            check_text(result, rel, p.read_bytes()[: MAX_TEXT_SCAN_BYTES + 1])
    return result


def scan_json(path: Path, allow_nested_packages: bool) -> SafetyResult:
    del allow_nested_packages
    result = SafetyResult(entries_scanned=1)
    check_entry_path(result, path.name, allow_nested_packages=True)
    check_text(result, path.name, path.read_bytes()[: MAX_TEXT_SCAN_BYTES + 1])
    return result


def scan_target(path: Path, allow_nested_packages: bool) -> SafetyResult:
    if path.is_dir():
        return scan_dir(path, allow_nested_packages)
    if path.suffix.lower() == ".json":
        return scan_json(path, allow_nested_packages)
    if zipfile.is_zipfile(path):
        return scan_zip(path, allow_nested_packages)
    raise ValueError("target_is_neither_directory_zip_nor_json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan a QRP handoff package for public-safe rules."
    )
    parser.add_argument("target", help="Zip file or directory to scan.")
    parser.add_argument(
        "--allow-nested-packages",
        action="store_true",
        help="Allow nested .zip/.bundle files; default is fail closed.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args(argv)

    target = Path(args.target)
    if not target.exists():
        print(f"ERROR: target not found: {target}")
        return 2

    try:
        result = scan_target(target, args.allow_nested_packages)
    except zipfile.BadZipFile as exc:
        print(f"ERROR: invalid zip file: {exc}")
        return 2
    except ValueError:
        print(
            "ERROR: target is neither a directory, zip file, nor JSON file: "
            f"{target}"
        )
        return 2

    summary = {
        "status": "PASS" if result.ok else "FAIL",
        "target": target.name,
        "entries_scanned": result.entries_scanned,
        "text_entries_scanned": result.text_entries_scanned,
        "errors": result.errors,
        "warnings": result.warnings,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"package_safety_status={summary['status']}")
        print(f"entries_scanned={result.entries_scanned}")
        print(f"text_entries_scanned={result.text_entries_scanned}")
        print(f"errors={len(result.errors)}")
        print(f"warnings={len(result.warnings)}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
