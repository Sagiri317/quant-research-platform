"""Self-test for tracked QRP handoff tooling structure."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QRP handoff tool self-tests.")
    parser.parse_args()
    required = {
        "handoff_contract.py",
        "receipt_contract.py",
        "build_handoff_package.py",
        "collect_handoff_receipts.py",
        "verify_handoff_package.py",
        "run_package_smoke.py",
        "self_test.py",
        "README.md",
    }
    missing = sorted(name for name in required if not (SCRIPT_DIR / name).exists())
    if missing:
        return fail("missing=" + ",".join(missing))
    retired_collector = "collect_" + "private_" + "smoke_" + "status.py"
    absent = {
        SCRIPT_DIR / retired_collector,
        SKILL_ROOT / "schemas" / "handoff" / "sidecar.schema.json",
        SKILL_ROOT / "schemas" / "handoff" / "task_metadata.schema.json",
    }
    present = sorted(path.as_posix() for path in absent if path.exists())
    if present:
        return fail("retired_files_present=" + ",".join(present))
    for path in sorted(SCRIPT_DIR.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    execution_paths = [
        SCRIPT_DIR / "build_handoff_package.py",
        SCRIPT_DIR / "collect_handoff_receipts.py",
        SCRIPT_DIR / "verify_handoff_package.py",
        SCRIPT_DIR / "self_test.py",
    ]
    forbidden_symbols = tuple(
        "".join(parts)
        for parts in (
            ("GENERATION", "_EVIDENCE"),
            ("BUNDLE", "_RECONSTRUCTION", "_EVIDENCE"),
            ("PRIVATE", "_SMOKE", "_STATUS", "_EVIDENCE"),
            ("PACKAGE", "_SMOKE", "_EVIDENCE"),
            ("HARD", "_SCAN", "_CLASSIFICATION", "_EVIDENCE"),
            ("TEST", "_CHECKS", "_EVIDENCE"),
            ("BUILDER", "_OWNED", "_HANDOFF", "_EVIDENCE"),
            ("verify", "_private", "_smoke", "_status"),
            ("verify", "_test", "_checks", "_evidence"),
            ("verify", "_generation", "_evidence"),
            ("verify", "_bundle", "_reconstruction", "_evidence"),
            ("verify", "_git", "_status"),
        )
    )
    offenders: list[str] = []
    for path in execution_paths:
        source = path.read_text(encoding="utf-8-sig")
        offenders.extend(
            f"{path.name}:{symbol}" for symbol in forbidden_symbols if symbol in source
        )
    if offenders:
        return fail("retired_execution_symbols=" + ",".join(offenders))
    readme = (SCRIPT_DIR / "README.md").read_text(encoding="utf-8-sig")
    if "handoff_contract.py" not in readme or "receipt_contract.py" not in readme:
        return fail("readme_missing_contract_owner")
    print("self_test_status=PASS")
    return 0


def fail(message: str) -> int:
    print("self_test_status=FAIL")
    print(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
