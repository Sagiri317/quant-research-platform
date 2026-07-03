from __future__ import annotations

import re
from pathlib import Path

from tests.helpers import maintenance_script_loader


def test_maintenance_script_loader_exposes_canonical_entrypoints_only() -> None:
    exported_names = {
        name
        for name in dir(maintenance_script_loader)
        if name.startswith("load_") and name.endswith("_module")
    }

    assert exported_names == {
        "load_handoff_script_module",
        "load_maintenance_script_module",
    }


def test_tests_use_canonical_maintenance_loader_names() -> None:
    forbidden_patterns = (
        re.compile("def _load_" + "governance_module"),
        re.compile("def _load_" + "contract_module"),
        re.compile("def _load_" + "governance_contract"),
        re.compile("def _load_" + "handoff_module"),
        re.compile("load_.* as _" + "load_"),
    )
    findings: list[str] = []
    for path in sorted(Path("tests").rglob("*.py")):
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern.search(source):
                findings.append(f"{path.as_posix()}:{pattern.pattern}")

    assert findings == []
