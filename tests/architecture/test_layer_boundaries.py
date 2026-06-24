from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.governance_fixtures import load_governance_contract

PROJECT_SRC = Path("src/quant_research_platform")


def test_owner_layer_import_boundaries_are_enforced_from_governance_contract() -> None:
    contract = load_governance_contract()
    hits: list[str] = []

    for rule in contract.layer_boundary_import_rules():
        root = PROJECT_SRC / rule["root"]
        for path, lineno, module in _iter_imports(root):
            if any(module.startswith(prefix) for prefix in rule["forbidden_prefixes"]):
                hits.append(f"{path.as_posix()}:{lineno}:{module}")

    assert hits == []


def test_storage_layout_does_not_own_run_output_or_evidence_roots() -> None:
    source = (PROJECT_SRC / "storage" / "layout.py").read_text(encoding="utf-8")

    assert "run_output" not in source
    assert "evidence_root" not in source
    assert "report_root" not in source


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    paths = (root,) if root.is_file() else tuple(sorted(root.rglob("*.py")))
    imports: list[tuple[Path, int, str]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)
