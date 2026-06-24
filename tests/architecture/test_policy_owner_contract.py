from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")


def test_policy_does_not_import_business_layers() -> None:
    policy_root = PROJECT_SRC / "policy"
    hits = [
        f"{path.as_posix()}:{lineno}:{module}"
        for path, lineno, module in _iter_imports(policy_root)
        if module.startswith("quant_research_platform.")
        and not module.startswith("quant_research_platform.policy.")
    ]

    assert hits == []


def test_contracts_do_not_own_share_safety_or_identifier_regexes() -> None:
    hits: list[str] = []
    blocked_terms = (
        "private_path",
        "provider",
        "source marker",
        "stock_code",
        "stock-code",
        "share_safety",
    )
    for path in sorted((PROJECT_SRC / "contracts").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        for term in blocked_terms:
            if term in lowered:
                hits.append(f"{path.as_posix()}:{term}")

    assert hits == []


def test_status_aggregation_rejects_top_level_pass_with_not_run_child() -> None:
    from quant_research_platform.policy.check_status import aggregate_check_status

    assert aggregate_check_status(["PASS", "NOT_RUN"]) == "FAIL"


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    if not root.exists():
        return ((root, 0, "missing-policy-package"),)
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
