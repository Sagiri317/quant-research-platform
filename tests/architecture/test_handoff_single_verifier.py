from __future__ import annotations

import ast
from pathlib import Path

HANDOFF_ROOT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/handoff"
)
SCRIPT_ROOT = Path(".agents/skills/quant-research-platform-maintenance/scripts")


def test_handoff_has_single_verifier_and_no_separate_safety_scripts() -> None:
    assert not (HANDOFF_ROOT / "handoff_safety.py").exists()
    assert not (SCRIPT_ROOT / "package_safety_check.py").exists()
    assert (HANDOFF_ROOT / "verify_handoff_package.py").is_file()


def test_handoff_builder_delegates_safety_to_verifier() -> None:
    builder = HANDOFF_ROOT / "build_handoff_package.py"
    source = builder.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(builder))
    forbidden_names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            token in node.name.lower() for token in ("sanitize", "forbidden", "private")
        ):
            forbidden_names.append(node.name)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and any(
                    token in target.id.lower()
                    for token in ("sanitize", "forbidden", "private")
                ):
                    forbidden_names.append(target.id)

    assert forbidden_names == []
    assert "verify_handoff_package" in source
