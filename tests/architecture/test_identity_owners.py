from __future__ import annotations

import ast
from pathlib import Path


def test_identity_and_common_envelope_each_have_one_production_owner() -> None:
    source_root = Path("src/quant_research_platform")
    software_owner = source_root / "storage" / "software_identity.py"
    envelope_owner = source_root / "contracts" / "business_manifest.py"

    software_factories = _definitions(source_root, "production_software_identity")
    envelope_factories = _definitions(source_root, "build_business_manifest_envelope")
    envelope_validators = _definitions(
        source_root,
        "validate_business_manifest_envelope",
    )

    assert software_factories == [software_owner]
    assert envelope_factories == [envelope_owner]
    assert envelope_validators == [envelope_owner]


def _definitions(source_root: Path, name: str) -> list[Path]:
    result: list[Path] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
            for node in ast.walk(tree)
        ):
            result.append(path)
    return result
