from __future__ import annotations

import ast
from pathlib import Path


def test_identity_and_common_envelope_each_have_one_production_owner() -> None:
    source_root = Path("src/quant_research_platform")
    software_owner = source_root / "storage" / "software_identity.py"
    envelope_owner = source_root / "contracts" / "business_manifest.py"

    owners = {
        "production_software_identity": [software_owner],
        "build_business_manifest_envelope": [envelope_owner],
        "validate_business_manifest_envelope": [envelope_owner],
    }
    definitions = {name: [] for name in owners}
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in definitions.keys() & names:
            definitions[name].append(path)
    assert definitions == owners
