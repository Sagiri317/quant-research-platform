from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

STRUCTURE_AUDIT = Path(
    ".agents/skills/quant-research-platform-maintenance/scripts/structure_audit.py"
)


def test_structure_audit_script_exists_and_is_internal_only() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert STRUCTURE_AUDIT.is_file()
    assert "structure_audit" not in pyproject


def test_structure_audit_json_gate_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(STRUCTURE_AUDIT),
            "--repo-root",
            ".",
            "--json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_id"] == "qrp-structure-audit"
    assert payload["status"] == "PASS"
    assert payload["checks"]


def test_negative_handoff_evidence_fields_are_semantic_specs() -> None:
    contract = _load_handoff_contract()
    specs = contract.ACTIVE_DENIED_HANDOFF_EVIDENCE_FIELDS

    assert specs
    assert all(
        {
            "field_name",
            "owner",
            "risk",
            "lifecycle",
            "acceptance_gate",
        }
        <= set(spec)
        for spec in specs
    )


def _load_handoff_contract():
    path = Path(
        ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
        "handoff_contract.py"
    )
    spec = importlib.util.spec_from_file_location("handoff_contract", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
