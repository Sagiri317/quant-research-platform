from __future__ import annotations

from pathlib import Path

from tests.helpers.maintenance_script_loader import load_maintenance_script_module

REPO_ROOT = Path.cwd()
VERIFY_GOVERNANCE = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "verify_governance.py"
)
verify_governance = load_maintenance_script_module("verify_governance")
governance_contract = load_maintenance_script_module("governance_contract")


def _unsafe_assignment_text() -> str:
    key = "".join(chr(code) for code in (116, 111, 107, 101, 110))
    return f'credential = "{key}=unsafe"\n'


def _legacy_versioned_api_prefix() -> str:
    return "/api/" + "v0.1"


def test_synthesized_string_gate_allows_current_version_surface(
    tmp_path: Path,
) -> None:
    module = verify_governance
    path = tmp_path / "tests" / "sample_version_strings.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                'label = "QRP" " 0.1"',
                'api = "/api/" "0.1"',
                'version = "0." "1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_synthesized_strings(
        result,
        tmp_path,
        ["tests/sample_version_strings.py"],
    )

    assert result.errors == []


def test_version_surface_policy_keeps_legal_tokens_out_of_retired_gate(
    tmp_path: Path,
) -> None:
    module = verify_governance
    contract = governance_contract
    legal_doc = tmp_path / "legal.md"
    legal_doc.write_text(
        "\n".join(
            [
                "QRP 0.1 is the current contract version.",
                "GET /api/0.1/catalog/factors",
                'definition_version = "0.1"',
                'workflow_version = "0.1"',
                'package = "handoff-sidecar/0.1"',
                'project_version = "0.1.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = module.Result()
    module.check_text_safety(result, tmp_path, ["legal.md"])

    assert result.errors == []
    assert "/api/0.1" in contract.VERSION_SURFACE_POLICY["allowed"]
    assert (
        _legacy_versioned_api_prefix()
        in contract.VERSION_SURFACE_POLICY["forbidden"]
    )


def test_version_surface_policy_rejects_legacy_v_api_prefix(tmp_path: Path) -> None:
    module = verify_governance
    path = tmp_path / "docs" / "api.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"GET {_legacy_versioned_api_prefix()}/catalog/factors\n",
        encoding="utf-8",
    )
    result = module.Result()

    module.check_text_safety(result, tmp_path, ["docs/api.md"])

    assert any("forbidden_versioned_api_prefix" in error for error in result.errors)
