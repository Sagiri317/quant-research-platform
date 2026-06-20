from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
VERIFY_GOVERNANCE = (
    REPO_ROOT
    / ".agents"
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
    / "verify_governance.py"
)


def _load_governance_module():
    spec = importlib.util.spec_from_file_location(
        "verify_governance",
        VERIFY_GOVERNANCE,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_governance_allows_sanitization_negative_fixtures_only() -> None:
    module = _load_governance_module()

    assert module._is_allowed_local_path_fixture(
        "tests/architecture/test_public_safe_evidence.py",
        '    "/mnt/",',
    )
    assert module._is_allowed_secret_fixture(
        "tests/interfaces/test_api_cli_01_contract.py",
        '                "token": "token=abcdef123456",',
    )
    assert not module._is_allowed_local_path_fixture(
        "src/quant_research_platform/example.py",
        'path = "C' + r':\Users\DELL\data\raw\csmar"',
    )
    assert not module._is_allowed_secret_fixture(
        "src/quant_research_platform/example.py",
        'token = "abcdef123456"',
    )
