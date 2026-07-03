from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.performance


def test_governance_tooling_lightweight_contracts_are_available() -> None:
    scripts_root = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
    )

    assert (scripts_root / "structure_audit.py").is_file()
    assert (scripts_root / "verify_governance.py").is_file()
    assert (scripts_root / "verify_code_map_flow.py").is_file()
    assert (scripts_root / "governance_contract.py").is_file()
