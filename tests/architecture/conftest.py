from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ast_trees,
    import_edges,
    module_to_path,
    path_to_layer,
    repo_python_files,
)

__all__ = (
    "ast_trees",
    "import_edges",
    "module_to_path",
    "path_to_layer",
    "repo_python_files",
)


@pytest.fixture(scope="session")
def live_source_tree_scan() -> dict[str, dict[str, object]]:
    contract = _load_governance_contract()
    return contract.source_tree_scan_groups(Path("."))


def _load_governance_contract():
    script = (
        Path(".agents")
        / "skills"
        / "quant-research-platform-maintenance"
        / "scripts"
        / "governance_contract.py"
    )
    spec = importlib.util.spec_from_file_location("governance_contract", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
