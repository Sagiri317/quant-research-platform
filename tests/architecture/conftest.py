from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ast_trees,
    import_edges,
    module_to_path,
    path_to_layer,
    repo_python_files,
)
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

__all__ = (
    "ast_trees",
    "import_edges",
    "module_to_path",
    "path_to_layer",
    "repo_python_files",
)


@pytest.fixture(scope="session")
def live_source_tree_scan() -> dict[str, dict[str, object]]:
    contract = load_maintenance_script_module("governance_contract")
    return contract.source_tree_scan_groups(Path("."))
