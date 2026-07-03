from __future__ import annotations

from pathlib import Path

from tests.helpers.import_graph import ImportEdge, import_edges_under
from tests.helpers.maintenance_script_loader import load_maintenance_script_module

PROJECT_SRC = Path("src/quant_research_platform")


def test_owner_layer_import_boundaries_are_enforced_from_governance_contract(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    contract = load_maintenance_script_module("governance_contract")
    hits: list[str] = []

    for rule in contract.layer_boundary_import_rules():
        root = PROJECT_SRC / rule["root"]
        for edge in import_edges_under(import_edges, root):
            if any(
                edge.module.startswith(prefix)
                for prefix in rule["forbidden_prefixes"]
            ):
                hits.append(f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}")

    assert hits == []


def test_storage_layout_does_not_own_run_output_or_evidence_roots() -> None:
    source = (PROJECT_SRC / "storage" / "layout.py").read_text(encoding="utf-8")

    assert "run_output" not in source
    assert "evidence_root" not in source
    assert "report_root" not in source
