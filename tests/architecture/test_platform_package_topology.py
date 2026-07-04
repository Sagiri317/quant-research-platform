from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import (
    parse_source,
)

SOURCE_ROOT = Path("src/quant_research_platform")
API_APP = SOURCE_ROOT / "interfaces/api/app.py"
PUBLIC_CLI = SOURCE_ROOT / "interfaces/cli/commands.py"
PRIVATE_CLI = SOURCE_ROOT / "interfaces/private_cli.py"

def test_platform_topology_has_required_packages() -> None:
    required = {
        "contracts",
        "storage",
        "catalog",
        "sources",
        "markets",
        "research",
        "workflows",
        "compute",
        "agents",
        "interfaces",
        "reports",
        "runtime",
        "artifacts",
        "policy",
        "profiles",
    }

    existing = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert existing == required


def test_platform_target_files_exist() -> None:
    required_files = {
        "contracts/ids.py",
        "contracts/time.py",
        "contracts/columns.py",
        "contracts/data.py",
        "contracts/factors.py",
        "contracts/labels.py",
        "contracts/datasets.py",
        "contracts/evaluations.py",
        "contracts/models.py",
        "contracts/predictions.py",
        "contracts/portfolios.py",
        "contracts/outputs.py",
        "contracts/workflows.py",
        "contracts/validation.py",
        "storage/layout.py",
        "storage/datasets.py",
        "storage/manifests.py",
        "storage/checksums.py",
        "catalog/processed.py",
        "catalog/factors.py",
        "catalog/labels.py",
        "catalog/datasets.py",
        "catalog/evaluations.py",
        "catalog/models.py",
        "catalog/predictions.py",
        "catalog/workflows.py",
        "markets/base.py",
        "markets/session_windows.py",
        "workflows/planner.py",
        "workflows/executor.py",
        "workflows/checkpoints.py",
        "workflows/local_verification.py",
        "agents/factor_factory.py",
        "agents/label_factory.py",
        "agents/dataset_planner.py",
        "agents/workflow_planner.py",
        "agents/proposals.py",
        "agents/safety.py",
        "reports/execution_metrics.py",
        "reports/schema_inventory.py",
        "reports/verification_summary.py",
        "artifacts/definitions.py",
        "artifacts/evidence.py",
        "artifacts/layout.py",
        "artifacts/outputs.py",
        "policy/check_status.py",
        "policy/identifiers.py",
        "policy/share_safety.py",
        "profiles/research.py",
        "runtime/logging.py",
        "runtime/profiling.py",
        "runtime/errors.py",
        "runtime/execution_metrics.py",
        "compute/dataframe/construction.py",
        "compute/dataframe/constraints.py",
        "compute/dataframe/turnover.py",
        "compute/dataframe/exposure.py",
        "compute/dataframe/portfolio_returns.py",
        "compute/tensor/construction.py",
        "compute/tensor/constraints.py",
        "compute/tensor/turnover.py",
        "compute/tensor/exposure.py",
        "compute/tensor/portfolio_returns.py",
        "sources/csmar/ids.py",
        "sources/csmar/raw_schemas.py",
        "sources/csmar/field_mappings.py",
        "sources/csmar/checksums.py",
        "sources/csmar/interim.py",
        "markets/implementations/a_share/processed/annotations.py",
        "markets/implementations/a_share/processed/materialization_io.py",
        "markets/implementations/a_share/processed/materializer.py",
        "markets/implementations/a_share/processed/panels.py",
        "markets/implementations/a_share/factors/definitions.py",
        "markets/implementations/a_share/factors/calculations.py",
        "markets/implementations/a_share/labels/definitions.py",
        "markets/implementations/a_share/labels/calculations.py",
        "markets/implementations/a_share/verification/aggregates.py",
        "research/labels/forward_returns.py",
        "interfaces/private_cli.py",
    }

    missing = [
        path for path in sorted(required_files) if not (SOURCE_ROOT / path).is_file()
    ]

    assert missing == []


def test_removed_root_directories_are_absent() -> None:
    retired_roots = {"config", "scripts", "web"}
    existing = {path.name for path in Path(".").iterdir() if path.is_dir()}

    assert existing.isdisjoint(retired_roots)


def test_report_placeholder_and_retired_research_modules_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "reports/bundles.py",
        SOURCE_ROOT / "reports/index.py",
        SOURCE_ROOT / "reports/manifests.py",
        SOURCE_ROOT / "reports/summaries.py",
        SOURCE_ROOT / "research/models" / ("base" + "line.py"),
        SOURCE_ROOT / "research/predictions/bundle.py",
        SOURCE_ROOT / "research/predictions/handoff.py",
    )

    assert [str(path) for path in retired if path.exists()] == []


def test_a_share_factor_builder_has_no_legacy_partition_fallback() -> None:
    path = SOURCE_ROOT / "markets/implementations/a_share/factors/build_factors.py"
    assert not path.exists()


def test_reviewed_package_inits_do_not_reexport_convenience_symbols() -> None:
    hits: list[str] = []
    for path in (
        SOURCE_ROOT / "research/evaluations/__init__.py",
        SOURCE_ROOT / "compute/tensor/__init__.py",
    ):
        tree = parse_source(path)
        for node in tree.body:
            if isinstance(node, ast.ImportFrom | ast.Import):
                hits.append(f"{path}:{node.lineno}: import")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        hits.append(f"{path}:{node.lineno}: __all__")
    assert hits == []
