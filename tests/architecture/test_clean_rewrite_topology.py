from __future__ import annotations

from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")


def test_clean_rewrite_topology_has_required_packages() -> None:
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
    }

    existing = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert existing == required


def test_retired_top_level_source_packages_are_absent() -> None:
    retired = {
        "artifacts",
        "datasets",
        "pipelines",
        "kernels",
        "core",
        "data",
        "data_layers",
        "data_artifacts",
        "research_datasets",
    }

    existing = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert existing.isdisjoint(retired)


def test_clean_rewrite_target_files_exist() -> None:
    required_files = {
        "contracts/ids.py",
        "contracts/time.py",
        "contracts/columns.py",
        "contracts/data.py",
        "contracts/factors.py",
        "contracts/labels.py",
        "contracts/masks.py",
        "contracts/universes.py",
        "contracts/datasets.py",
        "contracts/evaluations.py",
        "contracts/models.py",
        "contracts/predictions.py",
        "contracts/portfolios.py",
        "contracts/workflows.py",
        "contracts/validation.py",
        "storage/layout.py",
        "storage/parquet.py",
        "storage/manifests.py",
        "storage/partitions.py",
        "storage/checksums.py",
        "storage/privacy.py",
        "catalog/processed.py",
        "catalog/factors.py",
        "catalog/labels.py",
        "catalog/masks.py",
        "catalog/universes.py",
        "catalog/datasets.py",
        "catalog/evaluations.py",
        "catalog/models.py",
        "catalog/predictions.py",
        "catalog/workflows.py",
        "markets/base.py",
        "markets/calendars.py",
        "workflows/specs.py",
        "workflows/planner.py",
        "workflows/executor.py",
        "workflows/checkpoints.py",
        "agents/factor_factory.py",
        "agents/label_factory.py",
        "agents/dataset_planner.py",
        "agents/workflow_planner.py",
        "agents/proposals.py",
        "agents/safety.py",
        "reports/bundles.py",
        "reports/manifests.py",
        "reports/summaries.py",
        "runtime/privacy.py",
        "runtime/sanitization.py",
        "runtime/logging.py",
        "runtime/profiling.py",
        "runtime/errors.py",
    }

    missing = [
        path for path in sorted(required_files) if not (SOURCE_ROOT / path).is_file()
    ]

    assert missing == []


def test_removed_root_directories_are_absent() -> None:
    retired_roots = {"config", "scripts", "web"}
    existing = {path.name for path in Path(".").iterdir() if path.is_dir()}

    assert existing.isdisjoint(retired_roots)
