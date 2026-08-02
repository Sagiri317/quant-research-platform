from __future__ import annotations

from pathlib import Path

from tests.helpers.maintenance_script_loader import load_maintenance_script_module

verify_governance = load_maintenance_script_module("verify_governance")
governance_contract = load_maintenance_script_module("governance_contract")


def test_active_doc_scan_collects_current_non_empty_text_set() -> None:
    paths = verify_governance.active_doc_files(Path("."))

    assert paths
    assert "AGENTS.md" in paths
    assert (
        ".agents/skills/quant-research-platform-maintenance/SKILL.md" in paths
    )
    assert "docs/architecture/engineering-contract.md" in paths
    assert "docs/code-map/platform-flow.md" in paths
    assert all((Path(".") / path).is_file() for path in paths)


def test_governance_only_boundary_rejects_business_and_local_outputs() -> None:
    result = verify_governance.Result()

    verify_governance.check_governance_only(
        result,
        Path("."),
        [
            "docs/architecture/engineering-contract.md",
            "src/quant_research_platform/contracts/data.py",
            "data/research/private.parquet",
        ],
    )

    assert result.errors == [
        "path is outside governance-only allow list: "
        "src/quant_research_platform/contracts/data.py",
        "local-only root changed or included: data/research/private.parquet",
        "path is outside governance-only allow list: data/research/private.parquet",
        "blocked extension in changed path: data/research/private.parquet",
    ]


def test_governance_text_check_reuses_sensitive_scan_owner(tmp_path: Path) -> None:
    entry = next(
        item
        for item in governance_contract.SOURCE_TREE_ALLOWED_OCCURRENCES
        if item.rule_id.endswith(":sensitive_provider_marker")
    )
    relative = Path("docs") / "architecture" / "unexpected.md"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(entry.exact_line + "\n", encoding="utf-8")
    result = verify_governance.Result()

    verify_governance.check_text_safety(
        result,
        tmp_path,
        [relative.as_posix()],
    )

    assert result.checked_files == [relative.as_posix()]
    assert result.errors == [
        f"sensitive_text: {relative.as_posix()}:1:sensitive_provider_marker"
    ]


def test_current_active_docs_pass_sensitive_text_check() -> None:
    result = verify_governance.Result()
    paths = verify_governance.active_doc_files(Path("."))

    verify_governance.check_text_safety(result, Path(".").resolve(), paths)
    verify_governance.check_checked_files_non_empty(
        result,
        scan_scope="active-docs",
    )

    assert result.errors == []
