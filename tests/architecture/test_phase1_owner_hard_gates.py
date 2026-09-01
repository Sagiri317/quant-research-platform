from __future__ import annotations

from pathlib import Path

SOURCE_ROOT = Path("src/quant_research_platform")


def test_phase1_retired_surfaces_are_absent() -> None:
    retired = (
        SOURCE_ROOT / "workflows" / "local_verification.py",
        SOURCE_ROOT / "workflows" / "unified_research.py",
        SOURCE_ROOT
        / "markets"
        / "implementations"
        / "a_share"
        / "research_scenarios.py",
        SOURCE_ROOT
        / "markets"
        / "implementations"
        / "a_share"
        / "research_datasets.py",
        SOURCE_ROOT
        / "markets"
        / "implementations"
        / "crypto"
        / "research_scenarios.py",
        SOURCE_ROOT / "markets" / "implementations" / "crypto" / "research_datasets.py",
        SOURCE_ROOT / "research" / "portfolios" / "a_share.py",
    )
    assert all(not path.exists() for path in retired)


def test_phase1_research_and_profile_owners_have_no_provider_traversal() -> None:
    research_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (SOURCE_ROOT / "research").rglob("*.py")
    )
    assert "markets.implementations" not in research_sources
    assert "quant_research_platform.sources" not in research_sources

    profile = (SOURCE_ROOT / "profiles" / "research.py").read_text(encoding="utf-8")
    assert "write_dataset" not in profile
    assert "pandas" not in profile
    assert "memory_budget_bytes" not in profile


def test_phase1_has_one_common_processed_traversal_and_common_math_owner() -> None:
    processed_materializers = tuple(
        path
        for path in (SOURCE_ROOT / "markets" / "implementations").rglob(
            "*materializ*.py"
        )
    )
    assert processed_materializers == ()
    assert (SOURCE_ROOT / "research" / "factors" / "fundamental").is_dir()
    assert (SOURCE_ROOT / "research" / "factors" / "expectations").is_dir()
    assert (SOURCE_ROOT / "research" / "labels" / "engine.py").is_file()
    assert (SOURCE_ROOT / "research" / "portfolios" / "optimizer.py").is_file()


def test_phase1_production_source_has_no_retired_contract_markers() -> None:
    forbidden = (
        "purge_gap_steps",
        "standardized_ridge_regression",
        "execution_lag_steps",
        "return_source_column",
        "rebalance_frequency",
        "financial_disclosures",
        "DEFAULT_UNIVERSE",
        "AS_OF_MASK",
        "EVALUATION_MASK",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in SOURCE_ROOT.rglob("*.py")
    )
    assert all(marker not in source for marker in forbidden)
