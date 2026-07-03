from __future__ import annotations

from pathlib import Path

from tests.helpers.import_graph import import_edges_for_roots

PROJECT_SRC = Path("src/quant_research_platform")


def test_catalog_does_not_import_concrete_sources_markets_or_materializers() -> None:
    hits = [
        f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
        for edge in import_edges_for_roots((PROJECT_SRC / "catalog",))
        if edge.module.startswith(
            (
                "quant_research_platform.sources",
                "quant_research_platform.markets.implementations",
            )
        )
        or "materializer" in edge.module
    ]

    assert hits == []


def test_profiles_research_is_the_composition_root() -> None:
    profile = PROJECT_SRC / "profiles" / "research.py"
    assert profile.is_file()

    source = profile.read_text(encoding="utf-8")
    assert "ResearchProfile" in source
    assert "def research_profile(" in source
    assert "def research_profile_id(" in source
    assert "quant_research_platform.markets.implementations" in source
    assert "materializer" in source
    assert not (PROJECT_SRC / "catalog" / "profiles.py").exists()
