from __future__ import annotations

import quant_research_platform.profiles.research as research_profiles
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    materialization_dependencies,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.storage.layout import StorageLayout


def test_crypto_binance_1m_profile_builds_with_empty_prediction_and_portfolio() -> None:
    profile = research_profile("crypto_binance_1m")

    assert profile.profile_id == "crypto_binance_1m"
    assert profile.source_id == "binance"
    assert [item.source_id for item in profile.sources.list()] == ["binance"]
    assert [item.market_id for item in profile.markets.list()] == ["crypto"]
    assert profile.source_interim_catalogs.keys() == {"binance"}
    assert profile.research_datasets.list()[0].data_id == (
        "microstructure_forward_returns"
    )
    assert profile.evaluations.list()[0].run_id == (
        "microstructure_forward_returns_evaluation"
    )
    assert isinstance(profile.models, ModelCatalog)
    assert isinstance(profile.predictions, PredictionCatalog)
    assert isinstance(profile.portfolios, PortfolioCatalog)
    assert profile.models.list() == ()
    assert profile.predictions.list() == ()
    assert profile.portfolios.list() == ()


def test_crypto_binance_profile_imports_processed_dependency_window(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_import(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        research_profiles,
        "_materialize_binance_interim_data",
        capture_import,
    )
    core = TimeWindow("2026-07-06", "2026-07-06")
    dependency = materialization_dependencies.panel_dependency_window(core)

    research_profiles._materialize_crypto_binance_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        raw_root=tmp_path / "raw",
        start_date="2026-07-06",
        end_date="2026-07-06",
    )

    assert captured["start_date"] == dependency.start.isoformat(timespec="seconds")
    assert captured["end_date"] == dependency.end.isoformat(timespec="seconds")
