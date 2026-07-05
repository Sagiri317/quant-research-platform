from __future__ import annotations

from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.profiles.research import research_profile


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
