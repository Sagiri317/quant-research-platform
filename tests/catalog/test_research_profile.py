from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.markets import MarketRegistry
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.outputs import OutputCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios,
)
from tests.helpers import frequency_ids


def test_research_profile_is_only_public_profile_builder() -> None:
    import quant_research_platform.profiles.research as profiles

    assert set(profiles.__all__) == {
        "ResearchProfile",
        "research_profile",
        "research_profile_id",
    }


def test_research_profile_contains_active_catalogs() -> None:
    from quant_research_platform.profiles.research import (
        ResearchProfile,
        research_profile,
        research_profile_id,
    )

    profile = research_profile()

    assert isinstance(profile, ResearchProfile)
    assert profile.profile_id == "a_share_csmar_1d"
    assert (
        research_profile_id("a_share", "csmar", frequency_ids.DAILY_FREQUENCY_SLUG)
        == profile.profile_id
    )
    assert research_profile(profile.profile_id) == profile
    assert isinstance(profile.fields, FieldCatalog)
    assert isinstance(profile.factors, FactorCatalog)
    assert isinstance(profile.labels, LabelCatalog)
    assert isinstance(profile.processed, ProcessedDataCatalog)
    assert isinstance(profile.research_datasets, ResearchDatasetCatalog)
    assert isinstance(profile.outputs, OutputCatalog)
    assert isinstance(profile.evaluations, EvaluationCatalog)
    assert isinstance(profile.models, ModelCatalog)
    assert isinstance(profile.predictions, PredictionCatalog)
    assert isinstance(profile.portfolios, PortfolioCatalog)
    assert isinstance(profile.sources, SourceRegistry)
    assert isinstance(profile.markets, MarketRegistry)
    assert set(profile.source_interim_catalogs) == {"csmar"}
    assert profile.source_id == SourceId("csmar")
    assert callable(profile.materialize_interim_data)
    assert callable(profile.materialize_processed_data)


def test_research_profile_covers_required_models_predictions_and_signals() -> None:
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.research.models.algorithm_facts import (
        model_algorithm_ids,
    )

    profile = research_profile()
    algorithms = tuple(model_algorithm_ids())
    dataset = profile.research_datasets.get("price_volume_forward_returns")
    evaluation = profile.evaluations.get("signal_forward_return_20d")
    portfolio = research_scenarios.build_portfolio_catalog(
        execution_lag_steps=1,
    ).get("signal_forward_return_20d_portfolio")
    models = tuple(profile.models.list())
    predictions = tuple(
        research_scenarios.build_prediction_catalog(
            train_end="2014-06-30",
            test_start="2014-07-01",
        ).list()
    )
    model_ids = {model.model_id for model in models}

    assert profile.predictions.list() == ()
    assert profile.portfolios.list() == ()
    assert sorted(model.model_algorithm for model in models) == sorted(algorithms)
    assert len(predictions) == len(algorithms)
    assert {prediction.model_id for prediction in predictions} == model_ids
    assert all(
        prediction.research_dataset_id == dataset.data_id for prediction in predictions
    )
    assert dataset.factor_columns
    assert dataset.feature_columns == ("mom_20d_zscore",)
    assert [
        (
            transform.source_columns,
            transform.output_columns,
            transform.transform,
        )
        for transform in dataset.feature_transforms
    ] == [(("mom_20d",), ("mom_20d_zscore",), "cross_sectional_zscore")]
    assert [
        (signal.signal_id, signal.source_column, signal.source_kind)
        for signal in evaluation.signals
    ] == [
        ("mom_20d", "mom_20d", "factor"),
        ("rev_1d", "rev_1d", "factor"),
        ("mom_20d_zscore", "mom_20d_zscore", "feature"),
    ]
    assert [
        (signal.signal_id, signal.source_column, signal.source_kind)
        for signal in portfolio.signals
    ] == [
        ("mom_20d", "mom_20d", "factor"),
        ("rev_1d", "rev_1d", "factor"),
        ("mom_20d_zscore", "mom_20d_zscore", "feature"),
    ]


def test_catalog_profile_source_interim_catalog_requires_single_source(
    monkeypatch,
) -> None:
    import quant_research_platform.profiles.research as profiles

    def fake_catalog() -> object:
        return SimpleNamespace(
            list=lambda: [
                SimpleNamespace(source_id="synthetic_a"),
                SimpleNamespace(source_id="synthetic_b"),
            ]
        )

    monkeypatch.setattr(profiles, "csmar_interim_catalog", fake_catalog)

    with pytest.raises(ValueError, match="unexpected_source_interim_catalog_sources"):
        profiles.research_profile()
