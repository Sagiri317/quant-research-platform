from __future__ import annotations

from pathlib import Path
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
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import (
    research_scenarios,
)
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids


def test_research_profile_is_only_public_profile_builder() -> None:
    import quant_research_platform.profiles.research as profiles

    all_names = set(profiles.__all__)
    assert all_names == {"ResearchProfile", "research_profile", "research_profile_id"}
    assert "research_profile_for_scenario" not in vars(profiles)
    assert "resolve_a_share_prediction_catalog" not in vars(profiles)
    old_portfolio_resolver = "resolve_" + "a_share_" + "portfolio_catalog"
    assert old_portfolio_resolver not in vars(profiles)
    assert "default_catalogs" not in vars(profiles)
    assert "CatalogBundle" not in vars(profiles)
    assert "local_a_share_csmar_catalogs" not in vars(profiles)
    assert not [name for name in vars(profiles) if name.startswith("default_")]


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
    from quant_research_platform.research.models.supervised import model_algorithm_ids

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


def test_catalog_profile_private_processed_helper_forwards_typed_source_id(
    monkeypatch,
) -> None:
    import quant_research_platform.profiles.research as profiles

    captured: dict[str, object] = {}

    def fake_materialize(
        *,
        layout: StorageLayout,
        source_id: SourceId,
        time_window: TimeWindow,
        kinds: frozenset[ProcessedDataKind],
    ) -> str:
        captured.update(
            {
                "layout": layout,
                "source_id": source_id,
                "time_window": time_window,
                "kinds": kinds,
            }
        )
        return "ok"

    monkeypatch.setattr(
        profiles.processed_materializer_module,
        "materialize_processed_data",
        fake_materialize,
    )

    layout = StorageLayout(project_root=Path("project"))
    time_window = TimeWindow("2014-01-01", "2014-01-31")
    kinds = frozenset({ProcessedDataKind.PANEL})

    assert (
        profiles._materialize_processed_data(
            layout,
            SourceId("csmar"),
            time_window,
            kinds,
        )
        == "ok"
    )
    assert captured["layout"] == layout
    assert captured["source_id"] == SourceId("csmar")
    assert captured["time_window"] == time_window
    assert captured["kinds"] == kinds


def test_catalog_profile_private_interim_helper_forwards_kwargs(monkeypatch) -> None:
    import quant_research_platform.profiles.research as profiles

    captured: dict[str, object] = {}

    def fake_materialize(
        *,
        layout: StorageLayout,
        raw_root: Path,
        start_date: str,
        end_date: str,
    ) -> dict[str, object]:
        captured.update(
            {
                "layout": layout,
                "raw_root": raw_root,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        return captured

    monkeypatch.setattr(profiles, "_materialize_csmar_interim_data", fake_materialize)

    layout = StorageLayout(project_root=Path("project"))
    raw_root = Path("data/raw/csmar")
    result = profiles._materialize_a_share_csmar_interim_data(
        layout,
        raw_root,
        "2014-01-01",
        "2014-01-31",
    )

    assert result == {
        "layout": layout,
        "raw_root": raw_root,
        "start_date": "2014-01-01",
        "end_date": "2014-01-31",
    }


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
        profiles._a_share_csmar_source_interim_catalogs()


def test_catalog_profile_source_id_matches_interim_catalog() -> None:
    import quant_research_platform.profiles.research as profiles

    assert profiles._active_source_id() == SourceId("csmar")
    assert isinstance(
        next(iter(profiles._a_share_csmar_source_interim_catalogs().values())),
        InterimDataCatalog,
    )
