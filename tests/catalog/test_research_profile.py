from __future__ import annotations

import ast
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
from quant_research_platform.catalog.predictions import PredictionCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.catalog.sources import SourceRegistry
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.layout import StorageLayout

PROJECT_SRC = Path("src/quant_research_platform")


def test_research_profile_is_only_public_profile_builder() -> None:
    import quant_research_platform.catalog.profiles as profiles

    all_names = set(profiles.__all__)
    assert all_names == {"ResearchProfile", "research_profile"}
    assert "default_catalogs" not in vars(profiles)
    assert "CatalogBundle" not in vars(profiles)
    assert "local_a_share_csmar_catalogs" not in vars(profiles)
    assert not [name for name in vars(profiles) if name.startswith("default_")]


def test_research_profile_contains_active_catalogs() -> None:
    from quant_research_platform.catalog.profiles import (
        ResearchProfile,
        research_profile,
    )

    profile = research_profile()

    assert isinstance(profile, ResearchProfile)
    assert isinstance(profile.fields, FieldCatalog)
    assert isinstance(profile.factors, FactorCatalog)
    assert isinstance(profile.labels, LabelCatalog)
    assert isinstance(profile.processed, ProcessedDataCatalog)
    assert isinstance(profile.research_datasets, ResearchDatasetCatalog)
    assert isinstance(profile.reports, ReportCatalog)
    assert isinstance(profile.evaluations, EvaluationCatalog)
    assert isinstance(profile.models, ModelCatalog)
    assert isinstance(profile.predictions, PredictionCatalog)
    assert isinstance(profile.sources, SourceRegistry)
    assert isinstance(profile.markets, MarketRegistry)
    assert set(profile.source_interim_catalogs) == {"csmar"}
    assert profile.source_id == SourceId("csmar")
    assert callable(profile.materialize_interim_data)
    assert callable(profile.materialize_processed_data)


def test_research_profile_covers_required_models_predictions_and_signals() -> None:
    from quant_research_platform.catalog.profiles import research_profile
    from quant_research_platform.contracts.models import MODEL_ALGORITHMS

    profile = research_profile()
    dataset = profile.research_datasets.get("a_share_price_volume_forward_returns")
    evaluation = profile.evaluations.get("a_share_signal_forward_return_20d")
    portfolio = profile.portfolios.get(
        "a_share_signal_forward_return_20d_portfolio"
    )
    models = tuple(profile.models.list())
    predictions = tuple(profile.predictions.list())
    model_ids = {model.model_id for model in models}

    assert sorted(model.model_algorithm for model in models) == sorted(
        MODEL_ALGORITHMS
    )
    assert len(predictions) == len(MODEL_ALGORITHMS)
    assert {prediction.model_id for prediction in predictions} == model_ids
    assert all(
        prediction.research_dataset_id == dataset.data_id
        for prediction in predictions
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


def test_a_share_research_profiles_do_not_import_csmar() -> None:
    path = PROJECT_SRC / "markets/implementations/a_share/research_profiles.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    imports = _imported_modules(tree)
    assert not [
        module
        for module in imports
        if module.startswith("quant_research_platform.sources.csmar")
    ]


def test_catalog_profiles_is_only_prod_module_composing_csmar_and_a_share() -> None:
    composers: list[str] = []
    for path in (PROJECT_SRC).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _imported_modules(tree)
        has_csmar = any(
            module.startswith("quant_research_platform.sources.csmar")
            for module in imports
        )
        has_a_share = any(
            module.startswith("quant_research_platform.markets.implementations.a_share")
            for module in imports
        )
        if has_csmar and has_a_share:
            composers.append(path.as_posix())

    assert composers == ["src/quant_research_platform/catalog/profiles.py"]


def test_a_share_scenario_specs_are_not_owned_by_catalog_profile() -> None:
    profile_source = (PROJECT_SRC / "catalog/profiles.py").read_text(encoding="utf-8")
    scenario_source = (
        PROJECT_SRC / "markets/implementations/a_share/research_scenarios.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "EvaluationSpec(",
        "ModelSpec(",
        "PredictionSpec(",
        "build_a_share_evaluation_catalog",
        "build_a_share_model_catalog",
        "build_a_share_prediction_catalog",
    )

    assert [token for token in forbidden[:3] if token in profile_source] == []
    assert all(token in scenario_source for token in forbidden)


def test_reports_bundles_uses_report_local_catalog_not_profile_bundle() -> None:
    source = (PROJECT_SRC / "reports/report_artifacts.py").read_text(
        encoding="utf-8"
    )

    assert "quant_research_platform.catalog.profiles" not in source
    assert "local_a_share_csmar_catalogs" not in source
    assert "quant_research_platform.reports.catalog" in source
    assert "report_catalog(" in source


def test_catalog_profile_private_processed_helper_forwards_typed_source_id(
    monkeypatch,
) -> None:
    import quant_research_platform.catalog.profiles as profiles

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
        profiles.a_share_processed_materializer,
        "materialize_processed_data",
        fake_materialize,
    )

    layout = StorageLayout(project_root=Path("project"))
    time_window = TimeWindow("2014-01-01", "2014-01-31")
    kinds = frozenset({ProcessedDataKind.PANEL})

    assert (
        profiles._materialize_a_share_processed_data(
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
    import quant_research_platform.catalog.profiles as profiles

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
    import quant_research_platform.catalog.profiles as profiles

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
    import quant_research_platform.catalog.profiles as profiles

    assert profiles._a_share_csmar_source_id() == SourceId("csmar")
    assert isinstance(
        next(iter(profiles._a_share_csmar_source_interim_catalogs().values())),
        InterimDataCatalog,
    )


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules
