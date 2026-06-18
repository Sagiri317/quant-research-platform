from __future__ import annotations

from pathlib import Path


def test_catalog_profiles_does_not_own_a_share_scenario_parameters() -> None:
    source = Path("src/quant_research_platform/catalog/profiles.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "mom_20d",
        "rev_1d",
        "forward_return_20d",
        "2014-",
        "n_estimators",
        "max_train_rows",
        "model_algorithm",
        "prediction_id",
        "portfolio_id",
        "SignalSpec(",
        "direction=\"higher\"",
        "direction=\"lower\"",
        "column=\"",
    )
    for token in forbidden:
        assert token not in source


def test_a_share_scenarios_own_research_scenario_catalog_builders() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )

    for name in (
        "build_a_share_signal_specs",
        "build_a_share_model_specs",
        "build_a_share_prediction_specs",
        "build_a_share_evaluation_specs",
        "build_a_share_portfolio_specs",
        "build_a_share_model_catalog",
        "build_a_share_prediction_catalog",
        "build_a_share_evaluation_catalog",
        "build_a_share_portfolio_catalog",
        "default_a_share_evaluation_run_id",
        "default_a_share_prediction_smoke_id",
        "default_a_share_prediction_oos_period",
    ):
        assert hasattr(research_scenarios, name)
