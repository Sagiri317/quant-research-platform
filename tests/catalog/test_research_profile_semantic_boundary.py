from __future__ import annotations

import re
from pathlib import Path


def test_catalog_profiles_does_not_own_a_share_scenario_parameters() -> None:
    path = Path("src/quant_research_platform/catalog/profiles.py")
    source = path.read_text(encoding="utf-8")

    forbidden_patterns = (
        r"_PROFILE_",
        r"\btrain_end\b",
        r"\btest_start\b",
        r"\btest_end\b",
        r"\bevaluation_end\b",
        r"\bfrequency\b",
        r"\bexecution_lag_steps\b",
        r"20[0-9][0-9]-",
        r"\bmax_train_rows\b",
        r"\brandom_seed\b",
        r"\bxgboost\b",
        r"\blightgbm\b",
        r"\bportfolio_id\b",
        r"\bsignal_id\b",
        r"\bmodel_id\b",
        r"\brank_weighted\b",
        r"\bforward_return\b",
        r"\bmom_",
        r"\brev_",
        r"\bbuild_a_share_prediction_catalog\b",
        r"\bbuild_a_share_portfolio_catalog\b",
    )
    hits: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for pattern in forbidden_patterns:
            if re.search(pattern, line):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []


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


def test_a_share_scenario_builders_require_smoke_windows_and_frequency() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )

    try:
        research_scenarios.build_a_share_prediction_specs()
    except TypeError as exc:
        assert "train_end" in str(exc)
    else:  # pragma: no cover - failure branch for clear assertion message
        raise AssertionError("prediction specs must require explicit smoke dates")

    try:
        research_scenarios.build_a_share_portfolio_specs()
    except TypeError as exc:
        assert "frequency" in str(exc)
    else:  # pragma: no cover - failure branch for clear assertion message
        raise AssertionError("portfolio specs must require explicit frequency")

    source = Path(
        "src/quant_research_platform/markets/implementations/a_share/"
        "research_scenarios.py"
    ).read_text(encoding="utf-8")
    for token in (
        "DEFAULT_TRAIN_END",
        "DEFAULT_TEST_START",
        "DEFAULT_EVALUATION_PERIOD_END",
        'train_end: str = "2014-06-30"',
        'test_start: str = "2014-07-01"',
        'frequency: str = "daily"',
    ):
        assert token not in source
