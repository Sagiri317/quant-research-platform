from __future__ import annotations

import re
from pathlib import Path

from tests.helpers import frequency_ids


def test_research_profile_does_not_own_a_share_scenario_parameters() -> None:
    path = Path("src/quant_research_platform/profiles/research.py")
    source = path.read_text(encoding="utf-8")

    forbidden_patterns = (
        r"_PROFILE_",
        r"DEFAULT_",
        r"20[0-9][0-9]-",
        r"\btrain_end\s*=",
        r"\btest_start\s*=",
        r"\bfrequency\s*=",
        r"\bexecution_lag_steps\s*=",
    )
    hits: list[str] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        for pattern in forbidden_patterns:
            if re.search(pattern, line):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert hits == []
    assert "predictions=PredictionCatalog()" in source
    assert "portfolios=PortfolioCatalog()" in source


def test_a_share_scenarios_own_research_scenario_catalog_builders() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )

    for name in (
        "build_signal_specs",
        "build_model_specs",
        "build_prediction_specs",
        "build_evaluation_specs",
        "build_portfolio_specs",
        "build_model_catalog",
        "build_prediction_catalog",
        "build_evaluation_catalog",
        "build_portfolio_catalog",
        "default_evaluation_run_id",
        "default_prediction_smoke_id",
        "default_prediction_oos_period",
    ):
        assert hasattr(research_scenarios, name)


def test_a_share_scenario_builders_require_smoke_windows_and_frequency() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios,
    )

    try:
        research_scenarios.build_prediction_specs()
    except TypeError as exc:
        assert "train_end" in str(exc)
    else:  # pragma: no cover - failure branch for clear assertion message
        raise AssertionError("prediction specs must require explicit smoke dates")

    portfolio_specs = research_scenarios.build_portfolio_specs()
    assert portfolio_specs[0].rebalance_frequency == frequency_ids.DAILY_FREQUENCY_SLUG

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
