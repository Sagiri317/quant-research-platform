"""Hand-check full-domain projection and separate target/execution facts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.portfolios.construction import (
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.production import _validate_windows


@pytest.mark.parametrize(
    "neutrality,expected",
    [
        ("industry_size", [0.0, 0.5, 1 / 3, 1 / 6]),
        ("industry", [0.0, 0.5, 0.25, 0.25]),
        ("size", [1 / 6, 2 / 3, 1 / 6, 0.0]),
    ],
)
def test_projection_handcheck_includes_unpredicted_benchmark_assets(
    neutrality, expected
):
    assets = pd.Index(["a", "b", "c", "d"])
    caps = pd.Series([1.0, 2.0, 1.0, 2.0], index=assets)
    weights = build_simple_target_weights(
        pd.Series([np.nan, 1.0, np.nan, np.nan], index=assets),
        eligible=pd.Series(True, index=assets),
        float_market_cap=caps,
        industry=pd.Series(["A", "A", "B", "B"], index=assets)
        if neutrality != "size"
        else None,
        neutrality=neutrality,
    )
    assert weights.tolist() == pytest.approx(expected, abs=1e-8)
    assert weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert weights.ge(0).all()
    if neutrality != "size":
        assert weights.iloc[:2].sum() == pytest.approx(0.5)
    if neutrality != "industry":
        assert float(weights @ np.log(caps)) == pytest.approx(2 / 3 * np.log(2))


def test_projection_failure_does_not_silently_relax(monkeypatch):
    import quant_research_platform.research.portfolios.optimizer as optimizer

    monkeypatch.setattr(
        optimizer, "minimize", lambda *a, **kw: SimpleNamespace(success=False)
    )
    with pytest.raises(PortfolioTargetBlocked, match="target_neutrality_solver_failed"):
        build_simple_target_weights(
            pd.Series({"a": 1.0}),
            eligible=pd.Series({"a": True}),
            float_market_cap=pd.Series({"a": 1.0}),
            neutrality="size",
        )


def test_mixed_precision_windows_keep_whole_date_and_nanosecond_bounds():
    _validate_windows(
        TimeWindow("2023-01-02T12:00:00.000000001", "2023-01-02T23:59:59.999999999"),
        TimeWindow("2023-01-01", "2023-01-02"),
        TimeWindow("2023-01-02", "2023-01-02"),
    )
    with pytest.raises(ValueError, match="portfolio_windows_invalid"):
        _validate_windows(
            TimeWindow("2023-01-02", "2023-01-02"),
            TimeWindow("2023-01-01T00:00:00", "2023-01-02T23:59:59.999999998"),
            TimeWindow("2023-01-01", "2023-01-02"),
        )


def test_neutral_target_and_blocked_realized_exposure_are_distinct(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, window = _inputs()
    frames["panel"]["industry_standard_id"] = "fixture"
    frames["panel"]["industry_level1_code"] = frames["panel"].asset_id
    frames["execution"].loc[frames["execution"].asset_id.eq("a"), "can_enter"] = False
    candidate = PortfolioCandidateSpec(
        "neutral",
        "global",
        "top20_equal_weight",
        "ridge",
        "daily",
        0.5,
        None,
        neutrality="industry_size",
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        typed_batches=True,
    )
    with closing_iterator(batches):
        first = next(iter(batches))
    facts = {
        row.name: row.number if row.number is not None else row.text
        for row in first.diagnostics
        if row.asset_id == "__cash__"
    }
    assert facts["target_exposure:industry:fixture:a"] == pytest.approx(1 / 3)
    assert facts["benchmark_exposure:industry:fixture:a"] == pytest.approx(1 / 3)
    assert facts["realized_exposure:industry:fixture:a"] == 0.0
    assert facts["realized_exposure_status"] == "DEFINED"
    assert any(row.name == "raw_target_weight" for row in first.diagnostics)
    assert replace(candidate, neutrality="none").neutrality == "none"


def test_nonrebalance_holding_outside_benchmark_marks_both_exposures_unknown(
    tmp_path, monkeypatch
):
    from tests.research.test_portfolio_production import _build_producer, _inputs

    frames, fees, window = _inputs()
    for name in ("panel", "dataset", "execution"):
        last = frames[name].loc[frames[name].timestamp.eq("2023-01-10")]
        frames[name] = pd.concat(
            (frames[name], last.assign(timestamp=pd.Timestamp("2023-01-11"))),
            ignore_index=True,
        )
    frames["calendar"] = pd.concat(
        (
            frames["calendar"],
            frames["calendar"]
            .iloc[-1:]
            .assign(calendar_date=pd.Timestamp("2023-01-11")),
        ),
        ignore_index=True,
    )
    frames["panel"]["industry_standard_id"] = "fixture"
    frames["panel"]["industry_level1_code"] = frames["panel"].asset_id
    frames["dataset"].loc[
        frames["dataset"].asset_id.eq("b")
        & frames["dataset"].timestamp.ge("2023-01-09"),
        "eligible",
    ] = False
    candidate = PortfolioCandidateSpec(
        "neutral",
        "global",
        "top20_equal_weight",
        "ridge",
        "weekly-last-trading-day",
        0.5,
        None,
        neutrality="industry",
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=TimeWindow("2023-01-03", "2023-01-11"),
        input_window=TimeWindow("2022-12-01", "2023-01-11"),
        formal_dataset_window=TimeWindow("2022-12-01", "2023-01-11"),
        candidate=candidate,
        typed_batches=True,
    )
    matched = []
    for batch in batches:
        facts = {
            row.name: row.number if row.number is not None else row.text
            for row in batch.diagnostics
            if row.asset_id == "__cash__"
        }
        if facts.get("target_exposure_status") == "UNDEFINED":
            matched.append(facts)
            for kind in ("target", "realized"):
                assert facts[f"{kind}_exposure_reason"] == "outside_benchmark_domain"
                assert facts[f"{kind}_exposure_excluded_weight"] > 0.0
                assert not any(key.startswith(f"{kind}_exposure:") for key in facts)
    assert matched


def test_neutral_projection_satisfies_kkt_on_active_and_boundary_assets():
    from quant_research_platform.research.portfolios.optimizer import (
        _project_neutral_target,
    )

    context = pd.DataFrame(
        {
            "benchmark_weight": [0.2, 0.2, 0.2, 0.2, 0.2],
            "size": [-2.0, -1.0, 0.0, 1.0, 2.0],
        },
        index=list("abcde"),
    )
    target = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0], index=context.index)
    weights = _project_neutral_target(target, context).to_numpy()
    matrix = np.vstack((np.ones(5), context["size"]))
    gradient = weights - target.to_numpy()
    active = weights > 1e-9
    multiplier = np.linalg.lstsq(matrix[:, active].T, -gradient[active], rcond=None)[0]
    reduced_gradient = gradient + matrix.T @ multiplier
    assert matrix @ weights == pytest.approx(
        matrix @ context.benchmark_weight, abs=1e-8
    )
    assert reduced_gradient[active] == pytest.approx(np.zeros(active.sum()), abs=1e-7)
    assert (reduced_gradient[~active] >= -1e-7).all()
    assert (weights >= 0.0).all()
