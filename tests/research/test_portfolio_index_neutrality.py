"""Simple neutrality consumes actual index weights without enhanced risk inputs."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
)
from quant_research_platform.research.portfolios import production
from quant_research_platform.research.portfolios.construction import (
    _neutrality_context,
    build_simple_target_weights,
)


def _candidate(neutrality="industry"):
    return PortfolioCandidateSpec(
        "index-neutral",
        "global",
        "quantile_equal_weight",
        "champion",
        "daily",
        0.5,
        "csi500",
        neutrality=neutrality,
        industry_neutrality_limit=0.0,
    )


def test_candidate_index_reference_is_only_used_for_simple_neutrality():
    assert _candidate().risk_style_columns == ()
    with pytest.raises(ValueError, match="portfolio_candidate_construction_invalid"):
        _candidate("none")


def test_index_constituents_outside_eligible_pool_are_reference_only():
    weights = build_simple_target_weights(
        pd.Series({"a": 2.0, "b": 1.0}),
        eligible=pd.Series({"a": True, "b": True}),
        industry=pd.Series({"a": "A", "b": "B", "outside": "A"}),
        benchmark_weights=pd.Series({"outside": 0.8, "b": 0.2}),
        neutrality="industry",
        industry_neutrality_limit=0.0,
    )
    assert weights.to_dict() == pytest.approx({"a": 0.8, "b": 0.2, "outside": 0.0})


def test_index_industry_unknown_keeps_original_known_weights():
    context = _neutrality_context(
        pd.Series({"a": True, "b": True}),
        None,
        pd.Series({"a": "A", "b": "B", "outside": None}),
        "industry",
        benchmark_weights=pd.Series({"a": 0.5, "b": 0.45, "outside": 0.05}),
    )
    assert context.benchmark_weight.to_dict() == {"a": 0.5, "b": 0.45}
    assert context.attrs["unknown_industry_weight"] == pytest.approx(0.05)


def test_index_reference_cannot_buy_outside_pool_to_satisfy_exposure():
    with pytest.raises(PortfolioTargetBlocked, match="target_neutrality_infeasible"):
        build_simple_target_weights(
            pd.Series({"a": 2.0, "b": 1.0}),
            eligible=pd.Series({"a": True, "b": True}),
            industry=pd.Series({"a": "A", "b": "B", "outside": "C"}),
            benchmark_weights=pd.Series({"outside": 1.0}),
            neutrality="industry",
            industry_neutrality_limit=0.0,
        )


@pytest.mark.parametrize("missing", ["snapshot", "size"])
def test_missing_index_reference_preserves_positions_and_cash(missing):
    from tests.helpers.portfolio_inputs import decision_inputs as _decision_inputs

    inputs = _decision_inputs()
    inputs["candidate"] = _candidate("size" if missing == "size" else "industry")
    inputs["enhanced_decision"] = None
    inputs["benchmark_weights"] = (
        pd.Series(dtype=float)
        if missing == "snapshot"
        else pd.Series({"a": 0.4, "b": 0.5, "outside": 0.1})
    )
    inputs["panel"] = inputs["panel"].assign(
        industry_standard_id="fixture",
        industry_level1_code="A",
        total_market_cap=1000.0,
    )
    account, _, _, _, ledger = production._build_session(**inputs)
    assert account.nav_after == 100.0
    assert ledger.post_trade_notional.tolist() == [50.0, 50.0]
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert ledger.attrs["decision_availability"]["decision_skip_reason"] == (
        "target_neutrality_benchmark_empty"
        if missing == "snapshot"
        else "target_neutrality_size_missing"
    )


@pytest.mark.parametrize("neutrality", ["industry", "size"])
def test_simple_index_producer_uses_current_snapshot_without_risk_or_calibration(
    tmp_path, monkeypatch, neutrality
):
    from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs
    from tests.research.test_portfolio_production import _build_producer

    frames, fees, window = _inputs()
    # All three names are real current-frame rows; the external member is ineligible.
    for name in ("dataset", "panel", "execution"):
        extra = (
            frames[name].loc[frames[name].asset_id.eq("a")].assign(asset_id="outside")
        )
        frames[name] = pd.concat((frames[name], extra), ignore_index=True)
    frames["dataset"].loc[frames["dataset"].asset_id.eq("outside"), "eligible"] = False
    frames["dataset"] = frames["dataset"].drop(
        columns=["total_market_cap", "float_market_cap"]
    )
    frames["panel"] = frames["panel"].assign(industry_standard_id="fixture")
    frames["panel"]["industry_level1_code"] = frames["panel"].asset_id.map(
        {"a": "A", "b": "B", "outside": "A"}
    )
    frames["benchmark"] = (
        frames["panel"]
        .loc[frames["panel"].asset_id.ne("a"), ["asset_id", "timestamp"]]
        .assign(benchmark_id="csi500")
    )
    b_weight = np.where(frames["benchmark"].timestamp.le("2023-01-03"), 0.2, 0.4)
    frames["benchmark"]["weight"] = np.where(
        frames["benchmark"].asset_id.eq("b"), b_weight, 1 - b_weight
    )
    observed = []
    original = production._neutrality_context

    def context(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Simple index neutrality must not request enhanced risk/calibration"
        )

    monkeypatch.setattr(production, "_neutrality_context", context)
    monkeypatch.setattr(production, "FactorRiskState", forbidden)
    monkeypatch.setattr(production, "_enhanced_decision", forbidden)
    batches, calls = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=_candidate(neutrality),
        typed_batches=True,
    )
    batches = list(batches)
    assert batches
    assert observed
    assert observed[0].benchmark_weight.to_dict() == {
        "a": 0.0,
        "b": 0.2,
        "outside": 0.8,
    }
    assert all(
        result.benchmark_weight.to_dict() == {"a": 0.0, "b": 0.4, "outside": 0.6}
        for result in observed[1:]
    )
    assert all(not result.attrs["eligible"]["outside"] for result in observed)
    assert any(name == "benchmark" for name, _, _ in calls)
    assert all(name != "risk_context" for name, _, _ in calls)

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import PortfolioCapital

    def produce(emit):
        for batch in batches:
            emit(batch)

    path = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=PortfolioCapital("index-neutral", 100.0, "CNY", "cash_equity"),
        metadata={
            "window": {
                "start": "2023-01-03",
                "end": "2023-01-06",
                "access": "development",
            }
        },
        produce=produce,
    )
    persisted = list(read_portfolio_ledger_artifact(path).iter_chunks())
    assert persisted == batches
    for position, batch in enumerate(persisted):
        diagnostics = {(row.asset_id, row.name): row for row in batch.diagnostics}
        assert diagnostics["a", "benchmark_weight"].number == 0.0
        assert diagnostics["b", "benchmark_weight"].number == (
            0.2 if position == 0 else 0.4
        )
        assert all(row.asset_id != "outside" for row in batch.trades)
        if neutrality == "industry":
            assert diagnostics[
                "__cash__", "benchmark_exposure:industry:fixture:A"
            ].number == pytest.approx(0.8 if position == 0 else 0.6)


def test_unknown_index_industry_does_not_disable_feasible_known_industries():
    weights = build_simple_target_weights(
        pd.Series({"a": 2.0, "b": 1.0}),
        eligible=pd.Series({"a": True, "b": True}),
        industry=pd.Series({"a": "A", "b": "B", "outside": None}),
        benchmark_weights=pd.Series({"a": 0.5, "b": 0.45, "outside": 0.05}),
        neutrality="industry",
        industry_neutrality_limit=0.03,
    )
    assert weights["a"] == pytest.approx(0.53)
    assert weights["b"] == pytest.approx(0.47)
    assert weights.sum() == pytest.approx(1.0)


def test_fixed_reference_names_do_not_inflate_joint_constraint_rank():
    weights = build_simple_target_weights(
        pd.Series({"a": 2.0, "b": 1.0}),
        eligible=pd.Series({"a": True, "b": True}),
        industry=pd.Series({"a": "A", "b": "B", "x": "A", "y": "A"}),
        total_market_cap=pd.Series({"a": 100.0, "b": 400.0, "x": 50.0, "y": 200.0}),
        benchmark_weights=pd.Series({"b": 0.5, "x": 0.25, "y": 0.25}),
        neutrality="industry_size",
        industry_neutrality_limit=0.0,
        size_neutrality_limit=0.0,
    )
    assert weights.to_dict() == pytest.approx({"a": 0.5, "b": 0.5, "x": 0.0, "y": 0.0})
