"""Keep statistical size, regression weights and investable weights independent."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioTargetBlocked,
)
from quant_research_platform.research.portfolios.construction import (
    _neutrality_context,
    build_simple_target_weights,
)
from quant_research_platform.research.portfolios.production import _enhanced_risk_cross
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)


def test_risk_default_and_explicit_float_use_sqrt_objective_weights():
    x = np.linspace(-2.0, 2.0, 100)
    cross = pd.DataFrame(
        {
            "asset_id": [f"a{i}" for i in range(100)],
            "timestamp": pd.Timestamp("2024-01-02"),
            "industry_standard_id": "fixture",
            "industry_level1_code": "A",
            "style": x,
            "asset_return": 0.01 * x**2 + 0.02 * x,
            "total_market_cap": np.exp(2 * x),
            "float_market_cap": np.exp(-x),
        }
    )
    results = []
    for config in (
        FactorRiskConfig(),
        FactorRiskConfig(weight_column="float_market_cap"),
    ):
        fit = FactorRiskState(
            style_columns=("style",), config=config
        ).fit_return_session(cross)
        assert fit is not None
        rows = fit.exposures.index
        design = fit.exposures.to_numpy()
        cap = cross.set_index("asset_id").loc[rows, config.weight_column].to_numpy()
        multiplier = cap**0.25
        expected = np.linalg.lstsq(
            design * multiplier[:, None],
            cross.set_index("asset_id").loc[rows, "asset_return"].to_numpy()
            * multiplier,
            rcond=None,
        )[0]
        np.testing.assert_allclose(fit.factor_returns, expected, rtol=1e-12, atol=1e-14)
        results.append(fit.factor_returns.to_numpy())
    assert not np.allclose(*results)
    assert FactorRiskConfig().weight_column == "total_market_cap"
    with pytest.raises(ValueError, match="factor_risk_columns_invalid"):
        FactorRiskState(style_columns=("style",)).fit_return_session(
            cross.drop(columns="total_market_cap")
        )
    cross.loc[0, "total_market_cap"] = np.nan
    prepared = FactorRiskState(style_columns=("style",)).prepare_exposures(cross)
    assert "a0" not in set(prepared.asset_id)
    assert len(prepared) == 99


def test_production_risk_cross_consumes_only_selected_capitalization():
    from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs

    frames, _, _ = _inputs()
    panel = (
        frames["panel"]
        .loc[lambda x: x.timestamp.eq(x.timestamp.min())]
        .assign(
            period_return=0.01,
            total_market_cap=[100.0, 900.0],
            float_market_cap=[80.0, 30.0],
            industry_standard_id="fixture",
            industry_level1_code="A",
            is_nonfinancial_industry=True,
        )
    )
    for column, other in (
        ("total_market_cap", "float_market_cap"),
        ("float_market_cap", "total_market_cap"),
    ):
        actual = _enhanced_risk_cross(
            panel=panel.drop(columns=other),
            risk_frames=[],
            style_columns=(),
            weight_column=column,
        )
        assert actual[column].tolist() == panel[column].tolist()
        assert other not in actual
    with pytest.raises(KeyError, match="total_market_cap"):
        _enhanced_risk_cross(
            panel=panel.drop(columns="total_market_cap"),
            risk_frames=[],
            style_columns=(),
        )


def test_simple_size_neutrality_preserves_float_reference_and_can_select_float():
    assets = pd.Index(list("abcd"))
    floating = pd.Series([1.0, 2.0, 1.0, 2.0], index=assets)
    total = pd.Series([1.0, 2.0, 4.0, 8.0], index=assets)
    eligible = pd.Series(True, index=assets)
    signal = pd.Series([0.0, 1.0, 0.0, 0.0], index=assets)
    results = []
    for column, selected in (
        ("total_market_cap", total),
        ("float_market_cap", floating),
    ):
        context = _neutrality_context(
            eligible, floating, None, "size", size_market_cap=selected
        )
        np.testing.assert_allclose(context.benchmark_weight, floating / floating.sum())
        log_size = np.log(selected)
        median = np.median(log_size)
        mad = np.median(np.abs(log_size - median))
        clipped = log_size.clip(median - 3 * 1.4826 * mad, median + 3 * 1.4826 * mad)
        expected_size = (clipped - clipped.mean()) / clipped.std(ddof=0)
        np.testing.assert_allclose(context["size"], expected_size, atol=1e-12)
        weights = build_simple_target_weights(
            signal,
            eligible=eligible,
            float_market_cap=floating,
            total_market_cap=total,
            neutrality="size",
            size_neutrality_limit=0.0,
            size_cap_column=column,
        )
        assert weights @ context["size"] == pytest.approx(
            context.benchmark_weight @ context["size"], abs=1e-8
        )
        results.append(weights)
    assert not np.allclose(*results)
    default = build_simple_target_weights(
        signal,
        eligible=eligible,
        float_market_cap=floating,
        total_market_cap=total,
        neutrality="size",
        size_neutrality_limit=0.0,
    )
    np.testing.assert_allclose(default, results[0])
    cap_weighted = build_simple_target_weights(
        signal,
        eligible=eligible,
        float_market_cap=floating,
        total_market_cap=total,
        method="float_cap_weight",
        top_quantile=1.0,
    )
    np.testing.assert_allclose(cap_weighted, floating / floating.sum())


@pytest.mark.parametrize("missing", [None, np.nan, 0.0, -1.0])
def test_missing_total_blocks_only_size_without_reweighting_reference(missing):
    assets = pd.Index(list("abcd"))
    floating = pd.Series([1.0, 2.0, 1.0, 2.0], index=assets)
    total = (
        None if missing is None else pd.Series([1.0, 2.0, missing, 8.0], index=assets)
    )
    kwargs = dict(
        signal=pd.Series([0.0, 1.0, 0.0, 0.0], index=assets),
        eligible=pd.Series(True, index=assets),
        float_market_cap=floating,
        total_market_cap=total,
        industry=pd.Series("A", index=assets),
    )
    with pytest.raises(PortfolioTargetBlocked, match="target_neutrality_size_missing"):
        build_simple_target_weights(**kwargs, neutrality="size")
    for mode in ("none", "industry"):
        result = build_simple_target_weights(**kwargs, neutrality=mode)
        assert result.sum() == pytest.approx(1.0)
    result = build_simple_target_weights(
        **kwargs, neutrality="size", size_cap_column="float_market_cap"
    )
    assert result.sum() == pytest.approx(1.0)


def test_candidate_size_selection_is_persistable_and_validated():
    candidate = PortfolioCandidateSpec(
        "size",
        "global",
        "quantile_equal_weight",
        "ridge",
        "daily",
        0.2,
        None,
        neutrality="size",
    )
    assert asdict(candidate)["size_cap_column"] == "total_market_cap"
    assert (
        PortfolioCandidateSpec(
            **asdict(replace(candidate, size_cap_column="float_market_cap"))
        ).size_cap_column
        == "float_market_cap"
    )
    with pytest.raises(ValueError, match="portfolio_candidate_size_cap_column_invalid"):
        replace(candidate, size_cap_column="other")


@pytest.mark.parametrize("weight_column", ["total_market_cap", "float_market_cap"])
def test_backtest_risk_reads_only_configured_capitalization(
    tmp_path, monkeypatch, weight_column
):
    from quant_research_platform.research.portfolios import production
    from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
    from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs
    from tests.research.test_portfolio_production import _build_producer

    frames, fees, window = _inputs()
    frames["panel"] = (
        frames["panel"]
        .drop(columns="total_market_cap")
        .assign(
            period_return=0.01,
            industry_standard_id="fixture",
            industry_level1_code="A",
            is_nonfinancial_industry=True,
        )
    )
    frames["panel"][weight_column] = frames["panel"].asset_id.map(
        {"a": 100.0, "b": 900.0}
    )
    frames["benchmark"] = frames["panel"][["asset_id", "timestamp"]].assign(
        benchmark_id="csi500", weight=0.5
    )
    candidate = PortfolioCandidateSpec(
        "risk",
        "global",
        "benchmark_enhanced",
        "champion",
        "daily",
        None,
        "csi500",
        RISK_STYLE_COLUMNS,
    )
    observed = []
    original = production._enhanced_risk_cross

    def cross(**kwargs):
        result = original(**kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(production, "_enhanced_risk_cross", cross)
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
        risk_config=None
        if weight_column == "total_market_cap"
        else FactorRiskConfig(weight_column=weight_column),
    )
    assert list(batches)
    assert observed
    other = (
        "float_market_cap"
        if weight_column == "total_market_cap"
        else "total_market_cap"
    )
    assert all(other not in frame for frame in observed)
    assert observed[0][weight_column].tolist() == [100.0, 900.0]


@pytest.mark.parametrize("column", ["total_market_cap", "float_market_cap"])
def test_simple_production_reads_selected_size_and_keeps_reference(
    tmp_path, monkeypatch, column
):
    from quant_research_platform.research.portfolios import production
    from tests.helpers.portfolio_inputs import portfolio_inputs as _inputs
    from tests.research.test_portfolio_production import _build_producer

    frames, fees, window = _inputs()
    frames["dataset"]["total_market_cap"] = frames["dataset"].asset_id.map(
        {"a": 2000.0, "b": 9000.0}
    )
    expected = [2000.0, 9000.0] if column == "total_market_cap" else [1000.0, 2000.0]
    if column == "float_market_cap":
        frames["dataset"] = frames["dataset"].drop(columns="total_market_cap")
    observed = []
    original = production._neutrality_context

    def context(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append((kwargs["size_market_cap"].tolist(), result))
        return result

    monkeypatch.setattr(production, "_neutrality_context", context)
    candidate = PortfolioCandidateSpec(
        "size",
        "global",
        "quantile_equal_weight",
        "champion",
        "daily",
        0.5,
        None,
        neutrality="size",
        size_cap_column=column,
    )
    batches, _ = _build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        candidate=candidate,
    )
    assert list(batches)
    assert observed
    assert observed[0][0] == expected
    np.testing.assert_allclose(observed[0][1].benchmark_weight, [1 / 3, 2 / 3])


def test_missing_size_nonrebalance_marks_undefined_and_preserves_account():
    from quant_research_platform.research.portfolios.production import _build_session
    from tests.helpers.portfolio_inputs import decision_inputs as _decision_inputs

    inputs = _decision_inputs()
    inputs["candidate"] = replace(
        inputs["candidate"],
        construction="quantile_equal_weight",
        top_quantile=0.5,
        benchmark_id="csi500",
        risk_style_columns=(),
        rebalance_schedule="weekly-last-trading-day",
        neutrality="size",
    )
    inputs["enhanced_decision"] = None
    # Fixed-index size missingness does not remove otherwise eligible holdings.
    inputs["benchmark_weights"] = pd.Series({"a": 0.5, "b": 0.5})
    inputs["panel"]["total_market_cap"] = np.nan
    account, _, _, _, ledger = _build_session(**inputs)
    assert account.nav_after == 100.0
    assert ledger.post_trade_notional.tolist() == [50.0, 50.0]
    assert ledger.executed_delta_notional.eq(0.0).all()
    diagnostics = ledger.attrs["neutrality_exposures"]
    assert diagnostics["target_exposure_status"] == "UNDEFINED"
    assert diagnostics["target_exposure_reason"] == "target_neutrality_size_missing"
    inputs["candidate"] = replace(inputs["candidate"], rebalance_schedule="daily")
    account, _, optimized, _, ledger = _build_session(**inputs)
    assert optimized is None
    assert account.nav_after == 100.0
    assert ledger.executed_delta_notional.eq(0.0).all()
    assert (
        ledger.attrs["decision_availability"]["decision_skip_reason"]
        == "target_neutrality_size_missing"
    )
    assert (
        ledger.attrs["neutrality_exposures"]["realized_exposure_status"] == "UNDEFINED"
    )


@pytest.mark.parametrize("column", ["total_market_cap", "float_market_cap"])
def test_panel_selection_cannot_reintroduce_excluded_risk_capitalization(
    tmp_path, column
):
    from quant_research_platform.markets.implementations.a_share import (
        portfolio_sessions,
    )
    from quant_research_platform.research.portfolios.production import (
        _ENHANCED_PANEL_COLUMNS,
    )
    from tests.helpers.portfolio_inputs import closed_market_inputs as _native_inputs

    inputs, _, calendar, _ = _native_inputs(tmp_path, fault="normal_open")
    kwargs = dict(
        market_bars_root=inputs["market_bars"],
        stock_master_root=inputs["stock_master"],
        calendar=calendar,
        panel_columns=(*_ENHANCED_PANEL_COLUMNS, column),
        include_asset_sessions=False,
    )
    other = "float_market_cap" if column == "total_market_cap" else "total_market_cap"
    with pytest.raises(ValueError, match="portfolio_input_business_columns_excluded"):
        portfolio_sessions._portfolio_session_evidence(
            panel_root={
                "path": inputs["panel"],
                "columns": [*_ENHANCED_PANEL_COLUMNS, other],
            },
            **kwargs,
        )
    result = portfolio_sessions._portfolio_session_evidence(
        panel_root={
            "path": inputs["panel"],
            "columns": [*_ENHANCED_PANEL_COLUMNS, column],
        },
        **kwargs,
    )
    assert column in result[1].logical_columns
    assert other not in result[1].logical_columns


@pytest.mark.parametrize("column", ["total_market_cap", "float_market_cap"])
def test_native_backtest_persists_resolved_risk_and_size_defaults(tmp_path, column):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.portfolios.production import backtest
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.portfolio_inputs import _processed_specs
    from tests.helpers.portfolio_inputs import closed_market_inputs as _native_inputs
    from tests.research.direct_model_test_support import fit_frame

    inputs, _, _, panel = _native_inputs(tmp_path / "market", fault="normal_open")
    training = (
        read_dataset(inputs["dataset"])
        .loc[lambda x: x.timestamp.le("2022-12-20")]
        .copy()
    )
    training["signal"] = training.asset_id.str[1:].astype(float)
    training["label"] = training.signal * 0.001
    training["label_available_at"] = training.timestamp
    fitted = fit_frame(
        tmp_path / "learning",
        training,
        ModelSpec(
            model_id="ridge",
            model_algorithm="ridge_regression",
            input_factor_columns=("signal",),
            target_column="label",
            random_seed=42,
        ),
        market_id="a_share",
    )
    window = {"start": "2023-01-02", "end": "2023-01-06"}
    predicted = predict(
        {
            "model": str(fitted.root),
            "inputs": str(inputs["dataset"]),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    benchmark = panel[["market_id", "asset_id", "timestamp"]].assign(
        benchmark_id="csi500", weight=1 / panel.asset_id.nunique()
    )
    benchmark_root = tmp_path / "benchmark"
    write_dataset(
        benchmark_root, benchmark, spec=_processed_specs().get("benchmark_weights")
    )
    config = {
        "candidate": {
            "candidate_id": "native",
            "selection_group": "global",
            "construction": "benchmark_enhanced",
            "prediction_selector": "ridge",
            "rebalance_schedule": "daily",
            "top_quantile": None,
            "benchmark_id": "csi500",
            "risk_style_columns": ["signal"],
        },
        "inputs": {
            **{
                key: str(value)
                for key, value in inputs.items()
                if key != "risk_context"
            },
            "benchmark": str(benchmark_root),
            "risk_context": [str(inputs["dataset"])],
        },
        "predictions": [{"path": str(predicted), "window": window}],
        "output": str(tmp_path / "ledger"),
        "output_window": window,
        "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "formal_dataset_window": {"start": "2022-12-01", "end": "2023-01-06"},
        "prediction_window": window,
        "assumptions": {
            "initial_nav": 100.0,
            "slippage_bps": 0.0,
            "impact_bps": 0.0,
            "adv_participation": 1.0,
            "adv_sensitivities": [1.0],
        },
    }
    if column == "float_market_cap":
        config["risk_config"] = {"weight_column": column}
    artifact = read_portfolio_ledger_artifact(backtest(config))
    assert list(artifact.iter_chunks())
    assert artifact.metadata["candidate"]["size_cap_column"] == "total_market_cap"
    assert artifact.metadata["risk_config"]["weight_column"] == column
    assert artifact.metadata["risk_config"]["weighting"] == "sqrt_cap"
