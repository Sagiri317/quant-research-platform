"""Universal optional comparison through native accounts and persisted STAT tables."""

import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import write_crypto_episode_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    PortfolioEconomics,
    _record_execution_result,
)
from quant_research_platform.research.portfolios.comparison import comparison_tables
from quant_research_platform.research.portfolios.diagnostics import (
    crypto_statistics,
    portfolio_statistics,
)
from tests.helpers.benchmark_references import reference_input
from tests.research.test_optional_statistics import _cash_ledger


def _axis(returns, dates):
    return pd.DataFrame(
        {
            "candidate_id": "strategy",
            "execution_mode": "continuous",
            "episode_id": "continuous",
            "fold_id": "",
            "sample_role": "test",
            "start_at": dates[:-1],
            "end_at": dates[1:],
            "portfolio_return": returns,
            "portfolio_reason": [
                "" if np.isfinite(value) else "account_path_unknown"
                for value in returns
            ],
        }
    )


def _compare(axis, references, **options):
    sessions = options.pop(
        "sessions", pd.bdate_range(axis.start_at.min(), axis.end_at.max())
    )
    config = {"comparison_benchmarks": references, **options}
    analyses = options.pop("analyses", ["benchmark_comparison"])
    metadata = {
        "window": {"start": str(axis.start_at.min()), "end": str(axis.end_at.max())},
        "sample_role": "test",
        "execution_mode": "continuous",
        "candidate_ids": ["strategy"],
    }
    tables = comparison_tables(
        axis,
        config=config,
        analyses=analyses,
        currencies={"strategy": "CNY"},
        clock="trading_day_open",
        sessions=sessions,
        metadata=metadata,
    )
    return tables, metadata


@pytest.mark.parametrize(
    "construction",
    [
        "quantile_equal_weight",
        "quantile_equal_weight_half_step",
        "quantile_float_cap",
        "benchmark_enhanced",
    ],
)
def test_each_construction_defaults_to_performance_and_accepts_two_references(
    tmp_path, construction, monkeypatch
):
    base = PortfolioCandidateSpec(
        candidate_id="cash",
        selection_group="global",
        construction="quantile_equal_weight",
        prediction_selector="explicit",
        rebalance_schedule="daily",
        top_quantile=0.2,
        benchmark_id=None,
    )
    candidate = replace(
        base,
        construction=construction,
        top_quantile=None if construction == "benchmark_enhanced" else 0.2,
        benchmark_id="csi500" if construction == "benchmark_enhanced" else None,
        risk_style_columns=("size",) if construction == "benchmark_enhanced" else (),
    )
    source = tmp_path / "ledger"
    window = _cash_ledger(source, candidate=candidate)
    first = reference_input(tmp_path / "first.parquet", [100.0, 105.0, 110.0])
    second = reference_input(
        tmp_path / "second.parquet", [100.0, 90.0, 120.0], reference_id="other"
    )
    config = {
        "ledgers": [str(source)],
        "window": window,
        "comparison_benchmarks": [first, second],
        "output": str(tmp_path / "default"),
    }
    from quant_research_platform.research.portfolios import comparison

    original = comparison.open_dataset_session
    monkeypatch.setattr(
        comparison,
        "open_dataset_session",
        lambda **_: pytest.fail("unrequested reference read"),
    )
    default = read_statistics_artifact(portfolio_statistics(config))
    assert set(default._members) == {
        "portfolio_daily.parquet",
        "portfolio_performance.parquet",
    }
    monkeypatch.setattr(comparison, "open_dataset_session", original)
    output = read_statistics_artifact(
        portfolio_statistics(
            {
                **config,
                "output": str(tmp_path / "compared"),
                "analyses": ["performance", "benchmark_comparison"],
            }
        )
    )
    with (
        default.table("portfolio_daily.parquet") as before,
        output.table("portfolio_daily.parquet") as after,
    ):
        pd.testing.assert_frame_equal(before, after)
    with output.table("portfolio_benchmark_intervals.parquet") as intervals:
        assert len(intervals) == 4 and set(intervals.reference_id) == {
            "market",
            "other",
        }
        assert intervals.status.eq("DEFINED").all()


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_local_invalid_level_and_missing_reference_preserve_other_results(
    tmp_path, invalid
):
    dates = pd.bdate_range("2024-01-02", periods=5)
    axis = _axis([0.1, 0.02, -0.01, 0.03], dates)
    bad = reference_input(
        tmp_path / "bad.parquet",
        [100.0, 110.0, invalid, 112.0, 114.0],
        dates=dates,
        reference_id="bad",
    )
    good = reference_input(
        tmp_path / "good.parquet", [100.0, 105.0, 106.0, 107.0, 108.0], dates=dates
    )
    absent = {**good, "reference_id": "missing", "input": str(tmp_path / "missing")}
    tables, metadata = _compare(axis, [bad, good, absent])
    intervals = tables["portfolio_benchmark_intervals"]
    assert intervals.loc[intervals.reference_id.eq("bad"), "status"].tolist() == [
        "DEFINED",
        "UNDEFINED",
        "UNDEFINED",
        "DEFINED",
    ]
    assert (
        intervals.loc[intervals.reference_id.eq("market"), "status"].eq("DEFINED").all()
    )
    assert (
        intervals.loc[intervals.reference_id.eq("missing"), "reason"]
        .eq("reference_input_missing")
        .all()
    )
    assert (
        intervals.loc[intervals.reference_id.eq("bad"), "relative_wealth"]
        .iloc[1:]
        .isna()
        .all()
    )
    rows = tables["portfolio_benchmark_metrics"]
    assert rows.loc[rows.reference_id.eq("bad"), "observation_count"].eq(2).all()
    assert (
        rows.loc[rows.reference_id.eq("bad"), "planned_observation_count"].eq(4).all()
    )
    persisted = read_statistics_artifact(
        write_statistics_artifact(
            tmp_path / "statistics",
            metadata=metadata,
            tables=tables,
            table_group="portfolio",
        )
    )
    with persisted.table("portfolio_benchmark_intervals.parquet") as restored:
        pd.testing.assert_frame_equal(restored, intervals)


def test_endpoint_interval_percent_hand_math_and_complete_subinterval_coverage(
    tmp_path,
):
    dates = pd.bdate_range("2024-01-02", periods=3)
    axis = _axis([0.1, -0.04], dates)
    level = reference_input(
        tmp_path / "levels.parquet", [100.0, 105.0, 115.5], dates=dates
    )
    returns = reference_input(
        tmp_path / "returns.parquet",
        [5.0, 10.0, 0.0],
        dates=dates,
        reference_id="percent",
        representation="interval_simple_return",
        unit="percent",
    )
    tables, _ = _compare(axis, [level, returns])
    rows = tables["portfolio_benchmark_intervals"]
    a, b = [
        part.reset_index(drop=True)
        for _, part in rows.groupby("reference_id", sort=False)
    ]
    assert a.return_difference.tolist() == pytest.approx([0.05, -0.14])
    assert a.relative_wealth.tolist() == pytest.approx([1.1 / 1.05, 1.1 * 0.96 / 1.155])
    assert a.relative_wealth.iloc[0] - 1 != pytest.approx(0.05)
    np.testing.assert_allclose(
        a.benchmark_return, b.benchmark_return, rtol=1e-14, atol=0
    )
    combined = axis.iloc[:1].copy()
    combined.loc[0, "end_at"] = dates[2]
    combined.loc[0, "portfolio_return"] = 1.1 * 0.96 - 1
    aggregated, _ = _compare(combined, [returns])
    assert aggregated["portfolio_benchmark_intervals"].benchmark_return.iloc[
        0
    ] == pytest.approx(0.155)
    descriptor = deepcopy(returns)
    descriptor["return_unit"] = "decimal"
    with pytest.raises(ValueError, match="source_facts_conflict"):
        _compare(axis, [descriptor])


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("duplicate", "duplicate"),
        ("ambiguous", "ambiguous_series"),
        ("selector", "filter_column_not_found"),
        ("clock", "timestamp_basis_invalid"),
        ("unit", "return_unit_required"),
    ],
)
def test_reference_structure_errors_are_strict(tmp_path, mutation, error):
    dates = pd.bdate_range("2024-01-02", periods=3)
    ref = reference_input(
        tmp_path / "reference.parquet",
        [0.0, 0.1, 0.0],
        dates=dates,
        representation="interval_simple_return",
    )
    if mutation in {"duplicate", "ambiguous"}:
        frame = pd.read_parquet(ref["input"]["path"])
        extra = frame.copy()
        if mutation == "ambiguous":
            extra["series"] = "different"
            ref["input"].pop("row_filters")
        pd.concat([frame, extra]).to_parquet(ref["input"]["path"], index=False)
    elif mutation == "selector":
        ref["input"]["row_filters"] = [["not_a_column", "==", "market"]]
    elif mutation == "clock":
        ref["timestamp_basis"] = "midnight_is_open"
    else:
        ref["input"]["spec"].pop("column_units")
    with pytest.raises(ValueError, match=error):
        _compare(_axis([0.1, 0.1], dates), [ref])


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("valuation_currency", "USDT", "valuation_currency_mismatch"),
        ("timestamp_basis", "trading_day_close", "timestamp_basis_mismatch"),
        ("timestamp_basis", "utc", "timestamp_basis_mismatch"),
    ],
)
def test_incompatible_economic_clock_or_currency_is_not_applicable(
    tmp_path, field, value, reason
):
    dates = pd.bdate_range("2024-01-02", periods=3)
    ref = reference_input(
        tmp_path / "reference.parquet", [100.0, 105.0, 110.0], dates=dates
    )
    ref[field] = value
    tables, _ = _compare(_axis([0.1, 0.1], dates), [ref])
    rows = tables["portfolio_benchmark_intervals"]
    assert rows.status.eq("NOT_APPLICABLE").all() and rows.reason.eq(reason).all()
    assert rows.return_difference.isna().all() and rows.portfolio_return.notna().all()


def test_market_exposure_hac_and_regimes_use_intact_axis_and_descriptive_timing(
    tmp_path, monkeypatch
):
    dates = pd.bdate_range("2024-01-02", periods=7)
    market = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.04])
    portfolio = (
        0.002 + 1.5 * market + np.array([0.001, -0.001, -0.001, 0.001, 0.002, -0.002])
    )
    ref = reference_input(
        tmp_path / "reference.parquet",
        [*market, 0],
        dates=dates,
        representation="interval_simple_return",
    )
    options = {
        "analyses": ["market_exposure", "regime_comparison"],
        "market_exposure": {
            "hac_lag": 1,
            "confidence": 0.95,
            "minimum_observations": 3,
        },
        "regime_comparison": {"threshold": 0.0},
    }
    tables, metadata = _compare(_axis(portfolio, dates), [ref], **options)
    assert set(tables) == {"portfolio_market_exposure", "portfolio_regime_comparison"}
    exposure = tables["portfolio_market_exposure"].set_index("coefficient_id")
    beta = np.dot(market - market.mean(), portfolio - portfolio.mean()) / np.dot(
        market - market.mean(), market - market.mean()
    )
    assert exposure.loc["beta", "value"] == pytest.approx(beta)
    assert exposure.loc["market_return_intercept", "value"] == pytest.approx(
        portfolio.mean() - beta * market.mean()
    )
    assert "alpha" not in exposure.index
    assert metadata["comparison_method"]["regime_label_timing"] == "ex_post_descriptive"
    regime = tables["portfolio_regime_comparison"].set_index(["regime", "metric_id"])
    assert regime.loc[
        ("above_threshold", "mean_return_difference"), "value"
    ] == pytest.approx((portfolio - market)[market > 0].mean())
    import statsmodels.regression.linear_model as regression

    monkeypatch.setattr(
        regression,
        "OLS",
        lambda *_args, **_kwargs: pytest.fail("compressed HAC attempted"),
    )
    broken = portfolio.copy()
    broken[2] = np.nan
    missing, _ = _compare(_axis(broken, dates), [ref], **options)
    assert missing["portfolio_market_exposure"].status.eq("UNDEFINED").all()
    assert (
        missing["portfolio_market_exposure"].reason.eq("incomplete_paired_path").all()
    )


def test_numeric_market_exposure_does_not_import_plotting_dependencies():
    script = """
import sys
import numpy as np
import pandas as pd
from quant_research_platform.research.portfolios.comparison import _exposure

assert not any(
    name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules
)
market = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.04])
noise = np.array([0.001, -0.001, -0.001, 0.001, 0.002, -0.002])
portfolio = 0.002 + 1.5 * market + noise
dates = pd.bdate_range('2024-01-02', periods=7)
axis = pd.DataFrame({'start_at': dates[:-1], 'end_at': dates[1:]})
rows = _exposure({}, axis, portfolio, market, None, True,
                 {'minimum_observations': 3, 'hac_lag': 1, 'confidence': 0.95})
assert len(rows) == 4 and all(row['status'] == 'DEFINED' for row in rows)
assert not any(
    name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules
)
assert 'arch' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr


def _crypto_account(
    root, *, partial=False, mode="continuous", short=False, negative=False, dates=None
):
    dates = (
        dates
        if dates is not None
        else pd.date_range(
            "2024-01-01T23:55:00" if short else "2024-01-01",
            periods=3,
            freq="5min" if short else "D",
        )
    )
    capital = PortfolioCapital("crypto", 1000.0, "USDT", "linear_perpetual")

    def produce(emit):
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(
            dates[0].value,
            100.0,
            diagnostics=(
                PortfolioDiagnostic("btc_usdt", None, "oos_window_start", number=0.0),
            ),
        )
        account.fill(
            timestamp_ns=dates[0].value,
            order_id="entry",
            quantity=-20.0 if negative else 1.0,
            price=100.0,
            mark_price=100.0,
            fee=1.0,
        )
        account.mark(
            dates[1].value,
            200.0 if negative else 110.0,
            diagnostics=()
            if short
            else (
                PortfolioDiagnostic(
                    "btc_usdt",
                    None,
                    "utc_day_boundary",
                    number=float(dates[1].value // pd.Timedelta(days=1).value),
                ),
            ),
        )
        if not partial:
            account.mark(
                dates[2].value,
                210.0 if negative else 120.0,
                diagnostics=(
                    PortfolioDiagnostic("btc_usdt", None, "oos_window_end", number=0.0),
                ),
            )
        _record_execution_result(
            account,
            {
                "simulation": "comparison_fixture",
                "net_pnl": None if partial else -2201.0 if negative else 19.0,
                "created_orders": 1,
                "filled_orders": 1,
                "selection_status": "PARTIAL" if partial else "COMPLETE",
                "account_path_status": "PARTIAL" if partial else "COMPLETE",
                "account_stop_reason": "execution_unknown" if partial else None,
                "account_stop_ns": dates[1].value + 1 if partial else None,
            },
            dates[-1].value,
            210.0 if negative else 120.0,
        )

    metadata = {
        "window": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
        "sample_role": "test",
        "execution_mode": mode,
        "execution_style": "taker",
        "execution": {"position_size": 20.0, "taker_fee_bps": 0.0},
    }
    if mode == "continuous":
        write_portfolio_ledger_artifact(
            root, capital=capital, metadata=metadata, produce=produce
        )
    else:

        def episodes(emit):
            emit(
                start_ns=dates[0].value,
                stop_ns=dates[-1].value + 1,
                termination="PLANNED_END",
                stop_reasons=["planned_oos_end"],
                produce=produce,
            )

        write_crypto_episode_artifact(
            root, capital=capital, metadata=metadata, produce=episodes
        )
    return dates


@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
@pytest.mark.parametrize(
    "partial, short, negative",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_crypto_native_axis_partial_short_and_negative_equity(
    tmp_path, mode, partial, short, negative
):
    ledger = tmp_path / "ledger"
    dates = _crypto_account(
        ledger, partial=partial, mode=mode, short=short, negative=negative
    )
    ref = reference_input(
        tmp_path / "reference.parquet",
        [100.0, 105.0, 110.0],
        dates=dates,
        currency="USDT",
        clock="utc",
    )
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(ledger),
                "output": str(tmp_path / "statistics"),
                "analyses": ["benchmark_comparison", "market_exposure"],
                "comparison_benchmarks": [ref],
                "market_exposure": {
                    "hac_lag": 0,
                    "confidence": 0.95,
                    "minimum_observations": 3,
                },
            }
        )
    )
    assert "crypto_portfolio_daily.parquet" not in artifact._members
    with artifact.table("portfolio_benchmark_intervals.parquet") as rows:
        assert rows.start_at.iloc[0] == dates[0] and rows.end_at.iloc[-1] == dates[-1]
        assert not rows.episode_id.eq("all").any()
        if partial:
            assert (
                rows.status.iloc[-1] == "UNDEFINED"
                and rows.reason.iloc[-1] == "execution_unknown"
            )
        elif negative:
            assert (
                rows.portfolio_return.iloc[0] < -1 and rows.portfolio_wealth.iloc[0] < 0
            )
            assert rows.portfolio_reason.iloc[-1] == "nonpositive_start_nav"
        elif short:
            assert len(rows) == 1
            assert rows.portfolio_return.iloc[0] == pytest.approx(0.019)
            assert rows.relative_wealth.iloc[0] == pytest.approx(1.019 / 1.1)
        else:
            assert rows.portfolio_return.tolist() == pytest.approx([0.009, 10 / 1009])
            assert rows.relative_wealth.iloc[-1] == pytest.approx(1.019 / 1.1)
    with artifact.table("portfolio_benchmark_metrics.parquet") as metrics:
        if partial or negative:
            assert metrics.status.eq("UNDEFINED").all()
        if short:
            assert (
                metrics.loc[metrics.metric_id.eq("tracking_error"), "status"]
                .eq("UNDEFINED")
                .all()
            )
    with artifact.table("portfolio_market_exposure.parquet") as exposure:
        assert exposure.status.eq("UNDEFINED").all()


def test_missing_input_keeps_declared_units_but_rejects_bad_selector(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=4)
    ref = reference_input(
        tmp_path / "missing.parquet",
        [0.01, 0.02, 0.03, 0.0],
        dates=dates,
        representation="interval_simple_return",
    )
    (tmp_path / "missing.parquet").unlink()
    tables, _ = _compare(_axis([0.02, 0.01, 0.04], dates), [ref])
    assert (
        tables["portfolio_benchmark_intervals"]
        .reason.eq("reference_input_missing")
        .all()
    )
    ref["input"]["row_filters"][0][1] = "invalid_operator"
    with pytest.raises(ValueError, match="unsupported_dataset_physical_filter"):
        _compare(_axis([0.02, 0.01, 0.04], dates), [ref])


def test_constant_cash_is_explicit_zero_and_never_implicit_risk_free(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=4)
    ref = reference_input(
        tmp_path / "cash.parquet",
        [100.0, 105.0, 110.0, 115.0],
        dates=dates,
        return_basis="constant_cash",
        cost_basis="zero_cash",
    )
    axis = _axis([0.02, 0.01, 0.04], dates)
    with pytest.raises(ValueError, match="constant_cash_invalid"):
        _compare(axis, [ref])
    frame = pd.read_parquet(ref["input"]["path"])
    frame["value"] = 100.0
    frame.to_parquet(ref["input"]["path"], index=False)
    tables, _ = _compare(axis, [ref])
    assert tables["portfolio_benchmark_intervals"].benchmark_return.eq(0).all()
    with pytest.raises(ValueError, match="cash_is_not_risk_free"):
        _compare(
            axis,
            [ref],
            analyses=["market_exposure"],
            market_exposure={
                "hac_lag": 0,
                "confidence": 0.95,
                "minimum_observations": 3,
                "risk_free": ref,
            },
        )


def test_native_normalized_control_facts_are_inherited_and_conflicts_rejected(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.storage.datasets import write_dataset

    dates = pd.bdate_range("2024-01-02", periods=4)
    source = tmp_path / "control"
    facts = {
        "valuation_currency": "CNY",
        "timestamp_basis": "trading_day_open",
        "return_basis": "strategy_net",
        "cost_basis": "execution_costs",
    }
    spec = DataSpec(
        data_id="existing_control_intervals",
        frequency="1d",
        grain="asset_timestamp",
        column_roles={
            "timestamp": "timestamp",
            "end_at": "context",
            "value": "context",
            "series": "key",
        },
        primary_key=("timestamp", "series"),
        partition_columns=(),
        column_units={"value": "decimal"},
    )
    write_dataset(
        source,
        pd.DataFrame(
            {
                "timestamp": dates[:-1],
                "end_at": dates[1:],
                "value": [0.01, -0.01, 0.02],
                "series": "control",
            }
        ),
        spec=spec,
        metadata={"parameters": facts},
    )
    ref = {
        "reference_id": "control",
        "input": {"path": str(source), "row_filters": [["series", "==", "control"]]},
        "representation": "interval_simple_return",
        "start_column": "timestamp",
        "end_column": "end_at",
        "value_column": "value",
    }
    tables, metadata = _compare(
        _axis([0.02, 0.01, 0.04], dates),
        [],
        analyses=["control_comparison"],
        comparison_controls=[ref],
    )
    assert tables[
        "portfolio_benchmark_intervals"
    ].return_difference.tolist() == pytest.approx([0.01, 0.02, 0.02])
    assert metadata["comparison_method"]["control_design_matching"] == "not_established"
    ref["valuation_currency"] = "USD"
    with pytest.raises(ValueError, match="source_facts_conflict"):
        _compare(
            _axis([0.02, 0.01, 0.04], dates),
            [],
            analyses=["control_comparison"],
            comparison_controls=[ref],
        )


def test_requested_risk_free_defines_alpha_and_missing_rf_blocks_without_fallback(
    tmp_path,
):
    dates = pd.bdate_range("2024-01-02", periods=6)
    market = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
    risk_free = np.array([0.001, 0.002, 0.001, 0.003, 0.002])
    noise = np.array([0.0001, -0.0001, 0.0002, -0.0002, 0.0])
    portfolio = risk_free + 0.002 + 1.5 * (market - risk_free) + noise
    ref = reference_input(
        tmp_path / "reference.parquet",
        [*market, 0],
        dates=dates,
        representation="interval_simple_return",
    )
    rf = reference_input(
        tmp_path / "rf.parquet",
        [*risk_free, 0],
        dates=dates,
        representation="interval_simple_return",
        reference_id="risk_free",
    )
    kwargs = {
        "analyses": ["market_exposure"],
        "market_exposure": {
            "hac_lag": 1,
            "confidence": 0.95,
            "minimum_observations": 3,
            "risk_free": rf,
        },
    }
    tables, _ = _compare(_axis(portfolio, dates), [ref], **kwargs)
    rows = tables["portfolio_market_exposure"].set_index("coefficient_id")
    assert "alpha" in rows.index and "market_return_intercept" not in rows.index
    expected = np.linalg.lstsq(
        np.column_stack([np.ones(5), market - risk_free]),
        portfolio - risk_free,
        rcond=None,
    )[0]
    assert rows.loc["alpha", "value"] == pytest.approx(expected[0])
    assert rows.loc["beta", "value"] == pytest.approx(expected[1])
    assert (
        rows.planned_hypothesis_count.eq(2).all()
        and rows.defined_hypothesis_count.eq(2).all()
    )
    zero, metadata = _compare(_axis(risk_free, dates), [ref], **kwargs)
    estimates = zero["portfolio_market_exposure"].set_index("coefficient_id")
    assert estimates.loc[["alpha", "beta"], "value"].eq(0).all()
    assert estimates.loc[["alpha", "beta"], "inference_status"].eq("UNDEFINED").all()
    assert estimates.defined_hypothesis_count.eq(0).all()
    restored = read_statistics_artifact(
        write_statistics_artifact(
            tmp_path / "zero-excess",
            metadata=metadata,
            tables=zero,
            table_group="portfolio",
        )
    )
    with restored.table("portfolio_market_exposure.parquet") as rows:
        pd.testing.assert_frame_equal(rows, zero["portfolio_market_exposure"])
    invalid = zero["portfolio_market_exposure"].copy()
    invalid.loc[invalid.coefficient_id.eq("alpha"), "p_value"] = 0.0
    with pytest.raises(ValueError, match="statistics_exposure_inference_invalid"):
        write_statistics_artifact(
            tmp_path / "invalid-zero-excess",
            metadata=metadata,
            tables={"portfolio_market_exposure": invalid},
            table_group="portfolio",
        )
    (tmp_path / "rf.parquet").unlink()
    missing, _ = _compare(_axis(portfolio, dates), [ref], **kwargs)
    assert missing["portfolio_market_exposure"].status.eq("UNDEFINED").all()
    assert (
        missing["portfolio_market_exposure"].reason.eq("incomplete_paired_path").all()
    )


@pytest.mark.parametrize(
    "policy",
    [
        {"annualization_periods": -365},
        {"annualization_minimum_observations": -1},
        {"annualization_periods": True},
        {"annualization_minimum_observations": 2.5},
    ],
)
def test_comparison_only_crypto_validates_consumed_annualization_policy(
    tmp_path, policy
):
    ledger = tmp_path / "ledger"
    dates = _crypto_account(ledger)
    ref = reference_input(
        tmp_path / "reference.parquet",
        [100.0, 105.0, 110.0],
        dates=dates,
        currency="USDT",
        clock="utc",
    )
    with pytest.raises(ValueError, match="crypto_statistics_annualization_invalid"):
        crypto_statistics(
            {
                "ledger": str(ledger),
                "output": str(tmp_path / "statistics"),
                "analyses": ["benchmark_comparison"],
                "comparison_benchmarks": [ref],
                "statistics": policy,
            }
        )


@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
def test_crypto_non_daily_known_endpoints_preserve_actual_25hour_interval(
    tmp_path, mode
):
    dates = pd.DatetimeIndex(
        [
            "2024-01-01T12:00:00.000000123",
            "2024-01-01T13:00:00.000000123",
            "2024-01-02T13:00:00.000000123",
        ]
    )
    ledger = tmp_path / "ledger"
    _crypto_account(ledger, mode=mode, short=True, dates=dates)
    ref = reference_input(
        tmp_path / "reference.parquet",
        [100.0, 105.0, 110.0],
        dates=dates,
        currency="USDT",
        clock="utc",
    )
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(ledger),
                "output": str(tmp_path / "statistics"),
                "analyses": ["benchmark_comparison"],
                "comparison_benchmarks": [ref],
            }
        )
    )
    with artifact.table("portfolio_benchmark_intervals.parquet") as rows:
        assert len(rows) == 1
        assert rows.start_at.iloc[0] == dates[0] and rows.end_at.iloc[0] == dates[-1]
        assert rows.portfolio_return.iloc[0] == pytest.approx(0.019)
        assert rows.benchmark_return.iloc[0] == pytest.approx(0.1)
        assert rows.status.eq("DEFINED").all()
    with artifact.table("portfolio_benchmark_metrics.parquet") as metrics:
        row = metrics.set_index("metric_id").loc["tracking_error"]
        assert row.status == "UNDEFINED" and row.reason == "incomplete_daily_axis"


@pytest.mark.parametrize("mutation", ["filter", "value", "timestamp"])
def test_missing_file_cannot_hide_schema_known_column_errors(tmp_path, mutation):
    dates = pd.bdate_range("2024-01-02", periods=3)
    ref = reference_input(
        tmp_path / "missing.parquet", [100.0, 105.0, 110.0], dates=dates
    )
    (tmp_path / "missing.parquet").unlink()
    if mutation == "filter":
        ref["input"]["row_filters"][0][0] = "not_a_column"
    else:
        ref[f"{mutation}_column"] = "not_a_column"
    with pytest.raises(ValueError, match="column.*(found|missing)"):
        _compare(_axis([0.01, 0.02], dates), [ref])


@pytest.mark.parametrize(
    "cadence", ["no_calendar", "weekly", "omitted_session", "holiday"]
)
def test_native_calendar_evidence_localizes_daily_statistics(tmp_path, cadence):
    sessions = pd.bdate_range("2024-01-02", periods=30)
    if cadence == "weekly":
        dates = sessions[::5][:6]
    elif cadence == "omitted_session":
        dates = sessions[:8].delete(3)
    elif cadence == "holiday":
        # The explicit market calendar, not elapsed days, proves this long closure.
        sessions = sessions.delete([3, 4, 5, 6, 7])
        dates = sessions[:7]
    else:
        dates = sessions[:7]
    ledger = tmp_path / "ledger"
    window = _cash_ledger(ledger, dates=dates)
    market = np.array([0.01, -0.02, 0.03, 0.005, -0.01])[: len(dates) - 2]
    reference = reference_input(
        tmp_path / "reference.parquet",
        [*market, 0.0],
        dates=dates[1:],
        representation="interval_simple_return",
    )
    config = {
        "ledgers": [str(ledger)],
        "window": window,
        "analyses": [
            "performance",
            "benchmark_comparison",
            "market_exposure",
            "sharpe_inference",
            "cscv",
        ],
        "statistics": {"cscv_blocks": 4},
        "comparison_benchmarks": [reference],
        "market_exposure": {
            "minimum_observations": 3,
            "hac_lag": 1,
            "confidence": 0.95,
        },
        "output": str(tmp_path / "statistics"),
    }
    if cadence != "no_calendar":
        calendar = tmp_path / "calendar.parquet"
        pd.DataFrame(
            {"calendar_date": sessions, "source_market_type": 1, "is_trading_day": True}
        ).to_parquet(calendar, index=False)
        config["inputs"] = {
            "calendar": {
                "path": str(calendar),
                "spec": {
                    "data_id": "calendar",
                    "frequency": "1d",
                    "grain": "asset_timestamp",
                    "column_roles": {
                        "calendar_date": "timestamp",
                        "source_market_type": "key",
                        "is_trading_day": "context",
                    },
                    "primary_key": ["calendar_date", "source_market_type"],
                    "partition_columns": [],
                },
            }
        }
    artifact = read_statistics_artifact(portfolio_statistics(config))
    with artifact.table("portfolio_benchmark_intervals.parquet") as rows:
        assert rows.status.eq("DEFINED").all()
        assert rows.relative_wealth.notna().all()
    with artifact.table("portfolio_benchmark_metrics.parquet") as rows:
        annual = rows.loc[
            rows.metric_id.isin(
                ["annualized_return_difference", "tracking_error", "information_ratio"]
            )
        ]
        assert annual.status.eq(
            "DEFINED" if cadence == "holiday" else "UNDEFINED"
        ).all()
        if cadence != "holiday":
            assert annual.reason.eq("incomplete_daily_axis").all()
    with artifact.table("portfolio_performance.parquet") as rows:
        assert rows.loc[rows.metric_id.eq("terminal_nav"), "value"].iloc[0] == 1000
        annual = rows.loc[
            rows.metric_id.isin(["cagr", "annualized_volatility", "sharpe"])
        ]
        if cadence != "holiday":
            assert annual.reason.eq("incomplete_daily_axis").all()
        else:
            assert annual.loc[annual.metric_id.eq("cagr"), "value"].iloc[0] == 0
    with artifact.table("portfolio_robustness.parquet") as rows:
        if cadence != "holiday":
            assert rows.reason.eq("incomplete_daily_axis").all()
    with artifact.table("portfolio_market_exposure.parquet") as rows:
        if cadence != "holiday":
            assert rows.reason.eq("incomplete_daily_axis").all()
            assert rows.inference_status.isin(["UNDEFINED", "NOT_APPLICABLE"]).all()
        else:
            estimates = rows.loc[
                rows.coefficient_id.isin(["market_return_intercept", "beta"])
            ]
            assert estimates.status.eq("DEFINED").all()
            assert estimates.value.eq(0).all()
            assert estimates.inference_status.eq("UNDEFINED").all()
            assert estimates.inference_reason.eq("nonfinite_inference").all()
            assert (
                estimates[["t_statistic", "p_value", "fdr_adjusted_p_value"]]
                .isna()
                .all()
                .all()
            )
            assert estimates.standard_error.eq(0).all()
            assert estimates.defined_hypothesis_count.eq(0).all()
            assert estimates.planned_hypothesis_count.eq(2).all()
            assert (
                rows.loc[
                    rows.coefficient_id.isin(["correlation", "r_squared"]), "status"
                ]
                .eq("UNDEFINED")
                .all()
            )
