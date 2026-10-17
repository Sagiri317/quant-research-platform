"""Native numeric boundaries from the HEAD270 audit."""

from contextlib import closing, nullcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

from quant_research_platform.research.statistics import (
    cscv_pbo,
    fama_macbeth_period,
    hac_mean,
)


def test_native_validation_rank_tie_uses_unscaled_average_ranks():
    from quant_research_platform.research.models.trial_selection import (
        evaluate_model_trial_metrics_from_chunks,
        select_best_validation_trial,
    )

    rows = []
    for index, predictions in enumerate(((4, 3, 2, 1, 5), (5, 1, 2, 3, 4))):
        frame = pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2024-01-02"),
                "asset_id": list("abcde"),
                "label": np.arange(1.0, 6.0),
                "prediction": predictions,
            }
        )
        metrics = evaluate_model_trial_metrics_from_chunks(
            (frame,),
            model_id=f"candidate_{index}",
            label_column="label",
            prediction_column="prediction",
        )
        assert metrics["validation_rankic"] == 0.0
        rows.append(metrics)
    assert (
        select_best_validation_trial(
            pd.DataFrame(rows),
            trial_id_column="model_id",
            selection_metric="validation_rankic",
            maximize=True,
            validation_metric_columns=("validation_rankic",),
            test_metric_columns=(),
        )
        == "candidate_0"
    )


def test_cscv_true_permutation_ties_remain_undefined():
    values = np.tile([[0.1, 0.0], [-0.1, 0.1], [0.0, -0.1], [0.0, 0.0]], (10, 1))
    result = cscv_pbo(values, slice_count=10)
    assert result.combination_count == 252
    assert result.status == "UNDEFINED"
    assert result.reason == "candidate_ranking_tie"


@pytest.mark.parametrize("reverse", (False, True))
def test_hac_cancellation_preserves_representable_mean(reverse):
    values = np.array([1e200] * 10 + [1.0] + [-1e200] * 10)
    values = values[::-1] if reverse else values
    actual = hac_mean(values, max_lags=19)
    assert actual.mean == float(Fraction(1, 21))
    assert actual.observation_count == 21
    assert actual.standard_error > 0


@pytest.mark.parametrize("magnitude", (10.0, 1e308))
@pytest.mark.parametrize("weighted", (False, True))
def test_multi_factor_fmb_retains_orthogonal_small_coefficient(magnitude, weighted):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": [str(index) for index in range(100)],
            "x": np.repeat([0.0, 1.0], 50),
            "z": np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 20),
            "weight": 1.0,
        }
    )
    frame["label"] = np.where(frame.x == 0, magnitude, frame.z + 3)
    frame["weight"] = np.where(frame.x == 0, 2.0, 1.0)
    expected_warning = (
        pytest.warns(
            SingularMatrixWarning,
            match=(
                r"^The design matrix is rank-deficient\. "
                r"The model parameters are not uniquely determined\.$"
            ),
        )
        if magnitude == 10.0
        else nullcontext()
    )
    with expected_warning:
        actual = fama_macbeth_period(
            frame,
            return_column="label",
            factor_columns=("x", "z"),
            weight_column="weight" if weighted else None,
        )
    np.testing.assert_allclose(
        actual.coefficients,
        [magnitude, 3 - magnitude, 1 / 3 if weighted else 0.5],
        rtol=1e-14,
        atol=0,
    )


def test_multi_factor_nonunit_weighted_exact_fit():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("abcdef"),
            "x": [0.0, 0, 1, 1, 2, 2],
            "z": [0.0, 1, 0, 2, 1, 3],
            "weight": [1.0, 2, 3, 4, 5, 6],
        }
    )
    frame["label"] = 2 + 3 * frame.x - frame.z / 2
    actual = fama_macbeth_period(
        frame, return_column="label", factor_columns=("x", "z"), weight_column="weight"
    )
    np.testing.assert_allclose(actual.coefficients, [2, 3, -0.5], rtol=1e-14)


@pytest.mark.parametrize("half_life", (1, 2))
def test_weighted_scaler_finite_population_bound(half_life):
    from quant_research_platform.contracts.models import TrainingWeighting
    from quant_research_platform.research.models.scaler import (
        ModelMatrixScaler,
        equal_timestamp_sample_weights,
    )

    magnitude = np.finfo(float).max
    frame = pd.DataFrame({"x": [-magnitude, magnitude, -magnitude, magnitude]})
    weights = equal_timestamp_sample_weights(
        pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
        TrainingWeighting(half_life_periods=half_life),
        trading_sessions=pd.date_range("2024-01-02", "2024-01-03"),
    )
    scaler = ModelMatrixScaler.fit(frame, sample_weight=weights)
    assert scaler.mean_ == (0.0,)
    assert scaler.scale_[0] == pytest.approx(magnitude, rel=1e-15)
    np.testing.assert_allclose(scaler.transform(frame).x, [-1, 1, -1, 1], rtol=1e-15)


@pytest.mark.parametrize("index_start", [None, 0, 1000])
def test_standardized_residual_keeps_subnormal_half_units(index_start):
    from quant_research_platform.research.features.transforms import (
        _industry_size_residuals,
    )

    tiny = np.nextafter(0.0, 1.0)
    target = pd.Series(np.tile([0.0, tiny], 50))
    industries = pd.Series(["industry"] * 100)
    sizes = pd.Series(np.repeat(np.arange(50, dtype=float), 2))
    if index_start is None:
        target, sizes = target.to_numpy(), sizes.to_numpy()
    else:
        target.index = sizes.index = pd.RangeIndex(index_start, index_start + 100)
    actual = _industry_size_residuals(target, industries, sizes, rescale=True)
    np.testing.assert_array_equal(actual, np.tile([-1.0, 1.0], 50))
    raw = _industry_size_residuals(target, industries, sizes, rescale=False)
    np.testing.assert_array_equal(raw, np.zeros(100))


def test_adjacent_positive_price_labels_keep_distinct_model_targets(tmp_path):
    from quant_research_platform.research.datasets.assembler import (
        _a_share_model_target,
    )
    from quant_research_platform.research.labels.forward_returns import (
        add_forward_return_labels,
    )

    dates = pd.bdate_range("2024-01-02", periods=22)
    frame = pd.DataFrame(
        {
            "asset_id": np.repeat(["a", "b"], 22),
            "timestamp": np.tile(dates, 2),
            "adj_open": 3.0,
            "filling_flag": 0,
            "is_trading": True,
        }
    )
    exits = [np.nextafter(3.0, np.inf), np.nextafter(np.nextafter(3.0, np.inf), np.inf)]
    for asset, price in zip(("a", "b"), exits, strict=True):
        frame.loc[
            (frame.asset_id == asset) & (frame.timestamp == dates[-1]), "adj_open"
        ] = price
    labeled = add_forward_return_labels(frame, horizons=(20,), sessions=dates)
    labeled.to_parquet(tmp_path / "labels.parquet", index=False)
    restored = pd.read_parquet(tmp_path / "labels.parquet")
    first = restored.loc[restored.timestamp == dates[0]].assign(eligible=True)
    expected = [float((Fraction(value) - 3) / 3) for value in exits]
    np.testing.assert_allclose(first.forward_return_20d, expected, rtol=1e-14, atol=0)
    assert 0 < first.forward_return_20d.iloc[0] < first.forward_return_20d.iloc[1]
    np.testing.assert_allclose(_a_share_model_target(first), [-1, 1], rtol=1e-15)


@pytest.mark.parametrize("capital_value", (1000.0, 1e16))
def test_perpetual_published_pnl_keeps_small_cost_and_funding(tmp_path, capital_value):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        FundingEvent,
        PortfolioCapital,
    )
    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
    )

    capital = PortfolioCapital("fee", capital_value, "USDT", "linear_perpetual")
    metadata = {
        "sample_role": "validation",
        "window": {"start": "1970-01-01", "end": "1970-01-02"},
    }

    def produce(emit):
        economics = PortfolioEconomics(capital, "asset", emit)
        economics.fill(
            timestamp_ns=1,
            order_id="open",
            quantity=1.0,
            price=100.0,
            mark_price=100.0,
            fee=1.0,
        )
        economics.funding(FundingEvent(2, 0.01, 100.0), mark_price=economics.last_mark)
        economics.funding(FundingEvent(3, 0.01, 100.0), mark_price=economics.last_mark)
        summary = economics.mark(4, 100.0)
        assert summary["net_pnl"] == -3.0
        assert summary["gross_pnl"] == 0.0

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    result = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    assert result["net_pnl"] == -3.0
    assert result["gross_pnl"] == 0.0
    assert result["fees"] == 1.0
    assert result["funding"] == -2.0


@pytest.mark.parametrize("initial_nav", (100.0, 5e-13))
@pytest.mark.parametrize("trade", (False, True))
def test_public_builder_small_nav_cash_and_fee_publish(
    tmp_path, monkeypatch, initial_nav, trade
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        summarize_portfolio_ledger,
        write_portfolio_ledger_artifact,
    )
    from quant_research_platform.contracts.portfolios import (
        ExecutionAssumptions,
        PortfolioCapital,
    )
    from tests.research import test_portfolio_production as fixture

    frames, fees, window = fixture._inputs()
    frames["dataset"]["eligible"] = (
        frames["dataset"].asset_id.eq("b") if trade else False
    )
    frames["panel"][["open", "close"]] = 1.0
    fees["rate"] = 0.0
    fees.loc[fees.fee_code.eq("BROKER_COMMISSION") & fees.side.eq("buy"), "rate"] = 0.2
    assumptions = ExecutionAssumptions(initial_nav, 0.0, 0.0, 1.0, (1.0,))
    monkeypatch.setattr(fixture, "ExecutionAssumptions", lambda **unused: assumptions)
    chunks, _calls = fixture._build_producer(
        tmp_path,
        monkeypatch,
        frames=frames,
        fee_frame=fees,
        output_window=window,
        typed_batches=True,
    )
    capital = PortfolioCapital(
        "synthetic_quantile_equal_weight", initial_nav, "CNY", "cash_equity"
    )
    metadata = {
        "sample_role": "validation",
        "window": {"start": str(window.start), "end": str(window.end)},
    }

    def produce(emit):
        with closing(chunks):
            for batch in chunks:
                emit(batch)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    ledger = read_portfolio_ledger_artifact(root)
    batches = list(ledger.iter_chunks())
    assert len(batches) == 2
    for batch in batches:
        assert all(
            row.value >= 0 for row in batch.valuations if row.asset_id == "__cash__"
        )
    summary = summarize_portfolio_ledger(ledger.capital, iter(batches))
    expected_fee_ratio = 0.2 / 1.2 if trade else 0.0
    assert summary["fees"] / initial_nav == pytest.approx(expected_fee_ratio, abs=1e-15)
    assert summary["net_pnl"] / initial_nav == pytest.approx(
        -expected_fee_ratio, abs=1e-15
    )
    assert summary["gross_pnl"] / initial_nav == pytest.approx(0.0, abs=1e-15)
    assert summary["nav"] / initial_nav == pytest.approx(
        1 - expected_fee_ratio, abs=1e-15
    )
    if trade:
        assert sum(
            row.notional for batch in batches for row in batch.trades
        ) / initial_nav == pytest.approx(1 / 1.2, abs=1e-15)


def test_factor_inference_counts_original_dates_with_no_mature_or_finite_factor(
    tmp_path,
):
    from quant_research_platform.artifacts.statistics import (
        read_statistics_artifact,
        write_statistics_artifact,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.statistics_production import (
        build_a_share_factor_statistics_tables,
    )
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.research_specs import make_demo_research_dataset_spec
    from tests.helpers.statistics_artifacts import (
        statistics_config,
        statistics_metadata,
    )

    dates = pd.bdate_range("2023-01-02", periods=26)
    frame = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=str(asset),
                timestamp=timestamp,
                eligible=True,
                factor=(asset - 50) / 50,
                total_market_cap=1.0,
                forward_return_20d=0.1
                + 0.001 * (index + 1) * asset
                + 0.01 * np.sin(asset + index),
                forward_return_20d_available_at=timestamp,
            )
            for index, timestamp in enumerate(dates)
            for asset in range(100)
        ]
    )
    frame.loc[frame.timestamp.eq(dates[8]), "factor"] = np.nan
    frame.loc[frame.timestamp.eq(dates[12]), "forward_return_20d_available_at"] = dates[
        -1
    ] + pd.offsets.BDay(1)
    spec = make_demo_research_dataset_spec(
        data_id="factor_clock",
        factor_columns=("factor",),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
    )
    dataset_path = tmp_path / "factor-input"
    write_dataset(dataset_path, frame, spec=spec)
    config = statistics_config(factors=["factor"])
    config["evaluation"]["primary_label"] = "forward_return_20d"
    config["statistics"]["fama_macbeth_core_factors"] = []
    window = TimeWindow(dates[0].date(), dates[-1].date())
    tables = build_a_share_factor_statistics_tables(
        dataset_path=dataset_path,
        dataset_spec=spec,
        config=config,
        time_window=window,
    )
    metadata = statistics_metadata(window)
    root = write_statistics_artifact(
        tmp_path / "statistics", metadata=metadata, tables=tables, table_group="factor"
    )
    artifact = read_statistics_artifact(root)
    for member, effective in (
        ("factor_diagnostics.parquet", "observation_count"),
        ("fama_macbeth.parquet", "period_count"),
    ):
        with artifact.table(member) as actual:
            assert actual.input_date_count.eq(26).all()
            assert actual[effective].eq(24).all()
            assert (actual.input_date_count - actual[effective]).eq(2).all()


def test_valid_observation_clock_keeps_order_under_date_gaps():
    from quant_research_platform.research.statistics_production import (
        _prediction_summary,
    )

    values = np.sin(np.arange(25)) / 2
    regular = pd.DataFrame(
        {"__prediction_alpha__": values}, index=pd.bdate_range("2023-01-02", periods=25)
    )
    irregular = regular.copy()
    irregular.index = irregular.index + pd.to_timedelta(np.arange(25) ** 2, unit="D")
    options = dict(
        prediction_id="fixture",
        inference_blocker=None,
        max_lags=19,
        block_size=20,
        repetitions=32,
        seed=42,
    )
    pd.testing.assert_frame_equal(
        _prediction_summary(regular, **options),
        _prediction_summary(irregular, **options),
    )


@pytest.mark.parametrize("weighted", (False, True))
def test_ordinary_exact_fit_with_zero_coefficient_keeps_numeric_path(
    monkeypatch, weighted
):
    from quant_research_platform.research import statistics

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2024-01-02"),
            "asset_id": list("abcdefgh"),
            "x": np.repeat([0.0, 1.0, 2.0, 3.0], 2),
            "z": np.tile([-1.0, 1.0], 4),
            "weight": np.repeat([1.0, 2.0, 3.0, 4.0], 2),
        }
    )
    frame["label"] = 2.0 + frame.x

    def unexpected(*args):
        pytest.fail("Ordinary perfect fits do not need exact regression.")

    monkeypatch.setattr(statistics, "_exact_period_regression", unexpected)
    actual = statistics.fama_macbeth_period(
        frame,
        return_column="label",
        factor_columns=("x", "z"),
        weight_column="weight" if weighted else None,
    )
    np.testing.assert_allclose(actual.coefficients, [2, 1, 0], rtol=1e-14, atol=1e-14)


def test_test_set_keeps_rankic_by_and_explicit_exploratory_exclusion():
    from copy import deepcopy

    from quant_research_platform.research.protocols import a_share
    from quant_research_platform.research.statistics_production import (
        _paired_group_rankic,
    )
    from tests.helpers.research_configuration import A_SHARE_MODEL_ALGORITHMS
    from tests.helpers.statistics_artifacts import statistics_config

    manifest = statistics_config(sample_role="test", selection_group="quarterly")
    manifest["statistics"]["bootstrap_repetitions"] = 199
    families = tuple(A_SHARE_MODEL_ALGORITHMS)
    dates = pd.bdate_range("2024-01-02", periods=64)
    rng = np.random.default_rng(918)
    global_losses = pd.DataFrame(
        [
            dict(
                timestamp=date,
                family_id=family,
                asset_coverage_id="a" * 64,
                observation_count=100,
                sample_role="test",
                status="DEFINED",
                loss=rng.normal(0, 0.1),
            )
            for family in families
            for date in dates
        ]
    )
    quarterly = global_losses.copy()
    quarterly["loss"] += rng.normal(-0.01, 0.04, len(quarterly))

    def compare(value):
        return _paired_group_rankic(
            quarterly.assign(
                **{"global": -global_losses.loss, "quarterly": -quarterly.loss}
            ),
            selection_group="quarterly",
            family_ids=families,
            config=value,
        )

    actual = compare(manifest)
    assert actual.fdr_method.eq("BY").all()
    assert actual.fdr_adjusted_p_value.notna().all()
    selected = deepcopy(manifest)
    selected["sample_role"] = "validation"
    selected["selection_is_exploratory"] = True
    assert a_share._inference_blocker(selected) == "exploratory_selection_excluded"
    selected["selection_is_exploratory"] = False
    selected["sample_role"] = "test"
    selected["feature_recipe"] = {"role": "robustness"}
    assert a_share._inference_blocker(selected) == "feature_recipe_robustness_excluded"


@pytest.mark.parametrize("scale", (1.0, 1e-200, 1e-308))
@pytest.mark.parametrize("weighted", (False, True))
def test_fmb_recovers_before_converting_polluted_small_factor_slope(scale, weighted):
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [str(index) for index in range(100)],
            "x": np.repeat([0.0, 1.0], 50),
            "z": np.tile(np.arange(1.0, 6.0) * scale, 20),
            "label": np.repeat([1e308, 3.0], 50),
            "weight": np.repeat([2.0, 1.0], 50),
        }
    )
    actual = fama_macbeth_period(
        frame,
        return_column="label",
        factor_columns=("x", "z"),
        weight_column="weight" if weighted else None,
    )
    np.testing.assert_array_equal(
        actual.coefficients,
        [1e308, float(Fraction(3) - Fraction(1e308)), 0.0],
    )
    assert actual.white_reason == "white_test_undefined"


def test_white_constant_squared_residual_is_undefined_without_backend_fit(monkeypatch):
    from quant_research_platform.research import statistics

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": [str(index) for index in range(100)],
            "z": np.tile(np.arange(-2.0, 3.0), 20),
        }
    )
    frame["label"] = np.where(np.arange(100) < 50, 1e308, frame.z + 3.0)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Constant squared residuals have no defined White auxiliary R-squared."
        )

    monkeypatch.setattr(statistics, "OLS", forbidden)
    actual = statistics.fama_macbeth_period(
        frame,
        return_column="label",
        factor_columns=("z",),
    )
    assert actual.coefficients["z"] == 0.5
    assert actual.white_p_value is None
    assert actual.white_reason == "white_test_undefined"


@pytest.mark.parametrize(
    "capital_units,current,target,error",
    (
        (1, [], [], None),
        (1, [1.0], [1.0], None),
        (1, [2.0], [1.0], "execution_weights_sum_invalid"),
        (1, [1.0], [2.0], "execution_weights_sum_invalid"),
        (
            2,
            [0.26, 0.26, 0.48],
            [0.26, 0.26, 0.48],
            "execution_cash_after_sells_invalid",
        ),
    ),
)
def test_subnormal_capital_preserves_cash_or_rejects_unrepresentable_state(
    capital_units, current, target, error
):
    from datetime import date

    from quant_research_platform.contracts.portfolios import (
        ExecutionAssumptions,
        TradeCostComponents,
    )
    from quant_research_platform.research.portfolios.accounting import (
        execute_continuous_target,
    )

    nav = capital_units * np.nextafter(0.0, 1.0)
    assets = [str(index) for index in range(len(current))]
    expected = pytest.raises(ValueError, match=f"^{error}$") if error else nullcontext()
    with expected:
        actual = execute_continuous_target(
            pd.Series(current, index=assets, dtype=float),
            pd.Series(target, index=assets, dtype=float),
            nav=nav,
            average_daily_volume=pd.Series(nav, index=assets),
            can_enter=pd.Series(True, index=assets),
            can_exit=pd.Series(True, index=assets),
            execution_date=date(2023, 1, 3),
            cost_for_trade=lambda day, asset, side, amount, adv, settings: (
                TradeCostComponents(amount, 0.0, 0.0, 0.0, 0.0, 0.0, "a" * 64)
            ),
            cost_totals_for_buys=lambda assets, amounts, adv: np.zeros(len(assets)),
            assumptions=ExecutionAssumptions(nav, 0.0, 0.0, 1.0, (1.0,)),
            cost_source_identity="a" * 64,
        )
        assert actual.nav_after == nav
        assert actual.pre_trade_cash == (nav if not current else 0.0)
        assert actual.post_trade_cash == actual.pre_trade_cash
