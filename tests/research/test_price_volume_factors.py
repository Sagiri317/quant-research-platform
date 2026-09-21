from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.factors.engine import (
    calculate_factors_for_panel,
    factor_lookback_bars,
    factor_panel_input_columns,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    price_volume_factor_definitions,
)
from tests.helpers.research_configuration import CORE6_FACTORS, PV25_FACTORS

UNIQUE_FACTORS = (
    "momentum_20d",
    "momentum_60d",
    "upside_volatility_20d",
    "min_return_20d",
    "kurtosis_60d",
    "avg_amount_20d",
    "amount_change_20d",
    "volume_change_20d",
    "turnover_change_20d",
    "amount_volatility_20d",
    "volume_shock_20d",
    "return_volume_corr_20d",
    "return_turnover_corr_20d",
    "return_amount_imbalance_20d",
    "zero_amount_ratio_20d",
    "log_total_market_cap",
)


def test_price_volume_selections_use_one_definition_and_minimum_inputs():
    binding = dict(market_id=MarketId("a_share"), required_input_data_id="price_volume")
    standard = price_volume_factor_definitions(**binding, factor_columns=PV25_FACTORS)
    assert factor_lookback_bars(standard) == 252
    assert tuple(item.output_column for item in standard) == PV25_FACTORS
    core = price_volume_factor_definitions(**binding, factor_columns=CORE6_FACTORS)
    assert tuple(item.output_column for item in core) == CORE6_FACTORS
    reversal = price_volume_factor_definitions(
        **binding, factor_columns=("reversal_5d",)
    )
    assert factor_lookback_bars(reversal) == 5
    assert reversal[0].required_windows == (5,)
    x = np.arange(1.0, 301.0)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=len(x)),
            "adj_close": x,
            "period_return": np.sin(x) * 0.01,
            "high": x + 2.0,
            "low": x - 0.5,
            "close": x,
            "amount": x * 10.0,
            "volume": x + 10.0,
            "float_market_cap": x * 1_000.0,
            "total_market_cap": x * 2_000.0,
        }
    )
    names = (*PV25_FACTORS, *UNIQUE_FACTORS)
    definitions = price_volume_factor_definitions(**binding, factor_columns=names)
    assert len(definitions) == 41
    window = TimeWindow(
        panel.timestamp.iloc[280].date(), panel.timestamp.iloc[-1].date()
    )
    mixed = calculate_factors_for_panel(
        panel, definitions=definitions, output_window=window
    )
    for definition in definitions:
        minimum = panel.loc[:, factor_panel_input_columns((definition,))]
        single = calculate_factors_for_panel(
            minimum, definitions=(definition,), output_window=window
        )
        pd.testing.assert_frame_equal(
            single, mixed.loc[:, single.columns], check_exact=True
        )
    reverse = calculate_factors_for_panel(
        panel.iloc[::-1], definitions=definitions[::-1], output_window=window
    )
    pd.testing.assert_frame_equal(
        reverse.sort_values("timestamp").reset_index(drop=True),
        mixed.loc[:, reverse.columns],
        check_exact=True,
    )
    assert next(
        item for item in definitions if item.output_column == "momentum_252_20"
    ) == next(item for item in standard if item.output_column == "momentum_252_20")


def test_price_volume_unique_math_full_windows_endpoints_and_nulls():
    x = np.arange(1.0, 62.0)
    returns = np.resize(np.array([-0.02, 0.01, 0.03, -0.01, 0.0]), len(x))
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=len(x)),
            "adj_close": x,
            "period_return": returns,
            "amount": x,
            "volume": 2.0 * x,
            "float_market_cap": 1_000.0,
            "close": 10.0,
            "total_market_cap": 10_000.0,
        }
    )
    result = calculate_price_volume_factors(panel, factor_columns=UNIQUE_FACTORS)
    last = result.iloc[-1]
    flow, ret = x[-20:], returns[-20:]
    centered = returns[-60:] - returns[-60:].mean()
    variance = np.mean(centered**2)
    kurtosis = (
        59.0 / (58.0 * 57.0) * (61.0 * (np.mean(centered**4) / variance**2 - 3.0) + 6.0)
    )
    expected = {
        "momentum_20d": 61.0 / 41.0 - 1.0,
        "momentum_60d": 60.0,
        "upside_volatility_20d": np.sqrt((0.01**2 + 0.03**2) / 5.0),
        "min_return_20d": -0.02,
        "kurtosis_60d": kurtosis,
        "avg_amount_20d": 51.5,
        "amount_change_20d": 61.0 / 41.0 - 1.0,
        "volume_change_20d": 61.0 / 41.0 - 1.0,
        "turnover_change_20d": 61.0 / 41.0 - 1.0,
        "amount_volatility_20d": np.sqrt(665.0 / 19.0),
        "volume_shock_20d": 61.0 / 50.5 - 1.0,
        "return_volume_corr_20d": np.sum((ret - ret.mean()) * (flow - flow.mean()))
        / np.sqrt(np.sum((ret - ret.mean()) ** 2) * np.sum((flow - flow.mean()) ** 2)),
        "return_turnover_corr_20d": np.sum((ret - ret.mean()) * (flow - flow.mean()))
        / np.sqrt(np.sum((ret - ret.mean()) ** 2) * np.sum((flow - flow.mean()) ** 2)),
        "return_amount_imbalance_20d": np.sum(ret * flow) / np.sum(np.abs(ret) * flow),
        "zero_amount_ratio_20d": 0.0,
        "log_total_market_cap": np.log(10_000.0),
    }
    for name, value in expected.items():
        assert last[name] == pytest.approx(value), name
    scaled = calculate_price_volume_factors(
        panel.assign(period_return=returns * 1e200), factor_columns=("kurtosis_60d",)
    )
    assert scaled.loc[60, "kurtosis_60d"] == pytest.approx(kurtosis)
    for name in UNIQUE_FACTORS:
        if name != "log_total_market_cap":
            first = (
                60
                if name == "momentum_60d"
                else 59
                if name == "kurtosis_60d"
                else 20
                if name
                in {
                    "momentum_20d",
                    "amount_change_20d",
                    "volume_change_20d",
                    "turnover_change_20d",
                    "volume_shock_20d",
                }
                else 19
            )
            assert result[name].iloc[:first].isna().all(), name
            assert pd.notna(result[name].iloc[first]), name
    missing = panel.copy()
    missing.loc[50, ["amount", "volume", "period_return", "adj_close"]] = np.nan
    missing_result = calculate_price_volume_factors(
        missing, factor_columns=UNIQUE_FACTORS
    )
    assert (
        missing_result.loc[
            60,
            [
                "zero_amount_ratio_20d",
                "volume_shock_20d",
                "kurtosis_60d",
                "return_amount_imbalance_20d",
            ],
        ]
        .isna()
        .all()
    )
    assert missing_result.loc[60, "momentum_20d"] == result.loc[60, "momentum_20d"]
    constant = panel.assign(
        amount=0.0, volume=0.0, period_return=0.0, total_market_cap=0.0
    )
    zero = calculate_price_volume_factors(constant, factor_columns=UNIQUE_FACTORS).iloc[
        -1
    ]
    assert (
        zero[
            [
                "volume_shock_20d",
                "volume_change_20d",
                "turnover_change_20d",
                "amount_change_20d",
                "return_volume_corr_20d",
                "return_turnover_corr_20d",
                "return_amount_imbalance_20d",
                "kurtosis_60d",
                "log_total_market_cap",
            ]
        ]
        .isna()
        .all()
    )
    assert zero["zero_amount_ratio_20d"] == 1.0
    assert zero["amount_volatility_20d"] == 0.0
    assert zero["upside_volatility_20d"] == 0.0
    nonpositive = panel.assign(amount=1.0)
    nonpositive.loc[60, "amount"] = -2.0
    observed = calculate_price_volume_factors(
        nonpositive, factor_columns=("zero_amount_ratio_20d",)
    )
    assert observed.loc[60, "zero_amount_ratio_20d"] == 1.0 / 20.0


def test_price_volume_rejects_removed_names_before_calculation():
    for name in (
        "mom_20d",
        "rev_1d",
        "vol_20d",
        "skew_60d",
        "kurt_60d",
        "ret_amount_corr_20d",
        "amount_chg_20d",
        "intraday_range_20d",
        "zero_volume_ratio_20d",
        "momentum_120d",
    ):
        with pytest.raises(ValueError, match="unknown_price_volume_factor"):
            price_volume_factor_definitions(
                market_id=MarketId("a_share"),
                required_input_data_id="price_volume",
                factor_columns=(name,),
            )
        with pytest.raises(ValueError, match="unknown_price_volume_factor"):
            calculate_price_volume_factors(
                pd.DataFrame(), factor_columns=("reversal_5d", name)
            )


@pytest.mark.parametrize("size", [20, 60])
@pytest.mark.parametrize("scale", [1.0, 1e-10, 1e10])
def test_price_volume_skew_preserves_native_single_window_rounding(size, scale):
    values = np.random.default_rng(20260910).normal(size=320) * scale
    values[200] = np.nan
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    panel = pd.DataFrame(
        {"asset_id": "asset", "timestamp": dates, "period_return": values}
    )
    original = panel.copy(deep=True)
    actual = calculate_price_volume_factors(
        panel,
        factor_columns=(f"skewness_{size}d",),
        output_window=TimeWindow(dates[size - 1].date(), dates[-1].date()),
    )
    expected = pd.Series(
        [
            pd.Series(values[start : start + size]).skew(skipna=False)
            for start in range(len(values) - size + 1)
        ],
        name=f"skewness_{size}d",
    )
    pd.testing.assert_series_equal(
        actual[f"skewness_{size}d"], expected, check_exact=True
    )
    pd.testing.assert_frame_equal(panel, original, check_exact=True)


@pytest.mark.parametrize("value", [0.0, 0.01, -0.125])
@pytest.mark.parametrize("count", [1, 59, 60, 61])
def test_standardized_moments_require_nonzero_variance(value, count, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=count),
            "period_return": value,
        }
    )
    names = ("skewness_60d", "kurtosis_60d", "volatility_60d")
    result = calculate_price_volume_factors(panel, factor_columns=names)
    # Both standardized moments divide by a power of the zero second moment.
    assert result.skewness_60d.isna().all()
    assert result.kurtosis_60d.isna().all()
    assert result.volatility_60d.iloc[:59].isna().all()
    if count >= 60:
        assert result.volatility_60d.iloc[-1] == 0.0
    path = tmp_path / "moments.parquet"
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), path)
    restored = pq.read_table(path).to_pandas()
    pd.testing.assert_frame_equal(restored, result)


@pytest.mark.parametrize("missing", [None, 0, "all"])
@pytest.mark.parametrize("scale", [1.0, 1e-10, 1e10])
def test_skewness_independent_central_moment_oracle(missing, scale):
    from fractions import Fraction

    values = [-scale] * 40 + [2 * scale] * 20
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=60),
            "period_return": values,
        }
    )
    if missing == "all":
        panel["period_return"] = np.nan
    elif missing is not None:
        panel.loc[missing, "period_return"] = np.nan
    result = calculate_price_volume_factors(panel, factor_columns=("skewness_60d",))
    if missing is not None:
        assert result.skewness_60d.isna().all()
    else:
        # Exact mean=0, m2=2, m3=2; adjusted Fisher-Pearson sample skew.
        expected = (60 * 59) ** 0.5 / 58 * float(Fraction(2)) / 2**1.5
        assert result.skewness_60d.iloc[-1] == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
@pytest.mark.parametrize("kind", ["amount", "volume", "turnover"])
def test_return_correlations_preserve_units_and_persist(scale, kind, tmp_path):
    from decimal import Decimal, localcontext

    x = np.arange(1.0, 21.0) * scale
    y = np.arange(1.0, 21.0) ** 2
    with localcontext() as context:
        context.prec = 90
        left = [Decimal.from_float(float(value)) for value in x]
        right = [Decimal.from_float(float(value)) for value in y]
        left_mean, right_mean = sum(left) / 20, sum(right) / 20
        dx = [value - left_mean for value in left]
        dy = [value - right_mean for value in right]
        expected = float(
            sum(u * v for u, v in zip(dx, dy, strict=True))
            / (sum(v * v for v in dx) * sum(v * v for v in dy)).sqrt()
        )
    name = f"return_{kind}_corr_20d"
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=20),
            "period_return": x,
            "amount": y,
            "volume": y,
            "close": 1.0,
            "float_market_cap": 1.0,
        }
    )
    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_input_data_id="price_volume",
        factor_columns=(name,),
    )
    result = calculate_factors_for_panel(panel, definitions=definitions)
    assert result[name].iloc[:19].isna().all()
    assert result[name].iloc[-1] == expected
    path = tmp_path / "correlation.parquet"
    result.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result, check_exact=True)


@pytest.mark.parametrize("scale", [1.0, 1e-200, 1e200])
@pytest.mark.parametrize(
    "name",
    [
        "volatility_20d",
        "volatility_60d",
        "amount_volatility_20d",
        "turnover_volatility_20d",
        "downside_volatility_20d",
        "downside_volatility_60d",
        "upside_volatility_20d",
    ],
)
def test_volatility_representable_result_with_extreme_units(name, scale, tmp_path):
    from math import sqrt

    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_processed_dataset_spec,
    )

    count = 60 if name.endswith("60d") else 20
    values = np.array([-scale, 0, scale, scale] * (count // 4))
    # Each four-value block has mean 1/4 and centered sum of squares 11/4.
    # Negative and positive squared sums per observation are 1/4 and 1/2.
    oracle = scale * (
        0.5
        if name.startswith("downside")
        else sqrt(0.5)
        if name.startswith("upside")
        else sqrt(11 * count / (16 * (count - 1)))
    )
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=count).as_unit("ns"),
            "period_return": values,
            "amount": values + 2 * scale,
            "volume": values + 2 * scale,
            "close": 1.0,
            "float_market_cap": 1.0,
        }
    )
    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_input_data_id="price_volume",
        factor_columns=(name,),
    )
    result = calculate_factors_for_panel(panel, definitions=definitions)
    assert result[name].iloc[-1] == pytest.approx(oracle, rel=2e-15, abs=0)
    root = make_dataset_artifact_root(tmp_path, "volatility")
    write_dataset(
        root,
        result,
        spec=make_processed_dataset_spec("volatility", value_column=name),
    )
    persisted = read_dataset(root)
    pd.testing.assert_frame_equal(persisted, result, check_exact=True)
    assert persisted[name].iloc[-1] == pytest.approx(oracle, rel=2e-15, abs=0)


def test_expanded_pv_nine_formulas_complete_windows_and_nulls(tmp_path):
    from math import sqrt

    from quant_research_platform.research.factors.price_volume.definitions import (
        PRICE_VOLUME_LIBRARY,
    )

    names = PRICE_VOLUME_LIBRARY[-9:]
    x = np.arange(1.0, 251.0)
    returns = np.resize(np.array([-0.02, 0.0, 0.01, 0.03]), len(x))
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=250),
            "close": 10.0,
            "amount": x,
            "volume": x,
            "float_market_cap": 1000.0,
            "period_return": returns,
        }
    )
    result = calculate_price_volume_factors(panel, factor_columns=names)
    # Last 126 integers are 125..250, mean=187.5; sample variance=n(n+1)/12.
    # Last 20 returns contain five copies of each of the four values.
    ret = [-0.02, 0.0, 0.01, 0.03]
    mu = sum(ret) / 4
    m2 = sum((r - mu) ** 2 for r in ret) / 4
    m3 = sum((r - mu) ** 3 for r in ret) / 4
    expected = {
        "share_price": 10.0,
        "avg_amount_126d": 187.5,
        "amount_cv_126d": sqrt(126 * 127 / 12) / 187.5,
        "turnover_cv_126d": sqrt(126 * 127 / 12) / 187.5,
        "amihud_126d": sum(abs(returns[i]) / x[i] for i in range(124, 250)) / 126,
        "skewness_20d": sqrt(20 * 19) / 18 * m3 / m2**1.5,
        "top5_return_mean_20d": 0.03,
        "turnover_250d": 1.255,
        "abnormal_turnover_20_250": 240.5 / 125.5,
    }
    for name, value in expected.items():
        assert result[name].iloc[-1] == pytest.approx(value, abs=1e-14), name
        count = (
            1
            if name == "share_price"
            else 250
            if name in {"turnover_250d", "abnormal_turnover_20_250"}
            else 20
            if name in {"skewness_20d", "top5_return_mean_20d"}
            else 126
        )
        assert result[name].iloc[: count - 1].isna().all(), name
    path = tmp_path / "expanded.parquet"
    result.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result, check_exact=True)
    # A suspended market session stays in the window; absent facts cannot compress it.
    missing = panel.copy()
    missing.loc[240, ["amount", "volume", "period_return"]] = np.nan
    observed = calculate_price_volume_factors(missing, factor_columns=names)
    assert observed.loc[249, list(names[1:])].isna().all()
    zero = calculate_price_volume_factors(
        panel.assign(amount=0.0, volume=0.0, period_return=0.0), factor_columns=names
    )
    assert (
        zero.loc[
            249,
            [
                "amount_cv_126d",
                "turnover_cv_126d",
                "abnormal_turnover_20_250",
                "amihud_126d",
                "skewness_20d",
            ],
        ]
        .isna()
        .all()
    )
    # Suffix changes cannot affect earlier outputs.
    future = panel.copy()
    future.loc[249, ["amount", "volume", "period_return"]] *= 2
    changed = calculate_price_volume_factors(future, factor_columns=names)
    pd.testing.assert_frame_equal(changed.iloc[:-1], result.iloc[:-1], check_exact=True)


def test_market_model_factors_share_risk_math_and_window_identity(tmp_path):
    from quant_research_platform.compute.market_model import MarketModelRiskState
    from quant_research_platform.research.protocols.a_share import factor_catalog

    dates = pd.bdate_range("2020-01-01", periods=400)
    rows = [
        dict(
            market_id="a_share",
            asset_id=asset,
            timestamp=date,
            period_return=multiplier * ((i % 11) - 5) / 1000,
            float_market_cap=1000.0,
            is_a_share_scope=True,
            is_trading=True,
            is_st=False,
            is_pt=False,
            listing_age_trading_days=400,
            close=10.0,
            adj_close=10.0,
            volume=10.0,
            amount=100.0,
            turnover=0.1,
        )
        for i, date in enumerate(dates)
        for asset, multiplier in (("a", 1), ("b", 3))
    ]
    panel = pd.DataFrame(rows)
    definitions = tuple(
        d for d in factor_catalog().list() if d.family == "market_model"
    )
    result = calculate_factors_for_panel(panel, definitions=definitions)
    assert result.loc[result.timestamp < dates[252], "beta_252d"].isna().all()
    assert (
        result.loc[result.timestamp < dates[311], "residual_volatility_60d"]
        .isna()
        .all()
    )
    for asset, beta in (("a", 0.5), ("b", 1.5)):
        tail = result.loc[(result.asset_id == asset) & (result.timestamp >= dates[311])]
        assert np.allclose(tail.beta_252d, beta, rtol=1e-14, atol=0)
        assert np.allclose(tail.residual_volatility_60d, 0, atol=1e-16)
    state = MarketModelRiskState()
    for _, cross in panel.groupby("timestamp", sort=False):
        expected = state.observe(cross.assign(eligible=True))
    last = result.loc[result.timestamp == dates[-1]].set_index("asset_id")
    assert np.array_equal(last.beta_252d, expected.beta)
    assert np.array_equal(last.residual_volatility_60d, expected.residual_volatility)
    # The producer's bounded warmup must equal full-history output.
    bounded = calculate_factors_for_panel(
        panel.loc[panel.timestamp >= dates[350 - 312]],
        definitions=definitions,
        output_window=TimeWindow(dates[350], dates[-1]),
    )
    pd.testing.assert_frame_equal(
        bounded.reset_index(drop=True),
        result.loc[result.timestamp >= dates[350]].reset_index(drop=True),
        check_exact=True,
    )
    path = tmp_path / "market_model.parquet"
    bounded.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), bounded, check_exact=True)


def test_batched_volatility_matches_scalar_reference_across_units_and_windows():
    from statistics import stdev

    rng = np.random.default_rng(7)
    samples = [rng.normal(size=180) * scale for scale in (1.0, 1e-200, 1e200)]
    samples.extend(
        [
            1.0 + np.arange(180) * np.spacing(1.0),
            np.resize([-1e308, 1e308], 180),
            np.full(180, 0.125),
        ]
    )
    for values in samples:
        values[90] = np.nan
        panel = pd.DataFrame(
            {
                "asset_id": "asset",
                "timestamp": pd.bdate_range("2020-01-01", periods=len(values)),
                "period_return": values,
            }
        )
        actual = calculate_price_volume_factors(
            panel, factor_columns=("volatility_20d", "volatility_60d")
        )
        for size in (20, 60):
            expected = np.full(len(values), np.nan)
            for stop in range(size, len(values) + 1):
                window = values[stop - size : stop]
                if np.isfinite(window).all():
                    expected[stop - 1] = stdev(window)
            np.testing.assert_allclose(
                actual[f"volatility_{size}d"],
                expected,
                rtol=2e-15,
                atol=0,
                equal_nan=True,
            )


def test_long_rolling_means_match_exact_scalar_windows():
    from statistics import mean

    values = np.resize(np.array([1e200, -1e200, 1e-200, 3.0, 0.0, 1e200, -1e200]), 400)
    values[270] = np.nan
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": dates,
            "amount": values,
            "volume": values,
            "close": 1.0,
            "float_market_cap": 1.0,
        }
    )
    result = calculate_price_volume_factors(
        panel, factor_columns=("avg_amount_126d", "turnover_250d")
    )
    for name, size in (("avg_amount_126d", 126), ("turnover_250d", 250)):
        expected = np.full(len(values), np.nan)
        for stop in range(size, len(values) + 1):
            window = values[stop - size : stop]
            if np.isfinite(window).all():
                expected[stop - 1] = mean(window)
        np.testing.assert_array_equal(result[name], expected)
        bounded = calculate_price_volume_factors(
            panel.iloc[20:],
            factor_columns=(name,),
            output_window=TimeWindow(dates[300], dates[-1]),
        )
        np.testing.assert_array_equal(bounded[name], result[name].iloc[300:])
