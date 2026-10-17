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
    average = price_volume_factor_definitions(
        **binding, factor_columns=("avg_amount_20d",)
    )
    assert factor_lookback_bars(average) == 19
    assert average[0].required_windows == (20,)
    shock = price_volume_factor_definitions(
        **binding, factor_columns=("amount_shock_20d",)
    )
    assert factor_lookback_bars(shock) == 20
    x = np.arange(1.0, 301.0)
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "filling_flag": 0,
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=len(x)),
            "adj_close": x,
            "period_return": np.sin(x) * 0.01,
            "high": x + 2.0,
            "low": x - 0.5,
            "close": x,
            "amount": x * 10.0,
            "volume": x + 10.0,
            "turnover": (x + 10.0) / 1000.0,
            "float_market_cap": x * 1000.0,
            "total_market_cap": x * 2000.0,
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
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=len(x)),
            "adj_close": x,
            "period_return": returns,
            "amount": x,
            "volume": 2.0 * x,
            "turnover": 2.0 * x / 100.0,
            "float_market_cap": 1000.0,
            "close": 10.0,
            "total_market_cap": 10000.0,
        }
    )
    result = calculate_price_volume_factors(
        panel, min_fresh_fraction=1.0, factor_columns=UNIQUE_FACTORS
    )
    last = result.iloc[-1]
    flow, ret = (x[-20:], returns[-20:])
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
        "log_total_market_cap": np.log(10000.0),
    }
    for name, value in expected.items():
        assert last[name] == pytest.approx(value), name
    scaled = calculate_price_volume_factors(
        panel.assign(period_return=returns * 1e200),
        min_fresh_fraction=1.0,
        factor_columns=("kurtosis_60d",),
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
        missing, min_fresh_fraction=1.0, factor_columns=UNIQUE_FACTORS
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
        amount=0.0, volume=0.0, turnover=0.0, period_return=0.0, total_market_cap=0.0
    )
    zero = calculate_price_volume_factors(
        constant, min_fresh_fraction=1.0, factor_columns=UNIQUE_FACTORS
    ).iloc[-1]
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
        nonpositive, min_fresh_fraction=1.0, factor_columns=("zero_amount_ratio_20d",)
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
@pytest.mark.parametrize("scale", [1.0, 1e-10, 10000000000.0, 1e-200, 1e200])
def test_price_volume_skew_matches_independent_decimal_oracle(size, scale):
    values = np.random.default_rng(20260910).normal(size=320) * scale
    values[200] = np.nan
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    panel = pd.DataFrame(
        {"asset_id": "asset", "timestamp": dates, "period_return": values}
    )
    original = panel.copy(deep=True)
    actual = calculate_price_volume_factors(
        panel,
        min_fresh_fraction=1.0,
        factor_columns=(f"skewness_{size}d",),
        output_window=TimeWindow(dates[size - 1].date(), dates[-1].date()),
    )
    from decimal import Decimal, localcontext

    expected = []
    error_bounds = []
    for start in range(len(values) - size + 1):
        window = values[start : start + size]
        if np.isnan(window).any():
            expected.append(np.nan)
            error_bounds.append(0.0)
            continue
        with localcontext() as context:
            context.prec = 80
            data = [Decimal.from_float(float(value)) for value in window]
            n = Decimal(size)
            mean = sum(data) / n
            centered = [value - mean for value in data]
            m2 = sum(value**2 for value in centered) / n
            m3 = sum(value**3 for value in centered) / n
            correction = (n * (n - 1)).sqrt() / (n - 2)
            expected.append(float(correction * m3 / (m2 * m2.sqrt())))
            absolute_m3 = sum(abs(value) ** 3 for value in centered) / n
            unit = np.finfo("float64").eps
            gamma = 32 * size * unit / (1 - 32 * size * unit)
            error_bounds.append(
                gamma * float(correction * absolute_m3 / (m2 * m2.sqrt()))
            )
        single = calculate_price_volume_factors(
            panel.iloc[start : start + size],
            min_fresh_fraction=1.0,
            factor_columns=(f"skewness_{size}d",),
        )
        assert (
            single.iloc[-1][f"skewness_{size}d"]
            == actual.iloc[start][f"skewness_{size}d"]
        )
    observed = actual[f"skewness_{size}d"].to_numpy()
    expected = np.asarray(expected)
    assert np.array_equal(np.isnan(observed), np.isnan(expected))
    valid = np.isfinite(expected)
    assert np.all(
        np.abs(observed[valid] - expected[valid]) <= np.asarray(error_bounds)[valid]
    )
    pd.testing.assert_frame_equal(panel, original, check_exact=True)


@pytest.mark.parametrize("size", (20, 60))
@pytest.mark.parametrize(
    "case",
    (
        "adjacent",
        "adjacent_anchor",
        "tiny_adjacent",
        "large_adjacent",
        "subnormal_adjacent",
        "opposite_extremes",
        "constant",
        "missing",
    ),
)
def test_standardized_moments_preserve_adjacent_values_against_decimal(size, case):
    from decimal import Decimal, localcontext

    base = {
        "tiny_adjacent": 1e-200,
        "large_adjacent": 1e308,
        "subnormal_adjacent": np.nextafter(0.0, 1.0),
    }.get(case, 0.01)
    values = np.full(size, base)
    values[0 if case == "adjacent_anchor" else -1] = np.nextafter(base, np.inf)
    if case == "opposite_extremes":
        values = np.resize([-1e308, 1e308], size)
    elif case == "constant":
        values[:] = base
    elif case == "missing":
        values[0] = np.nan
    names = (f"skewness_{size}d",) + (("kurtosis_60d",) if size == 60 else ())
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=size),
            "period_return": values,
        }
    )
    actual = calculate_price_volume_factors(
        panel, min_fresh_fraction=1.0, factor_columns=names
    )
    assert actual.loc[: size - 2, list(names)].isna().all().all()
    if case in {"constant", "missing"}:
        assert actual.loc[size - 1, list(names)].isna().all()
        return
    with localcontext() as context:
        context.prec = 120
        data = [Decimal.from_float(float(value)) for value in values]
        n = Decimal(size)
        mean = sum(data) / n
        centered = [value - mean for value in data]
        m2 = sum(value**2 for value in centered) / n
        m3 = sum(value**3 for value in centered) / n
        expected = [(n * (n - 1)).sqrt() / (n - 2) * m3 / (m2 * m2.sqrt())]
        if size == 60:
            m4 = sum(value**4 for value in centered) / n
            expected.append(
                (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (m4 / m2**2 - 3) + 6)
            )
    np.testing.assert_allclose(
        actual.loc[size - 1, list(names)].to_numpy(dtype=float),
        [float(value) for value in expected],
        rtol=2e-14,
        atol=2e-14,
    )


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
@pytest.mark.parametrize("scale", [1.0, 1e-10, 10000000000.0, 1e-200, 1e200])
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
    result = calculate_price_volume_factors(
        panel, min_fresh_fraction=1.0, factor_columns=("skewness_60d",)
    )
    if missing is not None:
        assert result.skewness_60d.isna().all()
    else:
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
        left_mean, right_mean = (sum(left) / 20, sum(right) / 20)
        dx = [value - left_mean for value in left]
        dy = [value - right_mean for value in right]
        expected = float(
            sum((u * v for u, v in zip(dx, dy, strict=True)))
            / (sum(v * v for v in dx) * sum(v * v for v in dy)).sqrt()
        )
    name = f"return_{kind}_corr_20d"
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "filling_flag": 0,
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=20),
            "period_return": x,
            "amount": y,
            "volume": y,
            "turnover": y,
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
    # Dimensionless, well-conditioned 20-point correlation; allow rounding in
    # the independent floating reduction while retaining the Decimal oracle.
    assert result[name].iloc[-1] == pytest.approx(expected, rel=1e-13, abs=0)
    path = tmp_path / "correlation.parquet"
    result.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result, check_exact=True)


@pytest.mark.parametrize(
    "case", ("normal", "subnormal", "adjacent", "mixed", "missing")
)
def test_rolling_correlations_match_exact_windows_and_warmup(case):
    from quant_research_platform.compute.dataframe.cross_section import (
        _pearson_correlation_result,
    )
    from quant_research_platform.research.factors.price_volume.calculations import (
        _calculate_values,
    )

    rng = np.random.default_rng(719)
    x, y = rng.normal(size=160), rng.normal(size=160)
    if case == "subnormal":
        x = np.arange(1.0, 161.0) * np.nextafter(0.0, 1.0)
    elif case == "adjacent":
        x = 1e200 + np.arange(160) * np.spacing(1e200)
    elif case == "mixed":
        x[0], y[1] = np.nextafter(0.0, 1.0), 1e300
    elif case == "missing":
        x[45], y[98] = np.nan, np.inf
        x[60:82] = 1.0
    expected = np.full(len(x), np.nan)
    for index in range(19, len(x)):
        value, _ = _pearson_correlation_result(
            x[index - 19 : index + 1], y[index - 19 : index + 1]
        )
        expected[index] = np.nan if value is None else value
    for kind in ("amount", "volume", "turnover"):
        name = f"return_{kind}_corr_20d"
        values = {"period_return": x, kind: y}
        for start, stop in ((0, 160), (22, 41), (99, 160), (19, 20)):
            actual = _calculate_values(
                values, {}, name, start, stop, min_fresh_fraction=1.0
            )
            np.testing.assert_allclose(
                actual, expected[start:stop], rtol=1e-14, atol=1e-15
            )


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
    oracle = scale * (
        0.5
        if name.startswith("downside")
        else sqrt(0.5)
        if name.startswith("upside")
        else sqrt(11 * count / (16 * (count - 1)))
    )
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "filling_flag": 0,
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=count).as_unit("ns"),
            "period_return": values,
            "amount": values + 2 * scale,
            "volume": values + 2 * scale,
            "turnover": values + 2 * scale,
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
        root, result, spec=make_processed_dataset_spec("volatility", value_column=name)
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
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=250),
            "close": 10.0,
            "amount": x,
            "volume": x,
            "turnover": x / 100.0,
            "float_market_cap": 1000.0,
            "period_return": returns,
        }
    )
    result = calculate_price_volume_factors(
        panel, min_fresh_fraction=1.0, factor_columns=names
    )
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
    missing = panel.copy()
    missing.loc[240, ["amount", "volume", "turnover", "period_return"]] = np.nan
    observed = calculate_price_volume_factors(
        missing, min_fresh_fraction=1.0, factor_columns=names
    )
    assert observed.loc[249, list(names[1:])].isna().all()
    zero = calculate_price_volume_factors(
        panel.assign(amount=0.0, volume=0.0, turnover=0.0, period_return=0.0),
        min_fresh_fraction=1.0,
        factor_columns=names,
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
    future = panel.copy()
    future.loc[249, ["amount", "volume", "turnover", "period_return"]] *= 2
    changed = calculate_price_volume_factors(
        future, min_fresh_fraction=1.0, factor_columns=names
    )
    pd.testing.assert_frame_equal(changed.iloc[:-1], result.iloc[:-1], check_exact=True)


def test_market_model_factors_share_risk_math_and_window_identity(tmp_path):
    from quant_research_platform.compute.market_model import MarketModelRiskState

    dates = pd.bdate_range("2020-01-01", periods=400)
    rows = [
        dict(
            market_id="a_share",
            asset_id=asset,
            listing_date=pd.Timestamp("2000-01-01"),
            timestamp=date,
            period_return=multiplier * (i % 11 - 5) / 1000,
            float_market_cap=1000.0,
            is_a_share_scope=True,
            is_trading=True,
            is_st=False,
            is_pt=False,
            listing_age_trading_days=400,
            has_252_trading_sessions=True,
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
    from quant_research_platform.research.factors.engine import _market_model_frames

    result = pd.concat(
        _market_model_frames((panel,), trading_calendar=None, market_history=None),
        ignore_index=True,
    )
    assert result.loc[result.timestamp < dates[251], "beta_252d"].isna().all()
    assert (
        result.loc[result.timestamp < dates[295], "residual_volatility_60d"]
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
    bounded = pd.concat(
        _market_model_frames(
            (panel.loc[panel.timestamp >= dates[350 - 312]],),
            trading_calendar=None,
            market_history=None,
        ),
        ignore_index=True,
    )
    bounded = bounded.loc[bounded.timestamp >= dates[350]].reset_index(drop=True)
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
            panel,
            min_fresh_fraction=1.0,
            factor_columns=("volatility_20d", "volatility_60d"),
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


@pytest.mark.parametrize("extreme", (False, True))
def test_rolling_means_match_scalar_windows_with_numeric_tolerance(extreme):
    from fractions import Fraction
    from statistics import mean

    values = (
        np.full(400, 1e308)
        if extreme
        else np.resize([1e200, -1e200, 1e-200, 3.0, 0.0, 1e200, -1e200], 400)
    )
    values[270] = np.nan
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": dates,
            "amount": values,
            "volume": values,
            "turnover": values,
            "close": 1.0,
            "float_market_cap": 1.0,
        }
    )
    cases = (
        ("avg_amount_20d", 20, 0),
        ("avg_amount_126d", 126, 0),
        ("log_avg_amount_20d", 20, 0),
        ("turnover_20d", 20, 0),
        ("turnover_60d", 60, 0),
        ("turnover_250d", 250, 0),
        ("amount_shock_20d", 20, 1),
        ("volume_shock_20d", 20, 1),
    )
    result = calculate_price_volume_factors(
        panel,
        min_fresh_fraction=1.0,
        factor_columns=tuple((name for name, _, _ in cases)),
    )
    for name, size, shift in cases:
        expected = np.full(len(values), np.nan)
        for stop in range(size + shift, len(values) + 1):
            window = values[stop - size - shift : stop - shift]
            if np.isfinite(window).all() and np.isfinite(values[stop - 1]):
                average = mean(window)
                expected[stop - 1] = (
                    (
                        float(
                            Fraction(float(values[stop - 1]))
                            / (sum(map(Fraction, window)) / size)
                            - 1
                        )
                        if sum(map(Fraction, window)) != 0
                        else np.nan
                    )
                    if shift
                    else (np.log(average) if average > 0 else np.nan)
                    if name == "log_avg_amount_20d"
                    else average
                )
        np.testing.assert_allclose(
            result[name], expected, rtol=1e-12, atol=0, equal_nan=True
        )
        bounded = calculate_price_volume_factors(
            panel.iloc[20:],
            min_fresh_fraction=1.0,
            factor_columns=(name,),
            output_window=TimeWindow(dates[300], dates[-1]),
        )
        np.testing.assert_allclose(
            bounded[name], result[name].iloc[300:], rtol=1e-12, atol=0, equal_nan=True
        )


def test_turnover_uses_canonical_suspension_zero_and_preserves_unknown(tmp_path):
    from quant_research_platform.markets.implementations.a_share.processed import panels
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.storage.low_level_fixtures import (
        make_dataset_artifact_root,
        make_processed_dataset_spec,
    )

    raw = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=20).as_unit("ns"),
            "market_id": "a_share",
            "market_type": 1,
            "filling_flag": [0] * 19 + [1],
            "close": 10.0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "period_return": 0.0,
            "volume": 1.0,
            "amount": 10.0,
            "float_market_cap": 1000.0,
        }
    )
    raw.loc[19, "float_market_cap"] = np.nan
    raw["listing_date"] = raw["timestamp"].min()
    panel = panels._apply_suspension_semantics(raw)
    root = make_dataset_artifact_root(tmp_path, "canonical_turnover")
    selected = panel[["market_id", "asset_id", "timestamp", "turnover"]]
    write_dataset(
        root,
        selected,
        spec=make_processed_dataset_spec("canonical_turnover", value_column="turnover"),
    )
    persisted = read_dataset(root)
    result = calculate_price_volume_factors(persisted, factor_columns=("turnover_20d",))
    assert result.turnover_20d.iloc[-1] == pytest.approx(0.0095)
    raw.loc[19, "filling_flag"] = 0
    unknown = panels._apply_suspension_semantics(raw)
    assert calculate_price_volume_factors(
        unknown, factor_columns=("turnover_20d",)
    ).turnover_20d.iloc[-1] == pytest.approx(0.01)


@pytest.mark.parametrize("amount_scale", (1.0, 1e-308, 1e308))
@pytest.mark.parametrize("direction", (1.0, -1.0, 0.0))
def test_amount_flow_ratios_preserve_finite_units_and_undefined_denominators(
    amount_scale, direction
):
    names = ("signed_amount_flow_20d", "return_amount_imbalance_20d")
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=20),
            "period_return": direction * 0.1,
            "amount": amount_scale,
        }
    )
    result = calculate_price_volume_factors(panel, factor_columns=names)
    assert result.loc[:18, list(names)].isna().all().all()
    assert result[names[0]].iloc[-1] == direction
    if direction:
        assert result[names[1]].iloc[-1] == direction
    else:
        assert pd.isna(result[names[1]].iloc[-1])
    panel.loc[0, "amount"] = np.nan
    assert (
        calculate_price_volume_factors(panel, factor_columns=names)[list(names)]
        .isna()
        .all()
        .all()
    )
    panel["amount"] = 0.0
    assert (
        calculate_price_volume_factors(panel, factor_columns=names)[list(names)]
        .isna()
        .all()
        .all()
    )


def test_return_amount_products_preserve_opposing_extreme_units():
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=20),
            "period_return": [1e308, -1e-308] * 10,
            "amount": [1e-308, 1e308] * 10,
        }
    )
    result = calculate_price_volume_factors(
        panel, factor_columns=("return_amount_imbalance_20d",)
    )
    assert result.return_amount_imbalance_20d.iloc[-1] == 0.0


@pytest.mark.parametrize("scale", (1.0, 1e-200))
def test_market_model_residual_volatility_matches_decimal_oracle(scale):
    from decimal import Decimal, localcontext

    from quant_research_platform.compute.market_model import MarketModelRiskState

    left = np.resize(np.array([0.01, 0.02, -0.01, 0.03, 0.0]), 420) * scale
    right = np.resize(np.array([-0.01, 0.01, 0.02, -0.01, 0.04]), 420) * scale
    with localcontext() as context:
        context.prec = 100
        a = [Decimal.from_float(float(value)) for value in left]
        b = [Decimal.from_float(float(value)) for value in right]
        market = [(x + y) / 2 for x, y in zip(a, b, strict=True)]
        residuals = []
        for stop in range(len(a) - 60, len(a)):
            xs, ys = (market[stop - 251 : stop + 1], a[stop - 251 : stop + 1])
            mx, my = (sum(xs) / 252, sum(ys) / 252)
            beta = sum(
                ((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
            ) / sum((x - mx) ** 2 for x in xs)
            residuals.append(ys[-1] - my - beta * (xs[-1] - mx))
        mean = sum(residuals) / 60
        expected = float(
            (sum(((value - mean) ** 2 for value in residuals)) / 60).sqrt()
        )
    state = MarketModelRiskState()
    for left_value, right_value in zip(left, right, strict=True):
        observed = state.observe(
            pd.DataFrame(
                {
                    "listing_date": pd.Timestamp("2000-01-01"),
                    "asset_id": ("a", "b"),
                    "period_return": (left_value, right_value),
                    "float_market_cap": 1000.0,
                    "eligible": True,
                }
            )
        )
    assert observed.residual_volatility["a"] == pytest.approx(
        expected, rel=1e-12, abs=0
    )


def test_trend_efficiency_extreme_normal_price_path_persists_finite_ratio(tmp_path):
    from dataclasses import replace

    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    exponents = [0]
    for _ in range(2):
        exponents.append(exponents[-1] + 1023)
        for _ in range(29):
            exponents.append(exponents[-1] - 51)
    prices = np.ldexp(np.ones(61), exponents)
    returns = prices[1:] / prices[:-1] - 1.0
    assert np.isfinite(prices).all() and (prices >= np.finfo(float).tiny).all()
    np.testing.assert_array_equal(prices[:-1] * (1 + returns), prices[1:])
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "filling_flag": 0,
            "asset_id": "synthetic",
            "timestamp": pd.bdate_range("2020-01-02", periods=60).as_unit("ns"),
            "period_return": returns,
        }
    )
    calculated = calculate_price_volume_factors(
        frame, factor_columns=("trend_efficiency_60d",)
    )
    calculated.insert(0, "market_id", "a_share")
    base = scientific_catalogs().processed.get("price_volume_facts")
    spec = replace(
        base,
        column_roles={
            key: value
            for key, value in base.column_roles.items()
            if key in calculated.columns
        },
    )
    from tests.helpers.paths import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, calculated, spec=spec)
    saved = read_dataset(root)
    assert len(saved) == 60
    assert saved.trend_efficiency_60d.iloc[-1] == 1.0
    assert saved.trend_efficiency_60d.iloc[:-1].isna().all()


@pytest.mark.parametrize(
    "values",
    (
        [0.01] * 252 + [1e308] * 60,
        [0.01, 1e308] + [0.01, 0.02] * 156,
        [1e307, 5e307] * 157,
    ),
)
def test_market_model_extreme_equal_assets_keep_unit_beta(values):
    from quant_research_platform.compute.market_model import MarketModelRiskState

    state = MarketModelRiskState()
    for value in values:
        result = state.observe(
            pd.DataFrame(
                {
                    "listing_date": pd.Timestamp("2000-01-01"),
                    "asset_id": list("abcd"),
                    "period_return": value,
                    "float_market_cap": 1.0,
                }
            )
        )
    np.testing.assert_allclose(result.beta, 1.0, rtol=1e-14)
    np.testing.assert_array_equal(result.residual_volatility, 0.0)


@pytest.mark.parametrize("single_extreme", (False, True))
def test_amplitude_extreme_representable_mean_persists(tmp_path, single_extreme):
    from dataclasses import replace

    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs
    from tests.helpers.paths import StorageLayout

    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "filling_flag": 0,
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-02", periods=20).as_unit("ns"),
            "high": 1e308,
            "low": 0.5,
            "close": 1.0,
        }
    )
    from fractions import Fraction

    if single_extreme:
        frame[["high", "low", "close"]] = 0.1
        frame.loc[0, ["high", "low"]] = [1e308, 0.01]
    expected = float(
        sum(
            (Fraction(row.high) - Fraction(row.low)) / Fraction(row.close)
            for row in frame.itertuples()
        )
        / len(frame)
    )
    output = calculate_price_volume_factors(frame, factor_columns=("amplitude_20d",))
    output.insert(0, "market_id", "a_share")
    base = scientific_catalogs().processed.get("price_volume_facts")
    spec = replace(
        base,
        column_roles={
            key: value
            for key, value in base.column_roles.items()
            if key in output.columns
        },
    )
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, output, spec=spec)
    assert read_dataset(root).amplitude_20d.iloc[-1] == pytest.approx(
        expected, rel=2e-16
    )


def test_market_model_extreme_positive_prices_persist_usable_risk_styles(tmp_path):
    from quant_research_platform.research.portfolios.risk import FactorRiskState
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs
    from tests.helpers.paths import StorageLayout

    dates = pd.bdate_range("2020-01-02", periods=312).as_unit("ns")
    prices = np.empty(312)
    prices[0], prices[1] = 1e-307, 10.0
    for i in range(2, len(prices)):
        prices[i] = prices[i - 1] * (1.01 if i % 2 else 1.02)
    returns = prices / np.r_[prices[0], prices[:-1]] - 1
    assert np.isfinite(prices).all() and (prices > 0).all()
    panel = pd.DataFrame(
        [
            dict(
                market_id="a_share",
                asset_id=asset,
                listing_date=pd.Timestamp("2000-01-01"),
                timestamp=stamp,
                period_return=returns[i],
                float_market_cap=1000.0,
                is_a_share_scope=True,
                is_trading=True,
                is_st=False,
                is_pt=False,
                listing_age_trading_days=400,
                has_252_trading_sessions=True,
                close=prices[i],
                adj_close=prices[i],
                volume=10.0,
                amount=100.0,
                turnover=0.1,
            )
            for i, stamp in enumerate(dates)
            for asset in ("a", "b")
        ]
    )
    from quant_research_platform.research.factors.engine import _market_model_frames

    output = pd.concat(
        _market_model_frames((panel,), trading_calendar=None, market_history=None),
        ignore_index=True,
    )
    spec = scientific_catalogs().processed.get("market_model_factors")
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, output, spec=spec)
    saved = read_dataset(root)
    assert saved.loc[saved.timestamp.eq(dates[252]), "beta_252d"].eq(1.0).all()
    current = saved.loc[saved.timestamp.eq(dates[-1])].assign(
        industry_standard_id="CSRC2012",
        industry_level1_code="A",
        total_market_cap=1000.0,
        eligible=True,
    )
    assert current.residual_volatility_60d.eq(0.0).all()
    risk = FactorRiskState(style_columns=("beta_252d", "residual_volatility_60d"))
    assert len(risk.prepare_exposures(current)) == 2


def test_market_model_preserves_small_residual_beside_extreme_historical_mean():
    from fractions import Fraction

    from quant_research_platform.compute.market_model import MarketModelRiskState

    state = MarketModelRiskState()
    for index in range(253):
        left, right = (
            (0.0, 0.0)
            if index == 0
            else (1e308, 1e308)
            if index == 1
            else ((0.01, 0.02) if index % 2 else (0.02, 0.01))
        )
        state.observe(
            pd.DataFrame(
                {
                    "listing_date": pd.Timestamp("2000-01-01"),
                    "asset_id": ("a", "b"),
                    "period_return": (left, right),
                    "float_market_cap": 1.0,
                }
            )
        )
    x = [Fraction(value) for value in state._market_returns["a"]]
    y = [Fraction(value) for value in state._returns["a"]]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / sum(
        (a - mx) ** 2 for a in x
    )
    expected = float(y[-1] - my - slope * (x[-1] - mx))
    assert expected != 0.0
    assert state._residuals["a"][-1] == expected


def test_all_a_convex_mean_keeps_subnormal_mass_with_extreme_caps():
    from quant_research_platform.compute.market_model import MarketModelRiskState

    small = np.nextafter(0.0, 1.0)
    state = MarketModelRiskState(window=2)
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "asset_id": ("a", "b"),
            "period_return": (0.0, 0.0),
            "float_market_cap": (small, 1e308),
        }
    )
    state.observe(panel)
    state.observe(panel.assign(period_return=(1e308, 0.0)))
    assert state._market_returns["a"][-1] == small
    assert state._market_returns["b"][-1] == small


@pytest.mark.parametrize(
    "factor,values",
    (
        ("amount_cv_126d", [1e16, 1.0, -1e16] * 42),
        ("amount_cv_126d", [float(2**53), float(2**53 + 2), float(2**53 + 4)] * 42),
        (
            "turnover_cv_126d",
            [1.0, np.nextafter(1.0, 2.0), np.nextafter(np.nextafter(1.0, 2.0), 2.0)]
            * 42,
        ),
    ),
)
def test_coefficient_of_variation_keeps_exact_center_and_ratio(
    tmp_path, factor, values
):
    from dataclasses import replace
    from decimal import Decimal, localcontext
    from fractions import Fraction

    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    fractions = [Fraction(value) for value in values]
    center = sum(fractions) / len(fractions)
    variance = sum((value - center) ** 2 for value in fractions) / (len(fractions) - 1)
    with localcontext() as context:
        context.prec = 200
        deviation = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        expected = float(
            deviation / (Decimal(center.numerator) / Decimal(center.denominator))
        )
    frame = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-02", periods=126).as_unit("ns"),
            factor.split("_")[0]: values,
        }
    )
    result = calculate_price_volume_factors(frame, factor_columns=(factor,))
    prewrite_value = result[factor].iloc[-1]
    assert prewrite_value == pytest.approx(expected, rel=1e-12, abs=0)
    result.insert(0, "market_id", "a_share")
    base = scientific_catalogs().processed.get("price_volume_facts")
    spec = replace(
        base,
        column_roles={
            name: role for name, role in base.column_roles.items() if name in result
        },
    )
    from tests.helpers.paths import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, result, spec=spec)
    assert read_dataset(root)[factor].iloc[-1] == prewrite_value


def test_path_factors_complete_windows_zero_path_and_future_isolation():
    from quant_research_platform.research.factors.price_volume.definitions import (
        PRICE_VOLUME_PATH_FACTORS,
    )

    returns = [0.1, -0.1] * 10 + [1.0]
    panel = pd.DataFrame(
        {
            "asset_id": ["sample"] * 21,
            "timestamp": pd.date_range("2020-01-01", periods=21),
            "period_return": returns,
        }
    )
    result = calculate_price_volume_factors(
        panel, factor_columns=PRICE_VOLUME_PATH_FACTORS
    )
    assert result.loc[:18, list(PRICE_VOLUME_PATH_FACTORS)].isna().all().all()
    np.testing.assert_allclose(
        result.loc[19, list(PRICE_VOLUME_PATH_FACTORS)].to_numpy(dtype=float),
        [0.5, 0.05, 1.0],
    )
    changed = panel.copy()
    changed.loc[20, "period_return"] = -100.0
    pd.testing.assert_frame_equal(
        result.iloc[:20],
        calculate_price_volume_factors(
            changed, factor_columns=PRICE_VOLUME_PATH_FACTORS
        ).iloc[:20],
    )
    panel["period_return"] = 0.0
    zero = calculate_price_volume_factors(
        panel, factor_columns=PRICE_VOLUME_PATH_FACTORS
    )
    assert zero.loc[19, "positive_return_fraction_20d"] == 0.0
    assert zero.loc[19, list(PRICE_VOLUME_PATH_FACTORS[1:])].isna().all()
    panel.loc[10, "period_return"] = np.nan
    assert (
        calculate_price_volume_factors(panel, factor_columns=PRICE_VOLUME_PATH_FACTORS)
        .loc[19, list(PRICE_VOLUME_PATH_FACTORS)]
        .isna()
        .all()
    )


@pytest.mark.parametrize("scale", (1.0, np.nextafter(0.0, 1.0)))
@pytest.mark.parametrize("positive_count", (10, 15, 20))
def test_shock_and_log_preserve_exact_positive_window_mean(scale, positive_count):
    from decimal import Decimal, localcontext
    from fractions import Fraction

    amounts = [scale] * positive_count + [0.0] * (20 - positive_count) + [scale]
    panel = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=21),
            "amount": amounts,
            "volume": amounts,
        }
    )
    names = ("amount_shock_20d", "volume_shock_20d", "log_avg_amount_20d")
    actual = calculate_price_volume_factors(panel, factor_columns=names)
    center = sum(map(Fraction, amounts[:20])) / 20
    expected = float(Fraction(scale) / center - 1)
    assert actual.amount_shock_20d.iloc[-1] == expected
    assert actual.volume_shock_20d.iloc[-1] == expected
    with localcontext() as context:
        context.prec = 100
        log_mean = float((Decimal(center.numerator) / Decimal(center.denominator)).ln())
    assert actual.log_avg_amount_20d.iloc[19] == pytest.approx(log_mean, abs=1e-12)


def test_prefix_mean_factors_do_not_materialize_sliding_windows(monkeypatch):
    from quant_research_platform.research.factors.price_volume import calculations

    values = np.arange(1.0, 351.0)
    values[100] = np.nan
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2020-01-01", periods=len(values)),
            "amount": values,
            "turnover": values,
        }
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("mean factors must not scan dense sliding windows")

    monkeypatch.setattr(calculations, "sliding_window_view", unexpected)
    result = calculations.calculate_price_volume_factors(
        frame,
        factor_columns=(
            "avg_amount_20d",
            "avg_amount_126d",
            "amount_shock_20d",
            "turnover_250d",
            "abnormal_turnover_20_250",
        ),
    )
    assert result.loc[119, "avg_amount_20d"] == pytest.approx(111.0)
    assert result.loc[120, "avg_amount_20d"] == 111.5
    assert result.loc[120, "amount_shock_20d"] == pytest.approx(121.0 / 111.0 - 1)
    assert result.loc[121, "amount_shock_20d"] == (122.0 - 111.5) / 111.5


def test_unrepresentable_rolling_value_does_not_erase_independent_factor():
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=21),
            "period_return": np.resize([1.79e308, -1.79e308], 21),
        }
    )
    result = calculate_price_volume_factors(
        frame, factor_columns=("volatility_20d", "max_return_20d")
    )
    assert result.volatility_20d.isna().all()
    assert result.max_return_20d.iloc[19:].eq(1.79e308).all()


@pytest.mark.parametrize("factor", ["amount_cv_126d", "turnover_cv_126d"])
def test_cv_nonpositive_exact_center_cannot_retain_rounded_positive_ratio(factor):
    values = np.zeros(126)
    values[[10, 73, 7, 59]] = [1e16, -1e16, 1.0, -1.0]
    frame = pd.DataFrame(
        {
            "asset_id": "a",
            "timestamp": pd.bdate_range("2024-01-01", periods=126),
            factor.split("_")[0]: values,
        }
    )
    assert (
        calculate_price_volume_factors(frame, factor_columns=(factor,))[factor]
        .isna()
        .all()
    )
