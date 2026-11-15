"""Independent formulas, observation clocks and native persistence for new factors."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.history import past_ar1_innovations
from quant_research_platform.compute.market_model import (
    relative_downside_beta,
    rolling_partial_slope,
)
from quant_research_platform.research.factors.market_structure import calculate
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)

LIQUIDITY = "market_trading_liquidity_loss_beta_252d"
DOWNSIDE = "relative_downside_beta_252d"
LEADER = "industry_leader_lag_return_252d"


def test_forecast_innovation_is_fitted_strictly_before_current_observation():
    values = np.array([2, 5, 3, 7, 4, 9, 6, 5, 8, 2], dtype=float)
    expected = np.full(len(values), np.nan)
    for end in range(4, len(values)):
        dates = np.arange(max(1, end - 4), end)
        design = np.column_stack((np.ones(len(dates)), values[dates - 1]))
        coefficients = np.linalg.lstsq(design, values[dates], rcond=None)[0]
        expected[end] = values[end] - coefficients @ [1, values[end - 1]]
    actual = past_ar1_innovations(values, window=4, min_observations=3)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    changed = values.copy()
    changed[6:] += 100
    alternative = past_ar1_innovations(changed, window=4, min_observations=3)
    np.testing.assert_allclose(alternative[:6], actual[:6], atol=1e-12)
    assert alternative[6] - actual[6] == pytest.approx(100)
    assert np.isnan(past_ar1_innovations(np.ones(300))).all()


def test_downside_relative_beta_and_multiple_controls_match_independent_ols():
    market = np.tile([-3.0, -1.0, 1.0, 3.0], 63)
    own = np.where(market < 0, 2 * market, market)
    # Downside slope 2, all-days slope 1.5, hence relative downside beta 0.5.
    actual = relative_downside_beta(own, market)
    assert np.isnan(actual[:-1]).all()
    assert actual[-1] == pytest.approx(0.5)
    rng = np.random.default_rng(14)
    x = rng.normal(size=(320, 3))
    y = 0.3 + x @ [0.7, -0.4, 1.2] + rng.normal(size=320) * 0.01
    x[200, 0] = np.nan
    slope = rolling_partial_slope(y, x[:, 0], (x[:, 1], x[:, 2]))
    for end in (251, 275, 319):
        rows = slice(end - 251, end + 1)
        valid = np.isfinite(x[rows]).all(axis=1)
        design = np.column_stack((np.ones(valid.sum()), x[rows][valid]))
        expected = np.linalg.lstsq(design, y[rows][valid], rcond=None)[0][1]
        assert slope[end] == pytest.approx(expected, rel=1e-11, abs=1e-12)
    assert np.isnan(rolling_partial_slope(y, x[:, 1], (x[:, 1], x[:, 2]))).all()


def _prices(n=75):
    closing = 100 * (1.01 * 0.99) ** np.arange(n)
    opening = closing / 0.99
    return pd.DataFrame(
        dict(
            asset_id="a",
            timestamp=pd.bdate_range("2020-01-02", periods=n).as_unit("ns"),
            adj_close=closing,
            adj_open=opening,
        )
    )


def test_overnight_intraday_log_spreads_require_complete_paths_and_reset_lives():
    frame = _prices()
    names = tuple(f"overnight_intraday_spread_{width}d" for width in (20, 60))
    actual = calculate_price_volume_factors(frame, factor_columns=names)
    for width, name in zip((20, 60), names, strict=True):
        assert actual[name].iloc[:width].isna().all()
        np.testing.assert_allclose(
            actual[name].iloc[width:], width * np.log(1.01 / 0.99)
        )
    scaled = frame.assign(adj_open=frame.adj_open * 4, adj_close=frame.adj_close * 4)
    pd.testing.assert_frame_equal(
        actual, calculate_price_volume_factors(scaled, factor_columns=names)
    )
    frame.loc[40, "adj_open"] = np.nan
    missing = calculate_price_volume_factors(frame, factor_columns=names)
    assert missing[names[0]].iloc[40:60].isna().all()
    assert pd.notna(missing[names[0]].iloc[60])
    unknown_quote = _prices().assign(filling_flag=0)
    unknown_quote.loc[40, "filling_flag"] = 2
    unknown = calculate_price_volume_factors(unknown_quote, factor_columns=names)
    assert unknown[names[0]].iloc[40:61].isna().all()
    assert pd.notna(unknown[names[0]].iloc[61])
    frame["listing_date"] = pd.Timestamp("2019-01-01")
    frame.loc[50:, "listing_date"] = pd.Timestamp("2020-03-01")
    reset = calculate_price_volume_factors(
        frame, factor_columns=names, history_keys=("asset_id", "listing_date")
    )
    assert reset.loc[50:69, names[0]].isna().all()


def test_actual_limit_events_unknown_barriers_halts_and_first_observed_streak():
    frame = _prices(12).assign(
        close=[9, 10, 10, np.nan, 10, 9, 10, 10, 10, 12, 12, 12],
        high=[10, 10, 10, np.nan, 10, 9, 10, 10, 10, 12, 12, 12],
        limit_up_price=[10] * 9 + [12] * 3,
        price_limit_applicable=[1] * 8 + [0, 1, 1, 1],
        filling_flag=[0] * 10 + [1, 0],
    )
    names = ("limit_up_close_failure", "limit_up_streak")
    actual = calculate_price_volume_factors(frame, factor_columns=names)
    np.testing.assert_allclose(
        actual.limit_up_streak, [0, 1, 2, np.nan, np.nan, 0, 1, 2, 0, 1, 0, 1]
    )
    np.testing.assert_allclose(
        actual.limit_up_close_failure, [1, 0, 0, np.nan, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    truncated = calculate_price_volume_factors(frame.iloc[1:], factor_columns=names)
    assert truncated.limit_up_streak.iloc[:4].isna().all()
    frame.loc[6, "limit_up_price"] = np.nan
    unknown = calculate_price_volume_factors(frame, factor_columns=names)
    assert unknown.limit_up_streak.iloc[6:8].isna().all()
    assert unknown.limit_up_streak.iloc[8] == 0


def test_provider_limit_prices_persist_in_yuan_and_no_limit_event_overrides(tmp_path):
    from quant_research_platform.markets.implementations.a_share.processed.panels import (  # noqa: E501
        _apply_no_limit_exceptions,
        _canonical_bars,
    )
    from quant_research_platform.sources.csmar.field_mappings import RAW_TABLE_CONTRACTS
    from quant_research_platform.sources.csmar.interim import materialize_interim_data
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    raw.mkdir()
    row = dict.fromkeys(RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0)
    row.update(
        Stkcd="000001",
        Trddt="2020-01-02",
        Markettype=1,
        Trdsta=0,
        Filling=0,
        Ahshrtrd_D=0,
        Ahvaltrd_D=0,
        LimitStatus=0,
        Opnprc=9,
        Hiprc=10,
        Loprc=9,
        Clsprc=9,
        LimitUp=10,
        LimitDown=8,
    )
    _write_csmar(
        raw,
        "market_bars_1d",
        [row, {**row, "Stkcd": "000002", "LimitUp": "", "LimitDown": ""}],
    )
    root = tmp_path / "interim"
    materialize_interim_data(
        raw_root=raw,
        output_roots={"market_bars_1d": root},
        data_ids=("market_bars_1d",),
    )
    saved = read_dataset(root)
    assert saved.limit_up_price.iloc[0] == 10
    assert saved.limit_down_price.iloc[0] == 8
    panel = _canonical_bars(saved)
    assert panel.price_limit_applicable.iloc[0] == 1
    assert pd.isna(panel.price_limit_applicable.iloc[1])
    no_limit = pd.DataFrame(
        dict(source_asset_id=["000001"], timestamp=[pd.Timestamp("2020-01-02")])
    )
    panel = _apply_no_limit_exceptions(panel, no_limit)
    assert panel.price_limit_applicable.iloc[0] == 0
    assert panel[["limit_up_price", "limit_down_price"]].isna().all(axis=None)
    factors = calculate_price_volume_factors(
        panel, factor_columns=("limit_up_close_failure",)
    )
    assert factors.limit_up_close_failure.iloc[0] == 0
    assert pd.isna(factors.limit_up_close_failure.iloc[1])


def _market(days=820, assets=10):
    rng = np.random.default_rng(521)
    returns = rng.normal(0, 0.015, (days, assets))
    returns[1:, 0] += 0.6 * returns[:-1, -1]
    cap = np.arange(1, assets + 1, dtype=float) * 1e8
    dates = pd.bdate_range("2014-01-02", periods=days).as_unit("ns")
    frame = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id=np.tile([f"s{i:02}" for i in range(assets)], days),
            timestamp=np.repeat(dates, assets),
            listing_date=pd.Timestamp("2010-01-01"),
            period_return=returns.ravel(),
            float_market_cap=np.tile(cap, days),
            amount=np.exp(rng.normal(15, 0.5, (days, assets))).ravel(),
            industry_standard_id="CSRC",
            industry_level1_code="A",
            filling_flag=0,
            is_a_share_scope=True,
        )
    )
    return frame, returns, cap


@pytest.mark.parametrize("arrow_input", (False, True))
def test_full_market_innovation_and_leader_lag_match_independent_past_ols(arrow_input):
    frame, returns, cap = _market()
    if arrow_input:
        frame = frame.convert_dtypes(dtype_backend="pyarrow")
    names = (LIQUIDITY, DOWNSIDE, LEADER)
    result = calculate(
        frame, factor_columns=names, history_keys=("asset_id", "listing_date")
    )
    market = returns @ cap / cap.sum()
    market[0] = np.nan
    impact = abs(returns) * cap / frame.amount.to_numpy().reshape(returns.shape)
    level = np.median(impact, axis=1)
    level[0] = np.nan
    innovations = np.full(len(level), np.nan)
    for end in range(128, len(level)):
        positions = np.arange(max(1, end - 252), end)
        x, y = level[positions - 1], level[positions]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 126:
            coefficients = np.linalg.lstsq(
                np.column_stack((np.ones(valid.sum()), x[valid])), y[valid], rcond=None
            )[0]
            innovations[end] = level[end] - coefficients @ [1, level[end - 1]]
    scale = (
        pd.Series(innovations).shift().rolling(252, min_periods=126).std().to_numpy()
    )
    shock = innovations / scale
    rows = slice(len(frame) // len(cap) - 252, None)
    valid = np.isfinite(shock[rows]) & np.isfinite(market[rows])
    design = np.column_stack(
        (np.ones(valid.sum()), shock[rows][valid], market[rows][valid])
    )
    expected = -np.linalg.lstsq(design, returns[rows, 0][valid], rcond=None)[0][1]
    own = result.loc[result.asset_id.eq("s00")]
    assert own[LIQUIDITY].iloc[-1] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    # The top two of ten known prior-cap members lead; own leaders are excluded.
    for asset, peers in ((0, [8, 9]), (9, [8])):
        leaders = returns[:, peers] @ cap[peers] / cap[peers].sum()
        design = np.column_stack(
            (np.ones(252), leaders[-253:-1], returns[-253:-1, asset], market[-253:-1])
        )
        coefficient = np.linalg.lstsq(design, returns[-252:, asset], rcond=None)[0][1]
        expected = coefficient * leaders[-1]
        actual = result.loc[result.asset_id.eq(f"s{asset:02}"), LEADER].iloc[-1]
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
    # Current size cannot select current leaders: only yesterday's known cap can.
    changed = frame.copy()
    changed.loc[changed.timestamp.eq(changed.timestamp.max()), "float_market_cap"] = (
        1e12
    )
    pd.testing.assert_frame_equal(
        result,
        calculate(
            changed, factor_columns=names, history_keys=("asset_id", "listing_date")
        ),
    )
    # Missing selected leaders are unknown, not silently dropped from the basket.
    changed = frame.copy()
    changed.loc[
        changed.asset_id.eq("s09") & changed.timestamp.eq(changed.timestamp.max()),
        "period_return",
    ] = np.nan
    latest = calculate(changed, factor_columns=(LEADER,))
    assert pd.isna(latest.loc[latest.asset_id.eq("s00"), LEADER].iloc[-1])
    assert pd.notna(latest.loc[latest.asset_id.eq("s09"), LEADER].iloc[-1])


def test_native_single_overnight_spread_retains_unknown_quote_state(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_sources import _calendar_input
    from tests.markets.a_share.test_prepare_selector_clock import _save

    frame = _prices().assign(
        market_id="a_share",
        market_type=1,
        listing_date=pd.Timestamp("2000-01-01"),
        filling_flag=0,
    )
    frame.loc[40, "filling_flag"] = 2
    source = _save(tmp_path, "market_panel_1d", frame)
    calendar = _calendar_input(tmp_path / "calendar", {1: frame.timestamp})
    name = "overnight_intraday_spread_20d"
    result = prepare(
        {
            "market_id": "a_share",
            "data_id": "price_volume_facts",
            "inputs": {"market_panel_1d": source, "trading_calendar": str(calendar)},
            "factor_columns": [name],
            "time_window": {
                "start": str(frame.timestamp.iloc[39].date()),
                "end": str(frame.timestamp.iloc[61].date()),
            },
            "output_root": str(tmp_path / "spread"),
        }
    )
    saved = read_dataset(result).sort_values("timestamp")[name]
    assert saved.iloc[1:-1].isna().all()
    np.testing.assert_allclose(
        saved.iloc[[0, -1]], 20 * (np.log(1.01) - np.log(0.99)), atol=1e-12
    )


def test_native_prepare_and_assemble_preserve_raw_and_both_treatments(tmp_path):
    from quant_research_platform.contracts.risk import (
        RISK_STYLE_COLUMNS,
        FactorRiskModel,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets import inspect_dataset, read_dataset
    from tests.helpers.a_share_sources import _calendar_input
    from tests.helpers.research_processed_inputs import assembly_config, assembly_source
    from tests.markets.a_share.test_prepare_selector_clock import _save

    frame, returns, _ = _market(820, 120)
    closing = 100 * np.cumprod(1 + returns, axis=0)
    frame = frame.assign(
        market_type=1,
        adj_close=closing.ravel(),
        adj_open=(closing / (1 + returns / 2)).ravel(),
        close=10.0,
        high=10.5,
        limit_up_price=11.0,
        price_limit_applicable=1.0,
        is_trading=True,
        eligible=True,
        total_market_cap=frame.float_market_cap * 2,
    )
    dates = frame.timestamp.drop_duplicates()
    last = dates.iloc[-1]
    event = frame.asset_id.str[1:].astype(int) % 3
    frame.loc[frame.timestamp.eq(last) & event.eq(0), "high"] = 11.0
    frame.loc[frame.timestamp.eq(last) & event.eq(1), ["high", "close"]] = 11.0
    asset = frame.asset_id.str[1:].astype(int)
    for index, column in enumerate(RISK_STYLE_COLUMNS):
        frame[column] = np.sin(asset * (0.21 + index * 0.19)) + index + 2
    source = _save(tmp_path, "market_panel_1d", frame)
    calendar = _calendar_input(tmp_path / "calendar", {1: dates})
    window = {"start": str(last.date()), "end": str(last.date())}
    prices = (
        "overnight_intraday_spread_20d",
        "overnight_intraday_spread_60d",
        "limit_up_close_failure",
        "limit_up_streak",
    )
    inputs = {"market_panel_1d": source, "trading_calendar": str(calendar)}
    outputs = {}
    for data_id, names in (
        ("price_volume_facts", prices),
        ("market_structure_factors", (LIQUIDITY, DOWNSIDE, LEADER)),
    ):
        outputs[data_id] = str(
            prepare(
                dict(
                    market_id="a_share",
                    data_id=data_id,
                    inputs=inputs,
                    time_window=window,
                    factor_columns=list(names),
                    output_root=str(tmp_path / (data_id + "-output")),
                )
            )
        )
        saved = read_dataset(outputs[data_id]).sort_values("asset_id")
        expected = (
            calculate_price_volume_factors(frame, factor_columns=names)
            if data_id == "price_volume_facts"
            else calculate(frame, factor_columns=names)
        )
        expected = expected.loc[expected.timestamp.eq(last)].sort_values("asset_id")
        np.testing.assert_allclose(
            saved[list(names)], expected[list(names)], rtol=1e-9, atol=1e-11
        )
    context, _ = assembly_source(
        tmp_path, frame.loc[frame.timestamp.eq(last)], data_id="context"
    )
    for names, clip in (
        ((LIQUIDITY, DOWNSIDE, LEADER, *prices[:2]), "mad"),
        (prices[2:], "none"),
    ):
        assembled = assemble(
            assembly_config(
                tmp_path,
                context,
                inputs={"context": str(context), **outputs},
                time_window=window,
                factor_columns=list(names),
                filter_columns=["eligible"],
                feature_recipe="a_share_canonical",
                feature_groups=["raw", "pre_neutral", "processed"],
                feature_transform={"risk_neutralize": True, "clip": clip},
                output_root=str(tmp_path / ("assembled-" + clip)),
            )
        )
        saved = read_dataset(assembled)
        spec = inspect_dataset(assembled).spec
        transform = spec["feature_transforms"][0]
        assert transform["clip"] == clip
        assert transform["risk_model"] == FactorRiskModel().to_dict()
        assert spec["label_columns"] == []
        for name in names:
            assert saved[name].notna().all()
            assert saved["feature_pre_neutral_" + name].notna().all()
            assert saved["feature_" + name].notna().all()
        if clip == "none":
            assert set(saved.limit_up_close_failure) == {0, 1}
            values = saved.limit_up_close_failure.to_numpy()
            expected = (values - values.mean()) / values.std()
            np.testing.assert_allclose(
                saved.feature_pre_neutral_limit_up_close_failure, expected
            )
