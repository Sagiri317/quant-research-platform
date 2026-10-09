from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_derivative_factor_frames,
    iter_factor_frames,
    iter_panel_frames,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)
from quant_research_platform.research.protocols.crypto_hft import (
    HFT_DERIVATIVE_CHANNELS,
)
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.binance_hft import _coverage, coverage_lookup
from tests.markets.crypto.test_hft import _state, _trade


def _persist(tmp_path, data_id, frames):
    spec = hft_processed_catalog().get(data_id)
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, frames, spec=spec)
    return read_dataset(root)


def _panel(start=0, end=6100, batch=17, extreme=False):
    states = [_state(0), _state(1000, segment_id="s2")]
    if extreme:
        states = [
            replace(
                _state(0, bid_qty=1e308, ask_qty=1e308),
                bids=((950, 1e308),),
                asks=((1050, 1e308),),
            )
        ]
    return list(
        iter_panel_frames(
            states,
            [_trade(i, 8e307, maker_buyer=False) for i in range(100, end + 1, 100)]
            if extreme
            else [],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=start,
            end_ms=end,
            book_coverage={
                "s1": _coverage(0, end + 100) if extreme else _coverage(0, 1000),
                "s2": _coverage(1000, end + 100),
            },
            trade_coverage=coverage_lookup(
                {"trade-connection-1": _coverage(0, end + 100)}
            ),
            batch_rows=batch,
        )
    )


@pytest.mark.parametrize("start,batch", [(0, 1), (0, 71), (1000, 3)])
def test_derivative_independent_state_survives_execution_boundary(
    tmp_path, start, batch
):
    panel = _persist(
        tmp_path / "panel", "binance_futures_l2_panel_100ms", _panel(start, batch=batch)
    )
    events = [
        (
            0,
            "premiumIndex",
            dict(
                markPrice="102",
                indexPrice="101",
                lastFundingRate="0.0001",
                nextFundingTime=3600000,
                time=0,
            ),
            0,
        ),
        (0, "openInterest", dict(openInterest="10", time=0), 0),
        (900_000_000, "forceOrder", {"E": 900, "o": {"S": "SELL", "z": "2"}}, 0),
        (1150_000_000, "forceOrder", {"E": 1150, "o": {"S": "BUY", "z": "6"}}, 0),
    ]
    values = iter_derivative_factor_frames(
        (panel.iloc[i : i + batch] for i in range(0, len(panel), batch)),
        observations=events,
        max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, 10000),
    )
    result = _persist(
        tmp_path / "factor", "binance_futures_derivative_factors_100ms", values
    ).set_index("timestamp")
    row = result.loc[pd.Timestamp(1200, unit="ms")]
    assert row.mark_mid_basis == pytest.approx(0.02)
    assert row.log_open_interest == pytest.approx(np.log(10))
    assert row.premiumIndex_observed_at == pd.Timestamp(0)
    assert row.premiumIndex_age_ms == 1200
    assert row.time_since_observed_force_order == 0.05
    before = result.loc[pd.Timestamp(1100, unit="ms")]
    assert before.time_since_observed_force_order == 0.2
    assert result.open_interest_change_30s.isna().all()
    assert (
        result.loc[pd.Timestamp(6000, unit="ms"), "observed_force_order_imbalance_5s"]
        == 1
    )


@pytest.mark.parametrize("batch", [3, 53])
def test_finite_final_ratios_from_persisted_extreme_panel(tmp_path, batch):
    factors = (
        "top10_depth_per_spread",
        "near_depth_share_1_10",
        "trade_flow_imbalance_100ms",
        "trade_flow_imbalance_500ms",
        "trade_flow_imbalance_1s",
        "trade_flow_imbalance_5s",
        "depth_normalized_ofi_500ms",
        "depth_normalized_ofi_1s",
        "depth_normalized_ofi_5s",
        "taker_buy_pressure_500ms",
        "taker_sell_pressure_500ms",
    )
    frames = _panel(end=5100, batch=batch, extreme=True)
    for frame in frames:
        # Persisted absolute inputs are finite; only their sums exceed float range.
        frame["book_ofi_100ms"] = 8e307
        frame["observed_taker_buy_volume_100ms"] = 8e307
        frame["observed_taker_sell_volume_100ms"] = 4e307
        frame["observed_signed_trade_volume_100ms"] = 4e307
        frame["trade_volume_100ms"] = 1.2e308
    panel = _persist(tmp_path / "panel", "binance_futures_l2_panel_100ms", frames)
    for factor in factors:
        result = pd.concat(
            iter_factor_frames(
                (panel.iloc[i : i + batch] for i in range(0, len(panel), batch)),
                columns=(factor,),
            ),
            ignore_index=True,
        )
        if factor == "top10_depth_per_spread":
            expected = 2e307
        elif factor == "near_depth_share_1_10":
            expected = 1.0
        elif factor.startswith("trade_flow"):
            expected = float(Fraction(4e307) / (Fraction(8e307) + Fraction(4e307)))
        elif factor.startswith("depth_normalized"):
            n = {"500ms": 5, "1s": 10, "5s": 50}[factor.rsplit("_", 1)[1]]
            expected = float(n * Fraction(8e307) / (2 * Fraction(1e308)))
        elif factor == "taker_buy_pressure_500ms":
            expected = float(5 * Fraction(8e307) / Fraction(1e308))
        else:
            expected = float(5 * Fraction(4e307) / Fraction(1e308))
        assert np.isfinite(result[factor].iloc[-1])
        assert result[factor].iloc[-1] == expected


@pytest.mark.parametrize("half_life", [0.0005, 0.001])
@pytest.mark.parametrize("magnitude", [1e150, 1e300])
def test_actual_risk_estimate_retains_tail_after_latest_missing_session(
    half_life, magnitude
):
    config = FactorRiskConfig(
        minimum_cross_section=10,
        minimum_factor_dates=2,
        minimum_specific_dates=2,
        factor_half_life=half_life,
        specific_half_life=half_life,
        annualization=1.0,
        diagonal_shrinkage=0.0,
    )
    state = FactorRiskState(style_columns=(), config=config)
    frame = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{i}" for i in range(10)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "float_market_cap": 1.0,
            "asset_return": np.linspace(0.5 * magnitude, 1.5 * magnitude, 10),
        }
    )
    fitted = state.fit_return_session(frame)
    state.observe_return_session(frame)
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2020-01-02"), asset_return=0.0)
    )
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2020-01-03"), asset_return=np.nan)
    )
    result = state.estimate()
    with localcontext() as ctx:
        ctx.prec = 100
        tail = Decimal(2) ** (-Decimal(1) / Decimal.from_float(half_life))
        w = tail / (1 + tail)
        expected_cov = float(
            w * (1 - w) * Decimal.from_float(float(fitted.factor_returns.iloc[0])) ** 2
        )
        expected_specific = float(
            w * Decimal.from_float(float(fitted.residuals.iloc[0])) ** 2
        )
    assert expected_cov > 0 and expected_specific > 0
    assert result.factor_covariance.iloc[0, 0] == pytest.approx(
        expected_cov, rel=1e-12, abs=0
    )
    assert result.specific_variance.iloc[0] == pytest.approx(
        expected_specific, rel=1e-12, abs=0
    )
    assert np.isfinite(result.factor_covariance).all().all()


@pytest.mark.parametrize("batch", [3, 53])
def test_all_microstructure_factors_persist_finite_ratios_and_fail_true_overflow(
    tmp_path, batch
):
    frames = _panel(end=5100, batch=batch, extreme=True)
    for frame in frames:
        frame["book_ofi_100ms"] = 1e306
    panel = _persist(tmp_path / "panel", "binance_futures_l2_panel_100ms", frames)
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )
    from quant_research_platform.storage.datasets.reading import open_dataset_session

    panel_spec = hft_processed_catalog().get("binance_futures_l2_panel_100ms")
    panel_root = StorageLayout(tmp_path / "panel").processed_data_root(
        panel_spec.market_id, panel_spec.kind, panel_spec.data_id
    )
    factor_spec = hft_processed_catalog().get(
        "binance_futures_microstructure_factors_100ms"
    )
    factors = _persist(
        tmp_path / "factors",
        factor_spec.data_id,
        producer._panel_frame_producer(
            open_dataset_session(panel_root, columns=None, time_window=None),
            transform=iter_factor_frames,
            columns=factor_spec.required_columns,
            asset_ids=("btc_usdt",),
        ),
    )
    assert factors.top10_depth_per_spread.iloc[-1] == 2e307
    assert factors.near_depth_share_1_10.iloc[-1] == 1
    assert factors.trade_flow_imbalance_5s.iloc[-1] == 1
    assert factors.taker_buy_pressure_500ms.iloc[-1] == 4
    panel["book_ofi_100ms"] = 8e307
    with pytest.raises(OverflowError):
        list(iter_factor_frames([panel], columns=("ofi_500ms",)))


def test_force_order_ratio_uses_finite_final_mass_and_preserves_epoch_and_age():
    panel = pd.concat(_panel(end=6100), ignore_index=True)
    events = [
        (1050_000_000, "forceOrder", {"E": 1050, "o": {"S": "SELL", "z": "1e308"}}, 0),
        (1150_000_000, "forceOrder", {"E": 1150, "o": {"S": "BUY", "z": "1e308"}}, 0),
    ]

    def compute(frames, observations=events, age=10000):
        return pd.concat(
            iter_derivative_factor_frames(
                frames,
                observations=observations,
                max_age_ms=dict.fromkeys(HFT_DERIVATIVE_CHANNELS, age),
            ),
            ignore_index=True,
        )

    wide = compute([panel])
    narrow = compute([panel.loc[panel.timestamp.ge(pd.Timestamp(1100, unit="ms"))]])
    pd.testing.assert_frame_equal(
        wide.loc[wide.timestamp.ge(pd.Timestamp(1100, unit="ms"))].reset_index(
            drop=True
        ),
        narrow,
        check_exact=True,
    )
    mature = wide.set_index("timestamp").loc[pd.Timestamp(6000, unit="ms")]
    assert mature.observed_force_order_imbalance_5s == 0
    stale = compute([panel], age=10)
    assert stale.time_since_observed_force_order.isna().all()
    assert stale.forceOrder_observed_at.notna().any()
    reset = compute([panel], observations=events + [(1250_000_000, None, None, 1)])
    assert (
        reset.loc[
            reset.timestamp.ge(pd.Timestamp(1300, unit="ms")), "forceOrder_observed_at"
        ]
        .isna()
        .all()
    )


@pytest.mark.parametrize("half_life", [0.0005, 0.001, 63.0])
def test_ewma_covariance_is_invariant_to_common_missing_age(half_life):
    from quant_research_platform.research.portfolios.risk import _ewma_covariance

    arguments = dict(
        values=np.array([[1e150, -0.5e150], [0.0, 0.0]]),
        half_life=half_life,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
        columns=("a", "b"),
    )
    first = _ewma_covariance(ages=np.array([2, 1]), **arguments)
    shifted = _ewma_covariance(ages=np.array([2002, 2001]), **arguments)
    pd.testing.assert_frame_equal(first, shifted, check_exact=True)


def test_unrepresentable_final_risk_fails_explicitly():
    from quant_research_platform.research.portfolios.risk import _ewma_covariance

    with pytest.raises(ValueError, match="factor_risk_covariance_non_finite"):
        _ewma_covariance(
            np.array([[1e308], [-1e308]]),
            ages=np.array([1, 0]),
            half_life=63.0,
            diagonal_shrinkage=0.2,
            annualization_factor=252.0,
            columns=("a",),
        )
