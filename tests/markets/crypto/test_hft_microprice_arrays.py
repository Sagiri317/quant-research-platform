from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_factor_frames,
)
from quant_research_platform.research.models.scaler import ModelMatrixScaler


def _panel(bid, ask, *, asset="btc_usdt"):
    size = len(bid)
    return pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": asset,
            "timestamp_ms": np.arange(size) * 100,
            "segment_id": "a",
            "mid": 60000.0 if asset == "btc_usdt" else 3000.0,
            "book_spread": 0.01,
            "bid_depth_1": bid,
            "ask_depth_1": ask,
        }
    )


@pytest.mark.parametrize(
    "column,asset",
    [
        ("microprice", "btc_usdt"),
        ("microprice", "eth_usdt"),
        ("microprice_deviation", "btc_usdt"),
    ],
)
@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("extended", [True, False])
def test_microprice_selected_inputs_and_batches(
    column, asset, batch, extended, monkeypatch
):
    if not extended:
        monkeypatch.setattr(np, "longdouble", np.float64)
    bid = np.arange(1, 13) * 0.001
    ask = bid / 2
    ask[:3] = bid[:3]
    ask[3:6] = np.nextafter(bid[3:6], np.inf)
    frame = _panel(bid, ask, asset=asset)
    frame.loc[6:, "segment_id"] = "b"
    frame.loc[6:, "timestamp_ms"] += 100
    frame.loc[8, "bid_depth_1"] = np.nan
    frame.loc[9, ["bid_depth_1", "ask_depth_1"]] = 0.0
    frame.loc[10, "book_spread"] = 0.0
    frame.loc[11, "mid"] = np.nan
    if column == "microprice_deviation":
        frame = frame.drop(columns="mid")
    with np.errstate(all="raise"):
        result = pd.concat(
            iter_factor_frames(
                (frame.iloc[i : i + batch] for i in range(0, len(frame), batch)),
                columns=(column,),
            ),
            ignore_index=True,
        )
    expected = []
    for row in frame.to_dict("records"):
        if (
            not np.isfinite(row["bid_depth_1"])
            or row["bid_depth_1"] + row["ask_depth_1"] == 0
            or row["book_spread"] <= 0
            or (column == "microprice" and not np.isfinite(row["mid"]))
        ):
            expected.append(np.nan)
            continue
        bid, ask, spread = (
            Fraction(row[name])
            for name in ("bid_depth_1", "ask_depth_1", "book_spread")
        )
        exact = (bid - ask) / (bid + ask) * spread / 2
        if column == "microprice":
            exact += Fraction(row["mid"])
        expected.append(float(exact))
    np.testing.assert_array_equal(result[column].isna(), np.isnan(expected))
    # Relative tolerance keeps real sub-tick deviations observable near zero.
    np.testing.assert_allclose(result[column], expected, rtol=2e-12, atol=0)
    pd.testing.assert_frame_equal(
        result[["market_id", "asset_id", "timestamp_ms"]],
        frame[["market_id", "asset_id", "timestamp_ms"]],
        check_exact=True,
    )
    pd.testing.assert_series_equal(result.segment_id, frame.segment_id)


@pytest.mark.parametrize("boundary_pair", [False, True])
def test_microprice_constant_books_do_not_create_standardized_signals(boundary_pair):
    ask = np.arange(1, 51) * 0.001
    bid = ask * 2
    if boundary_pair:
        # Both exact expressions round to the same factor, despite differing sizes.
        bid, ask = [2.4573, 0.0021], [1.6382, 0.0014]
    result = pd.concat(
        iter_factor_frames([_panel(bid, ask)], columns=("microprice_deviation",)),
        ignore_index=True,
    )[["microprice_deviation"]]
    assert result.microprice_deviation.nunique() == 1
    scaler = ModelMatrixScaler.fit(result)
    assert not scaler.transform(result).to_numpy().any()


@pytest.mark.parametrize("depth", [1, 5, 10])
def test_plain_depth_projection_preserves_ties_and_real_changes(depth):
    from quant_research_platform.contracts.signals import SignalSpec
    from quant_research_platform.research.signals.normalization import (
        normalize_signal_frame,
    )

    bid = np.array([0.2, 0.6, 2.2, 0.6 + 1e-12, 0, np.nan, -1])
    ask = np.array([0.1, 0.3, 1.1, 0.3, 0, 1, 1])
    frame = _panel(bid, ask).drop(columns=["mid", "book_spread"])
    frame = frame.rename(
        columns={
            "bid_depth_1": f"bid_depth_{depth}",
            "ask_depth_1": f"ask_depth_{depth}",
        }
    )
    frame.loc[3:, "timestamp_ms"] += 100
    frame.loc[5:, "segment_id"] = "b"
    column = f"imbalance_{depth}"
    result = pd.concat(
        iter_factor_frames(
            (frame.iloc[i : i + 2] for i in range(0, len(frame), 2)),
            columns=(column,),
        ),
        ignore_index=True,
    )
    expected = [
        float((Fraction(b) - Fraction(a)) / (Fraction(b) + Fraction(a)))
        for b, a in zip(bid[:4], ask[:4], strict=True)
    ]
    np.testing.assert_allclose(result[column].iloc[:4], expected, rtol=2e-12, atol=0)
    assert result[column].iloc[4:].isna().all()
    assert result[column].iloc[:3].nunique() == 1
    assert result[column].iloc[3] > result[column].iloc[0]
    identity = ["market_id", "asset_id", "timestamp_ms", "segment_id"]
    pd.testing.assert_frame_equal(result[identity], frame[identity], check_exact=True)
    signals = result.iloc[:4].assign(timestamp=pd.Timestamp("2024-01-01"))
    for transform in ("rank", "zscore"):
        normalized = normalize_signal_frame(
            signals,
            (
                SignalSpec(
                    signal_id="signal",
                    source_kind="factor",
                    source_column=column,
                    transform=transform,
                ),
            ),
        ).signal
        assert normalized.iloc[:3].nunique() == 1
        assert normalized.iloc[3] > normalized.iloc[0]
    scaler = ModelMatrixScaler.fit(result.loc[:2, [column]])
    assert not scaler.transform(result.loc[:2, [column]]).to_numpy().any()
    assert scaler.transform(result.loc[3:3, [column]]).iloc[0, 0] > 0
