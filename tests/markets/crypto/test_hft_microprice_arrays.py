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
