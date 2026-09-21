"""Information assets use comparable matured features without selecting execution."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.hft import (
    iter_cross_asset_factor_frames,
    iter_factor_frames,
)
from tests.markets.crypto.test_spot_context import panel


def test_cross_asset_units_lag_segment_and_chunk_causality():
    btc = panel(spot=True)
    sol = btc.copy()
    sol["asset_id"] = "sol_usdt"
    sol[["mid", "book_spread"]] *= 0.01
    quantity = [
        key
        for key in sol
        if key.startswith(("bid_depth", "ask_depth", "observed_"))
        and key != "observed_utc"
    ]
    sol[quantity] *= 100
    sol["book_ofi_100ms"] *= 100
    btc_state = next(iter_factor_frames([btc]))
    sol_state = next(iter_factor_frames([sol]))
    targets = btc[["timestamp"]]

    def compute(target_frames, btc_frames, sol_frames):
        return pd.concat(
            iter_cross_asset_factor_frames(
                target_frames,
                information_frames={"btc_usdt": btc_frames, "sol_usdt": sol_frames},
                max_age_ms=200,
                batch_rows=7,
            ),
            ignore_index=True,
        )

    full = compute([targets], [btc_state], [sol_state])
    split = compute(
        [targets.iloc[i : i + 3] for i in range(0, 20, 3)],
        [btc_state.iloc[i : i + 4] for i in range(0, 20, 4)],
        [sol_state.iloc[i : i + 6] for i in range(0, 20, 6)],
    )
    pd.testing.assert_frame_equal(full, split, check_exact=True)
    for metric in ("return_100ms", "return_500ms", "normalized_ofi_500ms", "tfi_500ms"):
        np.testing.assert_allclose(
            full["btc_" + metric],
            full["sol_" + metric],
            rtol=1e-11,
            atol=1e-14,
            equal_nan=True,
        )
    assert full.sol_return_500ms.iloc[5] == pytest.approx(0.025)
    assert full.sol_lagged_return_100ms.iloc[5] == sol_state.return_100ms.iloc[4]
    assert full.sol_lagged_tfi_500ms.iloc[5] == 0.25
    changed = sol_state.copy()
    changed.loc[9:, "return_100ms"] = 99
    pd.testing.assert_frame_equal(
        full.iloc[:9],
        compute([targets], [btc_state], [changed]).iloc[:9],
        check_exact=True,
    )
    gap = sol_state.drop(index=range(8, 14)).copy()
    gap.loc[14, "return_100ms"] = np.nan
    limited = compute([targets], [btc_state], [gap])
    assert limited.sol_grid_age_ms.iloc[10] == 300
    assert pd.isna(limited.sol_return_100ms.iloc[10])
    assert pd.isna(limited.sol_lagged_return_100ms.iloc[14])
    assert limited.btc_return_100ms.equals(full.btc_return_100ms)
    missing = compute([targets], [btc_state], [])
    assert missing.filter(regex="^sol_").isna().all().all()
    assert missing.btc_return_100ms.equals(full.btc_return_100ms)
