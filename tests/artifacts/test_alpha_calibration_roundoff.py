from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.research.portfolios.accounting import (
    calibrate_expected_alpha,
    frozen_alpha_calibration,
)
from quant_research_platform.research.portfolios.production import (
    _ALPHA_DIAGNOSTICS,
    _completed_calibration_rows,
)
from quant_research_platform.research.predictions.outputs import _add_a_share_scores
from quant_research_platform.storage.identity import native_filesystem_path
from tests.artifacts.test_portfolio_ledger_artifact import (
    _alpha_envelope,
    _alpha_payload,
    _session,
)
from tests.helpers.portfolio_ledger import write_test_sessions


@pytest.mark.parametrize("intercept,noise", ((0.25, 0.0), (0.25, 0.001), (0.01, 0.0)))
def test_rank_and_open_return_roundoff_survives_final_json_and_freeze(
    tmp_path, intercept, noise
):
    dates = pd.date_range("2023-07-07", periods=8, freq="7D")
    rows = []
    for offset, timestamp in enumerate(dates):
        count = offset + 3
        assets = [f"synthetic_{index:06d}" for index in range(count)]
        prediction = pd.DataFrame(
            {"timestamp": timestamp, "family_id": "a_share", "asset_id": assets}
        )
        _add_a_share_scores(prediction, np.arange(count, dtype="float64"))
        alpha = prediction.set_index("asset_id")["alpha_score"]
        returns = (
            intercept
            + 0.2 * alpha.to_numpy()
            + noise * ((offset + np.arange(1, count + 1)) % 3 - 1)
        )
        alpha.index = pd.MultiIndex.from_arrays(
            (assets, [dates[0]] * count), names=("asset_id", "listing_date")
        )
        previous_open = pd.Series(1.0, index=alpha.index)
        endpoint = pd.DataFrame(
            {"asset_id": assets, "listing_date": dates[0], "adj_open": 1.0 + returns}
        )
        rows.extend(
            _completed_calibration_rows((alpha, previous_open, timestamp), endpoint)
        )
    inputs = pd.DataFrame(rows)
    assert len(inputs) == 52
    intervals = pd.DataFrame(
        {
            "timestamp": dates,
            "entry_at": dates + pd.Timedelta(days=3),
            "available_at": dates + pd.Timedelta(days=10),
        }
    )
    calibration = calibrate_expected_alpha(
        inputs, as_of=pd.Timestamp("2023-12-31"), scheduled_intervals=intervals
    )
    actual = inputs.holding_return.to_numpy()
    fitted = calibration.intercept + calibration.slope * inputs.alpha_score.to_numpy()
    within_resolution = np.abs(actual - fitted) <= 8 * np.finfo(
        np.float64
    ).eps * np.maximum(np.abs(actual), np.abs(fitted))
    numerical_zero = intercept == 0.25 and noise == 0.0
    assert bool(within_resolution.all()) == numerical_zero
    assert calibration.status == "DEFINED"
    assert calibration.inference_status == (
        "UNDEFINED" if numerical_zero else "DEFINED"
    )
    if numerical_zero:
        assert calibration.slope_standard_error == 0.0
        assert calibration.inference_reason == "alpha_calibration_zero_standard_error"
        assert (
            calibration.slope_t_statistic
            is calibration.slope_pvalue
            is calibration.slope_ci_low
            is calibration.slope_ci_high
            is None
        )
        assert calibration.weighted_r2 == 1.0
    else:
        assert calibration.slope_standard_error > 0.0
    frame = _session(
        "2023-09-01",
        "2023-09-04",
        pre_asset=0.0,
        target_asset=100.0,
        post_asset=100.0,
        pre_cash=100.0,
        target_cash=0.0,
        post_cash=0.0,
    )
    for column, field in _ALPHA_DIAGNOSTICS.items():
        frame[column] = pd.Series([None] * len(frame), dtype=object)
        frame.loc[frame.asset_id.eq("__cash__"), column] = getattr(calibration, field)
    envelope = _alpha_envelope("2023-12-31")
    root = write_test_sessions(
        tmp_path / "portfolio",
        envelope=envelope,
        ledger_chunks=(frame,),
        final_calibration=lambda: _alpha_payload(calibration, envelope),
    )
    before = Path(
        native_filesystem_path(root / "final_alpha_calibration.json")
    ).read_bytes()
    saved = json.loads(before)
    assert (
        saved["calibration"]["slope_standard_error"] == calibration.slope_standard_error
    )
    assert saved["calibration"]["slope_t_statistic"] == calibration.slope_t_statistic
    artifact = read_portfolio_ledger_artifact(root, expected_identity=root.name)
    frozen, _ = frozen_alpha_calibration(
        artifact.final_alpha_calibration,
        cutoff=pd.Timestamp("2023-12-31"),
        selection_group="global",
    )
    assert frozen == calibration
    summary = summarize_portfolio_ledger(artifact.capital, artifact.iter_chunks())
    assert (
        summary["alpha_calibration"]["inference_status"] == calibration.inference_status
    )
    assert (
        Path(native_filesystem_path(root / "final_alpha_calibration.json")).read_bytes()
        == before
    )
