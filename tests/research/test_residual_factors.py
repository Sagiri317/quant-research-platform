"""Independent residual arithmetic, causal exposures, and native preparation."""

from dataclasses import replace

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.factors.residual import (
    RESIDUAL_FACTOR_WINDOWS,
    _residual_drawdown,
    calculate_residual_factors,
)
from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)


def _panel():
    signs = 1.0 - 2.0 * ((np.arange(128)[:, None] >> np.arange(7)) & 1)
    dates = pd.bdate_range("2024-01-02", periods=70).as_unit("ns")
    amplitude = (np.arange(len(dates)) + 1) / 10000
    frames = []
    for day, timestamp in enumerate(dates):
        frames.append(
            pd.DataFrame(
                {
                    "market_id": "a_share",
                    "asset_id": [f"a{i:03d}" for i in range(128)],
                    "timestamp": timestamp,
                    "listing_date": pd.Timestamp("2010-01-01"),
                    "market_type": 4,
                    "is_a_share_scope": True,
                    "is_st": False,
                    "is_pt": False,
                    "has_252_trading_sessions": True,
                    "is_trading": True,
                    "industry_standard_id": "test",
                    "industry_level1_code": np.where(signs[:, 6] > 0, "A", "B"),
                    "total_market_cap": 1e9,
                    **{
                        column: signs[:, i]
                        for i, column in enumerate(RISK_STYLE_COLUMNS)
                    },
                    "period_return": 0.01
                    + 0.002 * signs[:, :5].sum(axis=1)
                    + amplitude[day] * signs[:, 5],
                }
            )
        )
    return pd.concat(frames, ignore_index=True), dates, signs[:, 5], amplitude


def _calculate(frame):
    return calculate_residual_factors(
        frame,
        style_columns=RISK_STYLE_COLUMNS,
        factor_columns=tuple(RESIDUAL_FACTOR_WINDOWS),
        history_keys=("asset_id", "listing_date"),
        min_fresh_fraction=0.75,
    )


def test_residual_windows_use_prior_exposures_and_actual_lifecycle():
    panel, dates, signs, amplitude = _panel()
    calculated = _calculate(panel)
    for column, window in RESIDUAL_FACTOR_WINDOWS.items():
        assert calculated.loc[calculated.timestamp < dates[window], column].isna().all()
        last = calculated.loc[calculated.timestamp.eq(dates[-1])].sort_values(
            "asset_id"
        )
        expected = (
            np.full(128, np.std(amplitude[-window:], ddof=0))
            if "volatility" in column
            else signs * amplitude[-window:].sum()
        )
        if "reversal" in column:
            expected *= -1
        if "drawdown" in column:
            # Each synthetic residual path is monotone with its asset's sign.
            expected = np.maximum(-signs * amplitude[-window:].sum(), 0)
        np.testing.assert_allclose(last[column], expected, rtol=1e-11, atol=1e-13)
    future = panel.copy()
    future.loc[future.timestamp >= dates[65], list(RISK_STYLE_COLUMNS)] = np.nan
    changed = _calculate(future)
    pd.testing.assert_frame_equal(
        calculated.loc[calculated.timestamp <= dates[65]],
        changed.loc[changed.timestamp <= dates[65]],
    )
    missing = panel.copy()
    missing.loc[
        missing.asset_id.eq("a000") & missing.timestamp.ge(dates[30]), "period_return"
    ] = np.nan
    missing.loc[
        missing.asset_id.eq("a001") & missing.timestamp.ge(dates[60]), "listing_date"
    ] = dates[60]
    result = _calculate(missing)
    assert (
        result.loc[
            result.asset_id.isin(["a000", "a001"]) & result.timestamp.eq(dates[-1]),
            "risk_residual_volatility_60d",
        ]
        .isna()
        .all()
    )


def test_additive_residual_drawdown_has_a_zero_origin_and_no_gap_compression():
    values = np.array([0.01, 0.02, -0.01, -0.03, np.nan, 0.01, 0.02, 0.03])
    actual = _residual_drawdown(values, 4)
    assert np.isnan(actual[:3]).all()
    # Additive path 1%,3%,2%,-1% has a 4pp current residual drawdown.
    np.testing.assert_allclose(actual[3], 0.04, rtol=1e-14, atol=1e-16)
    assert np.isnan(actual[4:]).all()
    assert np.isnan(_residual_drawdown(values[:2], 4)).all()


def test_residual_factors_prepare_and_native_read_keep_window_and_units(tmp_path):
    panel, dates, _, _ = _panel()
    catalog = processed_catalog()
    inputs = {}
    for name in ("market_panel_1d", "price_volume_facts", "market_model_factors"):
        spec = catalog.get(name)
        columns = [column for column in spec.required_columns if column in panel]
        spec = replace(
            spec,
            column_roles={column: spec.column_roles[column] for column in columns},
            column_units={
                column: unit
                for column, unit in spec.column_units.items()
                if column in columns
            },
        )
        root = tmp_path / name
        write_dataset(root, panel[columns], spec=spec)
        inputs[name] = str(root)
    calendar_dates = pd.date_range(dates[0], dates[-1]).as_unit("ns")
    calendar = pd.DataFrame(
        {
            "calendar_date": calendar_dates,
            "timestamp": calendar_dates,
            "source_market_type": 4,
            "is_trading_day": calendar_dates.dayofweek < 5,
            "source_day_of_week": calendar_dates.dayofweek,
            "source_state": np.where(calendar_dates.dayofweek < 5, "O", "C"),
        }
    )
    root = tmp_path / "calendar"
    write_dataset(root, calendar, spec=interim_catalog().get("trading_calendar"))
    inputs["trading_calendar"] = str(root)
    output = prepare(
        {
            "market_id": "a_share",
            "data_id": "risk_residual_factors",
            "inputs": inputs,
            "output_root": str(tmp_path / "output"),
            "time_window": {
                "start": str(dates[-1].date()),
                "end": str(dates[-1].date()),
            },
        }
    )
    saved = read_dataset(output).sort_values("asset_id").reset_index(drop=True)
    assert inspect_dataset(output).spec["column_units"] == {
        column: "decimal_return" for column in RESIDUAL_FACTOR_WINDOWS
    }
    expected = _calculate(panel)
    expected = (
        expected.loc[expected.timestamp.eq(dates[-1])]
        .sort_values("asset_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(saved.drop(columns="market_id"), expected)


def test_residual_factor_clipping_uses_prior_complete_cross_section():
    panel, dates, _, _ = _panel()
    dates = dates[:6]
    panel = panel.loc[panel.timestamp.isin(dates)].copy()
    values = np.r_[np.linspace(-1.0, 1.0, 100), np.linspace(3.0, 20.0, 28)]
    responses = 0.0001 * values**2 + 0.001 * np.cos(np.arange(128))
    panel["industry_level1_code"] = "one"
    panel["log_total_market_cap"] = np.tile(values, len(dates))
    panel["period_return"] = np.tile(responses, len(dates))
    # The final day's missing asset was observable in the entire prior cross section.
    panel = panel.loc[~(panel.timestamp.eq(dates[-1]) & panel.asset_id.eq("a127"))]
    actual = calculate_residual_factors(
        panel,
        style_columns=("log_total_market_cap",),
        factor_columns=("risk_residual_reversal_5d",),
        history_keys=("asset_id", "listing_date"),
        min_fresh_fraction=0.75,
    )
    median = np.median(values)
    bound = 3 * 1.4826 * np.median(np.abs(values - median))
    clipped = np.clip(values, median - bound, median + bound)
    design = np.column_stack((np.ones(128), (clipped - clipped.mean()) / clipped.std()))
    full_residual = (
        responses - design @ np.linalg.lstsq(design, responses, rcond=None)[0]
    )
    final_residual = (
        responses[:-1]
        - design[:-1] @ np.linalg.lstsq(design[:-1], responses[:-1], rcond=None)[0]
    )
    last = actual.loc[actual.timestamp.eq(dates[-1])].sort_values("asset_id")
    np.testing.assert_allclose(
        last.risk_residual_reversal_5d,
        -(4 * full_residual[:-1] + final_residual),
        rtol=1e-11,
        atol=1e-13,
    )
