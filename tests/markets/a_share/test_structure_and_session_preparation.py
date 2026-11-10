"""Native source-to-panel preparation for structure context and session labels."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.market_structure import calculate
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_interim_frames,
    _write_inputs,
)


def test_native_structure_and_intraday_targets_keep_history_and_maturity(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=66).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "panel", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    panel = read_dataset(inputs["market_panel_1d"])
    structure = read_dataset(
        _prepare(
            tmp_path / "structure",
            "market_structure_factors",
            inputs,
            dates[60],
            dates[64],
        )
    )
    direct = calculate(panel, history_keys=("asset_id", "listing_date"))
    expected = (
        direct.loc[direct.timestamp.between(dates[60], dates[64])]
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    actual = (
        structure.loc[:, expected.columns]
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    labels_root = _prepare(
        tmp_path / "session", "session_return_labels", inputs, dates[60], dates[64]
    )
    labels = read_dataset(labels_root)
    target = "next_session_intraday_return"
    assert labels[target].notna().all()
    first = labels.loc[labels.timestamp.eq(dates[60])].set_index("asset_id")
    following = panel.loc[panel.timestamp.eq(dates[61])].set_index("asset_id")
    pd.testing.assert_series_equal(
        first[target],
        (following.adj_close / following.adj_open - 1).rename(target),
        check_dtype=False,
    )
    maturity = pd.Timestamp(dates[61]) + pd.Timedelta(hours=7)
    assert first[f"{target}_available_at"].eq(maturity).all()
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.storage.datasets.reading import open_dataset_session

    source = open_dataset_session(labels_root)
    semantics = source.manifest["producer_diagnostics"]["label_semantics"][target]
    assert semantics["entry_offset_sessions"] == semantics["exit_offset_sessions"] == 1
    assert semantics["price_basis"] == "real_observed_adjusted_open_close"
    assembled = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "intraday_labels",
            "inputs": {"labels": str(labels_root)},
            "factor_columns": [],
            "feature_columns": [],
            "feature_recipe": None,
            "label_columns": [target],
            "time_window": {
                "start": str(dates[60].date()),
                "end": str(dates[64].date()),
            },
            "output_root": str(tmp_path / "labels_assembled"),
        }
    )
    saved = open_dataset_session(assembled)
    assert (
        saved.manifest["producer_diagnostics"]["label_semantics"][target] == semantics
    )


def test_daily_factor_units_follow_economic_quantities():
    from quant_research_platform.markets.implementations.a_share.processed.catalog import (  # noqa: E501
        processed_catalog,
    )
    from quant_research_platform.research.factors.price_volume.definitions import (
        PRICE_VOLUME_LIBRARY,
        PRICE_VOLUME_PATH_FACTORS,
    )

    catalog = processed_catalog()
    units = catalog.get("price_volume_facts").column_units
    assert set(units) == set((*PRICE_VOLUME_LIBRARY, *PRICE_VOLUME_PATH_FACTORS))
    for factor, unit in {
        "amount_volatility_20d": "CNY",
        "share_price": "CNY_per_share",
        "volatility_20d": "decimal_return",
        "turnover_volatility_20d": "share_turnover_fraction",
        "amihud_20d": "decimal_return_per_CNY",
        "volatility_compression_20_60": "dimensionless",
        "peak_age_20d": "trading_sessions",
    }.items():
        assert units[factor] == unit
    assert catalog.get("market_panel_1d").column_units["volume"] == "shares"


@pytest.mark.parametrize("unknown", ("exit", "listing"))
def test_structure_lifecycle_history_survives_publication_and_assembly(
    tmp_path, unknown
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.factors.market_structure import DESCRIPTIONS

    dates = pd.bdate_range("2024-01-02", periods=81).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    stocks = frames["stock_master"].copy()
    stocks["timestamp"] = stocks.listing_date
    own = stocks.loc[stocks.source_asset_id.eq("000001")]
    unknown_state = (
        {"delisting_date_unresolved": True}
        if unknown == "exit"
        else {"listing_date": pd.NaT}
    )
    frames["stock_master"] = pd.concat(
        [
            stocks,
            own.assign(timestamp=dates[22], **unknown_state),
            own.assign(timestamp=dates[27], delisting_date_unresolved=False),
        ],
        ignore_index=True,
    )
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "panel", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    panel = read_dataset(inputs["market_panel_1d"])
    axis = producer._calendar_panel_axis(panel, frames["trading_calendar"])
    own_axis = axis.loc[axis.asset_id.eq("000001")].sort_values("timestamp")
    assert own_axis.groupby("_calendar_segment").size().tolist() == (
        [81] if unknown == "exit" else [22, 1, 1, 1, 1, 1, 54]
    )
    root = _prepare(
        tmp_path / "structure", "market_structure_factors", inputs, dates[0], dates[-1]
    )
    structure = read_dataset(root)
    own = structure.loc[structure.asset_id.eq("000001")].set_index("timestamp")
    # Exit uncertainty does not erase observed prices. Unknown identity does:
    # the recovered listing needs twenty preceding closes again from day 27.
    position = np.arange(len(dates))
    valid = position >= 20
    if unknown == "listing":
        valid &= (position < 22) | (position >= 47)
    assert own.industry_return_relative_20d.notna().tolist() == valid.tolist()
    day = position[valid]
    own_return = (11.0 + day) / (11.0 + day - 20) - 1
    peer_return = (
        sum((initial + day) / (initial + day - 20) - 1 for initial in (12.0, 13.0)) / 2
    )
    np.testing.assert_allclose(
        own.loc[dates[valid], "industry_return_relative_20d"],
        own_return - peer_return,
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        own.loc[dates[22:27], "industry_amount_share"],
        (11.0 + position[22:27]) / (36.0 + 3 * position[22:27]),
        rtol=1e-12,
        atol=1e-14,
    )
    saved = read_dataset(
        assemble(
            {
                "market_id": "a_share",
                "dataset_id": "structure",
                "inputs": {"structure": str(root)},
                "factor_columns": list(DESCRIPTIONS),
                "label_columns": [],
                "feature_groups": ["raw"],
                "feature_recipe": None,
                "time_window": {
                    "start": str(dates[0].date()),
                    "end": str(dates[-1].date()),
                },
                "output_root": str(tmp_path / "assembled"),
            }
        )
    )
    pd.testing.assert_frame_equal(saved.loc[:, structure.columns], structure)
