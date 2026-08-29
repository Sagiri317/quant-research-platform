from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.factors import FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as a_share_factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as a_share_factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    market_panel_data_id,
)
from quant_research_platform.research.factors.price_volume.calculations import (
    calculate_price_volume_factors,
)
from quant_research_platform.research.factors.price_volume.columns import (
    PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS,
    price_volume_window_column,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    price_volume_factor_definitions,
)
from quant_research_platform.research.factors.price_volume.presets import (
    CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET,
    PRICE_VOLUME_SMOKE_PRESET,
    PRICE_VOLUME_STANDARD_PRESET,
    SMOKE_PRICE_VOLUME_FACTORS,
    price_volume_factor_columns,
)
from quant_research_platform.research.factors.price_volume.requirements import (
    price_volume_factor_input_columns,
    unsupported_price_volume_factors,
)
from tests.helpers.price_volume_reference import (
    assert_frame_matches_reference,
    reference_price_volume_factors,
)
from tests.helpers.synthetic_a_share import make_synthetic_factor_panel

MARKET_PANEL_DATA_ID = market_panel_data_id()
MOMENTUM_20D_COLUMN = price_volume_window_column("mom", 20)
REVERSAL_1D_COLUMN = price_volume_window_column("rev", 1)
FLOW_STANDARD_PRICE_VOLUME_FACTORS = (
    "amount_chg_20d",
    "amount_chg_60d",
    "amount_shock_20d",
    "amount_shock_60d",
    "amount_vol_20d",
    "amount_vol_60d",
    "volume_chg_20d",
    "volume_chg_60d",
    "volume_shock_20d",
    "volume_shock_60d",
    "ret_amount_corr_20d",
    "ret_amount_corr_60d",
    "ret_volume_corr_20d",
    "ret_volume_corr_60d",
    "ret_turnover_corr_20d",
    "ret_turnover_corr_60d",
    "trend_efficiency_20d",
    "trend_efficiency_60d",
    "signed_amount_flow_20d",
    "signed_amount_flow_60d",
    "ret_amount_imbalance_20d",
    "ret_amount_imbalance_60d",
)


def test_price_volume_standard_preset_defines_catalog_contract() -> None:
    columns = price_volume_factor_columns(preset=PRICE_VOLUME_STANDARD_PRESET)

    assert columns
    assert "mom_252d" in columns
    assert "rev_252d" in columns
    assert set(FLOW_STANDARD_PRICE_VOLUME_FACTORS).issubset(columns)
    assert price_volume_factor_columns() == columns
    assert price_volume_factor_columns(preset=PRICE_VOLUME_CORE_PRESET) == (
        CORE_PRICE_VOLUME_FACTORS
    )
    assert price_volume_factor_columns(preset=PRICE_VOLUME_SMOKE_PRESET) == (
        SMOKE_PRICE_VOLUME_FACTORS
    )
    assert not set(FLOW_STANDARD_PRICE_VOLUME_FACTORS).intersection(
        price_volume_factor_columns(preset=PRICE_VOLUME_CORE_PRESET)
    )
    assert not set(FLOW_STANDARD_PRICE_VOLUME_FACTORS).intersection(
        price_volume_factor_columns(preset=PRICE_VOLUME_SMOKE_PRESET)
    )
    with pytest.raises(ValueError, match="invalid_price_volume_preset"):
        price_volume_factor_columns(preset="unknown")


def test_price_volume_factor_definitions_are_output_and_input_driven() -> None:
    definitions = price_volume_factor_definitions(
        market_id=MarketId("a_share"),
        required_input_data_id=MARKET_PANEL_DATA_ID,
    )
    columns = price_volume_factor_columns()
    by_id = {definition.factor_id: definition for definition in definitions}

    assert tuple(definition.output_column for definition in definitions) == columns
    assert set(by_id) == set(columns)
    for definition in definitions:
        assert definition.factor_id == definition.output_column
        assert definition.input_columns == price_volume_factor_input_columns(
            definition.factor_id
        )
        assert definition.base_input_columns == definition.input_columns
        assert definition.required_input_data_id == MARKET_PANEL_DATA_ID
        assert definition.family == "price_volume"
        if definition.factor_id in SMOKE_PRICE_VOLUME_FACTORS:
            assert definition.factor_set is FactorSet.SMOKE
        elif definition.factor_id in CORE_PRICE_VOLUME_FACTORS:
            assert definition.factor_set is FactorSet.CORE
        else:
            assert definition.factor_set is FactorSet.STANDARD
            assert definition.to_manifest()["factor_set"] == "standard"

    assert by_id[REVERSAL_1D_COLUMN].input_columns == ("period_return",)
    assert by_id[MOMENTUM_20D_COLUMN].input_columns == ("adj_close",)
    assert by_id["amihud_20d"].input_columns == ("period_return", "amount")
    assert by_id["amplitude_20d"].input_columns == (
        "high",
        "low",
        "prev_close",
    )
    assert by_id["amount_chg_20d"].input_columns == ("amount",)
    assert by_id["volume_chg_20d"].input_columns == ("volume",)
    assert by_id["ret_amount_corr_20d"].input_columns == (
        "period_return",
        "amount",
    )
    assert by_id["ret_volume_corr_20d"].input_columns == (
        "period_return",
        "volume",
    )
    assert by_id["ret_turnover_corr_20d"].input_columns == (
        "period_return",
        "turnover",
    )
    assert by_id["trend_efficiency_20d"].input_columns == ("period_return",)
    assert by_id["signed_amount_flow_20d"].input_columns == (
        "period_return",
        "amount",
    )
    assert by_id["ret_amount_imbalance_20d"].input_columns == (
        "period_return",
        "amount",
    )


def test_price_volume_compute_uses_canonical_keys_and_prunes_outputs() -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=260))
    smoke_columns = price_volume_factor_columns(preset="smoke")

    result = calculate_price_volume_factors(panel, factor_columns=smoke_columns)

    assert set(smoke_columns).issubset(result.columns)
    assert not {
        "mom_252d",
        "rev_252d",
        "vol_252d",
        "skew_252d",
        "kurt_252d",
        "amihud_252d",
        "drawdown_252d",
    }.intersection(result.columns)
    assert result.sort_values(["asset_id", "timestamp"]).index.tolist() == list(
        result.index
    )


def test_price_volume_feature_exclusion_is_canonical_key_only() -> None:
    assert PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS == frozenset(
        {"market_id", "asset_id", "timestamp"}
    )


def test_price_volume_factors_handle_grouping_warmup_and_non_finite_edges() -> None:
    panel = make_synthetic_factor_panel(sessions=260)
    asset_a_dates = (
        panel.loc[panel["asset_id"].eq("asset_a"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    asset_b_dates = (
        panel.loc[panel["asset_id"].eq("asset_b"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[0]),
        "adj_close",
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[19]),
        ["amount", "prev_close"],
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[5]),
        ["total_market_cap", "float_market_cap"],
    ] = [0.0, -1.0]
    panel.loc[
        panel["asset_id"].eq("asset_b") & panel["timestamp"].eq(asset_b_dates[7]),
        "float_market_cap",
    ] = pd.NA
    shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)

    factor_columns = (
        "mom_252d",
        "rev_252d",
        "amihud_20d",
        "amplitude_20d",
        "log_float_market_cap",
        "log_total_market_cap",
    )
    result = calculate_price_volume_factors(
        shuffled,
        factor_columns=factor_columns,
    )
    result = result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    asset_a = result[result["asset_id"].eq("asset_a")].reset_index(drop=True)
    asset_b = result[result["asset_id"].eq("asset_b")].reset_index(drop=True)
    expected_asset_b_mom_252d = (
        asset_b.loc[252, "adj_close"] / asset_b.loc[0, "adj_close"] - 1.0
    )

    assert asset_a.loc[:251, "mom_252d"].isna().all()
    assert pd.isna(asset_a.loc[252, "mom_252d"])
    assert pd.isna(asset_a.loc[252, "rev_252d"])
    assert asset_b.loc[:251, "mom_252d"].isna().all()
    assert asset_b.loc[252, "mom_252d"] == pytest.approx(expected_asset_b_mom_252d)
    assert asset_b.loc[252, "rev_252d"] == pytest.approx(-expected_asset_b_mom_252d)
    assert pd.isna(asset_a.loc[19, "amihud_20d"])
    assert pd.isna(asset_a.loc[19, "amplitude_20d"])
    assert pd.isna(asset_a.loc[5, "log_total_market_cap"])
    assert pd.isna(asset_a.loc[5, "log_float_market_cap"])
    assert pd.isna(asset_b.loc[7, "log_float_market_cap"])
    assert not np.isinf(result.loc[:, factor_columns].to_numpy(dtype="float64")).any()


def test_price_volume_factors_reject_nonfinite_source_values() -> None:
    panel = make_synthetic_factor_panel(sessions=25, assets=("asset_a",))
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    denominator_date = ordered.loc[0, "timestamp"]
    rolling_window_end = ordered.loc[19, "timestamp"]

    panel.loc[panel["timestamp"].eq(denominator_date), "adj_close"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "amount"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "volume"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "prev_close"] = np.inf
    panel.loc[panel["timestamp"].eq(denominator_date), "turnover"] = np.inf
    panel.loc[panel["timestamp"].eq(rolling_window_end), "period_return"] = np.inf
    panel.loc[panel["timestamp"].eq(rolling_window_end), "high"] = np.inf

    factor_columns = (
        MOMENTUM_20D_COLUMN,
        "rev_20d",
        "max_return_20d",
        "amihud_20d",
        "amplitude_20d",
        "intraday_range_20d",
        "turnover_chg_20d",
        "amount_chg_20d",
        "amount_shock_20d",
        "volume_chg_20d",
        "volume_shock_20d",
        "ret_amount_corr_20d",
        "ret_volume_corr_20d",
        "ret_turnover_corr_20d",
        "trend_efficiency_20d",
        "signed_amount_flow_20d",
        "ret_amount_imbalance_20d",
    )
    with pytest.raises(ValueError, match="invalid_price_volume_numeric_values"):
        calculate_price_volume_factors(
            panel.sample(frac=1.0, random_state=17),
            factor_columns=factor_columns,
        )


def test_price_volume_flow_factors_are_past_only_and_chunk_stable() -> None:
    panel = make_synthetic_factor_panel(
        sessions=130,
        assets=("asset_a", "asset_b"),
    )
    factor_columns = FLOW_STANDARD_PRICE_VOLUME_FACTORS
    full = (
        calculate_price_volume_factors(panel, factor_columns=factor_columns)
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )
    chunked = _calculate_price_volume_in_session_chunks(
        panel,
        factor_columns=factor_columns,
        split_session_index=75,
        warmup_sessions=60,
    )
    cutoff = panel["timestamp"].sort_values().unique()[79]
    perturbed_future = panel.copy()
    future_mask = perturbed_future["timestamp"].gt(cutoff)
    for column in ("period_return", "amount", "volume", "turnover"):
        perturbed_future.loc[future_mask, column] = (
            pd.to_numeric(perturbed_future.loc[future_mask, column], errors="coerce")
            * 11.0
            + 7.0
        )
    with_future_noise = (
        calculate_price_volume_factors(
            perturbed_future,
            factor_columns=factor_columns,
        )
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )
    primary_key = ("market_id", "asset_id", "timestamp")

    assert full.loc[:18, "amount_shock_20d"].isna().all()
    assert full.loc[:18, "volume_shock_20d"].isna().all()
    assert full.loc[:19, "amount_chg_20d"].isna().all()
    assert full.loc[:59, "amount_chg_60d"].isna().all()
    assert full.loc[:, factor_columns].notna().any().all()
    assert not np.isinf(full.loc[:, factor_columns].to_numpy(dtype="float64")).any()
    assert_frame_matches_reference(
        full.loc[:, (*primary_key, *factor_columns)],
        chunked.loc[:, (*primary_key, *factor_columns)],
    )
    assert_frame_matches_reference(
        full.loc[
            full["timestamp"].le(cutoff),
            (*primary_key, *factor_columns),
        ].reset_index(drop=True),
        with_future_noise.loc[
            with_future_noise["timestamp"].le(cutoff),
            (*primary_key, *factor_columns),
        ].reset_index(drop=True),
    )


def test_price_volume_flow_factor_formulas_match_reference() -> None:
    panel = make_synthetic_factor_panel(
        sessions=90,
        assets=("asset_a", "asset_b"),
    )
    factor_columns = (
        "amount_chg_20d",
        "amount_shock_20d",
        "amount_vol_20d",
        "volume_chg_20d",
        "volume_shock_20d",
        "ret_amount_corr_20d",
        "ret_volume_corr_20d",
        "ret_turnover_corr_20d",
        "trend_efficiency_20d",
        "signed_amount_flow_20d",
        "ret_amount_imbalance_20d",
    )
    result = calculate_price_volume_factors(
        panel.sample(frac=1.0, random_state=29),
        factor_columns=factor_columns,
    )
    reference = reference_price_volume_factors(
        panel.sample(frac=1.0, random_state=29),
        factor_columns=factor_columns,
    )
    primary_key = ("market_id", "asset_id", "timestamp")

    assert result["ret_amount_corr_20d"].notna().any()
    assert result["trend_efficiency_20d"].dropna().between(0.0, 1.0).all()
    assert result["signed_amount_flow_20d"].dropna().between(-1.0, 1.0).all()
    assert result["ret_amount_imbalance_20d"].dropna().between(-1.0, 1.0).all()
    assert_frame_matches_reference(
        result.loc[:, (*primary_key, *factor_columns)].reset_index(drop=True),
        reference.loc[:, (*primary_key, *factor_columns)].reset_index(drop=True),
        rtol=1e-9,
    )


def test_price_volume_standard_formula_matches_reference() -> None:
    panel = make_synthetic_factor_panel(
        sessions=270,
        assets=("asset_b", "asset_a", "asset_c"),
    )
    for asset_id, adjustment in (
        ("asset_a", 0.72),
        ("asset_b", 0.81),
        ("asset_c", 0.68),
    ):
        ordered_dates = (
            panel.loc[panel["asset_id"].eq(asset_id), "timestamp"]
            .sort_values()
            .reset_index(drop=True)
        )
        dip_date = ordered_dates.iloc[35]
        rebound_date = ordered_dates.iloc[36]
        base_price = float(
            panel.loc[
                panel["asset_id"].eq(asset_id)
                & panel["timestamp"].eq(ordered_dates.iloc[34]),
                "adj_close",
            ].iloc[0]
        )
        panel.loc[
            panel["asset_id"].eq(asset_id) & panel["timestamp"].eq(dip_date),
            "adj_close",
        ] = base_price * adjustment
        panel.loc[
            panel["asset_id"].eq(asset_id) & panel["timestamp"].eq(rebound_date),
            "adj_close",
        ] = base_price * 1.03

    asset_a_dates = (
        panel.loc[panel["asset_id"].eq("asset_a"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    asset_b_dates = (
        panel.loc[panel["asset_id"].eq("asset_b"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    asset_c_dates = (
        panel.loc[panel["asset_id"].eq("asset_c"), "timestamp"]
        .sort_values()
        .reset_index(drop=True)
    )
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[0]),
        "adj_close",
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_a") & panel["timestamp"].eq(asset_a_dates[10]),
        "period_return",
    ] = np.nan
    panel.loc[
        panel["asset_id"].eq("asset_b") & panel["timestamp"].eq(asset_b_dates[19]),
        ["amount", "prev_close"],
    ] = 0.0
    panel.loc[
        panel["asset_id"].eq("asset_b") & panel["timestamp"].eq(asset_b_dates[60]),
        "turnover",
    ] = pd.NA
    panel.loc[
        panel["asset_id"].eq("asset_c") & panel["timestamp"].eq(asset_c_dates[7]),
        ["float_market_cap", "total_market_cap"],
    ] = [pd.NA, -1.0]
    panel.loc[
        panel["asset_id"].eq("asset_c") & panel["timestamp"].eq(asset_c_dates[20]),
        "high",
    ] = pd.NA
    shuffled = panel.sample(frac=1.0, random_state=23).reset_index(drop=True)
    factor_columns = price_volume_factor_columns()
    primary_key = ("market_id", "asset_id", "timestamp")
    expected_columns = (*primary_key, *factor_columns)

    result = calculate_price_volume_factors(
        shuffled,
        factor_columns=factor_columns,
    )
    reference = reference_price_volume_factors(
        shuffled,
        factor_columns=factor_columns,
    )

    assert len(result) == len(panel)
    assert tuple(result.loc[:, expected_columns].columns) == expected_columns
    assert result.sort_values(["asset_id", "timestamp"]).index.tolist() == list(
        result.index
    )
    assert result["drawdown_20d"].notna().any()
    assert result["drawdown_60d"].notna().any()
    assert result["skew_20d"].notna().any()
    assert result["kurt_20d"].notna().any()
    assert not np.isinf(result.loc[:, factor_columns].to_numpy(dtype="float64")).any()
    assert_frame_matches_reference(
        result.loc[:, expected_columns].reset_index(drop=True),
        reference.loc[:, expected_columns].reset_index(drop=True),
        rtol=1e-9,
    )


def test_a_share_factor_outputs_are_past_only_and_key_stable_from_registry() -> None:
    panel = make_synthetic_factor_panel(
        sessions=280,
        assets=("asset_b", "asset_a"),
    )
    definitions = tuple(a_share_factor_definitions.factor_catalog().list())
    output_columns = tuple(definition.output_column for definition in definitions)
    required_inputs = tuple(
        dict.fromkeys(
            column
            for definition in definitions
            for column in definition.input_columns
            if column not in {"market_id", "asset_id", "timestamp"}
        )
    )
    baseline = a_share_factor_calculations.calculate_factors_for_panel(
        panel,
        definitions=definitions,
    )
    cutoff = panel["timestamp"].sort_values().unique()[260]

    perturbed_future = panel.copy()
    future_mask = perturbed_future["timestamp"].gt(cutoff)
    for column in required_inputs:
        perturbed_future.loc[future_mask, column] = (
            pd.to_numeric(perturbed_future.loc[future_mask, column], errors="coerce")
            * 7.0
            + 3.0
        )
    with_future_noise = a_share_factor_calculations.calculate_factors_for_panel(
        perturbed_future,
        definitions=definitions,
    )
    shuffled_with_unrelated = panel.sample(frac=1.0, random_state=19).assign(
        unrelated_context="ignored"
    )
    reshuffled = a_share_factor_calculations.calculate_factors_for_panel(
        shuffled_with_unrelated,
        definitions=definitions,
    )

    compare_columns = ["market_id", "asset_id", "timestamp", *output_columns]
    baseline_past = _sorted_factor_output(
        baseline.loc[baseline["timestamp"].le(cutoff), compare_columns]
    )
    future_noise_past = _sorted_factor_output(
        with_future_noise.loc[
            with_future_noise["timestamp"].le(cutoff),
            compare_columns,
        ]
    )
    pd.testing.assert_frame_equal(baseline_past, future_noise_past)
    pd.testing.assert_frame_equal(
        _sorted_factor_output(baseline.loc[:, compare_columns]),
        _sorted_factor_output(reshuffled.loc[:, compare_columns]),
    )
    assert _key_set(baseline) == _key_set(panel)


def test_price_volume_rolling_moments_are_chunk_start_invariant() -> None:
    panel = make_synthetic_factor_panel(
        sessions=260,
        assets=("asset_a", "asset_b"),
    )
    factor_columns = ("skew_20d", "kurt_20d")
    full = (
        calculate_price_volume_factors(panel, factor_columns=factor_columns)
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )

    chunked = _calculate_price_volume_in_session_chunks(
        panel,
        factor_columns=factor_columns,
        split_session_index=130,
        warmup_sessions=20,
    )

    assert_frame_matches_reference(
        full.loc[:, ("market_id", "asset_id", "timestamp", *factor_columns)],
        chunked.loc[:, ("market_id", "asset_id", "timestamp", *factor_columns)],
    )
    assert (
        (full.loc[:, factor_columns].isna() == chunked.loc[:, factor_columns].isna())
        .all()
        .all()
    )


def test_price_volume_compute_fails_closed_when_required_input_missing() -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5)).drop(
        columns=["amount"]
    )

    with pytest.raises(ValueError, match="missing_price_volume_input_columns: amount"):
        calculate_price_volume_factors(panel, factor_columns=("amihud_20d",))


@pytest.mark.parametrize(
    ("factor_columns", "error_match"),
    [
        (("mom_2m",), "unsupported_price_volume_factors: mom_2m"),
        (
            ("log_float_market_cap_20d", "rev_1d_2d"),
            "unsupported_price_volume_factors: log_float_market_cap_20d, rev_1d_2d",
        ),
    ],
)
def test_price_volume_compute_fails_closed_for_unsupported_factor_names(
    factor_columns: tuple[str, ...],
    error_match: str,
) -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5))

    assert unsupported_price_volume_factors(factor_columns) == factor_columns
    with pytest.raises(ValueError, match=error_match):
        calculate_price_volume_factors(panel, factor_columns=factor_columns)


def test_price_volume_mom_1d_is_not_a_supported_factor() -> None:
    panel = _factor_panel_for_dates(pd.bdate_range("2014-01-02", periods=5))

    assert unsupported_price_volume_factors(("mom_1d",)) == ("mom_1d",)
    with pytest.raises(
        ValueError,
        match="unsupported_price_volume_factors: mom_1d",
    ):
        calculate_price_volume_factors(panel, factor_columns=("mom_1d",))


def test_price_volume_window_column_uses_frequency_suffix_contract() -> None:
    assert price_volume_window_column("mom", 7, DataFrequency.DAILY) == "mom_7d"
    assert price_volume_window_column("mom", 7, DataFrequency.MINUTE) == "mom_7m"
    assert (
        price_volume_window_column("ret_volume_corr", 7, DataFrequency.TICK)
        == "ret_volume_corr_7tick"
    )


def _calculate_price_volume_in_session_chunks(
    panel: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    split_session_index: int,
    warmup_sessions: int,
) -> pd.DataFrame:
    ordered = panel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    sessions = tuple(
        sorted(pd.to_datetime(ordered["timestamp"]).dt.normalize().unique())
    )
    parts = []
    for start_index, end_index in (
        (0, split_session_index),
        (split_session_index, len(sessions)),
    ):
        core_sessions = set(sessions[start_index:end_index])
        read_start_index = max(0, start_index - warmup_sessions)
        read_sessions = set(sessions[read_start_index:end_index])
        read_mask = (
            pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(read_sessions)
        )
        core_mask = (
            pd.to_datetime(ordered["timestamp"]).dt.normalize().isin(core_sessions)
        )
        calculated = calculate_price_volume_factors(
            ordered.loc[read_mask],
            factor_columns=factor_columns,
        )
        parts.append(calculated.loc[core_mask.loc[read_mask].to_numpy()])
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["asset_id", "timestamp"])
        .reset_index(drop=True)
    )


def _factor_panel_for_dates(dates) -> pd.DataFrame:
    rows = []
    for offset, date in enumerate(pd.to_datetime(dates), start=1):
        rows.append(
            {
                "market_id": "a_share",
                "asset_id": "asset_a",
                "timestamp": pd.Timestamp(date),
                "adj_close": float(100 + offset),
                "period_return": 0.001 * offset,
                "amount": 10000.0 + offset,
                "volume": 1000.0 + offset,
                "turnover": 0.01 + offset / 100000.0,
                "high": float(101 + offset),
                "low": float(99 + offset),
                "prev_close": float(99 + offset),
                "total_market_cap": float(1_000_000 + offset),
                "float_market_cap": float(500_000 + offset),
            }
        )
    return pd.DataFrame(rows)


def _sorted_factor_output(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )


def _key_set(frame: pd.DataFrame) -> set[tuple[object, object, object]]:
    keys = frame.loc[:, ["market_id", "asset_id", "timestamp"]]
    return set(keys.itertuples(index=False, name=None))
