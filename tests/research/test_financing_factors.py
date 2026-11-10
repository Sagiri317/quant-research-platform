"""Hand-computed disclosure joins use exact sessions and actual release versions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.financing import RECIPES, calculate


def inputs():
    sessions = pd.bdate_range("2024-01-02", periods=8).as_unit("ns")
    decisions = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["000001"] * 2,
            "timestamp": [
                sessions[6] + pd.Timedelta(hours=1),
                sessions[6] + pd.Timedelta(hours=2),
            ],
            "signal_session": sessions[5],
            "listing_date": sessions[0],
            "eligible": True,
            "float_market_cap": 1000.0,
            "amount": 200.0,
            "peak_drawdown_20d": 0.2,
            "amihud_20d": 2.0,
            "amihud_60d": 1.0,
            "a_float_shares": 1000.0,
            "volume": 200.0,
            "avg_volume_20d": 100.0,
            "avg_amount_20d": 200.0,
            "reversal_5d": -0.1,
            "price_up_acceleration_20d": 0.02,
        }
    )
    observations = pd.DataFrame(
        {
            "asset_id": "000001",
            "timestamp": sessions[:6],
            "available_at": sessions[1:7] + pd.Timedelta(minutes=30),
            "financing_balance": [50.0, 60, 70, 80, 90, 100],
            "financing_buy_amount": 20.0,
            "financing_repayment_amount": np.nan,
            "lending_balance": 10.0,
            "lending_quantity": [100.0, 90, 80, 70, 60, 50],
            "lending_repayment_quantity": 4.0,
        }
    )
    return decisions, observations, sessions


def test_financing_level_activity_change_and_liquidity_pressure_by_hand():
    decisions, observations, sessions = inputs()
    result = calculate(decisions, observations, sessions=sessions)
    for factor, expected in {
        "financing_balance_to_float_cap": 0.1,
        "financing_buy_intensity": 0.1,
        "lending_balance_to_float_cap": 0.01,
        "financing_balance_change_1d": 100 / 90 - 1,
        "financing_balance_change_5d": 1.0,
        "leveraged_liquidity_pressure": 0.02,
        "short_interest_to_float_shares": 0.05,
        "lending_days_to_cover_20d": 0.5,
        "lending_repayment_intensity": 0.02,
        "lending_net_cover_fraction_1d": 1 / 6,
        "lending_net_cover_fraction_5d": 0.5,
        "short_squeeze_pressure": 0.025,
        "long_deleveraging_pressure": 0.0,
        "financing_buy_after_drawdown": 0.02,
        "price_acceleration_x_leverage_expansion": 0.02,
    }.items():
        assert result[factor].tolist() == pytest.approx([expected] * 2)
    assert result.financing_repayment_intensity.isna().all()


def test_release_revision_and_missing_calendar_only_affect_dependents():
    decisions, observations, sessions = inputs()
    delayed = observations.iloc[[-1]].assign(
        financing_balance=120.0,
        financing_buy_amount=40.0,
        available_at=sessions[6] + pd.Timedelta(minutes=90),
    )
    reports = pd.concat([observations, delayed], ignore_index=True)
    result = calculate(decisions, reports, sessions=sessions)
    assert result.financing_balance_to_float_cap.tolist() == pytest.approx([0.1, 0.12])
    assert result.financing_buy_intensity.tolist() == pytest.approx([0.1, 0.2])
    without_calendar = calculate(decisions, reports)
    assert without_calendar.financing_balance_change_5d.isna().all()
    pd.testing.assert_series_equal(
        result.financing_buy_intensity, without_calendar.financing_buy_intensity
    )
    # A later revision of an earlier session changes that lag at its release only.
    revision = observations.iloc[[0]].assign(
        financing_balance=25.0,
        available_at=sessions[6] + pd.Timedelta(minutes=90),
    )
    result = calculate(decisions, pd.concat([reports, revision]), sessions=sessions)
    assert result.financing_balance_change_5d.tolist() == pytest.approx([1, 3.8])
    # No source report for the expected session: no silent stale balance carry.
    result = calculate(decisions, observations.iloc[:-1], sessions=sessions)
    assert result[list(RECIPES)].isna().all().all()


def test_lifecycle_and_unrelated_fields_do_not_gate_buy_activity():
    decisions, observations, sessions = inputs()
    decisions.loc[1, "listing_date"] = sessions[4]
    observations["financing_balance"] = np.nan
    result = calculate(decisions, observations, sessions=sessions)
    assert result.financing_buy_intensity.notna().all()
    assert result.financing_balance_change_5d.isna().all()
    selected = calculate(
        decisions.drop(
            columns=[
                "float_market_cap",
                "amihud_20d",
                "amihud_60d",
                "peak_drawdown_20d",
            ]
        ),
        observations.drop(
            columns=[
                "financing_balance",
                "lending_balance",
                "financing_repayment_amount",
            ]
        ),
        factor_columns=("financing_buy_intensity",),
    )
    assert selected.financing_buy_intensity.tolist() == pytest.approx([0.1, 0.1])


@pytest.mark.parametrize("reverse", [False, True])
def test_same_clock_report_withdrawal_only_invalidates_its_field(reverse):
    decisions, observations, sessions = inputs()
    withdrawn = observations.iloc[[-1]].assign(financing_buy_amount=np.nan)
    reports = pd.concat([observations, withdrawn], ignore_index=True)
    if reverse:
        reports = reports.iloc[::-1]
    result = calculate(decisions, reports, sessions=sessions)
    assert result.financing_buy_intensity.isna().all()
    assert result.financing_balance_to_float_cap.tolist() == pytest.approx([0.1, 0.1])


def test_invalid_business_numeric_cells_only_limit_their_financing_dependents():
    decisions, observations, sessions = inputs()
    observations["financing_balance"] = "invalid"
    observations["lending_balance"] = float("inf")
    decisions["amihud_20d"] = float("inf")
    result = calculate(decisions, observations, sessions=sessions)
    assert result.financing_buy_intensity.tolist() == pytest.approx([0.1, 0.1])
    assert result.financing_balance_change_1d.isna().all()
    assert result.lending_balance_to_float_cap.isna().all()
    assert result.leveraged_liquidity_pressure.isna().all()
    assert result.attrs["input_invalid_numeric_cells"]["financing_balance"] == len(
        observations
    )
    assert result.attrs["input_invalid_numeric_cells"]["amihud_20d"] == len(decisions)


def test_conservative_lag_uses_the_report_sessions_denominator_and_current_stress():
    decisions, observations, sessions = inputs()
    preceding = decisions.iloc[[0]].assign(
        timestamp=sessions[5] + pd.Timedelta(hours=1),
        signal_session=sessions[4],
        amount=100.0,
        float_market_cap=500.0,
    )
    full = pd.concat([preceding, decisions], ignore_index=True)
    result = calculate(
        full,
        observations,
        sessions=sessions,
        observation_lag_sessions=1,
    ).iloc[-2:]
    assert result.financing_buy_intensity.tolist() == pytest.approx([0.2, 0.2])
    assert result.financing_balance_to_float_cap.tolist() == pytest.approx([0.18, 0.18])
    assert result.financing_balance_change_1d.tolist() == pytest.approx(
        [90 / 80 - 1] * 2
    )
    assert result.leveraged_liquidity_pressure.tolist() == pytest.approx([0.036, 0.036])
    # Missing old denominators leave the independently known balance change valid.
    missing_history = calculate(
        decisions,
        observations,
        sessions=sessions,
        observation_lag_sessions=1,
    )
    assert missing_history.financing_buy_intensity.isna().all()
    assert missing_history.financing_balance_change_1d.notna().all()


@pytest.mark.parametrize("decision_unit,source_unit", [("us", "ns"), ("ns", "us")])
def test_mixed_pandas_datetime_resolutions_preserve_the_disclosure_join(
    decision_unit, source_unit
):
    decisions, observations, sessions = inputs()
    for name in ("timestamp", "signal_session", "listing_date"):
        decisions[name] = decisions[name].dt.as_unit(decision_unit)
    for name in ("timestamp", "available_at"):
        observations[name] = observations[name].dt.as_unit(source_unit)
    result = calculate(decisions, observations, sessions=sessions.as_unit("us"))
    assert result.financing_balance_change_5d.tolist() == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize(
    "decision_dtype,source_dtype",
    [
        ("object", "str"),
        ("str", "object"),
        ("string[pyarrow]", "string[python]"),
    ],
)
def test_native_string_key_carriers_have_the_same_financing_identity(
    decision_dtype, source_dtype
):
    decisions, observations, sessions = inputs()
    decisions["asset_id"] = decisions.asset_id.astype(decision_dtype)
    observations["asset_id"] = observations.asset_id.astype(source_dtype)
    result = calculate(decisions, observations, sessions=sessions)
    assert result.financing_buy_intensity.tolist() == pytest.approx([0.1, 0.1])


def test_native_preparation_keeps_public_clock_and_selected_factor(tmp_path):
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.public_research.interim import interim_catalog
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    decisions, observations, sessions = inputs()
    # PANEL's strict string owner and public raw normalization use different carriers.
    decisions["asset_id"] = decisions.asset_id.astype("object")
    decisions["execution_session"] = pd.Timestamp("2024-01-10")
    decisions["return_available_at"] = decisions.signal_session + pd.Timedelta(
        hours=7, minutes=30
    )
    path = tmp_path / "decisions"
    roles = dict.fromkeys(decisions, "context")
    roles.update(
        market_id="key", asset_id="key", timestamp="timestamp", eligible="mask"
    )
    write_dataset(
        path,
        decisions,
        spec=ProcessedDataSpec(
            data_id="decision_panel",
            market_id="a_share",
            kind="panel",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles=roles,
        ),
    )
    spec = interim_catalog().get("public_margin_balances")
    for column in spec.required_columns:
        if column not in observations:
            if column in spec.column_units:
                observations[column] = np.nan
            elif column in {"published_at", "retrieved_at"}:
                observations[column] = pd.Series(
                    pd.NaT, index=observations.index, dtype="datetime64[ns, UTC]"
                )
            elif column == "publication_date":
                observations[column] = pd.Series(
                    pd.NaT, index=observations.index, dtype="datetime64[ns]"
                )
            else:
                observations[column] = pd.Series(
                    "synthetic", index=observations.index, dtype="str"
                )
    raw = tmp_path / "observations"
    write_dataset(raw, observations.loc[:, list(spec.required_columns)], spec=spec)
    result = prepare(
        {
            "market_id": "a_share",
            "data_id": "financing_factors",
            "inputs": {"decision_panel": str(path), "public_margin_balances": str(raw)},
            "factor_columns": ["financing_buy_intensity"],
            "parameters": {"observation_lag_sessions": 0},
            "time_window": {"start": "2024-01-10", "end": "2024-01-10"},
            "output_root": str(tmp_path / "factors"),
        }
    )
    assert read_dataset(result).financing_buy_intensity.tolist() == pytest.approx(
        [0.1, 0.1]
    )
    from quant_research_platform.sources.csmar.interim import interim_catalog as csmar

    preceding = decisions.iloc[[0]].assign(
        timestamp=sessions[5] + pd.Timedelta(hours=1),
        signal_session=sessions[4],
        execution_session=sessions[5],
        return_available_at=sessions[4] + pd.Timedelta(hours=7, minutes=30),
        amount=100.0,
        float_market_cap=500.0,
    )
    historical = tmp_path / "historical_decisions"
    write_dataset(
        historical,
        pd.concat([preceding, decisions], ignore_index=True),
        spec=ProcessedDataSpec(
            data_id="decision_panel",
            market_id="a_share",
            kind="panel",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            column_roles=roles,
        ),
    )
    calendar_path = tmp_path / "calendar"
    calendar = pd.DataFrame(
        {
            "timestamp": sessions,
            "calendar_date": sessions,
            "source_market_type": 4,
            "is_trading_day": True,
            "source_day_of_week": sessions.dayofweek + 1,
            "source_state": "open",
        }
    )
    write_dataset(calendar_path, calendar, spec=csmar().get("trading_calendar"))
    lagged = prepare(
        {
            "market_id": "a_share",
            "data_id": "financing_factors",
            "inputs": {
                "decision_panel": str(historical),
                "public_margin_balances": str(raw),
                "trading_calendar": str(calendar_path),
            },
            "factor_columns": ["financing_buy_intensity"],
            "time_window": {"start": "2024-01-10", "end": "2024-01-10"},
            "output_root": str(tmp_path / "lagged_factors"),
        }
    )
    assert read_dataset(lagged).financing_buy_intensity.tolist() == pytest.approx(
        [0.2, 0.2]
    )
