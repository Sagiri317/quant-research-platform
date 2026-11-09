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


def test_native_preparation_keeps_public_clock_and_selected_factor(tmp_path):
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.public_research.interim import interim_catalog
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    decisions, observations, _ = inputs()
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
            "time_window": {"start": "2024-01-10", "end": "2024-01-10"},
            "output_root": str(tmp_path / "factors"),
        }
    )
    assert read_dataset(result).financing_buy_intensity.tolist() == pytest.approx(
        [0.1, 0.1]
    )
