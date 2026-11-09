"""Public leverage/sentiment states reuse causal macro history and true scopes."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.factors.macro_liquidity.calculations import (
    macro_states,
    parameters,
)
from quant_research_platform.research.factors.macro_liquidity.public_observations import (  # noqa: E501
    observations_for_series,
)


def test_finra_month_reference_and_publication_revisions_by_hand():
    dates = pd.date_range("2023-01-01", periods=13, freq="MS")
    frame = pd.DataFrame(
        {
            "asset_id": "US_MARGIN_ACCOUNTS",
            "timestamp": dates,
            "available_at": dates + pd.offsets.MonthEnd(0) + pd.Timedelta(days=10),
            "margin_debit": np.arange(13) * 10.0 + 100,
        }
    )
    revision = frame.iloc[[0]].assign(
        margin_debit=50, available_at=pd.Timestamp("2024-02-20")
    )
    history = observations_for_series(
        pd.concat([frame, revision]), ["us_finra_margin_debit"]
    )
    assert history.observation_date.dt.is_month_end.all()
    result = macro_states(
        history,
        ["2024-02-15", "2024-02-21"],
        columns=[
            "ml_us_finra_margin_debit_level",
            "ml_us_finra_margin_debit_growth",
            "ml_us_finra_margin_debit_yoy_growth",
        ],
        config=parameters({}),
    )
    assert result.ml_us_finra_margin_debit_level.tolist() == [220, 220]
    assert result.ml_us_finra_margin_debit_growth.tolist() == pytest.approx(
        [220 / 210 - 1] * 2
    )
    assert result.ml_us_finra_margin_debit_yoy_growth.tolist() == pytest.approx(
        [1.2, 3.4]
    )


def test_cftc_contracts_report_types_fields_and_clocks_remain_separate():
    frame = pd.DataFrame(
        {
            "asset_id": [
                "13874A:futures_only",
                "13874A:futures_only",
                "13874A:futures_and_options",
            ],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-09"]),
            "available_at": pd.to_datetime(
                ["2024-01-05T20:30Z", "2024-01-12T20:30Z", "2024-01-12T20:30Z"]
            ),
            "leveraged_long": [100.0, 120, 9999],
            "leveraged_short": [150.0, 140, 0],
            "open_interest": [500.0, 400, 1],
        }
    )
    names = [
        "us_cftc_sp500_leveraged_net_share",
        "us_cftc_sp500_leveraged_net_contracts",
    ]
    history = observations_for_series(frame, names)
    result = macro_states(
        history,
        ["2024-01-12T20:29Z", "2024-01-12T20:30Z"],
        columns=[
            "ml_us_cftc_sp500_leveraged_net_share_level",
            "ml_us_cftc_sp500_leveraged_net_contracts_change",
        ],
        config=parameters({}),
    )
    assert result.ml_us_cftc_sp500_leveraged_net_share_level.tolist() == pytest.approx(
        [-0.1, -0.05]
    )
    assert np.isnan(result.ml_us_cftc_sp500_leveraged_net_contracts_change.iloc[0])
    assert result.ml_us_cftc_sp500_leveraged_net_contracts_change.iloc[1] == 30
    frame.loc[1, "open_interest"] = 0
    changed = observations_for_series(frame, names).sort_values(
        ["series_id", "observation_date"]
    )
    assert changed.loc[changed.series_id.eq(names[0]), "value"].isna().sum() == 1
    assert changed.loc[changed.series_id.eq(names[1]), "value"].notna().all()


@pytest.mark.parametrize("bad", ["inf", "invalid", float("inf")])
def test_invalid_open_interest_only_limits_the_ratio_not_net_contracts(bad):
    frame = pd.DataFrame(
        {
            "asset_id": ["13874A:futures_only"],
            "timestamp": pd.Timestamp("2024-01-02"),
            "available_at": pd.Timestamp("2024-01-05", tz="UTC"),
            "leveraged_long": "30",
            "leveraged_short": "10",
            "open_interest": [bad],
        }
    )
    names = [
        "us_cftc_sp500_leveraged_net_share",
        "us_cftc_sp500_leveraged_net_contracts",
    ]
    result = observations_for_series(frame, names).set_index("series_id")
    assert pd.isna(result.loc[names[0], "value"])
    assert result.loc[names[1], "value"] == 20
    assert result.attrs["input_invalid_numeric_cells"]["open_interest"] == 1


def test_unknown_publication_and_conflicting_snapshots_are_not_invented():
    frame = pd.DataFrame(
        {
            "asset_id": "EQUITY PUT/CALL RATIO",
            "timestamp": pd.to_datetime(["2024-01-02"] * 3),
            "available_at": pd.to_datetime(["2024-01-03", "2024-01-03", None]),
            "put_call_ratio": [0.5, 0.8, 1.0],
        }
    )
    result = observations_for_series(frame, ["us_cboe_equity_put_call"])
    assert len(result) == 1 and result.value.isna().all()
    assert result.timestamp.eq(pd.Timestamp("2024-01-03", tz="UTC")).all()


@pytest.mark.parametrize(
    "values,known",
    [
        ([0.2, np.nan], False),
        ([np.nan, 0.2], False),
        ([0.2, 0.2], True),
        ([np.nan, np.nan], False),
    ],
)
def test_same_clock_withdrawals_cannot_be_coalesced_to_an_older_value(values, known):
    frame = pd.DataFrame(
        {
            "asset_id": "EQUITY PUT/CALL RATIO",
            "timestamp": pd.Timestamp("2024-01-02"),
            "available_at": pd.Timestamp("2024-01-03", tz="UTC"),
            "put_call_ratio": values,
        }
    )
    result = observations_for_series(frame, ["us_cboe_equity_put_call"])
    assert len(result) == 1
    assert bool(result.value.notna().iloc[0]) == known
    if known:
        assert result.value.iloc[0] == 0.2


def test_native_public_macro_inputs_need_no_unrelated_fred_source(tmp_path):
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.sources.public_research.interim import interim_catalog
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from quant_research_platform.storage.datasets.reading import open_dataset_session

    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "000001",
            "eligible": True,
            "timestamp": pd.to_datetime(["2024-01-12T01:20", "2024-01-15T01:20"]),
            "signal_session": pd.to_datetime(["2024-01-11", "2024-01-12"]),
            "execution_session": pd.to_datetime(["2024-01-12", "2024-01-15"]),
            "return_available_at": pd.to_datetime(
                ["2024-01-11T07:00", "2024-01-12T07:00"]
            ),
        }
    )
    panel["order_submit_at"] = panel.timestamp + pd.Timedelta(seconds=1)
    roles = dict.fromkeys(panel, "context")
    roles.update(
        market_id="key", asset_id="key", timestamp="timestamp", eligible="mask"
    )
    decisions = tmp_path / "decisions"
    write_dataset(
        decisions,
        panel,
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
    source = pd.DataFrame(
        {
            "asset_id": "13874A:futures_only",
            "source": "synthetic",
            "source_sha256": "hash",
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-09"]),
            "available_at": pd.to_datetime(["2024-01-05T20:30Z", "2024-01-12T20:30Z"]),
            "leveraged_long": [100.0, 120],
            "leveraged_short": [150.0, 140],
            "open_interest": ["500", "inf"],
            "report_type": "futures_only",
            "availability_basis": "assumed_release",
        }
    )
    spec = interim_catalog().get("public_cftc_positions")
    for column in spec.required_columns:
        if column not in source:
            if (
                column in spec.column_units
                and spec.column_units[column] != "identifier"
            ):
                source[column] = np.nan
            elif column in {"published_at", "retrieved_at"}:
                source[column] = pd.Series(
                    pd.NaT, index=source.index, dtype="datetime64[ns, UTC]"
                )
            elif column == "publication_date":
                source[column] = pd.Series(
                    pd.NaT, index=source.index, dtype="datetime64[ns]"
                )
            else:
                source[column] = "synthetic"
    public = tmp_path / "public"
    write_dataset(public, source.loc[:, list(spec.required_columns)], spec=spec)
    level = "ml_us_cftc_sp500_leveraged_net_share_level"
    contracts = "ml_us_cftc_sp500_leveraged_net_contracts_level"
    missing = "ml_us_cboe_equity_put_call_level"
    result = prepare(
        {
            "market_id": "a_share",
            "data_id": "macro_liquidity_factors",
            "inputs": {
                "decision_panel": str(decisions),
                "public_cftc_positions": str(public),
            },
            "factor_columns": [level, contracts, missing],
            "time_window": {"start": "2024-01-12", "end": "2024-01-15"},
            "output_root": str(tmp_path / "factors"),
        }
    )
    factors = read_dataset(result)
    assert factors[level].iloc[0] == pytest.approx(-0.1)
    assert pd.isna(factors[level].iloc[1])
    assert factors[contracts].tolist() == [-50, -20]
    assert factors[missing].isna().all()
    metadata = open_dataset_session(result).manifest["producer_diagnostics"]
    assert metadata["missing_public_sources"] == ["public_option_put_call"]
    assert not metadata["missing_macro_observations"]
    assert (
        metadata["public_source_clocks"]["public_cftc_positions"][
            "estimated_availability_rows"
        ]
        == 2
    )
    assert (
        metadata["public_source_clocks"]["public_cftc_positions"][
            "input_invalid_numeric_cells"
        ]["open_interest"]
        == 1
    )
