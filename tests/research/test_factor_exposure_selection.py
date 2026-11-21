"""Historical exposures and representations for factor selection."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.compute.dataframe.history import calendar_month_return
from quant_research_platform.research.datasets.assembler import _policy_inputs
from quant_research_platform.research.factors import distributions
from quant_research_platform.research.factors.search import _formula_identity
from quant_research_platform.research.factors.selection import cluster_exposures


def test_formula_identity_preserves_direction_and_undefined_division():
    assert _formula_identity("rank(a*b) + -( -c)") == _formula_identity("c+rank(b*a)")
    assert _formula_identity("1*(a+0)") == _formula_identity("a")
    assert _formula_identity("0*a") != _formula_identity("0")
    assert _formula_identity("a/b") != _formula_identity("b/a")
    assert _formula_identity("a") != _formula_identity("-a")
    assert _formula_identity("a/a") != _formula_identity("1")


def test_policy_projects_each_source_once_in_configured_order(tmp_path):
    table = tmp_path / "policy.csv"
    table.write_text(
        "analysis_grain,factor_column,research_role\nstock,x,signal\nstock,y,signal\nstock,z,state\n",
        encoding="utf-8",
    )
    selection = dict(
        policy=table,
        grains=["stock"],
        research_roles=["signal"],
        column_prefixes=["before_", ""],
    )
    roles = dict(x="factor", before_x="feature", y="factor", z="factor")
    assert _policy_inputs(selection, roles) == (["y"], ["before_x"])


def test_sign_flips_cluster_but_a_market_state_is_constant_within_each_date():
    x = np.arange(16, dtype=float)
    days = [
        pd.DataFrame(
            dict(
                asset_id=[f"a{i:02}" for i in range(16)],
                timestamp=day,
                original=x,
                inverse=-x,
                narrow=np.where(x % 2 == 0, x, np.nan),
                distinct=x % 3,
                common=float(i),
                missing=np.nan,
            )
        )
        for i, day in enumerate(pd.date_range("2020-01-06", periods=5, freq="7D"))
    ]
    cols = ["original", "inverse", "narrow", "distinct", "common", "missing"]
    result = cluster_exposures(
        days, cols, axis="cross_sectional", threshold=0.9, preferred=["original"]
    )
    assert result["retained"] == ["original", "distinct"]
    assert result["clusters"] == {"original": ["inverse", "narrow"]}
    assert result["constant"] == ["common"] and result["all_missing"] == ["missing"]
    for day in days:
        day["asset_id"] = "etf"
        day["timestamp"] += pd.to_timedelta(x, unit="s")
    time_series = cluster_exposures(
        days,
        cols,
        axis="single_asset_time_series",
        threshold=0.9,
        preferred=["original"],
    )
    assert set(time_series["retained"]) == {"original", "distinct", "common"}


def test_intraday_turnover_pairs_the_same_clock_and_asset_across_sessions():
    clock = np.repeat(np.arange(30), 2)
    exposure = np.tile([0.0, 1.0], 30)
    days = [
        pd.DataFrame(
            {
                "timestamp": day + pd.to_timedelta(clock, unit="s"),
                "asset_id": np.tile(["a", "b"], 30),
                "stable": exposure,
                "nearby": np.where(clock == i, 1 - exposure, exposure),
                "market_clock": clock,
            }
        )
        for i, day in enumerate(pd.date_range("2020-01-06", periods=5, freq="7D"))
    ]
    days[1] = days[1].iloc[:-1]
    result = cluster_exposures(
        days,
        ["nearby", "stable", "market_clock"],
        axis="cross_sectional",
        threshold=0.8,
    )
    assert result["retained"] == ["stable"]
    assert result["clusters"] == {"stable": ["nearby"]}
    assert result["constant"] == ["market_clock"]


def test_correlation_requires_pairs_in_three_different_weeks():
    x = np.arange(60, dtype=float)
    days = [
        pd.DataFrame(
            {
                "asset_id": x,
                "timestamp": day,
                "one": np.where((i == 0) | (x == 0), x, np.nan),
                "two": np.where((i == 0) | (x == 1), x, np.nan),
            }
        )
        for i, day in enumerate(pd.date_range("2020-01-06", periods=4, freq="7D"))
    ]
    result = cluster_exposures(
        days, ["one", "two"], axis="cross_sectional", threshold=0.9
    )
    assert result["retained"] == ["one", "two"]


def test_calendar_seasonality_uses_only_prior_same_month_returns():
    dates = pd.date_range("2015-12-31", "2021-03-31", freq="ME")
    rets = np.where(dates.month == 2, 0.01 * (dates.year - 2015), 0.0)
    prices = pd.Series(100 * np.cumprod(1 + rets))
    signal = calendar_month_return(prices, dates)
    assert np.isnan(signal[dates.get_loc("2018-02-28")])
    assert signal[dates.get_loc("2021-02-28")] == pytest.approx(0.03)
    prices.loc[dates >= "2021-02-01"] *= 2
    assert calendar_month_return(prices, dates)[
        dates.get_loc("2021-02-28")
    ] == pytest.approx(0.03)


def test_dividends_use_pre_ex_price_and_known_announcements_only(tmp_path):
    dates = pd.bdate_range("2016-01-04", "2020-06-12")
    frame = pd.DataFrame(
        dict(
            market_id="a_share",
            asset_id="a",
            timestamp=dates + pd.Timedelta(hours=7),
            signal_session=dates,
            execution_session=dates.shift(1),
            listing_date=dates[0],
            close=10.0,
            volatility_60d=0.01,
        )
    )
    events = pd.DataFrame(
        dict(
            kind="CA",
            source_asset_id="a",
            amount=[1.0, 1.0, 1.0, 2.0],
            timestamp=pd.to_datetime(
                ["2017-06-12", "2018-06-12", "2019-06-12", "2020-06-01"]
            ),
            announcement_date=pd.to_datetime(
                ["2017-05-01", "2018-05-01", "2019-05-01", "2020-06-09"]
            ),
        )
    )

    prices = frame[
        ["market_id", "asset_id", "signal_session", "listing_date", "close"]
    ].rename(columns={"signal_session": "timestamp"})
    # The old payout needs no pre-ex price to prove positive cash was paid.
    prices.loc[prices.timestamp.eq("2018-06-11"), "close"] = np.nan
    frame = frame.drop(columns="close")

    def calc(table):
        return distributions.calculate(
            frame,
            {"corporate_actions": table, "market_panel_1d": prices},
            columns=tuple(distributions.RECIPES),
        ).set_index("timestamp")

    first = calc(events.iloc[:3])
    full = calc(events)
    pd.testing.assert_frame_equal(first.loc[:"2020-06-09"], full.loc[:"2020-06-09"])
    row = full.loc["2020-06-10"].iloc[0]
    assert row.cash_dividend_yield_1y == pytest.approx(0.30)
    assert row.persistent_dividend_low_risk == pytest.approx(
        0.30 / (0.01 * np.sqrt(252))
    )

    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    inputs = {}
    for name, data, keys in (
        ("decision_panel", frame, ("market_id", "asset_id", "timestamp")),
        ("market_panel_1d", prices, ("market_id", "asset_id", "timestamp")),
        ("corporate_actions", events, ("source_asset_id", "timestamp", "kind")),
    ):
        path = tmp_path / name
        roles = dict.fromkeys(data, "context")
        roles.update(dict.fromkeys(keys, "key"), timestamp="timestamp")
        write_dataset(
            path,
            data,
            spec=DataSpec(
                data_id=name,
                frequency="1d",
                grain="asset_timestamp",
                primary_key=keys,
                column_roles=roles,
                partition_columns=(),
            ),
        )
        inputs[name] = str(path)
    artifact = prepare(
        dict(
            market_id="a_share",
            data_id="cash_distribution_factors",
            factor_columns=list(distributions.RECIPES),
            inputs=inputs,
            time_window={"start": "2020-06-10", "end": "2020-06-10"},
            output_root=str(tmp_path / "factors"),
        )
    )
    published = read_dataset(artifact).iloc[0]
    assert published.cash_dividend_yield_1y == pytest.approx(0.30)
    assert published.persistent_dividend_low_risk == pytest.approx(
        row.persistent_dividend_low_risk
    )


@pytest.mark.parametrize("algorithm", ["lstm_regressor", "transformer_regressor"])
def test_snapshot_branch_reads_last_real_row_not_padding_or_older_values(algorithm):
    import torch

    from quant_research_platform.research.models.neural import _network

    torch.manual_seed(7)
    net = _network(
        algorithm,
        2,
        dict(
            hidden_size=4,
            num_layers=1,
            sequence_length=3,
            n_heads=2,
            norm_first=True,
            dropout=0.0,
            snapshot_columns=["slow"],
            initialize_at_target_mean=True,
        ),
        columns=("fast", "slow"),
    ).eval()
    sample = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
    changed = sample.clone()
    changed[0, 0, 1] = 10000.0
    changed[0, 2, :] = -10000.0
    with torch.no_grad():
        assert net(sample, torch.tensor([2])).item() == 0.0
        assert net.output.weight.count_nonzero().item() == 0
        net.output.weight.fill_(1.0)
        assert torch.equal(
            net(sample, torch.tensor([2])), net(changed, torch.tensor([2]))
        )
        net.output.weight.zero_()
        net.output.weight[0, -1] = 1.0
        net.output.bias.zero_()
        assert net(sample, torch.tensor([2])).item() == 4.0
