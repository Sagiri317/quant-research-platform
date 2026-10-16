"""Native explicit preparation preserves source history and scientific boundaries."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_sources import (
    _prepare,
    _synthetic_financial_frames,
    _synthetic_interim_frames,
    _write_inputs,
)


@pytest.mark.parametrize(
    "barrier", ("known", "unknown", "raw_missing", "adjusted_missing")
)
@pytest.mark.parametrize("history_start", ("2010-01-04", "2020-01-02"))
def test_public_narrow_panel_retains_preceding_suspension_state(
    tmp_path, barrier, history_start
):
    dates = pd.bdate_range(history_start, periods=7).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    bars = frames["market_bars_1d"]
    asset = bars.source_asset_id.eq("000001")
    columns = ["open", "high", "low", "close", "adj_open", "adj_close"]
    bars.loc[asset, columns] = 100.0
    suspended = asset & bars.timestamp.between(dates[2], dates[4])
    bars.loc[suspended, "filling_flag"] = 1
    bars.loc[suspended, columns] = 50.0
    bars.loc[asset & bars.timestamp.eq(dates[5]), columns] = 120.0
    bars.loc[asset & bars.timestamp.eq(dates[5]), "period_return"] = 0.2
    if barrier == "unknown":
        frames["market_bars_1d"] = bars.loc[
            ~(asset & bars.timestamp.eq(dates[1]))
        ].copy()
    elif barrier == "raw_missing":
        bars.loc[
            asset & bars.timestamp.eq(dates[1]), ["open", "high", "low", "close"]
        ] = np.nan
    elif barrier == "adjusted_missing":
        bars.loc[asset & bars.timestamp.eq(dates[1]), ["adj_open", "adj_close"]] = (
            np.nan
        )
    inputs = _write_inputs(tmp_path / "source", frames)
    outputs = []
    for index, start in enumerate((dates[0], dates[3])):
        path = _prepare(
            tmp_path / f"result-{index}", "market_panel_1d", inputs, start, dates[5]
        )
        output = read_dataset(path)
        outputs.append(output.loc[output.timestamp.ge(dates[3])].reset_index(drop=True))
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)
    rows = outputs[1].loc[outputs[1].asset_id.eq("000001")].set_index("timestamp")
    if barrier in {"unknown", "raw_missing"}:
        assert pd.isna(rows.loc[dates[3], "close"])
    else:
        assert rows.loc[dates[3], "close"] == 100.0
    if barrier in {"unknown", "adjusted_missing"}:
        assert pd.isna(rows.loc[dates[3], "adj_close"])
    else:
        assert rows.loc[dates[3], "adj_close"] == 100.0
        assert rows.loc[dates[5], "period_return"] == pytest.approx(0.2)


@pytest.mark.parametrize("missing", (0, 2, 4))
@pytest.mark.parametrize("closed_market", (False, True))
def test_public_panel_keeps_calendar_endpoints_without_market_bars(
    tmp_path, missing, closed_market
):
    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    frames["market_bars_1d"] = (
        frames["market_bars_1d"]
        .loc[frames["market_bars_1d"].timestamp.ne(dates[missing])]
        .copy()
    )
    if closed_market:
        calendar = frames["trading_calendar"]
        calendar.loc[calendar.calendar_date.eq(dates[1]), "is_trading_day"] = False
        frames["trading_calendar"] = pd.concat(
            [calendar, calendar.assign(source_market_type=4, is_trading_day=True)],
            ignore_index=True,
        )
        bars = frames["market_bars_1d"]
        bars.loc[bars.timestamp.eq(dates[1]), "filling_flag"] = 2
    inputs = _write_inputs(tmp_path / "source", frames)
    saved = read_dataset(
        _prepare(tmp_path / "result", "market_panel_1d", inputs, dates[0], dates[-1])
    )
    assert len(saved) == (12 if closed_market else 15)
    assert saved.timestamp.nunique() == (4 if closed_market else 5)
    if closed_market:
        assert not saved.timestamp.eq(dates[1]).any()
    missing_rows = saved.loc[saved.timestamp.eq(dates[missing])]
    assert len(missing_rows) == 3
    assert missing_rows[["close", "adj_close", "is_trading"]].isna().all().all()


@pytest.mark.parametrize("block_rows", [300, 65536])
@pytest.mark.parametrize("known_closed", [False, True])
def test_market_model_keeps_only_known_closed_history_and_aligned_returns(
    tmp_path, monkeypatch, block_rows, known_closed
):
    from fractions import Fraction

    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.research.filtering import a_share_eligibility

    dates = pd.bdate_range("2020-01-02", periods=650).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    for data_id in ("stock_master", "stock_info_annual"):
        frames[data_id]["listing_date"] = dates[0]
    bars = frames["market_bars_1d"]
    for position, asset in enumerate(("000001", "000002", "000003")):
        mask = bars.source_asset_id.eq(asset)
        returns = 0.01 * np.sin(np.arange(650) * 0.23 + position) + 0.003 * np.cos(
            np.arange(650) * 0.13 + position
        )
        if asset == "000001" and known_closed:
            returns[640] = 0.0
        prices = (20.0 + position) * np.cumprod(1.0 + returns)
        for column in ("open", "close", "adj_open", "adj_close"):
            bars.loc[mask, column] = prices
        bars.loc[mask, "high"] = prices + 0.5
        bars.loc[mask, "low"] = prices - 0.5
        bars.loc[mask, "period_return"] = returns
        bars.loc[mask, "amount"] = prices * 100.0
        if asset != "000001":
            bars.loc[mask, "market_type"] = 4
    missing = bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[640])
    if known_closed:
        bars.loc[missing, "filling_flag"] = 2
    else:
        frames["market_bars_1d"] = bars.loc[~missing].copy()
    calendar = frames["trading_calendar"]
    other = calendar.assign(source_market_type=4)
    if known_closed:
        calendar.loc[calendar.calendar_date.eq(dates[640]), "is_trading_day"] = False
    frames["trading_calendar"] = pd.concat((calendar, other), ignore_index=True)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "result", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    inputs["market_model_factors"] = _prepare(
        tmp_path / "result", "market_model_factors", inputs, dates[0], dates[-1]
    )

    def restored(data_id):
        return read_dataset(inputs[data_id])

    panel = restored("market_panel_1d").sort_values(["timestamp", "asset_id"])
    factors = restored("market_model_factors").sort_values(["timestamp", "asset_id"])
    pd.testing.assert_frame_equal(
        factors.loc[:, ["market_id", "asset_id", "timestamp"]].reset_index(drop=True),
        panel.loc[:, ["market_id", "asset_id", "timestamp"]].reset_index(drop=True),
        check_exact=True,
    )
    actual = factors.loc[factors.asset_id.eq("000001")].set_index("timestamp")
    assert (
        actual.loc[dates[639], ["beta_252d", "residual_volatility_60d"]].notna().all()
    )
    if known_closed:
        assert dates[640] not in actual.index
    else:
        assert dates[640] in actual.index
        gap = panel.loc[panel.asset_id.eq("000001") & panel.timestamp.eq(dates[640])]
        assert gap.period_return.isna().all()
    # A single unknown pair does not erase sufficient observations in the
    # nominal window. Its missing lagged cap also invalidates the next pair.
    assert (
        actual.loc[dates[641], ["beta_252d", "residual_volatility_60d"]].notna().all()
    )
    previous_caps = {}
    history, residuals = [], []
    pending = Fraction(1)
    expected_beta = expected_volatility = np.nan
    for timestamp, cross in panel.groupby("timestamp", sort=False):
        eligible = cross.loc[a_share_eligibility(cross, "a_share_all")]
        denominator = sum(previous_caps.get(asset, 0.0) for asset in eligible.asset_id)
        market = (
            sum(
                previous_caps.get(row.asset_id, 0.0) * row.period_return
                for row in eligible.itertuples()
            )
            / denominator
            if np.isfinite(denominator) and denominator > 0
            else np.nan
        )
        if not known_closed and timestamp in (dates[640], dates[641]):
            # Required quote missing at 640; required lagged weight missing at
            # 641. Neither can be replaced by a survivor-only market proxy.
            market = np.nan
        stock = cross.loc[cross.asset_id.eq("000001")]
        if stock.empty:
            pending *= 1 + Fraction.from_float(market)
        else:
            interval = (
                float(pending * (1 + Fraction.from_float(market)) - 1)
                if np.isfinite(market)
                else np.nan
            )
            pending = Fraction(1)
            value = stock.period_return.iloc[0]
            value = np.nan if pd.isna(value) else float(value)
            history.append((interval, value))
            if not known_closed and timestamp in (dates[640], dates[641]):
                assert np.isnan(history[-1][0])
            expected_beta = residual = np.nan
            if len(history) >= 252:
                window = np.array(history[-252:])
                paired = window[np.isfinite(window).all(axis=1)]
                if len(paired) >= 189:
                    alpha, expected_beta = np.linalg.lstsq(
                        np.column_stack((np.ones(len(paired)), paired[:, 0])),
                        paired[:, 1],
                        rcond=None,
                    )[0]
                    if np.isfinite(window[-1]).all():
                        residual = value - alpha - expected_beta * interval
            residuals.append(residual)
            expected_volatility = np.nan
            if len(residuals) >= 60:
                window = np.asarray(residuals[-60:])
                valid = window[np.isfinite(window)]
                if len(valid) >= 45:
                    expected_volatility = np.std(valid, ddof=0)
            if timestamp == dates[641]:
                assert actual.loc[timestamp, "beta_252d"] == pytest.approx(
                    expected_beta, rel=1e-12
                )
                assert actual.loc[
                    timestamp, "residual_volatility_60d"
                ] == pytest.approx(expected_volatility, rel=1e-12, abs=1e-16)
        previous_caps.update(
            {
                asset: np.nan if pd.isna(cap) else float(cap)
                for asset, cap in zip(
                    cross.asset_id, cross.float_market_cap, strict=True
                )
            }
        )
    assert actual.iloc[-1].beta_252d == pytest.approx(expected_beta, rel=1e-12)
    assert actual.iloc[-1].residual_volatility_60d == pytest.approx(
        expected_volatility, rel=1e-12, abs=1e-16
    )


@pytest.mark.parametrize("block_rows", [3, 65536])
def test_listing_age_keeps_prior_market_sessions_in_native_narrow_panel(
    tmp_path, monkeypatch, block_rows
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.research.filtering import a_share_eligibility

    dates = pd.bdate_range("2020-01-02", periods=256).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    for data_id in ("stock_master", "stock_info_annual"):
        frames[data_id]["listing_date"] = dates[0]
    bars = frames["market_bars_1d"]
    bars.loc[
        bars.source_asset_id.eq("000001") & bars.timestamp.ge(dates[254]), "market_type"
    ] = 4
    moved = frames["trading_calendar"].assign(source_market_type=4)
    moved.loc[moved.calendar_date.isin(dates[:10]), "is_trading_day"] = False
    frames["trading_calendar"] = pd.concat(
        (frames["trading_calendar"], moved), ignore_index=True
    )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    inputs = _write_inputs(tmp_path / "source", frames)
    saved = (
        read_dataset(
            _prepare(
                tmp_path / "result", "market_panel_1d", inputs, dates[-4], dates[-1]
            )
        )
        .sort_values(["timestamp", "asset_id"])
        .reset_index(drop=True)
    )
    moved_asset = saved.loc[saved.asset_id.eq("000001")]
    assert moved_asset.timestamp.tolist() == dates[-4:].tolist()
    assert moved_asset.market_type.tolist() == [1, 1, 4, 4]
    assert moved_asset.listing_age_trading_days.tolist() == [253, 254, 255, 256]
    assert str(moved_asset.listing_age_trading_days.dtype) == "Int64"
    assert a_share_eligibility(moved_asset, "a_share_all").all()
    assert not saved.duplicated(["market_id", "asset_id", "timestamp"]).any()


@pytest.mark.parametrize("block_rows", [3, 65536])
@pytest.mark.parametrize("start_index", [0, 2])
def test_closed_source_market_event_survives_native_projection_and_unknown_barrier(
    tmp_path, monkeypatch, block_rows, start_index
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )

    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    for data_id in ("stock_master", "stock_info_annual"):
        frames[data_id]["listing_date"] = dates[0]
    bars = frames["market_bars_1d"]
    bars.loc[bars.timestamp.ge(dates[1]), "market_type"] = 4
    bars.loc[bars.timestamp.eq(dates[1]), "filling_flag"] = 2
    frames["market_bars_1d"] = bars.loc[bars.timestamp.ne(dates[2])].copy()
    calendar = frames["trading_calendar"]
    moved = calendar.assign(source_market_type=4)
    calendar.loc[calendar.calendar_date.isin(dates[1:3]), "is_trading_day"] = False
    moved.loc[moved.calendar_date.eq(dates[1]), "is_trading_day"] = False
    frames["trading_calendar"] = pd.concat((calendar, moved), ignore_index=True)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "history", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    inputs["price_volume_facts"] = _prepare(
        tmp_path / "result", "price_volume_facts", inputs, dates[start_index], dates[-1]
    )
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "result", "market_panel_1d", inputs, dates[start_index], dates[-1]
    )

    def restored(data_id):
        return (
            read_dataset(inputs[data_id])
            .sort_values(["timestamp", "asset_id"])
            .reset_index(drop=True)
        )

    panel = restored("market_panel_1d")
    expected_dates = dates[start_index:].difference(dates[1:2])
    assert panel.timestamp.drop_duplicates().tolist() == expected_dates.tolist()
    assert len(panel) == 3 * len(expected_dates)
    assert not panel.duplicated(["market_id", "asset_id", "timestamp"]).any()
    assert panel.loc[panel.timestamp.ge(dates[2]), "market_type"].eq(4).all()
    assert (
        panel.loc[panel.timestamp.eq(dates[2]), ["close", "adj_close", "is_trading"]]
        .isna()
        .all()
        .all()
    )
    factors = restored("price_volume_facts")
    assert factors.loc[factors.timestamp.eq(dates[3]), "reversal_1d"].isna().all()
    assert factors.loc[factors.timestamp.eq(dates[4]), "reversal_1d"].notna().all()


@pytest.mark.parametrize("block_rows", [3, 65536])
def test_public_panel_keeps_market_changes_across_closed_and_missing_slabs(
    tmp_path, monkeypatch, block_rows
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )

    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    bars = frames["market_bars_1d"]
    bars.loc[bars.timestamp.ge(dates[1]), "market_type"] = 4
    bars.loc[bars.timestamp.eq(dates[1]), "filling_flag"] = 2
    frames["market_bars_1d"] = bars.loc[bars.timestamp.ne(dates[2])].copy()
    calendar = frames["trading_calendar"]
    moved = calendar.assign(source_market_type=4)
    moved.loc[moved.calendar_date.eq(dates[1]), "is_trading_day"] = False
    frames["trading_calendar"] = pd.concat([calendar, moved], ignore_index=True)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    inputs = _write_inputs(tmp_path / "source", frames)
    saved = read_dataset(
        _prepare(tmp_path / "result", "market_panel_1d", inputs, dates[0], dates[-1])
    )
    assert saved.timestamp.nunique() == 4
    assert not saved.timestamp.eq(dates[1]).any()
    assert saved.loc[saved.timestamp.eq(dates[0]), "market_type"].eq(1).all()
    assert saved.loc[saved.timestamp.ge(dates[2]), "market_type"].eq(4).all()
    missing = saved.loc[saved.timestamp.eq(dates[2])]
    assert len(missing) == 3
    assert missing[["close", "adj_close", "is_trading"]].isna().all().all()


def test_public_expectations_window_preserves_fresh_preceding_forecasts(tmp_path):
    dates = pd.bdate_range("2020-04-01", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    declared = dates[2] - pd.Timedelta(days=7)
    frames["analyst_forecast"] = pd.DataFrame(
        {
            "source_asset_id": "000001",
            "timestamp": declared,
            "declare_date": declared,
            "forecast_end_date": pd.Timestamp("2020-12-31"),
            "institution_id": list("ABC"),
            "report_id": ["rA", "rB", "rC"],
            "analyst_ids": list("abc"),
            "forecast_eps": [1.0, 2.0, 3.0],
            "forecast_pe": 10.0,
        }
    )
    inputs = _write_inputs(tmp_path / "source", frames)
    inputs["market_panel_1d"] = _prepare(
        tmp_path / "history", "market_panel_1d", inputs, dates[0], dates[-1]
    )
    outputs = []
    for index, start in enumerate((dates[0], dates[2])):
        saved = read_dataset(
            _prepare(
                tmp_path / f"result-{index}",
                "expectations_factors",
                inputs,
                start,
                dates[-1],
            )
        )
        outputs.append(saved.loc[saved.timestamp.ge(dates[2])].reset_index(drop=True))
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)
    selected = outputs[1].loc[outputs[1].asset_id.eq("000001")]
    assert selected.forward_earnings_yield_fy1.notna().all()


@pytest.mark.parametrize("dataset_id", ("pv_all", "pv_expanded", "fusion_nf_core"))
def test_explicit_preparation_reaches_eligible_research_dataset(tmp_path, dataset_id):
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.protocols.a_share import (
        research_dataset_catalog,
    )

    sessions = pd.bdate_range(
        "2020-01-02", periods=600 if dataset_id == "pv_expanded" else 400
    ).as_unit("ns")
    frames = _synthetic_interim_frames(sessions)
    if dataset_id == "pv_expanded":
        bars = frames["market_bars_1d"]
        bars["period_return"] = np.sin(np.arange(len(bars)) / 3) * 0.01
    if dataset_id == "fusion_nf_core":
        frames.update(_synthetic_financial_frames())
    inputs = _write_inputs(tmp_path / "source", frames)
    steps = [
        "market_panel_1d",
        "execution_mask",
        "price_volume_facts",
        "forward_returns",
    ]
    steps.append("market_model_factors")
    if dataset_id == "fusion_nf_core":
        steps.extend(("fundamental_pit", "fundamental_factors"))
    for name in steps:
        inputs[name] = _prepare(
            tmp_path / "prepared", name, inputs, sessions[0], sessions[-1]
        )
    spec = research_dataset_catalog().get(dataset_id)
    output = read_dataset(
        assemble(
            {
                "market_id": "a_share",
                "dataset_id": dataset_id,
                "inputs": {key: str(inputs[key]) for key in spec.input_data_ids},
                "time_window": {
                    "start": str(sessions[0].date()),
                    "end": str(sessions[-1].date()),
                },
                "output_root": str(tmp_path / "dataset"),
            }
        )
    )
    assert {"eligible", "model_target"}.issubset(output.columns)
    assert {"scenario_universe", "can_enter", "can_exit"}.isdisjoint(output)
    assert output.eligible.any()
    assert output.loc[output.eligible, "model_target"].notna().any()
    if dataset_id == "pv_expanded":
        for name in (
            "share_price",
            "avg_amount_126d",
            "turnover_250d",
            "beta_252d",
            "residual_volatility_60d",
        ):
            assert output[name].notna().any(), name


@pytest.mark.parametrize("negative_component", ("current_ytd", "prior_fy", "prior_ytd"))
def test_negative_source_capex_propagates_through_native_pit_and_factor_revision(
    tmp_path, monkeypatch, negative_component
):
    from quant_research_platform.markets.implementations.a_share.fundamentals import (
        FINANCIAL_VALUE_COLUMNS,
    )
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )

    sessions = pd.bdate_range("2020-04-01", "2020-04-10").as_unit("ns")
    frames = _synthetic_interim_frames(sessions)
    financial = _synthetic_financial_frames()
    periods = pd.to_datetime(["2019-03-31", "2019-12-31", "2020-03-31"])
    for data_id in FINANCIAL_VALUE_COLUMNS:
        financial[data_id] = (
            financial[data_id]
            .loc[financial[data_id].report_period.isin(periods)]
            .copy()
        )
    financial["financial_balance"]["total_assets"] = 100.0
    cash = financial["financial_cashflow"]
    components = {
        "prior_ytd": (periods[0], 10.0),
        "prior_fy": (periods[1], 100.0),
        "current_ytd": (
            periods[2],
            100.0 if negative_component == "prior_fy" else 10.0,
        ),
    }
    for period, value in components.values():
        cash.loc[cash.report_period.eq(period), "capex_cash_paid"] = value
    for period, value in zip(periods, (-10.0, -30.0, -20.0), strict=True):
        cash.loc[cash.report_period.eq(period), "operating_cash_flow"] = value
    bad_period, _value = components[negative_component]
    cash.loc[cash.report_period.eq(bad_period), "capex_cash_paid"] = -5.0
    revision = cash.loc[cash.report_period.eq(bad_period)].copy()
    restored_value = 100.0 if negative_component == "prior_fy" else 15.0
    revision["capex_cash_paid"] = restored_value
    revision["is_corrected"] = True
    revision["correction_disclosure_date"] = pd.Timestamp("2020-04-06")
    revision["version_identity"] = "correction@2020-04-06"
    financial["financial_cashflow"] = pd.concat((cash, revision), ignore_index=True)
    disclosures = financial["financial_first_disclosure"]
    disclosures = disclosures.loc[disclosures.report_period.isin(periods)].copy()
    for period, disclosed in zip(
        periods, ("2019-04-20", "2020-01-20", "2020-04-02"), strict=True
    ):
        disclosures.loc[
            disclosures.report_period.eq(period), "first_disclosure_date"
        ] = pd.Timestamp(disclosed)
    financial["financial_first_disclosure"] = disclosures
    frames.update(financial)
    inputs = _write_inputs(tmp_path / "source", frames)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 6)
    for name in ("market_panel_1d", "fundamental_pit", "fundamental_factors"):
        inputs[name] = _prepare(
            tmp_path / "prepared", name, inputs, sessions[0], sessions[-1]
        )

    def restored(data_id):
        return (
            read_dataset(inputs[data_id])
            .sort_values(["timestamp", "asset_id"])
            .reset_index(drop=True)
        )

    pit = restored("fundamental_pit")
    factors = restored("fundamental_factors")
    assert not factors.duplicated(["market_id", "asset_id", "timestamp"]).any()
    assert len(factors) == 3 * len(sessions)
    assert factors.timestamp.dtype == np.dtype("datetime64[ns]")
    assert factors.capex_to_assets.dtype == np.dtype("float64")
    invalid = factors.timestamp.between("2020-04-03", "2020-04-06")
    assert factors.loc[invalid, "capex_to_assets"].isna().all()
    assert (
        pit.loc[
            pit.timestamp.between("2020-04-03", "2020-04-06"), "capex_cash_paid_ttm"
        ]
        .isna()
        .all()
    )
    expected_ttm = {"current_ytd": 105.0, "prior_fy": 190.0, "prior_ytd": 95.0}[
        negative_component
    ]
    assert (
        pit.loc[pit.timestamp.ge("2020-04-07"), "capex_cash_paid_ttm"]
        .eq(expected_ttm)
        .all()
    )
    assert (
        factors.loc[factors.timestamp.ge("2020-04-07"), "capex_to_assets"]
        .eq(expected_ttm / 100)
        .all()
    )
    assert (
        factors.loc[factors.timestamp.ge("2020-04-03"), "cfo_to_assets"].eq(-0.4).all()
    )
    if negative_component != "prior_fy":
        assert (
            pit.loc[pit.timestamp.lt("2020-04-03"), "capex_cash_paid_ttm"]
            .eq(100.0)
            .all()
        )
    evidence = read_dataset(inputs["financial_cashflow"])
    assert (
        evidence.loc[
            evidence.version_identity.eq("original")
            & evidence.report_period.eq(bad_period),
            "capex_cash_paid",
        ]
        .eq(-5.0)
        .all()
    )
    assert (
        evidence.loc[evidence.is_corrected, "capex_cash_paid"].eq(restored_value).all()
    )


def test_optional_source_absence_does_not_discover_neighboring_inputs(tmp_path):
    sessions = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    inputs = _write_inputs(tmp_path / "source", _synthetic_interim_frames(sessions))
    optional = set(
        processed_catalog().get("market_panel_1d").optional_source_input_data_ids
    )
    supplied = {name: path for name, path in inputs.items() if name not in optional}
    first = _prepare(
        tmp_path / "absent", "market_panel_1d", supplied, sessions[0], sessions[-1]
    )
    restored = read_dataset(first)
    assert len(restored) == 15
    assert restored.industry_level1_code.isna().all()
    supplied["industry_classification"] = inputs["industry_classification"]
    second = _prepare(
        tmp_path / "present", "market_panel_1d", supplied, sessions[0], sessions[-1]
    )
    assert len(read_dataset(second)) == 15
    pd.testing.assert_frame_equal(read_dataset(first), restored)


@pytest.mark.parametrize("failure", ("present_invalid", "required_absent"))
def test_only_optional_absence_can_proceed(tmp_path, failure):
    sessions = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    inputs = _write_inputs(tmp_path / "source", _synthetic_interim_frames(sessions))
    if failure == "present_invalid":
        inputs["industry_classification"] = tmp_path / "missing-provided-source"
    else:
        del inputs["market_cap"]
    with pytest.raises((ValueError, FileNotFoundError)):
        _prepare(
            tmp_path / "result", "market_panel_1d", inputs, sessions[0], sessions[-1]
        )
    assert not (tmp_path / "result" / "market_panel_1d").exists()
