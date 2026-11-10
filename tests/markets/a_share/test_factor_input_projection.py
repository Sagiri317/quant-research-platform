"""Factor producers retain exactly their mathematical clocks and input columns."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed.producer import (
    _align_panel_observations,
)
from quant_research_platform.research.factors.engine import factor_lookback_bars
from quant_research_platform.research.factors.market_structure import factor_definitions
from tests.markets.a_share.test_calendar_dependency_scope import (
    _calendar,
    _panel,
    _prepare,
)


@pytest.mark.parametrize(
    "factor,lookback,expected",
    [
        ("market_up_fraction", 0, 1.0),
        ("market_above_ma20_fraction", 19, 1.0),
        ("market_high_low_balance_60d", 60, 1.0),
        ("industry_return_relative_20d", 20, 0.0),
    ],
)
def test_structure_only_consumes_required_calendar_halo(
    tmp_path, factor, lookback, expected
):
    dates = pd.bdate_range("2023-01-02", periods=90)
    panel = pd.concat([_panel(dates, "a"), _panel(dates, "b")]).assign(
        is_a_share_scope=True,
        industry_standard_id="fixture",
        industry_level1_code="one",
    )
    result = _prepare(
        tmp_path,
        panel,
        _calendar({1: dates[80 - lookback : 81]}),
        "market_structure_factors",
        start=dates[80],
        end=dates[80],
        factor_columns=[factor],
    )
    assert result.asset_id.tolist() == ["a", "b"]
    np.testing.assert_allclose(result[factor], expected, rtol=1e-12, atol=1e-14)
    definition = next(d for d in factor_definitions() if d.output_column == factor)
    assert factor_lookback_bars((definition,)) == lookback
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import read_dataset

    config = {
        "market_id": "a_share",
        "data_id": "market_structure_factors",
        "inputs": {"market_panel_1d": str(tmp_path / "panel")},
        "time_window": {"start": str(dates[80].date()), "end": str(dates[80].date())},
        "factor_columns": [factor],
        "output_root": str(tmp_path / "without_calendar"),
    }
    if lookback:
        with pytest.raises(ValueError, match="data_operation_input_paths_mismatch"):
            prepare(config)
    else:
        pd.testing.assert_frame_equal(read_dataset(prepare(config)), result)
        with pytest.raises(FileNotFoundError):
            prepare(
                {
                    **config,
                    "inputs": {
                        **config["inputs"],
                        "trading_calendar": str(tmp_path / "missing"),
                    },
                    "output_root": str(tmp_path / "invalid_clock"),
                }
            )


def test_alignment_retains_gaps_observed_keys_and_consumed_domain():
    dates = pd.date_range("2024-01-02", periods=4).as_unit("ns")
    declared = pd.DataFrame(
        {"market_id": "a_share", "asset_id": "a", "timestamp": dates[[0, 2, 3]]}
    )
    selected = declared.iloc[[2, 0]].assign(amount=[3.0, 1.0])
    outside_halo = selected.iloc[[0]].assign(timestamp=dates[-1] + pd.Timedelta(days=1))
    dependencies = {
        "market_panel_1d": pd.concat([selected, outside_halo], ignore_index=True),
        "style": declared.iloc[[0]].assign(style=5.0),
    }
    panel, keys = _align_panel_observations(declared, dependencies)
    assert panel.timestamp.tolist() == dates[[0, 2, 3]].tolist()
    assert panel._panel_observed.tolist() == [True, False, True]
    np.testing.assert_allclose(panel.amount, [1.0, np.nan, 3.0], equal_nan=True)
    pd.testing.assert_frame_equal(
        keys.reset_index(drop=True), selected[declared.columns].reset_index(drop=True)
    )
    dependencies["style"] = dependencies["style"].assign(timestamp=dates[1])
    with pytest.raises(ValueError, match="a_share_input_key_outside_panel"):
        _align_panel_observations(declared, dependencies)


@pytest.mark.parametrize("instrument_type", ["stock", "equity_etf", "gold_etf"])
@pytest.mark.parametrize("end_second", [6, 9])
def test_native_hawkes_reads_event_dependencies_and_preserves_asset_type_math(
    tmp_path, monkeypatch, instrument_type, end_second
):
    import json

    from quant_research_platform.markets.implementations.a_share.processed import (
        high_frequency,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.protocols.a_share_hft import processed_specs
    from quant_research_platform.sources.venvoo import interim_spec, normalize
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.markets.a_share.test_high_frequency import (
        INSTRUMENTS,
        raw_events,
        raw_quotes,
    )

    instruments = [{**INSTRUMENTS[0], "instrument_type": instrument_type}]
    quotes = normalize(raw_quotes(), "hft_quotes", instruments)
    raw = raw_events("hft_trades")
    raw["trade_code"] = ["T", "T", "T", "?"]
    raw["time"] = [93000000, 93002000, 93005000, 93008000]
    raw["bs_flag"] = ["B", "S", "B", "B"]
    trades = normalize(raw, "hft_trades", instruments)
    orders = normalize(raw_events("hft_orders"), "hft_orders", instruments)
    panel = high_frequency.build_panel(quotes, orders, trades, max_gap_seconds=6)
    paths = {name: str(tmp_path / name) for name in ("hft_panel", "hft_trades")}
    write_dataset(
        paths["hft_panel"],
        panel,
        spec=next(spec for spec in processed_specs() if spec.data_id == "hft_panel"),
    )
    write_dataset(paths["hft_trades"], trades, spec=interim_spec("hft_trades"))
    model = tmp_path / "model"
    model.mkdir()
    baseline = np.array([1.0, 2.0])
    branching = np.array([[0.2, 0.1], [0.3, 0.1]])
    (model / "model.json").write_text(
        json.dumps(
            {
                "kind": "bivariate_exponential_hawkes",
                "fit_cutoff": "2018-02-14T00:00:00",
                "parameters": {
                    "baseline": baseline.tolist(),
                    "branching": branching.tolist(),
                    "decay": 2.0,
                },
            }
        )
    )
    reads = {}
    materialize = high_frequency.materialize_dataset_session

    def read(session):
        frame = materialize(session)
        reads[session.spec["data_id"]] = set(frame)
        assert frame.timestamp.max() <= pd.Timestamp(
            "2018-02-14T01:30:00"
        ) + pd.Timedelta(seconds=end_second)
        return frame

    monkeypatch.setattr(high_frequency, "materialize_dataset_session", read)
    root = prepare(
        {
            "market_id": "a_share",
            "data_id": "hft_hawkes_factors",
            "inputs": paths,
            "time_window": {
                "start": "2018-02-14T01:30:00",
                "end": f"2018-02-14T01:30:0{end_second}",
            },
            "output_root": str(tmp_path / "factors"),
            "parameters": {"model": str(model)},
        }
    )
    result = read_dataset(root)
    expected = np.tile(baseline, (3, 1))
    for i, query in enumerate((0, 3, 6)):
        for event, side in ((0, 0), (2, 1), (5, 0)):
            if event < query:
                expected[i] += 2 * np.exp(-2 * (query - event)) * branching[:, side]
    np.testing.assert_allclose(
        result[["hawkes_buy_intensity", "hawkes_sell_intensity"]].iloc[:3], expected
    )
    if end_second == 9:
        assert result.iloc[-1].filter(like="hawkes_").isna().all()
    else:
        assert len(result) == 3
    assert reads["hft_panel"] == {
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "segment_id",
    }
    assert reads["hft_trades"] == {
        "asset_id",
        "timestamp",
        "source_row",
        "event_kind",
        "side",
    }


@pytest.mark.parametrize(
    "factor,link,trade_fields,order_fields,expected",
    [
        ("trade_interval_seconds", None, (), None, [np.nan, 2, 3, 3]),
        ("trade_signed_run", None, ("side", "volume"), None, [1, 2, 3, 3]),
        (
            "trade_interval_return_interaction",
            None,
            ("price",),
            None,
            [
                np.nan,
                np.nan,
                2.5 * (10.02 / 10.01 - 1),
                2.5 * (10.02 / 10.01 - 1),
            ],
        ),
        ("observed_cancel_count_fraction", None, (), (), [np.nan, 0.5, 0, 0]),
        ("observed_order_age_seconds", None, None, None, [np.nan] * 4),
        (
            "observed_order_age_seconds",
            "ex_order_id",
            ("volume", "bid_order_id", "ask_order_id"),
            ("exchange", "order_type", "side", "volume", "ex_order_id"),
            [0, 1, 1, 4],
        ),
    ],
)
def test_native_event_projection_uses_catalog_dependencies(
    tmp_path, monkeypatch, factor, link, trade_fields, order_fields, expected
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        high_frequency,
    )
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.protocols.a_share_hft import processed_specs
    from quant_research_platform.sources.venvoo import interim_spec, normalize
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.markets.a_share.test_high_frequency import (
        INSTRUMENTS,
        raw_events,
        raw_quotes,
    )

    quotes = normalize(raw_quotes(), "hft_quotes", INSTRUMENTS)
    raw = raw_events("hft_trades").assign(trade_code="T", bs_flag="B")
    if factor == "observed_cancel_count_fraction":
        raw.loc[1, "trade_code"] = "C"
    trades = normalize(raw, "hft_trades", INSTRUMENTS)
    orders = normalize(raw_events("hft_orders"), "hft_orders", INSTRUMENTS)
    panel = high_frequency.build_panel(quotes, orders, trades, max_gap_seconds=6)
    paths = {
        name: str(tmp_path / name) for name in ("hft_panel", "hft_trades", "hft_orders")
    }
    for name, frame in (
        ("hft_panel", panel),
        ("hft_trades", trades),
        ("hft_orders", orders),
    ):
        spec = (
            next(spec for spec in processed_specs() if spec.data_id == name)
            if name == "hft_panel"
            else interim_spec(name)
        )
        write_dataset(paths[name], frame, spec=spec)
    reads = {}
    materialize = high_frequency.materialize_dataset_session

    def read(session):
        frame = materialize(session)
        reads[session.spec["data_id"]] = set(frame)
        return frame

    monkeypatch.setattr(high_frequency, "materialize_dataset_session", read)
    root = prepare(
        {
            "market_id": "a_share",
            "data_id": "hft_event_factors",
            "inputs": paths,
            "time_window": {"start": "2018-02-14", "end": "2018-02-14"},
            "output_root": str(tmp_path / "factors"),
            "factor_columns": [factor],
            "parameters": {"event_window": 2, "order_link_field": link},
        }
    )
    result = read_dataset(root)
    np.testing.assert_allclose(result[factor], expected, equal_nan=True)
    for name, fields in (("hft_trades", trade_fields), ("hft_orders", order_fields)):
        if fields is None:
            assert name not in reads
        else:
            assert reads[name] == {
                "asset_id",
                "timestamp",
                "source_row",
                "event_kind",
                *fields,
            }
