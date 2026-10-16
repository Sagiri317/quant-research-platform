from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    market_name_for_type,
)
from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.filtering import a_share_eligibility
from quant_research_platform.research.portfolios import (
    production as portfolio_production,
)
from quant_research_platform.research.protocols.a_share import factor_catalog
from quant_research_platform.runtime.dataframes import iter_complete_session_frames
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import native_filesystem_path
from tests.helpers.a_share_sources import (
    _published_inputs,
    _source_catalog,
    _synthetic_financial_frames,
    _synthetic_interim_frames,
)
from tests.helpers.paths import StorageLayout


def test_local_panel_window_keeps_public_industry_history_and_unknown_barrier(tmp_path):
    from quant_research_platform.sources.reference import interim
    from tests.helpers.a_share_raw import write_industry_reference_fixture

    dates = pd.bdate_range("2020-01-06", periods=5)
    source = _synthetic_interim_frames(dates)
    source["stock_master"]["timestamp"] = source["stock_master"]["listing_date"]
    source["industry_classification_versions"]["implement_date"] = pd.Timestamp(
        "2012-10-26"
    )
    source["industry_classification_sets"] = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012"] * 6,
            "industry_code": ["A", "A01", "B", "B01", "C", "C01"],
            "industry_name": [
                "old",
                "old leaf",
                "barrier",
                "blocked leaf",
                "new",
                "new leaf",
            ],
            "parent_industry_code": ["", "A", "", "B", "", "C"],
            "industry_level": ["1", "2", "1", "2", "1", "2"],
            "industry_id": [str(index) for index in range(6)],
        }
    )
    source["industry_classification"] = (
        source["industry_classification"]
        .iloc[:1]
        .assign(
            timestamp=pd.Timestamp("2020-01-07"),
            implement_date=pd.Timestamp("2020-01-07"),
            industry_level1_code="B01",
            industry_level1_name="blocked leaf",
        )
    )
    source["stock_info_changes"] = (
        source["stock_info_changes"]
        .iloc[:1]
        .assign(
            timestamp=pd.Timestamp("2020-01-07"),
            effective_date=pd.Timestamp("2020-01-07"),
            announcement_date=pd.NaT,
            value_after="blocked leaf",
        )
    )
    observations = tuple(
        (
            (
                snapshot_id,
                "CSRC2012",
                observed,
                published,
                [
                    {
                        "code": "000001",
                        "name": "synthetic issuer",
                        "category": category,
                        "major": "",
                        "industry": "01",
                        "source_locator": "page:1",
                    }
                ],
            )
            for snapshot_id, observed, published, category in (
                ("initial", "2019-12-31", "2020-01-03", "A"),
                ("late-old", "2020-01-06", "2020-01-08", "B"),
                ("recovery", "2020-01-08", "2020-01-09", "C"),
                ("future", "2020-01-09", "2020-01-12", "B"),
            )
        )
    )
    raw_root = tmp_path / "raw"
    write_industry_reference_fixture(raw_root, observations=observations)
    layout = StorageLayout(tmp_path)
    spec = processed_catalog().get("market_panel_1d")
    interim.materialize_interim_data(
        output_roots={
            interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID: layout.interim_data_root(
                "reference", interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
            )
        },
        raw_root=raw_root / "reference",
        data_ids=(interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,),
        end_date="2020-01-10",
    )
    snapshots = read_dataset(
        layout.interim_data_root(
            "reference", interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID
        )
    )
    assert snapshots["snapshot_id"].tolist() == ["initial", "late-old", "recovery"]
    for data_id in spec.source_input_data_ids:
        if data_id == interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID:
            continue
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    result = read_dataset(root)
    asset = result.loc[result["asset_id"].eq("000001")]
    assert asset["timestamp"].tolist() == list(dates)
    assert asset["industry_level1_code"].fillna("unknown").tolist() == [
        "A",
        "unknown",
        "unknown",
        "unknown",
        "C",
    ]
    assert asset["is_nonfinancial_industry"].fillna(False).tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_panel_window_preserves_listing_age_history_and_eligibility(tmp_path):
    dates = pd.bdate_range("2019-01-02", periods=256).as_unit("ns")
    requested = dates[250:255]
    source = _synthetic_interim_frames(dates)
    stock = source["stock_master"]
    stock["listing_date"] = [dates[0], dates[0], requested[0]]
    stock["timestamp"] = stock["listing_date"]
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(requested[0], requested[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    assert actual["timestamp"].tolist() == [
        timestamp for timestamp in requested for _ in range(3)
    ]
    assert actual["listing_age_trading_days"].tolist() == [
        age for offset in range(5) for age in (251 + offset, 251 + offset, 1 + offset)
    ]
    assert a_share_eligibility(actual, "a_share_all").tolist() == [
        False,
        False,
        False,
        *[True, True, False] * 4,
    ]


def test_persisted_panel_market_codes_reach_real_fee_mapping(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=66)
    source = _synthetic_interim_frames(dates)
    source["market_bars_1d"]["market_type"] = source["market_bars_1d"][
        "source_asset_id"
    ].map({"000001": 1.0, "000002": 4.0, "000003": 16.0})
    source["trading_calendar"] = pd.concat(
        [
            source["trading_calendar"].assign(source_market_type=code)
            for code in (1, 4, 16)
        ],
        ignore_index=True,
    )
    assert "market_type" not in source["stock_master"]
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    window = TimeWindow(dates[0], dates[-1])
    columns = ("market_id", "asset_id", "timestamp", "market_type")
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=window,
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    with closing_iterator(
        portfolio_production._iter_input_sessions(root, columns, window)
    ) as sessions:
        for timestamp in (dates[0], dates[64], dates[65]):
            panel = portfolio_production._read_next_session(
                sessions, columns, timestamp
            ).set_index("asset_id")
            assert {
                asset: market_name_for_type(panel.loc[asset, "market_type"])
                for asset in panel.index
            } == {"000001": "shanghai", "000002": "shenzhen", "000003": "shenzhen"}


def test_processed_blocks_preserve_full_history_math(
    tmp_path,
    monkeypatch,
):
    dates = pd.bdate_range("2020-01-02", periods=492)
    source = _synthetic_interim_frames(dates)
    source.update(_synthetic_financial_frames())
    disclosures = source["financial_first_disclosure"]
    disclosures["first_disclosure_date"] = disclosures["first_disclosure_date"].clip(
        lower=dates[80]
    )
    bars = source["market_bars_1d"]
    keep = ~(
        bars["source_asset_id"].eq("000001")
        & bars["timestamp"].between(dates[66], dates[250])
        | bars["source_asset_id"].eq("000002")
        & bars["timestamp"].between(dates[1], dates[190])
    )
    retained_rows = bars.loc[
        bars["source_asset_id"].eq("000001") & bars["timestamp"].eq(dates[100])
        | bars["source_asset_id"].eq("000002") & bars["timestamp"].eq(dates[5])
    ].copy()
    retained_rows["filling_flag"] = 1
    retained_rows["adj_close"] = 777.0
    bars = (
        pd.concat([bars.loc[keep], retained_rows], ignore_index=True)
        .sort_values(["timestamp", "source_asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    first = bars["source_asset_id"].eq("000002") & bars["timestamp"].eq(dates[0])
    bars.loc[first, ["open", "high", "low", "close", "adj_open", "adj_close"]] = np.nan
    source["market_bars_1d"] = bars
    stock = source["stock_master"]
    stock.loc[stock["source_asset_id"].eq("000003"), "listing_date"] = dates[70]
    stock.loc[stock["source_asset_id"].eq("000003"), "delisting_date"] = dates[330]
    layout = StorageLayout(tmp_path)
    window = TimeWindow(pd.Timestamp("2019-01-01"), dates[-1])
    target_ids = (
        "market_panel_1d",
        "execution_mask",
        "forward_returns",
        "price_volume_facts",
        "fundamental_pit",
        "fundamental_factors",
    )
    for data_id, frame in source.items():
        spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(spec.source_id, data_id), frame, spec=spec
        )
    source = {
        data_id: read_dataset(
            layout.interim_data_root(_source_catalog().get(data_id).source_id, data_id)
        )
        for data_id in source
    }
    eager = {}
    for data_id in target_ids:
        spec = processed_catalog().get(data_id)
        eager[data_id] = (
            producer._compute_processed_frame(
                spec,
                source={
                    **producer._prepare_panel_source(
                        source,
                        producer.panels._canonical_stock_master(source["stock_master"]),
                    ),
                    "market_bars_1d": source["market_bars_1d"],
                }
                if data_id == "market_panel_1d"
                else source,
                dependencies={key: eager[key] for key in spec.processed_input_data_ids},
                trading_sessions=tuple(dates),
                financial_pit=producer._prepare_pit_fundamentals(
                    producer._financial_pit_events(
                        source, required_sources=spec.source_input_data_ids
                    )
                )(producer._pit_observations(eager["market_panel_1d"]))
                if data_id == "fundamental_pit"
                else None,
            )
            .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
            .reset_index(drop=True)
        )
    prepare = producer._prepare_pit_fundamentals
    preparations = []
    observation_counts = []

    def observed_prepare(events):
        preparations.append(True)
        calculate = prepare(events)

        def observed_query(observations):
            observation_counts.append(len(observations))
            return calculate(observations)

        return observed_query

    monkeypatch.setattr(producer, "_prepare_pit_fundamentals", observed_prepare)
    default_batch_rows = producer.DEFAULT_BATCH_ROWS
    for data_id in target_ids:
        spec = processed_catalog().get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        monkeypatch.setattr(
            producer,
            "DEFAULT_BATCH_ROWS",
            200 if data_id == "fundamental_pit" else default_batch_rows,
        )
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=window,
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )
        actual = read_dataset(root)
        pd.testing.assert_frame_equal(actual, eager[data_id], check_exact=True)
        chunks = list(
            iter_complete_session_frames(
                read_dataset(root, mode=DatasetReadMode.CHUNKED),
                timestamp_column="timestamp",
            )
        )
        pd.testing.assert_frame_equal(
            pd.concat(chunks, ignore_index=True), eager[data_id], check_exact=True
        )
    assert preparations == [True]
    assert len(observation_counts) > 1
    assert sum(observation_counts) == len(eager["market_panel_1d"])
    mask = eager["execution_mask"]
    assert mask.loc[
        mask["asset_id"].eq("000001") & mask["timestamp"].eq(dates[63]), "can_enter"
    ].item()
    panel = eager["market_panel_1d"]
    reopened = panel.loc[
        panel["asset_id"].eq("000002") & panel["timestamp"].eq(dates[191])
    ]
    assert reopened["period_return"].item() == pytest.approx(0.01)


@pytest.mark.parametrize(
    "case",
    [
        "empty_window",
        "no_lifecycle",
        "holiday_fill",
        "closed_calendar",
        "late_lifecycle",
        "invalid_market_type",
    ],
)
def test_processed_window_boundaries_preserve_publish_state(tmp_path, case):
    dates = pd.bdate_range("2020-01-02", periods=130 if case == "late_lifecycle" else 4)
    frames = _synthetic_interim_frames(dates)
    if case == "empty_window":
        # The empty future window is proved by a terminated listing lifecycle.
        frames["stock_master"]["delisting_date"] = dates[-1]
    elif case == "no_lifecycle":
        frames["stock_master"]["listing_date"] = pd.Timestamp("2030-01-01")
    elif case == "late_lifecycle":
        frames["stock_master"]["listing_date"] = dates[70]
    elif case == "closed_calendar":
        frames["trading_calendar"]["is_trading_day"] = False
    elif case == "invalid_market_type":
        frames["market_bars_1d"]["market_type"] = 4.5
    elif case == "holiday_fill":
        bars = frames["market_bars_1d"]
        bars.loc[bars["timestamp"].eq(dates[-1]), "filling_flag"] = 2
    from tests.helpers.a_share_sources import _prepare, _write_inputs

    inputs = _write_inputs(tmp_path / "source", frames)
    start = pd.Timestamp("2030-01-01") if case == "empty_window" else dates[0]
    end = pd.Timestamp("2030-01-04") if case == "empty_window" else dates[-1]
    errors = {
        "holiday_fill": "holiday_fill_conflicts_with_open_session",
        "closed_calendar": "market_bar_session_missing_from_calendar",
        "invalid_market_type": "invalid_market_type",
    }
    output = tmp_path / "prepared" / "market_panel_1d"
    if case == "late_lifecycle":
        saved = read_dataset(
            _prepare(tmp_path / "prepared", "market_panel_1d", inputs, start, end)
        )
        assert len(saved) == 3 * (130 - 70)
        assert saved.timestamp.min() == dates[70]
    else:
        with pytest.raises(
            ValueError, match=errors.get(case, "processed_output_empty")
        ):
            _prepare(tmp_path / "prepared", "market_panel_1d", inputs, start, end)
        assert not output.exists()


@pytest.mark.parametrize("missing", (0, 2, 4, "outside_request"))
def test_panel_calendar_rejects_missing_requested_trade_dates_including_endpoints(
    tmp_path, missing
):
    dates = pd.bdate_range(
        "2020-01-02", periods=7 if missing == "outside_request" else 5
    ).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    if missing == "outside_request":
        requested = dates[1:-1]
        source["trading_calendar"] = (
            source["trading_calendar"]
            .loc[
                source["trading_calendar"].calendar_date.between(
                    requested[0], requested[-1]
                )
            ]
            .copy()
        )
    else:
        requested = dates
        source["trading_calendar"] = (
            source["trading_calendar"]
            .loc[source["trading_calendar"].calendar_date.ne(dates[missing])]
            .copy()
        )
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )

    def publish():
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=TimeWindow(requested[0], requested[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )

    if missing == "outside_request":
        publish()
        actual = read_dataset(root)
        assert len(actual) == 3 * len(requested)
        assert actual.timestamp.isin(requested).all()
    else:
        with pytest.raises(
            ValueError, match="market_bar_session_missing_from_calendar"
        ):
            publish()
        assert not root.exists()


def test_unknown_quote_breaks_suspension_baseline_across_persisted_slabs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 192)
    dates = pd.bdate_range("2020-01-02", periods=400).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    asset = bars.source_asset_id.eq("000002")
    prices = ["open", "high", "low", "close", "adj_open", "adj_close"]
    bars.loc[asset & bars.timestamp.eq(dates[61]), prices] = 100.0
    suspended = asset & bars.timestamp.between(dates[63], dates[-2])
    bars.loc[suspended, prices] = np.nan
    bars.loc[suspended, "filling_flag"] = 1
    bars.loc[asset & bars.timestamp.eq(dates[-1]), prices] = 120.0
    bars.loc[asset & bars.timestamp.eq(dates[-1]), "period_return"] = 0.1
    source["market_bars_1d"] = bars.loc[~(asset & bars.timestamp.eq(dates[62]))].copy()
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    decoded = {
        data_id: read_dataset(
            layout.interim_data_root(_source_catalog().get(data_id).source_id, data_id)
        )
        for data_id in spec.source_input_data_ids
    }
    compute = producer._compute_processed_frame
    raw_caps_before = decoded["market_cap"].copy(deep=True)
    prepared = producer._prepare_panel_source(
        decoded, producer.panels._canonical_stock_master(decoded["stock_master"])
    )
    caps_before = prepared["market_cap"].copy(deep=True)
    expected = compute(
        spec,
        source={
            **prepared,
            "market_bars_1d": decoded["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    slabs = []
    prepared_caps = []

    def observed(spec, **kwargs):
        slabs.append(tuple(kwargs["panel_sessions"]))
        prepared_caps.append(kwargs["source"]["market_cap"])
        return compute(spec, **kwargs)

    monkeypatch.setattr(producer, "_compute_processed_frame", observed)
    calls = {}
    for name in (
        "_canonical_stock_master",
        "_canonical_market_cap",
        "_prepare_industry_inputs",
    ):
        original = getattr(producer.panels, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] = calls.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(producer.panels, name, counted)
    stocks_before = decoded["stock_master"].copy(deep=True)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    assert calls == {
        "_canonical_stock_master": 1,
        "_canonical_market_cap": 1,
        "_prepare_industry_inputs": 1,
    }
    pd.testing.assert_frame_equal(
        decoded["stock_master"], stocks_before, check_exact=True
    )
    actual = read_dataset(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert len(slabs) > 1
    pd.testing.assert_frame_equal(
        decoded["market_cap"], raw_caps_before, check_exact=True
    )
    for caps in prepared_caps:
        pd.testing.assert_frame_equal(caps, caps_before, check_exact=True)
    assert any(dates[62] < slab[0] < dates[-1] for slab in slabs[1:])
    rows = actual.loc[actual.asset_id.eq("000002")].set_index("timestamp")
    assert (
        rows.loc[
            dates[62],
            ["close", "adj_close", "period_return", "volume", "amount", "turnover"],
        ]
        .isna()
        .all()
    )
    assert rows.loc[dates[63] : dates[-2], ["close", "adj_close"]].isna().all().all()
    assert (
        rows.loc[dates[63], ["period_return", "volume", "amount", "turnover"]]
        .eq(0.0)
        .all()
    )
    assert rows.loc[dates[-1], "period_return"] == 0.1
    assert rows.loc[dates[-1], "adj_close"] == 120.0


@pytest.mark.parametrize("target", ("price_volume_facts", "market_model_factors"))
def test_processed_narrow_output_uses_existing_panel_history(
    tmp_path, monkeypatch, target
):
    # Proven listing eligibility precedes the market-model and residual windows.
    dates = pd.bdate_range("2020-01-02", periods=620).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    calendar = pd.date_range(dates[0] - pd.offsets.BDay(300), dates[-1]).as_unit("ns")
    source["trading_calendar"] = pd.DataFrame(
        dict(
            timestamp=calendar,
            calendar_date=calendar,
            source_market_type=1,
            is_trading_day=calendar.dayofweek < 5,
            source_day_of_week=calendar.dayofweek,
            source_state=np.where(calendar.dayofweek < 5, "O", "C"),
        )
    )
    bars = source["market_bars_1d"]
    bars["period_return"] = np.sin(np.arange(len(bars)) / 3) * 0.01
    panel_spec = processed_catalog().get("market_panel_1d")
    spec = processed_catalog().get(target)
    panel = producer._compute_processed_frame(
        panel_spec,
        source={
            **producer._prepare_panel_source(
                source, producer.panels._canonical_stock_master(source["stock_master"])
            ),
            "market_bars_1d": source["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    expected = (
        pd.concat(
            producer.factor_engine._market_model_frames(
                (panel,),
                trading_calendar=source["trading_calendar"],
                market_history=None,
            ),
            ignore_index=True,
        )
        if target == "market_model_factors"
        else producer._compute_processed_frame(
            spec,
            source={"trading_calendar": source["trading_calendar"]},
            dependencies={"market_panel_1d": panel},
            trading_sessions=tuple(dates),
        )
    )
    core = dates[-20:]
    expected = (
        expected.loc[expected.timestamp.isin(core)]
        .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    layout = StorageLayout(tmp_path)
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        panel,
        spec=panel_spec,
    )
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    input_sessions = []
    compute = producer._compute_processed_frame

    def observed(spec, **kwargs):
        input_sessions.append(
            kwargs["dependencies"]["market_panel_1d"].timestamp.nunique()
        )
        return compute(spec, **kwargs)

    if target == "market_model_factors":
        market_model_frames = producer.factor_engine._market_model_frames

        def observed_market_models(panels, **kwargs):
            def observed_panels():
                for panel in panels:
                    input_sessions.append(panel.timestamp.nunique())
                    yield panel

            yield from market_model_frames(observed_panels(), **kwargs)

        monkeypatch.setattr(
            producer.factor_engine, "_market_model_frames", observed_market_models
        )
        monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 300)
    else:
        monkeypatch.setattr(producer, "_compute_processed_frame", observed)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(core[0], core[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.timestamp.min() == core[0]
    assert actual.timestamp.max() == core[-1]
    columns = [
        definition.output_column
        for definition in factor_catalog().list()
        if definition.output_column in spec.required_columns
    ]
    assert actual.loc[actual.timestamp.eq(core[0]), columns].notna().any().any()
    lookback = producer.factor_engine.factor_lookback_bars(
        tuple(
            definition
            for definition in factor_catalog().list()
            if definition.output_column in spec.required_columns
        )
    )
    assert input_sessions and max(input_sessions) <= (
        100 if target == "market_model_factors" else lookback + len(core)
    )
    assert max(input_sessions) < len(dates)


def test_financial_pit_calendar_file_order_preserves_formal_published_rows(tmp_path):
    from hashlib import sha256

    from quant_research_platform.sources.choice import interim as choice_interim
    from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
    from quant_research_platform.sources.choice.raw_exports import choice_session
    from quant_research_platform.sources.csmar import interim as csmar_interim
    from quant_research_platform.storage.identity import (
        filesystem_mkdir,
        filesystem_walk,
        filesystem_write_text,
    )
    from tests.helpers.choice_raw import SDK, response, write_css

    dates = pd.DatetimeIndex(["2019-01-30", "2019-01-31", "2019-02-01"])
    codes = ("000001.SZ", "000002.SZ", "000003.SZ")
    fields = tuple(field for columns in FINANCIAL_FIELDS.values() for field in columns)
    cases = {
        "ordered": ((0, 1), (2,)),
        "single": ((2, 0, 1),),
        "reversed": ((2,), (0, 1)),
    }
    calendars, panels, outputs = ([], [], [])
    calendar_hashes, panel_hashes = ([], [])
    for name, groups in cases.items():
        layout = StorageLayout(tmp_path / name)
        raw = layout.raw_source_root("csmar")
        filesystem_mkdir(raw, parents=True)
        originals = {}
        for number, positions in enumerate(groups, start=1):
            path = raw / f"TRDNEW_Cale{number}.csv"
            filesystem_write_text(
                path,
                "Markettype,Clddt,Daywk,State\n"
                + "".join(
                    f"4,{dates[index].date()},{index + 3},O\n" for index in positions
                ),
                encoding="utf-8",
            )
            originals[path] = Path(native_filesystem_path(path)).read_bytes()
        imported = csmar_interim.materialize_interim_data(
            raw_root=layout.raw_source_root("csmar"),
            output_roots={
                "trading_calendar": layout.interim_data_root(
                    "csmar", "trading_calendar"
                )
            },
            data_ids=("trading_calendar",),
        )
        calendar = read_dataset(imported.artifact_roots["trading_calendar"])
        assert len(calendar) == 3
        logical = calendar.sort_values(
            ["calendar_date", "source_market_type"]
        ).reset_index(drop=True)
        calendars.append(logical)
        calendar_hashes.append(
            sha256(
                pd.util.hash_pandas_object(logical, index=False).to_numpy().tobytes()
            ).hexdigest()
        )
        source = _synthetic_interim_frames(dates)
        source["market_bars_1d"]["market_type"] = 4
        panel_spec = processed_catalog().get("market_panel_1d")
        for data_id in panel_spec.source_input_data_ids:
            if data_id == "trading_calendar":
                continue
            source_spec = _source_catalog().get(data_id)
            write_dataset(
                layout.interim_data_root(source_spec.source_id, data_id),
                source[data_id],
                spec=source_spec,
            )
        with choice_session(SDK(response())) as session:
            write_css(
                session,
                layout,
                period="2018-12-31",
                end_date="2019-01-29",
                indicators=(
                    "STMTACTDATE",
                    "ACTUALISSUINGDATE",
                    "ITEMVALUECHANGEDATE",
                    *fields,
                ),
                data={
                    code: ["2019-01-29", None, None, *[100.0] * len(fields)]
                    for code in codes
                },
                branch="1",
            )
        choice_interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={
                spec.data_id: layout.interim_data_root("choice", spec.data_id)
                for spec in choice_interim.interim_catalog().list()
            },
            end_date="2019-02-01",
        )
        for data_id in ("market_panel_1d", "fundamental_pit"):
            spec = processed_catalog().get(data_id)
            root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
            with producer.produce_a_share_processed_data(
                spec,
                inputs=_published_inputs(layout, spec),
                time_window=TimeWindow(dates[0], dates[-1]),
                processed_catalog=processed_catalog(),
                source_catalog=_source_catalog(),
            ) as (frames, metadata):
                write_dataset(
                    root,
                    frames,
                    spec=spec,
                    metadata=metadata,
                )
            saved = read_dataset(root)
            assert tuple(saved.columns) == spec.required_columns
            assert len(saved) == 9 and (
                not saved.duplicated(list(spec.primary_key)).any()
            )
            if data_id == "market_panel_1d":
                panels.append(saved)
                panel_hashes.append(
                    tuple(
                        (
                            (
                                str((parent / file).relative_to(root)),
                                sha256(
                                    Path(
                                        native_filesystem_path(parent / file)
                                    ).read_bytes()
                                ).hexdigest(),
                            )
                            for parent, _directories, files in filesystem_walk(
                                root / "data"
                            )
                            for file in sorted(files)
                            if file.endswith(".parquet")
                        )
                    )
                )
            else:
                assert saved.has_legal_fundamental_pit.all()
                for columns in FINANCIAL_FIELDS.values():
                    for column in columns.values():
                        assert saved[column].eq(100.0).all()
                outputs.append(saved)
        assert all(
            (
                Path(native_filesystem_path(path)).read_bytes() == payload
                for path, payload in originals.items()
            )
        )
    for frames in (calendars, panels, outputs):
        assert len(frames) == 3
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frame, frames[0], check_exact=True)
    assert all(panel_hashes)
    assert len(set(calendar_hashes)) == len(set(panel_hashes)) == 1


@pytest.mark.parametrize("batch_rows", (6, 300))
def test_asynchronous_own_market_label_endpoints_and_mask_batches(
    tmp_path, monkeypatch, batch_rows
):
    dates = pd.bdate_range("2020-01-02", periods=9).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    own = bars.source_asset_id.eq("000001")
    bars.loc[own, "market_type"] = 4
    bars.loc[own & bars.timestamp.eq(dates[2]), "filling_flag"] = 2
    calendar = source["trading_calendar"]
    second = calendar.assign(source_market_type=4)
    second.loc[second.calendar_date.eq(dates[2]), "is_trading_day"] = False
    source["trading_calendar"] = pd.concat((calendar, second), ignore_index=True)
    layout = StorageLayout(tmp_path)
    panel_spec = processed_catalog().get("market_panel_1d")
    for data_id in panel_spec.source_input_data_ids:
        spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(spec.source_id, data_id),
            source[data_id],
            spec=spec,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = {}
    for data_id in ("market_panel_1d", "execution_mask", "forward_returns"):
        spec = processed_catalog().get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )
        saved[data_id] = read_dataset(root)
    panel = (
        saved["market_panel_1d"].query("asset_id == '000001'").set_index("timestamp")
    )
    assert dates[2] not in panel.index
    mask = saved["execution_mask"].query("asset_id == '000001'").set_index("timestamp")
    execution_at = panel.index[panel.index > dates[1]][0]
    actual_mask = portfolio_production._mask_values(
        mask.loc[[dates[1]]].reset_index().assign(execution_at=execution_at),
        pd.Index(["000001"]),
        execution_at,
        frozenset(),
    )
    assert actual_mask["can_enter"].all() and actual_mask["can_exit"].all()
    label = (
        saved["forward_returns"].query("asset_id == '000001'").set_index("timestamp")
    )
    for horizon in (1, 5):
        column = f"forward_return_{horizon}d"
        exit_date = panel.index[1 + horizon]
        assert (
            label.loc[dates[0], column]
            == (panel.loc[exit_date, "adj_open"] - panel.iloc[1].adj_open)
            / panel.iloc[1].adj_open
        )
        assert label.loc[dates[0], column + "_available_at"] == (
            pd.Timestamp(f"{exit_date.date()} 15:00", tz="Asia/Shanghai")
            .tz_convert("UTC")
            .tz_localize(None)
        )
    assert mask.iloc[-1][["can_enter", "can_exit"]].isna().all()


@pytest.mark.parametrize("missing_entry", (False, True))
def test_own_market_all_label_horizons_preserve_unknown_and_suspended_sessions(
    tmp_path, missing_entry
):
    dates = pd.bdate_range("2020-01-02", periods=135).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    own = bars.source_asset_id.eq("000001")
    bars.loc[own & bars.timestamp.lt(dates[65]), "market_type"] = 4
    bars.loc[own & bars.timestamp.eq(dates[2]), "filling_flag"] = 2
    if missing_entry:
        source["market_bars_1d"] = bars.loc[~(own & bars.timestamp.eq(dates[1]))].copy()
    else:
        bars.loc[own & bars.timestamp.eq(dates[3]), "filling_flag"] = 1
    calendar = source["trading_calendar"]
    second = calendar.assign(source_market_type=4)
    second.loc[second.calendar_date.isin([dates[2], dates[70]]), "is_trading_day"] = (
        False
    )
    source["trading_calendar"] = pd.concat((calendar, second), ignore_index=True)
    spec = processed_catalog().get("market_panel_1d")
    panel = producer._compute_processed_frame(
        spec,
        source={
            **producer._prepare_panel_source(
                source, producer.panels._canonical_stock_master(source["stock_master"])
            ),
            "market_bars_1d": source["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    axes = {
        asset: tuple(group.timestamp.sort_values())
        for asset, group in panel.groupby("asset_id")
    }
    assert dates[2] not in axes["000001"] and dates[70] in axes["000001"]
    assert dates[1] in axes["000001"] and dates[3] in axes["000001"]
    spec = processed_catalog().get("forward_returns")
    output = producer._compute_processed_frame(
        spec,
        source={},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(dates),
        asset_sessions=axes,
    )
    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, output, spec=spec)
    saved = read_dataset(root).query("asset_id == '000001'").set_index("timestamp")
    own_panel = panel.query("asset_id == '000001'").set_index("timestamp")
    for horizon in (1, 5, 10, 20, 60, 120):
        column = f"forward_return_{horizon}d"
        if missing_entry or horizon == 1:
            assert pd.isna(saved.loc[dates[0], column])
            assert pd.isna(saved.loc[dates[0], column + "_available_at"])
        else:
            exit_date = axes["000001"][horizon + 1]
            assert (
                saved.loc[dates[0], column]
                == (
                    own_panel.loc[exit_date, "adj_open"]
                    - own_panel.loc[dates[1], "adj_open"]
                )
                / own_panel.loc[dates[1], "adj_open"]
            )
            assert saved.loc[dates[0], column + "_available_at"] == (
                pd.Timestamp(f"{exit_date.date()} 15:00", tz="Asia/Shanghai")
                .tz_convert("UTC")
                .tz_localize(None)
            )


@pytest.mark.parametrize(
    "gap,batch_rows,old_market",
    (
        ("membership", 6, None),
        ("membership", 6, 1),
        ("membership", 6, 4),
        ("calendar", 6, None),
        ("membership", 300, None),
        ("calendar", 300, None),
    ),
)
def test_unobserved_initial_market_preserves_unknown_age_and_proven_252_sessions(
    tmp_path, monkeypatch, batch_rows, gap, old_market
):
    dates = pd.bdate_range("2019-01-02", periods=260).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    source["stock_master"]["listing_date"] = dates[0]
    source["stock_master"]["timestamp"] = dates[0]
    if gap == "membership":
        source["market_bars_1d"] = (
            source["market_bars_1d"]
            .loc[source["market_bars_1d"].timestamp.ge(dates[4])]
            .copy()
        )
    else:
        source["trading_calendar"] = (
            source["trading_calendar"]
            .loc[source["trading_calendar"].calendar_date.ge(dates[4])]
            .copy()
        )
    if old_market is not None:
        from quant_research_platform.sources.csmar.interim import (
            _transform_interim_chunk,
        )

        old_date = dates[0] - pd.Timedelta(days=10)
        assets = source["stock_master"].source_asset_id.unique()
        masters = pd.DataFrame(
            [
                {
                    "Stkcd": asset,
                    "Stknme": "synthetic",
                    "Listdt": listing,
                    "DelistedDate": end,
                    "Markettype": market,
                    "Nnindcd": "C",
                    "Nnindnme": "synthetic",
                    "Statco": "normal",
                    "Statdt": pd.NaT,
                }
                for asset in assets
                for listing, end, market in (
                    (old_date, dates[0] - pd.Timedelta(days=1), old_market),
                    (dates[0], pd.NaT, 1),
                )
            ]
        )
        source["stock_master"], _ = _transform_interim_chunk(
            masters, data_id="stock_master"
        )
        bars = source["market_bars_1d"]
        old = bars.drop_duplicates("source_asset_id").assign(
            timestamp=old_date, market_type=old_market
        )
        source["market_bars_1d"] = pd.concat((old, bars), ignore_index=True)
    spec = processed_catalog().get("market_panel_1d")
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    windows = (
        ((0, dates[:6]), (1, dates[-6:])) if gap == "membership" else ((1, dates[-6:]),)
    )
    for index, requested in windows:
        layout = StorageLayout(tmp_path / str(index))
        for data_id in spec.source_input_data_ids:
            source_spec = _source_catalog().get(data_id)
            write_dataset(
                layout.interim_data_root(source_spec.source_id, data_id),
                source[data_id],
                spec=source_spec,
            )
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=TimeWindow(requested[0], requested[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )
        actual = read_dataset(root)
        assert actual.listing_age_trading_days.isna().all()
        assert str(actual.listing_age_trading_days.dtype) == "Int64"
        assert str(actual.has_252_trading_sessions.dtype) == "boolean"
        if index == 0:
            unknown = actual.timestamp.lt(dates[4])
            assert actual.loc[unknown, "market_type"].isna().all()
            assert actual.has_252_trading_sessions.isna().all()
            assert not a_share_eligibility(actual, "a_share_all").any()
        elif gap == "calendar":
            assert actual.has_252_trading_sessions.all()
            assert a_share_eligibility(actual, "a_share_all").all()
        else:
            assert actual.has_252_trading_sessions.iloc[:3].isna().all()
            assert actual.has_252_trading_sessions.iloc[3:].all()
            assert (
                a_share_eligibility(actual, "a_share_all").tolist()
                == [False] * 3 + [True] * 15
            )


@pytest.mark.parametrize("batch_rows", (6, 3000))
@pytest.mark.parametrize("unknown_quote", (False, True))
def test_factor_warmup_uses_own_sessions_across_closure_and_migration(
    tmp_path, monkeypatch, batch_rows, unknown_quote
):
    dates = pd.bdate_range("2020-01-02", periods=300).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    own = bars.source_asset_id.eq("000001")
    bars.loc[own & bars.timestamp.lt(dates[200]), "market_type"] = 4
    closed = dates[60:90]
    bars.loc[own & bars.timestamp.isin(closed), "filling_flag"] = 2
    bars.loc[own & bars.timestamp.eq(dates[95]), "filling_flag"] = 1
    if unknown_quote:
        source["market_bars_1d"] = bars.loc[
            ~(own & bars.timestamp.eq(dates[245]))
        ].copy()
    second = source["trading_calendar"].assign(source_market_type=4)
    second.loc[second.calendar_date.isin(closed), "is_trading_day"] = False
    source["trading_calendar"] = pd.concat(
        (source["trading_calendar"], second), ignore_index=True
    )
    panel_spec = processed_catalog().get("market_panel_1d")
    spec = processed_catalog().get("price_volume_facts")
    panel = producer._compute_processed_frame(
        panel_spec,
        source={
            **producer._prepare_panel_source(
                source, producer.panels._canonical_stock_master(source["stock_master"])
            ),
            "market_bars_1d": source["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    own_panel = panel.loc[panel.asset_id.eq("000001")].set_index("timestamp")
    assert not own_panel.index.isin(closed).any()
    assert dates[95] in own_panel.index
    assert dates[245] in own_panel.index
    if unknown_quote:
        assert pd.isna(own_panel.loc[dates[245], "amount"])
    expected = producer._compute_processed_frame(
        spec,
        source={},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(dates),
    )
    core = dates[-10:]
    expected = (
        expected.loc[expected.timestamp.isin(core)]
        .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    layout = StorageLayout(tmp_path)
    calendar_spec = _source_catalog().get("trading_calendar")
    write_dataset(
        layout.interim_data_root(calendar_spec.source_id, "trading_calendar"),
        source["trading_calendar"],
        spec=calendar_spec,
    )
    write_dataset(
        layout.processed_data_root(
            panel_spec.market_id, panel_spec.kind, panel_spec.data_id
        ),
        panel,
        spec=panel_spec,
    )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(core[0], core[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.momentum_252_20.notna().all()
    row = actual.loc[actual.asset_id.eq("000001")].iloc[0]
    position = own_panel.index.get_loc(core[0])
    assert row.momentum_252_20 == (
        own_panel.iloc[position - 20].adj_close
        / own_panel.iloc[position - 252].adj_close
        - 1
    )


@pytest.mark.parametrize("batch_rows", (6, 3000))
def test_historical_calendar_gap_keeps_exact_age_unknown_across_slabs(
    tmp_path, monkeypatch, batch_rows
):
    dates = pd.bdate_range("2020-01-02", periods=260).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    source["stock_master"]["listing_date"] = [dates[0], dates[0], dates[200]]
    source["stock_master"]["timestamp"] = source["stock_master"]["listing_date"]
    source["trading_calendar"] = (
        source["trading_calendar"]
        .loc[source["trading_calendar"].calendar_date.ne(dates[100])]
        .copy()
    )
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[250], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(
            root,
            frames,
            spec=spec,
            metadata=metadata,
        )
    actual = read_dataset(root)
    old = actual.loc[actual.asset_id.ne("000003")]
    assert old.listing_age_trading_days.isna().all()
    assert (
        old.loc[old.timestamp.eq(dates[250]), "has_252_trading_sessions"].isna().all()
    )
    assert old.loc[old.timestamp.gt(dates[250]), "has_252_trading_sessions"].all()
    recent = actual.loc[actual.asset_id.eq("000003")]
    assert recent.listing_age_trading_days.tolist() == list(range(51, 61))
    assert not recent.has_252_trading_sessions.any()


@pytest.mark.parametrize(
    "missing",
    ("stock_master", "market_cap", "trading_no_limit", "special_treatment_changes"),
)
def test_missing_mandatory_panel_source_fails_before_materialization(
    tmp_path, monkeypatch, missing
):
    layout = StorageLayout(tmp_path)
    spec = processed_catalog().get("market_panel_1d")
    inputs = _published_inputs(layout, spec)
    del inputs[missing]

    def forbidden(*args, **kwargs):
        raise AssertionError("missing source reached materialization")

    monkeypatch.setattr(producer, "materialize_dataset_session", forbidden)
    from quant_research_platform.markets.prepare import prepare

    with pytest.raises(ValueError, match="data_operation_input_paths_mismatch"):
        prepare(
            {
                "market_id": "a_share",
                "data_id": "market_panel_1d",
                "inputs": {key: str(path) for key, path in inputs.items()},
                "time_window": {"start": "2020-01-02", "end": "2020-01-03"},
                "output_root": str(tmp_path / "output"),
            }
        )


@pytest.mark.parametrize("prior", (None, 2.0, 1e308))
def test_fundamental_adapter_reuses_prior_year_for_beginning_balances(
    prior, monkeypatch
):
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2020-08-03"),
        }
    )
    pit = pd.DataFrame(
        [
            {
                "asset_id": asset,
                "timestamp": pd.Timestamp("2020-08-03"),
                "metric_id": metric,
                "value": 1e308,
                "prior_year_value": prior,
                "beginning_value": prior,
            }
            for asset in ("a", "b")
            for metric in ("total_assets", "parent_equity")
        ]
    )
    pivots = []
    original = pd.DataFrame.pivot_table

    def pivot(frame, *args, **kwargs):
        values = kwargs.get("values")
        pivots.append(len(frame) * (len(values) if isinstance(values, list) else 1))
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "pivot_table", pivot)
    result = producer._fundamental_pit_frame(
        processed_catalog().get("fundamental_pit"), panel=panel, pit=pit
    )
    assert sum(pivots) <= 2 * len(pit)
    for metric in ("total_assets", "parent_equity"):
        pd.testing.assert_series_equal(
            result[f"beginning_{metric}"],
            result[f"prior_year_{metric}"],
            check_names=False,
            check_exact=True,
        )
        if prior is None:
            assert result[f"average_{metric}"].isna().all()
        elif prior == 1e308:
            assert result[f"average_{metric}"].eq(1e308).all()
        else:
            assert result[f"average_{metric}"].eq(5e307).all()
        expected_prior = np.nan if prior is None else prior
        np.testing.assert_array_equal(
            result[f"beginning_{metric}"], [expected_prior] * 2
        )


@pytest.mark.parametrize("batch_rows", (6, 3000))
@pytest.mark.parametrize("second_only", (False, True))
@pytest.mark.parametrize("relisted", (False, True))
def test_panel_preserves_proven_listing_lifecycles(
    tmp_path, monkeypatch, batch_rows, second_only, relisted
):
    dates = pd.bdate_range("2020-01-02", periods=8).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    from quant_research_platform.sources.csmar.interim import _transform_interim_chunk

    raw_master = pd.DataFrame(
        {
            "Stkcd": ["000001"] * (2 if relisted else 1),
            "Stknme": "synthetic",
            "Listdt": [dates[0], dates[5]] if relisted else [dates[0]],
            "DelistedDate": [dates[2], pd.NaT] if relisted else [pd.NaT],
            "Markettype": 1,
            "Nnindcd": "C",
            "Nnindnme": "synthetic",
            "Statco": "normal",
            "Statdt": pd.NaT,
        }
    )
    normalized, _ = _transform_interim_chunk(
        raw_master,
        data_id="stock_master",
    )
    stock = normalized.iloc[:1]
    revision = stock.assign(timestamp=dates[1], stock_name="revised")
    if relisted:
        normalized.loc[normalized.listing_date.eq(dates[0]), "delisting_date"] = pd.NaT

    source["stock_master"] = pd.concat([normalized, revision], ignore_index=True).iloc[
        ::-1
    ]
    bars = source["market_bars_1d"]
    bars = bars.loc[bars.source_asset_id.eq(stock.source_asset_id.iloc[0])].copy()
    if relisted:
        bars = bars.loc[~bars.timestamp.isin(dates[3:5])].copy()
    bars.loc[bars.timestamp.eq(dates[5]), "filling_flag"] = 1
    source["market_bars_1d"] = bars
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[5] if second_only else dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(
            root,
            frames,
            spec=spec,
            metadata=metadata,
        )
    actual = read_dataset(root)
    expected = (
        dates[5:]
        if second_only
        else (dates[:3].append(dates[5:]) if relisted else dates)
    )
    assert actual.timestamp.tolist() == expected.tolist()
    assert actual.stock_name.tolist() == [
        "synthetic" if day == dates[0] or (relisted and day >= dates[5]) else "revised"
        for day in expected
    ]
    assert actual.listing_age_trading_days.tolist() == [
        dates.get_loc(day) - (5 if relisted and day >= dates[5] else 0) + 1
        for day in expected
    ]
    suspended = actual.loc[actual.timestamp.eq(dates[5])]
    assert (
        suspended[["close", "adj_close", "open", "adj_open"]].isna().all().all()
        == relisted
    )


@pytest.mark.parametrize("batch_rows", (6, 3000))
@pytest.mark.parametrize("gap,suspended", ((1, False), (25, True)))
def test_native_lifecycle_history_does_not_join_relisting(
    tmp_path, monkeypatch, batch_rows, gap, suspended
):
    from quant_research_platform.sources.csmar import interim
    from tests.helpers.a_share_raw import _write_csmar

    dates = pd.bdate_range("2020-01-02", periods=330 + gap).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    masters = [
        dict(
            Stkcd="000001",
            Stknme="synthetic",
            Listdt=dates[0].date().isoformat(),
            DelistedDate=dates[299].date().isoformat(),
            Markettype=1,
            Nnindcd="C",
            Nnindnme="synthetic",
            Statco="normal",
            Statdt="",
        ),
        dict(
            Stkcd="000001",
            Stknme="synthetic",
            Listdt=dates[300 + gap].date().isoformat(),
            DelistedDate="",
            Markettype=1,
            Nnindcd="C",
            Nnindnme="synthetic",
            Statco="normal",
            Statdt="",
        ),
    ]
    _write_csmar(raw, "stock_master", masters[::-1])
    rows = []
    for i, stamp in enumerate(dates):
        if 300 <= i < 300 + gap:
            continue
        price = 10.0 if i < 300 else 100.0
        rows.append(
            dict(
                Stkcd="000001",
                Trddt=stamp.date().isoformat(),
                Markettype=1,
                Trdsta=0,
                Filling=int(suspended and i == 300 + gap),
                Opnprc=price,
                Hiprc=price,
                Loprc=price,
                Clsprc=price,
                Dnshrtrd=1.0,
                Dnvaltrd=10.0,
                Ahshrtrd_D=0.0,
                Ahvaltrd_D=0.0,
                Dsmvosd=1000.0,
                Dsmvtl3=1000.0,
                Dretwd=0.0,
                Adjprcwd=price,
                PreClosePrice=price,
                LimitStatus=0,
            )
        )
    _write_csmar(raw, "market_bars_1d", rows[::-1])
    _write_csmar(
        raw,
        "trading_calendar",
        [
            dict(
                Markettype=market,
                Clddt=stamp.date().isoformat(),
                Daywk=stamp.weekday() + 1,
                State="O" if stamp in dates else "C",
            )
            for market in (1, 4, 16, 32)
            for stamp in pd.date_range(dates[0], dates[-1])
        ],
    )
    imported = {"stock_master", "market_bars_1d", "trading_calendar"}
    interim.materialize_interim_data(
        raw_root=layout.raw_source_root("csmar"),
        output_roots={
            name: layout.interim_data_root("csmar", name) for name in imported
        },
        data_ids=tuple(sorted(imported)),
    )
    panel_spec = processed_catalog().get("market_panel_1d")
    for data_id in panel_spec.source_input_data_ids:
        if data_id not in imported:
            spec = _source_catalog().get(data_id)
            write_dataset(
                layout.interim_data_root(spec.source_id, data_id),
                source[data_id],
                spec=spec,
            )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    results = {}
    for data_id in (
        "market_panel_1d",
        "execution_mask",
        "forward_returns",
        "price_volume_facts",
    ):
        spec = processed_catalog().get(data_id)
        if data_id == "price_volume_facts":
            spec = replace(
                spec,
                column_roles={
                    k: v
                    for k, v in spec.column_roles.items()
                    if k in ("market_id", "asset_id", "timestamp", "reversal_1d")
                },
            )
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )
        results[data_id] = read_dataset(root).set_index("timestamp")
    panel = results["market_panel_1d"]
    assert panel.loc[dates[289], "listing_age_trading_days"] == 290
    assert a_share_eligibility(panel.reset_index(), "a_share_all").iloc[289]
    mask = results["execution_mask"]
    assert pd.isna(mask.loc[dates[299], "can_enter"])
    assert mask.loc[dates[300 + gap], "can_enter"]
    labels = results["forward_returns"]
    assert labels.loc[dates[270], "forward_return_20d"] == 0.0
    for horizon in (1, 20):
        edge = dates[299 - horizon]
        assert pd.isna(labels.loc[edge, f"forward_return_{horizon}d"])
        assert pd.isna(labels.loc[edge, f"forward_return_{horizon}d_available_at"])
    assert pd.isna(labels.loc[dates[289], "forward_return_20d"])
    factors = results["price_volume_facts"]
    assert pd.isna(factors.loc[dates[300 + gap], "reversal_1d"])
    assert factors.loc[dates[302 + gap], "reversal_1d"] == 0.0
    from quant_research_platform.research.portfolios.production import (
        _completed_calibration_rows,
    )

    entry = (
        panel.loc[[dates[289]]].reset_index().set_index(["asset_id", "listing_date"])
    )
    alpha = pd.Series(1.0, index=entry.index)
    pending = (alpha, entry.adj_open, dates[288])
    assert (
        _completed_calibration_rows(
            pending, panel.loc[[dates[302 + gap]]].reset_index()
        )
        == []
    )
    same_life = _completed_calibration_rows(
        pending, panel.loc[[dates[295]]].reset_index()
    )
    assert same_life == [
        {"timestamp": dates[288], "prediction_rank_score": 1.0, "holding_return": 0.0}
    ]


@pytest.mark.parametrize(
    "cap,close,volume",
    (
        (1e308, 1e-308, 1e308),
        (1e-308, 1e308, 1e-308),
        (3 * np.nextafter(0.0, 1.0), 2.0, 3 * np.nextafter(0.0, 1.0)),
        (3.0, 2.0, 3.0),
    ),
)
def test_native_panel_persists_finite_turnover_at_share_count_boundaries(
    tmp_path, cap, close, volume
):
    from fractions import Fraction

    dates = pd.bdate_range("2020-01-06", periods=3).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    bars[["close", "open", "high", "low"]] = close
    bars[["float_market_cap", "total_market_cap"]] = cap
    bars["volume"] = volume
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        source_spec = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    expected = float(Fraction(volume) * Fraction(close) / Fraction(cap))
    assert np.isfinite(actual.turnover).all()
    np.testing.assert_allclose(actual.turnover, expected, rtol=1e-15, atol=0)


def test_processed_axis_keeps_raw_dates_and_assets_before_membership_filter(tmp_path):
    dates = pd.bdate_range("2020-01-06", periods=3).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    extra_date = dates[-1] + pd.offsets.BDay()
    extra = source["market_bars_1d"].iloc[:2].copy()
    extra["timestamp"] = extra_date
    extra.iloc[0, extra.columns.get_loc("source_asset_id")] = "999999"
    extra.iloc[0, extra.columns.get_loc("market_type")] = 9
    extra.iloc[1, extra.columns.get_loc("filling_flag")] = 2
    bars = pd.concat((source["market_bars_1d"], extra), ignore_index=True)
    closed = (
        source["trading_calendar"]
        .iloc[-1:]
        .assign(calendar_date=extra_date, timestamp=extra_date, is_trading_day=False)
    )
    calendar = pd.concat((source["trading_calendar"], closed), ignore_index=True)
    spec = _source_catalog().get("market_bars_1d")
    root = StorageLayout(tmp_path).interim_data_root(spec.source_id, spec.data_id)
    write_dataset(root, bars, spec=spec)
    session = open_dataset_session(root, columns=None, time_window=None)
    stocks = producer.panels._canonical_stock_master(source["stock_master"])
    panel_axis = producer._processed_axis(
        session, panel_source=True, calendar=calendar, stock_master=stocks
    )
    raw = producer._processed_axis(
        session, panel_source=False, asset_column="source_asset_id"
    )
    assert panel_axis[5] == (raw[0], frozenset(raw[1].index))
    assert extra_date in panel_axis[5][0] and extra_date not in panel_axis[0]
    assert "999999" in panel_axis[5][1] and "999999" not in set(panel_axis[2].asset_id)


@pytest.mark.parametrize("calendar_state", ("unknown", "closed", "open"))
def test_native_panel_does_not_treat_unknown_calendar_as_closed(
    tmp_path, calendar_state
):
    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    bars.loc[bars.source_asset_id.eq("000001"), "market_type"] = 4
    source["market_bars_1d"] = bars.loc[
        ~(bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[2]))
    ]
    own = source["trading_calendar"].assign(source_market_type=4)
    if calendar_state == "unknown":
        own = own.loc[own.calendar_date.ne(dates[2])]
    elif calendar_state == "closed":
        own.loc[own.calendar_date.eq(dates[2]), "is_trading_day"] = False
    source["trading_calendar"] = pd.concat(
        (source["trading_calendar"], own), ignore_index=True
    )
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)

    def publish():
        with producer.produce_a_share_processed_data(
            spec,
            inputs=_published_inputs(layout, spec),
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )

    if calendar_state == "unknown":
        with pytest.raises(
            ValueError, match="a_share_trading_calendar_unknown_session"
        ):
            publish()
    else:
        publish()
        actual = read_dataset(root)
        row = actual.loc[actual.asset_id.eq("000001") & actual.timestamp.eq(dates[2])]
        if calendar_state == "closed":
            assert row.empty
        else:
            assert len(row) == 1 and row.close.isna().all()


def test_native_panel_retains_listing_before_first_known_company_status(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    source["stock_master"] = source["stock_master"].assign(
        timestamp=dates[3], status_change_date=dates[3], company_status="suspended"
    )
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        inputs=_published_inputs(layout, spec),
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=processed_catalog(),
        source_catalog=_source_catalog(),
    ) as (frames, metadata):
        write_dataset(root, frames, spec=spec, metadata=metadata)
    actual = read_dataset(root)
    assert len(actual) == 15
    assert actual.listing_date.notna().all()
    joined = producer.panels._asof_join_stock_master(
        actual.loc[:, ["asset_id", "timestamp"]],
        producer.panels._canonical_stock_master(source["stock_master"]),
    )
    assert joined.loc[joined.timestamp.lt(dates[3]), "company_status"].isna().all()
    assert (
        joined.loc[joined.timestamp.ge(dates[3]), "company_status"]
        .eq("suspended")
        .all()
    )


def test_bound_optional_source_cannot_disappear_into_absence(tmp_path):
    dates = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    spec = processed_catalog().get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        native = _source_catalog().get(data_id)
        write_dataset(
            layout.interim_data_root(native.source_id, data_id),
            source[data_id],
            spec=native,
        )
    inputs = _published_inputs(layout, spec)
    inputs["industry_classification"] = tmp_path / "missing-bound-source"
    with pytest.raises((ValueError, FileNotFoundError)):
        with producer.produce_a_share_processed_data(
            spec,
            inputs=inputs,
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=processed_catalog(),
            source_catalog=_source_catalog(),
        ) as (frames, _metadata):
            list(frames)


def test_price_only_panel_and_factors_do_not_require_industry_evidence(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from tests.helpers.a_share_sources import _prepare, _write_inputs

    dates = pd.bdate_range("2020-01-06", periods=30)
    source = _synthetic_interim_frames(dates)
    inputs = _write_inputs(tmp_path / "source", source)
    full = _prepare(tmp_path / "full", "market_panel_1d", inputs, dates[0], dates[-1])
    price_only = {
        key: value
        for key, value in inputs.items()
        if key
        not in {
            "a_share_industry_snapshots",
            "industry_classification",
            "industry_classification_sets",
            "industry_classification_versions",
            "stock_info_changes",
        }
    }
    missing = _prepare(
        tmp_path / "price-only", "market_panel_1d", price_only, dates[0], dates[-1]
    )
    left, right = read_dataset(full), read_dataset(missing)
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "is_nonfinancial_industry",
    ):
        assert right[column].isna().all()
    pd.testing.assert_frame_equal(
        left[["asset_id", "timestamp", "adj_open", "adj_close", "amount"]],
        right[["asset_id", "timestamp", "adj_open", "adj_close", "amount"]],
    )
    factors = []
    for name, panel in (("full", full), ("price-only", missing)):
        factors.append(
            read_dataset(
                prepare(
                    {
                        "market_id": "a_share",
                        "data_id": "price_volume_facts",
                        "inputs": {
                            "market_panel_1d": str(panel),
                            "trading_calendar": str(inputs["trading_calendar"]),
                        },
                        "factor_columns": ["reversal_5d"],
                        "time_window": {
                            "start": str(dates[0].date()),
                            "end": str(dates[-1].date()),
                        },
                        "output_root": str(tmp_path / name / "factors"),
                    }
                )
            )
        )
    pd.testing.assert_frame_equal(*factors)
    # Supplied selected evidence still crosses the native source validation boundary.
    broken = tmp_path / "malformed-industry.parquet"
    broken.write_bytes(b"invalid selected industry evidence")
    with pytest.raises((ValueError, OSError)):
        _prepare(
            tmp_path / "bad",
            "market_panel_1d",
            {**price_only, "a_share_industry_snapshots": broken},
            dates[0],
            dates[-1],
        )
