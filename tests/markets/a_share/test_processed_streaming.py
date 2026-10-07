from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.execution import (
    market_name_for_type,
)
from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.research.filtering import a_share_eligibility
from quant_research_platform.research.portfolios import (
    production as portfolio_production,
)
from quant_research_platform.runtime.dataframes import iter_complete_session_frames
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    open_dataset_session,
    read_dataset,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.markets.a_share.test_target_dag_chain import (
    _synthetic_financial_frames,
    _synthetic_interim_frames,
)


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
    profile = research_profile()
    layout = StorageLayout(tmp_path)
    spec = profile.processed.get("market_panel_1d")
    interim.materialize_interim_data(
        layout=layout,
        raw_root=raw_root / "reference",
        data_ids=(interim.A_SHARE_INDUSTRY_SNAPSHOTS_DATA_ID,),
        start_date="2020-01-06",
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
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(requested[0], requested[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
    actual = read_dataset(root)
    assert actual["timestamp"].tolist() == [
        timestamp for timestamp in requested for _ in range(3)
    ]
    assert actual["listing_age_trading_days"].tolist() == [
        age for offset in range(5) for age in (251 + offset, 251 + offset, 1 + offset)
    ]
    assert a_share_eligibility(actual, "pv_expanded").tolist() == [
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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    window = TimeWindow(dates[0], dates[-1])
    columns = ("market_id", "asset_id", "timestamp", "market_type")
    for data_id in spec.source_input_data_ids:
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=window,
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
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


def test_processed_blocks_preserve_full_history_math_and_release_failed_inputs(
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
    profile = research_profile()
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
        spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(spec.source_id, data_id), frame, spec=spec
        )
    source = {
        data_id: read_dataset(
            layout.interim_data_root(profile.interim.get(data_id).source_id, data_id),
            time_window=None
            if data_id in producer._FULL_HISTORY_SOURCE_IDS
            else producer.a_share.source_history_window(
                profile.interim.get(data_id), window
            ),
        )
        for data_id in source
    }
    eager = {}
    for data_id in target_ids:
        spec = profile.processed.get(data_id)
        eager[data_id] = (
            producer._compute_processed_frame(
                spec,
                source={
                    **producer._prepare_panel_source(source),
                    "market_bars_1d": source["market_bars_1d"],
                }
                if data_id == "market_panel_1d"
                else source,
                dependencies={key: eager[key] for key in spec.processed_input_data_ids},
                trading_sessions=tuple(dates),
                financial_pit=producer._prepare_pit_fundamentals(
                    producer._financial_pit_events(source)
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
        spec = profile.processed.get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        monkeypatch.setattr(
            producer,
            "DEFAULT_BATCH_ROWS",
            200 if data_id == "fundamental_pit" else default_batch_rows,
        )
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=window,
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
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
    if case == "no_lifecycle":
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
    base = research_profile()

    def source_identity(_root, data_id):
        return (data_id.encode().hex() + "0" * 64)[:64]

    def interim_producer(**kwargs):
        (data_id,) = kwargs["data_ids"]
        spec = base.interim.get(data_id)
        root = kwargs["layout"].target_root(
            data_id, kwargs["layout"].interim_data_root(spec.source_id, data_id)
        )
        write_dataset(
            root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
        )
        return InterimImportResult(
            artifact_roots={data_id: root},
            skipped=(),
            pruning_report_text="synthetic boundary",
        )

    profile = replace(
        base,
        producer_bindings={
            **base.producer_bindings,
            **{
                data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    interim_producer,
                    source_identity=source_identity,
                )
                for data_id in frames
            },
        },
    )
    layout = StorageLayout(tmp_path)

    def build(*, plan=False):
        return research.dataset_build(
            profile=profile,
            layout=layout,
            dataset_id="pv_all",
            start_date="2030-01-01" if case == "empty_window" else str(dates[0].date()),
            end_date="2030-01-04" if case == "empty_window" else str(dates[-1].date()),
            raw_root=tmp_path / "synthetic-raw",
            plan=plan,
        )

    errors = {
        "holiday_fill": "holiday_fill_conflicts_with_open_session",
        "closed_calendar": "a_share_trading_calendar_empty",
        "invalid_market_type": "invalid_market_type",
    }
    if case == "empty_window":
        planned = build(plan=True)
        core_identity = next(
            row["requested_identity"]
            for row in planned["targets"]
            if row["target_id"] == "market_panel_1d"
        )
    result = build()
    market = next(
        row for row in result["target_results"] if row["target_id"] == "market_panel_1d"
    )
    if case in errors:
        assert result["status"] == "FAILED"
        assert market["status"] == "FAILED"
        assert errors[case] in market["reason"]
        descendant = next(
            row for row in result["target_results"] if row["target_id"] == "pv_all"
        )
        assert descendant["status"] == "BLOCKED"
    elif case == "late_lifecycle":
        assert market["status"] == "BUILT"
    else:
        assert market["status"] == "BLOCKED"
        assert market["reason"] == "processed_target_empty"
    spec = profile.processed.get("market_panel_1d")
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    if case == "late_lifecycle":
        saved = read_dataset(layout.instance_root(root, market["instance_identity"]))
        assert len(saved) == 3 * (130 - 70)
        assert saved["timestamp"].min() == dates[70]
    elif case == "empty_window":
        assert result["status"] == "BLOCKED"
        assert market["instance_identity"] is None
        assert not layout.instance_root(root, core_identity).exists()
        descendant = next(
            row for row in result["target_results"] if row["target_id"] == "pv_all"
        )
        assert descendant["status"] == "BLOCKED"
        assert descendant["reason"].startswith("dependency_blocked: ")
        assert descendant["instance_identity"] is None
        dataset_root = layout.research_dataset_root(spec.market_id, "pv_all")
        assert not tuple(dataset_root.rglob("manifest.json"))
        (history_manifest,) = root.rglob("manifest.json")
        history_root = history_manifest.parent.parent
        assert history_root.name != core_identity
        session = open_dataset_session(history_root, columns=None, time_window=None)
        manifest = session.manifest
        common = manifest["common_manifest"]
        assert common["instance_identity"] == history_root.name
        assert common["research_window"] == {
            "start": "2011-01-01",
            "end": "2030-01-04",
            "access": "not_applicable",
        }
        assert manifest["row_count"] == 12
        saved = read_dataset(history_root)
        assert len(saved) == 12
        assert tuple(saved["timestamp"].sort_values().unique()) == tuple(dates)
    else:
        assert not tuple(root.rglob("manifest.json"))


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
            .loc[source["trading_calendar"].calendar_date.isin(requested)]
            .copy()
        )
    else:
        requested = dates
        source["trading_calendar"] = (
            source["trading_calendar"]
            .loc[source["trading_calendar"].calendar_date.ne(dates[missing])]
            .copy()
        )
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )

    def publish():
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=TimeWindow(requested[0], requested[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    for data_id in spec.source_input_data_ids:
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    decoded = {
        data_id: read_dataset(
            layout.interim_data_root(profile.interim.get(data_id).source_id, data_id)
        )
        for data_id in spec.source_input_data_ids
    }
    compute = producer._compute_processed_frame
    raw_caps_before = decoded["market_cap"].copy(deep=True)
    prepared = producer._prepare_panel_source(decoded)
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
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(dates[0], dates[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
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
    calendar = pd.bdate_range(dates[0] - pd.offsets.BDay(300), dates[-1]).as_unit("ns")
    source["trading_calendar"] = pd.DataFrame(
        dict(
            timestamp=calendar,
            calendar_date=calendar,
            source_market_type=1,
            is_trading_day=True,
            source_day_of_week=calendar.dayofweek,
            source_state="O",
        )
    )
    bars = source["market_bars_1d"]
    bars["period_return"] = np.sin(np.arange(len(bars)) / 3) * 0.01
    profile = research_profile()
    panel_spec = profile.processed.get("market_panel_1d")
    spec = profile.processed.get(target)
    panel = producer._compute_processed_frame(
        panel_spec,
        source={
            **producer._prepare_panel_source(source),
            "market_bars_1d": source["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    expected = producer._compute_processed_frame(
        spec,
        source={"trading_calendar": source["trading_calendar"]},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(dates),
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
        source_spec = profile.interim.get(data_id)
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
        layout=layout,
        time_window=TimeWindow(core[0], core[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
    actual = read_dataset(root)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.timestamp.min() == core[0]
    assert actual.timestamp.max() == core[-1]
    columns = [
        definition.output_column
        for definition in profile.factors.list()
        if definition.output_column in spec.required_columns
    ]
    assert actual.loc[actual.timestamp.eq(core[0]), columns].notna().any().any()
    lookback = producer.factor_engine.factor_lookback_bars(
        tuple(
            definition
            for definition in profile.factors.list()
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
        filesystem_read_bytes,
        filesystem_walk,
        filesystem_write_text,
    )
    from tests.helpers.choice_raw import SDK, response, write_css

    dates = pd.DatetimeIndex(["2019-01-30", "2019-01-31", "2019-02-01"])
    codes = ("000001.SZ", "000002.SZ", "000003.SZ")
    fields = tuple(field for columns in FINANCIAL_FIELDS.values() for field in columns)
    profile = research_profile()
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
            originals[path] = filesystem_read_bytes(path)
        imported = csmar_interim.materialize_interim_data(
            layout=layout, data_ids=("trading_calendar",)
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
        assert calendar.calendar_date.is_monotonic_increasing == (name != "reversed")
        source = _synthetic_interim_frames(dates)
        source["market_bars_1d"]["market_type"] = 4
        panel_spec = profile.processed.get("market_panel_1d")
        for data_id in panel_spec.source_input_data_ids:
            if data_id == "trading_calendar":
                continue
            source_spec = profile.interim.get(data_id)
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
        choice_interim.materialize_interim_data(layout=layout, end_date="2019-02-01")
        for data_id in ("market_panel_1d", "fundamental_pit"):
            spec = profile.processed.get(data_id)
            root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
            with producer.produce_a_share_processed_data(
                spec,
                layout=layout,
                time_window=TimeWindow(dates[0], dates[-1]),
                processed_catalog=profile.processed,
                source_catalog=profile.interim,
            ) as (frames, metadata, verify):
                write_dataset(
                    root,
                    frames,
                    spec=spec,
                    instance_metadata=metadata,
                    staged_validator=verify,
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
                                    filesystem_read_bytes(parent / file)
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
                filesystem_read_bytes(path) == payload
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
    profile = research_profile()
    layout = StorageLayout(tmp_path)
    panel_spec = profile.processed.get("market_panel_1d")
    for data_id in panel_spec.source_input_data_ids:
        spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(spec.source_id, data_id),
            source[data_id],
            spec=spec,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    saved = {}
    for data_id in ("market_panel_1d", "execution_mask", "forward_returns"):
        spec = profile.processed.get(data_id)
        root = layout.processed_data_root(spec.market_id, spec.kind, data_id)
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=TimeWindow(dates[0], dates[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
            )
        saved[data_id] = read_dataset(root)
    panel = (
        saved["market_panel_1d"].query("asset_id == '000001'").set_index("timestamp")
    )
    assert dates[2] not in panel.index
    mask = saved["execution_mask"].query("asset_id == '000001'").set_index("timestamp")
    actual_mask = portfolio_production._mask_values(
        mask.loc[[dates[1]]].reset_index(), pd.Index(["000001"])
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
            == panel.loc[exit_date, "adj_open"] / panel.iloc[1].adj_open - 1.0
        )
        assert label.loc[dates[0], column + "_available_at"] == exit_date
    assert mask.iloc[-1][["can_enter", "can_exit"]].isna().all()


@pytest.mark.parametrize("missing_entry", (False, True))
def test_own_market_all_label_horizons_preserve_unknown_and_suspended_sessions(
    tmp_path, missing_entry
):
    from quant_research_platform.markets.implementations.a_share.execution import (
        derive_execution_mask,
    )

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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    panel = producer._compute_processed_frame(
        spec,
        source={
            **producer._prepare_panel_source(source),
            "market_bars_1d": source["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=tuple(dates),
    )
    mask = derive_execution_mask(panel)
    axes = {
        asset: tuple(group.timestamp.sort_values())
        for asset, group in panel.groupby("asset_id")
    }
    assert dates[2] not in axes["000001"] and dates[70] in axes["000001"]
    assert dates[1] in axes["000001"] and dates[3] in axes["000001"]
    spec = profile.processed.get("forward_returns")
    output = producer._compute_processed_frame(
        spec,
        source={},
        dependencies={"market_panel_1d": panel, "execution_mask": mask},
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
        if missing_entry:
            assert pd.isna(saved.loc[dates[0], column])
        else:
            exit_date = axes["000001"][horizon + 1]
            assert (
                saved.loc[dates[0], column]
                == own_panel.loc[exit_date, "adj_open"]
                / own_panel.loc[dates[1], "adj_open"]
                - 1.0
            )
            assert saved.loc[dates[0], column + "_available_at"] == exit_date


@pytest.mark.parametrize("gap", ("membership", "calendar"))
@pytest.mark.parametrize("batch_rows", (6, 300))
def test_unobserved_initial_market_preserves_unknown_age_and_proven_252_sessions(
    tmp_path, monkeypatch, batch_rows, gap
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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    windows = (
        ((0, dates[:6]), (1, dates[-6:])) if gap == "membership" else ((1, dates[-6:]),)
    )
    for index, requested in windows:
        layout = StorageLayout(tmp_path / str(index))
        for data_id in spec.source_input_data_ids:
            source_spec = profile.interim.get(data_id)
            write_dataset(
                layout.interim_data_root(source_spec.source_id, data_id),
                source[data_id],
                spec=source_spec,
            )
        root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
        with producer.produce_a_share_processed_data(
            spec,
            layout=layout,
            time_window=TimeWindow(requested[0], requested[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, metadata, verify):
            write_dataset(
                root,
                frames,
                spec=spec,
                instance_metadata=metadata,
                staged_validator=verify,
            )
        actual = read_dataset(root)
        assert actual.listing_age_trading_days.isna().all()
        assert str(actual.listing_age_trading_days.dtype) == "Int64"
        assert str(actual.has_252_trading_sessions.dtype) == "boolean"
        if index == 0:
            unknown = actual.timestamp.lt(dates[4])
            assert actual.loc[unknown, "market_type"].isna().all()
            assert actual.has_252_trading_sessions.isna().all()
            assert not a_share_eligibility(actual, "pv_expanded").any()
        elif gap == "calendar":
            assert actual.has_252_trading_sessions.all()
            assert a_share_eligibility(actual, "pv_expanded").all()
        else:
            assert actual.has_252_trading_sessions.iloc[:3].isna().all()
            assert actual.has_252_trading_sessions.iloc[3:].all()
            assert (
                a_share_eligibility(actual, "pv_expanded").tolist()
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
    profile = research_profile()
    panel_spec = profile.processed.get("market_panel_1d")
    spec = profile.processed.get("price_volume_facts")
    panel = producer._compute_processed_frame(
        panel_spec,
        source={
            **producer._prepare_panel_source(source),
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
        layout=layout,
        time_window=TimeWindow(core[0], core[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
        )
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
    profile = research_profile()
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    for data_id in spec.source_input_data_ids:
        source_spec = profile.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(source_spec.source_id, data_id),
            source[data_id],
            spec=source_spec,
        )
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    with producer.produce_a_share_processed_data(
        spec,
        layout=layout,
        time_window=TimeWindow(dates[250], dates[-1]),
        processed_catalog=profile.processed,
        source_catalog=profile.interim,
    ) as (frames, metadata, verify):
        write_dataset(
            root, frames, spec=spec, instance_metadata=metadata, staged_validator=verify
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
