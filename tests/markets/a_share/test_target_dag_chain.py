from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.targets import ProducerBinding, ProducerKind
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    FINANCIAL_VALUE_COLUMNS,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.sources.csmar.interim import (
    InterimImportResult,
    normalize_raw_frame,
)
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research


@pytest.mark.parametrize("scenario", ["pv_all", "pv_expanded"])
def test_a_share_target_dag_builds_market_labels_and_eligible_dataset(
    tmp_path: Path, scenario: str
) -> None:
    sessions = pd.bdate_range(
        "2020-01-02", periods=600 if scenario == "pv_expanded" else 400
    )
    frames = _synthetic_interim_frames(sessions)
    if scenario == "pv_expanded":
        bars = frames["market_bars_1d"]
        bars["period_return"] = np.sin(np.arange(len(bars)) / 3) * 0.01
    base = research_profile()

    def source_identity(_raw_root: Path, data_id: str) -> str:
        return (f"{data_id}:{len(sessions)}".encode().hex() + "0" * 64)[:64]

    def producer(**kwargs: object) -> InterimImportResult:
        data_ids = kwargs["data_ids"]
        assert isinstance(data_ids, tuple) and len(data_ids) == 1
        data_id = data_ids[0]
        spec = base.interim.get(data_id)
        output_root = kwargs["layout"].target_root(
            data_id, kwargs["layout"].interim_data_root(spec.source_id, data_id)
        )
        write_dataset(
            output_root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
        )

    bindings = {
        data_id: ProducerBinding(
            ProducerKind.INTERIM, producer, source_identity=source_identity
        )
        for data_id in frames
    }
    profile = replace(base, producer_bindings={**base.producer_bindings, **bindings})
    layout = StorageLayout(project_root=tmp_path)
    result = research.dataset_build(
        profile=profile,
        layout=layout,
        dataset_id=scenario,
        start_date=sessions[0].date().isoformat(),
        end_date=sessions[-1].date().isoformat(),
        raw_root=tmp_path / "synthetic-raw",
    )
    assert result["status"] == "COMPLETE"
    target_results = {item["target_id"]: item for item in result["target_results"]}
    assert target_results["market_panel_1d"]["status"] == "BUILT"
    assert target_results["execution_mask"]["status"] == "BUILT"
    assert target_results["forward_returns"]["status"] == "BUILT"
    assert target_results[scenario]["status"] == "BUILT"
    logical_root = layout.research_dataset_root("a_share", scenario)
    output = read_dataset(
        layout.instance_root(
            logical_root, target_results[scenario]["instance_identity"]
        )
    )
    assert {"eligible", "model_target"}.issubset(output.columns)
    assert {"scenario_universe", "can_enter", "can_exit"}.isdisjoint(output)
    assert output["eligible"].any()
    assert output.loc[output["eligible"], "model_target"].notna().any()
    if scenario == "pv_expanded":
        assert "fundamental_pit" not in target_results
        assert "market_model_factors" in target_results
        for name in (
            "share_price",
            "avg_amount_126d",
            "turnover_250d",
            "beta_252d",
            "residual_volatility_60d",
        ):
            assert output[name].notna().any(), name


def test_fundamental_pit_and_factors_reach_the_same_target_dag(tmp_path: Path) -> None:
    sessions = pd.bdate_range("2020-01-02", periods=400)
    frames = _synthetic_interim_frames(sessions)
    frames.update(_synthetic_financial_frames())
    base = research_profile()
    raw_root = tmp_path / "synthetic-raw"
    source_root = raw_root / "csmar"
    source_root.mkdir(parents=True)
    for data_id in frames:
        (source_root / f"{data_id}.raw").write_bytes(data_id.encode())

    def source_identity(_raw_root: Path, data_id: str) -> str:
        return hashlib.sha256(data_id.encode()).hexdigest()

    def producer(**kwargs: object) -> InterimImportResult:
        data_id = kwargs["data_ids"][0]
        spec = base.interim.get(data_id)
        output_root = kwargs["layout"].target_root(
            data_id, kwargs["layout"].interim_data_root(spec.source_id, data_id)
        )
        write_dataset(
            output_root,
            frames[data_id],
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
        )

    bindings = {
        data_id: ProducerBinding(
            ProducerKind.INTERIM, producer, source_identity=source_identity
        )
        for data_id in frames
    }
    profile = replace(base, producer_bindings={**base.producer_bindings, **bindings})
    result = research.dataset_build(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        dataset_id="fusion_nf_core",
        start_date=sessions[0].date().isoformat(),
        end_date=sessions[-1].date().isoformat(),
        raw_root=raw_root,
    )
    assert result["status"] == "COMPLETE"
    target_results = {item["target_id"]: item for item in result["target_results"]}
    assert target_results["fundamental_pit"]["status"] == "BUILT"
    assert target_results["fundamental_factors"]["status"] == "BUILT"
    assert target_results["fusion_nf_core"]["status"] == "BUILT"
    output = read_dataset(
        StorageLayout(project_root=tmp_path).instance_root(
            StorageLayout(project_root=tmp_path).research_dataset_root(
                "a_share", "fusion_nf_core"
            ),
            target_results["fusion_nf_core"]["instance_identity"],
        )
    )
    assert output["eligible"].any()
    assert output["model_target"].notna().any()


@pytest.mark.parametrize("negative_component", ("current_ytd", "prior_fy", "prior_ytd"))
def test_negative_source_capex_propagates_through_native_pit_and_factor_revision(
    tmp_path, monkeypatch, negative_component
):
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", 6)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "synthetic-raw",
        start_date="2020-04-01",
        end_date="2020-04-10",
        data_ids=("fundamental_factors",),
    )
    assert result["status"] == "COMPLETE"
    targets = {item["target_id"]: item for item in result["target_results"]}

    def restored(data_id):
        spec = profile.processed.get(data_id)
        return (
            read_dataset(
                layout.instance_root(
                    layout.processed_data_root(spec.market_id, spec.kind, data_id),
                    targets[data_id]["instance_identity"],
                )
            )
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
    raw_spec = profile.interim.get("financial_cashflow")
    evidence = read_dataset(
        layout.instance_root(
            layout.interim_data_root(raw_spec.source_id, raw_spec.data_id),
            targets[raw_spec.data_id]["instance_identity"],
        )
    )
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


def _synthetic_financial_frames() -> dict[str, pd.DataFrame]:
    assets = ("000001", "000002", "000003")
    periods = pd.to_datetime(
        [
            "2019-03-31",
            "2019-06-30",
            "2019-09-30",
            "2019-12-31",
            "2020-03-31",
            "2020-06-30",
            "2020-09-30",
            "2020-12-31",
        ]
    )
    frames: dict[str, pd.DataFrame] = {}
    for data_id, columns in FINANCIAL_VALUE_COLUMNS.items():
        rows = [(asset, period) for asset in assets for period in periods]
        frames[data_id] = pd.DataFrame(
            {
                "source_asset_id": [asset for asset, _period in rows],
                "timestamp": [period for _asset, period in rows],
                "report_period": [period for _asset, period in rows],
                "report_type": "A",
                "is_corrected": [False] * len(rows),
                "correction_disclosure_date": pd.Series(
                    pd.NaT, index=range(len(rows)), dtype="datetime64[ns]"
                ),
                "version_identity": "original",
                **{
                    column: [float(index + 1) for index in range(len(rows))]
                    for column in columns
                },
            }
        )
    frames["financial_first_disclosure"] = pd.DataFrame(
        {
            "asset_id": [asset for asset in assets for _period in periods],
            "report_period": [period for _asset in assets for period in periods],
            "first_disclosure_date": [
                period + pd.Timedelta(days=20)
                for _asset in assets
                for period in periods
            ],
        }
    )
    return frames


def _synthetic_interim_frames(sessions: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    assets = ("000001", "000002", "000003")
    rows = [
        (asset, timestamp, asset_index, date_index)
        for date_index, timestamp in enumerate(sessions)
        for asset_index, asset in enumerate(assets, start=1)
    ]
    price = np.array(
        [10.0 + date_index + asset_index for _, _, asset_index, date_index in rows]
    )
    market_bars = pd.DataFrame(
        {
            "source_asset_id": [row[0] for row in rows],
            "timestamp": [row[1] for row in rows],
            "trade_date": [row[1] for row in rows],
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 100.0,
            "amount": price * 100.0,
            "after_hours_volume": 0.0,
            "after_hours_amount": 0.0,
            "prev_close": price - 0.1,
            "limit_status": 0.0,
            "period_return": 0.01,
            "adj_close": price,
            "adj_open": price + 0.1,
            "float_market_cap": 1000000.0,
            "total_market_cap": 1200000.0,
            "turnover": np.nan,
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0],
            "stock_name": assets,
            "listing_date": sessions[0] - pd.Timedelta(days=400),
            "delisting_date": pd.NaT,
            "company_status": "normal",
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
            "status_change_date": sessions[0] - pd.Timedelta(days=400),
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=400),
            "share_change_date": sessions[0] - pd.Timedelta(days=400),
            "total_shares": 120000.0,
            "a_float_shares": 100000.0,
        }
    )
    calendar = pd.DataFrame(
        {
            "timestamp": sessions,
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
            "source_day_of_week": sessions.dayofweek,
            "source_state": "O",
        }
    )
    industry = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "industry_standard_id": "CSRC2012",
            "implement_date": sessions[0] - pd.Timedelta(days=1),
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
        }
    )
    industry_sets = pd.DataFrame(
        {
            "industry_code": "A01",
            "industry_standard_id": "CSRC2012",
            "industry_name": "industry",
            "parent_industry_code": "A",
            "industry_id": "1",
            "industry_level": "1",
        },
        index=[0],
    )
    industry_versions = pd.DataFrame(
        {
            "industry_standard_id": "CSRC2012",
            "industry_standard_name": "CSRC",
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "implement_date": sessions[0] - pd.Timedelta(days=1),
            "abandon_date": pd.NaT,
            "institution_name": "CSRC",
        },
        index=[0],
    )
    listing_changes = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "announcement_date": sessions[0] - pd.Timedelta(days=2),
            "effective_date": sessions[0] - pd.Timedelta(days=1),
            "changed_item": "status",
            "value_before": "",
            "value_after": "normal",
        }
    )
    annual = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0] - pd.Timedelta(days=1),
            "effective_date": sessions[0] - pd.Timedelta(days=1),
            "industry_level1_code": "A01",
            "industry_level1_name": "industry",
            "listing_date": sessions[0] - pd.Timedelta(days=400),
            "listing_state": "normal",
        }
    )
    stock_changes = listing_changes.assign(
        changed_item="P0207行业代码", value_after=industry["industry_level1_name"]
    )
    no_limit = pd.DataFrame(
        {
            "source_asset_id": [assets[0]],
            "timestamp": [sessions[0] - pd.Timedelta(days=1)],
            "trade_date": [sessions[0] - pd.Timedelta(days=1)],
            "reason": ["P8606"],
        }
    )
    special = normalize_raw_frame(
        pd.DataFrame(
            {
                "Symbol": [assets[0]],
                "Chgtype": ["BA"],
                "Annoudt": [sessions[0] - pd.Timedelta(days=2)],
                "Execudt": [sessions[0] - pd.Timedelta(days=1)],
            }
        ),
        "special_treatment_changes",
    )
    snapshots = pd.DataFrame(
        {
            "source_asset_id": assets,
            "timestamp": sessions[0],
            "observation_date": sessions[0] - pd.Timedelta(days=2),
            "publication_date": sessions[0] - pd.Timedelta(days=1),
            "industry_standard_id": "CSRC2012",
            "industry_category_code": "A",
            "industry_major_code": "",
            "industry_code": "01",
            "company_name": "synthetic issuer",
            "snapshot_id": "earlier-public-observation",
            "source_locator": "page:1",
        }
    )
    return {
        "a_share_industry_snapshots": snapshots,
        "market_bars_1d": market_bars,
        "stock_master": stock_master,
        "market_cap": market_cap,
        "trading_calendar": calendar,
        "industry_classification": industry,
        "industry_classification_sets": industry_sets,
        "industry_classification_versions": industry_versions,
        "listing_status_changes": listing_changes,
        "stock_info_changes": stock_changes,
        "stock_info_annual": annual,
        "trading_no_limit": no_limit,
        "special_treatment_changes": special,
    }


def _windowed_source_profile(frames, calls, *, reject_before=None, reject_at=None):
    base = research_profile()
    frames = dict(frames)
    sessions = pd.DatetimeIndex(frames["trading_calendar"].calendar_date)
    calendar = pd.bdate_range(sessions[0] - pd.offsets.BDay(300), sessions[-1]).as_unit(
        "ns"
    )
    history_calendar = pd.DataFrame(
        dict(
            timestamp=calendar,
            calendar_date=calendar,
            source_market_type=1,
            is_trading_day=True,
            source_day_of_week=calendar.dayofweek,
            source_state="O",
        )
    )
    frames["trading_calendar"] = pd.concat(
        [
            history_calendar.loc[history_calendar.calendar_date.lt(sessions[0])],
            frames["trading_calendar"],
        ],
        ignore_index=True,
    )

    def identity(_raw_root, data_id):
        return hashlib.sha256(f"{data_id}:{len(frames[data_id])}".encode()).hexdigest()

    def produce(**kwargs):
        (data_id,) = kwargs["data_ids"]
        spec = base.interim.get(data_id)
        start, end = (
            pd.Timestamp(kwargs["start_date"]),
            pd.Timestamp(kwargs["end_date"]),
        )
        calls.append((data_id, start, end))
        if data_id == "market_bars_1d" and (
            reject_before is not None
            and start < reject_before
            or (reject_at is not None and start == reject_at)
        ):
            raise ValueError("synthetic_history_unavailable")
        frame = frames[data_id]
        if "timestamp" in frame:
            frame = frame.loc[frame.timestamp.le(end)]
        if data_id == "market_bars_1d":
            frame = frame.loc[frame.timestamp.ge(start)]
        root = kwargs["layout"].target_root(
            data_id, kwargs["layout"].interim_data_root(spec.source_id, data_id)
        )
        write_dataset(
            root, frame, spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return InterimImportResult(
            artifact_roots={data_id: root},
            skipped=(),
        )

    return replace(
        base,
        producer_bindings={
            **base.producer_bindings,
            **{
                data_id: ProducerBinding(
                    ProducerKind.INTERIM, produce, source_identity=identity
                )
                for data_id in frames
            },
        },
    )


def test_public_narrow_build_keeps_mixed_terminal_windows_lineage_and_reuse(
    tmp_path, monkeypatch
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )

    opened = []
    original_open = producer.open_dataset_session

    def open_session(root, **kwargs):
        opened.append((root, kwargs["time_window"]))
        return original_open(root, **kwargs)

    monkeypatch.setattr(producer, "open_dataset_session", open_session)
    dates = pd.bdate_range("2020-01-02", periods=425).as_unit("ns")
    requested = dates[410:420]
    frames = _synthetic_interim_frames(dates)
    frames["market_bars_1d"]["period_return"] = (
        np.sin(np.arange(len(frames["market_bars_1d"])) / 3) * 0.01
    )
    calls = []
    profile = _windowed_source_profile(frames, calls)
    layout = StorageLayout(tmp_path)
    targets = ("market_panel_1d", "price_volume_facts", "market_model_factors")
    arguments = dict(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(requested[0].date()),
        end_date=str(requested[-1].date()),
        data_ids=targets,
    )
    plan = research.data_materialize(**arguments, plan=True)
    assert plan["status"] == "COMPLETE"
    assert len({entry["target_id"] for entry in plan["targets"]}) == len(
        plan["targets"]
    )
    assert not calls and (not tuple(tmp_path.rglob("manifest.json")))
    first = research.data_materialize(**arguments)
    assert first["status"] == "COMPLETE", first["blockers"]
    result = {entry["target_id"]: entry for entry in first["target_results"]}
    assert len(result) == len(first["target_results"])
    planned = {entry["target_id"]: entry for entry in plan["targets"]}
    panel_spec = profile.processed.get("market_panel_1d")
    panel_root = layout.processed_data_root(
        panel_spec.market_id, panel_spec.kind, panel_spec.data_id
    )
    public_panel_id = result["market_panel_1d"]["instance_identity"]
    history_panel_id = planned["price_volume_facts"]["upstream_identities"][
        "market_panel_1d"
    ]
    assert history_panel_id != public_panel_id
    assert (
        planned["market_model_factors"]["upstream_identities"]["market_panel_1d"]
        == history_panel_id
    )
    public_panel = read_dataset(layout.instance_root(panel_root, public_panel_id))
    history_panel = read_dataset(layout.instance_root(panel_root, history_panel_id))
    assert public_panel.timestamp.min() == requested[0]
    assert (
        public_panel.timestamp.max() == history_panel.timestamp.max() == requested[-1]
    )
    assert history_panel.timestamp.min() <= dates[0]
    prior = history_panel.loc[history_panel.timestamp.lt(dates[0])]
    assert prior.listing_date.eq(dates[0] - pd.Timedelta(days=400)).all()
    assert prior[["listing_age_trading_days", "close"]].isna().all().all()
    for data_id in ("stock_master", "stock_info_changes"):
        windows = [window for root, window in opened if data_id in root.parts]
        assert windows and all(window is None for window in windows)
    for target in targets[1:]:
        spec = profile.processed.get(target)
        root = layout.processed_data_root(spec.market_id, spec.kind, target)
        actual = read_dataset(
            layout.instance_root(root, result[target]["instance_identity"])
        )
        expected = (
            pd.concat(
                producer.factor_engine._market_model_frames(
                    [history_panel],
                    trading_calendar=read_dataset(
                        layout.instance_root(
                            layout.interim_data_root("csmar", "trading_calendar"),
                            planned["market_model_factors"]["upstream_identities"][
                                "trading_calendar"
                            ],
                        )
                    ),
                    market_history=None,
                ),
                ignore_index=True,
            ).loc[:, list(spec.required_columns)]
            if target == "market_model_factors"
            else producer._compute_processed_frame(
                spec,
                source={},
                dependencies={"market_panel_1d": history_panel},
                trading_sessions=tuple(dates[:420]),
            )
        )
        expected = (
            expected.loc[expected.timestamp.isin(requested)]
            .sort_values(["timestamp", "market_id", "asset_id"], kind="mergesort")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert actual.timestamp.min() == requested[0]
        assert actual.timestamp.max() == requested[-1]
    before_reuse = tuple(calls)
    second = research.data_materialize(**arguments)
    assert second["status"] == "COMPLETE"
    assert all(entry["status"] == "REUSED" for entry in second["target_results"])
    assert tuple(calls) == before_reuse
    assert {
        entry["target_id"]: entry["instance_identity"]
        for entry in second["target_results"]
    } == {key: entry["instance_identity"] for key, entry in result.items()}
    dataset = research.dataset_build(
        profile=profile,
        layout=layout,
        dataset_id="pv_all",
        raw_root=tmp_path / "raw",
        start_date=str(requested[0].date()),
        end_date=str(requested[-1].date()),
    )
    assert dataset["status"] == "COMPLETE", dataset["blockers"]
    dataset_result = next(
        entry for entry in dataset["target_results"] if entry["target_id"] == "pv_all"
    )
    restored = read_dataset(
        layout.instance_root(
            layout.research_dataset_root("a_share", "pv_all"),
            dataset_result["instance_identity"],
        )
    )
    assert (
        restored.timestamp.min() == requested[0]
        and restored.timestamp.max() == requested[-1]
    )
    first_asset = restored.loc[restored.asset_id.eq("000001")].set_index("timestamp")
    history_asset = history_panel.loc[history_panel.asset_id.eq("000001")].set_index(
        "timestamp"
    )
    entry = history_asset.loc[dates[405], "adj_close"]
    expected_reversal = -(history_asset.loc[requested[0], "adj_close"] - entry) / entry
    assert first_asset.loc[requested[0], "reversal_5d"] == expected_reversal
    assert restored["forward_return_20d"].isna().all()


@pytest.mark.parametrize("failed_window", ("history", "core"))
def test_internal_history_failure_blocks_exact_dependent_without_hiding_failure(
    tmp_path, failed_window
):
    dates = pd.bdate_range("2020-01-02", periods=40).as_unit("ns")
    requested = dates[-10:]
    from contextlib import contextmanager

    profile = _windowed_source_profile(_synthetic_interim_frames(dates), [])
    binding = profile.producer_bindings["market_panel_1d"]

    @contextmanager
    def produce_panel(*args, **kwargs):
        start = pd.Timestamp(kwargs["time_window"].start)
        if (start < requested[0]) == (failed_window == "history"):
            raise ValueError("synthetic_history_unavailable")
        with binding.producer(*args, **kwargs) as result:
            yield result

    profile = replace(
        profile,
        producer_bindings={
            **profile.producer_bindings,
            "market_panel_1d": replace(binding, producer=produce_panel),
        },
    )
    result = research.data_materialize(
        profile=profile,
        layout=StorageLayout(tmp_path),
        raw_root=tmp_path / "raw",
        start_date=str(requested[0].date()),
        end_date=str(requested[-1].date()),
        data_ids=("market_panel_1d", "price_volume_facts"),
    )
    assert result["status"] == "FAILED"
    assert any(
        entry["target_id"] == "market_panel_1d"
        and entry["reason"] == "processed_build_failed:synthetic_history_unavailable"
        for entry in result["blockers"]
    )
    targets = {entry["target_id"]: entry for entry in result["target_results"]}
    assert len(targets) == len(result["target_results"])
    if failed_window == "history":
        assert targets["market_panel_1d"]["status"] == "BUILT"
        assert targets["price_volume_facts"]["status"] == "BLOCKED"
        assert "market_panel_1d" in targets["price_volume_facts"]["reason"]
    else:
        assert targets["market_panel_1d"]["status"] == "FAILED"
        assert targets["price_volume_facts"]["status"] == "BUILT"


def test_mixed_window_rebuild_reports_each_instance_and_preserves_old_good(
    tmp_path, monkeypatch
):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )

    dates = pd.bdate_range("2020-01-02", periods=40).as_unit("ns")
    requested = dates[-10:]
    profile = _windowed_source_profile(_synthetic_interim_frames(dates), [])
    layout = StorageLayout(tmp_path)
    arguments = dict(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(requested[0].date()),
        end_date=str(requested[-1].date()),
        data_ids=("market_panel_1d", "price_volume_facts"),
    )
    initial = research.data_materialize(**arguments)
    assert initial["status"] == "COMPLETE"
    plan = research.data_materialize(**arguments, plan=True)
    targets = {entry["target_id"]: entry for entry in plan["targets"]}
    core_identity = targets["market_panel_1d"]["requested_identity"]
    history_identity = targets["price_volume_facts"]["upstream_identities"][
        "market_panel_1d"
    ]
    assert core_identity != history_identity
    spec = profile.processed.get("market_panel_1d")
    logical_root = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)
    old_roots = [
        layout.instance_root(logical_root, identity)
        for identity in (core_identity, history_identity)
    ]

    def originals():
        return {
            (root.name, path.relative_to(root).as_posix()): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for root in old_roots
            for path in root.rglob("*")
            if path.is_file()
        }

    before = originals()
    build_target = research._build_target
    compute = producer._compute_processed_frame
    core_panel_active = False

    def altered_build(**kwargs):
        nonlocal core_panel_active
        core_panel_active = (
            kwargs["spec"].target_id == "market_panel_1d"
            and pd.Timestamp(kwargs["time_window"].start) == requested[0]
        )
        try:
            return build_target(**kwargs)
        finally:
            core_panel_active = False

    def altered_compute(spec, **kwargs):
        output = compute(spec, **kwargs)
        if core_panel_active:
            output["close"] *= 1.001
        return output

    monkeypatch.setattr(research, "_build_target", altered_build)
    monkeypatch.setattr(producer, "_compute_processed_frame", altered_compute)
    rebuilt = research.data_materialize(**arguments, rebuild=True)
    assert rebuilt["status"] == "FAILED"
    evidence = [
        entry
        for entry in rebuilt["rebuild_evidence"]
        if entry["target_id"] == "market_panel_1d"
    ]
    assert len(evidence) == 1
    assert {
        entry["instance_identity"]: entry["comparison_status"] for entry in evidence
    } == {core_identity: "CONTENT_MISMATCH"}
    assert originals() == before
    public = {entry["target_id"]: entry for entry in rebuilt["target_results"]}
    assert public["market_panel_1d"]["status"] == "FAILED"
    assert public["price_volume_facts"]["status"] == "BUILT"


def test_mixed_window_interruption_preserves_completed_public_result_sink(
    tmp_path, monkeypatch
):
    dates = pd.bdate_range("2020-01-02", periods=40).as_unit("ns")
    requested = dates[-10:]
    profile = _windowed_source_profile(_synthetic_interim_frames(dates), [])
    build = research._build_target
    primary = KeyboardInterrupt("synthetic_factor_interruption")

    def interrupt_factor(**kwargs):
        if kwargs["spec"].target_id == "price_volume_facts":
            raise primary
        return build(**kwargs)

    monkeypatch.setattr(research, "_build_target", interrupt_factor)
    sink = {}
    with pytest.raises(KeyboardInterrupt) as caught:
        research._execute_action(
            profile=profile,
            protocol=profile.protocol(
                scenario_id=research._default_scenario_id(profile),
                access_window="development",
            ),
            research_window=research._research_window(
                profile.protocol(
                    scenario_id=research._default_scenario_id(profile),
                    access_window="development",
                ),
                None,
                start_date=str(requested[0].date()),
                end_date=str(requested[-1].date()),
            ),
            action="data materialize",
            terminal_targets=("market_panel_1d", "price_volume_facts"),
            layout=StorageLayout(tmp_path),
            raw_root=tmp_path / "raw",
            result_sink=sink,
        )
    assert caught.value is primary
    assert sink["market_panel_1d"].status.value == "BUILT"
    assert "price_volume_facts" not in sink
    spec = profile.processed.get("market_panel_1d")
    layout = StorageLayout(tmp_path)
    panel = read_dataset(
        layout.instance_root(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            sink["market_panel_1d"].instance_identity,
        )
    )
    assert (
        panel.timestamp.min() == requested[0] and panel.timestamp.max() == requested[-1]
    )


@pytest.mark.parametrize(
    "barrier", ("known", "unknown", "raw_missing", "adjusted_missing")
)
def test_public_narrow_panel_retains_preceding_suspension_state(tmp_path, barrier):
    dates = pd.bdate_range("2020-01-02", periods=7).as_unit("ns")
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    outputs = []
    for start in (dates[0], dates[3]):
        result = research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=tmp_path / "raw",
            start_date=str(start.date()),
            end_date=str(dates[5].date()),
            data_ids=("market_panel_1d",),
        )
        assert result["status"] == "COMPLETE", result["blockers"]
        item = next(
            row
            for row in result["target_results"]
            if row["target_id"] == "market_panel_1d"
        )
        spec = profile.processed.get("market_panel_1d")
        output = read_dataset(
            layout.instance_root(
                layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
                item["instance_identity"],
            )
        )
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        data_ids=("market_panel_1d",),
    )
    assert result["status"] == "COMPLETE", result["blockers"]
    item = next(
        row for row in result["target_results"] if row["target_id"] == "market_panel_1d"
    )
    spec = profile.processed.get("market_panel_1d")
    saved = read_dataset(
        layout.instance_root(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            item["instance_identity"],
        )
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        data_ids=("market_panel_1d", "market_model_factors"),
    )
    assert result["status"] == "COMPLETE", result["blockers"]
    targets = {item["target_id"]: item for item in result["target_results"]}

    def restored(data_id):
        spec = profile.processed.get(data_id)
        return read_dataset(
            layout.instance_root(
                layout.processed_data_root(spec.market_id, spec.kind, data_id),
                targets[data_id]["instance_identity"],
            )
        )

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
    if not known_closed:
        assert (
            actual.loc[dates[641], ["beta_252d", "residual_volatility_60d"]]
            .isna()
            .all()
        )
        return
    assert dates[640] not in actual.index
    assert (
        actual.loc[dates[641], ["beta_252d", "residual_volatility_60d"]].notna().all()
    )
    previous_caps = {}
    history, residuals = [], []
    pending = Fraction(1)
    expected_beta = expected_volatility = np.nan
    for timestamp, cross in panel.groupby("timestamp", sort=False):
        eligible = cross.loc[a_share_eligibility(cross, "pv_expanded")]
        denominator = sum(previous_caps.get(asset, 0.0) for asset in eligible.asset_id)
        market = (
            sum(
                previous_caps.get(row.asset_id, 0.0) * row.period_return
                for row in eligible.itertuples()
            )
            / denominator
            if denominator
            else np.nan
        )
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
            if np.isfinite(interval) and np.isfinite(value):
                history.append((interval, value))
            else:
                history.clear()
            residual = np.nan
            if len(history) >= 252:
                values = np.array(history[-252:])
                alpha, expected_beta = np.linalg.lstsq(
                    np.column_stack((np.ones(252), values[:, 0])),
                    values[:, 1],
                    rcond=None,
                )[0]
                residual = values[-1, 1] - alpha - expected_beta * values[-1, 0]
            residuals.append(residual)
            if len(residuals) >= 60:
                expected_volatility = np.std(residuals[-60:], ddof=0)
            if timestamp == dates[641]:
                assert actual.loc[timestamp, "beta_252d"] == pytest.approx(
                    expected_beta, rel=1e-12
                )
                assert actual.loc[
                    timestamp, "residual_volatility_60d"
                ] == pytest.approx(expected_volatility, rel=1e-12, abs=1e-16)
        previous_caps.update(
            dict(zip(cross.asset_id, cross.float_market_cap, strict=True))
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(dates[-4].date()),
        end_date=str(dates[-1].date()),
        data_ids=("market_panel_1d",),
    )
    assert result["status"] == "COMPLETE", result["blockers"]
    target = next(
        item
        for item in result["target_results"]
        if item["target_id"] == "market_panel_1d"
    )
    spec = profile.processed.get("market_panel_1d")
    saved = (
        read_dataset(
            layout.instance_root(
                layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
                target["instance_identity"],
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
    assert a_share_eligibility(moved_asset, "pv_all").all()
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(dates[start_index].date()),
        end_date=str(dates[-1].date()),
        data_ids=("market_panel_1d", "price_volume_facts"),
    )
    assert result["status"] == "COMPLETE", result["blockers"]
    targets = {item["target_id"]: item for item in result["target_results"]}

    def restored(data_id):
        spec = profile.processed.get(data_id)
        return (
            read_dataset(
                layout.instance_root(
                    layout.processed_data_root(spec.market_id, spec.kind, data_id),
                    targets[data_id]["instance_identity"],
                )
            )
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=tmp_path / "raw",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        data_ids=("market_panel_1d",),
    )
    assert result["status"] == "COMPLETE", result["blockers"]
    item = next(
        row for row in result["target_results"] if row["target_id"] == "market_panel_1d"
    )
    spec = profile.processed.get("market_panel_1d")
    saved = read_dataset(
        layout.instance_root(
            layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
            item["instance_identity"],
        )
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
    profile = _windowed_source_profile(frames, [])
    layout = StorageLayout(tmp_path)
    outputs = []
    for start in (dates[0], dates[2]):
        result = research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=tmp_path / "raw",
            start_date=str(start.date()),
            end_date=str(dates[-1].date()),
            data_ids=("expectations_factors",),
        )
        assert result["status"] == "COMPLETE", result["blockers"]
        item = next(
            row
            for row in result["target_results"]
            if row["target_id"] == "expectations_factors"
        )
        spec = profile.processed.get("expectations_factors")
        saved = read_dataset(
            layout.instance_root(
                layout.processed_data_root(spec.market_id, spec.kind, spec.data_id),
                item["instance_identity"],
            )
        )
        outputs.append(saved.loc[saved.timestamp.ge(dates[2])].reset_index(drop=True))
    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=True)
    selected = outputs[1].loc[outputs[1].asset_id.eq("000001")]
    assert selected.forward_earnings_yield_fy1.notna().all()


def test_optional_source_absence_is_bound_and_changes_identity_when_present(tmp_path):
    sessions = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(sessions)
    base = research_profile()
    panel = base.processed.get("market_panel_1d")
    absent = set(panel.optional_source_input_data_ids)
    calls = []
    layout = StorageLayout(tmp_path)
    # A default-root artifact must never override the bound absent source.
    for data_id in absent:
        spec = base.interim.get(data_id)
        write_dataset(
            layout.interim_data_root(spec.source_id, data_id),
            frames[data_id],
            spec=spec,
        )

    def identity(_root, data_id):
        return (
            None if data_id in absent else hashlib.sha256(data_id.encode()).hexdigest()
        )

    def produce(**kwargs):
        (data_id,) = kwargs["data_ids"]
        calls.append(data_id)
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
        return InterimImportResult(artifact_roots={data_id: root}, skipped=())

    profile = replace(
        base,
        producer_bindings={
            **base.producer_bindings,
            **{
                key: ProducerBinding(
                    ProducerKind.INTERIM, produce, source_identity=identity
                )
                for key in frames
            },
        },
    )

    def run(data_id, *, plan=False):
        return research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=tmp_path / "raw",
            start_date="2020-01-02",
            end_date="2020-01-08",
            data_ids=(data_id,),
            plan=plan,
        )

    first = run("market_panel_1d")
    assert first["status"] == "COMPLETE"
    first_targets = {x["target_id"]: x for x in first["target_results"]}
    assert absent.isdisjoint(first_targets) and absent.isdisjoint(calls)
    first_identity = first_targets["market_panel_1d"]["instance_identity"]
    root = layout.instance_root(
        layout.processed_data_root(panel.market_id, panel.kind, panel.data_id),
        first_identity,
    )
    restored = read_dataset(root)
    assert len(restored) == 15
    assert restored.industry_level1_code.isna().all()
    direct = run("industry_classification")
    assert direct["status"] == "BLOCKED"
    absent.remove("industry_classification")
    second = run("market_panel_1d")
    assert second["status"] == "COMPLETE"
    second_targets = {x["target_id"]: x for x in second["target_results"]}
    assert second_targets["market_panel_1d"]["instance_identity"] != first_identity
    assert second_targets["industry_classification"]["status"] == "BUILT"
    planned = run("market_panel_1d", plan=True)
    panel_plan = next(
        item for item in planned["targets"] if item["target_id"] == "market_panel_1d"
    )
    assert "industry_classification" in panel_plan["upstream_target_ids"]
    assert (
        panel_plan["upstream_identities"]["industry_classification"]
        == (second_targets["industry_classification"]["instance_identity"])
    )
    assert read_dataset(root).equals(restored)


@pytest.mark.parametrize("failure", ("present_invalid", "required_absent"))
def test_optional_source_only_absence_is_qualified(tmp_path, failure):
    sessions = pd.bdate_range("2020-01-02", periods=5).as_unit("ns")
    frames = _synthetic_interim_frames(sessions)
    profile = _windowed_source_profile(frames, [])
    original = profile.producer_bindings["industry_classification"]
    if failure == "present_invalid":

        def invalid(**kwargs):
            raise ValueError("synthetic_bad_source")

        binding = replace(original, producer=invalid)
        key = "industry_classification"
    else:
        key = "market_cap"
        binding = replace(
            profile.producer_bindings[key], source_identity=lambda *_: None
        )
    profile = replace(
        profile, producer_bindings={**profile.producer_bindings, key: binding}
    )
    result = research.data_materialize(
        profile=profile,
        layout=StorageLayout(tmp_path),
        raw_root=tmp_path / "raw",
        start_date="2020-01-02",
        end_date="2020-01-08",
        data_ids=("market_panel_1d",),
    )
    assert result["status"] != "COMPLETE"
    assert not any(
        x["target_id"] == "market_panel_1d" and x["status"] in {"BUILT", "REUSED"}
        for x in result["target_results"]
    )
