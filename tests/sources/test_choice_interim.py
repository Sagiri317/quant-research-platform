from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.research import read_research_run_record
from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    construct_ttm_events,
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.choice import interim
from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
from quant_research_platform.sources.choice.raw_exports import choice_session
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_is_dir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_unlink,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.choice_raw import SDK, response, write_css
from tests.helpers.research_configuration import research_configuration

FIELDS = tuple(field for fields in FINANCIAL_FIELDS.values() for field in fields)
DATE_FIELDS = ("STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE")
CODE = "000001.SZ"


def _file_hashes(root: Path) -> dict[Path, str]:
    return {
        (parent / name).relative_to(root): hashlib.sha256(
            filesystem_read_bytes(parent / name)
        ).hexdigest()
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


def _disclosure(
    session, layout, *, period="2019-12-31", first="2020/1/30", events=None
):
    return write_css(
        session,
        layout,
        period=period,
        end_date="2025-12-31",
        indicators=DATE_FIELDS,
        data={CODE: [first, None, events]},
    )


def _snapshot(
    session, layout, day, values, *, branch="1", period="2019-12-31", fields=FIELDS
):
    return write_css(
        session,
        layout,
        period=period,
        end_date=day,
        indicators=fields,
        data={CODE: values},
        branch=branch,
    )


def _read(result, policy):
    return {
        data_id: read_dataset(
            root, mode=DatasetReadMode.MATERIALIZED, resource_policy=policy
        )
        for data_id, root in result.artifact_roots.items()
    }


@pytest.mark.parametrize("field", ["TINCOMESTATEMENT_61", "TBALANCESTATEMENT_88"])
def test_native_null_revisions_preserve_history_and_next_session_visibility(
    tmp_path, field
):
    layout = StorageLayout(tmp_path)
    position = FIELDS.index(field)
    before = [100.0] * len(FIELDS)
    corrected = before.copy()
    corrected[position] = 200.0
    withdrawn = corrected.copy()
    withdrawn[position] = None
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01,2020-02-02")
        _snapshot(session, layout, "2020-01-30", before)
        _snapshot(session, layout, "2020-01-30", [None] * len(FIELDS), branch="3")
        for day, values in (("2020-02-01", corrected), ("2020-02-02", withdrawn)):
            for branch in ("1", "3"):
                _snapshot(session, layout, day, values, branch=branch)
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            start_date="2019-01-01",
            end_date="2019-12-31",
            resource_policy=policy,
        )
        frames = _read(result, policy)
        assert set(frames) == {*FINANCIAL_FIELDS, "financial_first_disclosure"}
        for data_id in frames:
            profile = research_profile()
            assert str(profile.interim.get(data_id).source_id) == "choice"
            assert (
                profile.producer_bindings[data_id].producer
                is interim.materialize_interim_data
            )
        data_id = next(
            name for name, fields in FINANCIAL_FIELDS.items() if field in fields
        )
        column = FINANCIAL_FIELDS[data_id][field]
        values = frames[data_id][column]
        assert values.iloc[:2].tolist() == [100.0, 200.0] and pd.isna(values.iloc[2])
        assert frames[data_id].report_period.eq(pd.Timestamp("2019-12-31")).all()
        sessions = pd.DatetimeIndex(["2020-01-31", "2020-02-03", "2020-02-04"])
        events = financial_events_from_interim(
            {name: frames[name] for name in FINANCIAL_FIELDS},
            frames["financial_first_disclosure"],
            trading_sessions=sessions,
        )
        ttm = construct_ttm_events(events)
        selected = ttm.loc[ttm.metric_id.eq(column)]
        assert selected.available_at.tolist() == list(sessions[:2])
        assert selected.value.iloc[0] == 100.0 and pd.isna(selected.value.iloc[1])
        pit = materialize_pit_fundamentals(
            pd.DataFrame({"asset_id": ["000001"] * 2, "timestamp": sessions[:2]}),
            selected,
        )
        assert pit.value.iloc[0] == 100.0 and pd.isna(pit.value.iloc[1])
        assert policy.used_memory_bytes == 0


def test_same_coordinate_overlap_compares_values_and_keeps_request_scope(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        first, _ = _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", list(map(float, range(14))))
        _snapshot(
            session,
            layout,
            "2020-01-30",
            list(reversed(list(map(float, range(14))))),
            fields=tuple(reversed(FIELDS)),
        )
        _disclosure(session, layout, period="2018-12-31")
    with runtime_resource_policy_scope() as policy:
        identity = interim.source_content_identity(
            layout.raw_source_root("choice"), "financial_balance"
        )
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("financial_balance",),
            start_date="2019-01-01",
            end_date="2019-12-31",
            resource_policy=policy,
        )
        frame = _read(result, policy)["financial_balance"]
        assert len(frame) == 1 and frame.total_assets.iloc[0] == 0.0
        assert identity == interim.source_content_identity(
            layout.raw_source_root("choice"), "financial_balance"
        )
        with pytest.raises(
            PortfolioTargetBlocked, match="choice_first_disclosure_snapshot_missing"
        ):
            interim.materialize_interim_data(
                layout=layout,
                data_ids=("financial_income",),
                resource_policy=policy,
            )
        assert filesystem_is_dir(first) and policy.used_memory_bytes == 0


def test_disclosure_and_financial_superset_is_one_raw_response(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        write_css(
            session,
            layout,
            period="2019-12-31",
            end_date="2020-01-30",
            indicators=(*DATE_FIELDS, *FIELDS),
            data={CODE: ["2020/1/30", None, None, *([100.0] * 14)]},
        )
    with runtime_resource_policy_scope() as policy:
        frames = _read(
            interim.materialize_interim_data(layout=layout, resource_policy=policy),
            policy,
        )
        assert frames["financial_balance"].total_assets.tolist() == [100.0]
        assert frames["financial_first_disclosure"].first_disclosure_date.tolist() == [
            pd.Timestamp("2020-01-30")
        ]


def test_disclosure_date_list_order_and_duplicates_are_semantically_equal(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01,2020-02-03")
        write_css(
            session,
            layout,
            period="2019-12-31",
            end_date="2025-12-31",
            indicators=tuple(reversed(DATE_FIELDS)),
            data={
                CODE: [
                    ["2020/2/3", "2020-02-01", "2020/2/3"],
                    [],
                    ["2020-01-30", "2020/1/30"],
                ]
            },
        )
    with runtime_resource_policy_scope() as policy:
        frames = _read(
            interim.materialize_interim_data(
                layout=layout,
                data_ids=("financial_first_disclosure",),
                resource_policy=policy,
            ),
            policy,
        )
        assert frames["financial_first_disclosure"].first_disclosure_date.tolist() == [
            pd.Timestamp("2020-01-30")
        ]
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("field", DATE_FIELDS)
def test_partial_disclosure_completeness_obeys_requested_report_period(tmp_path, field):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
        write_css(
            session,
            layout,
            period="2018-12-31",
            end_date="2025-12-31",
            indicators=(field,),
            data={CODE: ["2019/1/30" if field == "STMTACTDATE" else None]},
        )
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            start_date="2019-01-01",
            end_date="2019-12-31",
            resource_policy=policy,
        )
        frames = _read(result, policy)
        assert set(frames) == {*FINANCIAL_FIELDS, "financial_first_disclosure"}
        assert frames["financial_balance"].total_assets.tolist() == [100.0]
        for data_id in ("financial_balance", "financial_first_disclosure"):
            with pytest.raises(
                PortfolioTargetBlocked, match="choice_disclosure_field_missing"
            ) as caught:
                interim.materialize_interim_data(
                    layout=StorageLayout(tmp_path / "unscoped"),
                    raw_root=layout.raw_source_root("choice"),
                    data_ids=(data_id,),
                    resource_policy=policy,
                )
            assert "report_period=2018-12-31" in " ".join(caught.value.__notes__)
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("foreign", [None, "period", "end_date", "branch"])
def test_unfetched_field_is_blocked_and_other_coordinates_cannot_fill_it(
    tmp_path, financial, foreign
):
    layout = StorageLayout(tmp_path)
    fields = FIELDS if financial else DATE_FIELDS
    values = [None] * len(fields) if financial else ["2020/1/30", None, None]
    data_id = "financial_balance" if financial else "financial_first_disclosure"
    day = "2020-01-30" if financial else "2025-12-31"
    with choice_session(SDK(response())) as session:
        if financial:
            _disclosure(session, layout)
        write_css(
            session,
            layout,
            period="2019-12-31",
            end_date=day,
            indicators=fields[:-1],
            data={CODE: values[:-1]},
        )
        if foreign is not None:
            write_css(
                session,
                layout,
                period="2020-12-31" if foreign == "period" else "2019-12-31",
                end_date="2025-01-01" if foreign == "end_date" else day,
                branch="3" if foreign == "branch" else "1",
                indicators=fields[-1:],
                data={CODE: values[-1:]},
            )
    with runtime_resource_policy_scope() as policy:
        before = interim.source_content_identity(
            layout.raw_source_root("choice"), data_id
        )
        reason = (
            "choice_financial_snapshot_field_missing"
            if financial
            else "choice_disclosure_field_missing"
        )
        with pytest.raises(PortfolioTargetBlocked, match=reason) as caught:
            interim.materialize_interim_data(
                layout=layout, data_ids=(data_id,), resource_policy=policy
            )
        details = " ".join(caught.value.__notes__)
        assert any(field in details for field in fields)
        assert (
            "report_period=" in details and "EndDate=" in details and "type=" in details
        )
        if foreign is None:
            assert fields[-1] in details
        assert not filesystem_exists(layout.interim_data_root("choice", data_id))
        assert policy.used_memory_bytes == 0
    if foreign is None:
        with choice_session(SDK(response())) as session:
            write_css(
                session,
                layout,
                period="2019-12-31",
                end_date=day,
                indicators=fields[-1:],
                data={CODE: values[-1:]},
            )
        with runtime_resource_policy_scope() as policy:
            assert before != interim.source_content_identity(
                layout.raw_source_root("choice"), data_id
            )
            frames = _read(
                interim.materialize_interim_data(
                    layout=layout, data_ids=(data_id,), resource_policy=policy
                ),
                policy,
            )
            assert len(frames[data_id]) == 1
            if financial:
                assert (
                    frames[data_id][list(FINANCIAL_FIELDS[data_id].values())]
                    .isna()
                    .all()
                    .all()
                )
            assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("null", [False, True])
def test_overlapping_single_field_conflict_including_provider_null_is_rejected(
    tmp_path, financial, null
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        if financial:
            _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
        write_css(
            session,
            layout,
            period="2019-12-31",
            end_date="2020-01-30" if financial else "2025-12-31",
            indicators=FIELDS[:1] if financial else DATE_FIELDS[:1],
            data={CODE: [None if null else 99.0 if financial else "2020/1/31"]},
        )
    with pytest.raises(
        ValueError,
        match="choice_snapshot_conflict"
        if financial
        else "choice_disclosure_field_conflict",
    ) as caught:
        interim.materialize_interim_data(layout=layout, data_ids=("financial_balance",))
    assert any("request_identities=" in note for note in caught.value.__notes__)
    assert not filesystem_exists(
        layout.interim_data_root("choice", "financial_balance")
    )


def test_same_day_branch_conflict_is_not_hidden_by_the_current_canonical_value(
    tmp_path,
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01")
        _snapshot(session, layout, "2020-01-30", [100.0] * 14)
        for branch, old, new in (("1", 90.0, 100.0), ("3", 80.0, 110.0)):
            _snapshot(session, layout, "2020-01-31", [old] * 14, branch=branch)
            _snapshot(session, layout, "2020-02-01", [new] * 14, branch=branch)
    with pytest.raises(ValueError, match="choice_financial_version_values_conflict"):
        interim.materialize_interim_data(layout=layout, data_ids=("financial_balance",))
    assert not filesystem_exists(
        layout.interim_data_root("choice", "financial_balance")
    )


def test_missing_predecessor_is_not_a_proven_all_null_snapshot(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01")
        _snapshot(session, layout, "2020-01-30", [100.0] * 14)
        _snapshot(session, layout, "2020-02-01", [100.0] * 14)
        _snapshot(session, layout, "2020-02-01", [110.0] * 14, branch="3")
    with pytest.raises(
        PortfolioTargetBlocked, match="choice_revision_predecessor_missing"
    ):
        interim.materialize_interim_data(layout=layout, data_ids=("financial_balance",))
    with choice_session(SDK(response())) as session:
        _snapshot(session, layout, "2020-01-31", [None] * 14, branch="3")
    with runtime_resource_policy_scope() as policy:
        frame = _read(
            interim.materialize_interim_data(
                layout=layout,
                data_ids=("financial_balance",),
                resource_policy=policy,
            ),
            policy,
        )["financial_balance"]
        assert frame.total_assets.tolist() == [100.0, 110.0]


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("initial", "choice_first_disclosure_snapshot_missing"),
        ("revision", "choice_revision_snapshot_missing"),
        ("predecessor", "choice_revision_predecessor_missing"),
        ("chronology", "choice_disclosure_chronology_unresolved"),
    ],
)
def test_known_missing_coordinates_are_blocked_in_the_real_run_and_report(
    tmp_path, mode, reason
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(
            session,
            layout,
            events="2020-01-29" if mode == "chronology" else "2020-02-01",
        )
        if mode != "initial":
            _snapshot(session, layout, "2020-01-30", [100.0] * 14)
        if mode == "predecessor":
            for branch in ("1", "3"):
                _snapshot(session, layout, "2020-02-01", [110.0] * 14, branch=branch)
    profile = research_profile()
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=research_configuration(development=("2019-03-04", "2019-03-15")),
    )
    record = read_research_run_record(layout.research_run_path(result["run_id"]))
    targets = {row.target_id: row for row in record.target_results}
    assert targets["financial_balance"].status.value == "BLOCKED"
    assert targets["financial_balance"].reason == reason
    assert targets["pv_all_champion_csi500_enhanced_ledger"].status.value == "BLOCKED"
    report = research.report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert report["status"] != "COMPLETE"
    assert reason in str(report["report"]["sections"]["readiness/blockers"])


def test_source_primary_exception_survives_index_cleanup_failure(tmp_path, monkeypatch):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", [100.0] * 14)
    actual_rows = interim._rows
    actual_close = interim.ExactPrimaryKeyIndex.close
    primary = KeyboardInterrupt("synthetic_choice_primary")

    def rows(*args):
        yield from actual_rows(*args)
        raise primary

    def close(index):
        actual_close(index)
        if index.primary_key == ("source_coordinate",):
            raise RuntimeError("synthetic_choice_cleanup")

    monkeypatch.setattr(interim, "_rows", rows)
    monkeypatch.setattr(interim.ExactPrimaryKeyIndex, "close", close)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(KeyboardInterrupt) as caught:
            interim.materialize_interim_data(
                layout=layout,
                data_ids=("financial_balance",),
                resource_policy=policy,
            )
        assert caught.value is primary
        assert any("RuntimeError" in note for note in primary.__notes__)
        assert policy.used_memory_bytes == 0
    assert not filesystem_exists(
        layout.interim_data_root("choice", "financial_balance")
    )


def _persisted_financials_feed_all_nineteen_factors_and_keep_nulls(tmp_path):
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.contracts.data import (
        DataFrequency,
        DataGrain,
        ProcessedDataKind,
        ProcessedDataSpec,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.research.factors.fundamental.definitions import (
        FUNDAMENTAL_FACTORS,
    )
    from quant_research_platform.storage.datasets.writing import write_dataset

    layout = StorageLayout(tmp_path)
    profile = research_profile()
    current = {
        "total_assets": 100.0,
        "total_liabilities": 40.0,
        "parent_equity": 60.0,
        "cash": 10.0,
        "short_term_borrowings": 5.0,
        "current_portion_noncurrent_liabilities": 2.0,
        "long_term_borrowings": 8.0,
        "bonds_payable": 1.0,
        "operating_revenue": 30.0,
        "operating_cost": 20.0,
        "consolidated_net_income": 8.0,
        "parent_net_income": 7.0,
        "operating_cash_flow": 9.0,
        "capex_cash_paid": 4.0,
    }
    prior = {
        **current,
        "total_assets": 80.0,
        "parent_equity": 50.0,
        "parent_net_income": 6.0,
        "operating_revenue": 25.0,
    }
    names = tuple(
        name for fields in FINANCIAL_FIELDS.values() for name in fields.values()
    )
    with choice_session(SDK(response())) as session:
        for period, first, values in (
            ("2018-12-31", "2019-01-30", prior),
            ("2019-12-31", "2020-01-30", current),
        ):
            _disclosure(
                session,
                layout,
                period=period,
                first=first,
                events="2020-02-01" if period == "2019-12-31" else None,
            )
            _snapshot(
                session, layout, first, [values[name] for name in names], period=period
            )
        _snapshot(session, layout, "2020-01-30", [None] * 14, branch="3")
        _snapshot(
            session,
            layout,
            "2020-02-01",
            [None if name == "parent_net_income" else current[name] for name in names],
        )
        _snapshot(session, layout, "2020-02-01", [None] * 14, branch="3")

    with runtime_resource_policy_scope() as policy:
        imported = interim.materialize_interim_data(
            layout=layout,
            start_date="2018-01-01",
            end_date="2019-12-31",
            resource_policy=policy,
        )
        persisted = _read(imported, policy)
        sessions = pd.bdate_range("2018-01-01", "2020-02-04")
        calendar_spec = profile.interim.get("trading_calendar")
        calendar = pd.DataFrame(
            {
                "timestamp": sessions,
                "calendar_date": sessions,
                "source_market_type": 4,
                "is_trading_day": True,
                "source_day_of_week": sessions.dayofweek + 1,
                "source_state": "O",
            }
        )
        write_dataset(
            layout.interim_data_root(calendar_spec.source_id, calendar_spec.data_id),
            calendar.loc[:, list(calendar_spec.required_columns)],
            spec=calendar_spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        market = pd.DataFrame(
            {
                "market_id": "a_share",
                "asset_id": ["000001", "000001"],
                "timestamp": pd.to_datetime(["2020-01-31", "2020-02-03"]),
                "total_market_cap": 200.0,
            }
        )
        market_spec = ProcessedDataSpec(
            data_id="market_panel_1d",
            market_id="a_share",
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            kind=ProcessedDataKind.PANEL,
            column_roles={
                "market_id": ColumnRole.KEY,
                "asset_id": ColumnRole.KEY,
                "timestamp": ColumnRole.TIMESTAMP,
                "total_market_cap": ColumnRole.CONTEXT,
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
        )
        write_dataset(
            layout.processed_data_root(
                "a_share", market_spec.kind, market_spec.data_id
            ),
            market,
            spec=market_spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        for data_id in ("fundamental_pit", "fundamental_factors"):
            spec = profile.processed.get(data_id)
            produced = producer.produce_a_share_processed_data(
                spec,
                layout=layout,
                time_window=TimeWindow("2018-01-01", "2020-02-04"),
                processed_catalog=profile.processed,
                source_catalog=profile.interim,
                resource_policy=policy,
            )
            root = layout.processed_data_root("a_share", spec.kind, data_id)
            with produced as (frames, metadata, verify):
                write_dataset(
                    root,
                    frames,
                    spec=spec,
                    resource_policy=policy,
                    instance_metadata=metadata,
                    staged_validator=verify,
                )
            persisted[data_id] = read_dataset(root, resource_policy=policy)
        saved = read_dataset(root, resource_policy=policy)
        expected = [
            0.035,
            0.3,
            0.15,
            0.045,
            7 / 55,
            8 / 90,
            10 / 90,
            1 / 3,
            0.1,
            -1 / 90,
            0.25,
            0.2,
            1 / 90,
            0.2,
            4 / 90,
            0.4,
            0.16,
            0.1,
            0.06,
        ]
        assert saved.loc[0, list(FUNDAMENTAL_FACTORS)].tolist() == pytest.approx(
            expected
        )
        assert (
            saved.loc[1, ["earnings_yield", "roe", "earnings_change_to_assets"]]
            .isna()
            .all()
        )
        assert saved.loc[1, "roa"] == pytest.approx(8 / 90)
        assert policy.used_memory_bytes == 0
        return persisted


@pytest.mark.parametrize("group_size", [1, 2])
def test_choice_persisted_financials_feed_all_nineteen_factors_and_keep_nulls(
    tmp_path, monkeypatch, group_size
):
    full = _persisted_financials_feed_all_nineteen_factors_and_keep_nulls(
        tmp_path / "full"
    )
    original_write_css = write_css

    def split_css(session, layout, *, indicators, data, **kwargs):
        disclosure = set(indicators).issubset(DATE_FIELDS)
        if disclosure:
            original_write_css(
                session, layout, indicators=indicators, data=data, **kwargs
            )
        for start in range(0, len(indicators), group_size):
            selected = tuple(reversed(indicators[start : start + group_size]))
            selected_data = {}
            for code, cells in data.items():
                values = []
                for field in selected:
                    value = cells[indicators.index(field)]
                    if disclosure:
                        dates = (
                            []
                            if value is None
                            else value.split(",")
                            if isinstance(value, str)
                            else value
                        )
                        value = [day.replace("-", "/") for day in reversed(dates)] * 2
                    values.append(value)
                selected_data[code] = values
            result = original_write_css(
                session,
                layout,
                indicators=selected,
                data=selected_data,
                **kwargs,
            )
        return result

    monkeypatch.setattr(__name__ + ".write_css", split_css)
    split = _persisted_financials_feed_all_nineteen_factors_and_keep_nulls(
        tmp_path / "split"
    )
    assert (
        set(full)
        == set(split)
        == {
            *FINANCIAL_FIELDS,
            "financial_first_disclosure",
            "fundamental_pit",
            "fundamental_factors",
        }
    )
    for data_id in full:
        pd.testing.assert_frame_equal(full[data_id], split[data_id], check_exact=True)


@pytest.mark.parametrize("mutation", ["add", "remove", "change"])
def test_raw_mutation_before_publish_preserves_old_good(
    tmp_path, monkeypatch, mutation
):
    layout = StorageLayout(tmp_path / "project")
    extra = StorageLayout(tmp_path / "extra")
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        snapshot, _ = _snapshot(session, layout, "2020-01-30", [100.0] * 14)
        added, _ = _snapshot(session, extra, "2020-01-29", [100.0] * 14)
    good = interim.materialize_interim_data(
        layout=layout, data_ids=("financial_balance",)
    )
    good_root = good.artifact_roots["financial_balance"]
    before = _file_hashes(good_root)
    candidate_layout = StorageLayout(tmp_path / "candidate")
    candidate = candidate_layout.interim_data_root("choice", "financial_balance")
    actual_rows = interim._rows

    def mutating_rows(*args):
        yield from actual_rows(*args)
        if mutation == "add":
            shutil.copytree(
                native_filesystem_path(added),
                native_filesystem_path(
                    layout.raw_source_root("choice") / "css" / added.name
                ),
            )
        elif mutation == "remove":
            filesystem_unlink(snapshot / "response.csv")
        else:
            path = snapshot / "response.csv"
            filesystem_write_bytes(
                path, filesystem_read_bytes(path).replace(b"100.0", b"101.0", 1)
            )

    monkeypatch.setattr(interim, "_rows", mutating_rows)
    with pytest.raises(ValueError, match="choice_raw_identity_changed"):
        interim.materialize_interim_data(
            layout=candidate_layout,
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
    assert not filesystem_exists(candidate)
    assert before == _file_hashes(good_root)


def test_bulk_import_reuses_only_adjacent_mode_index_and_keeps_published_identity(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        snapshot, _ = _snapshot(session, layout, "2020-01-30", [100.0] * 14)
    original_index = interim._raw_index
    opened = []
    active = 0

    @contextmanager
    def observed_index(root, financial, policy, *, members):
        nonlocal active
        active += 1
        assert active == 1
        opened.append(financial)
        try:
            with original_index(root, financial, policy, members=members) as index:
                yield index
        finally:
            active -= 1

    monkeypatch.setattr(interim, "_raw_index", observed_index)
    good = interim.materialize_interim_data(layout=layout)
    assert opened == [True, False]
    before = {
        root / relative: digest
        for root in good.artifact_roots.values()
        for relative, digest in _file_hashes(root).items()
    }
    original_write = interim.write_dataset
    published = []
    candidate = StorageLayout(tmp_path / "candidate")

    def mutate_after_first_publication(*args, **kwargs):
        result = original_write(*args, **kwargs)
        published.append(kwargs["spec"].data_id)
        if len(published) == 1:
            before.update(
                {
                    args[0] / relative: digest
                    for relative, digest in _file_hashes(args[0]).items()
                }
            )
            raw = snapshot / "response.csv"
            filesystem_write_bytes(
                raw, filesystem_read_bytes(raw).replace(b"100.0", b"101.0", 1)
            )
        return result

    monkeypatch.setattr(interim, "write_dataset", mutate_after_first_publication)
    with pytest.raises(ValueError, match="choice_raw_identity_changed"):
        interim.materialize_interim_data(
            layout=candidate, raw_root=layout.raw_source_root("choice")
        )
    assert published == ["financial_balance"]
    assert opened == [True, False, True]
    assert active == 0
    assert not filesystem_exists(
        candidate.interim_data_root("choice", "financial_income")
    )
    assert all(
        hashlib.sha256(filesystem_read_bytes(path)).hexdigest() == digest
        for path, digest in before.items()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "metadata_missing",
        "metadata_type",
        "request_missing",
        "arguments_type",
        "arguments_count",
        "cutoff_type",
        "csv_missing",
        "json_corrupt",
    ],
)
def test_raw_metadata_invalid_is_source_local(tmp_path, mutation):
    import json

    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        root, _ = _disclosure(session, layout)
    sidecar = root / "response.metadata.json"
    payload = json.loads(filesystem_read_text(sidecar, encoding="utf-8"))
    if mutation == "metadata_missing":
        del payload["metadata"]
    elif mutation == "metadata_type":
        payload["metadata"] = []
    elif mutation == "request_missing":
        del payload["metadata"]["request"]
    elif mutation == "arguments_type":
        payload["metadata"]["request"]["arguments"] = None
    elif mutation == "arguments_count":
        payload["metadata"]["request"]["arguments"] = ["x"]
    elif mutation == "cutoff_type":
        payload["metadata"]["request"]["cutoff"] = 1
    elif mutation == "csv_missing":
        filesystem_unlink(root / "response.csv")
    filesystem_write_text(
        sidecar,
        "{" if mutation == "json_corrupt" else json.dumps(payload),
        encoding="utf-8",
    )
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(ValueError):
            interim.source_content_identity(
                layout.raw_source_root("choice"), "financial_first_disclosure"
            )
        assert policy.used_memory_bytes == 0


def test_invalid_choice_metadata_preserves_unrelated_target_and_report(tmp_path):
    import json

    from tests.sources.reference.test_interim import _write_reference_root

    layout = StorageLayout(tmp_path)
    _write_reference_root(layout.project_root / "data/raw")
    with choice_session(SDK(response())) as session:
        root, _ = _disclosure(session, layout)
    sidecar = root / "response.metadata.json"
    payload = json.loads(filesystem_read_text(sidecar, encoding="utf-8"))
    del payload["metadata"]["request"]
    filesystem_write_text(sidecar, json.dumps(payload), encoding="utf-8")
    profile = research_profile()
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=research_configuration(development=("2019-03-04", "2019-03-15")),
    )
    record = read_research_run_record(layout.research_run_path(result["run_id"]))
    targets = {row.target_id: row for row in record.target_results}
    assert targets["financial_first_disclosure"].status.value == "BLOCKED"
    assert targets["financial_first_disclosure"].reason == "source_identity_invalid"
    assert targets["a_share_fee_schedule"].status.value == "BUILT"
    report = research.report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert report["status"] != "COMPLETE"
    assert "source_identity_invalid" in str(
        report["report"]["sections"]["readiness/blockers"]
    )


@pytest.mark.parametrize(
    "error", [PermissionError("synthetic_permission"), OSError(5, "synthetic_io")]
)
@pytest.mark.parametrize("boundary", ["metadata", "directory"])
def test_choice_fatal_io_is_not_source_blocked(tmp_path, monkeypatch, error, boundary):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)

    def fail(_path):
        raise error

    if boundary == "metadata":
        monkeypatch.setattr(interim, "read_manifest", fail)
    else:
        original = interim.filesystem_scandir

        def scan(path):
            if str(path) == str(layout.raw_source_root("choice") / "css"):
                raise error
            return original(path)

        monkeypatch.setattr(interim, "filesystem_scandir", scan)
    with pytest.raises(type(error)) as caught:
        research.data_materialize(
            profile=research_profile(),
            layout=layout,
            raw_root=None,
            start_date="2019-01-01",
            end_date="2019-12-31",
            data_ids=("financial_first_disclosure",),
        )
    assert caught.value is error
