from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
from quant_research_platform.markets.implementations.a_share.fundamentals import (
    _pit_observations,
    _prepare_pit_fundamentals,
    construct_ttm_events,
    financial_events_from_interim,
)
from quant_research_platform.sources.choice import interim
from quant_research_platform.sources.choice.fields import FINANCIAL_FIELDS
from quant_research_platform.sources.choice.raw_exports import (
    ChoiceRequest,
    choice_session,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_unlink,
    filesystem_walk,
    filesystem_write_text,
    native_filesystem_path,
)
from tests.helpers.choice_raw import SDK, response, write_css
from tests.helpers.paths import StorageLayout

FIELDS = tuple(field for fields in FINANCIAL_FIELDS.values() for field in fields)
DATE_FIELDS = ("STMTACTDATE", "ACTUALISSUINGDATE", "ITEMVALUECHANGEDATE")
CODE = "000001.SZ"


def _file_hashes(root: Path) -> dict[Path, str]:
    return {
        (parent / name).relative_to(root): hashlib.sha256(
            Path(native_filesystem_path(parent / name)).read_bytes()
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


def _read(result):
    return {
        data_id: read_dataset(root, mode=DatasetReadMode.MATERIALIZED)
        for data_id, root in result.artifact_roots.items()
    }


@pytest.mark.parametrize("extra", ["ReportType=3", "Type=1,ReportType=2"])
def test_financial_css_rejects_ctr_report_type_without_remapping(tmp_path, extra):
    from datetime import date

    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        session.sdk.response = response(
            Codes=[CODE],
            Indicators=list(FIELDS),
            Dates=["2020-01-30"],
            Data={CODE: [100.0] * len(FIELDS)},
        )
        request = ChoiceRequest(
            "css",
            (
                CODE,
                ",".join(FIELDS),
                f"ReportDate=2019-12-31,EndDate=2020-01-30,{extra}",
            ),
            "synthetic-official-shape",
            date(2025, 12, 31),
        )
        session.fetch(
            layout.raw_source_root("choice") / request.endpoint / request.identity(),
            request,
        )
    before = _file_hashes(layout.raw_source_root("choice"))
    with pytest.raises(ValueError, match="choice_financial_css_reporttype_unsupported"):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
    assert _file_hashes(layout.raw_source_root("choice")) == before
    assert not filesystem_exists(
        layout.interim_data_root("choice", "financial_balance")
    )


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
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("choice", key)
            for key in tuple(spec.data_id for spec in interim.interim_catalog().list())
        },
        raw_root=layout.raw_source_root("choice"),
        start_date="2019-01-01",
        end_date="2020-02-02",
    )
    frames = _read(result)
    assert set(frames) == {*FINANCIAL_FIELDS, "financial_first_disclosure"}
    data_id = next(
        (name for name, fields in FINANCIAL_FIELDS.items() if field in fields)
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
    pit = _prepare_pit_fundamentals(selected)(
        _pit_observations(
            pd.DataFrame({"asset_id": ["000001"] * 2, "timestamp": sessions[:2]})
        )
    )
    assert pit.value.iloc[0] == 100.0 and pd.isna(pit.value.iloc[1])


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
    identity = _file_hashes(layout.raw_source_root("choice"))
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("choice", key)
            for key in ("financial_balance",)
        },
        raw_root=layout.raw_source_root("choice"),
        data_ids=("financial_balance",),
        start_date="2019-01-01",
        end_date="2020-02-02",
    )
    frame = _read(result)["financial_balance"]
    assert len(frame) == 1 and frame.total_assets.iloc[0] == 0.0
    assert identity == _file_hashes(layout.raw_source_root("choice"))
    with pytest.raises(
        PortfolioTargetBlocked, match="choice_first_disclosure_snapshot_missing"
    ):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_income",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_income",),
        )


@pytest.mark.parametrize(
    "compact_period,compact_end", [(True, False), (False, True), (True, True)]
)
def test_request_date_formats_share_semantic_coordinates_and_preserve_raw(
    tmp_path, compact_period, compact_end
):
    layout = StorageLayout(tmp_path)
    period = "20191231" if compact_period else "2019-12-31"
    day = "20200130" if compact_end else "2020-01-30"
    with choice_session(SDK(response())) as session:
        first, first_request = _disclosure(session, layout)
        duplicate, duplicate_request = write_css(
            session,
            layout,
            period=period,
            end_date="20251231",
            indicators=DATE_FIELDS,
            data={CODE: ["2020/1/30", None, None]},
        )
        _snapshot(session, layout, day, [100.0] * len(FIELDS), period=period)
    assert (
        first != duplicate and first_request.identity() != duplicate_request.identity()
    )
    before = _file_hashes(layout.raw_source_root("choice"))
    identity = _file_hashes(layout.raw_source_root("choice"))
    frames = _read(
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in tuple(
                    spec.data_id for spec in interim.interim_catalog().list()
                )
            },
            raw_root=layout.raw_source_root("choice"),
        )
    )
    assert set(frames) == {*FINANCIAL_FIELDS, "financial_first_disclosure"}
    assert len(frames["financial_first_disclosure"]) == 1
    assert frames["financial_balance"].total_assets.tolist() == [100.0]
    assert frames["financial_balance"].report_period.tolist() == [
        pd.Timestamp("2019-12-31")
    ]
    assert identity == _file_hashes(layout.raw_source_root("choice"))
    assert _file_hashes(layout.raw_source_root("choice")) == before


def test_disclosure_and_financial_superset_is_one_raw_response(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        write_css(
            session,
            layout,
            period="2019-12-31",
            end_date="2020-01-30",
            indicators=(*DATE_FIELDS, *FIELDS),
            data={CODE: ["2020/1/30", None, None, *[100.0] * 14]},
        )
    frames = _read(
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in tuple(
                    spec.data_id for spec in interim.interim_catalog().list()
                )
            },
            raw_root=layout.raw_source_root("choice"),
        )
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
    frames = _read(
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_first_disclosure",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_first_disclosure",),
        )
    )
    assert frames["financial_first_disclosure"].first_disclosure_date.tolist() == [
        pd.Timestamp("2020-01-30")
    ]


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
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("choice", key)
            for key in tuple(spec.data_id for spec in interim.interim_catalog().list())
        },
        raw_root=layout.raw_source_root("choice"),
        start_date="2019-01-01",
        end_date="2020-02-02",
    )
    frames = _read(result)
    assert set(frames) == {*FINANCIAL_FIELDS, "financial_first_disclosure"}
    assert frames["financial_balance"].total_assets.tolist() == [100.0]
    for data_id in ("financial_balance", "financial_first_disclosure"):
        with pytest.raises(
            PortfolioTargetBlocked, match="choice_disclosure_field_missing"
        ) as caught:
            interim.materialize_interim_data(
                output_roots={
                    key: StorageLayout(tmp_path / "unscoped").interim_data_root(
                        "choice", key
                    )
                    for key in (data_id,)
                },
                raw_root=layout.raw_source_root("choice"),
                data_ids=(data_id,),
            )
        assert "report_period=2018-12-31" in " ".join(caught.value.__notes__)


@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("foreign", [None, "period", "end_date", "branch"])
def test_unfetched_field_is_blocked_and_other_coordinates_cannot_fill_it(
    tmp_path, financial, foreign
):
    layout = StorageLayout(tmp_path)
    fields = tuple(FINANCIAL_FIELDS["financial_balance"]) if financial else DATE_FIELDS
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
    before = _file_hashes(layout.raw_source_root("choice"))
    reason = (
        "choice_financial_snapshot_field_missing"
        if financial
        else "choice_disclosure_field_missing"
    )
    with pytest.raises(PortfolioTargetBlocked, match=reason) as caught:
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key) for key in (data_id,)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=(data_id,),
        )
    details = " ".join(caught.value.__notes__)
    assert any(field in details for field in fields)
    assert (
        "report_period=" in details and "EndDate=" in details and ("type=" in details)
    )
    if foreign is None:
        assert fields[-1] in details
    assert not filesystem_exists(layout.interim_data_root("choice", data_id))
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
        assert before != _file_hashes(layout.raw_source_root("choice"))
        frames = _read(
            interim.materialize_interim_data(
                output_roots={
                    key: layout.interim_data_root("choice", key) for key in (data_id,)
                },
                raw_root=layout.raw_source_root("choice"),
                data_ids=(data_id,),
            )
        )
        assert len(frames[data_id]) == 1
        if financial:
            assert (
                frames[data_id][list(FINANCIAL_FIELDS[data_id].values())]
                .isna()
                .all()
                .all()
            )


@pytest.mark.parametrize("financial", [False, True])
@pytest.mark.parametrize("null", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_overlapping_single_field_conflict_including_provider_null_is_rejected(
    tmp_path, financial, null, compact
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        if financial:
            _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
        write_css(
            session,
            layout,
            period="20191231" if compact else "2019-12-31",
            end_date=("20200130" if financial else "20251231")
            if compact
            else ("2020-01-30" if financial else "2025-12-31"),
            indicators=FIELDS[:1] if financial else DATE_FIELDS[:1],
            data={CODE: [None if null else 99.0 if financial else "2020/1/31"]},
        )
    with pytest.raises(
        ValueError,
        match="choice_snapshot_conflict"
        if financial
        else "choice_disclosure_field_conflict",
    ) as caught:
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
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
        _snapshot(session, layout, "2020-01-30", [100.0] * 14, branch="3")
        for branch, old, new in (("1", 100.0, 90.0), ("3", 100.0, 110.0)):
            _snapshot(session, layout, "2020-01-31", [old] * 14, branch=branch)
            _snapshot(session, layout, "2020-02-01", [new] * 14, branch=branch)
    with pytest.raises(ValueError, match="choice_financial_version_values_conflict"):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
    assert not filesystem_exists(
        layout.interim_data_root("choice", "financial_balance")
    )


@pytest.mark.parametrize("after", (150.0, 200.0))
def test_explicit_predecessor_must_match_known_branch_history(tmp_path, after):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01")
        for branch in ("1", "3"):
            _snapshot(session, layout, "2020-01-30", [100.0] * 14, branch=branch)
            _snapshot(session, layout, "2020-01-31", [150.0] * 14, branch=branch)
            _snapshot(session, layout, "2020-02-01", [after] * 14, branch=branch)
    with pytest.raises(
        PortfolioTargetBlocked,
        match="choice_revision_predecessor_conflicts_with_history",
    ):
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
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
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
    with choice_session(SDK(response())) as session:
        _snapshot(session, layout, "2020-01-31", [None] * 14, branch="3")
    frame = _read(
        interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("choice", key)
                for key in ("financial_balance",)
            },
            raw_root=layout.raw_source_root("choice"),
            data_ids=("financial_balance",),
        )
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
    with pytest.raises(PortfolioTargetBlocked, match=reason):
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={"financial_balance": tmp_path / "balance"},
            data_ids=("financial_balance",),
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
    from types import SimpleNamespace

    from quant_research_platform.catalog.processed import InterimDataCatalog
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_module,
    )

    processed_catalog = processed_module.processed_catalog
    from quant_research_platform.sources.csmar.interim import (
        interim_catalog as csmar_catalog,
    )

    source_specs = {
        spec.data_id: spec
        for catalog in (csmar_catalog(), interim.interim_catalog())
        for spec in catalog.list()
    }
    profile = SimpleNamespace(
        interim=InterimDataCatalog(source_specs.values()), processed=processed_catalog()
    )
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
            native = [values[name] for name in names]
            native[FIELDS.index("TINCOMESTATEMENT_60")] = 8.0
            native[FIELDS.index("TINCOMESTATEMENT_61")] = (
                6.0 if period == "2018-12-31" else 7.0
            )
            _snapshot(session, layout, first, native, period=period)
        _snapshot(session, layout, "2020-01-30", [None] * 14, branch="3")
        correction = [current[name] for name in names]
        correction[FIELDS.index("TINCOMESTATEMENT_60")] = 8.0
        correction[FIELDS.index("TINCOMESTATEMENT_61")] = None
        _snapshot(session, layout, "2020-02-01", correction)
        _snapshot(session, layout, "2020-02-01", [None] * 14, branch="3")
    imported = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("choice", key)
            for key in tuple(spec.data_id for spec in interim.interim_catalog().list())
        },
        raw_root=layout.raw_source_root("choice"),
        start_date="2018-01-01",
        end_date="2020-02-02",
    )
    persisted = _read(imported)
    original = (
        persisted["financial_income"]
        .loc[
            persisted["financial_income"].report_period.eq(pd.Timestamp("2019-12-31"))
            & persisted["financial_income"].version_identity.eq("original")
        ]
        .iloc[0]
    )
    assert (original.consolidated_net_income, original.parent_net_income) == (8.0, 7.0)
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
    )
    market = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["000001", "000001"],
            "timestamp": pd.to_datetime(["2020-01-31", "2020-02-03"]),
            "listing_date": pd.Timestamp("2010-01-01"),
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
            "listing_date": ColumnRole.CONTEXT,
            "total_market_cap": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
    )
    write_dataset(
        layout.processed_data_root("a_share", market_spec.kind, market_spec.data_id),
        market,
        spec=market_spec,
    )
    for data_id in ("fundamental_pit", "fundamental_factors"):
        spec = profile.processed.get(data_id)
        produced = producer.produce_a_share_processed_data(
            spec,
            inputs={
                **{
                    name: layout.interim_data_root(
                        profile.interim.get(name).source_id, name
                    )
                    for name in spec.source_input_data_ids
                },
                **{
                    name: layout.processed_data_root(
                        "a_share", profile.processed.get(name).kind, name
                    )
                    for name in spec.processed_input_data_ids
                },
            },
            time_window=TimeWindow("2018-01-01", "2020-02-04"),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        )
        root = layout.processed_data_root("a_share", spec.kind, data_id)
        with produced as (frames, metadata):
            write_dataset(
                root,
                frames,
                spec=spec,
                metadata=metadata,
            )
        persisted[data_id] = read_dataset(root)
    saved = read_dataset(root)
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
    assert saved.loc[0, list(FUNDAMENTAL_FACTORS)].tolist() == pytest.approx(expected)
    assert (
        saved.loc[1, ["earnings_yield", "roe", "earnings_change_to_assets"]]
        .isna()
        .all()
    )
    assert saved.loc[1, "roa"] == pytest.approx(8 / 90)
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
                session, layout, indicators=selected, data=selected_data, **kwargs
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


def test_bulk_import_shares_selected_reads_and_preserves_table_evidence(
    tmp_path, monkeypatch
):
    from collections import Counter

    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence

    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
    data_ids = (*FINANCIAL_FIELDS, "financial_first_disclosure")
    expected = {}
    for data_id in data_ids:
        target = tmp_path / "single" / data_id
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={data_id: target},
            data_ids=(data_id,),
        )
        expected[data_id] = (
            read_dataset(target),
            read_dataset_manifest_evidence(target)["source_evidence"],
            _file_hashes(target),
        )
    reads = Counter()
    original = interim.read_choice_response

    def counted(root, *args, **kwargs):
        reads[root] += 1
        return original(root, *args, **kwargs)

    monkeypatch.setattr(interim, "read_choice_response", counted)
    roots = {data_id: tmp_path / "bulk" / data_id for data_id in data_ids}
    result = interim.materialize_interim_data(
        raw_root=layout.raw_source_root("choice"), output_roots=roots, data_ids=data_ids
    )
    assert len(reads) == 2 and set(reads.values()) == {1}
    assert set(result.artifact_roots) == set(data_ids)
    for data_id, root in roots.items():
        frame, evidence, hashes = expected[data_id]
        pd.testing.assert_frame_equal(read_dataset(root), frame, check_exact=True)
        assert read_dataset_manifest_evidence(root)["source_evidence"] == evidence
        assert _file_hashes(tmp_path / "single" / data_id) == hashes


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
    payload = json.loads(
        Path(native_filesystem_path(sidecar)).read_text(encoding="utf-8")
    )
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
    with pytest.raises(ValueError):
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={"financial_first_disclosure": tmp_path / "invalid-output"},
            data_ids=("financial_first_disclosure",),
        )


def test_invalid_choice_metadata_preserves_unrelated_target_and_report(tmp_path):
    import json

    from tests.sources.reference.test_interim import _write_reference_root

    layout = StorageLayout(tmp_path)
    _write_reference_root(layout.project_root / "data/raw")
    with choice_session(SDK(response())) as session:
        root, _ = _disclosure(session, layout)
    sidecar = root / "response.metadata.json"
    payload = json.loads(
        Path(native_filesystem_path(sidecar)).read_text(encoding="utf-8")
    )
    del payload["metadata"]["request"]
    filesystem_write_text(sidecar, json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={"financial_first_disclosure": tmp_path / "disclosures"},
            data_ids=("financial_first_disclosure",),
        )
    from quant_research_platform.sources.reference.interim import (
        materialize_interim_data,
    )

    fees = tmp_path / "fees"
    materialize_interim_data(
        raw_root=layout.raw_source_root("reference"),
        output_roots={"a_share_fee_schedule": fees},
    )
    assert not read_dataset(fees).empty


@pytest.mark.parametrize(
    "error", [PermissionError("synthetic_permission"), OSError(5, "synthetic_io")]
)
@pytest.mark.parametrize("boundary", ["metadata", "directory"])
def test_choice_fatal_io_is_not_source_blocked(tmp_path, monkeypatch, error, boundary):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)

    def fail(_path, **_kwargs):
        raise error

    if boundary == "metadata":
        monkeypatch.setattr(interim, "read_qrp_json_file", fail)
    else:
        original = interim.filesystem_scandir

        def scan(path):
            if str(path) == str(layout.raw_source_root("choice") / "css"):
                raise error
            return original(path)

        monkeypatch.setattr(interim, "filesystem_scandir", scan)
    with pytest.raises(type(error)) as caught:
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots={"financial_first_disclosure": tmp_path / "disclosures"},
            data_ids=("financial_first_disclosure",),
            start_date="2019-01-01",
            end_date="2019-12-31",
        )
    assert caught.value is error


def test_requested_knowledge_cutoff_excludes_future_disclosures_and_missing_revisions(
    tmp_path,
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01")
        _disclosure(session, layout, period="2019-09-30", first="2020-02-03")
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
        _snapshot(session, layout, "2020-01-30", [None] * len(FIELDS), branch="3")
    arguments = dict(raw_root=layout.raw_source_root("choice"), start_date="2019-01-01")
    roots = {
        name: tmp_path / ("earlier-" + name)
        for name in ("financial_balance", "financial_first_disclosure")
    }
    interim.materialize_interim_data(
        **arguments, output_roots=roots, data_ids=tuple(roots), end_date="2020-01-31"
    )
    saved = {name: read_dataset(root) for name, root in roots.items()}
    assert saved["financial_balance"].total_assets.tolist() == [100.0]
    assert saved["financial_balance"].report_period.tolist() == [
        pd.Timestamp("2019-12-31")
    ]
    assert saved["financial_first_disclosure"].first_disclosure_date.tolist() == [
        pd.Timestamp("2020-01-30")
    ]
    old_hashes = {name: _file_hashes(root) for name, root in roots.items()}
    with pytest.raises(
        PortfolioTargetBlocked, match="choice_revision_snapshot_missing"
    ):
        interim.materialize_interim_data(
            **arguments,
            output_roots={"financial_balance": tmp_path / "missing-revision"},
            data_ids=("financial_balance",),
            end_date="2020-02-02",
        )
    interim.materialize_interim_data(
        **arguments,
        output_roots={"financial_first_disclosure": tmp_path / "later-disclosures"},
        data_ids=("financial_first_disclosure",),
        end_date="2020-02-02",
    )
    assert all((_file_hashes(root) == old_hashes[name] for name, root in roots.items()))
    with choice_session(SDK(response())) as session:
        for branch in ("1", "3"):
            _snapshot(
                session, layout, "2020-02-01", [200.0] * len(FIELDS), branch=branch
            )
    root = tmp_path / "complete-balance"
    interim.materialize_interim_data(
        **arguments,
        output_roots={"financial_balance": root},
        data_ids=("financial_balance",),
        end_date="2020-02-02",
    )
    frame = read_dataset(root)
    assert frame.total_assets.tolist() == [100.0, 200.0]
    assert frame.correction_disclosure_date.iloc[1] == pd.Timestamp("2020-02-01")
    assert all((_file_hashes(root) == old_hashes[name] for name, root in roots.items()))


def test_financial_tables_only_require_their_own_fields(tmp_path):
    layout = StorageLayout(tmp_path)
    balance_fields = tuple(FINANCIAL_FIELDS["financial_balance"])
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(
            session,
            layout,
            "2020-01-30",
            [100.0] * len(balance_fields),
            fields=balance_fields,
        )
    before = _file_hashes(layout.raw_source_root("choice"))
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("choice", key)
            for key in ("financial_balance",)
        },
        raw_root=layout.raw_source_root("choice"),
        data_ids=("financial_balance",),
    )
    frame = _read(result)["financial_balance"]
    assert len(frame) == 1 and frame.total_assets.iloc[0] == 100.0
    for data_id in ("financial_income", "financial_cashflow"):
        with pytest.raises(
            PortfolioTargetBlocked, match="choice_first_disclosure_snapshot_missing"
        ):
            interim.materialize_interim_data(
                output_roots={
                    key: layout.interim_data_root("choice", key) for key in (data_id,)
                },
                raw_root=layout.raw_source_root("choice"),
                data_ids=(data_id,),
            )
    assert _file_hashes(layout.raw_source_root("choice")) == before


@pytest.mark.parametrize(
    "period,day", [("2021-12-31", "2022-01-30"), ("2019-12-31", "2020-02-01")]
)
def test_unselected_financial_values_do_not_block_historical_window(
    tmp_path, period, day
):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout, events="2020-02-01")
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
    arguments = dict(
        raw_root=layout.raw_source_root("choice"),
        data_ids=("financial_balance",),
        start_date="2019-01-01",
        end_date="2020-01-31",
    )
    baseline = tmp_path / "baseline"
    interim.materialize_interim_data(
        **arguments, output_roots={"financial_balance": baseline}
    )
    with choice_session(SDK(response())) as session:
        if period != "2019-12-31":
            _disclosure(session, layout, period=period, first=day)
        _snapshot(
            session,
            layout,
            day,
            ["invalid-unused-numeric"] * len(FIELDS),
            period=period,
        )
    raw_hashes = _file_hashes(layout.raw_source_root("choice"))
    historical = tmp_path / "historical"
    interim.materialize_interim_data(
        **arguments, output_roots={"financial_balance": historical}
    )
    pd.testing.assert_frame_equal(
        read_dataset(historical), read_dataset(baseline), check_exact=True
    )
    assert _file_hashes(layout.raw_source_root("choice")) == raw_hashes
    with pytest.raises(ValueError, match="choice_financial_value_invalid"):
        interim.materialize_interim_data(
            **{**arguments, "end_date": day},
            output_roots={"financial_balance": tmp_path / "selected-invalid"},
        )


def test_bulk_financial_reports_remain_table_specific(tmp_path):
    layout = StorageLayout(tmp_path)
    with choice_session(SDK(response())) as session:
        _disclosure(session, layout)
        _snapshot(session, layout, "2020-01-30", [100.0] * len(FIELDS))
        _snapshot(
            session,
            layout,
            "2020-02-01",
            [200.0],
            period="2019-09-30",
            fields=("TCASHFLOWSTATEMENT_39",),
        )
    # An orphan cash-flow report must not enter the balance-sheet report index.
    roots = {
        name: tmp_path / name for name in ("financial_balance", "financial_cashflow")
    }
    with pytest.raises(PortfolioTargetBlocked, match="choice_first_disclosure_missing"):
        interim.materialize_interim_data(
            raw_root=layout.raw_source_root("choice"),
            output_roots=roots,
            data_ids=tuple(roots),
        )
    assert read_dataset(roots["financial_balance"]).total_assets.tolist() == [100.0]
    assert not filesystem_exists(roots["financial_cashflow"])
