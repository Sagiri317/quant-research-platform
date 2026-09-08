from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import interim, raw_exports, raw_schemas
from tests.helpers.runtime_resources import capture_test_policy


def _analyst_rows():
    return [
        {
            "Stkcd": "000001",
            "ReportID": report,
            "AnanmID": authors,
            "DeclareDate": declared,
            "Fenddt": end,
            "InstitutionID": institution,
            "Feps": eps,
            "Fpe": 10,
        }
        for report, authors, declared, end, institution, eps in (
            ("9007199254740993", "001,002", "2025-01-03", "2025-12-31", "A", 18),
            ("9007199254740992", "001,002", "2025-01-03", "2025-12-31", "A", 22),
            ("9007199254740993", "001,002", "2025-01-03", "2026-12-31", "A", 30),
            ("2", "", "2025-01-03", "2025-12-31", "B", 12),
            ("3", "003", "2025-01-03", "2025-12-31", "C", 13),
            ("4", "004", "", "2025-12-31", "D", 100),
            ("5", "005", "2025-01-03", "2025-12-31", "", 100),
            ("6", "006", " ", "2025-12-31", " ", 100),
        )
    ]


@pytest.mark.parametrize("chunksize", (1, 3, 100))
def test_analyst_raw_reports_persist_and_feed_institution_consensus(
    tmp_path, chunksize
):
    import json

    import numpy as np

    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows())
    raw_path = next(raw.glob("*.csv"))
    original = raw_path.read_bytes()
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("analyst_forecast",),
            chunksize=chunksize,
            start_date="2025-01-06",
            end_date="2025-01-07",
            resource_policy=policy,
        )
        root = result.artifact_roots["analyst_forecast"]
        saved = read_dataset(root, resource_policy=policy)
        manifest = json.loads((root / "_manifest/manifest.json").read_text())
        assert manifest["producer_diagnostics"]["source_row_counts"] == {
            "raw_rows": 8,
            "missing_declare_date_rows": 2,
            "missing_institution_id_rows": 2,
            "excluded_missing_required_rows": 3,
            "retained_before_date_filter_rows": 5,
            "retained_after_date_filter_rows": 5,
            "excluded_date_range_rows": 0,
        }
        spec = interim.interim_catalog().get("analyst_forecast")
        assert spec.primary_key == (
            "source_asset_id",
            "timestamp",
            "report_id",
            "forecast_end_date",
        )
        assert spec.column_roles["forecast_end_date"] is ColumnRole.KEY
        assert spec.column_roles["timestamp"] is ColumnRole.TIMESTAMP
        assert len(saved) == 5
        assert saved.report_id.str.startswith("900719925474099").sum() == 3
        assert saved.loc[saved.institution_id.eq("A"), "analyst_ids"].tolist() == [
            "001,002",
            "001,002",
            "001,002",
        ]
        assert saved.loc[saved.institution_id.eq("B"), "analyst_ids"].isna().all()
        assert saved.timestamp.eq(saved.declare_date).all()
        sessions = pd.bdate_range("2025-01-03", "2025-01-07").as_unit("ns")
        panel = pd.DataFrame(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": sessions,
                "close": 10.0,
            }
        )
        processed_spec = research_profile().processed.get("expectations_factors")
        actual = producer._compute_processed_frame(
            processed_spec,
            source={"analyst_forecast": saved},
            dependencies={"market_panel_1d": panel},
            trading_sessions=tuple(sessions),
        )
        processed_root = layout.processed_data_root(
            "a_share", "factor", "expectations_factors"
        )
        write_dataset(
            processed_root,
            actual,
            spec=processed_spec,
            resource_policy=policy,
            borrowed_input=True,
        )
        persisted = read_dataset(processed_root, resource_policy=policy)
        pd.testing.assert_frame_equal(persisted, actual, check_exact=True)
        assert persisted.iloc[0, 3:].isna().all()
        assert persisted.loc[1, "forward_earnings_yield_fy1"] == 1.3
        assert persisted.loc[1, "forecast_dispersion_fy1"] == 1.0
        assert persisted.loc[1, "log_analyst_coverage"] == pytest.approx(np.log(3))
        assert persisted.forward_earnings_yield_fy2.isna().all()
        assert policy.used_memory_bytes == 0
    assert raw_path.read_bytes() == original


def test_analyst_all_excluded_persists_zero_rows_and_counts(tmp_path):
    import json

    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows()[5:])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("analyst_forecast",),
            chunksize=1,
            resource_policy=policy,
        )
        root = result.artifact_roots["analyst_forecast"]
        empty = read_dataset(root, resource_policy=policy)
        assert empty.empty
        manifest = json.loads((root / "_manifest/manifest.json").read_text())
        counts = manifest["producer_diagnostics"]["source_row_counts"]
        assert counts["raw_rows"] == counts["excluded_missing_required_rows"] == 3
        assert counts["retained_after_date_filter_rows"] == 0
        sessions = pd.bdate_range("2025-01-03", periods=2).as_unit("ns")
        panel = pd.DataFrame(
            {
                "market_id": "a_share",
                "asset_id": "000001",
                "timestamp": sessions,
                "close": 10.0,
            }
        )
        with pytest.raises(ValueError, match="BLOCKED_EXTERNAL_DATA: analyst_forecast"):
            producer._compute_processed_frame(
                research_profile().processed.get("expectations_factors"),
                source={"analyst_forecast": empty},
                dependencies={"market_panel_1d": panel},
                trading_sessions=tuple(sessions),
            )
        assert policy.used_memory_bytes == 0


def test_analyst_duplicate_report_key_is_folded_and_counted_across_chunks(tmp_path):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    row = _analyst_rows()[0]
    _write_csmar(raw, "analyst_forecast", [row, row])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("analyst_forecast",),
            chunksize=1,
            resource_policy=policy,
        )
        root = result.artifact_roots["analyst_forecast"]
        assert len(read_dataset(root, resource_policy=policy)) == 1
        assert (
            read_dataset_manifest_evidence(root)["producer_diagnostics"][
                "exact_duplicate_count"
            ]
            == 1
        )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("column", ("Markettype", "Daywk"))
def test_calendar_direct_boolean_is_not_an_integer(column):
    row = {"Markettype": "4", "Clddt": "2025-01-03", "Daywk": "5", "State": "O"}
    row[column] = True
    with pytest.raises(ValueError, match="invalid_trading_calendar"):
        interim.normalize_raw_frame(pd.DataFrame([row]), "trading_calendar")


@pytest.mark.parametrize(
    "column,value",
    (
        ("DeclareDate", "invalid"),
        ("Fenddt", "invalid"),
        ("ReportID", ""),
        ("Feps", "infinite"),
    ),
)
def test_invalid_analyst_raw_preserves_published_dataset(tmp_path, column, value):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows()[:1])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout, data_ids=("analyst_forecast",), resource_policy=policy
        )
        root = result.artifact_roots["analyst_forecast"]
        before = read_dataset(root, resource_policy=policy)
        rows = _analyst_rows()[:1]
        rows[0][column] = value
        rows[0]["InstitutionID"] = ""
        _write_csmar(raw, "analyst_forecast", rows)
        with pytest.raises(ValueError, match="invalid_"):
            interim.materialize_interim_data(
                layout=StorageLayout(tmp_path / "candidate"),
                raw_root=raw,
                data_ids=("analyst_forecast",),
                resource_policy=policy,
            )
        pd.testing.assert_frame_equal(
            before, read_dataset(root, resource_policy=policy)
        )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "column,value",
    (
        ("Markettype", "1.9"),
        ("Daywk", "3.5"),
        ("Markettype", ""),
        ("Daywk", "inf"),
        ("Markettype", "True"),
    ),
)
def test_calendar_raw_rejects_nonintegers_and_preserves_old_good(
    tmp_path, column, value
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    row = {"Markettype": "4", "Clddt": "2025-01-03", "Daywk": "5", "State": "O"}
    _write_csmar(raw, "trading_calendar", [row])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout, data_ids=("trading_calendar",), resource_policy=policy
        )
        root = result.artifact_roots["trading_calendar"]
        before = read_dataset(root, resource_policy=policy)
        assert before.source_market_type.tolist() == [4]
        assert before.source_day_of_week.tolist() == [5]
        assert before.is_trading_day.tolist() == [True]
        row[column] = value
        _write_csmar(raw, "trading_calendar", [row])
        with pytest.raises(ValueError, match="invalid_"):
            interim.materialize_interim_data(
                layout=StorageLayout(tmp_path / "candidate"),
                raw_root=raw,
                data_ids=("trading_calendar",),
                resource_policy=policy,
            )
        pd.testing.assert_frame_equal(
            before, read_dataset(root, resource_policy=policy)
        )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("state", ("SYNTHETIC_PRIVATE_STATE", None))
def test_calendar_state_error_never_contains_source_row_value(state) -> None:
    frame = pd.DataFrame(
        {
            "calendar_date": ["2023-01-03"],
            "source_market_type": [1],
            "source_day_of_week": [2],
            "source_state": [state],
        }
    )
    with pytest.raises(ValueError) as raised:
        interim._normalize_trading_calendar_frame(frame)
    assert str(raised.value) == "invalid_trading_calendar_state"


@pytest.mark.parametrize(
    "market,day,filling,after,accepted",
    [
        (1, "2019-07-22", 0, None, True),
        (64, "2025-03-19", 0, None, True),
        (32, "2019-07-21", 0, None, True),
        (32, "2019-07-22", 0, None, False),
        (16, "2020-08-23", 0, None, True),
        (16, "2020-08-24", 0, None, False),
        (16, "2020-08-24", 1, None, True),
        (16, "2020-08-24", 2, None, True),
        (16, "2020-08-24", 3, None, False),
        (1, "2026-01-05", 0, None, False),
        (16, "2026-01-05", 0, 0.0, True),
        (32, "2020-01-02", 0, 2.0, True),
    ],
)
def test_after_hours_nulls_follow_known_historical_applicability(
    market, day, filling, after, accepted
) -> None:
    row = dict.fromkeys(
        raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
    )
    row.update(
        Stkcd="000001",
        Trddt=day,
        Markettype=market,
        Filling=filling,
        Ahshrtrd_D=after,
        Ahvaltrd_D=after,
    )
    if not accepted:
        with pytest.raises(
            ValueError, match="invalid_csmar_(after_hours|filling_flag)"
        ):
            interim.normalize_raw_frame(pd.DataFrame([row]), "market_bars_1d")
    else:
        result = interim.normalize_raw_frame(pd.DataFrame([row]), "market_bars_1d")
        expected = 0.0 if after is None else after
        assert (
            result.after_hours_volume.iloc[0]
            == result.after_hours_amount.iloc[0]
            == expected
        )
        assert result.volume.iloc[0] == result.amount.iloc[0] == 1.0 + expected


def test_after_hours_materialization_preserves_totals_and_old_good(tmp_path) -> None:
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    row = dict.fromkeys(
        raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
    )
    row.update(
        Stkcd="000001",
        Trddt="2019-08-01",
        Markettype=32,
        Filling=0,
        Ahshrtrd_D=2.0,
        Ahvaltrd_D=3.0,
    )
    _write_csmar(raw, "market_bars_1d", [row])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout, data_ids=("market_bars_1d",), resource_policy=policy
        )
        root = result.artifact_roots["market_bars_1d"]
        loaded = read_dataset(root, resource_policy=policy)
        assert loaded.volume.iloc[0] == 3.0 and loaded.amount.iloc[0] == 4.0
        before = {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }
        row["Ahvaltrd_D"] = None
        _write_csmar(raw, "market_bars_1d", [row])
        with pytest.raises(
            ValueError, match="invalid_csmar_after_hours_amount_missing"
        ):
            interim.materialize_interim_data(
                layout=StorageLayout(tmp_path / "candidate"),
                raw_root=raw,
                data_ids=("market_bars_1d",),
                resource_policy=policy,
            )
        assert {
            path.relative_to(root): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        } == before
        assert policy.used_memory_bytes == 0


def _collect_raw_export(
    path: Path,
    dataset: str,
    *,
    chunksize: int = 1024,
) -> pd.DataFrame:
    chunks = _collect_raw_chunks(path, dataset, chunksize=chunksize)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


@pytest.mark.parametrize("memory_cap", (None, 16 * 1024**2))
def test_index_weight_handoff_releases_raw_frame_before_persisting_normalized_rows(
    tmp_path, monkeypatch, memory_cap
):
    import weakref

    from quant_research_platform.storage.datasets import writing
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    root = layout.raw_source_root("csmar")
    root.mkdir(parents=True)
    sessions = pd.bdate_range("2019-03-01", periods=21 if memory_cap is None else 63)
    _write_csmar(
        root,
        "index_weights",
        (
            {
                "Indexcd": benchmark,
                "Enddt": timestamp.date().isoformat(),
                "Stkcd": f"{index:06}",
                "Constdnme": f"synthetic_{index:06}",
                "Weight": 1.0,
            }
            for benchmark in (
                ("000852", "000905")
                if memory_cap is None
                else ("000852", "000852", "000905")
            )
            for timestamp in sessions
            for index in range(1, 101)
        ),
    )
    references = []
    actual_frame = raw_exports._raw_frame
    actual_declared = writing._WriteStats.declared_frame

    def tracked_frame(*args):
        frame = actual_frame(*args)
        references.append(weakref.ref(frame))
        return frame

    def declared_frame(self, frame, **kwargs):
        assert references and all(reference() is None for reference in references)
        return actual_declared(self, frame, **kwargs)

    monkeypatch.setattr(raw_exports, "_raw_frame", tracked_frame)
    monkeypatch.setattr(writing._WriteStats, "declared_frame", declared_frame)
    with runtime_resource_policy_scope(explicit_cap_bytes=memory_cap) as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("index_weights",),
            resource_policy=policy,
        )
        saved = read_dataset(
            result.artifact_roots["index_weights"], resource_policy=policy
        )
        assert len(saved) == 200 * len(sessions)
        assert (
            saved.groupby(["source_asset_id", "timestamp"]).weight.sum().eq(1.0).all()
        )
        assert policy.used_memory_bytes == 0


def _collect_raw_chunks(
    path: Path,
    dataset: str,
    *,
    chunksize: int,
) -> list[pd.DataFrame]:
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    producer = raw_exports.iter_private_export_chunks_with_policy(
        path,
        dataset=dataset,
        chunksize=chunksize,
        resource_policy=policy,
    )
    chunks: list[pd.DataFrame] = []
    with runtime_resource_policy_scope(policy):
        try:
            bound = next(producer)
            while True:
                lease = policy.lease("test.csmar.consumer", bound)
                try:
                    try:
                        owned = producer.send(lease)
                    except StopIteration:
                        return chunks
                    if not isinstance(owned, OwnedValue):
                        raise AssertionError("invalid_bound_first_output")
                    chunks.append(owned.value)
                    try:
                        bound = producer.send(None)
                    except StopIteration:
                        return chunks
                finally:
                    lease.close()
        finally:
            producer.close()


def _financial_fixture_row(
    columns: tuple[str, ...],
    *,
    invalid_numeric_column: str | None = None,
) -> list[str]:
    return [
        (
            "000001"
            if column == "Stkcd"
            else "测试"
            if column == "ShortName"
            else "0"
            if column == "IfCorrect"
            else ""
            if column == "DeclareDate"
            else "2020-01-02"
            if column == "Accper"
            else "A"
            if column == "Typrep"
            else "bad"
            if column == invalid_numeric_column
            else "1"
        )
        for column in columns
    ]


def test_index_weights_raw_date_and_constituent_name_remain_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "csmar__index_weights__synthetic__full__export-20260101.csv"
    path.write_text(
        "Indexcd,Enddt,Stkcd,Constdnme,Weight\n000905,2019-01-02,000001,合成证券,1.0\n",
        encoding="utf-8-sig",
    )
    raw = _collect_raw_export(path, "index_weights")
    assert raw["Enddt"].tolist() == ["2019-01-02"]
    assert raw["Constdnme"].tolist() == ["合成证券"]
    normalized = interim.normalize_raw_frame(raw, "index_weights")
    assert normalized["effective_date"].tolist() == [pd.Timestamp("2019-01-02")]
    assert normalized["constituent_asset_id"].tolist() == ["000001"]
    assert normalized["weight"].tolist() == [0.01]


def test_raw_csv_batch_accounting_visits_row_sizes_only_linearly(tmp_path, monkeypatch):
    from quant_research_platform.policy.lifecycle import closing_iterator

    dates = pd.date_range("2020-01-01", periods=512).strftime("%Y-%m-%d").tolist()
    path = tmp_path / "csmar__trading_calendar__synthetic__full__export-20260101.csv"
    path.write_text(
        "Markettype,Clddt,Daywk,State\n"
        + "".join(f"4,{stamp},1,O\n" for stamp in dates),
        encoding="utf-8",
    )
    actual_records = raw_exports._iter_csv_records
    size_visits = 0

    class CountedRow(list):
        def __len__(self):
            nonlocal size_visits
            size_visits += 1
            return super().__len__()

    def observed_records(*args, **kwargs):
        with closing_iterator(actual_records(*args, **kwargs)) as records:
            for header, row, source_bytes in records:
                yield header, CountedRow(row), source_bytes

    monkeypatch.setattr(raw_exports, "_iter_csv_records", observed_records)
    result = _collect_raw_export(path, "trading_calendar", chunksize=len(dates))
    assert result["Clddt"].tolist() == dates
    assert size_visits <= 16 * len(dates), size_visits


def test_raw_export_discovery_contract_keeps_csv_only(
    tmp_path: Path,
) -> None:
    csv_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.csv"
    )
    xlsx_export = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.xlsx"
    )
    txt_description = tmp_path / (
        "csmar__trading_calendar__all_market_types__19901219_20261231__"
        "export-20260605.txt"
    )
    csv_export.write_text("Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n")
    xlsx_export.write_bytes(b"spreadsheet-bytes")
    txt_description.write_text("Clddt\tcalendar date\n", encoding="utf-8")

    assert raw_exports.SUPPORTED_RAW_EXPORT_SUFFIXES == {".csv"}
    assert [
        path
        for _dataset, path, _identity in raw_exports.dataset_files_with_identity_by_id(
            tmp_path
        )
    ] == [csv_export]
    assert raw_schemas.expected_metadata_sidecar_path(
        csv_export
    ) == csv_export.with_suffix(
        ".metadata.json",
    )
    assert not raw_schemas.inspect_raw_export_file(csv_export).has_metadata
    csv_export.with_suffix(".metadata.json").write_text("{}", encoding="utf-8")
    assert raw_schemas.inspect_raw_export_file(csv_export).has_metadata
    with pytest.raises(ValueError, match="Invalid CSMAR raw filename"):
        raw_schemas.parse_raw_filename(xlsx_export)
    with pytest.raises(ValueError, match="Cannot infer CSMAR dataset"):
        raw_schemas.inspect_raw_export_file(txt_description)


def test_csmar_source_identity_is_selected_content_only(tmp_path: Path) -> None:
    bars = tmp_path / (
        "csmar__market_bars_1d__all_market_types__full__export-20260605.csv"
    )
    unrelated = tmp_path / (
        "csmar__stock_master__all_market_types__full__export-20260605.csv"
    )
    bars.write_text(
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
        + "\n",
        encoding="utf-8",
    )
    unrelated.write_text(
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["stock_master"].required_columns)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="runtime_resource_policy_not_active"):
        interim.source_content_identity(tmp_path, "market_bars_1d")

    with runtime_resource_policy_scope(None) as policy:
        first = interim.source_content_identity(tmp_path, "market_bars_1d")
        assert first is not None

        sidecar = bars.with_suffix(".metadata.json")
        sidecar.write_text("{}", encoding="utf-8")
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first
        sidecar.write_text('{"row_count": 1}', encoding="utf-8")
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first

        unrelated.write_text(
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["stock_master"].required_columns)
            + "\n# unrelated\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first

        bars.write_text(
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
            + "\n# changed\n",
            encoding="utf-8",
        )
        second = interim.source_content_identity(tmp_path, "market_bars_1d")
        assert second is not None
        assert second != first

        renamed = tmp_path / "renamed-market-file.csv"
        bars.rename(renamed)
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == second
        bars = renamed

        shard = tmp_path / (
            "csmar__market_bars_1d__all_market_types__full__"
            "part-01__export-20260605.csv"
        )
        shard.write_text(
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
            + "\n",
            encoding="utf-8",
        )
        third = interim.source_content_identity(tmp_path, "market_bars_1d")
        assert third is not None
        assert third != second
        shard.unlink()
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == second
        assert policy.used_memory_bytes == 0


def test_csmar_source_identity_uses_only_one_recursive_root(tmp_path: Path) -> None:
    primary = tmp_path / "raw"
    nested = primary / "calendar" / "history"
    sibling = tmp_path / "sibling-root"
    nested.mkdir(parents=True)
    sibling.mkdir()
    columns = raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns
    (nested / "native-calendar.csv").write_text(
        ",".join(columns) + "\n",
        encoding="utf-8",
    )
    (sibling / "native-calendar.csv").write_text(
        ",".join(columns) + "\n",
        encoding="utf-8",
    )

    with runtime_resource_policy_scope(None) as policy:
        identity = interim.source_content_identity(primary, "trading_calendar")
        assert identity is not None
        raw_only = primary / "raw_only"
        raw_only.mkdir()
        raw_only_file = raw_only / "native-calendar.csv"
        raw_only_file.write_text(
            ",".join(columns) + "\n4,2020-01-03,5,C\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        raw_only_file.write_text(
            ",".join(columns) + "\n4,2020-01-04,6,O\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        raw_only_file.unlink()
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        assert interim.source_content_identity(sibling, "trading_calendar") is not None
        assert interim.source_content_identity(primary, "stock_master") is None
        assert policy.used_memory_bytes == 0
    assert not tuple(tmp_path.glob(".k-*/k.db"))


def test_csmar_identity_ignores_unrequested_bad_schema_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    calendar = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    calendar.write_text(
        "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / (
        "csmar__special_treatment_changes__all_market_types__full__export-20260605.csv"
    )
    unrelated.write_text("bad,header\n", encoding="utf-8")

    with runtime_resource_policy_scope(None) as policy:
        identity = interim.source_content_identity(tmp_path, "trading_calendar")
        assert identity is not None
        assert policy.used_memory_bytes == 0
    assert not tuple(tmp_path.parent.glob(".k-*/k.db"))


@pytest.mark.parametrize(
    "mutation",
    ("planned", "reading", "staged", "staged_added", "staged_removed", "unrelated"),
)
def test_csmar_materialization_binds_inspected_content_to_plan(
    tmp_path, monkeypatch, mutation
):
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from quant_research_platform.workflows import research

    layout = StorageLayout(tmp_path)
    source = tmp_path / "raw" / "csmar"
    source.mkdir(parents=True)
    calendar = source / "native-calendar.csv"
    original = "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n"
    calendar.write_text(original, encoding="utf-8")
    built = research.data_materialize(
        profile=research_profile(),
        layout=layout,
        raw_root=source.parent,
        start_date=None,
        end_date=None,
        data_ids=("trading_calendar",),
    )
    result = built["target_results"][0]
    assert result["status"] == "BUILT"
    root = layout.instance_root(
        layout.interim_data_root("csmar", "trading_calendar"),
        result["instance_identity"],
    )
    metadata = {
        "common_manifest": read_dataset_manifest_evidence(root)["common_manifest"]
    }
    old_good = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    target = StorageLayout(tmp_path / "candidate").interim_data_root(
        "csmar", "trading_calendar"
    )
    target.parent.mkdir(parents=True)
    unrelated = target.parent / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    previous_members = set(target.parent.iterdir())

    def mutate():
        if mutation == "staged_added":
            (source / "extra-calendar.csv").write_text(original, encoding="utf-8")
        elif mutation == "staged_removed":
            calendar.unlink()
        else:
            calendar.write_text(original.replace(",O", ",C"), encoding="utf-8")

    if mutation == "planned":
        mutate()
    elif mutation == "reading":
        actual_hash = interim.content_hash_for_files

        def hash_then_mutate(*args, **kwargs):
            identity = actual_hash(*args, **kwargs)
            mutate()
            return identity

        monkeypatch.setattr(interim, "content_hash_for_files", hash_then_mutate)
    elif mutation.startswith("staged"):
        actual_write = interim.write_dataset

        def write(*args, **kwargs):
            verify = kwargs.pop("staged_validator", None)

            def mutate_before_verify(receipt):
                mutate()
                if verify is not None:
                    verify(receipt)

            return actual_write(*args, **kwargs, staged_validator=mutate_before_verify)

        monkeypatch.setattr(interim, "write_dataset", write)
    else:
        (
            source / "csmar__stock_master__all_market_types__full__export-20260605.csv"
        ).write_text("bad,header\n", encoding="utf-8")

    with runtime_resource_policy_scope() as policy:

        def materialize():
            return interim.materialize_interim_data(
                layout=layout.with_target_roots({"trading_calendar": target}),
                raw_root=source,
                data_ids=("trading_calendar",),
                start_date=None,
                end_date=None,
                instance_metadata=metadata,
                resource_policy=policy,
            )

        if mutation == "unrelated":
            materialize()
            pd.testing.assert_frame_equal(
                read_dataset(target, resource_policy=policy),
                read_dataset(root, resource_policy=policy),
                check_exact=True,
            )
            assert (
                read_dataset_manifest_evidence(target)["common_manifest"]
                == metadata["common_manifest"]
            )
            assert set(target.parent.iterdir()) == previous_members | {target}
        else:
            with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
                materialize()
            assert set(target.parent.iterdir()) == previous_members
            assert not target.exists()
        assert policy.used_memory_bytes == 0
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == old_good
    assert unrelated.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "dataset",
    ("financial_balance", "financial_income", "financial_cashflow"),
)
def test_financial_raw_decoder_preserves_text_and_normalizes_values(
    tmp_path: Path,
    dataset: str,
) -> None:
    path = tmp_path / f"{dataset}.csv"
    columns = raw_exports.RAW_TABLE_CONTRACTS[dataset].required_columns
    row = _financial_fixture_row(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow(row)

    decoded = _collect_raw_export(path, dataset, chunksize=1)
    normalized = interim.normalize_raw_frame(decoded, dataset)

    assert normalized.loc[0, "stock_name"] == "测试"
    assert normalized.loc[0, "report_type"] == "A"
    assert normalized.loc[0, "report_period"] == pd.Timestamp("2020-01-02")


@pytest.mark.parametrize(
    "dataset",
    ("financial_balance", "financial_income", "financial_cashflow"),
)
def test_financial_raw_decoder_rejects_invalid_numeric_fact(
    tmp_path: Path,
    dataset: str,
) -> None:
    path = tmp_path / f"{dataset}.csv"
    columns = raw_exports.RAW_TABLE_CONTRACTS[dataset].required_columns
    numeric_column = next(
        column
        for column in columns
        if column
        not in {
            "Stkcd",
            "ShortName",
            "IfCorrect",
            "DeclareDate",
            "Accper",
            "Typrep",
        }
    )
    row = _financial_fixture_row(
        columns,
        invalid_numeric_column=numeric_column,
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerow(row)

    with pytest.raises(ValueError, match="invalid_csmar_numeric_value"):
        _collect_raw_export(path, dataset, chunksize=1)


def test_provider_header_bound_accepts_160_and_rejects_161_fields(
    tmp_path: Path,
) -> None:
    required = raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns
    for width in (160, 161):
        columns = (
            *required,
            *(f"Optional{index}" for index in range(width - len(required))),
        )
        path = tmp_path / f"provider-{width}.csv"
        path.write_text(",".join(columns) + "\n", encoding="utf-8")
        if width == 160:
            inspection = raw_schemas.inspect_raw_export_file(path)
            assert inspection.columns == columns
        else:
            with pytest.raises(ValueError, match="csmar_raw_export_schema_invalid"):
                raw_schemas.inspect_raw_export_file(path)


def test_financial_correction_date_order_is_validated() -> None:
    result = interim._parse_financial_correction_date(
        pd.Series(["2020-02-01,2020-01-02", "2020-01-03", ""])
    )

    assert result.tolist() == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-01-03"),
        pd.NaT,
    ]
    with pytest.raises(ValueError, match="invalid_financial_correction_date"):
        interim._parse_financial_correction_date(
            pd.Series(["2020-01-01,2020-02-01,2020-01-15"])
        )


def test_csmar_identity_rejects_requested_bad_schema_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    requested = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    requested.write_text("bad,header\n", encoding="utf-8")

    with runtime_resource_policy_scope(None) as policy:
        with pytest.raises(ValueError, match="csmar_raw_export_"):
            interim.source_content_identity(tmp_path, "trading_calendar")
        assert policy.used_memory_bytes == 0
    assert not tuple(tmp_path.parent.glob(".k-*/k.db"))


def test_dirty_index_info_csv_audit_counts_and_semantics(tmp_path: Path) -> None:
    index_path = tmp_path / "index_info_dirty.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,-9666,all shares,"
        '1,2,3,4,5,"formula, English comma, 中文标点"\n'
        "000300,large,2014-01-03,2013-12-31,1000,blue chips,"
        ',4,7,5,9,"formula with embedded line\nbreak and ""quotes"""\n'
        "000905,mid,2014-01-06,2013-12-31,500,text continues\n"
        "over physical line,1,2,3,4,5,formula after merge\n"
        "000999,bad-enum,2014-01-07,2013-12-31,600,scope,8,9,10,6,10,formula\n",
        encoding="utf-8",
    )

    audit = _audit_index_info_csv(index_path)
    raw = _collect_raw_export(index_path, "index_info")
    normalized = interim.normalize_raw_frame(raw, "index_info")
    pruned, _ = interim._apply_output_column_rules_validated(
        normalized,
        "index_info",
        overrides=interim._validated_column_overrides({}),
    )

    assert audit == {
        "metadata_row_count": 1,
        "physical_line_count": 7,
        "record_start_count": 4,
        "logical_records": 4,
        "merged_continuation_rows": 2,
        "ambiguous_rows": 0,
        "malformed_rows": 0,
        "dedup_rows": 0,
        "final_rows": 4,
        "out_of_enum_counts": {
            "Idxinfo05": 1,
            "Idxinfo06": 1,
            "Idxinfo07": 1,
            "Idxinfo08": 1,
            "Idxinfo09": 1,
        },
    }
    by_asset = pruned.set_index("source_asset_id")
    assert pd.isna(by_asset.loc["000985", "base_point"])
    assert by_asset.loc["000300", "start_date"] == pd.Timestamp("2014-01-03")
    assert by_asset.loc["000300", "base_date"] == pd.Timestamp("2013-12-31")
    assert "embedded line\nbreak" in by_asset.loc[
        "000300",
        "calculation_formula",
    ].replace("\r\n", "\n")
    assert by_asset.loc["000905", "calculation_formula"] == "formula after merge"
    assert pd.isna(by_asset.loc["000999", "weighting_method"])
    assert pd.isna(by_asset.loc["000999", "publisher"])
    assert pd.isna(by_asset.loc["000999", "index_category"])
    assert pd.isna(by_asset.loc["000999", "index_type"])
    assert pd.isna(by_asset.loc["000999", "market_belonging"])


def test_ambiguous_index_info_long_row_fails_closed(tmp_path: Path) -> None:
    index_path = tmp_path / "index_info_ambiguous.csv"
    index_path.write_text(
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,1000,scope,"
        "1,2,3,4,5,ambiguous,1,2,3,4,5,tail\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        _collect_raw_export(index_path, "index_info")


def test_index_info_parser_streams_bounded_chunks_without_a_full_frame_surface(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index_info.csv"
    rows = [
        {
            "Indexcd": f"00030{position}",
            "Idxinfo01": f"index-{position}",
            "Idxinfo11": "2014-01-02",
            "Idxinfo02": "2013-12-31",
            "Idxinfo03": "1000",
            "Idxinfo04": "scope",
            "Idxinfo05": "1",
            "Idxinfo06": "2",
            "Idxinfo07": "3",
            "Idxinfo08": "4",
            "Idxinfo09": "5",
            "Idxinfo10": 'formula, with "quotes"',
        }
        for position in range(3)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)

    chunks = _collect_raw_chunks(path, "index_info", chunksize=1)

    assert [len(chunk) for chunk in chunks] == [1, 1, 1]
    assert pd.concat(chunks, ignore_index=True)["Indexcd"].tolist() == [
        "000300",
        "000301",
        "000302",
    ]


def test_read_export_header_preserves_keyboard_interrupt_over_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("header_interrupted")
    close_calls = 0
    path = tmp_path / "export.csv"
    path.write_text('"quoted",header\n', encoding="utf-8")
    original_close = raw_schemas._BoundedCsvRecords.close

    def interrupted(_self):
        raise primary

    def failing_close(self):
        nonlocal close_calls
        close_calls += 1
        original_close(self)
        raise OSError("header_close_failed")

    monkeypatch.setattr(raw_schemas._BoundedCsvRecords, "_parse_record", interrupted)
    monkeypatch.setattr(raw_schemas._BoundedCsvRecords, "close", failing_close)
    with pytest.raises(KeyboardInterrupt) as raised:
        raw_schemas.read_export_header(path)
    assert raised.value is primary
    assert close_calls == 1


@pytest.mark.parametrize("text_width", (1, 16_384))
def test_non_index_raw_iteration_bounds_requested_chunks_across_text_widths(
    tmp_path: Path,
    text_width: int,
) -> None:
    path = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "Markettype": ["4"] * 5,
            "Clddt": pd.date_range("2020-01-01", periods=5).astype(str),
            "Daywk": ["3"] * 5,
            "State": ["x" * text_width] * 5,
        }
    ).to_csv(path, index=False)

    chunks = _collect_raw_chunks(path, "trading_calendar", chunksize=4)

    assert all(0 < len(chunk) <= 4 for chunk in chunks)
    assert sum(map(len, chunks)) == 5


def _audit_index_info_csv(path: Path) -> dict[str, object]:
    physical_line_count = len(path.read_text(encoding="utf-8").splitlines())
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    logical_rows = [row for _, row, _ in raw_exports.iter_index_info_rows(path)]
    raw = _collect_raw_export(path, "index_info")
    enum_specs = {
        "Idxinfo05": frozenset({"1", "2", "3", "4"}),
        "Idxinfo06": frozenset({"1", "2", "3", "4"}),
        "Idxinfo07": frozenset({"1", "2", "3", "4", "5", "6", "7"}),
        "Idxinfo08": frozenset({"1", "2", "3", "4", "5"}),
        "Idxinfo09": frozenset({"1", "2", "3", "4", "5", "6", "7", "8", "9"}),
    }
    header_index = {name: index for index, name in enumerate(header)}
    out_of_enum_counts = {}
    for column, allowed in enum_specs.items():
        index = header_index[column]
        out_of_enum_counts[column] = sum(
            1
            for row in logical_rows
            if row[index].strip() != "" and row[index].strip() not in allowed
        )
    return {
        "metadata_row_count": 1,
        "physical_line_count": physical_line_count,
        "record_start_count": sum(
            int(raw_exports._looks_like_index_info_record_start(row)) for row in rows
        ),
        "logical_records": len(raw),
        "merged_continuation_rows": sum(1 for row in rows if len(row) < len(header)),
        "ambiguous_rows": 0,
        "malformed_rows": 0,
        "dedup_rows": 0,
        "final_rows": len(raw),
        "out_of_enum_counts": out_of_enum_counts,
    }


def test_index_market_preserves_nullable_official_open_without_close_substitution():
    raw = pd.DataFrame(
        {
            "Indexcd": ["H00905", "H00852"],
            "Idxtrd01": ["2023-01-03"] * 2,
            "Idxtrd02": ["123.5", None],
            "Idxtrd03": ["125"] * 2,
            "Idxtrd04": ["120"] * 2,
            "Idxtrd05": ["124"] * 2,
            "Idxtrd06": ["1"] * 2,
            "Idxtrd07": ["1"] * 2,
            "Idxtrd08": ["0.5"] * 2,
            "Idxtrd09": ["fixture"] * 2,
        }
    )
    actual = interim.normalize_raw_frame(raw, "index_market").set_index(
        "source_asset_id"
    )
    assert actual.loc["H00905", "open"] == 123.5
    assert pd.isna(actual.loc["H00852", "open"])
    assert "open" in interim.INDEX_RETURN_INTERIM_COLUMNS


@pytest.mark.parametrize("width", (32, 154, 160))
def test_legal_provider_width_uses_bounded_unquoted_fast_path(
    tmp_path, monkeypatch, width
):
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    path = tmp_path / "legal.csv"
    fields = [f"field_{position}" for position in range(width)]
    path.write_text(
        ",".join(fields) + "\n" + ",".join(["1"] * width) + "\n", encoding="utf-8"
    )

    def forbidden(_self):
        raise AssertionError("legal_unquoted_row_entered_byte_parser")

    monkeypatch.setattr(raw_schemas._BoundedCsvRecords, "_parse_record", forbidden)
    with runtime_resource_policy_scope() as policy:
        records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        try:
            header, _ = next(records)
            assert raw_schemas.materialize_header_record(header) == fields
            record, _ = next(records)
            assert record.materialize(max_fields=160) == ["1"] * width
            with pytest.raises(StopIteration):
                next(records)
        finally:
            records.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "data_id", ("market_bars_1d", "analyst_forecast", "trading_calendar")
)
def test_csmar_exact_duplicates_and_unordered_chunks_persist_identical_content(
    tmp_path, data_id
):
    import hashlib

    from quant_research_platform.storage.datasets.comparison import compare_datasets
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    if data_id == "analyst_forecast":
        source = _analyst_rows()
        source[0]["AnanmID"] = "张三,李四"
        source[1]["Feps"] = ""
    elif data_id == "trading_calendar":
        source = [
            {"Markettype": 1, "Clddt": day, "Daywk": weekday, "State": "O"}
            for day, weekday in (("2020-01-02", 4), ("2020-01-03", 5))
        ]
    else:
        row = dict.fromkeys(
            raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns, 1.0
        )
        row.update(
            Stkcd="000001", Markettype=1, Filling=0, Ahshrtrd_D=0.0, Ahvaltrd_D=0.0
        )
        source = [{**row, "Trddt": day} for day in ("2020-01-02", "2020-01-03")]
    rows = [*reversed(source), dict(source[0])]
    roots, frames, hashes = [], [], []
    spec = interim.interim_catalog().get(data_id)
    for size in (1, 2, 8):
        layout = StorageLayout(tmp_path / str(size))
        raw = layout.raw_source_root("csmar")
        raw.mkdir(parents=True)
        _write_csmar(raw, data_id, rows)
        hashes.append(hashlib.sha256(next(raw.glob("*.csv")).read_bytes()).hexdigest())
        with runtime_resource_policy_scope() as policy:
            result = interim.materialize_interim_data(
                layout=layout,
                data_ids=(data_id,),
                chunksize=size,
                resource_policy=policy,
            )
            root = result.artifact_roots[data_id]
            roots.append(root)
            saved = read_dataset(root, resource_policy=policy)
            frames.append(
                saved.sort_values(list(spec.primary_key)).reset_index(drop=True)
            )
            metadata = read_dataset_manifest_evidence(root)["producer_diagnostics"]
            assert metadata["exact_duplicate_count"] == 1
            if data_id == "analyst_forecast":
                counts = metadata["source_row_counts"]
                assert counts["raw_rows"] == len(rows)
                assert counts["excluded_missing_required_rows"] + len(saved) + metadata[
                    "exact_duplicate_count"
                ] == len(rows)
                assert {"9007199254740992", "9007199254740993"}.issubset(
                    saved.report_id
                )
                assert "张三,李四" in set(saved.analyst_ids.dropna())
                assert saved.forecast_eps.isna().any()
            assert str(saved.timestamp.dtype) == "datetime64[ns]"
            assert not saved.duplicated(list(spec.primary_key)).any()
            assert policy.used_memory_bytes == 0
    assert len(set(hashes)) == 1
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], frame, check_exact=True)
    with runtime_resource_policy_scope() as policy:
        for root in roots[1:]:
            assert (
                compare_datasets(roots[0], root, resource_policy=policy).status
                == "MATCH"
            )
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("chunksize", (1, 2))
def test_csmar_conflicting_duplicate_keeps_prior_published_dataset(tmp_path, chunksize):
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    first = _analyst_rows()[0]
    _write_csmar(raw, "analyst_forecast", [first])
    with runtime_resource_policy_scope() as policy:
        original = interim.materialize_interim_data(
            layout=layout, data_ids=("analyst_forecast",), resource_policy=policy
        ).artifact_roots["analyst_forecast"]
        old_good = {
            path.relative_to(original): path.read_bytes()
            for path in original.rglob("*")
            if path.is_file()
        }
        _write_csmar(raw, "analyst_forecast", [first, {**first, "Feps": 999}])
        candidate = layout.instance_root(original, "f" * 64)
        with pytest.raises(
            ValueError, match="conflicting_csmar_primary_key_duplicates"
        ):
            interim.materialize_interim_data(
                layout=layout.with_target_roots({"analyst_forecast": candidate}),
                data_ids=("analyst_forecast",),
                chunksize=chunksize,
                resource_policy=policy,
            )
        assert not candidate.exists()
        assert {
            path.relative_to(original): path.read_bytes()
            for path in original.rglob("*")
            if path.is_file()
        } == old_good
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cancel", (False, True))
def test_csmar_injected_capacity_preserves_pending_duplicates_next_batch_and_cleanup(
    tmp_path, monkeypatch, cancel
):
    from quant_research_platform.runtime.resources import (
        RuntimeResourcePolicy,
        pandas_frame_payload_bytes,
    )
    from quant_research_platform.storage.datasets import writing
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    row = {
        "Indexcd": "000852",
        "Enddt": "2019-03-01",
        "Stkcd": "000001",
        "Constdnme": "synthetic",
        "Weight": 1.0,
    }
    _write_csmar(
        raw,
        "index_weights",
        [row, row, {**row, "Stkcd": "000002"}, {**row, "Stkcd": "000003"}],
    )
    transform = interim._transform_interim_chunk
    observed = []
    injected = []

    def transform_with_one_capacity_failure(frame, **kwargs):
        if not injected:
            injected.append(True)
            size = pandas_frame_payload_bytes(frame)
            raise interim._InterimCapacityExceeded(2 * size, 2 * size)
        observed.append(len(frame))
        return transform(frame, **kwargs)

    pending = []
    create_lease = RuntimeResourcePolicy.lease

    def record_pending(policy, owner, reserved_bytes):
        lease = create_lease(policy, owner, reserved_bytes)
        if owner == "source.csmar.interim.pending":
            pending.append(lease)
        return lease

    primary = KeyboardInterrupt("pending_consumer_cancel")
    with runtime_resource_policy_scope(explicit_cap_bytes=16 * 1024**2) as policy:
        original = interim.materialize_interim_data(
            layout=layout,
            data_ids=("index_weights",),
            chunksize=2,
            resource_policy=policy,
        ).artifact_roots["index_weights"]
        old_good = {
            path.relative_to(original): path.read_bytes()
            for path in original.rglob("*")
            if path.is_file()
        }
        candidate = layout.instance_root(original, "e" * 64)
        monkeypatch.setattr(
            interim, "_transform_interim_chunk", transform_with_one_capacity_failure
        )
        monkeypatch.setattr(RuntimeResourcePolicy, "lease", record_pending)
        if cancel:

            def fail_consumer(_self, _frame, **_kwargs):
                raise primary

            monkeypatch.setattr(writing._WriteStats, "declared_frame", fail_consumer)
        request = dict(
            layout=layout.with_target_roots({"index_weights": candidate}),
            data_ids=("index_weights",),
            chunksize=2,
            resource_policy=policy,
        )
        if cancel:
            with pytest.raises(KeyboardInterrupt) as caught:
                interim.materialize_interim_data(**request)
            assert caught.value is primary
            assert not candidate.exists()
        else:
            interim.materialize_interim_data(**request)
            saved = read_dataset(candidate, resource_policy=policy)
            assert saved.constituent_asset_id.tolist() == ["000001", "000002", "000003"]
            assert observed == [1, 1, 2]
            assert (
                read_dataset_manifest_evidence(candidate)["producer_diagnostics"][
                    "exact_duplicate_count"
                ]
                == 1
            )
        assert pending and all(not lease.active for lease in pending)
        assert {
            relative: (original / relative).read_bytes() for relative in old_good
        } == old_good
        assert policy.used_memory_bytes == 0
