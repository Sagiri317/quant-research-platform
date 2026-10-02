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
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_rename,
    filesystem_unlink,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from tests.helpers.runtime_resources import capture_test_policy


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): filesystem_read_bytes(parent / name)
        for parent, _directories, files in filesystem_walk(root)
        for name in files
    }


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
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows())
    raw_path = next(filesystem_glob(raw, "*.csv"))
    original = filesystem_read_bytes(raw_path)
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
        manifest = json.loads(
            filesystem_read_text(root / "_manifest/manifest.json", encoding="utf-8")
        )
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
    assert filesystem_read_bytes(raw_path) == original


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
    filesystem_mkdir(raw, parents=True)
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
        manifest = json.loads(
            filesystem_read_text(root / "_manifest/manifest.json", encoding="utf-8")
        )
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
    filesystem_mkdir(raw, parents=True)
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
    filesystem_mkdir(raw, parents=True)
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
    filesystem_mkdir(raw, parents=True)
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
    filesystem_mkdir(raw, parents=True)
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
        before = _file_bytes(root)
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
        assert _file_bytes(root) == before
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
    filesystem_mkdir(root, parents=True)
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
    filesystem_write_text(
        path,
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
    filesystem_write_text(
        path,
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
    filesystem_write_text(
        csv_export, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
    )
    filesystem_write_bytes(xlsx_export, b"spreadsheet-bytes")
    filesystem_write_text(txt_description, "Clddt\tcalendar date\n", encoding="utf-8")

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
    filesystem_write_text(
        csv_export.with_suffix(".metadata.json"), "{}", encoding="utf-8"
    )
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
    filesystem_write_text(
        bars,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
        + "\n",
        encoding="utf-8",
    )
    filesystem_write_text(
        unrelated,
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
        filesystem_write_text(sidecar, "{}", encoding="utf-8")
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first
        filesystem_write_text(sidecar, '{"row_count": 1}', encoding="utf-8")
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first

        filesystem_write_text(
            unrelated,
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["stock_master"].required_columns)
            + "\n# unrelated\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == first

        filesystem_write_text(
            bars,
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
            + "\n# changed\n",
            encoding="utf-8",
        )
        second = interim.source_content_identity(tmp_path, "market_bars_1d")
        assert second is not None
        assert second != first

        renamed = tmp_path / "renamed-market-file.csv"
        filesystem_rename(bars, renamed)
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == second
        bars = renamed

        shard = tmp_path / (
            "csmar__market_bars_1d__all_market_types__full__"
            "part-01__export-20260605.csv"
        )
        filesystem_write_text(
            shard,
            ",".join(raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns)
            + "\n",
            encoding="utf-8",
        )
        third = interim.source_content_identity(tmp_path, "market_bars_1d")
        assert third is not None
        assert third != second
        filesystem_unlink(shard)
        assert interim.source_content_identity(tmp_path, "market_bars_1d") == second
        assert policy.used_memory_bytes == 0


def test_csmar_source_identity_uses_only_one_recursive_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(tmp_path)
    )
    primary = tmp_path / "raw"
    nested = primary / "calendar" / "history"
    sibling = tmp_path / "sibling-root"
    filesystem_mkdir(nested, parents=True)
    filesystem_mkdir(sibling)
    columns = raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns
    filesystem_write_text(
        nested / "native-calendar.csv",
        ",".join(columns) + "\n",
        encoding="utf-8",
    )
    filesystem_write_text(
        sibling / "native-calendar.csv",
        ",".join(columns) + "\n",
        encoding="utf-8",
    )

    with runtime_resource_policy_scope(None) as policy:
        identity = interim.source_content_identity(primary, "trading_calendar")
        assert identity is not None
        raw_only = primary / "raw_only"
        filesystem_mkdir(raw_only)
        raw_only_file = raw_only / "native-calendar.csv"
        filesystem_write_text(
            raw_only_file,
            ",".join(columns) + "\n4,2020-01-03,5,C\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        filesystem_write_text(
            raw_only_file,
            ",".join(columns) + "\n4,2020-01-04,6,O\n",
            encoding="utf-8",
        )
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        filesystem_unlink(raw_only_file)
        assert interim.source_content_identity(primary, "trading_calendar") == identity
        assert interim.source_content_identity(sibling, "trading_calendar") is not None
        assert interim.source_content_identity(primary, "stock_master") is None
        assert policy.used_memory_bytes == 0
    assert not tuple(filesystem_glob(tmp_path, ".k-*/k.db"))


def test_csmar_identity_ignores_unrequested_bad_schema_and_cleans_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir",
        lambda: str(tmp_path.parent),
    )
    calendar = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    filesystem_write_text(
        calendar,
        "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n",
        encoding="utf-8",
    )
    unrelated = tmp_path / (
        "csmar__special_treatment_changes__all_market_types__full__export-20260605.csv"
    )
    filesystem_write_text(unrelated, "bad,header\n", encoding="utf-8")

    with runtime_resource_policy_scope(None) as policy:
        identity = interim.source_content_identity(tmp_path, "trading_calendar")
        assert identity is not None
        assert policy.used_memory_bytes == 0
    assert not tuple(filesystem_glob(tmp_path.parent, ".k-*/k.db"))


def test_csmar_source_identity_streams_inventory_with_accounted_members(
    tmp_path, monkeypatch
):
    import hashlib
    import sys

    from quant_research_platform.storage import identity as identity_owner

    scratch = tmp_path / "scratch"
    filesystem_mkdir(scratch)
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir", lambda: str(scratch)
    )
    real_hash = identity_owner._hash_file_content
    real_content_hash = interim.content_hash_for_files
    reservations = []

    def hash_member(*args, **kwargs):
        metadata = sys._getframe(1).f_locals["metadata"]
        reservations.append(metadata.reserved_bytes)
        return real_hash(*args, **kwargs)

    def hash_stream(files, **kwargs):
        assert iter(files) is files
        return real_content_hash(files, **kwargs)

    monkeypatch.setattr(identity_owner, "_hash_file_content", hash_member)
    monkeypatch.setattr(interim, "content_hash_for_files", hash_stream)
    payload = b"Markettype,Clddt,Daywk,State\n"
    peaks = []
    for count in (1, 64):
        raw = tmp_path / str(count)
        filesystem_mkdir(raw)
        for index in range(count):
            filesystem_write_bytes(
                raw / f"csmar__trading_calendar__p{index}__full__export-20260101.csv",
                payload,
            )
        reservations.clear()
        with runtime_resource_policy_scope(explicit_cap_bytes=8 * 1024**2) as policy:
            actual = interim.source_content_identity(raw, "trading_calendar")
            assert policy.used_memory_bytes == 0
        expected = hashlib.sha256()
        for _ in range(count):
            for value in ("trading_calendar", hashlib.sha256(payload).hexdigest()):
                encoded = value.encode("utf-8")
                expected.update(len(encoded).to_bytes(8, "big"))
                expected.update(encoded)
        assert actual == expected.hexdigest()
        assert len(reservations) == count
        peaks.append(max(reservations))
        assert not tuple(filesystem_iterdir(scratch))
    assert peaks[1] > peaks[0] * 16


def test_csmar_materialization_replays_each_dataset_from_one_inventory(
    tmp_path, monkeypatch
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    for index in range(2):
        shard = raw / str(index)
        filesystem_mkdir(shard, parents=True)
        stamp = f"2020-01-0{index + 2}"
        _write_csmar(
            shard,
            "trading_calendar",
            [{"Markettype": 4, "Clddt": stamp, "Daywk": index + 4, "State": "O"}],
        )
        _write_csmar(
            shard,
            "index_weights",
            [
                {
                    "Indexcd": "000905",
                    "Enddt": stamp,
                    "Stkcd": "000001",
                    "Constdnme": "合成",
                    "Weight": 1,
                }
            ],
        )
    data_ids = ("trading_calendar", "index_weights")
    single = StorageLayout(tmp_path / "single")
    combined = StorageLayout(tmp_path / "combined")
    real_content_hash = interim.content_hash_for_files
    calls = 0

    def hash_stream(files, **kwargs):
        nonlocal calls
        assert iter(files) is files
        calls += 1
        return real_content_hash(files, **kwargs)

    monkeypatch.setattr(interim, "content_hash_for_files", hash_stream)
    with runtime_resource_policy_scope() as policy:
        expected = {}
        for data_id in data_ids:
            result = interim.materialize_interim_data(
                layout=single, raw_root=raw, data_ids=(data_id,), resource_policy=policy
            )
            expected[data_id] = read_dataset(
                result.artifact_roots[data_id], resource_policy=policy
            )
        result = interim.materialize_interim_data(
            layout=combined, raw_root=raw, data_ids=data_ids, resource_policy=policy
        )
        assert set(result.artifact_roots) == set(data_ids)
        for data_id in data_ids:
            actual = read_dataset(
                result.artifact_roots[data_id], resource_policy=policy
            )
            pd.testing.assert_frame_equal(actual, expected[data_id], check_exact=True)
            assert len(actual) == 2
        assert policy.used_memory_bytes == 0
    # Each selected dataset is hashed before reading and again before publication.
    assert calls == 8


@pytest.mark.parametrize("member_count", (1, 32))
@pytest.mark.parametrize(
    "mutation",
    ("planned", "reading", "staged", "staged_added", "staged_removed", "unrelated"),
)
def test_csmar_materialization_binds_inspected_content_to_plan(
    tmp_path, monkeypatch, mutation, member_count
):
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from quant_research_platform.workflows import research

    layout = StorageLayout(tmp_path)
    source = tmp_path / "raw" / "csmar"
    filesystem_mkdir(source, parents=True)
    calendar = source / "native-calendar.csv"
    original = "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n"
    filesystem_write_text(calendar, original, encoding="utf-8")
    for index in range(member_count - 1):
        filesystem_write_text(
            source / f"calendar-{index}.csv",
            original.splitlines()[0] + "\n",
            encoding="utf-8",
        )
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
    old_good = _file_bytes(root)
    target = StorageLayout(tmp_path / "candidate").interim_data_root(
        "csmar", "trading_calendar"
    )
    filesystem_mkdir(target.parent, parents=True)
    unrelated = target.parent / "keep.txt"
    filesystem_write_text(unrelated, "keep", encoding="utf-8")
    previous_members = set(filesystem_iterdir(target.parent))

    def mutate():
        if mutation == "staged_added":
            filesystem_write_text(
                source / "extra-calendar.csv", original, encoding="utf-8"
            )
        elif mutation == "staged_removed":
            filesystem_unlink(calendar)
        else:
            filesystem_write_text(
                calendar, original.replace(",O", ",C"), encoding="utf-8"
            )

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
        filesystem_write_text(
            source / "csmar__stock_master__all_market_types__full__export-20260605.csv",
            "bad,header\n",
            encoding="utf-8",
        )

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
            assert set(filesystem_iterdir(target.parent)) == previous_members | {target}
        else:
            with pytest.raises(ValueError, match="csmar_raw_export_identity_changed"):
                materialize()
            assert set(filesystem_iterdir(target.parent)) == previous_members
            assert not filesystem_exists(target)
        assert policy.used_memory_bytes == 0
    assert _file_bytes(root) == old_good
    assert filesystem_read_text(unrelated, encoding="utf-8") == "keep"


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
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as handle:
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
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as handle:
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
        filesystem_write_text(path, ",".join(columns) + "\n", encoding="utf-8")
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "quant_research_platform.storage.exact_keys.gettempdir",
        lambda: str(tmp_path.parent),
    )
    requested = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    filesystem_write_text(requested, "bad,header\n", encoding="utf-8")

    with runtime_resource_policy_scope(None) as policy:
        with pytest.raises(ValueError, match="csmar_raw_export_"):
            interim.source_content_identity(tmp_path, "trading_calendar")
        assert policy.used_memory_bytes == 0
    assert not tuple(filesystem_glob(tmp_path.parent, ".k-*/k.db"))


def test_dirty_index_info_csv_audit_counts_and_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    scratch = tmp_path / "scratch"
    filesystem_mkdir(scratch)
    monkeypatch.setattr(
        raw_schemas.tempfile, "tempdir", native_filesystem_path(scratch)
    )
    from quant_research_platform.storage import _scratch_ownership

    temporary_file = _scratch_ownership.TemporaryFile
    handles = []

    def open_scratch(*args, **kwargs):
        assert Path(raw_schemas.tempfile.gettempdir()) == Path(
            native_filesystem_path(scratch)
        )
        handle = temporary_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(_scratch_ownership, "TemporaryFile", open_scratch)
    index_path = tmp_path / "index_info_dirty.csv"
    filesystem_write_text(
        index_path,
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

    assert len(handles) >= 2 and all(handle.closed for handle in handles)
    assert not list(filesystem_iterdir(scratch))

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


@pytest.mark.parametrize(
    "tail,expected",
    [
        ("scope,extra,1,2,3,4,5,formula,tail", ("scope,extra", "formula,tail")),
        ("scope,1,2,3,4,5,ambiguous,1,2,3,4,5,tail", None),
    ],
)
def test_index_info_long_row_requires_one_unambiguous_field_split(
    tmp_path: Path, tail, expected
) -> None:
    index_path = tmp_path / "index_info_long.csv"
    filesystem_write_text(
        index_path,
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,1000," + tail + "\n",
        encoding="utf-8",
    )
    if expected is None:
        with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
            _collect_raw_export(index_path, "index_info")
    else:
        actual = _collect_raw_export(index_path, "index_info")
        assert len(actual) == 1
        assert (actual.iloc[0]["Idxinfo04"], actual.iloc[0]["Idxinfo10"]) == expected


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
    pd.DataFrame(rows).to_csv(native_filesystem_path(path), index=False)

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
    filesystem_write_text(path, '"quoted\nvalue",header\n', encoding="utf-8")
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
    ).to_csv(native_filesystem_path(path), index=False)

    chunks = _collect_raw_chunks(path, "trading_calendar", chunksize=4)

    assert all(0 < len(chunk) <= 4 for chunk in chunks)
    assert sum(map(len, chunks)) == 5


def _audit_index_info_csv(path: Path) -> dict[str, object]:
    physical_line_count = len(filesystem_read_text(path, encoding="utf-8").splitlines())
    with open_filesystem_file(path, "r", encoding="utf-8", newline="") as handle:
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


@pytest.mark.parametrize("quoted", [False, True])
@pytest.mark.parametrize("width", (32, 154, 160))
def test_legal_provider_width_uses_bounded_fast_path(
    tmp_path, monkeypatch, width, quoted
):
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    path = tmp_path / "legal.csv"
    fields = [f"field_{position}" for position in range(width)]
    if quoted:
        fields[0] = '字段,"示例"'
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle, quoting=csv.QUOTE_ALL if quoted else csv.QUOTE_MINIMAL
        )
        writer.writerows((fields, ["1"] * width))

    def forbidden(_self):
        raise AssertionError("legal_csv_row_entered_byte_parser")

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
        filesystem_mkdir(raw, parents=True)
        _write_csmar(raw, data_id, rows)
        hashes.append(
            hashlib.sha256(
                filesystem_read_bytes(next(filesystem_glob(raw, "*.csv")))
            ).hexdigest()
        )
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
    filesystem_mkdir(raw, parents=True)
    first = _analyst_rows()[0]
    _write_csmar(raw, "analyst_forecast", [first])
    with runtime_resource_policy_scope() as policy:
        original = interim.materialize_interim_data(
            layout=layout, data_ids=("analyst_forecast",), resource_policy=policy
        ).artifact_roots["analyst_forecast"]
        old_good = _file_bytes(original)
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
        assert not filesystem_exists(candidate)
        assert _file_bytes(original) == old_good
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
    filesystem_mkdir(raw, parents=True)
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
        old_good = _file_bytes(original)
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
            assert not filesystem_exists(candidate)
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
            relative: filesystem_read_bytes(original / relative)
            for relative in old_good
        } == old_good
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("primary_type", (MemoryError, OSError, KeyboardInterrupt))
def test_csmar_inspection_keeps_fatal_source_failure(
    tmp_path, monkeypatch, primary_type
):
    path = tmp_path / "native-calendar.csv"
    filesystem_write_text(
        path, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
    )
    primary = primary_type("primary")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(raw_exports, "inspect_raw_export_file", fail)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(primary_type) as caught:
            interim.source_content_identity(tmp_path, "trading_calendar")
        assert caught.value is primary
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "code,st,pt",
    [
        ("AB", True, False),
        ("AD", True, False),
        ("BC", False, True),
        ("BA", False, False),
        ("AS", None, None),
        ("DT", None, None),
        ("TX", None, None),
    ],
)
def test_special_treatment_destination_state(code, st, pt):
    result = interim.normalize_raw_frame(
        pd.DataFrame(
            {
                "Symbol": ["000001"],
                "Chgtype": [code],
                "Annoudt": ["2020-01-02"],
                "Execudt": ["2020-01-01"],
            }
        ),
        "special_treatment_changes",
    )
    assert result["change_type"].iloc[0] == code
    for column, expected in (("is_st", st), ("is_pt", pt)):
        assert str(result[column].dtype) == "boolean"
        if expected is None:
            assert pd.isna(result[column].iloc[0])
        else:
            assert bool(result[column].iloc[0]) is expected


@pytest.mark.parametrize("code", ["A", "B", "ZZ", "ABA", "ab"])
def test_special_treatment_rejects_non_provider_transition(code):
    with pytest.raises(ValueError, match="invalid_special_treatment_change_type"):
        interim.normalize_raw_frame(
            pd.DataFrame(
                {
                    "Symbol": ["000001"],
                    "Chgtype": [code],
                    "Annoudt": ["2020-01-02"],
                    "Execudt": ["2020-01-01"],
                }
            ),
            "special_treatment_changes",
        )


@pytest.mark.parametrize("chunksize,bad_numeric", ((1, False), (8, False), (1, True)))
def test_index_membership_raw_text_survives_public_materialization(
    tmp_path, chunksize, bad_numeric
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    rows = [
        {
            "Indexcd": "000905",
            "Chgsmp01": effective,
            "Chgsmp02": code,
            "Chgsmp03": name,
            "Chgsmp04": "invalid" if bad_numeric else "1",
            "Chgsmp05": "1",
            "Chgsmp06": announced,
            "Chgsmp07": "SZSE",
        }
        for effective, code, name, announced in (
            ("2020-01-06", "000001", "示例甲", "2020-01-03"),
            ("2020-01-07", "000002", "示例乙", "2020-01-04"),
        )
    ]
    _write_csmar(raw, "index_membership", rows)
    path = next(filesystem_glob(raw, "*.csv"))
    original = filesystem_read_bytes(path)
    with runtime_resource_policy_scope() as policy:
        if bad_numeric:
            with pytest.raises(ValueError, match="invalid_csmar_numeric_value"):
                interim.materialize_interim_data(
                    layout=layout,
                    raw_root=raw,
                    data_ids=("index_membership",),
                    chunksize=chunksize,
                    resource_policy=policy,
                )
        else:
            result = interim.materialize_interim_data(
                layout=layout,
                raw_root=raw,
                data_ids=("index_membership",),
                chunksize=chunksize,
                resource_policy=policy,
            )
            frame = read_dataset(
                result.artifact_roots["index_membership"], resource_policy=policy
            )
            frame = frame.sort_values("constituent_stock_code")
            assert frame.constituent_stock_code.tolist() == ["000001", "000002"]
            assert frame.constituent_asset_id.nunique() == 2
            assert frame.constituent_name.tolist() == ["示例甲", "示例乙"]
            assert frame.market_code.tolist() == ["SZSE", "SZSE"]
            assert frame.effective_date.dt.strftime("%Y-%m-%d").tolist() == [
                "2020-01-06",
                "2020-01-07",
            ]
            assert frame.announcement_date.dt.strftime("%Y-%m-%d").tolist() == [
                "2020-01-03",
                "2020-01-04",
            ]
        assert policy.used_memory_bytes == 0
    assert filesystem_read_bytes(path) == original


@pytest.mark.parametrize("chunksize", [1, 10])
def test_index_membership_preserves_native_codes_and_venue_type_identity(
    tmp_path, chunksize
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    identities = [
        ("000001", "SSE", 1),
        ("000001", "SZSE", 1),
        ("000001", "SSE", 3),
        ("123456789", "CFETS", 3),
        ("DEMO", "NASDAQ", 1),
        ("DEMO", "", 1),
    ]
    _write_csmar(
        raw,
        "index_membership",
        [
            {
                "Indexcd": "source_index",
                "Chgsmp01": "2020-01-06",
                "Chgsmp02": code,
                "Chgsmp03": "Synthetic",
                "Chgsmp04": "1",
                "Chgsmp05": str(kind),
                "Chgsmp06": "2020-01-03",
                "Chgsmp07": market,
            }
            for code, market, kind in identities
        ],
    )
    with runtime_resource_policy_scope() as policy:
        imported = interim.materialize_interim_data(
            layout=layout,
            raw_root=raw,
            data_ids=("index_membership",),
            chunksize=chunksize,
            resource_policy=policy,
        )
        saved = read_dataset(
            imported.artifact_roots["index_membership"], resource_policy=policy
        )
        actual = {
            (code, "" if pd.isna(market) else market, int(kind))
            for code, market, kind in saved[
                ["constituent_stock_code", "market_code", "security_type"]
            ].itertuples(index=False, name=None)
        }
        assert actual == set(identities)
        assert len(saved) == saved.constituent_asset_id.nunique() == len(identities)
        assert saved.market_code.isna().sum() == 1
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("kind", [1, 1.0, None])
def test_index_membership_identity_uses_normalized_provider_types(kind):
    row = {
        "Indexcd": "source_index",
        "Chgsmp01": "2020-01-06",
        "Chgsmp02": "CODE123",
        "Chgsmp03": "Synthetic",
        "Chgsmp04": 1,
        "Chgsmp05": kind,
        "Chgsmp06": "2020-01-03",
        "Chgsmp07": None,
    }
    actual = interim.normalize_raw_frame(pd.DataFrame([row]), "index_membership")
    baseline = interim.normalize_raw_frame(
        pd.DataFrame([{**row, "Chgsmp05": None if kind is None else 1}]),
        "index_membership",
    )
    assert actual.constituent_asset_id.equals(baseline.constituent_asset_id)
    assert actual.market_code.isna().all()
    assert actual.constituent_stock_code.iloc[0] == "CODE123"
    if kind is None:
        assert actual.security_type.isna().all()
    else:
        assert actual.security_type.iloc[0] == 1
        assert pd.api.types.is_integer_dtype(actual.security_type.dtype)


@pytest.mark.parametrize("size", [46080, 460800])
@pytest.mark.parametrize("suffix", ["", "中", "中😀"])
@pytest.mark.parametrize("finish", ["normal", "early", "invalid"])
@pytest.mark.parametrize("quoted", [False, True])
def test_csv_unicode_fast_heap_is_covered_by_live_leases(
    tmp_path, size, suffix, finish, quoted
):
    import os
    import subprocess
    import sys
    from textwrap import dedent

    # tracemalloc is process-wide: earlier tests can make this parser's first
    # scratch path trigger a multi-MiB resize of Python's global intern table.
    root = Path(__file__).resolve().parents[2]
    script = dedent(
        """
        import sys
        from pathlib import Path
        import pytest
        from tests.correctness.test_raw_csmar_semantics import (
            _assert_csv_unicode_heap_coverage,
        )

        with pytest.MonkeyPatch.context() as monkeypatch:
            _assert_csv_unicode_heap_coverage(
                Path(sys.argv[1]), monkeypatch, int(sys.argv[2]),
                sys.argv[3], sys.argv[4], sys.argv[5] == "True",
            )
        """
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(tmp_path),
            str(size),
            suffix,
            finish,
            str(quoted),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(root / "src"), str(root))),
            **{
                key: "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "BLIS_NUM_THREADS",
                )
            },
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr


def _assert_csv_unicode_heap_coverage(
    tmp_path, monkeypatch, size, suffix, finish, quoted
):
    import tempfile
    import tracemalloc

    import psutil

    from quant_research_platform.runtime.resources import MemoryLease
    from tests.helpers.runtime_resources import capture_test_policy

    scratch = tmp_path / "scratch"
    filesystem_mkdir(scratch)
    monkeypatch.setenv("TMPDIR", str(native_filesystem_path(scratch)))
    monkeypatch.setattr(tempfile, "tempdir", None)
    expected = ["a" * size + suffix, "b" * size + suffix]
    encoded_fields = [f'"{value}"' for value in expected] if quoted else expected
    payload = (",".join(encoded_fields) + "\n").encode()
    if finish == "invalid":
        payload = payload[:-1] + b"\xff\n"
    path = tmp_path / "unicode.csv"
    filesystem_write_bytes(path, payload)
    process = psutil.Process()
    descriptors = (
        process.num_fds if hasattr(process, "num_fds") else process.num_handles
    )
    fd_before, threads_before = descriptors(), process.num_threads()
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        shrink = MemoryLease.shrink_to

        def shrink_after_covered_peak(lease, actual_bytes):
            if lease.owner != "source.csmar.csv.record":
                return shrink(lease, actual_bytes)
            assert tracemalloc.get_traced_memory()[1] <= policy.used_memory_bytes
            shrink(lease, actual_bytes)
            assert tracemalloc.get_traced_memory()[0] <= policy.used_memory_bytes
            tracemalloc.reset_peak()

        monkeypatch.setattr(MemoryLease, "shrink_to", shrink_after_covered_peak)
        records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        tracemalloc.start()
        try:
            if finish == "invalid":
                with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
                    next(records)
            else:
                record, raw_bytes = next(records)
                assert raw_bytes == len(payload)
                actual = record.materialize(max_fields=160)
                peak = tracemalloc.get_traced_memory()[1]
                # Input/expected are allocated before tracing. RSS is deliberately
                # not substituted for the Python heap covered by these owners.
                assert peak <= policy.used_memory_bytes
                assert actual == expected
                if finish == "normal":
                    del actual, record
                    with pytest.raises(StopIteration):
                        next(records)
        finally:
            records.close()
            tracemalloc.stop()
        assert policy.used_memory_bytes == 0
    # Earlier test workers can retire while this parser runs on Windows.
    assert descriptors() <= fd_before
    assert all(Path(item.path).name != path.name for item in process.open_files())
    assert process.num_threads() <= threads_before
    assert list(filesystem_iterdir(scratch)) == []


def test_csv_unicode_insufficient_budget_fails_before_decode_and_releases(tmp_path):
    from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
    from tests.helpers.runtime_resources import capture_test_policy

    path = tmp_path / "unicode.csv"
    filesystem_write_text(
        path, "a" * 460800 + "😀," + "b" * 460800 + "😀\n", encoding="utf-8"
    )
    with runtime_resource_policy_scope(
        capture_test_policy(
            available_memory_bytes=64 * 1024**2, explicit_cap_bytes=4 * 1024**2
        )
    ) as policy:
        records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        try:
            with pytest.raises(DatasetReadCapacityExceeded):
                next(records)
        finally:
            records.close()
        assert policy.used_memory_bytes == 0


def test_expired_csv_records_do_not_retain_previous_unicode_payload(tmp_path):
    import tracemalloc

    from tests.helpers.runtime_resources import capture_test_policy

    text = "a" * 460800 + "😀," + "b" * 460800 + "😀\n"
    path = tmp_path / "records.csv"
    filesystem_write_text(path, text * 3, encoding="utf-8")
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        rows = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        expired = []
        tracemalloc.start()
        try:
            for record, _ in rows:
                assert tracemalloc.get_traced_memory()[1] <= policy.used_memory_bytes
                fields = record.iter_fields()
                next(fields)
                expired.append((record, fields))
            assert tracemalloc.get_traced_memory()[0] < 128 * 1024
            for record, fields in expired:
                with pytest.raises(RuntimeError, match="csmar_csv_record_expired"):
                    record.materialize()
                with pytest.raises(RuntimeError, match="csmar_csv_record_expired"):
                    next(fields)
        finally:
            rows.close()
            tracemalloc.stop()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "failure_type", (FileNotFoundError, NotADirectoryError, OSError, PermissionError)
)
def test_csmar_parent_failure_keeps_workflow_semantics(
    tmp_path, monkeypatch, failure_type
):
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows import research

    raw = tmp_path / "raw/csmar"
    filesystem_mkdir(raw, parents=True)
    source = raw / "native-calendar.csv"
    filesystem_write_text(
        source, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
    )
    arguments = dict(
        profile=research_profile(),
        raw_root=raw.parent,
        start_date="2020-01-01",
        end_date="2020-01-03",
        data_ids=("trading_calendar",),
    )
    normal_layout = StorageLayout(tmp_path / "normal")
    failed_layout = StorageLayout(tmp_path / "failed")
    with runtime_resource_policy_scope() as policy:
        result = research.data_materialize(layout=normal_layout, **arguments)
        assert result["status"] == "COMPLETE"
        originals = _file_bytes(normal_layout.project_root)
        primary = failure_type("parent_failure")
        native_check = interim.require_unaliased_path
        native_stat = interim.filesystem_stat
        armed = False
        calls = []

        def checked(path, *, error):
            nonlocal armed
            result = native_check(path, error=error)
            if error == "csmar_interim_source_parent_invalid":
                armed = True
            return result

        def failed_stat(path, *args, **kwargs):
            nonlocal armed
            if armed and path == raw:
                armed = False
                calls.append(path)
                raise primary
            return native_stat(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(interim, "require_unaliased_path", checked)
            patch.setattr(interim, "filesystem_stat", failed_stat)
            if failure_type in (FileNotFoundError, NotADirectoryError):
                failed = research.data_materialize(layout=failed_layout, **arguments)
                assert failed["status"] == "FAILED"
                assert failed["target_results"][0]["reason"] == (
                    "interim_build_failed:csmar_interim_source_parent_invalid"
                )
            else:
                with pytest.raises(failure_type) as caught:
                    research.data_materialize(layout=failed_layout, **arguments)
                assert caught.value is primary
        assert calls == [raw]
        assert policy.used_memory_bytes == 0
        assert _file_bytes(normal_layout.project_root) == originals
        assert not tuple(filesystem_glob(failed_layout.project_root, "**/.a-*"))
        assert not tuple(
            filesystem_glob(failed_layout.project_root, "**/manifest.json")
        )


@pytest.mark.parametrize("exponent", (0, 600, -600))
def test_adjusted_open_preserves_units_through_storage_and_labels(tmp_path, exponent):
    import math

    import numpy as np

    from quant_research_platform.research.labels.forward_returns import (
        add_forward_return_labels,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    sessions = pd.date_range("2019-08-01", periods=3)
    prices = [math.ldexp(multiplier, exponent) for multiplier in (1.0, 1.0, 1.5)]
    rows = []
    for timestamp, price in zip(sessions, prices, strict=True):
        row = dict.fromkeys(
            raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
        )
        row.update(
            Stkcd="000001",
            Trddt=timestamp.date().isoformat(),
            Markettype=32,
            Filling=0,
            Opnprc=price,
            Clsprc=price,
            Adjprcwd=price,
            Hiprc=price,
            Loprc=price,
        )
        rows.append(row)
    _write_csmar(raw, "market_bars_1d", rows)
    with runtime_resource_policy_scope() as policy:
        built = interim.materialize_interim_data(
            layout=layout, data_ids=("market_bars_1d",), resource_policy=policy
        )
        loaded = read_dataset(
            built.artifact_roots["market_bars_1d"], resource_policy=policy
        )
        # open == close == adjusted close: the exact adjusted open is that price.
        np.testing.assert_array_equal(loaded.adj_open, loaded.open)
        panel = loaded.rename(columns={"source_asset_id": "asset_id"}).assign(
            can_enter=True
        )
        labeled = add_forward_return_labels(panel, horizons=(1,), sessions=sessions)
        assert labeled.forward_return_1d.iloc[0] == pytest.approx(0.5)
        assert labeled.forward_return_1d_available_at.iloc[0] == sessions[2]
        assert labeled.forward_return_1d.iloc[1:].isna().all()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "opening,adjusted,closing",
    [
        (2.0**600, 2.0**-600, 2.0**-600),
        (2.0**-600, 2.0**600, 2.0**600),
        (2.0**600, 2.0**600, 2.0**600),
        (2.0**-600, 2.0**-600, 2.0**-600),
    ],
)
def test_adjusted_open_avoids_intermediate_product_and_ratio_loss(
    opening, adjusted, closing
):
    from fractions import Fraction

    row = dict.fromkeys(
        raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
    )
    row.update(
        Stkcd="000001",
        Trddt="2019-08-01",
        Markettype=32,
        Filling=0,
        Opnprc=opening,
        Clsprc=closing,
        Adjprcwd=adjusted,
    )
    expected = float(Fraction(opening) * Fraction(adjusted) / Fraction(closing))
    actual = interim.normalize_raw_frame(pd.DataFrame([row]), "market_bars_1d")
    assert actual.adj_open.iloc[0] == expected


@pytest.mark.parametrize(
    "opening,adjusted,closing,expected",
    [
        (1.0, 1.0, 0.0, None),
        (1.0, 1.0, None, None),
        (None, 1.0, 1.0, None),
        (1.0, None, 1.0, None),
        (0.0, 1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0, 0.0),
    ],
)
def test_adjusted_open_keeps_missing_and_zero_close_semantics(
    opening, adjusted, closing, expected
):
    row = dict.fromkeys(
        raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
    )
    row.update(
        Stkcd="000001",
        Trddt="2019-08-01",
        Markettype=32,
        Filling=0,
        Opnprc=opening,
        Clsprc=closing,
        Adjprcwd=adjusted,
    )
    actual = interim.normalize_raw_frame(
        pd.DataFrame([row]), "market_bars_1d"
    ).adj_open.iloc[0]
    if expected is None:
        assert pd.isna(actual)
    else:
        assert actual == expected


def test_csmar_pending_copy_admits_payload_before_concurrent_allocation(
    tmp_path, monkeypatch
):
    import sys
    import threading

    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(
        raw,
        "index_weights",
        [
            dict(
                Indexcd="000852",
                Enddt="2019-03-01",
                Stkcd=f"{i + 1:06d}",
                Constdnme="synthetic",
                Weight=25.0,
            )
            for i in range(4)
        ],
    )
    transform = interim._transform_interim_chunk
    classify = ExactPrimaryKeyIndex.classify_comparison_table
    copy = pd.DataFrame.copy
    workers, releases, worker_errors, copies, refusals = [], [], [], [], []
    armed = False
    first = True
    with runtime_resource_policy_scope(explicit_cap_bytes=16 * 1024**2) as policy:

        def hold_remaining_capacity():
            ready, release = threading.Event(), threading.Event()

            def worker():
                _payload = None
                lease = None
                try:
                    size = (
                        policy.explicit_memory_cap_bytes - policy.used_memory_bytes - 1
                    )
                    lease = policy.lease("test.concurrent_payload", size)
                    _payload = bytearray(size)
                    ready.set()
                    if not release.wait(10):
                        raise AssertionError("concurrent_payload_timeout")
                except BaseException as error:
                    worker_errors.append(type(error).__name__)
                    ready.set()
                finally:
                    _payload = None
                    if lease is not None:
                        lease.close()

            thread = threading.Thread(target=worker)
            workers.append(thread)
            releases.append(release)
            thread.start()
            assert ready.wait(10)
            assert not worker_errors
            return release, thread

        def transform_under_pressure(frame, **kwargs):
            nonlocal first, armed
            if not first:
                return transform(frame, **kwargs)
            first = False
            release, thread = hold_remaining_capacity()
            try:
                return transform(frame, **kwargs)
            except interim._InterimCapacityExceeded:
                refusals.append(True)
                armed = True
                raise
            finally:
                release.set()
                thread.join(10)
                assert not thread.is_alive()

        def classify_then_contend(self, *args, **kwargs):
            nonlocal armed
            result = classify(self, *args, **kwargs)
            if armed:
                armed = False
                hold_remaining_capacity()
            return result

        def observe_copy(frame, *args, **kwargs):
            caller = sys._getframe(1)
            if (
                caller.f_code.co_name == "_iter_pruned_interim_frames"
                and caller.f_locals.get("next_pending_lease") is not None
            ):
                copies.append(len(frame))
            return copy(frame, *args, **kwargs)

        monkeypatch.setattr(
            interim, "_transform_interim_chunk", transform_under_pressure
        )
        monkeypatch.setattr(
            ExactPrimaryKeyIndex, "classify_comparison_table", classify_then_contend
        )
        monkeypatch.setattr(pd.DataFrame, "copy", observe_copy)
        try:
            with pytest.raises(MemoryError, match="^runtime_memory_budget_exceeded$"):
                interim.materialize_interim_data(
                    layout=layout,
                    data_ids=("index_weights",),
                    chunksize=2,
                    resource_policy=policy,
                )
        finally:
            for release in releases:
                release.set()
            for worker in workers:
                worker.join(10)
                assert not worker.is_alive()
        assert refusals == [True]
        assert not copies
        assert not worker_errors
        assert policy.used_memory_bytes == 0
    assert not list(filesystem_glob(tmp_path, "**/manifest.json"))


def test_raw_table_identity_ignores_parent_directory_and_rejects_conflict(tmp_path):
    payload = "Stkcd,Shrchgdt,Nshrttl,Nshra\n000001,2024-01-02,100,50\n"
    outputs = []
    for directory in ("stock_master", "market_cap"):
        root = tmp_path / directory
        filesystem_mkdir(root)
        path = root / "unknown.csv"
        filesystem_write_text(path, payload, encoding="utf-8")
        outputs.append(raw_schemas.inspect_raw_export_file(path).dataset)
    assert outputs == ["market_cap", "market_cap"]
    conflict = tmp_path / "TRDNEW_Co.csv"
    filesystem_write_text(conflict, payload, encoding="utf-8")
    with pytest.raises(ValueError, match="table_identity_conflict"):
        raw_schemas.inspect_raw_export_file(conflict)


@pytest.mark.parametrize(
    ("code", "column"),
    [
        ("invalid_index_info_date", "start_date"),
        ("invalid_index_info_date", "base_date"),
        ("invalid_stock_master_date", "listing_date"),
        ("invalid_stock_master_date", "delisting_date"),
        ("invalid_stock_master_date", "status_change_date"),
        ("invalid_industry_version_date", "abandon_date"),
        ("invalid_analyst_date", "declare_date"),
        ("invalid_listing_status_date", "announcement_date"),
        ("invalid_stock_info_change_date", "announcement_date"),
    ],
)
def test_controlled_csmar_date_reason_survives_workflow_sanitizer(code, column):
    from quant_research_platform.workflows.research import _semantic_failure_code

    with pytest.raises(ValueError) as raised:
        interim._parse_optional_csmar_date(
            pd.Series(["invalid-date"]), column, error_code=code
        )
    assert _semantic_failure_code(raised.value) == f"{code}:{column}"
    valid = interim._parse_optional_csmar_date(
        pd.Series(["", " ", None, "2020-01-01"]), column, error_code=code
    )
    assert valid.isna().tolist() == [True, True, True, False]
    assert str(valid.dtype) == "datetime64[ns]"


def test_controlled_csmar_numeric_reasons_survive_workflow_sanitizer():
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS
    from quant_research_platform.workflows.research import _semantic_failure_code

    datasets = [
        (
            "analyst_forecast",
            _analyst_rows()[0],
            ("forecast_eps", "forecast_pe"),
            "invalid_analyst_value",
        )
    ]
    for data_id, columns in interim.FINANCIAL_COLUMNS_BY_DATA_ID.items():
        numeric = [c for c in columns if c not in interim.FINANCIAL_BASE_COLUMNS]
        inverse = {value: key for key, value in COLUMN_MAPPINGS[data_id].items()}
        row = {
            "Stkcd": "000001",
            "ShortName": "synthetic",
            "IfCorrect": 0,
            "DeclareDate": None,
            "Accper": "2020-12-31",
            "Typrep": "A",
            **{inverse[column]: 2.5 for column in numeric},
        }
        datasets.append((data_id, row, numeric, "invalid_financial_value"))
    checked = 0
    for data_id, row, columns, code in datasets:
        inverse = {value: key for key, value in COLUMN_MAPPINGS[data_id].items()}
        valid = interim.normalize_raw_frame(pd.DataFrame([row]), data_id)
        for column in columns:
            assert float(valid.loc[0, column]) == float(row[inverse[column]])
            assert pd.api.types.is_float_dtype(valid[column].dtype)
            with pytest.raises(ValueError) as raised:
                interim.normalize_raw_frame(
                    pd.DataFrame([{**row, inverse[column]: "invalid-number"}]), data_id
                )
            assert _semantic_failure_code(raised.value) == f"{code}:{column}"
            checked += 1
    assert checked == 16
    for message in ("private value", "/private/input.csv", "private\nvalue", "a" * 97):
        assert _semantic_failure_code(ValueError(message)) == "valueerror"


def test_csmar_date_reason_survives_research_run_record_and_report(tmp_path):
    import json

    from quant_research_platform.artifacts.research import read_research_run_record
    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows import research
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.research_configuration import research_configuration

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    inverse = {value: key for key, value in COLUMN_MAPPINGS["index_info"].items()}
    _write_csmar(
        raw,
        "index_info",
        [
            {
                inverse["index_code"]: "000905",
                inverse["base_date"]: "2020-01-01",
                inverse["start_date"]: "invalid-date",
            }
        ],
    )
    profile = research_profile()
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=research_configuration(development=("2019-01-01", "2019-12-31")),
    )
    expected = "interim_build_failed:invalid_index_info_date:start_date"
    target = next(
        item for item in result["target_results"] if item["target_id"] == "index_info"
    )
    assert result["status"] == target["status"] == "FAILED"
    assert target["reason"] == expected
    record = read_research_run_record(layout.research_run_path(result["run_id"]))
    saved = next(
        item for item in record.target_results if item.target_id == "index_info"
    )
    assert record.status.value == saved.status.value == "FAILED"
    assert saved.reason == expected
    report = research.report_build(
        profile=profile, layout=layout, run_id=result["run_id"]
    )
    assert report["source_run_status"] == "FAILED"
    assert expected in json.dumps(report)


@pytest.mark.parametrize(
    "available,cap",
    [(64 * 1024**2, 8 * 1024**2), (64 * 1024**2, 384 * 1024), (384 * 1024, None)],
)
def test_csv_scratch_remains_admitted_when_extra_fields_leave_the_frame(
    tmp_path, monkeypatch, available, cap
):
    import os
    import tempfile

    from quant_research_platform.runtime.resources import pandas_frame_payload_bytes
    from quant_research_platform.sources.csmar.field_mappings import RAW_TABLE_CONTRACTS
    from quant_research_platform.storage import _scratch_ownership

    scratch = tmp_path / "scratch"
    filesystem_mkdir(scratch)
    monkeypatch.setattr(tempfile, "tempdir", native_filesystem_path(scratch))
    monkeypatch.setattr(_scratch_ownership, "_memory_filesystem", lambda _path: True)
    temporary_file = _scratch_ownership.TemporaryFile
    handles = []

    def tracked_file(*args, **kwargs):
        handle = temporary_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(_scratch_ownership, "TemporaryFile", tracked_file)
    columns = RAW_TABLE_CONTRACTS["financial_balance"].required_columns
    values = {column: "1" for column in columns}
    values.update(
        Stkcd="000001",
        ShortName="Synthetic",
        IfCorrect="0",
        DeclareDate="2020-04-01",
        Accper="2019-12-31",
        Typrep="A",
    )
    path = tmp_path / "extra.csv"
    filesystem_write_text(
        path,
        ",".join((*columns, "OptionalProviderField"))
        + "\n"
        + ",".join(values[column] for column in columns)
        + ',"'
        + "x" * (896 * 1024)
        + '\nx"\n',
        encoding="utf-8",
    )
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=available, explicit_cap_bytes=cap)
    ) as policy:
        chunks = raw_exports.iter_private_export_chunks_with_policy(
            path, dataset="financial_balance", chunksize=1, resource_policy=policy
        )
        lease = policy.lease("test.csmar.producer", next(chunks))
        try:
            if cap == 8 * 1024**2:
                owned = chunks.send(lease)
                assert "OptionalProviderField" not in owned.value
                assert owned.value.Stkcd.tolist() == ["000001"]
                scratch_bytes = sum(
                    os.fstat(handle.fileno()).st_size
                    for handle in handles
                    if not handle.closed
                )
                assert scratch_bytes > 896 * 1024
                assert policy.used_memory_bytes >= (
                    scratch_bytes + pandas_frame_payload_bytes(owned.value)
                )
                owned.release(owner="test.csmar.consumer")
            else:
                with pytest.raises(
                    ValueError, match="^storage_scratch_exceeds_memory_capacity$"
                ):
                    chunks.send(lease)
        finally:
            chunks.close()
            lease.close()
        assert policy.used_memory_bytes == 0
    assert handles and all(handle.closed for handle in handles)
    assert list(filesystem_iterdir(scratch)) == []


def test_index_info_continuation_scratch_survives_the_reconstructed_frame_handoff(
    tmp_path, monkeypatch
):
    import os
    import tempfile

    from quant_research_platform.runtime.resources import pandas_frame_payload_bytes
    from quant_research_platform.storage import _scratch_ownership

    scratch = tmp_path / "scratch"
    filesystem_mkdir(scratch)
    monkeypatch.setattr(tempfile, "tempdir", native_filesystem_path(scratch))
    monkeypatch.setattr(_scratch_ownership, "_memory_filesystem", lambda _path: True)
    temporary_file = _scratch_ownership.TemporaryFile
    handles = []

    def tracked_file(*args, **kwargs):
        handle = temporary_file(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(_scratch_ownership, "TemporaryFile", tracked_file)
    count = 393216
    path = tmp_path / "continuation.csv"
    # Keep the logical record's source bytes identical across platforms.
    filesystem_write_bytes(
        path,
        b"Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        b"Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        b"000985,synthetic,2014-01-02,2013-12-31,1000,scope,1,2,3,4,5,formula\n"
        + b"#\n" * count
        + b"000300,next,2014-01-03,2013-12-31,1000,scope,1,2,3,4,5,end\n",
    )
    with runtime_resource_policy_scope(
        capture_test_policy(
            available_memory_bytes=64 * 1024**2, explicit_cap_bytes=32 * 1024**2
        )
    ) as policy:
        chunks = raw_exports.iter_private_export_chunks_with_policy(
            path, dataset="index_info", chunksize=1, resource_policy=policy
        )
        lease = policy.lease("test.csmar.producer", next(chunks))
        try:
            owned = chunks.send(lease)
            assert owned.value.Idxinfo10.iloc[0] == "formula" + "\n#" * count
            scratch_bytes = sum(
                os.fstat(handle.fileno()).st_size
                for handle in handles
                if not handle.closed
            )
            assert scratch_bytes >= 5 * count
            assert policy.used_memory_bytes >= (
                scratch_bytes + pandas_frame_payload_bytes(owned.value)
            )
            owned.release(owner="test.csmar.consumer")
        finally:
            chunks.close()
            lease.close()
        assert policy.used_memory_bytes == 0
    assert handles and all(handle.closed for handle in handles)
    assert list(filesystem_iterdir(scratch)) == []


def test_expired_slow_record_releases_scratch_before_fast_records(tmp_path):
    import os

    path = tmp_path / "slow_fast.csv"
    # End the slow record at the buffer boundary so the next records take the fast path.
    filesystem_write_bytes(path, b'"' + b"s" * (8192 - 5) + b'\nx"\n' + b"a,b\n" * 3)
    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2)
    ) as policy:
        records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        try:
            slow, _ = next(records)
            owner = slow._owner
            assert os.fstat(owner._spool.fileno()).st_size > 0
            for _ in range(3):
                fast, _ = next(records)
                assert fast.materialize() == ["a", "b"]
                assert owner._decoded_fields is not None
                assert os.fstat(owner._spool.fileno()).st_size == 0
            with pytest.raises(RuntimeError, match="csmar_csv_record_expired"):
                slow.materialize()
        finally:
            records.close()
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_csmar_releases_decoded_batch_before_normalization(
    tmp_path, monkeypatch, cancel
):
    import weakref

    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    class TrackedRow(list):
        pass

    class TrackedNumericText(str):
        pass

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    rows = []
    for day, price in ((1, 10.0), (2, 11.0), (3, 12.0)):
        row = dict.fromkeys(
            raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
        )
        row.update(
            Stkcd="000001",
            Trddt=f"2019-08-{day:02}",
            Markettype=32,
            Filling=0,
            Opnprc=price,
            Hiprc=price,
            Loprc=price,
            Clsprc=price,
            Adjprcwd=price,
        )
        rows.append(row)
    _write_csmar(raw, "market_bars_1d", rows)
    records = raw_exports._iter_csv_records
    transform = interim._transform_interim_chunk
    references, observations = [], []
    primary = KeyboardInterrupt("decoded_batch_consumer_cancel")

    def tracked_records(*args, **kwargs):
        with closing_iterator(records(*args, **kwargs)) as source:
            for header, row, source_bytes in source:
                tracked = TrackedRow(row)
                position = header.index("Opnprc")
                tracked[position] = TrackedNumericText(tracked[position])
                references.append(
                    (weakref.ref(tracked), weakref.ref(tracked[position]))
                )
                try:
                    yield header, tracked, source_bytes
                finally:
                    tracked = None

    def observe_transform(frame, **kwargs):
        assert len(references) == len(frame) == 3
        # The suspended row iterator may still own its final row.
        assert all(ref() is None for pair in references[:-1] for ref in pair)
        observations.append(len(frame))
        if cancel:
            raise primary
        return transform(frame, **kwargs)

    with runtime_resource_policy_scope(explicit_cap_bytes=32 * 1024**2) as policy:
        original = interim.materialize_interim_data(
            layout=layout,
            data_ids=("market_bars_1d",),
            chunksize=3,
            resource_policy=policy,
        ).artifact_roots["market_bars_1d"]
        expected = read_dataset(original, resource_policy=policy)
        old_good = _file_bytes(original)
        candidate = StorageLayout(tmp_path / "comparison").interim_data_root(
            "csmar", "market_bars_1d"
        )
        monkeypatch.setattr(raw_exports, "_iter_csv_records", tracked_records)
        monkeypatch.setattr(interim, "_transform_interim_chunk", observe_transform)
        request = dict(
            layout=layout.with_target_roots({"market_bars_1d": candidate}),
            data_ids=("market_bars_1d",),
            chunksize=3,
            resource_policy=policy,
        )
        if cancel:
            with pytest.raises(KeyboardInterrupt) as caught:
                interim.materialize_interim_data(**request)
            assert caught.value is primary
            assert not filesystem_exists(candidate)
        else:
            interim.materialize_interim_data(**request)
            actual = read_dataset(candidate, resource_policy=policy)
            pd.testing.assert_frame_equal(actual, expected)
        assert observations == [3]
        remaining = [
            type(ref()).__name__
            for pair in references
            for ref in pair
            if ref() is not None
        ]
        assert not remaining, remaining
        assert _file_bytes(original) == old_good
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize(
    "case",
    ["index_lookahead", "index_lookahead_unicode", "projected_field", "column_label"],
)
def test_csmar_reader_payload_survives_output_lease_close(tmp_path, monkeypatch, case):
    import sys
    import weakref

    from quant_research_platform.runtime.resources import pandas_frame_payload_bytes

    class TrackedText(str):
        pass

    width = 900_000 if case == "index_lookahead_unicode" else 600_000
    wide_text = "x" * (width - 1) + ("😀" if case == "index_lookahead_unicode" else "x")
    path = tmp_path / "reader.csv"
    if case == "projected_field":
        dataset = "market_cap"
        header = "Stkcd,Shrchgdt,Nshrttl,Nshra,OptionalProviderField"
        text = (
            header
            + "\n000001,2019-08-01,100,50,"
            + "x" * width
            + "\n000001,2019-08-02,200,100,small\n"
        )
    else:
        dataset = "index_info"
        header = (
            "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
            "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10"
        )
        if case == "column_label":
            text = (
                header
                + ","
                + "x" * width
                + "\n000985,short,2014-01-02,2013-12-31,1000,scope,1,2,3,4,5,end,1\n"
            )
        else:
            text = (
                header
                + "\n000985,short,2014-01-02,2013-12-31,1000,scope,1,2,3,4,5,end\n"
                + "000300,"
                + wide_text
                + ",2014-01-03,2013-12-31,1000,scope,1,2,3,4,5,end\n"
            )
    filesystem_write_text(path, text, encoding="utf-8")
    del text
    references = []
    parse = raw_schemas._BoundedCsvRecords._try_fast_fields

    def tracked_fields(reader):
        result = parse(reader)
        if result is not None:
            fields = result[0]
            for position, value in enumerate(fields):
                if len(value) == width:
                    fields[position] = TrackedText(value)
                    references.append(weakref.ref(fields[position]))
        return result

    monkeypatch.setattr(
        raw_schemas._BoundedCsvRecords, "_try_fast_fields", tracked_fields
    )
    with runtime_resource_policy_scope(
        capture_test_policy(
            available_memory_bytes=64 * 1024**2, explicit_cap_bytes=32 * 1024**2
        )
    ) as policy:
        chunks = raw_exports.iter_private_export_chunks_with_policy(
            path, dataset=dataset, chunksize=1, resource_policy=policy
        )
        lease = policy.lease("test.csmar.producer", next(chunks))
        try:
            owned = chunks.send(lease)
            assert len(references) == 1 and references[0]() is not None
            retained = sys.getsizeof(references[0]())
            assert retained >= width
            if case == "column_label":
                assert len(owned.value.columns[-1]) == width
                assert lease.reserved_bytes >= (
                    retained + pandas_frame_payload_bytes(owned.value)
                )
            elif case.startswith("index_lookahead"):
                assert owned.value.Idxinfo01.tolist() == ["short"]
            else:
                assert "OptionalProviderField" not in owned.value
                assert owned.value.Nshrttl.tolist() == [100.0]
            assert (
                sum(
                    item.reserved_bytes
                    for item in policy._leases
                    if item.owner.startswith("source.csmar.")
                )
                >= retained
            )
            del owned
            next_bound = chunks.send(None)
            lease.close()
            # The reader still holds its last record/lookahead after output closes.
            assert references[0]() is not None
            assert (
                sum(
                    item.reserved_bytes
                    for item in policy._leases
                    if item.owner.startswith("source.csmar.")
                )
                >= retained
            )
            lease = policy.lease("test.csmar.next", next_bound)
            if case == "column_label":
                with pytest.raises(StopIteration):
                    chunks.send(lease)
            else:
                owned = chunks.send(lease)
                if case.startswith("index_lookahead"):
                    assert owned.value.Idxinfo01.iloc[0] == wide_text
                else:
                    assert owned.value.Nshrttl.tolist() == [200.0]
                del owned
        finally:
            chunks.close()
            lease.close()
        assert all(reference() is None for reference in references)
        assert policy.used_memory_bytes == 0


def _nullable_index_numeric_rows(data_id):
    if data_id == "index_weights":
        return [
            {
                "Indexcd": "000905",
                "Enddt": "2019-01-02",
                "Stkcd": f"{position + 1:06}",
                "Constdnme": f"synthetic-{position}",
                "Weight": value,
            }
            for position, value in enumerate((None, 25.0, 25.0, 50.0))
        ]
    return [
        {
            "Indexcd": "H00905",
            "Idxtrd01": f"2019-01-{position + 2:02}",
            "Idxtrd02": 100.0,
            "Idxtrd03": 101.0,
            "Idxtrd04": 99.0,
            "Idxtrd05": 100.0,
            "Idxtrd06": 1.0,
            "Idxtrd07": 1.0,
            "Idxtrd08": value,
            "Idxtrd09": "synthetic-index",
        }
        for position, value in enumerate((None, -2.5, 0.0, 2.5))
    ]


@pytest.mark.parametrize("chunksize", (1, 4))
@pytest.mark.parametrize(
    "data_id,column,expected",
    (
        ("index_weights", "weight", (float("nan"), 0.25, 0.25, 0.5)),
        ("index_market", "period_return", (float("nan"), -0.025, 0.0, 0.025)),
    ),
)
def test_index_source_null_numeric_values_persist_without_claiming_complete_weights(
    tmp_path, chunksize, data_id, column, expected
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.markets.implementations.a_share.benchmark import (
        iter_benchmark_weight_snapshots,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, data_id, _nullable_index_numeric_rows(data_id))
    original = _file_bytes(raw)
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=(data_id,),
            chunksize=chunksize,
            resource_policy=policy,
        )
        root = result.artifact_roots[data_id]
        saved = read_dataset(root, resource_policy=policy).sort_values(
            "constituent_asset_id" if data_id == "index_weights" else "timestamp"
        )
        pd.testing.assert_series_equal(
            saved[column].reset_index(drop=True),
            pd.Series(expected, name=column, dtype="float64"),
            check_exact=True,
        )
        nulls = 0
        for path in filesystem_glob(root, "**/*.parquet"):
            table = pq.ParquetFile(native_filesystem_path(path)).read(columns=[column])
            assert table.schema.field(column).type == pa.float64()
            nulls += table[column].null_count
        assert nulls == 1
        if data_id == "index_weights":
            info = pd.DataFrame(
                {
                    "source_asset_id": ["000905", "000852"],
                    "index_name": ["中证小盘500指数", "中证1000指数"],
                }
            )
            # The known weights sum to one, but the missing member is still unknown.
            assert saved.weight.sum() == 1.0
            with pytest.raises(ValueError, match="a_share_benchmark_weight_invalid"):
                list(iter_benchmark_weight_snapshots(info, [saved]))
        assert policy.used_memory_bytes == 0
    assert _file_bytes(raw) == original


@pytest.mark.parametrize(
    "data_id,raw_column,bad_value,reason",
    [
        (data_id, raw_column, value, reason)
        for data_id, raw_column, reason in (
            ("index_weights", "Weight", "invalid_index_weight"),
            ("index_market", "Idxtrd08", "invalid_index_return"),
        )
        for value in ("invalid-number", float("inf"), -float("inf"))
    ]
    + [("index_weights", "Weight", -1.0, "invalid_index_weight")],
)
def test_nullable_index_source_numeric_values_still_reject_invalid_facts(
    data_id, raw_column, bad_value, reason
):
    rows = _nullable_index_numeric_rows(data_id)
    rows[0][raw_column] = bad_value
    with pytest.raises(ValueError, match=reason):
        interim.normalize_raw_frame(pd.DataFrame(rows), data_id)


@pytest.mark.parametrize("cap", (512 * 1024, 8 * 1024**2))
@pytest.mark.parametrize(
    "payload,expected",
    (
        (b'"1",2\r\n', [["1", "2"]]),
        (b'\xef\xbb\xbf"a""b",2\r\n', [['a"b', "2"]]),
        (b'"line\r\nbreak",2\r\n', [["line\r\nbreak", "2"]]),
        (b'"1"\r"2"\n"3"', [["1"], ["2"], ["3"]]),
        (b'"",', [["", ""]]),
        (
            b'"' + b"x" * 8190 + b'""y",2\r\n',
            [["x" * 8190 + '"y', "2"]],
        ),
        (b'"1"0,2\n', None),
        (b'"1" ,2\n', None),
        (b'"1"\t,2\n', None),
        (b'"unterminated,2\n', None),
        (b'"' + b"x" * 8190 + b'"0,2\n', None),
    ),
)
def test_csv_strict_grammar_matches_across_memory_budgets(
    tmp_path, cap, payload, expected
):
    import io

    path = tmp_path / "grammar.csv"
    filesystem_write_bytes(path, payload)
    oracle = csv.reader(
        io.StringIO(payload.decode("utf-8-sig"), newline=""), strict=True
    )
    if expected is None:
        with pytest.raises(csv.Error):
            list(oracle)
    else:
        assert list(oracle) == expected

    with runtime_resource_policy_scope(
        capture_test_policy(available_memory_bytes=64 * 1024**2, explicit_cap_bytes=cap)
    ) as policy:
        records = raw_schemas.iter_export_csv_records(path, resource_policy=policy)
        actual, modes = [], []
        try:
            if expected is None:
                with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
                    next(records)
            else:
                for record, _ in records:
                    modes.append(record._owner._decoded_fields is not None)
                    actual.append(record.materialize())
                assert actual == expected
                if cap == 512 * 1024:
                    assert modes and not any(modes)
                elif payload == b'"1",2\r\n':
                    assert modes == [True]
        finally:
            records.close()
        assert policy.used_memory_bytes == 0


def test_malformed_quoted_numeric_cannot_publish_over_old_good(tmp_path):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows()[:1])
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout, data_ids=("analyst_forecast",), resource_policy=policy
        )
        root = result.artifact_roots["analyst_forecast"]
        old_good = _file_bytes(root)
        before = read_dataset(root, resource_policy=policy)
        path = next(filesystem_glob(raw, "*.csv"))
        payload = filesystem_read_bytes(path)
        assert payload.count(b",18,10") == 1
        malformed = payload.replace(b",18,10", b',"1"0,10')
        filesystem_write_bytes(path, malformed)
        candidate = layout.instance_root(root, "f" * 64)
        with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
            interim.materialize_interim_data(
                layout=layout.with_target_roots({"analyst_forecast": candidate}),
                data_ids=("analyst_forecast",),
                resource_policy=policy,
            )
        assert not filesystem_exists(candidate)
        assert _file_bytes(root) == old_good
        pd.testing.assert_frame_equal(
            before, read_dataset(root, resource_policy=policy)
        )
        assert filesystem_read_bytes(path) == malformed
        assert policy.used_memory_bytes == 0
