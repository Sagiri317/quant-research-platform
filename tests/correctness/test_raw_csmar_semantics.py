from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.sources.csmar import interim, raw_exports, raw_schemas
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_glob,
    filesystem_mkdir,
    filesystem_rename,
    filesystem_unlink,
    filesystem_walk,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)
from tests.helpers.source_preparation import prepared_source_evidence


def _file_bytes(root: Path) -> dict[Path, bytes]:
    return {
        (parent / name).relative_to(root): Path(
            native_filesystem_path(parent / name)
        ).read_bytes()
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


@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize("unused_value", [0.125, "invalid-unused-numeric"])
@pytest.mark.parametrize(
    "data_id,source_column,kept_column",
    (
        ("market_bars_1d", "Dretnd", "return_without_dividend"),
        ("stock_master", "Markettype", "market_type"),
        ("market_cap", "Nshrb", "b_float_shares"),
    ),
)
def test_csmar_fixed_projection_preserves_dtypes_duplicates_and_raw(
    tmp_path, chunksize, data_id, source_column, kept_column, unused_value
):

    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    row = dict.fromkeys(raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns, 1.0)
    row.update(Stkcd="000001")
    if data_id == "market_bars_1d":
        row.update(
            Trddt="2020-01-02",
            Markettype=1,
            Trdsta=0,
            Filling=0,
            Ahshrtrd_D=0.0,
            Ahvaltrd_D=0.0,
            LimitStatus=0,
        )
    elif data_id == "stock_master":
        row.update(
            Stknme="synthetic",
            Listdt="2020-01-02",
            DelistedDate="",
            Statdt="2020-01-02",
            Statco="normal",
            Markettype=4,
        )
    else:
        row.update(Shrchgdt="2020-01-02")
    row[source_column] = unused_value
    extra = (
        ()
        if source_column in row
        and source_column in raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns
        else (source_column,)
    )
    _write_csmar(
        raw, data_id, [row, {**row, "Stkcd": "000002"}, row], extra_columns=extra
    )
    raw_bytes = _file_bytes(raw)
    default = interim.materialize_interim_data(
        output_roots={
            key: StorageLayout(tmp_path / "default").interim_data_root("csmar", key)
            for key in (data_id,)
        },
        raw_root=raw,
        data_ids=(data_id,),
        chunksize=chunksize,
    )
    default_frame = (
        read_dataset(default.artifact_roots[data_id])
        .sort_values("source_asset_id")
        .reset_index(drop=True)
    )
    assert kept_column not in default_frame
    root = default.artifact_roots[data_id]
    saved = default_frame
    from quant_research_platform.storage.manifests import read_qrp_json_file

    _raw_spec, declared = read_qrp_json_file(root / "spec.json")
    assert set(declared["column_roles"]) == set(saved.columns)
    assert tuple(saved.columns) == interim.INTERIM_COLUMNS_BY_DATA_ID[data_id]
    assert saved.source_asset_id.tolist() == ["000001", "000002"]
    assert str(saved.timestamp.dtype) == "datetime64[ns]"
    assert (
        read_dataset_manifest_evidence(root)["producer_diagnostics"][
            "exact_duplicate_count"
        ]
        == 1
    )
    assert _file_bytes(raw) == raw_bytes


@pytest.mark.parametrize("chunksize", (1, 3, 100))
def test_analyst_raw_reports_persist_and_feed_institution_consensus(
    tmp_path, chunksize
):
    import json

    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows())
    raw_path = next(filesystem_glob(raw, "*.csv"))
    original = Path(native_filesystem_path(raw_path)).read_bytes()
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("analyst_forecast",),
        chunksize=chunksize,
        start_date="2025-01-06",
        end_date="2025-01-07",
    )
    root = result.artifact_roots["analyst_forecast"]
    saved = read_dataset(root)
    manifest = json.loads(
        Path(native_filesystem_path(root / "_manifest/manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    assert manifest["producer_diagnostics"]["source_row_counts"] == {
        "raw_rows": 8,
        "missing_declare_date_rows": 2,
        "missing_institution_id_rows": 2,
        "excluded_missing_required_rows": 2,
        "retained_before_date_filter_rows": 6,
        "retained_after_date_filter_rows": 6,
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
    assert len(saved) == 6
    assert saved.institution_id.isna().sum() == 1
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
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_module,
    )

    processed_catalog = processed_module.processed_catalog

    processed_spec = processed_catalog().get("expectations_factors")
    actual = producer._compute_processed_frame(
        processed_spec,
        source={"analyst_forecast": saved},
        dependencies={"market_panel_1d": panel},
        trading_sessions=tuple(sessions),
        expectations=producer._prepare_expectations(
            saved.rename(columns={"source_asset_id": "asset_id"}),
            trading_sessions=sessions,
        ),
    )
    processed_root = layout.processed_data_root(
        "a_share", "factor", "expectations_factors"
    )
    write_dataset(processed_root, actual, spec=processed_spec)
    persisted = read_dataset(processed_root)
    pd.testing.assert_frame_equal(persisted, actual, check_exact=True)
    assert persisted.iloc[0, 3:].isna().all()
    assert (
        persisted.loc[
            1,
            [
                "forward_earnings_yield_fy1",
                "forecast_dispersion_fy1",
                "log_analyst_coverage",
            ],
        ]
        .isna()
        .all()
    )
    assert persisted.forward_earnings_yield_fy2.isna().all()
    assert Path(native_filesystem_path(raw_path)).read_bytes() == original


def test_analyst_all_excluded_persists_zero_rows_and_counts(tmp_path):
    import json

    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, "analyst_forecast", [_analyst_rows()[5], _analyst_rows()[7]])
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("analyst_forecast",),
        chunksize=1,
    )
    root = result.artifact_roots["analyst_forecast"]
    empty = read_dataset(root)
    assert empty.empty
    manifest = json.loads(
        Path(native_filesystem_path(root / "_manifest/manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    counts = manifest["producer_diagnostics"]["source_row_counts"]
    assert counts["raw_rows"] == counts["excluded_missing_required_rows"] == 2
    assert counts["retained_after_date_filter_rows"] == 0
    sessions = pd.bdate_range("2025-01-03", periods=2).as_unit("ns")
    from types import SimpleNamespace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog as processed_module,
    )

    processed_catalog = processed_module.processed_catalog
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.a_share_sources import _synthetic_interim_frames

    profile = SimpleNamespace(
        processed=processed_catalog(), interim=interim.interim_catalog()
    )
    panel_spec = profile.processed.get("market_panel_1d")
    facts = _synthetic_interim_frames(sessions)
    panel = producer._compute_processed_frame(
        panel_spec,
        source={
            **producer._prepare_panel_source(
                facts, producer.panels._canonical_stock_master(facts["stock_master"])
            ),
            "market_bars_1d": facts["market_bars_1d"],
        },
        dependencies={},
        trading_sessions=sessions,
    )
    write_dataset(
        layout.processed_data_root("a_share", panel_spec.kind, panel_spec.data_id),
        panel,
        spec=panel_spec,
    )
    calendar_spec = profile.interim.get("trading_calendar")
    write_dataset(
        layout.interim_data_root("csmar", "trading_calendar"),
        facts["trading_calendar"].loc[:, list(calendar_spec.required_columns)],
        spec=calendar_spec,
    )
    with pytest.raises(ValueError, match="BLOCKED_EXTERNAL_DATA: analyst_forecast"):
        with producer.produce_a_share_processed_data(
            profile.processed.get("expectations_factors"),
            inputs={
                **{
                    name: layout.interim_data_root(
                        profile.interim.get(name).source_id, name
                    )
                    for name in profile.processed.get(
                        "expectations_factors"
                    ).source_input_data_ids
                },
                "market_panel_1d": layout.processed_data_root(
                    "a_share", panel_spec.kind, panel_spec.data_id
                ),
            },
            time_window=TimeWindow(sessions[0], sessions[-1]),
            processed_catalog=profile.processed,
            source_catalog=profile.interim,
        ) as (frames, _metadata):
            list(frames)


@pytest.mark.parametrize("column", ("Markettype", "Daywk"))
def test_calendar_direct_boolean_is_not_an_integer(column):
    row = {"Markettype": "4", "Clddt": "2025-01-03", "Daywk": "5", "State": "O"}
    row[column] = True
    result = interim.normalize_raw_frame(pd.DataFrame([row]), "trading_calendar")
    counts = result.attrs["source_row_counts"]
    target = "source_market_type" if column == "Markettype" else "source_day_of_week"
    assert counts[f"invalid_{target}_rows"] == 1
    if column == "Markettype":
        assert result.empty
        assert counts["excluded_missing_required_rows"] == 1
    else:
        assert result.source_day_of_week.isna().all()
        assert result.is_trading_day.tolist() == [True]
        assert result.timestamp.tolist() == [pd.Timestamp("2025-01-03")]
    assert str(result.timestamp.dtype) == "datetime64[ns]"
    assert str(result.source_day_of_week.dtype) == "Int64"


@pytest.mark.parametrize(
    "column,value",
    (
        ("DeclareDate", "invalid"),
        ("Fenddt", "invalid"),
        ("ReportID", ""),
        ("Feps", "infinite"),
    ),
)
@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize("all_bad", (False, True))
def test_invalid_analyst_facts_are_local_to_required_rows_or_optional_cells(
    tmp_path, column, value, chunksize, all_bad
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    row = {**_analyst_rows()[0], column: value}
    rows = [row] if all_bad else [row, {**_analyst_rows()[0], "Stkcd": "000002"}]
    _write_csmar(raw, "analyst_forecast", rows)
    original = _file_bytes(raw)
    candidate = tmp_path / "candidate"
    interim.materialize_interim_data(
        output_roots={"analyst_forecast": candidate},
        raw_root=raw,
        data_ids=("analyst_forecast",),
        chunksize=chunksize,
    )
    saved = read_dataset(candidate).sort_values("source_asset_id")
    optional = column == "Feps"
    assert len(saved) == (len(rows) if optional else len(rows) - 1)
    assert str(saved.timestamp.dtype) == "datetime64[ns]"
    assert str(saved.forecast_end_date.dtype) == "datetime64[ns]"
    assert pd.api.types.is_float_dtype(saved.forecast_eps.dtype)
    if optional:
        assert pd.isna(saved.forecast_eps.iloc[0])
        assert saved.forecast_pe.iloc[0] == 10
    if not all_bad:
        good = saved.loc[saved.source_asset_id.eq("000002")].iloc[0]
        assert good.forecast_eps == 18
        assert good.report_id == "9007199254740993"
    counts = read_dataset_manifest_evidence(candidate)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == len(rows)
    assert counts["retained_after_date_filter_rows"] == len(saved)
    assert counts.get("excluded_missing_required_rows", 0) == (0 if optional else 1)
    if optional:
        assert counts["invalid_forecast_eps_rows"] == 1
    assert _file_bytes(raw) == original


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
@pytest.mark.parametrize("chunksize", (1, 100))
def test_calendar_raw_invalid_codes_preserve_independent_dates_and_states(
    tmp_path, column, value, chunksize
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    valid = {"Markettype": "4", "Clddt": "2025-01-06", "Daywk": "1", "State": "C"}
    bad = {**valid, "Clddt": "2025-01-03", "State": "O", column: value}
    _write_csmar(raw, "trading_calendar", [bad, valid])
    original = _file_bytes(raw)
    candidate = tmp_path / "candidate"
    interim.materialize_interim_data(
        output_roots={"trading_calendar": candidate},
        raw_root=raw,
        data_ids=("trading_calendar",),
        chunksize=chunksize,
    )
    saved = read_dataset(candidate).sort_values("timestamp")
    assert saved.iloc[-1].timestamp == pd.Timestamp("2025-01-06")
    assert saved.iloc[-1].source_day_of_week == 1
    assert not saved.iloc[-1].is_trading_day
    if column == "Daywk":
        assert len(saved) == 2
        assert pd.isna(saved.iloc[0].source_day_of_week)
        assert saved.iloc[0].is_trading_day
    else:
        assert len(saved) == 1
    counts = read_dataset_manifest_evidence(candidate)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == 2
    assert counts["retained_after_date_filter_rows"] == len(saved)
    assert counts.get("excluded_missing_required_rows", 0) == (column == "Markettype")
    if value:
        target = (
            "source_market_type" if column == "Markettype" else "source_day_of_week"
        )
        assert counts[f"invalid_{target}_rows"] == 1
    assert _file_bytes(raw) == original


def test_calendar_raw_preserves_integer_codes(tmp_path):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    row = {"Markettype": "4", "Clddt": "2025-01-03", "Daywk": "5", "State": "O"}
    _write_csmar(raw, "trading_calendar", [row])
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("trading_calendar",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("trading_calendar",),
    )
    root = result.artifact_roots["trading_calendar"]
    before = read_dataset(root)
    assert before.source_market_type.tolist() == [4]
    assert before.source_day_of_week.tolist() == [5]
    assert before.is_trading_day.tolist() == [True]


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
    result = interim.normalize_raw_frame(pd.DataFrame([row]), "market_bars_1d")
    assert len(result) == 1
    assert result.close.iloc[0] == 1.0
    assert result.timestamp.iloc[0] == pd.Timestamp(day)
    if accepted:
        expected = 0.0 if after is None else after
        assert (
            result.after_hours_volume.iloc[0]
            == result.after_hours_amount.iloc[0]
            == expected
        )
        assert result.volume.iloc[0] == result.amount.iloc[0] == 1.0 + expected
    else:
        assert (
            result[["after_hours_volume", "after_hours_amount", "volume", "amount"]]
            .isna()
            .all()
            .all()
        )
    if filling == 3:
        assert pd.isna(result.filling_flag.iloc[0])


def test_after_hours_materialization_preserves_known_totals_and_local_unknowns(
    tmp_path,
) -> None:
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
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
    original_root = tmp_path / "original"
    interim.materialize_interim_data(
        output_roots={"market_bars_1d": original_root},
        raw_root=raw,
        data_ids=("market_bars_1d",),
    )
    loaded = read_dataset(original_root)
    assert loaded.volume.iloc[0] == 3.0 and loaded.amount.iloc[0] == 4.0
    before = _file_bytes(original_root)
    _write_csmar(
        raw, "market_bars_1d", [{**row, "Ahvaltrd_D": None}, {**row, "Stkcd": "000002"}]
    )
    raw_before = _file_bytes(raw)
    candidate = tmp_path / "candidate"
    interim.materialize_interim_data(
        output_roots={"market_bars_1d": candidate},
        raw_root=raw,
        data_ids=("market_bars_1d",),
        chunksize=1,
    )
    saved = read_dataset(candidate).set_index("source_asset_id")
    assert len(saved) == 2
    assert saved.loc["000001", "volume"] == 3.0
    assert pd.isna(saved.loc["000001", "after_hours_amount"])
    assert pd.isna(saved.loc["000001", "amount"])
    assert saved.loc["000001", "close"] == 1.0
    assert saved.loc["000002", "volume"] == 3.0
    assert saved.loc["000002", "amount"] == 4.0
    assert _file_bytes(original_root) == before
    assert _file_bytes(raw) == raw_before


@pytest.mark.parametrize("bad", ["invalid", float("inf"), -float("inf")])
@pytest.mark.parametrize(
    "column,target", [("Ahshrtrd_D", "volume"), ("Ahvaltrd_D", "amount")]
)
def test_invalid_after_hours_facts_are_not_historical_structural_zero(
    bad, column, target
):
    row = dict.fromkeys(
        raw_exports.RAW_TABLE_CONTRACTS["market_bars_1d"].required_columns, 1.0
    )
    row.update(
        Stkcd="000001",
        Trddt="2019-01-02",
        Markettype=1,
        Filling=0,
        Ahshrtrd_D=None,
        Ahvaltrd_D=None,
    )
    row[column] = bad
    result = interim.normalize_raw_frame(pd.DataFrame([row]), "market_bars_1d")
    assert pd.isna(result[target].iloc[0])
    assert pd.isna(result[f"after_hours_{target}"].iloc[0])
    other = "amount" if target == "volume" else "volume"
    assert result[other].iloc[0] == 1.0
    assert result[f"after_hours_{other}"].iloc[0] == 0.0
    assert result.close.iloc[0] == 1.0
    assert result.attrs["source_row_counts"][f"invalid_after_hours_{target}_rows"] == 1


def _collect_raw_export(
    path: Path,
    dataset: str,
    *,
    chunksize: int = 1024,
) -> pd.DataFrame:
    chunks = _collect_raw_chunks(path, dataset, chunksize=chunksize)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _collect_raw_chunks(
    path: Path, dataset: str, *, chunksize: int
) -> list[pd.DataFrame]:
    return list(
        raw_exports.iter_private_export_chunks(
            path, dataset=dataset, chunksize=chunksize
        )
    )


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
    assert [path for _dataset, path in raw_exports.dataset_files_by_id(tmp_path)] == [
        csv_export
    ]
    selection = list(raw_exports.dataset_files_by_id(tmp_path))
    expected = _collect_raw_export(csv_export, "trading_calendar")
    sidecar = csv_export.with_suffix(".metadata.json")
    for payload in ("{}", "changed metadata"):
        filesystem_write_text(sidecar, payload, encoding="utf-8")
        assert list(raw_exports.dataset_files_by_id(tmp_path)) == selection
        pd.testing.assert_frame_equal(
            _collect_raw_export(csv_export, "trading_calendar"), expected
        )
    sidecar.unlink()
    sidecar.mkdir()
    assert list(raw_exports.dataset_files_by_id(tmp_path)) == selection
    pd.testing.assert_frame_equal(
        _collect_raw_export(csv_export, "trading_calendar"), expected
    )
    with pytest.raises(ValueError, match="Invalid CSMAR raw filename"):
        raw_schemas.parse_raw_filename(xlsx_export)


def test_csmar_preparation_evidence_is_selected_content_only(tmp_path: Path) -> None:
    bars = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    unrelated = tmp_path / (
        "csmar__stock_master__all_market_types__full__export-20260605.csv"
    )
    filesystem_write_text(
        bars,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns)
        + "\n",
        encoding="utf-8",
    )
    filesystem_write_text(
        unrelated,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["stock_master"].required_columns)
        + "\n",
        encoding="utf-8",
    )

    first = prepared_source_evidence(interim, tmp_path, "trading_calendar")
    assert first is not None

    sidecar = bars.with_suffix(".metadata.json")
    filesystem_write_text(sidecar, "{}", encoding="utf-8")
    assert prepared_source_evidence(interim, tmp_path, "trading_calendar") == first
    filesystem_write_text(sidecar, '{"row_count": 1}', encoding="utf-8")
    assert prepared_source_evidence(interim, tmp_path, "trading_calendar") == first

    filesystem_write_text(
        unrelated,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["stock_master"].required_columns)
        + "\n# unrelated\n",
        encoding="utf-8",
    )
    assert prepared_source_evidence(interim, tmp_path, "trading_calendar") == first

    filesystem_write_text(
        bars,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns)
        + ",unused\n",
        encoding="utf-8",
    )
    second = prepared_source_evidence(interim, tmp_path, "trading_calendar")
    assert second is not None
    assert second != first

    renamed = tmp_path / "renamed-calendar-file.csv"
    filesystem_rename(bars, renamed)
    assert prepared_source_evidence(interim, tmp_path, "trading_calendar") == second
    bars = renamed

    shard = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__part-01__export-20260605.csv"
    )
    filesystem_write_text(
        shard,
        ",".join(raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns)
        + "\n",
        encoding="utf-8",
    )
    third = prepared_source_evidence(interim, tmp_path, "trading_calendar")
    assert third is not None
    assert third != second
    filesystem_unlink(shard)
    assert prepared_source_evidence(interim, tmp_path, "trading_calendar") == second


def test_csmar_source_identity_uses_only_one_recursive_root(
    tmp_path: Path,
) -> None:
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

    identity = prepared_source_evidence(interim, primary, "trading_calendar")
    assert identity is not None
    raw_only = primary / "raw_only"
    filesystem_mkdir(raw_only)
    raw_only_file = raw_only / "native-calendar.csv"
    filesystem_write_text(
        raw_only_file,
        ",".join(columns) + "\n4,2020-01-03,5,C\n",
        encoding="utf-8",
    )
    assert prepared_source_evidence(interim, primary, "trading_calendar") == identity
    filesystem_write_text(
        raw_only_file,
        ",".join(columns) + "\n4,2020-01-04,6,O\n",
        encoding="utf-8",
    )
    assert prepared_source_evidence(interim, primary, "trading_calendar") == identity
    filesystem_unlink(raw_only_file)
    assert prepared_source_evidence(interim, primary, "trading_calendar") == identity
    assert prepared_source_evidence(interim, sibling, "trading_calendar") is not None
    assert prepared_source_evidence(interim, primary, "stock_master") is None


def test_csmar_identity_ignores_unrequested_bad_schema(
    tmp_path: Path,
) -> None:
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

    identity = prepared_source_evidence(interim, tmp_path, "trading_calendar")
    assert identity is not None


def test_csmar_source_identity_hashes_every_inventory_member(tmp_path):
    import hashlib

    payload = b"Markettype,Clddt,Daywk,State\n"
    for count in (1, 64):
        raw = tmp_path / str(count)
        filesystem_mkdir(raw)
        for index in range(count):
            filesystem_write_bytes(
                raw / f"csmar__trading_calendar__p{index}__full__export-20260101.csv",
                payload,
            )
        actual = prepared_source_evidence(interim, raw, "trading_calendar")
        expected = hashlib.sha256()
        for _ in range(count):
            for value in ("trading_calendar", hashlib.sha256(payload).hexdigest()):
                encoded = value.encode("utf-8")
                expected.update(len(encoded).to_bytes(8, "big"))
                expected.update(encoded)
        assert actual == expected.hexdigest()


def test_csmar_materialization_replays_each_dataset_from_one_inventory(
    tmp_path, monkeypatch
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

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
    expected = {}
    for data_id in data_ids:
        result = interim.materialize_interim_data(
            output_roots={
                key: single.interim_data_root("csmar", key) for key in (data_id,)
            },
            raw_root=raw,
            data_ids=(data_id,),
        )
        expected[data_id] = read_dataset(result.artifact_roots[data_id])
    inventory_calls = 0
    original_inventory = interim.dataset_files_by_id

    def counted_inventory(*args, **kwargs):
        nonlocal inventory_calls
        inventory_calls += 1
        return original_inventory(*args, **kwargs)

    monkeypatch.setattr(interim, "dataset_files_by_id", counted_inventory)
    result = interim.materialize_interim_data(
        output_roots={
            key: combined.interim_data_root("csmar", key) for key in data_ids
        },
        raw_root=raw,
        data_ids=data_ids,
    )
    assert inventory_calls == 1
    assert set(result.artifact_roots) == set(data_ids)
    for data_id in data_ids:
        actual = read_dataset(result.artifact_roots[data_id])
        pd.testing.assert_frame_equal(actual, expected[data_id], check_exact=True)
        assert len(actual) == 2


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
def test_financial_raw_retains_invalid_numeric_fact_and_interim_localizes_it(
    tmp_path: Path, dataset: str
) -> None:
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS

    path = tmp_path / f"{dataset}.csv"
    columns = raw_exports.RAW_TABLE_CONTRACTS[dataset].required_columns
    numeric_column = next(
        column
        for column in columns
        if column
        not in {"Stkcd", "ShortName", "IfCorrect", "DeclareDate", "Accper", "Typrep"}
    )
    bad = _financial_fixture_row(columns, invalid_numeric_column=numeric_column)
    good = _financial_fixture_row(columns)
    good[columns.index("Stkcd")] = "000002"
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows([bad, good])
    before = Path(native_filesystem_path(path)).read_bytes()
    decoded = _collect_raw_export(path, dataset, chunksize=1)
    assert decoded[numeric_column].tolist() == ["bad", "1"]
    normalized = interim.normalize_raw_frame(decoded, dataset).set_index(
        "source_asset_id"
    )
    target = COLUMN_MAPPINGS[dataset][numeric_column]
    assert pd.isna(normalized.loc["000001", target])
    assert normalized.loc["000002", target] == 1.0
    for column in (
        set(interim.FINANCIAL_COLUMNS_BY_DATA_ID[dataset])
        - set(interim.FINANCIAL_BASE_COLUMNS)
        - {target}
    ):
        assert normalized[column].tolist() == [1.0, 1.0]
    assert normalized.version_identity.tolist() == ["original", "original"]
    assert normalized.attrs["source_row_counts"][f"invalid_{target}_rows"] == 1
    assert Path(native_filesystem_path(path)).read_bytes() == before


def test_provider_header_accepts_extra_optional_fields(
    tmp_path: Path,
) -> None:
    required = raw_exports.RAW_TABLE_CONTRACTS["trading_calendar"].required_columns
    for width in (160, 161, 200):
        columns = (
            *required,
            *(f"Optional{index}" for index in range(width - len(required))),
        )
        path = tmp_path / f"provider-{width}.csv"
        filesystem_write_text(path, ",".join(columns) + "\n", encoding="utf-8")
        assert tuple(raw_schemas.read_export_header(path)) == columns
        selected = list(raw_exports.dataset_files_by_id(tmp_path))
        assert all(dataset == "trading_calendar" for dataset, _ in selected)
        assert _collect_raw_export(path, "trading_calendar").empty


def test_financial_correction_date_order_is_validated_locally() -> None:
    result = interim._parse_financial_correction_date(
        pd.Series(
            [
                "2020-02-01,2020-01-02",
                "2020-01-03",
                "",
                "2020-01-01,2020-02-01,2020-01-15",
                "2020-01-01,2020-01-01",
                "invalid-date",
            ]
        )
    )
    assert result.iloc[0] == pd.Timestamp("2020-02-01")
    assert result.iloc[1] == pd.Timestamp("2020-01-03")
    assert result.iloc[2:].isna().all()
    assert str(result.dtype) == "datetime64[ns]"


def test_csmar_identity_rejects_requested_bad_schema(
    tmp_path: Path,
) -> None:
    requested = tmp_path / (
        "csmar__trading_calendar__all_market_types__full__export-20260605.csv"
    )
    filesystem_write_text(requested, "bad,header\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing CSMAR columns"):
        prepared_source_evidence(interim, tmp_path, "trading_calendar")


@pytest.mark.parametrize("order", ("first_empty", "last_empty", "all_empty"))
@pytest.mark.parametrize("chunksize", (1, 2))
def test_analyst_fixed_projection_is_independent_of_empty_chunk_order(
    tmp_path, order, chunksize
):

    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    valid = _analyst_rows()[0]
    invalid = {**valid, "ReportID": "discarded", "DeclareDate": ""}
    rows = (
        [invalid, valid]
        if order == "first_empty"
        else [valid, invalid]
        if order == "last_empty"
        else [invalid]
    )
    _write_csmar(raw, "analyst_forecast", rows)
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("analyst_forecast",),
        chunksize=chunksize,
    )
    saved = read_dataset(result.artifact_roots["analyst_forecast"])
    assert "analyst_ids" in saved
    assert len(saved) == (0 if order == "all_empty" else 1)


def test_index_info_named_header_reordering_preserves_complete_row(tmp_path):
    fields = [
        "Indexcd",
        "Idxinfo01",
        "Idxinfo11",
        "Idxinfo02",
        "Idxinfo03",
        "Idxinfo04",
        "Idxinfo05",
        "Idxinfo06",
        "Idxinfo07",
        "Idxinfo08",
        "Idxinfo09",
        "Idxinfo10",
    ]
    row = [
        "000300",
        "CSI 300",
        "2014-01-02",
        "2013-12-31",
        "1000",
        "scope",
        "1",
        "2",
        "3",
        "4",
        "5",
        "formula",
    ]
    frames = []
    for name, positions in (
        ("native", list(range(12))),
        ("reordered", [1, 0, *range(2, 12)]),
    ):
        path = tmp_path / f"{name}.csv"
        with open_filesystem_file(path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([fields[i] for i in positions])
            writer.writerow([row[i] for i in positions])
        frames.append(_collect_raw_export(path, "index_info").loc[:, fields])
    pd.testing.assert_frame_equal(*frames)


def test_quoted_index_info_csv_preserves_semantics(tmp_path: Path) -> None:
    index_path = tmp_path / "index_info_quoted.csv"
    with open_filesystem_file(
        index_path, "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Indexcd",
                "Idxinfo01",
                "Idxinfo11",
                "Idxinfo02",
                "Idxinfo03",
                "Idxinfo04",
                "Idxinfo05",
                "Idxinfo06",
                "Idxinfo07",
                "Idxinfo08",
                "Idxinfo09",
                "Idxinfo10",
            ]
        )
        writer.writerows(
            [
                [
                    "000985",
                    "benchmark",
                    "2014-01-02",
                    "2013-12-31",
                    "-9666",
                    "all shares",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "formula, English comma, 中文标点",
                ],
                [
                    "000300",
                    "large",
                    "2014-01-03",
                    "2013-12-31",
                    "1000",
                    "blue chips",
                    "",
                    "4",
                    "7",
                    "5",
                    "9",
                    'formula with embedded line\nbreak and "quotes"',
                ],
                [
                    "000905",
                    "mid",
                    "2014-01-06",
                    "2013-12-31",
                    "500",
                    "text continues\nover physical line",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "formula after merge",
                ],
            ]
        )

    raw = _collect_raw_export(index_path, "index_info")
    normalized = interim.normalize_raw_frame(raw, "index_info")
    pruned = normalized.loc[:, list(interim.INTERIM_COLUMNS_BY_DATA_ID["index_info"])]

    assert len(raw) == len(pruned) == 3
    by_asset = pruned.set_index("source_asset_id")
    assert pd.isna(by_asset.loc["000985", "base_point"])
    assert by_asset.loc["000985", "calculation_formula"] == (
        "formula, English comma, 中文标点"
    )
    assert '"quotes"' in by_asset.loc["000300", "calculation_formula"]
    assert by_asset.loc["000300", "start_date"] == pd.Timestamp("2014-01-03")
    assert by_asset.loc["000300", "base_date"] == pd.Timestamp("2013-12-31")
    assert "embedded line\nbreak" in by_asset.loc[
        "000300",
        "calculation_formula",
    ].replace("\r\n", "\n")
    assert (
        by_asset.loc["000905", "sample_scope"] == "text continues\nover physical line"
    )
    assert by_asset.loc["000905", "calculation_formula"] == "formula after merge"


@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize(
    "isolated_column",
    (None, "Idxinfo05", "Idxinfo06", "Idxinfo07", "Idxinfo08", "Idxinfo09"),
)
def test_index_info_preserves_optional_source_codes_and_discloses_unknowns(
    tmp_path, chunksize, isolated_column
):
    import json

    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    row = dict(
        Indexcd="000985",
        Idxinfo01="benchmark",
        Idxinfo11="2014-01-02",
        Idxinfo02="2013-12-31",
        Idxinfo03="1000",
        Idxinfo04="scope",
        Idxinfo05="1",
        Idxinfo06="2",
        Idxinfo07="3",
        Idxinfo08="4",
        Idxinfo09="5",
        Idxinfo10="formula",
    )
    # Disjoint IDs retain each column's original logical group in one publication.
    code_groups = (
        ("Idxinfo05", 4),
        ("Idxinfo06", 4),
        ("Idxinfo07", 7),
        ("Idxinfo08", 5),
        ("Idxinfo09", 9),
    )
    rows = []
    unsupported = []
    unsupported_counts = {column: 0 for column, _maximum in code_groups}
    all_unsupported = isolated_column is not None
    if all_unsupported:
        # Keep the whole-column unsupported boundary, independent of other groups.
        code_groups = tuple(item for item in code_groups if item[0] == isolated_column)
    for group, (column, maximum) in enumerate(code_groups):
        supported = [
            dict(row, Indexcd=f"{group * 1000 + number:06d}", **{column: value})
            for number, value in enumerate(["", *map(str, range(1, maximum + 1))])
        ]
        bad_rows = [
            dict(row, Indexcd=f"{group * 1000 + 100 + number:06d}", **{column: value})
            for number, value in enumerate(
                (str(maximum + 1), "unknown", "-9666", "1;2")
            )
        ]
        other_column = "Idxinfo06" if column == "Idxinfo05" else "Idxinfo05"
        bad_rows[0][other_column] = "unknown"
        unsupported_counts[column] += 4
        unsupported_counts[other_column] += 1
        unsupported.extend(bad_rows)
        rows.extend(
            bad_rows
            if all_unsupported
            else [bad_rows[0], *supported, *bad_rows[1:], supported[0]]
        )
    extra = tuple(
        name
        for name in row
        if name not in raw_exports.RAW_TABLE_CONTRACTS["index_info"].required_columns
    )
    _write_csmar(raw, "index_info", rows, extra_columns=extra)
    raw_path = next(filesystem_glob(raw, "*.csv"))
    original = Path(native_filesystem_path(raw_path)).read_bytes()
    assert len(_collect_raw_export(raw_path, "index_info", chunksize=chunksize)) == len(
        rows
    )
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_info",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("index_info",),
        chunksize=chunksize,
    )
    assert not result.skipped
    root = result.artifact_roots["index_info"]
    saved = read_dataset(root)
    assert tuple(saved.columns) == interim.INDEX_INFO_INTERIM_COLUMNS
    assert set(saved.source_asset_id) == {item["Indexcd"] for item in rows}
    assert len(saved) == len({item["Indexcd"] for item in rows})
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS

    by_asset = saved.set_index("source_asset_id")
    for item in unsupported:
        for source_column in raw_exports.INDEX_INFO_SOURCE_CODE_COLUMNS:
            output_column = COLUMN_MAPPINGS["index_info"][source_column]
            assert (
                by_asset.loc[item["Indexcd"], output_column] == item[source_column]
            ), (item["Indexcd"], source_column)
    manifest = json.loads(
        Path(native_filesystem_path(root / "_manifest/manifest.json")).read_text(
            encoding="utf-8"
        )
    )
    retained = len(rows)
    assert manifest["producer_diagnostics"]["source_row_counts"] == {
        "raw_rows": len(rows),
        **{
            f"unsupported_{name}_rows": unsupported_counts[name]
            for name in raw_exports.INDEX_INFO_SOURCE_CODE_COLUMNS
        },
        "unsupported_source_code_rows": 4 * len(code_groups),
        "invalid_base_date_rows": 0,
        "invalid_start_date_rows": 0,
        "excluded_missing_timestamp_rows": 0,
        "retained_before_date_filter_rows": retained,
        "retained_after_date_filter_rows": retained,
        "excluded_date_range_rows": 0,
    }
    assert manifest["producer_diagnostics"]["exact_duplicate_count"] == (
        len(code_groups) * int(not all_unsupported)
    )
    assert Path(native_filesystem_path(raw_path)).read_bytes() == original


def test_index_info_structure_failure_keeps_atomic_failure_and_old_good(tmp_path):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    good = dict(
        Indexcd="000905",
        Idxinfo01="benchmark",
        Idxinfo11="2014-01-02",
        Idxinfo02="2013-12-31",
        Idxinfo05="1",
    )
    _write_csmar(raw, "index_info", [good], extra_columns=("Idxinfo05",))
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_info",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("index_info",),
    )
    root = result.artifact_roots["index_info"]
    before = _file_bytes(root)
    _write_csmar(
        raw,
        "index_info",
        [
            good,
            dict(good, Indexcd="000852", Idxinfo05="unknown", Idxinfo11="invalid-date"),
        ],
        extra_columns=("Idxinfo05",),
    )
    raw_path = next(filesystem_glob(raw, "*.csv"))
    with raw_path.open("a", encoding="utf-8") as stream:
        stream.write("broken,row\n")
    candidate = StorageLayout(tmp_path / "candidate")
    with pytest.raises(ValueError, match="malformed_csmar_csv_row"):
        interim.materialize_interim_data(
            output_roots={
                key: candidate.interim_data_root("csmar", key)
                for key in ("index_info",)
            },
            raw_root=raw,
            data_ids=("index_info",),
            chunksize=100,
        )
    assert _file_bytes(root) == before
    assert len(read_dataset(root)) == 1
    assert not filesystem_exists(candidate.interim_data_root("csmar", "index_info"))
    assert not tuple(filesystem_glob(tmp_path / "candidate", "**/.a-*"))


@pytest.mark.parametrize("unknown_code", ("000905", "000852"))
def test_index_info_benchmark_ignores_optional_unknown_codes(tmp_path, unknown_code):
    from quant_research_platform.markets.implementations.a_share.benchmark import (
        iter_benchmark_weight_snapshots,
    )
    from quant_research_platform.markets.implementations.a_share.ids import (
        FORMAL_BENCHMARKS,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    rows = [
        dict(
            Indexcd=code,
            Idxinfo01=name,
            Idxinfo11="2014-01-02",
            Idxinfo02="2013-12-31",
            Idxinfo05="5;3" if code == unknown_code else "1",
        )
        for code, (_, name) in FORMAL_BENCHMARKS.items()
    ]
    _write_csmar(raw, "index_info", rows, extra_columns=("Idxinfo05",))
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_info",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("index_info",),
        chunksize=1,
    )
    saved = read_dataset(result.artifact_roots["index_info"])
    assert set(saved.source_asset_id) == set(FORMAL_BENCHMARKS)
    for code in FORMAL_BENCHMARKS:
        weights = pd.DataFrame(
            dict(
                source_asset_id=[code],
                constituent_asset_id=["000001"],
                effective_date=["2020-01-02"],
                weight=[1.0],
            )
        )
        actual = pd.concat(iter_benchmark_weight_snapshots(saved, [weights]))
        assert set(actual.benchmark_id) == {FORMAL_BENCHMARKS[code][0]}
        wrong_identity = saved.copy()
        wrong_identity.loc[wrong_identity.source_asset_id.eq(code), "index_name"] = (
            "wrong"
        )
        with pytest.raises(
            ValueError, match="^a_share_benchmark_index_identity_invalid$"
        ):
            list(iter_benchmark_weight_snapshots(wrong_identity, [weights]))


@pytest.mark.parametrize(
    "tail",
    [
        "scope,extra,1,2,3,4,5,formula,tail",
        "scope,1,2,3,4,5,ambiguous,1,2,3,4,5,tail",
        "text continues\nover physical line,1,2,3,4,5,formula",
        'scope,1,2,3,4,5,"unterminated',
    ],
)
def test_index_info_rejects_unrepaired_records(tmp_path: Path, tail) -> None:
    index_path = tmp_path / "index_info_malformed.csv"
    filesystem_write_text(
        index_path,
        "Indexcd,Idxinfo01,Idxinfo11,Idxinfo02,Idxinfo03,Idxinfo04,"
        "Idxinfo05,Idxinfo06,Idxinfo07,Idxinfo08,Idxinfo09,Idxinfo10\n"
        "000985,benchmark,2014-01-02,2013-12-31,1000," + tail + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
        _collect_raw_export(index_path, "index_info")


@pytest.mark.parametrize("reordered", [False, True])
def test_index_info_interim_localizes_bad_named_code_and_raw_preserves_it(
    tmp_path, reordered
):
    fields = ["Indexcd", "Idxinfo01", "Idxinfo11", "Idxinfo02"]
    bad = ["invalid code", "000300", "2014-01-02", "2013-12-31"]
    good = ["000905", "independent index", "2014-01-02", "2013-12-31"]
    positions = [1, 0, 2, 3] if reordered else list(range(4))
    path = tmp_path / "index.csv"
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([fields[i] for i in positions])
        writer.writerows([[row[i] for i in positions] for row in [bad, good]])
    before = Path(native_filesystem_path(path)).read_bytes()
    raw = _collect_raw_export(path, "index_info")
    assert raw.Indexcd.tolist() == ["invalid code", "000905"]
    assert raw.Idxinfo01.tolist() == ["000300", "independent index"]
    normalized = interim.normalize_raw_frame(raw, "index_info")
    assert normalized.source_asset_id.tolist() == ["000905"]
    assert normalized.index_name.tolist() == ["independent index"]
    assert normalized.timestamp.tolist() == [pd.Timestamp("2014-01-02")]
    assert normalized.attrs["source_row_counts"]["excluded_missing_required_rows"] == 1
    assert Path(native_filesystem_path(path)).read_bytes() == before


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
    original_close = raw_schemas._CsvRecords.close

    def interrupted(_self):
        raise primary

    def failing_close(self):
        nonlocal close_calls
        close_calls += 1
        original_close(self)
        raise OSError("header_close_failed")

    monkeypatch.setattr(raw_schemas._CsvRecords, "__next__", interrupted)
    monkeypatch.setattr(raw_schemas._CsvRecords, "close", failing_close)
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
def test_legal_provider_width_and_quoting_preserve_fields(tmp_path, quoted):
    width = 160
    path = tmp_path / "legal.csv"
    fields = [f"field_{position}" for position in range(width)]
    if quoted:
        fields[0] = '字段,"示例"'
    with open_filesystem_file(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle, quoting=csv.QUOTE_ALL if quoted else csv.QUOTE_MINIMAL
        )
        writer.writerows((fields, ["1"] * width))

    records = raw_schemas.iter_export_csv_records(path)
    try:
        header = next(records)
        assert raw_schemas.validate_export_header(header) == fields
        record = next(records)
        assert record == ["1"] * width
        with pytest.raises(StopIteration):
            next(records)
    finally:
        records.close()


@pytest.mark.parametrize(
    "data_id,single",
    [
        ("market_bars_1d", False),
        ("analyst_forecast", False),
        ("analyst_forecast", True),
        ("trading_calendar", False),
    ],
)
def test_csmar_exact_duplicates_and_unordered_chunks_persist_identical_content(
    tmp_path, data_id, single
):
    import hashlib

    from quant_research_platform.storage.datasets.comparison import compare_datasets
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

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
    if single:
        source = source[:1]
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
                Path(
                    native_filesystem_path(next(filesystem_glob(raw, "*.csv")))
                ).read_bytes()
            ).hexdigest()
        )
        result = interim.materialize_interim_data(
            output_roots={
                key: layout.interim_data_root("csmar", key) for key in (data_id,)
            },
            raw_root=layout.raw_source_root("csmar"),
            data_ids=(data_id,),
            chunksize=size,
        )
        root = result.artifact_roots[data_id]
        roots.append(root)
        saved = read_dataset(root)
        frames.append(saved.sort_values(list(spec.primary_key)).reset_index(drop=True))
        metadata = read_dataset_manifest_evidence(root)["producer_diagnostics"]
        assert metadata["exact_duplicate_count"] == 1
        if data_id == "analyst_forecast":
            counts = metadata["source_row_counts"]
            assert counts["raw_rows"] == len(rows)
            assert counts["excluded_missing_required_rows"] + len(saved) + metadata[
                "exact_duplicate_count"
            ] == len(rows)
            if single:
                assert len(saved) == 1
            else:
                assert {"9007199254740992", "9007199254740993"}.issubset(
                    saved.report_id
                )
            assert "张三,李四" in set(saved.analyst_ids.dropna())
            if not single:
                assert saved.forecast_eps.isna().any()
        assert str(saved.timestamp.dtype) == "datetime64[ns]"
        assert not saved.duplicated(list(spec.primary_key)).any()
    assert len(set(hashes)) == 1
    for frame in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], frame, check_exact=True)
    for root in roots[1:]:
        assert compare_datasets(roots[0], root).status == "MATCH"


@pytest.mark.parametrize("chunksize", (1, 2))
@pytest.mark.parametrize(
    "raw_column,value,column",
    [("Feps", 999, "forecast_eps"), ("InstitutionID", "disputed", "institution_id")],
)
def test_analyst_same_key_conflict_preserves_independent_reports_and_prior_dataset(
    tmp_path, chunksize, raw_column, value, column
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    first = _analyst_rows()[0]
    _write_csmar(raw, "analyst_forecast", [first])
    original = tmp_path / "original"
    interim.materialize_interim_data(
        output_roots={"analyst_forecast": original},
        raw_root=raw,
        data_ids=("analyst_forecast",),
    )
    old_good = _file_bytes(original)
    neighbor = {
        **first,
        "ReportID": "independent-report",
        "InstitutionID": "B",
        "Feps": 12,
    }
    _write_csmar(
        raw, "analyst_forecast", [first, {**first, raw_column: value}, neighbor]
    )
    raw_before = _file_bytes(raw)
    candidate = tmp_path / "candidate"
    interim.materialize_interim_data(
        output_roots={"analyst_forecast": candidate},
        raw_root=raw,
        data_ids=("analyst_forecast",),
        chunksize=chunksize,
    )
    saved = read_dataset(candidate).set_index("report_id")
    assert len(saved) == 2
    assert pd.isna(saved.loc[first["ReportID"], column])
    assert saved.loc[first["ReportID"], "forecast_pe"] == 10
    if column == "forecast_eps":
        assert saved.loc[first["ReportID"], "institution_id"] == "A"
    else:
        assert saved.loc[first["ReportID"], "forecast_eps"] == 18
    assert saved.loc["independent-report", "forecast_eps"] == 12
    assert saved.loc["independent-report", "institution_id"] == "B"
    assert "version_identity" not in saved
    counts = read_dataset_manifest_evidence(candidate)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts[f"conflicting_{column}_rows"] == 2
    assert counts["raw_rows"] == 3
    assert _file_bytes(original) == old_good
    assert _file_bytes(raw) == raw_before


@pytest.mark.parametrize("primary_type", (MemoryError, OSError, KeyboardInterrupt))
def test_csmar_preparation_keeps_fatal_source_failure(
    tmp_path, monkeypatch, primary_type
):
    path = tmp_path / "native-calendar.csv"
    filesystem_write_text(
        path, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
    )
    primary = primary_type("primary")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(raw_schemas, "open_filesystem_file", fail)
    with pytest.raises(primary_type) as caught:
        prepared_source_evidence(interim, tmp_path, "trading_calendar")
    assert caught.value is primary


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
@pytest.mark.parametrize("chunksize", (1, 100))
def test_special_treatment_unknown_transition_keeps_event_and_valid_neighbor(
    tmp_path, code, chunksize
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    row = {
        "Symbol": "000001",
        "Chgtype": code,
        "Annoudt": "2020-01-02",
        "Execudt": "2020-01-01",
    }
    _write_csmar(
        raw,
        "special_treatment_changes",
        [row, {**row, "Symbol": "000002", "Chgtype": "AB"}],
    )
    before = _file_bytes(raw)
    output = tmp_path / "interim"
    interim.materialize_interim_data(
        raw_root=raw,
        data_ids=("special_treatment_changes",),
        output_roots={"special_treatment_changes": output},
        chunksize=chunksize,
    )
    saved = read_dataset(output).set_index("source_asset_id")
    assert len(saved) == 2
    assert saved.loc["000001", "change_type"] == code
    assert saved.loc["000001", ["is_st", "is_pt"]].isna().all()
    assert saved.loc["000001", "timestamp"] == pd.Timestamp("2020-01-01")
    assert saved.loc["000001", "announcement_date"] == pd.Timestamp("2020-01-02")
    assert bool(saved.loc["000002", "is_st"])
    assert not bool(saved.loc["000002", "is_pt"])
    assert str(saved.is_st.dtype) == str(saved.is_pt.dtype) == "boolean"
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["unsupported_transition_rows"] == 1
    assert counts["raw_rows"] == counts["retained_after_date_filter_rows"] == 2
    assert _file_bytes(raw) == before


@pytest.mark.parametrize("code", ["", " ", None])
@pytest.mark.parametrize("all_bad", [False, True])
def test_special_treatment_blank_event_key_is_local_and_typed(tmp_path, code, all_bad):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    row = {
        "Symbol": "000001",
        "Chgtype": code,
        "Annoudt": "2020-01-02",
        "Execudt": "2020-01-01",
    }
    rows = [row] if all_bad else [row, {**row, "Symbol": "000002", "Chgtype": "AB"}]
    _write_csmar(raw, "special_treatment_changes", rows)
    before = _file_bytes(raw)
    output = tmp_path / "interim"
    interim.materialize_interim_data(
        raw_root=raw,
        data_ids=("special_treatment_changes",),
        output_roots={"special_treatment_changes": output},
        chunksize=1,
    )
    saved = read_dataset(output)
    assert saved.source_asset_id.tolist() == ([] if all_bad else ["000002"])
    assert str(saved.timestamp.dtype) == "datetime64[ns]"
    assert str(saved.is_st.dtype) == str(saved.is_pt.dtype) == "boolean"
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == len(rows)
    assert counts["excluded_missing_required_rows"] == 1
    assert counts["retained_after_date_filter_rows"] == len(rows) - 1
    assert _file_bytes(raw) == before


@pytest.mark.parametrize("announcement", ["invalid", "9999-12-31", "", None])
def test_special_treatment_unknown_announcement_never_fabricates_visibility(
    announcement,
):
    result = interim.normalize_raw_frame(
        pd.DataFrame(
            [
                {
                    "Symbol": "000001",
                    "Chgtype": "AB",
                    "Annoudt": announcement,
                    "Execudt": "2020-01-01",
                },
                {
                    "Symbol": "000002",
                    "Chgtype": "BA",
                    "Annoudt": "2020-01-02",
                    "Execudt": "2020-01-01",
                },
            ]
        ),
        "special_treatment_changes",
    ).set_index("source_asset_id")
    assert len(result) == 2
    assert pd.isna(result.loc["000001", "announcement_date"])
    assert result.timestamp.eq(pd.Timestamp("2020-01-01")).all()
    assert result.loc["000002", "announcement_date"] == pd.Timestamp("2020-01-02")
    assert bool(result.loc["000001", "is_st"])
    assert not bool(result.loc["000002", "is_st"])
    if announcement:
        assert result.attrs["source_row_counts"]["invalid_announcement_date_rows"] == 1


@pytest.mark.parametrize("chunksize,bad_numeric", ((1, False), (8, False), (1, True)))
def test_index_membership_raw_text_survives_public_materialization(
    tmp_path, chunksize, bad_numeric
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

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
    original = Path(native_filesystem_path(path)).read_bytes()
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_membership",)
        },
        raw_root=raw,
        data_ids=("index_membership",),
        chunksize=chunksize,
    )
    frame = read_dataset(result.artifact_roots["index_membership"])
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
    if bad_numeric:
        assert frame.change_type.isna().all()
        from quant_research_platform.storage.manifests import (
            read_dataset_manifest_evidence,
        )

        counts = read_dataset_manifest_evidence(
            result.artifact_roots["index_membership"]
        )["producer_diagnostics"]["source_row_counts"]
        assert counts["invalid_change_type_rows"] == 2
    else:
        assert frame.change_type.tolist() == [1.0, 1.0]
    assert Path(native_filesystem_path(path)).read_bytes() == original


@pytest.mark.parametrize("chunksize", [1, 10])
def test_index_membership_preserves_native_codes_and_venue_type_identity(
    tmp_path, chunksize
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

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
    imported = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_membership",)
        },
        raw_root=raw,
        data_ids=("index_membership",),
        chunksize=chunksize,
    )
    saved = read_dataset(imported.artifact_roots["index_membership"])
    actual = {
        (code, "" if pd.isna(market) else market, int(kind))
        for code, market, kind in saved[
            ["constituent_stock_code", "market_code", "security_type"]
        ].itertuples(index=False, name=None)
    }
    assert actual == set(identities)
    assert len(saved) == saved.constituent_asset_id.nunique() == len(identities)
    assert saved.market_code.isna().sum() == 1


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


@pytest.mark.parametrize(
    "failure_type", (FileNotFoundError, NotADirectoryError, OSError, PermissionError)
)
def test_csmar_parent_failure_keeps_workflow_semantics(
    tmp_path, monkeypatch, failure_type
):
    from tests.helpers.paths import StorageLayout

    raw = tmp_path / "raw/csmar"
    filesystem_mkdir(raw, parents=True)
    source = raw / "native-calendar.csv"
    filesystem_write_text(
        source, "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n", encoding="utf-8"
    )
    arguments = dict(
        raw_root=raw,
        start_date="2020-01-01",
        end_date="2020-01-03",
        data_ids=("trading_calendar",),
    )
    normal_layout = StorageLayout(tmp_path / "normal")
    failed_layout = StorageLayout(tmp_path / "failed")
    interim.materialize_interim_data(
        output_roots={"trading_calendar": normal_layout.project_root / "calendar"},
        **arguments,
    )
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
        error_type = (
            ValueError
            if failure_type in (FileNotFoundError, NotADirectoryError)
            else failure_type
        )
        with pytest.raises(error_type) as caught:
            interim.materialize_interim_data(
                output_roots={
                    "trading_calendar": failed_layout.project_root / "calendar"
                },
                **arguments,
            )
        if error_type is ValueError:
            assert str(caught.value) == "csmar_interim_source_parent_invalid"
        else:
            assert caught.value is primary
    assert calls == [raw]
    assert _file_bytes(normal_layout.project_root) == originals
    assert not tuple(filesystem_glob(failed_layout.project_root, "**/.a-*"))
    assert not tuple(filesystem_glob(failed_layout.project_root, "**/manifest.json"))


@pytest.mark.parametrize("price_scale", (0.5, 20.0, 800.0))
@pytest.mark.parametrize("split", (False, True))
def test_adjusted_open_preserves_units_through_storage_and_labels(
    tmp_path, price_scale, split
):
    import numpy as np

    from quant_research_platform.research.labels.forward_returns import (
        add_forward_return_labels,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    sessions = pd.date_range("2019-08-01", periods=3)
    prices = [price_scale * multiplier for multiplier in (1.0, 1.0, 1.5)]
    rows = []
    for position, (timestamp, price) in enumerate(zip(sessions, prices, strict=True)):
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
        if split:
            split_price = prices[0] / (2 if position == 2 else 1)
            row.update(
                Opnprc=split_price,
                Clsprc=split_price,
                Hiprc=split_price,
                Loprc=split_price,
                Adjprcwd=prices[0],
            )
        rows.append(row)
    _write_csmar(raw, "market_bars_1d", rows)
    built = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("market_bars_1d",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("market_bars_1d",),
    )
    loaded = read_dataset(built.artifact_roots["market_bars_1d"])
    np.testing.assert_allclose(loaded.adj_open, loaded.adj_close, rtol=1e-14, atol=0)
    panel = loaded.rename(columns={"source_asset_id": "asset_id"}).assign(
        is_trading=True
    )
    labeled = add_forward_return_labels(panel, horizons=(1,), sessions=sessions)
    assert labeled.forward_return_1d.iloc[0] == pytest.approx(0.0 if split else 0.5)
    assert labeled.forward_return_1d_available_at.iloc[0] == sessions[2] + pd.Timedelta(
        hours=7
    )
    assert labeled.forward_return_1d.iloc[1:].isna().all()


@pytest.mark.parametrize(
    "opening,adjusted,closing",
    [
        (12.0, 20.0, 10.0),
        (2.35, 4.71, 2.34),
        (799.3, 1201.7, 800.1),
    ],
)
def test_adjusted_open_matches_independent_price_ratio(opening, adjusted, closing):
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
    assert actual.adj_open.iloc[0] == pytest.approx(expected, rel=1e-14, abs=0)


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


def test_raw_table_identity_ignores_parent_directory_and_rejects_conflict(tmp_path):
    payload = "Stkcd,Shrchgdt,Nshrttl,Nshra\n000001,2024-01-02,100,50\n"
    outputs = []
    for directory in ("stock_master", "market_cap"):
        root = tmp_path / directory
        filesystem_mkdir(root)
        path = root / "unknown.csv"
        filesystem_write_text(path, payload, encoding="utf-8")
        [(dataset, selected_path)] = list(raw_exports.dataset_files_by_id(root))
        assert selected_path == path
        outputs.append(dataset)
        frame = _collect_raw_export(path, dataset)
        assert frame.loc[0, "Stkcd"] == "000001"
    assert outputs == ["market_cap", "market_cap"]
    conflict = tmp_path / "TRDNEW_Co.csv"
    filesystem_write_text(conflict, payload, encoding="utf-8")
    with pytest.raises(ValueError, match="table_identity_conflict"):
        for dataset, path in raw_exports.dataset_files_by_id(tmp_path):
            _collect_raw_export(path, dataset)


def test_optional_csmar_dates_keep_invalid_cells_unknown_without_losing_neighbors():
    result = interim._parse_optional_csmar_date(
        pd.Series(["invalid-date", "9999-12-31", "", " ", None, "2020-01-01"])
    )
    assert result.isna().tolist() == [True, True, True, True, True, False]
    assert result.iloc[-1] == pd.Timestamp("2020-01-01")
    assert str(result.dtype) == "datetime64[ns]"


@pytest.mark.parametrize("bad", ["invalid-number", float("inf"), -float("inf")])
def test_optional_csmar_numeric_cells_are_local_and_diagnosed(bad):
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS

    datasets = [
        ("analyst_forecast", _analyst_rows()[0], ("forecast_eps", "forecast_pe"))
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
        datasets.append((data_id, row, numeric))
    checked = 0
    for data_id, row, columns in datasets:
        inverse = {value: key for key, value in COLUMN_MAPPINGS[data_id].items()}
        valid = interim.normalize_raw_frame(pd.DataFrame([row]), data_id)
        for column in columns:
            assert float(valid.loc[0, column]) == float(row[inverse[column]])
            assert pd.api.types.is_float_dtype(valid[column].dtype)
            result = interim.normalize_raw_frame(
                pd.DataFrame(
                    [{**row, inverse[column]: bad}, {**row, "Stkcd": "000002"}]
                ),
                data_id,
            ).set_index("source_asset_id")
            assert len(result) == 2
            assert pd.isna(result.loc["000001", column])
            assert result.loc["000002", column] == valid.loc[0, column]
            for other in set(columns) - {column}:
                assert result[other].tolist() == [valid.loc[0, other]] * 2
            assert result.attrs["source_row_counts"][f"invalid_{column}_rows"] == 1
            checked += 1
    assert checked == 16


def test_index_date_error_is_diagnosed_without_blocking_valid_base_date(tmp_path):

    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

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
    from quant_research_platform.storage.datasets.reading import read_dataset

    result = interim.materialize_interim_data(
        raw_root=raw,
        output_roots={"index_info": tmp_path / "index-info"},
        data_ids=("index_info",),
    )
    saved = read_dataset(result.artifact_roots["index_info"])
    assert len(saved) == 1
    assert saved["timestamp"].iloc[0] == pd.Timestamp("2020-01-01")
    assert pd.isna(saved["start_date"].iloc[0])


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
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, data_id, _nullable_index_numeric_rows(data_id))
    original = _file_bytes(raw)
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in (data_id,)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=(data_id,),
        chunksize=chunksize,
    )
    root = result.artifact_roots[data_id]
    saved = read_dataset(root).sort_values(
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
def test_nullable_index_source_invalid_numeric_values_stay_local(
    data_id, raw_column, bad_value, reason
):
    rows = _nullable_index_numeric_rows(data_id)
    rows[0][raw_column] = bad_value
    result = interim.normalize_raw_frame(pd.DataFrame(rows), data_id)
    target = "weight" if data_id == "index_weights" else "period_return"
    assert len(result) == len(rows)
    assert pd.isna(result[target].iloc[0])
    expected = [0.25, 0.25, 0.5] if target == "weight" else [-0.025, 0.0, 0.025]
    assert result[target].iloc[1:].tolist() == expected
    assert str(result[target].dtype) == "float64"
    diagnostic = (
        "weight"
        if bad_value == -1.0
        else "weight_pct"
        if target == "weight"
        else "return_1d_pct"
    )
    assert result.attrs["source_row_counts"][f"invalid_{diagnostic}_rows"] == 1


@pytest.mark.parametrize(
    "payload,expected",
    (
        (b'"1",2\r\n', [["1", "2"]]),
        (b"a,2\r", [["a", "2"]]),
        (b"a,2\rb,3", [["a", "2"], ["b", "3"]]),
        (b"a,2\r\nb,3\rc,4\n", [["a", "2"], ["b", "3"], ["c", "4"]]),
        ('"中文😀\r\n引号""",2\r'.encode(), [['中文😀\r\n引号"', "2"]]),
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
def test_csv_strict_grammar_matches_standard_parser(tmp_path, payload, expected):
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

    records = raw_schemas.iter_export_csv_records(path)
    actual = []
    try:
        if expected is None:
            with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
                next(records)
        else:
            for record in records:
                actual.append(record)
            assert actual == expected
    finally:
        records.close()


def test_malformed_quoted_numeric_cannot_publish_over_old_good(tmp_path):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    filesystem_mkdir(raw, parents=True)
    _write_csmar(raw, "analyst_forecast", _analyst_rows()[:1])
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("analyst_forecast",),
    )
    root = result.artifact_roots["analyst_forecast"]
    old_good = _file_bytes(root)
    before = read_dataset(root)
    path = next(filesystem_glob(raw, "*.csv"))
    payload = Path(native_filesystem_path(path)).read_bytes()
    assert payload.count(b",18,10") == 1
    malformed = payload.replace(b",18,10", b',"1"0,10')
    filesystem_write_bytes(path, malformed)
    candidate = tmp_path / "new-candidate"
    with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
        interim.materialize_interim_data(
            output_roots={"analyst_forecast": candidate},
            raw_root=raw,
            data_ids=("analyst_forecast",),
        )
    assert not filesystem_exists(candidate)
    assert _file_bytes(root) == old_good
    pd.testing.assert_frame_equal(before, read_dataset(root))
    assert Path(native_filesystem_path(path)).read_bytes() == malformed


def _short_analyst_csv_records(count):
    header = b"Stkcd,ReportID,AnanmID,DeclareDate,Fenddt,InstitutionID,Feps,Fpe"
    rows = []
    for position in range(count):
        fields = [
            f"{position % 3 + 1:06}",
            str(9007199254740993 + position),
            "000007" if position % 3 else "",
            "2025-01-03",
            "2025-12-31",
            "000009",
            "2.5" if position % 3 != 1 else "",
            "10.5" if position % 3 != 2 else "",
        ]
        rows.append(fields)
    return header, rows


@pytest.fixture
def real_csv_handles(monkeypatch):
    handles = []
    real_open = raw_schemas.open_filesystem_file

    def open_source(*args, **kwargs):
        handle = real_open(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(raw_schemas, "open_filesystem_file", open_source)
    yield handles
    assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize(
    "endings,terminated",
    (
        ((b"\n",), True),
        ((b"\r\n",), True),
        ((b"\r",), True),
        ((b"\r",), False),
        ((b"\r", b"\r\n", b"\n"), True),
    ),
)
def test_cr_only_physical_fragments_preserve_interim_rows(
    tmp_path, real_csv_handles, endings, terminated
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.paths import StorageLayout

    count = 3
    header, rows = _short_analyst_csv_records(count)
    records = [header, *(",".join(row).encode() for row in rows)]
    payload = b"".join(
        record
        + (endings[position % len(endings)] if terminated or position < count else b"")
        for position, record in enumerate(records)
    )
    layout = StorageLayout(tmp_path)
    raw_root = layout.raw_source_root("csmar")
    filesystem_mkdir(raw_root, parents=True)
    path = raw_root / "csmar__analyst_forecast__synthetic__full__export-20260101.csv"
    filesystem_write_bytes(path, payload)
    expected = pd.DataFrame(
        [
            {
                "source_asset_id": row[0],
                "timestamp": pd.Timestamp(row[3]),
                "report_id": row[1],
                "analyst_ids": row[2] or pd.NA,
                "declare_date": pd.Timestamp(row[3]),
                "forecast_end_date": pd.Timestamp(row[4]),
                "institution_id": row[5],
                "forecast_eps": float(row[6]) if row[6] else float("nan"),
                "forecast_pe": float(row[7]) if row[7] else float("nan"),
            }
            for row in rows
        ]
    )
    for column in ("source_asset_id", "report_id", "analyst_ids", "institution_id"):
        expected[column] = expected[column].astype("string")
    for column in ("timestamp", "declare_date", "forecast_end_date"):
        expected[column] = expected[column].astype("datetime64[ns]")
    for column in ("forecast_eps", "forecast_pe"):
        expected[column] = expected[column].astype("float64")
    keys = ["source_asset_id", "timestamp", "report_id", "forecast_end_date"]
    expected = expected.sort_values(keys).reset_index(drop=True)
    assert raw_schemas.read_export_header(path) == tuple(header.decode().split(","))
    identity = prepared_source_evidence(interim, raw_root, "analyst_forecast")
    assert identity is not None
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=raw_root,
        data_ids=("analyst_forecast",),
    )
    saved = read_dataset(result.artifact_roots["analyst_forecast"])
    assert len(saved) == count and not saved.duplicated(keys).any()
    pd.testing.assert_frame_equal(
        saved.sort_values(keys).reset_index(drop=True), expected, check_exact=True
    )
    assert prepared_source_evidence(interim, raw_root, "analyst_forecast") == identity
    assert Path(native_filesystem_path(path)).read_bytes() == payload


def test_long_record_after_short_header_can_publish(tmp_path, real_csv_handles):
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.paths import StorageLayout

    header, rows = _short_analyst_csv_records(1)
    row = rows[0]
    # One field exceeds Python csv's default 128 KiB field limit.
    row[5] = "x" * (128 * 1024 + 1)
    record = ",".join(row).encode()
    payload = header + b"\n" + record + b"\n"
    layout = StorageLayout(tmp_path)
    raw_root = layout.raw_source_root("csmar")
    filesystem_mkdir(raw_root, parents=True)
    path = raw_root / "csmar__analyst_forecast__synthetic__full__export-20260101.csv"
    filesystem_write_bytes(path, payload)
    target = layout.interim_data_root("csmar", "analyst_forecast")
    assert not filesystem_exists(target)
    assert raw_schemas.read_export_header(path) == tuple(header.decode().split(","))
    assert prepared_source_evidence(interim, raw_root, "analyst_forecast") is not None
    interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("analyst_forecast",)
        },
        raw_root=raw_root,
        data_ids=("analyst_forecast",),
    )
    saved = read_dataset(target)
    assert len(saved) == 1
    assert saved["institution_id"].iloc[0] == row[5]
    assert Path(native_filesystem_path(path)).read_bytes() == payload


def test_cr_reader_early_close_releases_real_handles(tmp_path, real_csv_handles):
    header, rows = _short_analyst_csv_records(3)
    payload = b"\r".join([header, *(",".join(row).encode() for row in rows)]) + b"\r"
    path = tmp_path / "early-close.csv"
    filesystem_write_bytes(path, payload)
    records = raw_schemas.iter_export_csv_records(path)
    try:
        record = next(records)
        assert record == header.decode().split(",")
        record = next(records)
        assert record == rows[0]
    finally:
        records.close()
    assert Path(native_filesystem_path(path)).read_bytes() == payload


@pytest.mark.parametrize("payload", (b"bad\xff,2\n", b'"bad\r\n\xff",2\r\n'))
def test_invalid_utf8_is_rejected_in_csv_records(tmp_path, real_csv_handles, payload):
    path = tmp_path / "invalid-utf8.csv"
    filesystem_write_bytes(path, payload)
    records = raw_schemas.iter_export_csv_records(path)
    try:
        with pytest.raises(ValueError, match="^malformed_csmar_csv_row$"):
            next(records)
    finally:
        records.close()
    assert Path(native_filesystem_path(path)).read_bytes() == payload


@pytest.mark.parametrize("ending", ("\n", "\r", "\r\n"))
def test_quoted_csv_record_preserves_embedded_newlines(tmp_path, ending):
    import csv
    import io

    value = 'prefix\r\n"quoted",tail'
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator=ending).writerow([value, "second"])
    payload = buffer.getvalue().encode()
    path = tmp_path / "quoted.csv"
    filesystem_write_bytes(path, payload)
    assert list(raw_schemas.iter_export_csv_records(path)) == [[value, "second"]]


@pytest.mark.parametrize(
    "omitted",
    (
        None,
        "Idxinfo03",
        "Idxinfo04",
        "Idxinfo05",
        "Idxinfo06",
        "Idxinfo07",
        "Idxinfo08",
        "Idxinfo09",
        "Idxinfo10",
    ),
)
def test_index_info_projects_extra_text_and_persists_optional_columns(
    tmp_path, omitted
):
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw = layout.raw_source_root("csmar")
    raw.mkdir(parents=True)
    row = dict(
        zip(
            COLUMN_MAPPINGS["index_info"],
            (
                "000300",
                "CSI 300",
                "2014-01-02",
                "2013-12-31",
                "1000",
                "scope",
                "1",
                "2",
                "3",
                "4",
                "5",
                "formula",
            ),
            strict=True,
        )
    )
    if omitted:
        row.pop(omitted)
    row["VendorDescription"] = "unused provider text"
    fields = list(row)
    fields.reverse()
    path = raw / "csmar__index_info__synthetic__full__export-20260101.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    original = path.read_bytes()
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_info",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("index_info",),
    )
    saved = read_dataset(result.artifact_roots["index_info"])
    assert tuple(saved.columns) == interim.INDEX_INFO_INTERIM_COLUMNS
    assert saved.source_asset_id.tolist() == ["000300"]
    assert saved.index_name.tolist() == ["CSI 300"]
    assert "VendorDescription" not in saved
    if omitted:
        column = COLUMN_MAPPINGS["index_info"][omitted]
        assert saved[column].isna().all()
        assert str(saved[column].dtype) == (
            "float64" if column == "base_point" else "string"
        )
    assert path.read_bytes() == original


def test_index_info_name_is_a_required_source_fact(tmp_path):
    path = tmp_path / "index.csv"
    path.write_text("Indexcd,Idxinfo11,Idxinfo02\n000300,2014-01-02,2013-12-31\n")
    with pytest.raises(ValueError, match="missing CSMAR columns"):
        interim.normalize_raw_frame(
            _collect_raw_export(path, "index_info"), "index_info"
        )


def test_complete_chain_index_info_fixture_has_native_price_and_return_identities(
    tmp_path, monkeypatch
):
    from quant_research_platform.markets.implementations.a_share.ids import (
        FORMAL_BENCHMARKS,
    )
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers import a_share_raw
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    writer = a_share_raw._write_csmar

    class CapturedIndexInfo(Exception):
        pass

    def write_index(root, data_id, rows, **kwargs):
        if data_id == "index_info":
            writer(root, data_id, rows, **kwargs)
            raise CapturedIndexInfo

    monkeypatch.setattr(a_share_raw, "_write_csmar", write_index)
    total_returns = {"csi500": "H00905", "csi1000": "H00852"}
    with pytest.raises(CapturedIndexInfo):
        a_share_raw.write_a_share_raw_fixture(
            layout, benchmark_total_return_indices=total_returns
        )
    result = interim.materialize_interim_data(
        output_roots={
            key: layout.interim_data_root("csmar", key) for key in ("index_info",)
        },
        raw_root=layout.raw_source_root("csmar"),
        data_ids=("index_info",),
    )
    saved = read_dataset(result.artifact_roots["index_info"]).set_index(
        "source_asset_id"
    )
    assert len(saved) == 4
    for code, (benchmark, name) in FORMAL_BENCHMARKS.items():
        assert saved.loc[code, "index_name"] == name
        assert saved.loc[total_returns[benchmark], "index_name"] == f"{name}全收益"


@pytest.mark.parametrize("chunksize", [1, 100])
@pytest.mark.parametrize(
    "data_id,date_column,value_column",
    [
        ("market_bars_1d", "Trddt", "Clsprc"),
        ("adjustment", "TradingDate", "FwardFactor"),
        ("market_cap", "Shrchgdt", "Nshrttl"),
    ],
)
def test_interim_owns_source_window_and_raw_preserves_all_business_values(
    tmp_path, chunksize, data_id, date_column, value_column
):
    from quant_research_platform.sources.csmar.field_mappings import COLUMN_MAPPINGS
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    row = dict.fromkeys(raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns, 1.0)
    row["Symbol" if data_id == "adjustment" else "Stkcd"] = "000001"
    row[date_column] = "2020-01-02"
    if data_id == "market_bars_1d":
        row.update(
            Markettype=1, Trdsta=0, Filling=0, Ahshrtrd_D=0, Ahvaltrd_D=0, LimitStatus=0
        )
    _write_csmar(raw, data_id, [row])
    arguments = dict(
        raw_root=raw,
        data_ids=(data_id,),
        start_date="2020-01-01",
        end_date="2020-12-31",
        chunksize=chunksize,
    )
    baseline = tmp_path / "baseline"
    interim.materialize_interim_data(**arguments, output_roots={data_id: baseline})
    future = {**row, date_column: "2026-01-02", value_column: "invalid-later-value"}
    if data_id == "market_bars_1d":
        future.update(Ahshrtrd_D="", Ahvaltrd_D="")
    _write_csmar(raw, data_id, [row, future])
    original = _file_bytes(raw)
    path = next(filesystem_glob(raw, "*.csv"))
    decoded = pd.concat(
        raw_exports.iter_private_export_chunks(
            path,
            dataset=data_id,
            chunksize=chunksize,
        ),
        ignore_index=True,
    )
    assert len(decoded) == 2
    assert decoded[date_column].tolist() == ["2020-01-02", "2026-01-02"]
    assert decoded[value_column].iloc[1] == "invalid-later-value"
    selected = tmp_path / "selected"
    interim.materialize_interim_data(**arguments, output_roots={data_id: selected})
    pd.testing.assert_frame_equal(
        read_dataset(selected), read_dataset(baseline), check_exact=True
    )
    counts = read_dataset_manifest_evidence(selected)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == 2
    assert counts["excluded_date_range_rows"] == 1
    assert counts["retained_after_date_filter_rows"] == 1
    expanded = tmp_path / "expanded"
    interim.materialize_interim_data(
        **{**arguments, "end_date": "2026-12-31"}, output_roots={data_id: expanded}
    )
    saved = read_dataset(expanded).sort_values("timestamp").reset_index(drop=True)
    target = COLUMN_MAPPINGS[data_id][value_column]
    assert len(saved) == 2
    assert saved.loc[0, target] == 1.0
    assert pd.isna(saved.loc[1, target])
    assert saved.loc[1, "timestamp"] == pd.Timestamp("2026-01-02")
    if data_id == "market_bars_1d":
        assert saved.loc[1, "open"] == 1.0
        assert pd.isna(saved.loc[1, "adj_open"])
    assert _file_bytes(raw) == original
    _write_csmar(raw, data_id, [row, {**future, date_column: "unknown-date"}])
    unknown_before = _file_bytes(raw)
    unknown = tmp_path / "unknown"
    interim.materialize_interim_data(**arguments, output_roots={data_id: unknown})
    pd.testing.assert_frame_equal(
        read_dataset(unknown), read_dataset(baseline), check_exact=True
    )
    counts = read_dataset_manifest_evidence(unknown)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == 2
    assert counts["excluded_missing_required_rows"] == 1
    assert counts["retained_after_date_filter_rows"] == 1
    assert _file_bytes(raw) == unknown_before


def test_multi_target_diagnostics_do_not_inherit_previous_table_counts(tmp_path):
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    _write_csmar(raw, "analyst_forecast", _analyst_rows())
    _write_csmar(
        raw,
        "trading_calendar",
        [dict(Markettype=4, Clddt="2020-01-02", Daywk=4, State="O")],
    )
    roots = {name: tmp_path / name for name in ("analyst_forecast", "trading_calendar")}
    metadata = {"producer_diagnostics": {"caller_fact": "preserved"}}
    interim.materialize_interim_data(
        raw_root=raw, output_roots=roots, data_ids=tuple(roots), metadata=metadata
    )
    analyst = read_dataset_manifest_evidence(roots["analyst_forecast"])[
        "producer_diagnostics"
    ]
    calendar = read_dataset_manifest_evidence(roots["trading_calendar"])[
        "producer_diagnostics"
    ]
    assert analyst["source_row_counts"]["raw_rows"] == len(_analyst_rows())
    assert calendar["exact_duplicate_count"] == 0
    assert calendar["caller_fact"] == "preserved"
    assert calendar["source_row_counts"] == {
        "raw_rows": 1,
        "retained_before_date_filter_rows": 1,
        "retained_after_date_filter_rows": 1,
        "excluded_date_range_rows": 0,
    }
    assert "missing_institution_id_rows" not in calendar["source_row_counts"]
    assert metadata == {"producer_diagnostics": {"caller_fact": "preserved"}}


@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize("all_undated", (False, True))
def test_index_info_unusable_dates_are_local_to_timed_rows(
    tmp_path, chunksize, all_undated
):
    import json

    from quant_research_platform.markets.implementations.a_share.benchmark import (
        iter_benchmark_weight_snapshots,
    )
    from quant_research_platform.sources.csmar.ids import SOURCE_ID
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.helpers.a_share_raw import _write_csmar
    from tests.helpers.paths import StorageLayout

    layout = StorageLayout(tmp_path)
    raw_root = layout.raw_source_root(SOURCE_ID)
    filesystem_mkdir(raw_root, parents=True)
    valid = dict(
        Indexcd="000905",
        Idxinfo01="中证小盘500指数",
        Idxinfo11="2014-01-02",
        Idxinfo02="2013-12-31",
        Idxinfo04="source replacement \ufffd retained",
    )
    undated = [
        dict(valid, Indexcd="000852", Idxinfo11="9999-12-31", Idxinfo02=""),
        dict(valid, Indexcd="000853", Idxinfo11="invalid-date", Idxinfo02="bad-date"),
        dict(valid, Indexcd="000854", Idxinfo11="", Idxinfo02=""),
    ]
    rows = (
        undated
        if all_undated
        else [
            valid,
            dict(valid, Indexcd="000906", Idxinfo11="9999-12-31"),
            dict(valid, Indexcd="000907", Idxinfo11="invalid-date"),
            dict(valid, Indexcd="000908", Idxinfo02="bad-date"),
            *undated,
        ]
    )
    _write_csmar(raw_root, "index_info", rows, extra_columns=("Idxinfo04",))
    raw_path = next(filesystem_glob(raw_root, "*.csv"))
    before = Path(native_filesystem_path(raw_path)).read_bytes()
    result = interim.materialize_interim_data(
        output_roots={"index_info": layout.interim_data_root(SOURCE_ID, "index_info")},
        raw_root=raw_root,
        data_ids=("index_info",),
        chunksize=chunksize,
    )
    assert not result.skipped
    root = result.artifact_roots["index_info"]
    saved = read_dataset(root)
    assert len(saved) == (0 if all_undated else 4)
    assert not saved[["source_asset_id", "timestamp"]].isna().any().any()
    if not all_undated:
        by_code = saved.set_index("source_asset_id")
        assert by_code.loc["000905", "timestamp"] == pd.Timestamp("2014-01-02")
        assert by_code.loc["000905", "base_date"] == pd.Timestamp("2013-12-31")
        for code in ("000906", "000907"):
            assert pd.isna(by_code.loc[code, "start_date"])
            assert by_code.loc[code, "timestamp"] == pd.Timestamp("2013-12-31")
        assert pd.isna(by_code.loc["000908", "base_date"])
        assert by_code.loc["000908", "timestamp"] == pd.Timestamp("2014-01-02")
        assert saved.sample_scope.eq(valid["Idxinfo04"]).all()
    manifest = json.loads(
        Path(native_filesystem_path(root / "_manifest/manifest.json")).read_text()
    )
    counts = manifest["producer_diagnostics"]["source_row_counts"]
    assert counts["raw_rows"] == len(rows)
    assert counts["invalid_start_date_rows"] == (2 if all_undated else 4)
    assert counts["invalid_base_date_rows"] == (1 if all_undated else 2)
    assert counts["excluded_missing_timestamp_rows"] == 3
    assert counts["retained_before_date_filter_rows"] == len(saved)
    assert counts["retained_after_date_filter_rows"] == len(saved)
    assert counts["excluded_date_range_rows"] == 0
    assert Path(native_filesystem_path(raw_path)).read_bytes() == before
    weights = pd.DataFrame(
        dict(
            source_asset_id=["000905"],
            constituent_asset_id=["000001"],
            effective_date=["2020-01-02"],
            weight=[1.0],
        )
    )
    if all_undated:
        with pytest.raises(
            ValueError, match="a_share_benchmark_index_identity_invalid"
        ):
            list(iter_benchmark_weight_snapshots(saved, [weights]))
    else:
        assert len(pd.concat(iter_benchmark_weight_snapshots(saved, [weights]))) == 1


@pytest.mark.parametrize(
    ("dates", "expected_dates"),
    (
        (
            ("2020-01-02", "2020/01/03", "20200104", "", None),
            ("2020-01-02", "2020-01-03", "2020-01-04", None, None),
        ),
        (
            ("2020-01-02T08:00:00+08:00", "2020-01-03T00:00:00Z", "", None),
            ("2020-01-02", "2020-01-03", None, None),
        ),
    ),
)
def test_index_date_vectorization_preserves_dates_and_missingness(
    dates, expected_dates
):
    values = pd.Series(dates, dtype="string")
    frame = pd.DataFrame(
        {
            "Indexcd": [f"{i:06d}" for i in range(len(values))],
            "Idxinfo01": "benchmark",
            "Idxinfo11": values,
            "Idxinfo02": "2013-12-31",
        }
    )
    normalized = interim.normalize_raw_frame(frame, "index_info")
    expected = pd.Series(expected_dates, dtype="datetime64[ns]")
    pd.testing.assert_series_equal(normalized.start_date, expected, check_names=False)
    assert normalized.attrs["source_row_counts"]["invalid_start_date_rows"] == 0


@pytest.mark.parametrize(
    "data_id", ("financial_balance", "financial_income", "financial_cashflow")
)
@pytest.mark.parametrize(
    "correction_dates",
    ("invalid-date", "2020-01-01,2020-02-01,2020-01-15", "2020-01-01,2020-01-01"),
)
def test_invalid_financial_correction_chronology_never_becomes_original_version(
    data_id, correction_dates
):
    columns = raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns
    original = dict(zip(columns, _financial_fixture_row(columns), strict=True))
    bad = {**original, "IfCorrect": "1", "DeclareDate": correction_dates}
    good = {**original, "Stkcd": "000002"}
    normalized = interim.normalize_raw_frame(pd.DataFrame([bad, good]), data_id)
    assert normalized.source_asset_id.tolist() == ["000002"]
    assert normalized.version_identity.tolist() == ["original"]
    assert normalized.is_corrected.tolist() == [False]
    assert normalized.correction_disclosure_date.isna().all()
    assert normalized.report_period.tolist() == [pd.Timestamp("2020-01-02")]
    counts = normalized.attrs["source_row_counts"]
    assert counts["raw_rows"] == 2
    assert counts["excluded_missing_required_rows"] == 1
    assert counts["retained_before_date_filter_rows"] == 1


@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize("bad_delisting_date", ("invalid-date", "9999-12-31"))
def test_stock_master_unresolved_delisting_keeps_later_status_and_neighbors(
    tmp_path, chunksize, bad_delisting_date
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    first = {
        "Stkcd": "000001",
        "Stknme": "first status",
        "Listdt": "2010-01-04",
        "DelistedDate": "",
        "Markettype": 4,
        "Nnindcd": "C",
        "Nnindnme": "industry",
        "Statco": "normal",
        "Statdt": "2020-01-02",
    }
    later = {
        **first,
        "Stknme": "later status",
        "Statdt": "2020-01-06",
        "DelistedDate": bad_delisting_date,
    }
    neighbor = {**first, "Stkcd": "000002", "Statdt": "2020-01-06"}
    known_end = {**neighbor, "Stkcd": "000003", "DelistedDate": "2020-02-03"}
    _write_csmar(raw, "stock_master", [later, neighbor, first, known_end])
    before = _file_bytes(raw)
    decoded = _collect_raw_export(next(filesystem_glob(raw, "*.csv")), "stock_master")
    assert decoded.DelistedDate.iloc[0] == bad_delisting_date
    assert len(decoded) == 4
    output = tmp_path / "interim"
    interim.materialize_interim_data(
        raw_root=raw,
        data_ids=("stock_master",),
        output_roots={"stock_master": output},
        chunksize=chunksize,
    )
    saved = read_dataset(output)
    history = saved.loc[saved.source_asset_id.eq("000001")].sort_values("timestamp")
    assert history.timestamp.tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-06"),
    ]
    assert history.stock_name.tolist() == ["first status", "later status"]
    assert history.delisting_date.isna().all()
    assert history.delisting_date_unresolved.tolist() == [False, True]
    assert history.listing_date.eq(pd.Timestamp("2010-01-04")).all()
    neighbors = saved.loc[saved.source_asset_id.ne("000001")].set_index(
        "source_asset_id"
    )
    assert pd.isna(neighbors.loc["000002", "delisting_date"])
    assert neighbors.loc["000003", "delisting_date"] == pd.Timestamp("2020-02-03")
    assert not neighbors.delisting_date_unresolved.any()
    assert pd.api.types.is_bool_dtype(saved.delisting_date_unresolved.dtype)
    assert not saved.delisting_date_unresolved.isna().any()
    assert not saved.duplicated(["source_asset_id", "timestamp"]).any()
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["invalid_delisting_date_rows"] == 1
    assert counts["raw_rows"] == counts["retained_after_date_filter_rows"] == 4
    assert counts["excluded_missing_timestamp_rows"] == 0
    assert _file_bytes(raw) == before


@pytest.mark.parametrize("chunksize", (1, 100))
@pytest.mark.parametrize("listing_date", ("invalid-date", "9999-12-31", "", None))
def test_stock_master_unknown_listing_date_keeps_status_time_without_fabrication(
    tmp_path, chunksize, listing_date
):
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    filesystem_mkdir(raw)
    first = {
        "Stkcd": "000001",
        "Stknme": "stock",
        "Listdt": "2010-01-04",
        "DelistedDate": "",
        "Markettype": 4,
        "Nnindcd": "C",
        "Nnindnme": "industry",
        "Statco": "normal",
        "Statdt": "2020-01-02",
    }
    later = {**first, "Listdt": listing_date, "Statdt": "2020-01-06"}
    neighbor = {**first, "Stkcd": "000002", "Statdt": "2020-01-06"}
    _write_csmar(raw, "stock_master", [first, later, neighbor])
    before = _file_bytes(raw)
    output = tmp_path / "interim"
    interim.materialize_interim_data(
        raw_root=raw,
        data_ids=("stock_master",),
        output_roots={"stock_master": output},
        chunksize=chunksize,
    )
    saved = read_dataset(output)
    history = saved.loc[saved.source_asset_id.eq("000001")].sort_values("timestamp")
    assert len(history) == 2
    assert history.listing_date.iloc[0] == pd.Timestamp("2010-01-04")
    assert pd.isna(history.listing_date.iloc[1])
    assert (
        history.timestamp.iloc[1]
        == history.status_change_date.iloc[1]
        == pd.Timestamp("2020-01-06")
    )
    assert saved.loc[saved.source_asset_id.eq("000002"), "listing_date"].tolist() == [
        pd.Timestamp("2010-01-04")
    ]
    assert not saved.delisting_date_unresolved.any()
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == counts["retained_after_date_filter_rows"] == 3
    assert counts["excluded_missing_timestamp_rows"] == 0
    assert counts.get("invalid_listing_date_rows", 0) == (1 if listing_date else 0)
    assert _file_bytes(raw) == before


def test_no_limit_description_missing_preserves_identified_events():
    from quant_research_platform.markets.implementations.a_share.processed import panels

    events = interim.normalize_raw_frame(
        pd.DataFrame(
            {
                "TradingDate": ["2020-01-02"] * 3,
                "Symbol": ["000001", "000002", "000003"],
                "Reason": ["IPO", None, " "],
            }
        ),
        "trading_no_limit",
    )
    assert len(events) == 3
    assert events.reason.isna().tolist() == [False, True, True]
    panel = pd.DataFrame(
        {
            "asset_id": ["000001", "000002", "000003", "000004"],
            "timestamp": pd.to_datetime(["2020-01-02"] * 4),
            "limit_status": [1.0] * 4,
        }
    )
    assert panels._apply_no_limit_exceptions(panel, events).limit_status.tolist() == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_index_weight_invalid_constituent_is_excluded_locally():
    result = interim.normalize_raw_frame(
        pd.DataFrame(
            {
                "Indexcd": ["000300", "000300"],
                "Enddt": ["2020-01-02"] * 2,
                "Stkcd": ["000001", "bad-code"],
                "Constdnme": ["valid", "unknown"],
                "Weight": [1.0, 2.0],
            }
        ),
        "index_weights",
    )
    assert result.constituent_asset_id.tolist() == ["000001"]
    assert result.weight.tolist() == [0.01]
    assert result.attrs["source_row_counts"]["excluded_missing_required_rows"] == 1


@pytest.mark.parametrize(
    "dates,expected_rows,excluded_invalid,excluded_window",
    (
        (("invalid-date", "9999-12-31"), 0, 2, 0),
        (("2019-12-31", "2020-01-06"), 0, 0, 2),
        (
            ("invalid-date", "2019-12-31", "2020-01-02", "2020-01-03"),
            2,
            1,
            1,
        ),
    ),
)
def test_daily_bar_requested_window_preserves_typed_schema(
    tmp_path, dates, expected_rows, excluded_invalid, excluded_window
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    data_id = "market_bars_1d"
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "result"
    row = dict.fromkeys(raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns, 1.0)
    row.update(
        Stkcd="000001",
        Markettype=1,
        Trdsta=0,
        Filling=0,
        Ahshrtrd_D=0.0,
        Ahvaltrd_D=0.0,
        LimitStatus=0,
        Dretwd=0.025,
    )
    _write_csmar(raw, data_id, [dict(row, Trddt=date) for date in dates])
    original = _file_bytes(raw)
    result = interim.materialize_interim_data(
        raw_root=raw,
        output_roots={data_id: output},
        data_ids=(data_id,),
        start_date="2020-01-02",
        end_date="2020-01-03",
        chunksize=1,
    )
    assert _file_bytes(raw) == original
    assert result.skipped == ()
    assert result.artifact_roots == {data_id: output}
    saved = read_dataset(output)
    assert len(saved) == expected_rows
    assert tuple(saved.columns) == interim.MARKET_BARS_INTERIM_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(saved.timestamp)
    assert saved.source_asset_id.tolist() == ["000001"] * expected_rows
    assert saved.period_return.tolist() == [0.025] * expected_rows
    assert saved.timestamp.between("2020-01-02", "2020-01-03").all()
    members = list(filesystem_glob(output, "**/*.parquet"))
    assert members
    for member in members:
        schema = pq.ParquetFile(native_filesystem_path(member)).schema_arrow
        assert schema.field("source_asset_id").type == pa.string()
        assert pa.types.is_timestamp(schema.field("timestamp").type)
        assert not any(pa.types.is_null(field.type) for field in schema)
    counts = read_dataset_manifest_evidence(output)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert counts["raw_rows"] == len(dates)
    assert counts.get("excluded_missing_required_rows", 0) == excluded_invalid
    assert counts["excluded_date_range_rows"] == excluded_window
    assert counts["retained_after_date_filter_rows"] == expected_rows


@pytest.mark.parametrize(
    "chunksize,reverse,split_files", [(1, False, False), (100, True, True)]
)
def test_daily_bar_conflicts_preserve_agreed_facts_and_returns(
    tmp_path, chunksize, reverse, split_files
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    data_id = "market_bars_1d"
    row = dict.fromkeys(raw_exports.RAW_TABLE_CONTRACTS[data_id].required_columns, 1.0)
    row.update(
        Stkcd="000001",
        Trddt="2020-01-02",
        Markettype=1,
        Trdsta=0,
        Filling=0,
        Ahshrtrd_D=0.0,
        Ahvaltrd_D=0.0,
        LimitStatus=0,
        Opnprc=10.0,
        Clsprc=10.0,
        Adjprcwd=10.0,
        Dretwd=0.025,
    )
    rows = [
        row,
        {**row, "Stkcd": "000002"},
        {**row, "Dnshrtrd": 2.0},
        {**row, "Stkcd": "000003"},
        {**row, "Stkcd": "000004"},
        {**row, "Stkcd": "000003", "Opnprc": 20.0},
        {**row, "Stkcd": "000004", "Opnprc": 20.0, "Clsprc": 20.0},
        dict(row),
    ]
    if reverse:
        rows.reverse()
    raw = tmp_path / "raw"
    raw.mkdir()
    if split_files:
        # Repeated keys cross files, not merely chunks within one file.
        for name, part in (("a", rows[:4]), ("b", rows[4:])):
            directory = raw / name
            directory.mkdir()
            _write_csmar(directory, data_id, part)
    else:
        _write_csmar(raw, data_id, rows)
    before = _file_bytes(raw)
    result = interim.materialize_interim_data(
        raw_root=raw,
        output_roots={data_id: tmp_path / "result"},
        data_ids=(data_id,),
        chunksize=chunksize,
    )
    root = result.artifact_roots[data_id]
    saved = read_dataset(root).set_index("source_asset_id").sort_index()
    assert saved.index.tolist() == ["000001", "000002", "000003", "000004"]
    assert saved.period_return.tolist() == [0.025] * 4
    assert saved.amount.tolist() == [1.0] * 4
    assert pd.isna(saved.loc["000001", "volume"])
    assert saved.loc["000001", ["open", "close", "adj_open"]].tolist() == [10.0] * 3
    assert saved.loc["000002", ["open", "close", "volume"]].tolist() == [
        10.0,
        10.0,
        1.0,
    ]
    assert pd.isna(saved.loc["000003", "open"])
    assert pd.isna(saved.loc["000003", "adj_open"])
    assert saved.loc["000003", "close"] == 10.0
    # Both complete alternatives establish the same adjusted opening price.
    assert saved.loc["000004", ["open", "close"]].isna().all()
    assert saved.loc["000004", "adj_open"] == 10.0
    diagnostics = read_dataset_manifest_evidence(root)["producer_diagnostics"]
    assert diagnostics["exact_duplicate_count"] == 1
    counts = diagnostics["source_row_counts"]
    assert counts["conflicting_volume_rows"] == 2
    assert counts["conflicting_open_rows"] == 4
    assert counts["conflicting_close_rows"] == 2
    assert counts["conflicting_adj_open_rows"] == 2
    assert counts["raw_rows"] == len(rows)
    assert _file_bytes(raw) == before


@pytest.mark.parametrize("chunksize,reverse", [(1, False), (8, True)])
def test_index_market_conflicts_are_field_local_across_chunks(
    tmp_path, chunksize, reverse
):
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    rows = _nullable_index_numeric_rows("index_market")[1:3]
    rows.append({**rows[0], "Idxtrd05": 110.0, "Idxtrd08": 3.0})
    if reverse:
        rows.reverse()
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_csmar(raw, "index_market", rows)
    result = interim.materialize_interim_data(
        raw_root=raw,
        output_roots={"index_market": tmp_path / "result"},
        data_ids=("index_market",),
        chunksize=chunksize,
    )
    root = result.artifact_roots["index_market"]
    saved = read_dataset(root).sort_values("timestamp").reset_index(drop=True)
    assert len(saved) == 2
    assert saved.open.tolist() == [100.0, 100.0]
    assert saved.index_name.tolist() == ["synthetic-index", "synthetic-index"]
    assert pd.isna(saved.close.iloc[0]) and pd.isna(saved.period_return.iloc[0])
    assert saved.close.iloc[1] == 100.0 and saved.period_return.iloc[1] == 0.0
    counts = read_dataset_manifest_evidence(root)["producer_diagnostics"][
        "source_row_counts"
    ]
    assert (
        counts["conflicting_close_rows"]
        == counts["conflicting_period_return_rows"]
        == 2
    )


def test_optional_reference_conflicts_preserve_consumed_facts(tmp_path, subtests):
    from quant_research_platform.markets.implementations.a_share.benchmark import (
        iter_benchmark_weight_snapshots,
    )
    from quant_research_platform.markets.implementations.a_share.processed import panels
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    info = {
        "Indexcd": "000905",
        "Idxinfo01": "中证小盘500指数",
        "Idxinfo11": "2014-01-02",
        "Idxinfo02": "2013-12-31",
        "Idxinfo05": "1",
    }
    no_limit = {"Symbol": "000001", "TradingDate": "2020-01-02", "Reason": "IPO"}
    weight = {
        "Indexcd": "000905",
        "Enddt": "2020-01-02",
        "Stkcd": "000001",
        "Constdnme": "first name",
        "Weight": 60.0,
    }
    cases = {
        "index_info": (
            [
                info,
                {**info, "Idxinfo05": "2"},
                {**info, "Indexcd": "000852", "Idxinfo01": "中证1000指数"},
            ],
            "weighting_method",
            ("Idxinfo05",),
        ),
        "trading_no_limit": (
            [
                no_limit,
                {**no_limit, "Reason": "relisting"},
                {**no_limit, "Symbol": "000002"},
            ],
            "reason",
            (),
        ),
        "index_weights": (
            [
                weight,
                {**weight, "Constdnme": "changed name"},
                {**weight, "Stkcd": "000002", "Weight": 40.0},
            ],
            "constituent_name",
            (),
        ),
    }
    benchmark_info = pd.DataFrame(
        {"source_asset_id": ["000905"], "index_name": ["中证小盘500指数"]}
    )
    benchmark_weights = pd.DataFrame(
        {
            "source_asset_id": ["000905"],
            "constituent_asset_id": ["000001"],
            "effective_date": [pd.Timestamp("2020-01-02")],
            "weight": [1.0],
        }
    )
    for data_id, (rows, column, extra) in cases.items():
        raw = tmp_path / data_id / "raw"
        raw.mkdir(parents=True)
        _write_csmar(raw, data_id, rows, extra_columns=extra)
        for chunksize in (1, 100):
            with subtests.test(data_id=data_id, chunksize=chunksize):
                root = tmp_path / data_id / str(chunksize)
                interim.materialize_interim_data(
                    raw_root=raw,
                    output_roots={data_id: root},
                    data_ids=(data_id,),
                    chunksize=chunksize,
                )
                saved = read_dataset(root)
                assert len(saved) == 2
                assert saved[column].isna().sum() == 1
                counts = read_dataset_manifest_evidence(root)["producer_diagnostics"][
                    "source_row_counts"
                ]
                assert counts[f"conflicting_{column}_rows"] == 2
                if data_id == "trading_no_limit":
                    panel = pd.DataFrame(
                        {
                            "asset_id": ["000001", "000002", "000003"],
                            "timestamp": pd.to_datetime(["2020-01-02"] * 3),
                            "limit_status": [1.0] * 3,
                        }
                    )
                    assert panels._apply_no_limit_exceptions(
                        panel, saved
                    ).limit_status.tolist() == [0.0, 0.0, 1.0]
                else:
                    frames = iter_benchmark_weight_snapshots(
                        saved if data_id == "index_info" else benchmark_info,
                        [saved if data_id == "index_weights" else benchmark_weights],
                    )
                    actual = pd.concat(frames, ignore_index=True).sort_values(
                        "asset_id"
                    )
                    assert actual.weight.tolist() == (
                        [0.6, 0.4] if data_id == "index_weights" else [1.0]
                    )


def test_unconsumed_index_name_conflict_does_not_block_benchmark_publication(tmp_path):
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.research.portfolios.production import (
        _iter_benchmark_weights,
    )
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
    from tests.helpers.a_share_raw import _write_csmar

    raw = tmp_path / "raw"
    raw.mkdir()
    info = dict(Idxinfo11="2014-01-02", Idxinfo02="2013-12-31")
    _write_csmar(
        raw,
        "index_info",
        [
            dict(info, Indexcd="000905", Idxinfo01="中证小盘500指数"),
            dict(info, Indexcd="000852", Idxinfo01="中证1000指数"),
            dict(info, Indexcd="000852", Idxinfo01="conflicting name"),
        ],
    )
    _write_csmar(
        raw,
        "index_weights",
        [
            dict(
                Indexcd="000905",
                Enddt="2020-01-02",
                Stkcd=asset,
                Constdnme="synthetic",
                Weight=weight,
            )
            for asset, weight in (("000001", 25.0), ("000002", 75.0))
        ],
    )
    before = _file_bytes(raw)
    roots = {name: tmp_path / name for name in ("index_info", "index_weights")}
    interim.materialize_interim_data(
        raw_root=raw, output_roots=roots, data_ids=tuple(roots), chunksize=1
    )
    saved_info = read_dataset(roots["index_info"]).set_index("source_asset_id")
    assert set(saved_info.index) == {"000852", "000905"}
    assert pd.isna(saved_info.loc["000852", "index_name"])
    assert saved_info.loc["000905", "index_name"] == "中证小盘500指数"
    counts = read_dataset_manifest_evidence(roots["index_info"])[
        "producer_diagnostics"
    ]["source_row_counts"]
    assert counts["conflicting_index_name_rows"] == 2
    output = prepare(
        dict(
            market_id="a_share",
            data_id="benchmark_weights",
            inputs={key: str(root) for key, root in roots.items()},
            time_window={"start": "2020-01-02", "end": "2020-01-02"},
            output_root=str(tmp_path / "benchmark"),
        )
    )
    saved = read_dataset(output)
    assert saved.benchmark_id.tolist() == ["csi500", "csi500"]
    assert saved.asset_id.tolist() == ["000001", "000002"]
    assert saved.timestamp.eq(pd.Timestamp("2020-01-02")).all()
    assert saved.weight.tolist() == [0.25, 0.75]
    (weights,) = _iter_benchmark_weights(
        output, "csi500", pd.DatetimeIndex(["2020-01-02"])
    )
    assert weights.index.tolist() == ["000001", "000002"]
    assert weights.tolist() == [0.25, 0.75]
    assert _file_bytes(raw) == before
