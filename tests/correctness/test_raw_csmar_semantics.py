from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import interim, raw_exports, raw_schemas
from tests.helpers.runtime_resources import capture_test_policy


def _collect_raw_export(
    path: Path,
    dataset: str,
    *,
    chunksize: int = 1024,
) -> pd.DataFrame:
    chunks = _collect_raw_chunks(path, dataset, chunksize=chunksize)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def test_index_weight_handoff_releases_raw_frame_before_persisting_normalized_rows(
    tmp_path, monkeypatch
):
    import weakref

    from quant_research_platform.storage.datasets import writing
    from quant_research_platform.storage.datasets.reading import read_dataset
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.a_share_raw import _write_csmar

    layout = StorageLayout(tmp_path)
    root = layout.raw_source_root("csmar")
    root.mkdir(parents=True)
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
            for benchmark in ("000852", "000905")
            for timestamp in pd.bdate_range("2019-03-01", "2019-03-29")
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
    with runtime_resource_policy_scope() as policy:
        result = interim.materialize_interim_data(
            layout=layout,
            data_ids=("index_weights",),
            resource_policy=policy,
        )
        saved = read_dataset(
            result.artifact_roots["index_weights"], resource_policy=policy
        )
        assert len(saved) == 4200
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("header_interrupted")
    close_calls = 0

    class FailingHandle:
        def fileno(self):
            return 7

        def read(self, _limit: int = -1):
            raise primary

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()
            return False

        def close(self):
            nonlocal close_calls
            close_calls += 1
            raise OSError("header_close_failed")

    class ExportPath:
        suffix = ".csv"

        def open(self, *_args, **_kwargs):
            return FailingHandle()

    monkeypatch.setattr(raw_schemas, "Path", lambda _path: ExportPath())
    identity = raw_schemas.DatasetFileIdentity(1, 2, 3, 4)
    monkeypatch.setattr(raw_schemas, "dataset_file_identity", lambda _path: identity)
    monkeypatch.setattr(
        raw_schemas.os,
        "fstat",
        lambda _fileno: SimpleNamespace(
            st_dev=1,
            st_ino=2,
            st_size=3,
            st_mtime_ns=4,
        ),
    )

    with pytest.raises(KeyboardInterrupt) as raised:
        raw_schemas.read_export_header("export.csv")

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
