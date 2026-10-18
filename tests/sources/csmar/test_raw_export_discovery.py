from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.sources.csmar import field_mappings, raw_exports


def test_requested_dataset_rejects_ambiguous_header(tmp_path: Path) -> None:
    columns = tuple(
        dict.fromkeys(
            (
                *field_mappings.RAW_TABLE_CONTRACTS["stock_master"].required_columns,
                *field_mappings.RAW_TABLE_CONTRACTS["market_cap"].required_columns,
            )
        )
    )
    (tmp_path / "unknown.csv").write_text(
        ",".join(columns) + "\n" + ",".join("1" for _ in columns) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cannot infer CSMAR dataset from raw header"):
        tuple(
            raw_exports.dataset_files_by_id(
                tmp_path, requested_datasets=("market_cap",)
            )
        )


@pytest.mark.parametrize("error_type", [GeneratorExit, KeyboardInterrupt, MemoryError])
def test_csv_constructor_preserves_primary_when_close_fails(
    tmp_path, monkeypatch, error_type
):
    from quant_research_platform.sources.csmar import raw_schemas

    primary = error_type("constructor-primary")
    path = tmp_path / "synthetic.csv"
    path.write_bytes(b"a,b\n1,2\n")
    handle = path.open("rb")

    class FailingHandle:
        def fileno(self):
            return handle.fileno()

        def readable(self):
            raise primary

        def close(self):
            handle.close()
            raise OSError("close-secondary")

    monkeypatch.setattr(
        raw_schemas, "open_filesystem_file", lambda *_a: FailingHandle()
    )
    with pytest.raises(error_type) as raised:
        raw_schemas._CsvRecords(path)
    assert raised.value is primary
    assert handle.closed
    assert primary.__notes__ == ["CSMAR source close also failed: OSError"]


def test_canonical_three_dataset_root_ignores_sibling_archive(tmp_path):
    import csv

    import pandas as pd

    from quant_research_platform.sources.csmar import interim

    raw = tmp_path / "data" / "raw" / "csmar"
    archive = tmp_path / "data" / "archive" / "csmar"
    archive.mkdir(parents=True)
    (archive / "STK_ITEMCHANGE.csv").write_text(
        "unrepaired,invalid\n", encoding="utf-8"
    )
    rows = {
        "listing_status_changes": {
            "Symbol": "000001",
            "DeclareDate": "2020-01-02",
            "ChangeDate": "2020-01-03",
            "ChangedItem": "上市状态",
            "ValueBefore": "正常",
            "ValueAfter": "变更",
        },
        "special_treatment_changes": {
            "Symbol": "000001",
            "Chgtype": "ST",
            "Chgreas": "合成原因",
            "ShortNamebc": "合成前",
            "ShortNameac": "合成后",
            "Annoudt": "2020-01-02",
            "Execudt": "2020-01-03",
            "Chgrsdis": "说明,第一行\n第二行",
            "Content": '合成"内容"',
            "TypeAuditOpinbc": "unconsumed",
            "ISSustManaNonStandExpl": "unconsumed",
        },
        "index_info": {
            "Indexcd": "000300",
            "Idxinfo01": "synthetic index",
            "Idxinfo11": "2020-01-02",
            "Idxinfo02": "2020-01-01",
        },
    }
    for dataset, row in rows.items():
        directory = raw / dataset
        directory.mkdir(parents=True)
        table = field_mappings.RAW_TABLE_CONTRACTS[dataset].table_code
        with (directory / f"{table}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    members = tuple(raw_exports.dataset_files_by_id(raw))
    assert {dataset for dataset, _ in members} == set(rows)
    assert len(members) == 3
    for dataset, path in members:
        assert path.is_relative_to(raw)
        chunks = list(
            raw_exports.iter_private_export_chunks(
                path,
                dataset=dataset,
                chunksize=1,
            )
        )
        actual = interim.normalize_raw_frame(pd.concat(chunks), dataset)
        projected = pd.DataFrame(
            [
                {
                    name: value
                    for name, value in rows[dataset].items()
                    if name in field_mappings.COLUMN_MAPPINGS[dataset]
                }
            ]
        )
        expected = interim.normalize_raw_frame(projected.astype("string"), dataset)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        if dataset == "special_treatment_changes":
            assert len(rows[dataset]) == 11
            assert set(chunks[0]) == {"Symbol", "Chgtype", "Annoudt", "Execudt"}
            assert actual["change_type"].tolist() == ["ST"]
            assert actual["effective_date"].tolist() == [pd.Timestamp("2020-01-03")]


@pytest.mark.parametrize("exhaust", [False, True])
def test_csv_touch_does_not_change_consumed_provenance(tmp_path, exhaust):
    import hashlib
    import os

    from quant_research_platform.sources.csmar.raw_schemas import (
        iter_export_csv_records,
    )

    path = tmp_path / "source.csv"
    raw = b"value\n1\n"
    path.write_bytes(raw)
    hashes = []
    rows = iter_export_csv_records(path, content_hashes=hashes)
    assert next(rows) == ["value"]
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    if exhaust:
        assert list(rows) == [["1"]]
    rows.close()
    assert hashes == ([hashlib.sha256(raw).hexdigest()] if exhaust else [])


@pytest.mark.parametrize(
    "operation", ("early_close", "header", "duplicate_header", "parse")
)
def test_csv_closes_once_and_preserves_parse_or_header_failure(
    tmp_path, monkeypatch, operation
):
    from quant_research_platform.sources.csmar import raw_schemas

    path = tmp_path / "source.csv"
    payload = b'a,b\n"unclosed\n' if operation == "parse" else b"a,a\n1,2\n"
    if operation == "header":
        payload = b"a,b\n1,2\n"
    path.write_bytes(payload)
    closed = []
    close = raw_schemas._CsvRecords.close

    def close_then_fail(records):
        closed.append(records)
        close(records)
        raise OSError("close_failed")

    monkeypatch.setattr(raw_schemas._CsvRecords, "close", close_then_fail)
    primary_error = operation in {"duplicate_header", "parse"}
    error = (
        "malformed_csmar_csv_row"
        if operation == "parse"
        else "duplicate_csmar_provider_column"
    )
    with pytest.raises(
        ValueError if primary_error else OSError,
        match=error if primary_error else "close_failed",
    ) as caught:
        if operation in {"header", "duplicate_header"}:
            raw_schemas.read_export_header(path)
        else:
            rows = raw_schemas.iter_export_csv_records(path)
            if operation == "early_close":
                assert next(rows) == ["a", "a"]
                rows.close()
            else:
                list(rows)
    assert len(closed) == 1
    if primary_error:
        assert caught.value.__notes__ == ["iterator close also failed: OSError"]
    if operation in {"early_close", "parse"}:
        rows.close()
        assert len(closed) == 1


@pytest.mark.parametrize(
    "data_id,columns,error",
    (
        (
            "listing_status_changes",
            (
                "Symbol",
                "DeclareDate",
                "ChangeDate",
                "ChangedItem",
                "ValueBefore",
                "ValueAfter",
            ),
            "invalid_listing_status_date",
        ),
        (
            "stock_info_changes",
            (
                "Symbol",
                "AnnouncementDate",
                "ImplementDate",
                "ChangedItem",
                "Value_Before",
                "Value_After",
            ),
            "invalid_stock_info_change_date",
        ),
    ),
)
def test_state_change_dates_survive_csv_to_interim(tmp_path, data_id, columns, error):
    import pandas as pd

    from quant_research_platform.sources.csmar import interim
    from quant_research_platform.storage.datasets import read_dataset

    source = pd.DataFrame(
        [
            ("000001", "", "2020-01-03", "上市状态", "正常", "变更"),
            ("000001", "2020-01-01", "2020-01-02", "上市状态", "变更", "正常"),
        ],
        columns=columns,
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    table = field_mappings.RAW_TABLE_CONTRACTS[data_id].table_code
    source.to_csv(raw / f"{table}.csv", index=False)
    output = tmp_path / "interim"
    interim.materialize_interim_data(
        raw_root=raw,
        output_roots={data_id: output},
        data_ids=(data_id,),
        chunksize=1,
    )
    actual = read_dataset(output).sort_values("timestamp").reset_index(drop=True)
    assert actual.source_asset_id.tolist() == ["000001", "000001"]
    assert actual.timestamp.tolist() == list(
        pd.to_datetime(["2020-01-02", "2020-01-03"])
    )
    assert actual.effective_date.equals(actual.timestamp)
    assert actual.announcement_date.iloc[0] == pd.Timestamp("2020-01-01")
    assert pd.isna(actual.announcement_date.iloc[1])
    assert actual.value_after.tolist() == ["正常", "变更"]
    mapped = source.rename(columns=field_mappings.COLUMN_MAPPINGS[data_id])
    mapped.loc[0, "effective_date"] = "bad-date"
    with pytest.raises(ValueError, match=error):
        interim._normalize_state_change_frame(mapped, data_id)
