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
            raw_exports.dataset_files_with_identity_by_id(
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
    members = tuple(raw_exports.dataset_files_with_identity_by_id(raw))
    assert {dataset for dataset, _, _ in members} == set(rows)
    assert len(members) == 3
    for dataset, path, identity in members:
        assert path.is_relative_to(raw)
        chunks = list(
            raw_exports.iter_private_export_chunks(
                path,
                dataset=dataset,
                chunksize=1,
                expected_identity=identity,
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
