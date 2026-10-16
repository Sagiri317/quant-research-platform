"""Equity statement rows retain source and subject identity, not guessed uniqueness."""

import csv

from quant_research_platform.sources.csmar.field_mappings import RAW_TABLE_CONTRACTS
from quant_research_platform.sources.csmar.interim import interim_catalog
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets import read_dataset


def test_source_and_subject_distinguish_statement_rows_after_native_roundtrip(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "FN_FN046.csv"
    header = RAW_TABLE_CONTRACTS["equity_changes"].required_columns
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(
            [
                ("000001", "2020-12-31", "A", "1", "001"),
                ("000001", "2020-12-31", "A", "2", "001"),
                ("000001", "2020-12-31", "A", "1", "002"),
            ]
        )
    output = tmp_path / "interim"
    prepare_source(
        {
            "source_id": "csmar",
            "raw_root": str(raw),
            "outputs": {"equity_changes": str(output)},
            "chunksize": 1,
        }
    )
    frame = read_dataset(output)
    key = interim_catalog().get("equity_changes").primary_key
    assert key == (
        "source_asset_id",
        "timestamp",
        "report_type",
        "data_source",
        "subject_code",
    )
    assert len(frame) == 3
    assert not frame.duplicated(list(key)).any()
    assert set(zip(frame.data_source, frame.subject_code, strict=True)) == {
        ("1", "001"),
        ("2", "001"),
        ("1", "002"),
    }
