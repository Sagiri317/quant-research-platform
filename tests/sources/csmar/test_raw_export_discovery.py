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
