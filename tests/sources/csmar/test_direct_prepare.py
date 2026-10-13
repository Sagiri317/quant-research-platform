"""Direct source operations distinguish absent, empty, and damaged exports."""

import csv

import pytest

from quant_research_platform.sources.csmar.interim import materialize_interim_data
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets import read_dataset


def test_empty_declared_source_preserves_native_schema(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with (raw / "calendar.csv").open("w", newline="") as handle:
        csv.writer(handle).writerow(("Markettype", "Clddt", "Daywk", "State"))
    output = tmp_path / "empty-calendar"
    result = prepare_source(
        {
            "source_id": "csmar",
            "raw_root": str(raw),
            "outputs": {"trading_calendar": str(output)},
            "start_date": None,
            "end_date": None,
        }
    )
    assert result.skipped == ()
    frame = read_dataset(output)
    assert frame.empty
    assert str(frame.timestamp.dtype) == "datetime64[ns]"
    assert str(frame.is_trading_day.dtype) == "bool"


def test_absent_requested_table_is_not_an_empty_present_table(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    output = tmp_path / "calendar"
    result = materialize_interim_data(
        raw_root=raw,
        output_roots={"trading_calendar": output},
        data_ids=("trading_calendar",),
    )
    assert result.skipped == ("trading_calendar",)
    assert not output.exists()
    (raw / "calendar.csv").write_text("Markettype,Clddt,Daywk,State\n4,bad-date,1,O\n")
    with pytest.raises(ValueError):
        materialize_interim_data(
            raw_root=raw,
            output_roots={"trading_calendar": output},
            data_ids=("trading_calendar",),
        )
    assert not output.exists()
