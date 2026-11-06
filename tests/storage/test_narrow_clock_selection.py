"""Physical filter clocks do not leak into output or consume unrelated maturity."""

from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets import reading, write_dataset
from tests.storage.test_selectable_dataset import frame, spec


@pytest.mark.parametrize("native", [False, True])
def test_narrow_projection_keeps_parent_file_year_rows_and_clock(tmp_path, native):
    values = frame().assign(label=1.0, mature_at=pd.Timestamp("2025-01-01"))
    declared = replace(
        spec(),
        partition_columns=("year",) if native else (),
        column_roles={**spec().column_roles, "label": "label", "mature_at": "context"},
        label_availability_columns={"label": "mature_at"},
    )
    root = tmp_path / "source"
    root.mkdir()
    if native:
        root = root / "native"
        write_dataset(root, values, spec=declared)
        path = next((root / "data" / "year=2021").glob("*.parquet"))
        options = {}
    else:
        path = root / "selected.parquet"
        values.to_parquet(path, index=False)
        (root / "excluded.parquet").write_bytes(b"not selected")
        options = {"spec": declared}
    # An incompatible unconsumed clock must not gate this factor-only read.
    table = pq.ParquetFile(path).read()
    index = table.schema.get_field_index("mature_at")
    pq.write_table(
        table.set_column(index, "mature_at", pa.array(["unconsumed"] * len(table))),
        path,
    )
    parent = reading.open_dataset_session(
        root,
        files=[str(path.relative_to(root))],
        years=[2021],
        time_window=TimeWindow("2021-01-01", "2021-01-01"),
        row_filters=(("asset_id", "in", ("a",)),),
        columns=("value",),
        **options,
    )
    narrowed = parent.filtered_view(
        TimeWindow("2021-01-01T00:00:00.000000001", "2021-01-02T00:00:00")
    ).projected_view(("value",))
    assert set(narrowed.physical_columns) == {"value", "timestamp", "asset_id"}
    assert narrowed.read_paths == (path,)
    result = reading.materialize_dataset_session(narrowed, batch_rows=1)
    assert list(result) == ["value"]
    assert result.value.tolist() == [5.0]
    assert reading.materialize_dataset_session(parent).value.tolist() == [3.0, 5.0]
    if native:
        with pytest.raises(ValueError, match="dataset_schema_mismatch"):
            reading.materialize_dataset_session(parent.projected_view(("mature_at",)))
