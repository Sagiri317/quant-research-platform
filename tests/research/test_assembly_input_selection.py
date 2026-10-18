"""Persisted source selectors and single application of the assembly window."""

import pandas as pd
import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.datasets import assembler
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    reading,
)
from tests.helpers.research_processed_inputs import assembly_config as _config
from tests.helpers.research_processed_inputs import assembly_frame as _frame
from tests.helpers.research_processed_inputs import assembly_source as _source


@pytest.mark.parametrize("storage", ["native", "parquet"])
def test_assembly_persists_scientific_input_selection(tmp_path, storage):
    frame = pd.concat(
        [
            _frame(4).assign(timestamp=date, eligible=True)
            for date in pd.date_range("2023-12-31", "2024-01-04")
        ],
        ignore_index=True,
    )
    root, _ = _source(tmp_path, frame)
    source = {"path": str(root)}
    if storage == "parquet":
        declared = inspect_dataset(root).spec
        source["spec"] = {
            key: declared[key]
            for key in (
                "data_id",
                "frequency",
                "grain",
                "primary_key",
                "partition_columns",
                "column_roles",
            )
        }
        root = tmp_path / "bare"
        root.mkdir()
        for year, rows in frame.groupby(frame.timestamp.dt.year):
            rows.to_parquet(root / f"{year}.parquet", index=False)
        source["path"] = str(root)
    source.update(
        columns=["momentum_20d"],
        files=sorted(str(path.relative_to(root)) for path in root.rglob("*.parquet")),
        years=[2024],
        time_window={"start": "2024-01-01", "end": "2024-01-03"},
        row_filters=[
            ["asset_id", "in", ["asset_000", "asset_001"]],
            ["eligible", "==", True],
        ],
    )
    config = _config(
        tmp_path,
        root,
        inputs={"selected": source},
        time_window={"start": "2024-01-02", "end": "2024-01-04"},
    )
    output = assembler.assemble(config)
    facts = inspect_dataset(output)
    assert facts.manifest["inputs"] == config["inputs"]
    assert facts.manifest["parameters"]["time_window"] == config["time_window"]
    columns = ["market_id", "asset_id", "timestamp", "momentum_20d"]
    assert facts.manifest["parameters"]["input_columns"] == {"selected": columns}
    expected = (
        frame.loc[
            frame.asset_id.isin(["asset_000", "asset_001"])
            & frame.timestamp.between("2024-01-02", "2024-01-03"),
            columns,
        ]
        .astype({"timestamp": "datetime64[ns]"})
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(read_dataset(output), expected)
    # Persisted selectors remain descriptive; replay is an explicit new operation.
    replay = assembler.assemble(
        {
            **config,
            "inputs": facts.manifest["inputs"],
            "output_root": str(tmp_path / "replayed"),
        }
    )
    pd.testing.assert_frame_equal(read_dataset(replay), expected)


@pytest.mark.parametrize(
    "input_window", [None, {"start": "2024-01-03", "end": "2024-01-05"}]
)
@pytest.mark.parametrize("intraday", [False, True])
def test_assembly_applies_overall_window_once_to_each_reader(
    tmp_path, monkeypatch, input_window, intraday
):
    stamps = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-02T00:00:00.000000001",
            "2024-01-02T12:00:00",
            "2024-01-03",
            "2024-01-04",
            "2024-01-04T00:00:00.000000001",
            "2024-01-04T23:59:59.999999999",
            "2024-01-05",
        ],
        format="mixed",
    )
    frame = pd.concat(
        [
            _frame(1).assign(timestamp=stamp, momentum_20d=float(index))
            for index, stamp in enumerate(stamps)
        ],
        ignore_index=True,
    )
    base, _ = _source(tmp_path / "base", frame)
    auxiliary, _ = _source(
        tmp_path / "auxiliary", frame.rename(columns={"momentum_20d": "context"})
    )
    overall = {"start": "2024-01-02", "end": "2024-01-04"}
    if intraday:
        overall = {
            "start": "2024-01-02T01:00:00.000000001+01:00",
            "end": "2024-01-04T01:00:00+01:00",
        }
    inputs = {
        name: {"path": str(root), "time_window": input_window}
        for name, root in (("base", base), ("auxiliary", auxiliary))
    }
    lower = (
        pd.Timestamp(TimeWindow(**input_window).start)
        if input_window is not None
        else pd.Timestamp("2024-01-02") + pd.Timedelta(int(intraday), "ns")
    )
    upper = pd.Timestamp("2024-01-04" if intraday else "2024-01-04T23:59:59.999999999")
    selected = frame.loc[frame.timestamp.between(lower, upper)]
    scans = []
    consumed = {base: [], auxiliary: []}
    original = reading.iter_dataset_file_batches

    def measured(session, **kwargs):
        scans.append(session.root)
        for path, batch in original(session, **kwargs):
            consumed[session.root].extend(batch.timestamp.tolist())
            yield path, batch

    monkeypatch.setattr(reading, "iter_dataset_file_batches", measured)
    output = assembler.assemble(
        _config(
            tmp_path,
            base,
            inputs=inputs,
            context_columns=["context"],
            time_window=overall,
        )
    )
    # Both inputs scan once and consume only the exact intersection, regardless
    # of whether the reader represents a day end with < or <= predicates.
    assert len(scans) == 2 and set(scans) == {base, auxiliary}
    assert consumed == {
        base: selected.timestamp.tolist(),
        auxiliary: selected.timestamp.tolist(),
    }
    monkeypatch.setattr(reading, "iter_dataset_file_batches", original)
    actual = read_dataset(output)
    expected = (
        selected.assign(context=selected.momentum_20d)
        .astype({"timestamp": "datetime64[ns]"})
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
