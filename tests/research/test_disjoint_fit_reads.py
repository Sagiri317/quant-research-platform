"""Fit consumes only the selected windows and owns only model-specific keys."""

import pandas as pd
import pytest

from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from tests.correctness.test_direct_model_operations import data, fit_config


def request(tmp_path):
    source = data(tmp_path)
    config = fit_config(tmp_path, source)
    config.update(
        fit_windows=[
            {"start": "2020-01-01", "end": "2020-01-01"},
            {"start": "2020-01-05", "end": "2020-01-05"},
        ],
        fit_cutoff="2020-01-05",
    )
    return source, config


def test_fit_indexes_only_selected_rows_once(tmp_path, monkeypatch):
    _, config = request(tmp_path)
    indexed = []
    original = ExactPrimaryKeyIndex.add_table

    def observe(self, table):
        indexed.extend(table.column("timestamp").to_pylist())
        return original(self, table)

    monkeypatch.setattr(ExactPrimaryKeyIndex, "add_table", observe)
    result, _ = load_model(fit(config))
    assert result.metadata["rows_after_input_label_dropna"] == 6
    assert indexed == list(pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-05"] * 3))


@pytest.mark.parametrize("day", [1, 3, 5])
def test_fit_finite_validation_is_scoped_to_selected_windows(tmp_path, day):
    source, config = request(tmp_path)
    frame = pd.read_parquet(source)
    frame.loc[frame.timestamp.dt.day.eq(day), "x"] = float("inf")
    frame.to_parquet(source, index=False)
    if day == 3:
        result, _ = load_model(fit(config))
        assert result.metadata["rows_after_input_label_dropna"] == 6
    else:
        with pytest.raises(ValueError, match="dataset_infinite_value"):
            fit(config)


@pytest.mark.parametrize("day", [1, 5])
@pytest.mark.parametrize("wide_key", [False, True])
def test_fit_rejects_model_axis_duplicates_in_each_window(tmp_path, day, wide_key):
    source, config = request(tmp_path)
    frame = pd.read_parquet(source)
    if wide_key:
        frame["revision"] = 0
        config["input_spec"]["primary_key"].append("revision")
        config["input_spec"]["column_roles"]["revision"] = "key"
    duplicate = frame.loc[frame.timestamp.dt.day.eq(day)].iloc[[0]].copy()
    if wide_key:
        duplicate["revision"] = 1
    pd.concat([frame, duplicate], ignore_index=True).to_parquet(source, index=False)
    error = (
        "model_training_keys_invalid" if wide_key else "duplicate_dataset_primary_key"
    )
    with pytest.raises(ValueError, match=error):
        fit(config)


def test_fit_keeps_parent_selection_and_maturity(tmp_path):
    source, config = request(tmp_path)
    frame = pd.read_parquet(source)
    frame.loc[frame.asset_id.eq("c"), "label_available_at"] = pd.Timestamp("2020-01-10")
    frame.to_parquet(source, index=False)
    spec = config.pop("input_spec")
    spec["label_availability_columns"] = {"label": "label_available_at"}
    config["inputs"] = {
        "path": str(source),
        "spec": spec,
        "time_window": {"start": "2020-01-02", "end": "2020-01-06"},
    }
    result, _ = load_model(fit(config))
    assert result.metadata["rows_after_input_label_dropna"] == 2


@pytest.mark.parametrize(
    "first,second,valid",
    [
        (("2020-01-01", "2020-01-01"), ("2020-01-01T12:00", "2020-01-01T13:00"), False),
        (("2020-01-01", "2020-01-01"), ("2020-01-02T00:00", "2020-01-02T01:00"), True),
        (
            ("2020-01-01T00:00:00.000000001", "2020-01-01T00:00:00.000000002"),
            ("2020-01-01T00:00:00.000000002", "2020-01-01T00:00:00.000000003"),
            False,
        ),
        (
            ("2020-01-01T00:00:00.000000001", "2020-01-01T00:00:00.000000002"),
            ("2020-01-01T00:00:00.000000003", "2020-01-01T00:00:00.000000004"),
            True,
        ),
    ],
)
def test_fit_windows_are_effectively_disjoint(first, second, valid):
    from quant_research_platform.contracts.models import _fit_windows

    windows = [{"start": start, "end": end} for start, end in (first, second)]
    if valid:
        assert len(_fit_windows(windows, pd.Timestamp("2020-01-05"))) == 2
    else:
        with pytest.raises(ValueError, match="fit_windows_must_be_ordered_disjoint"):
            _fit_windows(windows, pd.Timestamp("2020-01-05"))


def test_fit_keeps_complete_cross_sections_across_reader_batches(tmp_path, monkeypatch):
    from quant_research_platform.research.models import production

    _, config = request(tmp_path)
    reader = production.iter_dataset_chunks
    monkeypatch.setattr(
        production, "iter_dataset_chunks", lambda session: reader(session, batch_rows=2)
    )
    result, _ = load_model(fit(config))
    assert result.metadata["rows_after_input_label_dropna"] == 6
    assert result.metadata["training_asset_ids"] == ["a", "b", "c"]
    assert result.metadata["train_start"] == "2020-01-01"
    assert result.metadata["train_end"] == "2020-01-05"


def test_fit_accepts_empty_first_window_and_asset_selection(tmp_path):
    _, config = request(tmp_path)
    config["fit_windows"][0] = {"start": "2019-12-31", "end": "2019-12-31"}
    config["asset_ids"] = ["a", "c"]
    result, _ = load_model(fit(config))
    assert result.metadata["rows_after_input_label_dropna"] == 2
    assert result.metadata["training_asset_ids"] == ["a", "c"]
