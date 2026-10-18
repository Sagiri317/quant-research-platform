"""Native preparation preserves ON/ID clocks, coverage, and declared sessions."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
from tests.helpers.a_share_sources import _calendar_input
from tests.research.test_prepare_selection import _input


def _inputs(tmp_path, *, native=True):
    dates = pd.bdate_range("2023-12-01", periods=45).as_unit("ns")
    frame = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": day,
                "listing_date": pd.Timestamp("2020-01-01"),
                "market_type": 4 if asset == "a" else 1,
                "adj_open": 100.0 + i + number,
                "adj_close": 101.0 + i + number,
                "is_trading": True,
                "selected": i != 2,
                "filling_flag": 0,
            }
            for number, asset in enumerate(("a", "b", "c"))
            for i, day in enumerate(dates)
            if asset != "a" or i % 2 == 0
        ]
    )
    return frame, {
        "trading_calendar": str(
            _calendar_input(tmp_path / "calendar", {1: dates, 4: dates[::2]})
        ),
        "market_panel_1d": _input(tmp_path, "market_panel_1d", frame, native=native),
    }


def _request(tmp_path, inputs, frame, **extra):
    return {
        "market_id": "a_share",
        "data_id": "on_id_decomposition",
        "inputs": inputs,
        "time_window": {
            "start": str(frame.timestamp.min().date()),
            "end": str(frame.timestamp.max().date()),
        },
        "output_root": str(tmp_path),
        **extra,
    }


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("block_rows", (3, 100000))
def test_prepare_roundtrip_and_narrow_slice(tmp_path, monkeypatch, native, block_rows):
    frame, inputs = _inputs(tmp_path, native=native)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", block_rows)
    full_root = prepare(
        _request(tmp_path / "full", inputs, frame, parameters={"horizons": [1, 20]})
    )
    full = read_dataset(full_root)
    dates = sorted(frame.timestamp.unique())
    window = {
        "start": str(pd.Timestamp(dates[18]).date()),
        "end": str(pd.Timestamp(dates[22]).date()),
    }
    narrow = read_dataset(
        prepare(
            {
                **_request(
                    tmp_path / "narrow", inputs, frame, parameters={"horizons": [1, 20]}
                ),
                "time_window": window,
            }
        )
    )
    expected = full.loc[
        full.timestamp.between(window["start"], window["end"])
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        narrow.select_dtypes(exclude="floating"),
        expected.select_dtypes(exclude="floating"),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(narrow.isna(), expected.isna())
    pd.testing.assert_frame_equal(
        narrow, expected, check_exact=False, rtol=1e-12, atol=0
    )
    session = open_dataset_session(full_root)
    spec = session.spec
    assert spec["column_units"]["on_id_endpoint_return_20d"] == "decimal_return"
    assert spec["column_units"]["on_id_holding_intraday_log_return_20d"] == "log_return"
    assert all(
        spec["column_roles"][column] == "context"
        for column in spec["label_availability_columns"].values()
    )
    assert str(full.on_id_endpoint_return_20d.dtype) == "float64"
    assert str(full.on_id_endpoint_return_20d_available_at.dtype) == "datetime64[ns]"
    assert str(full.on_id_endpoint_return_20d_reason.dtype) == "string"
    assert full.on_id_decomposition_qualified_20d.dtype == bool
    assert (
        session.manifest["producer_diagnostics"]["label_semantics"][
            "on_id_endpoint_return_20d"
        ]["corporate_action_validation"]
        == "not_performed"
    )
    a = full.loc[full.asset_id.eq("a")].iloc[0]
    a_axis = frame.loc[frame.asset_id.eq("a"), "timestamp"].sort_values()
    assert a.on_id_endpoint_return_1d_available_at == a_axis.iloc[2] + pd.Timedelta(
        hours=7
    )


def test_price_filter_keeps_full_declared_axis_and_unknown_prices(tmp_path):
    frame, inputs = _inputs(tmp_path)
    inputs["market_panel_1d"]["row_filters"] = [["selected", "==", True]]
    result = read_dataset(
        prepare(
            _request(tmp_path / "filtered", inputs, frame, parameters={"horizons": [1]})
        )
    )
    assert len(result) == int(frame.selected.sum())
    assert result.timestamp.isin(frame.loc[frame.selected, "timestamp"]).all()
    b = result.loc[result.asset_id.eq("b")].reset_index(drop=True)
    assert pd.isna(b.on_id_endpoint_return_1d.iloc[0])
    assert b.on_id_endpoint_return_1d_reason.iloc[0] == "missing_price"
    assert pd.isna(b.on_id_endpoint_return_1d.iloc[1])
    assert np.isfinite(b.on_id_holding_intraday_log_return_1d.iloc[0])
    assert b.on_id_holding_intraday_log_return_1d_available_at.iloc[0] == (
        frame.loc[frame.asset_id.eq("b"), "timestamp"].iloc[1] + pd.Timedelta(hours=7)
    )
    assert pd.isna(b.on_id_endpoint_return_1d_available_at.iloc[0])


@pytest.mark.parametrize("native", [False, True])
def test_selected_asset_clock_does_not_require_unrelated_market(tmp_path, native):
    frame, inputs = _inputs(tmp_path, native=native)
    dates = pd.DatetimeIndex(frame.timestamp.unique()).sort_values()
    frame = frame.loc[~(frame.asset_id.eq("b") & frame.timestamp.gt(dates[-4]))]
    (tmp_path / "missing-tail").mkdir()
    inputs["market_panel_1d"] = _input(
        tmp_path / "missing-tail", "market_panel_1d", frame, native=native
    )
    expected = read_dataset(
        prepare(
            _request(tmp_path / "complete", inputs, frame, parameters={"horizons": [1]})
        )
    )
    expected = expected.loc[expected.asset_id.eq("b")].reset_index(drop=True)
    inputs["market_panel_1d"]["row_filters"] = [["asset_id", "==", "b"]]
    inputs["trading_calendar"] = str(
        _calendar_input(tmp_path / "selected-calendar", {1: dates})
    )
    actual = read_dataset(
        prepare(
            _request(tmp_path / "selected", inputs, frame, parameters={"horizons": [1]})
        )
    )
    assert actual.asset_id.unique().tolist() == ["b"]
    pd.testing.assert_frame_equal(actual, expected)


def test_noncontiguous_year_selection_keeps_sessions_unknown(tmp_path):
    frame, inputs = _inputs(tmp_path)
    inputs["market_panel_1d"]["years"] = [2024]
    result = read_dataset(
        prepare(
            _request(tmp_path / "selected", inputs, frame, parameters={"horizons": [1]})
        )
    )
    assert len(result) == int(frame.timestamp.dt.year.eq(2024).sum())
    assert result.timestamp.dt.year.eq(2024).all()
    first = result.loc[result.asset_id.eq("b")].iloc[0]
    assert np.isfinite(first.on_id_endpoint_return_1d)
    assert first.on_id_endpoint_return_1d_available_at == (
        result.loc[result.asset_id.eq("b"), "timestamp"].iloc[2] + pd.Timedelta(hours=7)
    )


@pytest.mark.parametrize("horizons", ([], [2], [True], [1, 1], "20"))
def test_invalid_horizons_fail_before_input_io(tmp_path, horizons):
    with pytest.raises(ValueError, match="on_id_horizons_invalid"):
        prepare(
            {
                "market_id": "a_share",
                "data_id": "on_id_decomposition",
                "inputs": {},
                "parameters": {"horizons": horizons},
                "time_window": {"start": "2024-01-01", "end": "2024-01-02"},
                "output_root": str(tmp_path / "output"),
            }
        )


def test_assembled_selected_label_semantics_and_real_evaluate_cutoffs(tmp_path):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from quant_research_platform.research.datasets.assembler import assemble
    from quant_research_platform.research.evaluations.runner import evaluate
    from tests.helpers.research_processed_inputs import assembly_source as _source

    frame, inputs = _inputs(tmp_path)
    labels_root = prepare(
        _request(tmp_path / "labels", inputs, frame, parameters={"horizons": [1, 20]})
    )
    first_date = frame.timestamp.min()
    signal = frame.loc[
        frame.timestamp.eq(first_date), ["market_id", "asset_id", "timestamp"]
    ].copy()
    signal["momentum_20d"] = np.arange(len(signal), dtype=float)
    signal_root, _ = _source(tmp_path, signal)
    labels = [
        "on_id_pre_entry_log_return",
        "on_id_holding_intraday_log_return_1d",
        "on_id_holding_overnight_log_return_1d",
        "on_id_endpoint_return_1d",
    ]
    window = {"start": str(first_date.date()), "end": str(first_date.date())}
    assembled = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "on_id_conditions",
            "inputs": {
                "external_table": str(signal_root),
                "on_id_decomposition": str(labels_root),
            },
            "base_input_data_id": "external_table",
            "factor_columns": ["momentum_20d"],
            "label_columns": labels,
            "feature_recipe": None,
            "time_window": window,
            "output_root": str(tmp_path / "assembled"),
        }
    )
    session = open_dataset_session(assembled)
    assert set(session.manifest["producer_diagnostics"]["label_semantics"]) == set(
        labels
    )
    assert (
        "on_id_endpoint_return_20d"
        not in session.manifest["producer_diagnostics"]["label_semantics"]
    )
    actual = read_dataset(assembled)
    for number, label in enumerate(labels):
        mature_column = session.spec["label_availability_columns"][label]
        maturity = actual[mature_column].min()
        for step in (-1, 0, 1):
            cutoff = maturity + pd.Timedelta(step, "ns")
            output = tmp_path / f"eval-{number}-{step}"
            evaluate(
                {
                    "inputs": str(assembled),
                    "output": str(output),
                    "window": window,
                    "cutoff": cutoff.isoformat() + "Z",
                    "evaluation": {
                        "evaluation_id": f"test-{number}-{step}",
                        "market_id": "a_share",
                        "research_dataset_id": "on_id_conditions",
                        "signals": [
                            {
                                "signal_id": "frozen",
                                "source_kind": "factor",
                                "source_column": "momentum_20d",
                            }
                        ],
                        "label_columns": [label],
                        "filter_mode": "unfiltered",
                        "n_groups": 2,
                    },
                }
            )
            manifest, tables = read_evaluation_outputs(output)
            metadata = manifest["metadata"]
            assert metadata["input_facts"]["label_units"] == {
                label: session.spec["column_units"][label]
            }
            assert metadata["input_facts"]["label_semantics"] == {
                label: session.manifest["producer_diagnostics"]["label_semantics"][
                    label
                ]
            }
            assert metadata["label_availability_columns"] == {label: mature_column}
            expected = int(
                (actual[label].notna() & actual[mature_column].le(cutoff)).sum()
            )
            assert tables["metrics"].n_obs.sum() == expected


def test_native_endpoint_parity_lifecycle_and_local_invalid_prices(tmp_path):
    frame, inputs = _inputs(tmp_path)
    valid_root = prepare(
        _request(tmp_path / "valid", inputs, frame, parameters={"horizons": [1, 20]})
    )
    legacy = read_dataset(
        prepare(
            {
                **_request(tmp_path / "legacy", inputs, frame),
                "data_id": "forward_returns",
                "label_columns": ["forward_return_1d", "forward_return_20d"],
            }
        )
    )
    valid = read_dataset(valid_root)
    for horizon in (1, 20):
        actual = valid[f"on_id_endpoint_return_{horizon}d"]
        expected = legacy[f"forward_return_{horizon}d"]
        pd.testing.assert_series_equal(
            actual.isna(), expected.isna(), check_names=False
        )
        pd.testing.assert_series_equal(
            actual, expected, check_names=False, check_exact=False, rtol=1e-12, atol=0
        )
    dates = sorted(frame.timestamp.unique())
    a = frame.asset_id.eq("a")
    frame.loc[a & frame.timestamp.ge(dates[6]), "listing_date"] = dates[6]
    frame.loc[frame.asset_id.eq("b") & frame.timestamp.eq(dates[0]), "listing_date"] = (
        pd.NaT
    )
    frame.loc[frame.asset_id.eq("c") & frame.timestamp.eq(dates[1]), "adj_close"] = (
        np.inf
    )
    frame.loc[frame.asset_id.eq("c") & frame.timestamp.eq(dates[2]), "adj_open"] = 0
    (tmp_path / "changed").mkdir()
    changed = _input(tmp_path / "changed", "market_panel_1d", frame, native=False)
    inputs["market_panel_1d"] = changed
    result = read_dataset(
        prepare(
            _request(
                tmp_path / "changed-output",
                inputs,
                frame,
                parameters={"horizons": [1, 20]},
            )
        )
    )
    b = result.loc[result.asset_id.eq("b")].iloc[0]
    assert b.on_id_pre_entry_log_return_reason == "unknown_listing_lifecycle"
    c = result.loc[result.asset_id.eq("c")].iloc[0]
    assert c.on_id_holding_intraday_log_return_1d_reason == "invalid_price"
    assert c.on_id_endpoint_return_1d_reason == "invalid_price"
    assert np.isfinite(c.on_id_endpoint_return_20d)
    at_boundary = result.loc[
        result.asset_id.eq("a") & result.timestamp.eq(dates[2])
    ].iloc[0]
    assert at_boundary.on_id_endpoint_return_1d_reason == "lifecycle_boundary"
    assert np.isfinite(at_boundary.on_id_holding_intraday_log_return_1d)


def test_native_file_selection_uses_inventory_without_compacting_clock(tmp_path):
    frame, inputs = _inputs(tmp_path)
    panel = open_dataset_session(inputs["market_panel_1d"]["path"])
    selected = [
        str(path.relative_to(panel.root))
        for path in panel.all_paths
        if "year=2024" in str(path)
    ]
    assert selected
    inputs["market_panel_1d"]["files"] = selected
    result = read_dataset(
        prepare(
            _request(
                tmp_path / "selected-files", inputs, frame, parameters={"horizons": [1]}
            )
        )
    )
    assert len(result) == int(frame.timestamp.dt.year.eq(2024).sum())
    assert result.timestamp.dt.year.eq(2024).all()
    row = result.loc[result.asset_id.eq("b")].iloc[0]
    assert np.isfinite(row.on_id_endpoint_return_1d)


def test_price_labels_need_no_execution_mask(tmp_path):
    frame, inputs = _inputs(tmp_path)
    assert set(inputs) == {"market_panel_1d", "trading_calendar"}
    result = read_dataset(
        prepare(
            _request(tmp_path / "output", inputs, frame, parameters={"horizons": [1]})
        )
    )
    assert result.on_id_endpoint_return_1d.notna().any()


def test_actual_panel_axis_survives_market_migration_and_suspension(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=12).as_unit("ns")
    own = dates[[0, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "migrating",
            "timestamp": own,
            "listing_date": pd.Timestamp("2020-01-01"),
            "adj_open": np.arange(100.0, 110.0),
            "adj_close": np.arange(101.0, 111.0),
            "market_type": [1, 1, 1, 4, 4, 4, 4, 4, 4, 4],
            "is_trading": True,
            "filling_flag": 0,
        }
    )
    frame.loc[2, ["adj_open", "adj_close"]] = np.nan
    frame.loc[2, "is_trading"] = False
    inputs = {
        "market_panel_1d": _input(tmp_path, "market_panel_1d", frame, native=True),
        "trading_calendar": str(
            _calendar_input(tmp_path / "calendar", {1: own, 4: dates})
        ),
    }
    result = read_dataset(
        prepare(
            _request(
                tmp_path / "output", inputs, frame, parameters={"horizons": [1, 5]}
            )
        )
    )
    assert result.on_id_endpoint_return_5d_available_at.iloc[0] == own[
        6
    ] + pd.Timedelta(hours=7)
    assert result.on_id_endpoint_return_5d.iloc[0] == pytest.approx(
        (106.0 - 101.0) / 101.0, rel=1e-12, abs=0
    )
    assert result.on_id_decomposition_reason_5d.iloc[0] == "missing_price"
    assert result.on_id_endpoint_return_1d_reason.iloc[0] == "missing_price"


def test_bare_directory_member_selection_cannot_define_a_sparse_clock(tmp_path):
    frame, inputs = _inputs(tmp_path, native=False)
    dates = sorted(frame.timestamp.unique())[:6]
    for data_id in ("market_panel_1d",):
        directory = tmp_path / f"parts-{data_id}"
        directory.mkdir()
        source = frame
        for i, day in enumerate(dates):
            source.loc[source.timestamp.eq(day)].to_parquet(
                directory / f"part-{i}.parquet", index=False
            )
        inputs[data_id] = {
            **inputs[data_id],
            "path": str(directory),
            "files": ["part-0.parquet", "part-2.parquet", "part-4.parquet"],
        }
    with pytest.raises(ValueError, match="a_share_panel_session_domain_unproven"):
        prepare(
            _request(tmp_path / "output", inputs, frame, parameters={"horizons": [1]})
        )


def test_native_infinite_price_is_local_but_general_reader_stays_strict(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame, inputs = _inputs(tmp_path)
    panel = open_dataset_session(inputs["market_panel_1d"]["path"])
    day = sorted(frame.timestamp.unique())[1]
    for path in panel.all_paths:
        table = pq.ParquetFile(path).read()
        affected = np.array(
            [
                (asset == "c" and timestamp == day)
                for asset, timestamp in zip(
                    table["asset_id"].to_pylist(),
                    table["timestamp"].to_pylist(),
                    strict=True,
                )
            ]
        )
        if affected.any():
            values = table["adj_close"].to_numpy().copy()
            values[affected] = np.inf
            table = table.set_column(
                table.schema.get_field_index("adj_close"), "adj_close", pa.array(values)
            )
            pq.write_table(table, path)
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        read_dataset(panel.root)
    result = read_dataset(
        prepare(
            _request(tmp_path / "result", inputs, frame, parameters={"horizons": [1]})
        )
    )
    first = result.loc[result.asset_id.eq("c")].iloc[0]
    assert first.on_id_holding_intraday_log_return_1d_reason == "invalid_price"
    assert first.on_id_holding_overnight_log_return_1d_reason == "invalid_price"
    assert np.isfinite(first.on_id_endpoint_return_1d)


def test_private_infinity_tolerance_does_not_relax_other_consumed_columns(tmp_path):
    from dataclasses import replace

    from quant_research_platform.storage.datasets.reading import (
        _dataset_input_options,
        materialize_dataset_session,
    )

    frame, _ = _inputs(tmp_path)
    frame["unrelated"] = np.inf
    frame.loc[0, "adj_close"] = np.inf
    # Reuse the actual PANEL declaration while adding a consumed context column.
    source = _input(tmp_path, "market_panel_1d", frame, native=False)
    session = open_dataset_session(**_dataset_input_options(source))
    session = replace(session, _allow_infinite_columns=("adj_open", "adj_close"))
    with pytest.raises(ValueError, match="dataset_infinite_value"):
        materialize_dataset_session(session)
