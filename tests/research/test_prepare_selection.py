"""Public preparation consumes selected mathematical inputs and label horizons."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import producer
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_LABEL_DATA_ID,
    HFT_PANEL_DATA_ID,
)
from quant_research_platform.markets.implementations.crypto.processed.catalog import (
    hft_processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session


def _input(root, name, frame, *, native, crypto=False):
    catalog = hft_processed_catalog() if crypto else processed_catalog()
    spec = catalog.get(name)
    roles = {column: spec.column_roles.get(column, "context") for column in frame}
    spec = replace(spec, column_roles=roles)
    if native:
        path = root / name
        write_dataset(path, frame, spec=spec)
        return {"path": str(path)}
    path = root / (name + ".parquet")
    frame.to_parquet(path, index=False)
    return {
        "path": str(path),
        "spec": {
            "data_id": name,
            "frequency": str(spec.frequency),
            "grain": str(spec.grain),
            "primary_key": list(spec.primary_key),
            "partition_columns": [],
            "column_roles": {
                column: str(role) for column, role in spec.column_roles.items()
            },
        },
    }


def _request(tmp_path, name, inputs, start, end, *, market="a_share", **options):
    return {
        "market_id": market,
        "data_id": name,
        "inputs": inputs,
        "time_window": {"start": start, "end": end},
        "output_root": str(tmp_path / "output"),
        **options,
    }


@pytest.mark.parametrize("native", [False, True])
def test_pit_only_factor_selects_business_columns_and_support_keys(tmp_path, native):
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["a", "b"],
            "timestamp": pd.to_datetime(["2024-01-02"] * 2),
            "cash": [25.0, 60.0],
            "total_assets": [100.0, 100.0],
            "has_legal_fundamental_pit": True,
            "selected": [True, False],
        }
    )
    source = _input(tmp_path, "fundamental_pit", frame, native=native)
    source.update(
        columns=["cash", "total_assets"], row_filters=[["selected", "==", True]]
    )
    request = _request(
        tmp_path,
        "fundamental_factors",
        {"fundamental_pit": source},
        "2024-01-02",
        "2024-01-02",
        factor_columns=["cash_to_assets"],
    )
    result = read_dataset(prepare(request))
    assert result.asset_id.tolist() == ["a"]
    assert result.cash_to_assets.tolist() == [0.25]
    assert result.has_legal_fundamental_pit.tolist() == [True]
    assert set(result) == {
        "market_id",
        "asset_id",
        "timestamp",
        "cash_to_assets",
        "has_legal_fundamental_pit",
    }
    request["factor_columns"] = ["book_to_market"]
    with pytest.raises(ValueError, match="data_operation_input_paths_mismatch"):
        prepare(request)


@pytest.mark.parametrize("native", [False, True])
def test_price_dependent_fundamental_retains_market_cap_input(tmp_path, native):
    keys = {
        "market_id": ["a_share"],
        "asset_id": ["a"],
        "timestamp": pd.to_datetime(["2024-01-02"]),
    }
    pit = _input(
        tmp_path,
        "fundamental_pit",
        pd.DataFrame(
            {**keys, "parent_equity": [25.0], "has_legal_fundamental_pit": [True]}
        ),
        native=native,
    )
    panel = _input(
        tmp_path,
        "market_panel_1d",
        pd.DataFrame({**keys, "total_market_cap": [100.0]}),
        native=native,
    )
    result = read_dataset(
        prepare(
            _request(
                tmp_path,
                "fundamental_factors",
                {"fundamental_pit": pit, "market_panel_1d": panel},
                "2024-01-02",
                "2024-01-02",
                factor_columns=["book_to_market"],
            )
        )
    )
    assert result.book_to_market.tolist() == [0.25]


@pytest.mark.parametrize("native", [False, True])
def test_selected_a_share_label_bounds_real_lookahead_and_persisted_schema(
    tmp_path, monkeypatch, native
):
    dates = pd.bdate_range("2024-01-02", periods=10).as_unit("ns")
    keys = {"market_id": "a_share", "asset_id": "a", "timestamp": dates}
    panel = _input(
        tmp_path,
        "market_panel_1d",
        pd.DataFrame(
            {**keys, "adj_open": np.arange(10.0, 20.0), "listing_date": dates[0]}
        ),
        native=native,
    )
    mask = _input(
        tmp_path,
        "execution_mask",
        pd.DataFrame({**keys, "can_enter": True}),
        native=native,
    )
    panel["columns"] = ["adj_open"]
    mask["columns"] = ["can_enter"]
    observed = []
    materialize = producer.materialize_dataset_session

    def record(session):
        frame = materialize(session)
        observed.append(frame.timestamp.max())
        return frame

    monkeypatch.setattr(producer, "materialize_dataset_session", record)
    result_root = prepare(
        _request(
            tmp_path,
            "forward_returns",
            {"market_panel_1d": panel, "execution_mask": mask},
            "2024-01-02",
            "2024-01-02",
            label_columns=["forward_return_1d"],
        )
    )
    result = read_dataset(result_root)
    assert max(observed) == dates[2]
    assert result.forward_return_1d.tolist() == pytest.approx([1 / 11])
    assert result.forward_return_1d_available_at.tolist() == [
        dates[2] + pd.Timedelta(hours=1, minutes=30)
    ]
    assert set(result) == {
        "market_id",
        "asset_id",
        "timestamp",
        "forward_return_1d",
        "forward_return_1d_available_at",
    }
    assert open_dataset_session(result_root).spec["label_availability_columns"] == {
        "forward_return_1d": "forward_return_1d_available_at"
    }


@pytest.mark.parametrize(
    "labels",
    [
        [],
        ["forward_return_1d", "forward_return_1d"],
        ["forward_return_1d_available_at"],
        ["unknown"],
        "forward_return_1d",
    ],
)
def test_label_selection_is_nonempty_unique_and_names_values(tmp_path, labels):
    request = _request(
        tmp_path,
        "forward_returns",
        {},
        "2024-01-02",
        "2024-01-02",
        label_columns=labels,
    )
    with pytest.raises(ValueError, match="data_operation_label_columns_invalid"):
        prepare(request)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "label,fields,expected",
    [
        ("future_mid_return_500ms", {"mid": np.arange(10.0, 16.0)}, 0.5),
        ("future_mid_direction_500ms", {"mid": np.arange(10.0, 16.0)}, 1.0),
        (
            "maker_bid_sell_pressure_proxy_500ms",
            {"best_bid_qty": 10.0, "observed_taker_sell_volume_100ms": 1.0},
            0.5,
        ),
    ],
)
def test_crypto_exact_label_selection_does_not_require_sibling_family(
    tmp_path, native, label, fields, expected
):
    dates = pd.date_range("2024-01-02", periods=6, freq="100ms").as_unit("ns")
    frame = pd.DataFrame(
        {
            "market_id": "crypto",
            "asset_id": "btc_usdt",
            "timestamp": dates,
            "segment_id": "s1",
            **fields,
        }
    )
    source = _input(tmp_path, HFT_PANEL_DATA_ID, frame, native=native, crypto=True)
    source["columns"] = list(fields)
    result_root = prepare(
        _request(
            tmp_path,
            HFT_LABEL_DATA_ID,
            {HFT_PANEL_DATA_ID: source},
            dates[0].isoformat(),
            dates[0].isoformat(),
            market="crypto",
            label_columns=[label],
            parameters={"asset_ids": ["btc_usdt"]},
        )
    )
    result = read_dataset(result_root)
    assert result[label].tolist() == [expected]
    assert result[label + "_available_at"].tolist() == [dates[5]]
    assert set(result) == {
        "market_id",
        "asset_id",
        "timestamp",
        label,
        label + "_available_at",
    }
    metadata = open_dataset_session(result_root).manifest
    assert [item["output_column"] for item in metadata["parameters"]["labels"]] == [
        label
    ]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("history_start", [0, 10])
def test_selected_factor_keeps_actual_history_before_output_window(
    tmp_path, native, history_start
):
    dates = pd.bdate_range("2024-01-02", periods=25).as_unit("ns")
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "a",
            "timestamp": dates,
            "listing_date": dates[0],
            "adj_close": np.arange(1.0, 26.0),
        }
    )
    source = _input(tmp_path, "market_panel_1d", frame, native=native)
    source.update(
        columns=["adj_close"],
        years=[2024],
        time_window={
            "start": str(dates[history_start].date()),
            "end": str(dates[-1].date()),
        },
    )
    result = read_dataset(
        prepare(
            _request(
                tmp_path,
                "price_volume_facts",
                {"market_panel_1d": source},
                str(dates[-1].date()),
                str(dates[-1].date()),
                factor_columns=["momentum_20d"],
            )
        )
    )
    if history_start:
        assert result.momentum_20d.isna().all()
    else:
        assert result.momentum_20d.tolist() == [4.0]


@pytest.mark.parametrize("batch_rows", [3, producer.DEFAULT_BATCH_ROWS])
@pytest.mark.parametrize("unknown_gap", [False, True])
def test_panel_warmup_keeps_suspension_state_without_consuming_old_industry_conflict(
    tmp_path, monkeypatch, batch_rows, unknown_gap
):
    from tests.helpers.a_share_sources import (
        _prepare,
        _synthetic_interim_frames,
        _write_inputs,
    )

    dates = pd.bdate_range("2023-01-09", periods=4).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    frames["a_share_industry_snapshots"] = frames["a_share_industry_snapshots"].iloc[:0]
    frames["industry_classification"] = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 3,
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-05"]),
            "industry_standard_id": ["CSRC2012", "CAPCO", "CAPCO"],
            "implement_date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-05"]
            ),
            "industry_level1_code": ["A01", "B01", "C01"],
            "industry_level1_name": ["A leaf", "B leaf", "C leaf"],
        }
    )
    frames["stock_info_changes"] = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 3,
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-05"]),
            "effective_date": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-05"]
            ),
            "announcement_date": pd.to_datetime(
                ["2023-01-09", "2023-01-09", "2023-01-10"]
            ),
            "changed_item": ["P0207行业代码", "P0221行业代码", "P0221行业代码"],
            "value_before": [""] * 3,
            "value_after": ["A leaf", "B leaf", "C leaf"],
        }
    )
    frames["industry_classification_versions"] = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "CAPCO"],
            "industry_standard_name": ["CSRC", "CAPCO"],
            "timestamp": pd.to_datetime(["2012-10-26", "2023-01-01"]),
            "implement_date": pd.to_datetime(["2012-10-26", "2023-01-01"]),
            "abandon_date": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
            "institution_name": ["CSRC", "CAPCO"],
        }
    )
    frames["industry_classification_sets"] = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012"] * 2 + ["CAPCO"] * 4,
            "industry_code": ["A", "A01", "B", "B01", "C", "C01"],
            "industry_name": ["A", "A leaf", "B", "B leaf", "C", "C leaf"],
            "parent_industry_code": ["", "A", "", "B", "", "C"],
            "industry_level": ["1", "2", "1", "2", "1", "2"],
            "industry_id": [str(i) for i in range(6)],
        }
    )
    bars = frames["market_bars_1d"]
    suspended = bars.source_asset_id.eq("000001") & bars.timestamp.gt(dates[0])
    bars.loc[suspended, ["filling_flag", "trading_status_code"]] = 1
    bars.loc[suspended, ["open", "high", "low", "close", "adj_close", "adj_open"]] = (
        np.nan
    )
    initial_close = bars.loc[
        bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[0]), "close"
    ].item()
    if unknown_gap:
        frames["market_bars_1d"] = bars.loc[
            ~(bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[1]))
        ].copy()
    inputs = _write_inputs(tmp_path / "inputs", frames)
    monkeypatch.setattr(producer, "DEFAULT_BATCH_ROWS", batch_rows)
    root = _prepare(
        tmp_path / "selected", "market_panel_1d", inputs, dates[-1], dates[-1]
    )
    result = read_dataset(root).query("asset_id == '000001'")
    assert result.industry_level1_code.tolist() == ["C"]
    if unknown_gap:
        assert result.adj_close.isna().all()
    else:
        assert result.adj_close.tolist() == [initial_close]
    assert result.listing_date.tolist() == [frames["stock_master"].listing_date.iloc[0]]
    # The prelisting market membership is unproved in this fixture.
    assert result.listing_age_trading_days.isna().all()
    with pytest.raises(ValueError, match="industry_history_standard_conflict"):
        _prepare(
            tmp_path / "historical", "market_panel_1d", inputs, dates[0], dates[-1]
        )


@pytest.mark.parametrize("freshness", [0, -1, 30.0, True, "30"])
def test_prepare_freshness_rejects_nonpositive_or_noninteger_values(
    tmp_path, freshness
):
    request = _request(
        tmp_path,
        "expectations_factors",
        {},
        "2024-01-02",
        "2024-01-02",
        parameters={"freshness_days": freshness},
    )
    with pytest.raises(ValueError, match="data_operation_freshness_days_invalid"):
        prepare(request)


@pytest.mark.parametrize("native", [False, True])
def test_prepare_source_member_selection_keeps_only_selected_assets_and_history(
    tmp_path, native
):
    from tests.helpers.a_share_sources import (
        _source_catalog,
        _synthetic_interim_frames,
        _write_inputs,
    )

    dates = pd.bdate_range("2023-01-09", periods=4).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    sources = _write_inputs(tmp_path / "source", frames)
    selected = {
        "path": str(sources["market_bars_1d"]),
        "files": ["data/year=2023/part-000.parquet"],
    }
    if native:
        (sources["market_bars_1d"] / "data" / "unselected.parquet").write_bytes(
            b"not parquet"
        )
    else:
        root = tmp_path / "bare"
        root.mkdir()
        frames["market_bars_1d"].to_parquet(root / "selected.parquet", index=False)
        (root / "unselected.parquet").write_bytes(b"not parquet")
        spec = _source_catalog().get("market_bars_1d")
        selected.update(
            path=str(root),
            files=["selected.parquet"],
            spec={
                "data_id": spec.data_id,
                "frequency": str(spec.frequency),
                "grain": str(spec.grain),
                "primary_key": list(spec.primary_key),
                "partition_columns": [],
                "column_roles": {
                    name: str(role) for name, role in spec.column_roles.items()
                },
            },
        )
    selected.update(
        years=[2023],
        row_filters=[["source_asset_id", "==", "000001"]],
        time_window={"start": str(dates[0].date()), "end": str(dates[-1].date())},
    )
    inputs = {
        name: str(sources[name])
        for name in processed_catalog().get("market_panel_1d").source_input_data_ids
    }
    inputs["market_bars_1d"] = selected
    output = prepare(
        _request(
            tmp_path,
            "market_panel_1d",
            inputs,
            str(dates[-1].date()),
            str(dates[-1].date()),
        )
    )
    result = read_dataset(output)
    assert result.asset_id.tolist() == ["000001"]
    assert result.timestamp.tolist() == [dates[-1]]
    assert result.close.tolist() == [14.0]
