"""Source selection preserves history, entitlements and sample multiplicity."""

import json
from collections import Counter
from dataclasses import asdict

import pandas as pd
import pytest

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share import portfolio_sessions
from quant_research_platform.research.portfolios import production


def test_session_inputs_intersect_years_source_window_history_files_and_filters(
    tmp_path,
):
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "a", "a", "b", "a"],
            "timestamp": pd.to_datetime(
                ["2021-01-02", "2022-01-02", "2023-01-02", "2023-01-02", "2023-01-03"]
            ),
            "value": [1.0, 2.0, 3.0, 99.0, 4.0],
            "chosen": [True, True, True, False, True],
        }
    )
    frame.to_parquet(tmp_path / "selected.parquet", index=False)
    (tmp_path / "unselected.parquet").write_bytes(b"unselected malformed member")
    spec = DataSpec(
        data_id="selected",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("asset_id", "timestamp"),
        partition_columns=(),
        column_roles={
            "asset_id": "key",
            "timestamp": "timestamp",
            "value": "context",
            "chosen": "context",
        },
    )
    source = {
        "path": str(tmp_path),
        "spec": spec,
        "files": ["selected.parquet"],
        "columns": ["value"],
        "years": [2021, 2023],
        "time_window": {"start": "2021-01-01", "end": "2023-01-02"},
        "row_filters": [["chosen", "==", True]],
    }
    frames = list(
        production._iter_input_sessions(
            source,
            ("asset_id", "timestamp", "value"),
            TimeWindow("2022-01-01", "2024-01-01"),
            business_columns=("value",),
        )
    )
    assert len(frames) == 1
    assert frames[0].to_dict("list") == {
        "asset_id": ["a"],
        "timestamp": [pd.Timestamp("2023-01-02")],
        "value": [3.0],
    }
    assert (
        list(
            production._iter_input_sessions(
                {**source, "years": []},
                ("asset_id", "timestamp", "value"),
                TimeWindow("2022-01-01", "2024-01-01"),
            )
        )
        == []
    )


@pytest.mark.parametrize("exclude_history", [False, True])
def test_raw_rights_entitlement_is_independent_of_adjusted_panel_history(
    tmp_path, monkeypatch, exclude_history
):
    from tests.sources.reference.test_equity_events import native_equity_inputs

    inputs, _, calendar = native_equity_inputs(tmp_path)
    calls = Counter()
    original = portfolio_sessions.open_dataset_session

    def observed(*args, **kwargs):
        session = original(*args, **kwargs)
        calls[session.root] += 1
        return session

    monkeypatch.setattr(portfolio_sessions, "open_dataset_session", observed)
    panel = {
        "path": str(inputs["panel"]),
        "years": [2023] if exclude_history else [2022, 2023],
        "columns": ["open", "close", "amount"],
    }
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root={
                "path": str(inputs["market_bars"]),
                "row_filters": [["filling_flag", "in", [0, 1, 2]]],
            },
            stock_master_root={"path": str(inputs["stock_master"])},
            equity_events_root={
                "path": str(inputs["equity_events"]),
                "columns": [
                    "value_per_old_unit",
                    "remaining_old_units_per_old_unit",
                    "successor_units_per_old_unit",
                ],
            },
            panel_root=panel,
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2023-01-06", "2023-01-06"),
        )
    )
    assert len(views) == 1
    assert views[0].rights_events.entitlement_proved.tolist() == [True]
    assert calls[inputs["panel"]] == 1


@pytest.mark.parametrize("native_schedule", [False, True])
def test_calibration_selection_preserves_duplicate_observations(
    tmp_path, native_schedule
):
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.portfolio_inputs import _interim_specs

    calendar_dates = pd.date_range("2023-01-01", "2023-02-02")
    calendar_root = tmp_path / "calendar"
    write_dataset(
        calendar_root,
        pd.DataFrame(
            {
                "timestamp": calendar_dates,
                "calendar_date": calendar_dates,
                "source_market_type": 1,
                "is_trading_day": True,
            }
        ),
        spec=_interim_specs().get("trading_calendar"),
    )
    days = pd.date_range("2023-01-01", periods=8)
    observations = pd.DataFrame(
        [
            {
                "timestamp": day,
                "prediction_rank_score": score,
                "holding_return": 0.01 * score + i * 0.0001,
                "include": i < 6,
            }
            for i, day in enumerate(days)
            for score in (-1.0, -1.0, 0.0, 1.0)
        ]
    )
    intervals = pd.DataFrame(
        {
            "timestamp": days,
            "entry_at": days + pd.Timedelta(days=1),
            "available_at": days + pd.Timedelta(days=2),
        }
    )
    observations.to_parquet(tmp_path / "observations.parquet", index=False)
    intervals.to_parquet(tmp_path / "intervals.parquet", index=False)
    if native_schedule:
        from quant_research_platform.storage.datasets.writing import write_dataset

        root = tmp_path / "intervals"
        write_dataset(
            root,
            intervals,
            spec=DataSpec(
                data_id="intervals",
                frequency="1d",
                grain="market_timestamp",
                primary_key=("timestamp",),
                partition_columns=(),
                column_roles={
                    "timestamp": "timestamp",
                    "entry_at": "context",
                    "available_at": "context",
                },
            ),
        )
        schedule = {
            "path": str(root),
            "columns": ["entry_at"],
            "time_window": {"start": "2023-01-01", "end": "2023-01-06"},
        }
    else:
        schedule = str(tmp_path / "intervals.parquet")
        intervals.iloc[:6].to_parquet(schedule, index=False)
    options = {
        "path": str(tmp_path / "observations.parquet"),
        "years": [2021, 2023],
        "columns": ["prediction_rank_score", "holding_return"],
        "time_window": {"start": "2023-01-01", "end": "2023-01-31"},
        "row_filters": [["include", "==", True]],
    }
    result = production.calibrate(
        {
            "observations": options,
            "scheduled_intervals": schedule,
            "rebalance_schedule": "daily",
            "calendar": {
                "path": str(calendar_root),
                "time_window": {"start": "2023-01-01", "end": "2023-01-08"},
            },
            "cutoff": "2023-02-01",
            "selection_group": "global",
            "output": str(tmp_path / "calibration.json"),
        }
    )
    expected = production.calibrate_expected_alpha(
        observations.loc[observations.include],
        scheduled_intervals=intervals.iloc[:6],
        trading_sessions=days,
        as_of=pd.Timestamp("2023-02-01"),
    )
    assert expected.status == "DEFINED"
    assert json.loads(result.read_text())["calibration"] == asdict(expected)
    with pytest.raises(ValueError, match="portfolio_input_business_columns_excluded"):
        production.calibrate(
            {
                "observations": {**options, "columns": ["prediction_rank_score"]},
                "scheduled_intervals": schedule,
                "rebalance_schedule": "daily",
                "calendar": str(calendar_root),
                "cutoff": "2023-02-01",
                "selection_group": "global",
                "output": str(tmp_path / "excluded.json"),
            }
        )
    assert not (tmp_path / "excluded.json").exists()


def test_risk_context_selection_uses_selected_columns_not_unrelated_schema(tmp_path):
    from contextlib import ExitStack

    from quant_research_platform.storage.datasets.writing import write_dataset

    spec = DataSpec(
        data_id="risk",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("asset_id", "timestamp"),
        partition_columns=(),
        column_roles={
            "asset_id": "key",
            "timestamp": "timestamp",
            "value": "context",
            "unrelated": "context",
        },
    )
    root = tmp_path / "risk"
    write_dataset(
        root,
        pd.DataFrame(
            {
                "asset_id": ["a"],
                "timestamp": [pd.Timestamp("2023-01-02")],
                "value": [1.0],
                "unrelated": [2.0],
            }
        ),
        spec=spec,
    )
    with ExitStack() as stack:
        contexts = production._risk_context_input_sessions(
            roots=({"path": str(root), "columns": ["value"]},),
            style_columns=("value",),
            time_window=TimeWindow("2023-01-01", "2023-01-03"),
            retained=stack,
        )
        columns, frames = contexts[0]
        assert columns == ("timestamp", "asset_id", "value")
        assert next(frames).value.tolist() == [1.0]
    with (
        ExitStack() as stack,
        pytest.raises(ValueError, match="portfolio_risk_context_schema_invalid"),
    ):
        production._risk_context_input_sessions(
            roots=({"path": str(root), "columns": ["unrelated"]},),
            style_columns=("value",),
            time_window=TimeWindow("2023-01-01", "2023-01-03"),
            retained=stack,
        )


def test_market_axis_reprojections_keep_filter_support_during_migration(tmp_path):
    from quant_research_platform.markets.implementations.a_share.processed import (
        panels,
        producer,
    )
    from quant_research_platform.storage.datasets.reading import open_dataset_session

    days = pd.date_range("2023-01-02", periods=3)
    frame = pd.DataFrame(
        {
            "source_asset_id": ["a"] * 3,
            "timestamp": days,
            "market_type": [1, 4, 4],
            "filling_flag": [0, 0, 0],
            "selected": [True, False, True],
        }
    )
    path = tmp_path / "bars.parquet"
    frame.to_parquet(path, index=False)
    session = open_dataset_session(
        path,
        row_filters=(("selected", "==", True),),
        spec=DataSpec(
            data_id="bars",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("source_asset_id", "timestamp"),
            partition_columns=(),
            column_roles={
                "source_asset_id": "key",
                "timestamp": "timestamp",
                "market_type": "context",
                "filling_flag": "context",
                "selected": "context",
            },
        ),
    )
    stocks = panels._canonical_stock_master(
        pd.DataFrame(
            {
                "source_asset_id": ["a"],
                "timestamp": [days[0]],
                "listing_date": [days[0]],
                "delisting_date": [pd.NaT],
                "company_status": ["active"],
            }
        )
    )
    calendar = pd.DataFrame(
        [
            {"calendar_date": day, "source_market_type": market, "is_trading_day": True}
            for day in days
            for market in (1, 4)
        ]
    )
    _, _, membership, _, _, _ = producer._processed_axis(
        session, panel_source=True, calendar=calendar, stock_master=stocks
    )
    assert membership.timestamp.tolist() == [days[0], days[2]]
    assert membership.market_type.tolist() == [1, 4]


@pytest.mark.parametrize("operation", [production.backtest, production.calibrate])
def test_stage_configuration_rejects_non_json_types_before_opening_inputs(
    tmp_path, operation
):
    with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
        operation(
            {
                "observations": {
                    "path": str(tmp_path / "missing"),
                    "time_window": TimeWindow("2020-01-01", "2020-01-02"),
                }
            }
        )
    assert not list(tmp_path.iterdir())
