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


@pytest.mark.parametrize("selected_assets", [[], ["a0"]])
def test_portfolio_bar_read_preserves_selected_history_and_empty_dates(
    tmp_path, monkeypatch, selected_assets
):
    from tests.research.test_closed_market_portfolios import _native_inputs

    layout, roots, calendar, _ = _native_inputs(tmp_path)
    materialize = portfolio_sessions.materialize_dataset_session
    bar_reads = []

    def observed(session):
        frame = materialize(session)
        if tuple(frame.columns) == portfolio_sessions._MARKET_FACT_COLUMNS:
            bar_reads.append(frame)
        return frame

    monkeypatch.setattr(portfolio_sessions, "materialize_dataset_session", observed)
    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root={
                "path": str(layout["market_bars"]),
                "time_window": {"start": "2022-12-15", "end": "2023-01-06"},
                "row_filters": [
                    ["source_asset_id", "in", selected_assets or ["absent"]]
                ],
            },
            stock_master_root=layout["stock_master"],
            panel_root=roots["market_panel_1d"],
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2023-01-02", "2023-01-04"),
        )
    )
    assert [view.timestamp for view in views] == list(
        pd.date_range("2023-01-02", "2023-01-04")
    )
    assert len(bar_reads) == 1
    bars = bar_reads[0]
    assert set(bars.source_asset_id) == set(selected_assets)
    assert bars.timestamp.between("2022-12-15", "2023-01-04").all()
    if selected_assets:
        assert bars.timestamp.min() == pd.Timestamp("2022-12-15")
        assert bars.timestamp.max() == pd.Timestamp("2023-01-04")
    else:
        assert bars.empty


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
    _, _, membership, _, _ = producer._processed_axis(
        session, panel_source=True, calendar=calendar, stock_master=stocks
    )
    assert membership.timestamp.tolist() == [days[0], days[2]]
    assert membership.market_type.tolist() == [1, 4]


@pytest.mark.parametrize("source_name", ("panel", "execution"))
@pytest.mark.parametrize("selected", (False, True))
def test_bounded_backtest_ignores_future_bad_keys_and_keeps_required_history(
    tmp_path, source_name, selected
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.contracts.portfolios import (
        ExecutionAssumptions,
        PortfolioCandidateSpec,
    )
    from tests.helpers.portfolio_ledger import session_frame
    from tests.research.test_closed_market_portfolios import _native_inputs

    inputs, _, _, _ = _native_inputs(tmp_path, fault="normal_open")
    root = inputs[source_name]
    if selected:
        inputs[source_name] = {
            "path": str(root),
            "files": sorted(
                str(path.relative_to(root)) for path in root.rglob("*.parquet")
            ),
            "columns": ["open", "close", "amount"]
            if source_name == "panel"
            else ["can_enter", "can_exit"],
            "years": [2022, 2023],
            "time_window": {"start": "2022-12-01", "end": "2023-01-06"},
            "row_filters": [["market_id", "==", "a_share"]],
        }

    class Predictions:
        def values_for_frame(self, frame, **kwargs):
            return [
                10.0 if str(asset).startswith("a") else 1.0 for asset in frame.asset_id
            ], [True] * len(frame)

        def coverage_counts(self):
            return 0, 0

    def backtest(end):
        return list(
            production.backtest_a_share(
                candidate=PortfolioCandidateSpec(
                    selection_group="global",
                    prediction_selector="champion",
                    candidate_id="bounded",
                    construction="quantile_equal_weight",
                    rebalance_schedule="daily",
                    top_quantile=0.5,
                    benchmark_id=None,
                ),
                inputs=inputs,
                output_window=TimeWindow("2022-12-30", end),
                input_window=TimeWindow("2022-12-01", end),
                formal_dataset_window=TimeWindow("2022-12-01", end),
                prediction_lookup=Predictions(),
                assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
            )
        )

    expected = backtest("2023-01-05")
    part = next(root.rglob("year=2023/*.parquet"))
    table = pq.ParquetFile(part).read()
    frame = table.to_pandas()
    position = frame.index[frame.timestamp.eq("2023-01-06")][0]
    frame.loc[position, "asset_id"] = None
    pq.write_table(pa.Table.from_pandas(frame, schema=table.schema), part)

    actual = backtest("2023-01-05")
    assert actual == expected
    assert len(actual) == 3
    assert any(batch.trades for batch in actual)
    first = session_frame(actual[0]).set_index("asset_id")
    assert first.loc["a0", "adv20"] == 1000.0
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        backtest("2023-01-06")


@pytest.mark.parametrize("source_name", ("stock_master", "equity_events"))
@pytest.mark.parametrize("selected", (False, True))
def test_future_lifecycle_and_rights_keys_do_not_affect_current_sessions(
    tmp_path, source_name, selected
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tests.research.test_closed_market_portfolios import _native_inputs
    from tests.sources.reference.test_equity_events import native_equity_inputs

    if source_name == "stock_master":
        inputs, _, calendar, _ = _native_inputs(
            tmp_path, fault="future_lifecycle_append"
        )
        key, columns, filters = (
            "source_asset_id",
            ["company_status"],
            [["company_status", "==", "active"]],
        )
    else:
        inputs, _, calendar = native_equity_inputs(
            tmp_path,
            events=lambda event: [
                event,
                dict(
                    event,
                    event_id="future",
                    effective_date="2023-01-09",
                    publication_date="2023-01-09",
                ),
            ],
        )
        key, columns, filters = (
            "event_id",
            [
                "value_per_old_unit",
                "remaining_old_units_per_old_unit",
                "successor_units_per_old_unit",
            ],
            [["kind", "==", "quote"]],
        )
    root = inputs[source_name]
    source = {"path": str(root)}
    if selected:
        source.update(
            files=sorted(
                str(path.relative_to(root)) for path in root.rglob("*.parquet")
            ),
            columns=columns,
            years=[2022, 2023],
            time_window={"start": "2022-12-01", "end": "2023-01-10"},
            row_filters=filters,
        )

    def sessions(end, chosen=source):
        roots = {**inputs, source_name: chosen}
        return list(
            portfolio_sessions.iter_portfolio_sessions(
                market_bars_root=roots["market_bars"],
                stock_master_root=roots["stock_master"],
                equity_events_root=roots.get("equity_events"),
                panel_root=roots["panel"],
                panel_columns=production._PANEL_COLUMNS,
                calendar=calendar,
                time_window=TimeWindow("2023-01-02", end),
            )
        )

    expected = sessions("2023-01-06")
    damaged = 0
    for part in root.rglob("*.parquet"):
        table = pq.ParquetFile(part).read()
        frame = table.to_pandas()
        future = frame.timestamp.gt("2023-01-06")
        if future.any():
            damaged += int(future.sum())
            frame.loc[future, key] = None
            pq.write_table(pa.Table.from_pandas(frame, schema=table.schema), part)
    assert damaged == 1
    actual = sessions("2023-01-06")
    assert len(actual) == 5
    for before, after in zip(expected, actual, strict=True):
        for name in before.__dataclass_fields__:
            old, new = getattr(before, name), getattr(after, name)
            if isinstance(old, pd.DataFrame):
                pd.testing.assert_frame_equal(old, new, check_exact=True)
            else:
                assert old == new
    with pytest.raises(ValueError, match="dataset_primary_key_null"):
        sessions("2023-01-10")
    narrower = {**source, "time_window": {"start": "2022-12-01", "end": "2023-01-06"}}
    assert len(sessions("2023-01-10", narrower)) == 5


def test_corporate_amount_selection_preserves_cash_and_structural_companions(tmp_path):
    from fractions import Fraction

    from quant_research_platform.research.portfolios.accounting import (
        EquityEntitlements,
    )
    from quant_research_platform.sources.csmar.interim import interim_catalog
    from quant_research_platform.storage.datasets import write_dataset
    from tests.research.test_closed_market_portfolios import _native_inputs

    inputs, _, calendar, _ = _native_inputs(tmp_path, fault="normal_open")
    source = pd.DataFrame(
        {
            "source_asset_id": ["a0"],
            "timestamp": pd.to_datetime(["2023-01-05"]),
            "kind": ["CA"],
            "market_type": pd.Series([1], dtype="Int64"),
            "announcement_date": pd.to_datetime(["2023-01-02"]),
            "settlement_date": pd.to_datetime(["2023-01-06"]),
            "settlement_date_unresolved": False,
            "amount": [0.25],
            "subscription_price": [float("nan")],
        }
    )
    root = tmp_path / "selected-actions"
    write_dataset(root, source, spec=interim_catalog().get("corporate_actions"))

    def sessions(columns):
        return portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=inputs["market_bars"],
            stock_master_root=inputs["stock_master"],
            panel_root=inputs["panel"],
            panel_columns=production._PANEL_COLUMNS,
            corporate_actions_root={"path": str(root), "columns": columns},
            calendar=calendar,
            time_window=TimeWindow("2023-01-02", "2023-01-06"),
        )

    book = EquityEntitlements()
    units = pd.Series({"a0": Fraction(100)}, dtype="object")
    lives = pd.Series({"a0": pd.Timestamp("2022-12-01")})
    paid = 0.0
    for view in sessions(["amount"]):
        units, lives, cash, _ = book.process(
            view.corporate_events, pd.DataFrame(), units, lives, view.timestamp
        )
        paid += cash
        book.register(view.corporate_events, units, lives, view.timestamp)
    assert paid == 25.0
    assert units["a0"] == 100
    with pytest.raises(ValueError, match="portfolio_input_business_columns_excluded"):
        list(sessions(["market_type"]))


def test_portfolio_upper_bound_preserves_narrower_panel_and_execution_selectors(
    tmp_path,
):
    from tests.research.test_closed_market_portfolios import _native_inputs

    inputs, _, calendar, _ = _native_inputs(tmp_path, fault="normal_open")

    def selected(name, columns):
        root = inputs[name]
        return {
            "path": str(root),
            "files": sorted(
                str(path.relative_to(root))
                for path in root.rglob("year=2023/*.parquet")
            ),
            "columns": columns,
            "years": [2023],
            "time_window": {"start": "2022-12-15", "end": "2023-01-03"},
            "row_filters": [["asset_id", "in", ["a0"]]],
        }

    views = list(
        portfolio_sessions.iter_portfolio_sessions(
            market_bars_root=inputs["market_bars"],
            stock_master_root=inputs["stock_master"],
            panel_root=selected("panel", ["open", "close", "amount"]),
            execution_root=selected("execution", ["can_enter", "can_exit"]),
            panel_columns=production._PANEL_COLUMNS,
            calendar=calendar,
            time_window=TimeWindow("2023-01-02", "2023-01-05"),
        )
    )
    assert [set(view.panel.asset_id) for view in views] == [
        {"a0"},
        {"a0"},
        set(),
        set(),
    ]
    assert views[0].execution.execution_at.tolist() == [pd.Timestamp("2023-01-03")]
    assert all(set(view.execution.asset_id) == {"a0"} for view in views)
    assert views[-1].execution.timestamp.tolist() == [pd.Timestamp("2023-01-03")]
    assert views[-1].execution.execution_at.isna().all()


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


@pytest.mark.parametrize(
    "name",
    [
        "dataset",
        "panel",
        "execution",
        "calendar",
        "fees",
        "market_bars",
        "stock_master",
        "corporate_actions",
    ],
)
@pytest.mark.parametrize("market", [None, "a_share", "crypto"])
def test_public_portfolio_respects_explicit_input_market(tmp_path, name, market):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from tests.research.test_portfolio_missing_predictions import (
        _public_prediction_config,
        _replace_public_input,
    )

    config = _public_prediction_config(tmp_path)
    _replace_public_input(
        tmp_path,
        config,
        name,
        lambda frame: (
            frame
            if market is None or "market_id" not in frame
            else frame.assign(market_id=market)
        ),
        market=market,
    )
    if market == "crypto":
        with pytest.raises(ValueError, match="^input_market_mismatch$"):
            production.backtest(config)
    else:
        ledger = read_portfolio_ledger_artifact(production.backtest(config))
        assert len(list(ledger.iter_chunks())) == 3


@pytest.mark.parametrize("market", [None, "a_share", "crypto"])
def test_risk_context_checks_only_consumed_declared_market(tmp_path, market):
    from contextlib import ExitStack

    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.storage.datasets import write_dataset

    fields = dict(
        data_id="risk",
        frequency="1d",
        grain="asset_timestamp",
        column_roles={"asset_id": "key", "timestamp": "timestamp", "style": "factor"},
        primary_key=("asset_id", "timestamp"),
        partition_columns=(),
    )
    spec = (
        DataSpec(**fields)
        if market is None
        else ProcessedDataSpec(**fields, market_id=market, kind="factor")
    )
    frame = pd.DataFrame(
        {"asset_id": ["a"], "timestamp": pd.to_datetime(["2023-01-02"]), "style": [1.0]}
    )
    root = tmp_path / "risk"
    write_dataset(root, frame, spec=spec)
    with ExitStack() as channels:
        if market == "crypto":
            with pytest.raises(ValueError, match="^input_market_mismatch$"):
                production._risk_context_input_sessions(
                    roots=(root,),
                    style_columns=("style",),
                    time_window=TimeWindow("2023-01-02", "2023-01-02"),
                    retained=channels,
                )
        else:
            inputs = production._risk_context_input_sessions(
                roots=(root,),
                style_columns=("style",),
                time_window=TimeWindow("2023-01-02", "2023-01-02"),
                retained=channels,
            )
            pd.testing.assert_frame_equal(
                next(inputs[0][1]).reset_index(drop=True),
                frame.astype({"timestamp": "datetime64[ns]"}).loc[
                    :, list(inputs[0][0])
                ],
            )
        assert (
            production._risk_context_input_sessions(
                roots=(root,),
                style_columns=(),
                time_window=TimeWindow("2023-01-02", "2023-01-02"),
                retained=channels,
            )
            == {}
        )


@pytest.mark.parametrize("market", [None, "a_share", "crypto"])
def test_benchmark_consumption_respects_declared_market(tmp_path, market):
    from quant_research_platform.contracts.data import ProcessedDataSpec
    from quant_research_platform.storage.datasets import write_dataset

    fields = dict(
        data_id="benchmark",
        frequency="1d",
        grain="asset_timestamp",
        column_roles={
            "asset_id": "key",
            "timestamp": "timestamp",
            "benchmark_id": "context",
            "weight": "context",
        },
        primary_key=("asset_id", "timestamp"),
        partition_columns=(),
    )
    spec = (
        DataSpec(**fields)
        if market is None
        else ProcessedDataSpec(**fields, market_id=market, kind="annotation")
    )
    frame = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": pd.to_datetime(["2023-01-02"]),
            "benchmark_id": ["index"],
            "weight": [1.0],
        }
    )
    root = tmp_path / "benchmark"
    write_dataset(root, frame, spec=spec)
    weights = production._iter_benchmark_weights(
        root, "index", pd.DatetimeIndex(["2023-01-02"])
    )
    if market == "crypto":
        with pytest.raises(ValueError, match="^input_market_mismatch$"):
            list(weights)
    else:
        assert [row.to_dict() for row in weights] == [{"a": 1.0}]


@pytest.mark.parametrize("market", [None, "a_share", "crypto"])
def test_equity_event_consumption_respects_declared_market(tmp_path, market):
    from tests.research.test_portfolio_missing_predictions import _replace_public_input
    from tests.sources.reference.test_equity_events import native_equity_inputs

    inputs, _, calendar = native_equity_inputs(tmp_path / "market")
    _replace_public_input(
        tmp_path,
        {"inputs": inputs},
        "equity_events",
        lambda frame: frame,
        market=market,
    )
    views = portfolio_sessions.iter_portfolio_sessions(
        market_bars_root=inputs["market_bars"],
        stock_master_root=inputs["stock_master"],
        equity_events_root=inputs["equity_events"],
        panel_root=inputs["panel"],
        panel_columns=production._PANEL_COLUMNS,
        calendar=calendar,
        time_window=TimeWindow("2022-12-30", "2023-01-06"),
    )
    if market == "crypto":
        with pytest.raises(ValueError, match="^input_market_mismatch$"):
            list(views)
    else:
        events = pd.concat([view.rights_events for view in views], ignore_index=True)
        assert events.event_id.tolist() == ["quote-1"]
        assert events.value_per_old_unit.tolist() == [2.5]
