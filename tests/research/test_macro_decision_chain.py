"""Native macro inputs retain their real decision clock through assembly and fitting."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.data import ProcessedDataSpec
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.markets.implementations.a_share.processed.macro import (
    decision_panel,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.portfolios.daily import (
    _decision_session_frames,
    _prediction_alpha,
)
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.protocols.a_share import factor_catalog
from quant_research_platform.sources.macro.interim import (
    interim_catalog,
    read_release_csv,
)
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)
from quant_research_platform.storage.datasets.reading import open_dataset_session


def test_native_preopen_macro_assembly_fit_predict_and_account_keys(tmp_path):
    sessions = pd.bdate_range("2024-01-04", periods=8)
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": ["A", "B"] * 7,
            "timestamp": sessions[:-1].repeat(2),
            "eligible": True,
            "industry_standard_id": ["CSRC2012", "CAPCO"] * 7,
            "industry_level1_code": "A",
            "volatility_60d": [1.0, 3.0] * 7,
            "target": np.repeat(np.arange(1, 8), 2) * 0.001,
            "target_available_at": sessions[1:].repeat(2) + pd.Timedelta(hours=7),
        }
    )
    roles = {name: "context" for name in frame}
    roles.update(
        market_id="key",
        asset_id="key",
        timestamp="timestamp",
        eligible="mask",
        target="label",
    )
    spec = ProcessedDataSpec(
        data_id="daily",
        market_id="a_share",
        kind="panel",
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        frequency="1d",
        grain="asset_timestamp",
        column_roles=roles,
        column_units={"target": "decimal_return"},
        label_availability_columns={"target": "target_available_at"},
    )
    daily = tmp_path / "daily"
    write_dataset(daily, frame, spec=spec)
    calendar = pd.DataFrame(
        {"calendar_date": sessions, "source_market_type": 1, "is_trading_day": True}
    )
    calendar_path = tmp_path / "calendar.parquet"
    calendar.to_parquet(calendar_path, index=False)
    calendar_input = {
        "path": str(calendar_path),
        "spec": {
            "data_id": "calendar",
            "frequency": "1d",
            "grain": "source_market_calendar_date",
            "primary_key": ["calendar_date", "source_market_type"],
            "partition_columns": [],
            "column_roles": {
                "calendar_date": "timestamp",
                "source_market_type": "key",
                "is_trading_day": "mask",
            },
        },
    }
    decisions = prepare(
        {
            "market_id": "a_share",
            "data_id": "decision_panel",
            "inputs": {"daily_dataset": str(daily), "trading_calendar": calendar_input},
            "time_window": {
                "start": str(sessions[0].date()),
                "end": str(sessions[-2].date()),
            },
            "output_root": str(tmp_path / "decisions"),
            "parameters": {
                "mode": "preopen",
                "local_time": "09:00",
                "latency_seconds": 1,
            },
        }
    )
    releases = pd.DataFrame(
        {
            "series_id": "us_nfci",
            "observation_date": sessions[:-1],
            "value": np.arange(1, 8, dtype=float),
            "unit": "index",
            "frequency": "W",
            "source_id": "synthetic",
            "source_series": "NFCI",
            "source_timezone": "America/Chicago",
            "source_url": "https://example.invalid/test",
            "retrieved_at": "2024-12-31T00:00Z",
            "published_at": [
                (day + pd.Timedelta(minutes=30)).isoformat() + "Z"
                for day in sessions[1:]
            ],
        }
    )
    # A post-decision revision cannot change Monday's 09:00 input.
    revision = releases.iloc[[1]].assign(value=99, published_at="2024-01-08T02:00:00Z")
    release_file = tmp_path / "releases.csv"
    volatility = releases.assign(series_id="us_vix", source_series="VIX", frequency="D")
    pd.concat([releases, revision, volatility], ignore_index=True).to_csv(
        release_file, index=False
    )
    macro = tmp_path / "macro"
    write_dataset(
        macro,
        read_release_csv(release_file),
        spec=interim_catalog().get("macro_observations"),
    )
    column = "ml_us_nfci_level"
    industry_column = "ml_us_vix_change_x_industry_volatility_60d"
    columns = [column, industry_column]
    window = {"start": str(sessions[1].date()), "end": str(sessions[-1].date())}
    factors = prepare(
        {
            "market_id": "a_share",
            "data_id": "macro_liquidity_factors",
            "inputs": {
                "macro_observations": str(macro),
                "decision_panel": str(decisions),
            },
            "time_window": window,
            "factor_columns": columns,
            "output_root": str(tmp_path / "factors"),
        }
    )
    dataset = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "macro_test",
            "inputs": {
                "decision_panel": str(decisions),
                "macro_liquidity_factors": str(factors),
            },
            "base_input_data_id": "decision_panel",
            "time_window": window,
            "factor_columns": columns,
            "label_columns": ["target"],
            "filter_columns": ["eligible"],
            "output_root": str(tmp_path / "dataset"),
        }
    )
    actual = read_dataset(dataset)
    assert inspect_dataset(dataset).spec["feature_transforms"] == []
    assert actual.groupby("signal_session")[column].first().tolist() == list(
        range(1, 8)
    )
    assert actual.timestamp.iloc[2] == pd.Timestamp("2024-01-08T01:00")
    assert actual[industry_column].iloc[:2].isna().all()
    np.testing.assert_allclose(
        actual[industry_column].iloc[2:], np.tile([-1 / np.sqrt(2), 1 / np.sqrt(2)], 6)
    )
    assert factor_catalog().get(column).family == "macro_liquidity"
    model = ModelSpec(
        model_id="macro_test",
        model_algorithm="linear_regression",
        input_factor_columns=(column,),
        target_column="target",
    )
    trained = fit(
        {
            "market_id": "a_share",
            "inputs": str(dataset),
            "output": str(tmp_path / "model"),
            "model": model_spec_payload(model),
            "label_unit": "decimal_return",
            "fit_windows": [
                {"start": "2024-01-05T01:00:00", "end": "2024-01-09T01:00:00"}
            ],
            "fit_cutoff": "2024-01-09T08:00:00",
        }
    )
    predictions = predict(
        {
            "model": str(trained),
            "inputs": str(dataset),
            "output": str(tmp_path / "predictions"),
            "prediction_window": {
                "start": "2024-01-10T01:00:00",
                "end": "2024-01-15T01:00:00",
            },
        }
    )
    predicted, _ = read_predictions(predictions)
    np.testing.assert_allclose(
        predicted.prediction_score, np.repeat(np.arange(4, 8), 2) * 0.001
    )
    first = actual.loc[actual.signal_session.eq(sessions[1])]
    account = next(_decision_session_frames(iter([first]), sessions))
    assert account.timestamp.eq(sessions[1]).all()

    class Lookup:
        def values_for_frame(self, rows, **kwargs):
            assert rows.timestamp.eq(pd.Timestamp("2024-01-08T01:00")).all()
            return np.array([1.0, 2.0]), np.array([True, True])

    assert _prediction_alpha(account, Lookup()).tolist() == [1, 2]


def test_decision_clock_rejects_information_after_opening_price_formation():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-05"])})
    sessions = pd.to_datetime(["2024-01-05", "2024-01-08"])
    for clock, latency in [("09:25", 0), ("09:24:59", 1), ("09:29", 0)]:
        with pytest.raises(ValueError, match="before_opening_auction"):
            decision_panel(
                frame,
                sessions,
                {"mode": "preopen", "local_time": clock, "latency_seconds": latency},
            )
    close = decision_panel(
        frame, sessions, {"mode": "close", "local_time": "16:00", "latency_seconds": 1}
    )
    assert close.timestamp.iloc[0] == pd.Timestamp("2024-01-05T08:00")


@pytest.mark.parametrize(
    "missing_column", [None, "listing_date", "period_return", "trading_calendar"]
)
def test_native_macro_beta_resets_listing_history_without_blocking_levels(
    tmp_path, missing_column
):
    sessions = pd.bdate_range("2024-01-02", periods=17)
    levels = np.arange(16, dtype=float) ** 2
    frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "A",
            "market_type": 1,
            "timestamp": sessions[:-1],
            "listing_date": [sessions[0]] * 5
            + [sessions[5]] * 4
            + [pd.NaT]
            + [sessions[5]] * 6,
            "eligible": True,
            "period_return": np.r_[np.nan, 2 * np.diff(levels)],
        }
    )
    if missing_column not in (None, "trading_calendar"):
        frame = frame.drop(columns=missing_column)
    frame = decision_panel(
        frame,
        sessions,
        {"mode": "preopen", "local_time": "09:00", "latency_seconds": 0},
    )
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(
        market_id="key", asset_id="key", timestamp="timestamp", eligible="mask"
    )
    decision_root = tmp_path / "decisions"
    write_dataset(
        decision_root,
        frame,
        spec=ProcessedDataSpec(
            data_id="decision_panel",
            market_id="a_share",
            kind="panel",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=(),
            frequency="1d",
            grain="asset_timestamp",
            column_roles=roles,
        ),
    )
    releases = pd.DataFrame(
        {
            "series_id": "us_nfci",
            "observation_date": sessions[:-1],
            "value": levels,
            "unit": "index",
            "frequency": "W",
            "source_id": "synthetic",
            "source_series": "NFCI",
            "source_timezone": "UTC",
            "source_url": "https://example.invalid/test",
            "retrieved_at": "2024-12-31T00:00Z",
            "published_at": [day.isoformat() + "Z" for day in sessions[:-1]],
        }
    )
    release_file = tmp_path / "releases.csv"
    releases.to_csv(release_file, index=False)
    macro = tmp_path / "macro"
    write_dataset(
        macro,
        read_release_csv(release_file),
        spec=interim_catalog().get("macro_observations"),
    )
    calendar_path = tmp_path / "calendar.parquet"
    dates = pd.date_range(sessions[0], sessions[-1])
    pd.DataFrame(
        {
            "calendar_date": dates,
            "source_market_type": 1,
            "is_trading_day": dates.isin(sessions),
        }
    ).to_parquet(calendar_path, index=False)
    calendar_input = {
        "path": str(calendar_path),
        "spec": {
            "data_id": "calendar",
            "frequency": "1d",
            "grain": "source_market_calendar_date",
            "primary_key": ["calendar_date", "source_market_type"],
            "partition_columns": [],
            "column_roles": {
                "calendar_date": "timestamp",
                "source_market_type": "key",
                "is_trading_day": "mask",
            },
        },
    }
    factors = prepare(
        {
            "market_id": "a_share",
            "data_id": "macro_liquidity_factors",
            "inputs": {
                "macro_observations": str(macro),
                "decision_panel": str(decision_root),
                **(
                    {"trading_calendar": calendar_input}
                    if missing_column != "trading_calendar"
                    else {}
                ),
            },
            "time_window": {
                "start": str(sessions[1].date()),
                "end": str(sessions[-1].date()),
            },
            "factor_columns": ["ml_us_nfci_level", "ml_us_nfci_beta"],
            "parameters": {
                "beta_window": 4,
                "beta_min_observations": 3,
                "beta_min_changes": 3,
            },
            "output_root": str(tmp_path / "factors"),
        }
    )
    actual = read_dataset(factors).sort_values("timestamp")
    missing = open_dataset_session(factors).manifest["producer_diagnostics"][
        "macro_beta_missing_inputs"
    ]
    assert missing == ([] if missing_column is None else [missing_column])
    # Preopen levels include the new midnight release; return betas stop at close.
    np.testing.assert_allclose(actual.ml_us_nfci_level, np.r_[levels[1:], levels[-1]])
    expected = np.full(16, np.nan)
    if missing_column is None:
        expected[[3, 4, 8, 13, 14, 15]] = 2
    np.testing.assert_allclose(actual.ml_us_nfci_beta, expected, equal_nan=True)


def test_macro_factor_prepare_rejects_conflicting_panel_market(tmp_path):
    sessions = pd.bdate_range("2024-01-02", periods=2)
    frame = decision_panel(
        pd.DataFrame(
            {"market_id": ["crypto"], "asset_id": ["A"], "timestamp": sessions[:1]}
        ),
        sessions,
        {"mode": "preopen", "local_time": "09:00", "latency_seconds": 0},
    )
    roles = dict.fromkeys(frame.columns, "context")
    roles.update(market_id="key", asset_id="key", timestamp="timestamp")
    root = tmp_path / "wrong-market"
    write_dataset(
        root,
        frame,
        spec=ProcessedDataSpec(
            data_id="decision_panel",
            market_id="crypto",
            kind="panel",
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=(),
            frequency="1d",
            grain="asset_timestamp",
            column_roles=roles,
        ),
    )
    with pytest.raises(ValueError, match="^input_market_mismatch$"):
        prepare(
            {
                "market_id": "a_share",
                "data_id": "macro_liquidity_factors",
                "inputs": {
                    "decision_panel": str(root),
                    "macro_observations": str(tmp_path / "unconsumed"),
                },
                "factor_columns": ["ml_us_nfci_level"],
                "time_window": {"start": "2024-01-03", "end": "2024-01-03"},
                "output_root": str(tmp_path / "factors"),
            }
        )
    assert not (tmp_path / "factors").exists()


def test_decision_panel_carries_consumed_label_definition_and_feature_lineage(tmp_path):
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.models.persisted import load_model
    from tests.helpers.research_processed_inputs import (
        assembly_config,
        assembly_frame,
        assembly_source,
    )

    frame = assembly_frame().assign(
        eligible=True,
        industry_standard_id="test",
        industry_level1_code="A",
        log_total_market_cap=np.arange(120) / 10,
        forward_return_20d=np.sin(np.arange(120)) / 100,
        forward_return_20d_available_at=pd.Timestamp("2024-02-01T07:00"),
    )
    _, source_spec = assembly_source(
        tmp_path,
        frame,
        availability={"forward_return_20d": "forward_return_20d_available_at"},
        units={"forward_return_20d": "decimal_return"},
    )
    source = tmp_path / "source"
    semantics = {"entry": "next_open", "horizon_sessions": 20}
    write_dataset(
        source,
        frame,
        spec=source_spec,
        metadata={
            "producer_diagnostics": {
                "label_semantics": {"forward_return_20d": semantics}
            }
        },
    )
    daily = assemble(
        assembly_config(
            tmp_path,
            source,
            feature_recipe="a_share_canonical",
            label_columns=["model_target"],
            filter_columns=["eligible"],
            target_transform="raw",
        )
    )
    calendar = tmp_path / "calendar"
    write_dataset(
        calendar,
        pd.DataFrame(
            {
                "calendar_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "source_market_type": 1,
                "is_trading_day": True,
            }
        ),
        spec=DataSpec(
            data_id="calendar",
            frequency="1d",
            grain="source_market_calendar_date",
            primary_key=("calendar_date", "source_market_type"),
            partition_columns=(),
            column_roles={
                "calendar_date": "timestamp",
                "source_market_type": "key",
                "is_trading_day": "mask",
            },
        ),
    )
    output = prepare(
        {
            "market_id": "a_share",
            "data_id": "decision_panel",
            "inputs": {"daily_dataset": str(daily), "trading_calendar": str(calendar)},
            "time_window": {"start": "2024-01-02", "end": "2024-01-02"},
            "parameters": {
                "mode": "preopen",
                "local_time": "09:00",
                "latency_seconds": 0,
            },
            "output_root": str(tmp_path / "decisions"),
        }
    )
    source_values, carried = read_dataset(daily), read_dataset(output)
    pd.testing.assert_frame_equal(
        carried[list(source_values.columns)].drop(columns="timestamp"),
        source_values.drop(columns="timestamp"),
    )
    facts = inspect_dataset(output).manifest["producer_diagnostics"]
    assert facts["label_semantics"]["forward_return_20d"] == semantics
    assert (
        facts["reused_feature_transforms"]["feature_momentum_20d"]
        == (inspect_dataset(daily).spec["feature_transforms"][0])
    )
    model = fit(
        {
            "market_id": "a_share",
            "inputs": str(output),
            "output": str(tmp_path / "model"),
            "model": {
                "model_id": "carried",
                "model_algorithm": "linear_regression",
                "input_feature_columns": ["feature_momentum_20d"],
                "target_column": "model_target",
            },
            "fit_windows": [{"start": "2024-01-03", "end": "2024-01-03"}],
            "fit_cutoff": "2024-02-01T08:00:00",
        }
    )
    _, fitted_facts = load_model(model)
    assert fitted_facts["target_definition"] == {
        "source_label": "forward_return_20d",
        "target_transform": "raw",
        "target_clip": "none",
        "label_semantics": semantics,
    }


@pytest.mark.parametrize("case", ["complete", "late_predictions", "closed_only"])
def test_preopen_dataset_runs_shared_daily_account_on_next_session(tmp_path, case):
    from quant_research_platform.contracts.portfolios import (
        ExecutionAssumptions,
        PortfolioCandidateSpec,
    )
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.markets.implementations.a_share.calendar import (
        unified_trading_sessions,
    )
    from quant_research_platform.research.portfolios.daily import backtest_a_share
    from tests.helpers.portfolio_inputs import closed_market_inputs

    inputs, roots, calendar, _ = closed_market_inputs(
        tmp_path, fault="closed_only_axis" if case == "closed_only" else None
    )
    extra = calendar.iloc[-len(calendar.source_market_type.unique()) :].copy()
    extra["calendar_date"] = pd.Timestamp("2023-01-09")
    extra["timestamp"] = pd.Timestamp("2023-01-09")
    extra["is_trading_day"] = True
    calendar = pd.concat([calendar, extra], ignore_index=True)
    source_spec = inspect_dataset(roots["trading_calendar"]).spec
    from quant_research_platform.contracts.data import InterimDataSpec

    calendar_spec = InterimDataSpec(
        data_id=source_spec["data_id"],
        source_id=source_spec["source_id"],
        frequency=source_spec["frequency"],
        grain=source_spec["grain"],
        primary_key=tuple(source_spec["primary_key"]),
        partition_columns=("year",),
        column_roles=source_spec["column_roles"],
    )
    calendar_path = tmp_path / "extended_calendar"
    write_dataset(calendar_path, calendar, spec=calendar_spec)
    decisions = prepare(
        {
            "market_id": "a_share",
            "data_id": "decision_panel",
            "inputs": {
                "daily_dataset": str(inputs["dataset"]),
                "trading_calendar": str(calendar_path),
            },
            "time_window": {
                "start": "2022-12-30" if case == "late_predictions" else "2022-12-01",
                "end": "2023-01-06",
            },
            "parameters": {
                "mode": "preopen",
                "local_time": "09:00",
                "latency_seconds": 1,
            },
            "output_root": str(tmp_path / "account_decisions"),
        }
    )
    seen = []

    class Lookup:
        def values_for_frame(self, rows, **kwargs):
            seen.extend(rows.timestamp.tolist())
            assert rows.timestamp.dt.hour.eq(1).all()
            return np.where(rows.asset_id.str.startswith("a"), 10.0, 1.0), np.ones(
                len(rows), dtype=bool
            )

        def coverage_counts(self):
            return 0, 0

    settings = dict(
        candidate=PortfolioCandidateSpec(
            selection_group="global",
            prediction_selector="champion",
            candidate_id="preopen",
            construction="quantile_equal_weight",
            rebalance_schedule="daily",
            top_quantile=0.5,
            benchmark_id=None,
        ),
        inputs={**inputs, "dataset": decisions, "calendar": calendar_path},
        output_window=TimeWindow("2022-12-30", "2023-01-06"),
        input_window=TimeWindow("2022-12-01", "2023-01-06"),
        formal_dataset_window=TimeWindow("2022-12-01", "2023-01-06"),
        prediction_lookup=Lookup(),
        assumptions=ExecutionAssumptions(
            1_000_000, 0, 0, 1, (1,), minimum_commission=0
        ),
    )
    batches = list(backtest_a_share(**settings))
    assert len(batches) == 4
    assert min(seen) == pd.Timestamp("2023-01-02T01:00")
    trades = [trade for batch in batches for trade in batch.trades]
    assert (
        trades
        and pd.Timestamp(trades[0].timestamp_ns).date().isoformat() == "2023-01-02"
    )
    assert pd.Timestamp("2023-01-09") in unified_trading_sessions(calendar)
    from quant_research_platform.contracts.policies import PolicyDecision

    policy_times = []
    policy_batches = []
    stream = backtest_a_share(
        **{
            **settings,
            "prediction_lookup": None,
            "decision_columns": ("signal",),
        }
    )
    try:
        event = next(stream)
        while True:
            if isinstance(event, PolicyDecision):
                policy_times.append(event.timestamp)
                event = stream.send(
                    pd.Series(1 / len(event.features), index=event.features.index)
                )
            else:
                policy_batches.append(event)
                event = next(stream)
    except StopIteration:
        pass
    finally:
        stream.close()
    assert policy_times[0] == pd.Timestamp("2023-01-02T01:00")
    assert pd.DatetimeIndex(policy_times).is_monotonic_increasing
    if case == "closed_only":
        assert any(batch.trades for batch in policy_batches)
        assert all(
            trade.quantity > 0 for batch in policy_batches for trade in batch.trades
        )

    from dataclasses import replace

    from quant_research_platform.markets.implementations.a_share.processed.catalog import (  # noqa: E501
        processed_catalog,
    )

    benchmark = tmp_path / "benchmark"
    base = read_dataset(inputs["dataset"])
    benchmark_rows = pd.MultiIndex.from_product(
        [unified_trading_sessions(calendar), base.asset_id.unique()],
        names=["timestamp", "asset_id"],
    ).to_frame(index=False)
    write_dataset(
        benchmark,
        benchmark_rows.assign(
            market_id="a_share",
            benchmark_id="csi500",
            weight=1 / base.asset_id.nunique(),
        ),
        spec=processed_catalog().get("benchmark_weights"),
    )
    seen.clear()
    enhanced = list(
        backtest_a_share(
            **{
                **settings,
                "candidate": replace(
                    settings["candidate"],
                    construction="benchmark_enhanced",
                    benchmark_id="csi500",
                    top_quantile=None,
                    risk_style_columns=("signal",),
                ),
                "inputs": {
                    **settings["inputs"],
                    "benchmark": benchmark,
                    "risk_context": [inputs["dataset"]],
                },
                "prediction_window": TimeWindow(
                    "2023-01-02T01:00:00", "2023-01-06T01:00:00"
                ),
                "on_final_calibration": lambda result: None,
            }
        )
    )
    assert len(enhanced) == 4
    assert min(seen) == pd.Timestamp("2023-01-02T01:00")
