from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.research.portfolios.runner as portfolio_runner
from quant_research_platform.artifacts._output_bundle_validation import (
    _read_prediction_output_bundle,
)
from quant_research_platform.catalog.portfolios import PortfolioCatalog
from quant_research_platform.compute.dataframe.performance import (
    calculate_additive_cumulative_returns,
    summarize_performance,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.portfolios import (
    PortfolioDiagnosticsConfig,
    PortfolioSpec,
    WeightConfig,
)
from quant_research_platform.contracts.predictions import PredictionSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.portfolios.engine import (
    _construct_portfolio_positions,
    compute_exposure_summary,
    summarize_concentration,
    summarize_period_breakdown,
)
from quant_research_platform.research.portfolios.runner import (
    build_portfolio_result_from_research_dataset,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids
from tests.helpers.research_specs import (
    make_portfolio_research_dataset_spec,
    make_research_dataset_catalog,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.prediction_output_helpers import write_prediction_output_and_read


def test_portfolio_read_columns_include_diagnostics_config_columns() -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_column="signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
        diagnostics_config=PortfolioDiagnosticsConfig(
            capacity_amount_column="amount",
            capacity_float_market_cap_column="float_market_cap",
            numeric_exposure_columns=("beta",),
            categorical_exposure_columns=("industry",),
        ),
    )

    assert portfolio_runner._portfolio_read_columns(spec) == (
        "market_id",
        "asset_id",
        "timestamp",
        "signal",
        "forward_return",
        "amount",
        "float_market_cap",
        "beta",
        "industry",
    )


def test_portfolio_streaming_diagnostics_match_canonical_two_signal_chunks() -> None:
    timestamps = pd.date_range("2024-01-01", periods=6, freq="D")
    net_returns = {
        "a": (-0.10, 0.02, 0.02, 0.02, 0.02, 0.02),
        "b": (0.02, -0.01, 0.03, -0.02, 0.04, -0.01),
    }
    return_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        for signal, values in net_returns.items():
            net = values[index]
            return_rows.append(
                {
                    "timestamp": timestamp,
                    "signal": signal,
                    "return_source_column": "label",
                    "portfolio_mode": "long_short",
                    "long_return": net,
                    "short_return": 0.0,
                    "gross_portfolio_return": net,
                    "turnover": 1.0,
                    "cost": 0.0,
                    "net_portfolio_return": net,
                    "long_count": 1,
                    "short_count": 1,
                }
            )
            position_rows.extend(
                (
                    {
                        "timestamp": timestamp,
                        "asset": f"{signal}-long",
                        "signal": signal,
                        "position": 0.6,
                    },
                    {
                        "timestamp": timestamp,
                        "asset": f"{signal}-short",
                        "signal": signal,
                        "position": -0.4,
                    },
                )
            )
    returns = pd.DataFrame(return_rows)
    positions = pd.DataFrame(position_rows)
    chunks = tuple(
        (
            returns.loc[returns["timestamp"].isin(timestamps[start : start + 2])]
            .reset_index(drop=True),
            positions.loc[
                positions["timestamp"].isin(timestamps[start : start + 2])
            ].reset_index(drop=True),
        )
        for start in range(0, len(timestamps), 2)
    )

    expected_additive_cumulative = calculate_additive_cumulative_returns(returns)
    expected = {
        "additive_cumulative_returns": expected_additive_cumulative,
        "performance_summary": summarize_performance(returns),
        "exposure_summary": compute_exposure_summary(positions, returns),
        "concentration": summarize_concentration(positions),
        "period_breakdown": summarize_period_breakdown(returns),
    }

    summary_state = portfolio_runner._PortfolioSummaryState()
    for return_chunk, position_chunk in chunks:
        portfolio_runner._update_portfolio_summary_state(
            summary_state,
            positions=position_chunk,
            portfolio_returns=return_chunk,
        )
    summary_state.returns_complete = True
    actual_additive_cumulative = pd.concat(
        list(portfolio_runner._iter_portfolio_additive_cumulative_chunks(summary_state)),
        ignore_index=True,
    )
    try:
        actual = {
            "additive_cumulative_returns": actual_additive_cumulative,
            **portfolio_runner._portfolio_summary_outputs(summary_state),
        }
        for column in (
            "sharpe_like",
            "sortino_like",
            "calmar_like",
            "skew",
            "kurtosis",
            "profit_factor",
            "payoff_ratio",
            "average_win",
            "average_loss",
        ):
            assert actual["performance_summary"][column].dtype == "float64"
        for column in (
            "sharpe_like_reason",
            "sortino_like_reason",
            "calmar_like_reason",
            "skew_reason",
            "kurtosis_reason",
            "profit_factor_reason",
            "payoff_ratio_reason",
            "average_win_reason",
            "average_loss_reason",
        ):
            assert str(actual["performance_summary"][column].dtype) == "string"

        mismatches: list[str] = []
        for name, expected_frame in expected.items():
            try:
                pd.testing.assert_frame_equal(
                    actual[name].reset_index(drop=True),
                    expected_frame.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    rtol=0.0,
                    atol=1e-12,
                )
            except AssertionError as exc:
                mismatches.append(f"{name}: {exc}")
        assert not mismatches, "\n\n".join(mismatches)
        a_cumulative = actual_additive_cumulative[
            actual_additive_cumulative["signal"] == "a"
        ].sort_values("timestamp")
        assert a_cumulative["high_water_mark"].tolist() == pytest.approx(
            [0.0] * len(timestamps)
        )
        assert a_cumulative["drawdown"].tolist() == pytest.approx(
            [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0]
        )
        a_summary = actual["performance_summary"].loc[
            lambda frame: frame["signal"] == "a"
        ].iloc[0]
        assert a_summary["max_drawdown"] == pytest.approx(-0.10)
        assert a_summary["drawdown_duration"] == len(timestamps)
        assert a_summary["max_drawdown_start"] == ""
        assert a_summary["max_drawdown_valley"] == timestamps[0].isoformat()
        assert a_summary["max_drawdown_recovery"] == timestamps[-1].isoformat()
    finally:
        portfolio_runner._close_portfolio_summary_state(summary_state)


@pytest.mark.parametrize(
    "values",
    (
        (0.01,),
        (0.01, 0.02),
        (-0.01, -0.02),
        (0.01, 0.01),
    ),
)
def test_portfolio_streaming_summary_matches_edge_semantics(
    values: tuple[float, ...],
) -> None:
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(values)),
            "signal": ["factor"] * len(values),
            "return_source_column": ["label"] * len(values),
            "portfolio_mode": ["long_short"] * len(values),
            "long_return": values,
            "short_return": [0.0] * len(values),
            "gross_portfolio_return": values,
            "turnover": [1.0] * len(values),
            "cost": [0.0] * len(values),
            "net_portfolio_return": values,
            "long_count": [1] * len(values),
            "short_count": [1] * len(values),
        }
    )
    expected = summarize_performance(returns)
    summary_state = portfolio_runner._PortfolioSummaryState()
    try:
        for row in returns.itertuples(index=False, name=None):
            portfolio_runner._append_portfolio_return_summary(
                summary_state,
                pd.DataFrame([row], columns=returns.columns),
            )
        actual = portfolio_runner._portfolio_performance_summary(summary_state)
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
    finally:
        portfolio_runner._close_portfolio_summary_state(summary_state)


def test_portfolio_streaming_stats_matches_full_across_fetch_boundary() -> None:
    values = [0.10, 0.10, 0.0] + [0.0] * 4093 + [-0.30, -0.10]
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(values)),
            "signal": ["a"] * len(values),
            "return_source_column": ["label"] * len(values),
            "portfolio_mode": ["long_short"] * len(values),
            "long_return": values,
            "short_return": [0.0] * len(values),
            "gross_portfolio_return": values,
            "turnover": [1.0] * len(values),
            "cost": [0.0] * len(values),
            "net_portfolio_return": values,
            "long_count": [1] * len(values),
            "short_count": [1] * len(values),
        }
    )
    returns = pd.concat(
        [returns, returns.iloc[[0]].assign(signal="b")],
        ignore_index=True,
    )
    expected = summarize_performance(returns)
    summary_state = portfolio_runner._PortfolioSummaryState()
    try:
        portfolio_runner._append_portfolio_return_summary(summary_state, returns)
        actual = portfolio_runner._portfolio_performance_summary(summary_state)
        actual_additive = pd.concat(
            list(
                portfolio_runner._iter_portfolio_additive_cumulative_chunks(
                    summary_state
                )
            ),
            ignore_index=True,
        )
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=False,
            rtol=0.0,
            atol=1e-12,
        )
        for signal in ("a", "b"):
            summary_row = actual.loc[actual["signal"] == signal].iloc[0]
            additive_tail = actual_additive.loc[
                actual_additive["signal"] == signal
            ].iloc[-1]
            assert (
                summary_row["final_additive_cumulative_gross_return"]
                == additive_tail["additive_cumulative_gross_return"]
            )
            assert (
                summary_row["final_additive_cumulative_net_return"]
                == additive_tail["additive_cumulative_net_return"]
            )
    finally:
        portfolio_runner._close_portfolio_summary_state(summary_state)


def test_portfolio_return_summary_checks_disk_before_sqlite_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def reject_disk(_path, *, reserve_bytes: int):  # noqa: ANN001
        calls.append(reserve_bytes)
        if len(calls) > 1:
            raise OSError("portfolio_disk_preflight")

    monkeypatch.setattr(
        portfolio_runner,
        "require_free_disk_headroom",
        reject_disk,
    )
    returns = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "signal": ["factor"],
            "return_source_column": ["label"],
            "portfolio_mode": ["long_short"],
            "long_return": [0.01],
            "short_return": [0.0],
            "gross_portfolio_return": [0.01],
            "turnover": [1.0],
            "cost": [0.0],
            "net_portfolio_return": [0.01],
            "long_count": [1],
            "short_count": [1],
        }
    )
    state = portfolio_runner._PortfolioSummaryState()

    with pytest.raises(OSError, match="portfolio_disk_preflight"):
        portfolio_runner._append_portfolio_return_summary(state, returns)

    assert len(calls) == 2
    assert state.return_database is not None
    assert (
        state.return_database.execute("SELECT COUNT(*) FROM returns").fetchone()[0]
        == 0
    )
    portfolio_runner._close_portfolio_summary_state(state)
    assert state.return_database is None
    assert state.return_scratch is None


def test_portfolio_return_summary_grows_main_and_temp_page_caps_before_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def allow_disk(_path, *, reserve_bytes: int) -> None:  # noqa: ANN001
        calls.append(reserve_bytes)

    monkeypatch.setattr(
        portfolio_runner,
        "require_free_disk_headroom",
        allow_disk,
    )
    database = sqlite3.connect(":memory:")
    try:
        database.execute(
            f"PRAGMA page_size={portfolio_runner._PORTFOLIO_SQLITE_PAGE_BYTES}"
        )
        database.execute("PRAGMA temp_store=FILE")
        portfolio_runner._admit_portfolio_disk_growth(
            database,
            Path("."),
            row_count=0,
        )
        startup_caps = {
            schema: int(
                database.execute(f"PRAGMA {schema}.max_page_count").fetchone()[0]
            )
            for schema in ("main", "temp")
        }
        portfolio_runner._admit_portfolio_disk_growth(
            database,
            Path("."),
            row_count=4096,
        )
        first_batch_caps = {
            schema: int(
                database.execute(f"PRAGMA {schema}.max_page_count").fetchone()[0]
            )
            for schema in ("main", "temp")
        }
        portfolio_runner._admit_portfolio_disk_growth(
            database,
            Path("."),
            row_count=4096,
        )
        for schema, startup_cap in startup_caps.items():
            assert (
                int(
                    database.execute(
                        f"PRAGMA {schema}.max_page_count"
                    ).fetchone()[0]
                )
                > startup_cap
            )
        assert (
            int(database.execute("PRAGMA temp.max_page_count").fetchone()[0])
            > first_batch_caps["temp"]
        )
        assert len(calls) == 3
    finally:
        database.close()


def test_portfolio_return_summary_admits_cumulative_query_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    phases: list[str] = []

    def allow_disk(_path, *, reserve_bytes: int) -> None:  # noqa: ANN001
        calls.append(reserve_bytes)

    real_ordered_stats = portfolio_runner._ordered_performance_stats

    def observe_ordered_stats(summary_state):  # noqa: ANN001
        phases.append("query")
        return real_ordered_stats(summary_state)

    monkeypatch.setattr(portfolio_runner, "require_free_disk_headroom", allow_disk)
    monkeypatch.setattr(
        portfolio_runner,
        "_ordered_performance_stats",
        observe_ordered_stats,
    )
    rows = 4096
    returns = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min"),
            "signal": ["factor"] * rows,
            "return_source_column": ["label"] * rows,
            "portfolio_mode": ["long_short"] * rows,
            "long_return": [0.01] * rows,
            "short_return": [0.0] * rows,
            "gross_portfolio_return": [0.01] * rows,
            "turnover": [1.0] * rows,
            "cost": [0.0] * rows,
            "net_portfolio_return": [0.01] * rows,
            "long_count": [1] * rows,
            "short_count": [1] * rows,
        }
    )
    state = portfolio_runner._PortfolioSummaryState()
    try:
        portfolio_runner._append_portfolio_return_summary(state, returns)
        portfolio_runner._append_portfolio_return_summary(state, returns)
        portfolio_runner._portfolio_summary_outputs(state)
        assert phases == ["query"]
        assert calls[-1] >= (
            2
            * 2
            * rows
            * portfolio_runner._PORTFOLIO_SQLITE_ROW_DISK_BYTES
        )
        assert calls[-1] > calls[-2]
    finally:
        portfolio_runner._close_portfolio_summary_state(state)


def test_portfolio_return_summary_rejects_cumulative_disk_before_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    query_calls: list[str] = []

    def reject_aggregate_disk(_path, *, reserve_bytes: int) -> None:  # noqa: ANN001
        calls.append(reserve_bytes)
        if len(calls) == 3:
            raise OSError("portfolio_aggregate_disk_preflight")

    real_ordered_stats = portfolio_runner._ordered_performance_stats

    def observe_ordered_stats(summary_state):  # noqa: ANN001
        query_calls.append("ordered")
        return real_ordered_stats(summary_state)

    monkeypatch.setattr(
        portfolio_runner,
        "require_free_disk_headroom",
        reject_aggregate_disk,
    )
    monkeypatch.setattr(
        portfolio_runner,
        "_ordered_performance_stats",
        observe_ordered_stats,
    )
    returns = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "signal": ["factor"],
            "return_source_column": ["label"],
            "portfolio_mode": ["long_short"],
            "long_return": [0.01],
            "short_return": [0.0],
            "gross_portfolio_return": [0.01],
            "turnover": [1.0],
            "cost": [0.0],
            "net_portfolio_return": [0.01],
            "long_count": [1],
            "short_count": [1],
        }
    )
    state = portfolio_runner._PortfolioSummaryState()
    try:
        with pytest.raises(OSError, match="portfolio_aggregate_disk_preflight"):
            portfolio_runner._append_portfolio_return_summary(state, returns)
            portfolio_runner._portfolio_summary_outputs(state)
        assert len(calls) == 3
        assert query_calls == []
    finally:
        portfolio_runner._close_portfolio_summary_state(state)


def test_portfolio_summary_admits_before_frame_build_and_holds_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024 * 1024,
        explicit_cap_bytes=512 * 1024,
    )
    summary_state = portfolio_runner._PortfolioSummaryState(
        resource_policy=policy,
        position_stats={"signal": {}},
    )
    observed_used: list[int] = []

    def build_summary(_state) -> pd.DataFrame:  # noqa: ANN001
        observed_used.append(policy.used_memory_bytes)
        return pd.DataFrame({"value": [1]})

    for name in (
        "_portfolio_performance_summary",
        "_portfolio_exposure_summary",
        "_portfolio_concentration_summary",
        "_portfolio_period_breakdown",
    ):
        monkeypatch.setattr(portfolio_runner, name, build_summary)

    with runtime_resource_policy_scope(policy):
        outputs = portfolio_runner._portfolio_summary_outputs(summary_state)
        assert outputs
        assert observed_used
        assert observed_used[0] == policy.used_memory_bytes
        assert policy.used_memory_bytes > 0
        portfolio_runner._close_portfolio_summary_state(summary_state)
        assert policy.used_memory_bytes == 0


def test_portfolio_summary_admits_stats_batch_before_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returns = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "signal": ["factor"],
            "return_source_column": ["label"],
            "portfolio_mode": ["long_short"],
            "long_return": [0.01],
            "short_return": [0.0],
            "gross_portfolio_return": [0.01],
            "turnover": [1.0],
            "cost": [0.0],
            "net_portfolio_return": [0.01],
            "long_count": [1],
            "short_count": [1],
        }
    )
    summary_state = portfolio_runner._PortfolioSummaryState()
    portfolio_runner._append_portfolio_return_summary(summary_state, returns)
    bound = portfolio_runner._portfolio_summary_memory_bound(summary_state)
    assert bound > 128 * 1024
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024 * 1024,
        explicit_cap_bytes=128 * 1024,
    )
    summary_state.resource_policy = policy
    built = False

    def fail_if_built(_state) -> pd.DataFrame:  # noqa: ANN001
        nonlocal built
        built = True
        return pd.DataFrame()

    monkeypatch.setattr(
        portfolio_runner,
        "_portfolio_performance_summary",
        fail_if_built,
    )
    try:
        with runtime_resource_policy_scope(policy):
            with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
                portfolio_runner._portfolio_summary_outputs(summary_state)
        assert not built
        assert policy.used_memory_bytes == 0
    finally:
        if summary_state.return_database is not None:
            portfolio_runner._close_portfolio_summary_state(summary_state)


def test_portfolio_summary_insufficient_headroom_fails_before_build_and_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024 * 1024,
        explicit_cap_bytes=32 * 1024,
    )
    summary_state = portfolio_runner._PortfolioSummaryState(
        resource_policy=policy,
        position_stats={"signal": {}},
    )
    built = False

    def fail_if_built(_state) -> pd.DataFrame:  # noqa: ANN001
        nonlocal built
        built = True
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(
        portfolio_runner,
        "_portfolio_performance_summary",
        fail_if_built,
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            portfolio_runner._portfolio_summary_outputs(summary_state)
        assert not built
        assert policy.used_memory_bytes == 0

    reopened = capture_test_policy(available_memory_bytes=4 * 1024 * 1024)
    with runtime_resource_policy_scope(reopened):
        assert portfolio_runner.current_runtime_resource_policy() is reopened


def test_portfolio_state_admission_precedes_source_pull_and_reopens(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024 * 1024,
        explicit_cap_bytes=64 * 1024,
    )
    source_pulls: list[bool] = []

    def source_chunks(**_kwargs):  # noqa: ANN003
        source_pulls.append(True)
        return iter(())

    monkeypatch.setattr(portfolio_runner, "_portfolio_input_chunks", source_chunks)
    with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
        build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
            resource_policy=policy,
        )

    assert source_pulls == []
    assert policy.used_memory_bytes == 0
    assert portfolio_runner.current_runtime_resource_policy() is None
    reopened = capture_test_policy(available_memory_bytes=4 * 1024 * 1024)
    with runtime_resource_policy_scope(reopened):
        assert portfolio_runner.current_runtime_resource_policy() is reopened


def test_portfolio_output_chunks_keep_one_lease_across_state_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PortfolioSpec(
        portfolio_id="portfolio_demo",
        market_id=MarketId("a_share"),
        research_dataset_id="research_demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_column="signal",
                source_kind="factor",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=0,
    )
    chunks = iter(
        (
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-01")],
                    "asset_id": ["asset"],
                    "signal": [1.0],
                    "forward_return": [0.01],
                }
            ),
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2024-01-02")],
                    "asset_id": ["asset"],
                    "signal": [1.0],
                    "forward_return": [0.02],
                }
            ),
        )
    )
    policy = capture_test_policy(
        available_memory_bytes=4 * 1024 * 1024,
        explicit_cap_bytes=128 * 1024,
    )
    lease_starts: list[int] = []
    original_lease = portfolio_runner.RuntimeResourcePolicy.lease

    def observe_lease(current, owner, reserved_bytes):  # noqa: ANN001
        lease_starts.append(current.used_memory_bytes)
        return original_lease(current, owner, reserved_bytes)

    monkeypatch.setattr(
        portfolio_runner.RuntimeResourcePolicy,
        "lease",
        observe_lease,
    )
    with runtime_resource_policy_scope(policy):
        output_chunks = portfolio_runner._iter_portfolio_output_chunks(
            chunks,
            spec,
            source_columns_by_kind={"factor": ("signal",)},
            resource_policy=policy,
        )
        assert next(output_chunks)
        assert policy.used_memory_bytes > 0
        assert next(output_chunks)
        assert policy.used_memory_bytes > 0
        with pytest.raises(StopIteration):
            next(output_chunks)

    assert lease_starts == [0]
    assert policy.used_memory_bytes == 0


def test_managed_portfolio_accepts_true_peak_and_rejects_before_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-04"] * 3),
            "asset_id": ["A", "B", "D"],
            "raw": [1.0, 1.0, 1.0],
            "forward_return": [0.4, -0.2, 0.0],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="lease_peak",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="raw",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=1,
        weight_config=WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    lease_requests: list[int] = []
    original_lease = portfolio_runner.RuntimeResourcePolicy.lease

    def observe_lease(current, owner, reserved_bytes):  # noqa: ANN001
        lease_requests.append(reserved_bytes)
        return original_lease(current, owner, reserved_bytes)

    monkeypatch.setattr(
        portfolio_runner.RuntimeResourcePolicy,
        "lease",
        observe_lease,
    )
    policy = capture_test_policy(
        available_memory_bytes=1024**2,
        explicit_cap_bytes=1024**2,
    )
    with runtime_resource_policy_scope(policy):
        outputs = list(
            portfolio_runner._iter_portfolio_output_chunks(
                (frame,),
                spec,
                source_columns_by_kind={"factor": ("raw",)},
                resource_policy=policy,
            )
        )
    assert len(outputs) == 1
    assert policy.used_memory_bytes == 0
    assert lease_requests
    formal_bound = lease_requests[0]

    called: list[bool] = []

    def reject_kernel(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("portfolio kernel allocated before admission")

    monkeypatch.setattr(
        portfolio_runner,
        "_construct_portfolio_positions",
        reject_kernel,
    )
    insufficient = capture_test_policy(
        available_memory_bytes=1024**2,
        explicit_cap_bytes=formal_bound - 1,
    )
    with runtime_resource_policy_scope(insufficient):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            list(
                portfolio_runner._iter_portfolio_output_chunks(
                    (frame,),
                    spec,
                    source_columns_by_kind={"factor": ("raw",)},
                    resource_policy=insufficient,
                )
            )
    assert called == []
    assert insufficient.used_memory_bytes == 0


def _assert_portfolio_chunks_match_full(
    frame: pd.DataFrame,
    spec: PortfolioSpec,
    chunk_timestamp_counts: tuple[int, ...],
) -> None:
    full = _construct_portfolio_positions(
        frame,
        spec,
        source_columns_by_kind={"factor": ("raw",)},
    )
    timestamp_groups = [
        group.copy() for _, group in frame.groupby("timestamp", sort=True)
    ]
    chunks: list[pd.DataFrame] = []
    offset = 0
    for count in chunk_timestamp_counts:
        chunks.append(pd.concat(timestamp_groups[offset : offset + count]))
        offset += count
    assert offset == len(timestamp_groups)
    output_chunks = list(
        portfolio_runner._iter_portfolio_output_chunks(
            chunks,
            spec,
            source_columns_by_kind={"factor": ("raw",)},
            resource_policy=None,
        )
    )
    for output_name in (
        "positions",
        "turnover",
        "cost_breakdown",
        "portfolio_returns",
        "diagnostics",
        "capacity_diagnostics",
        "numeric_exposure_diagnostics",
        "categorical_exposure_diagnostics",
    ):
        actual = pd.concat(
            [chunk[output_name] for chunk in output_chunks],
            ignore_index=True,
        )
        expected = getattr(full, output_name).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("chunk_timestamp_counts", [(1, 1, 1, 1), (2, 2), (1, 3)])
def test_sparse_portfolio_temporal_state_matches_full_across_chunk_partitions(
    chunk_timestamp_counts: tuple[int, ...],
) -> None:
    frame = pd.DataFrame(
        {
            "asset_id": ["A", "B", "C", "D", "B", "D", "A", "B", "D"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                    "2026-01-04",
                ]
            ),
            "raw": [10.0, -10.0, -2.0, 2.0, -5.0, 5.0, 1.0, 1.0, 1.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.0],
        }
    )
    spec = PortfolioSpec(
        portfolio_id="sparse",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="raw",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=1,
        weight_config=WeightConfig(top_quantile=1 / 3, bottom_quantile=1 / 3),
    )

    _assert_portfolio_chunks_match_full(frame, spec, chunk_timestamp_counts)


@pytest.mark.parametrize("execution_lag_steps", [0, 1, 2])
def test_dense_portfolio_temporal_state_matches_full_for_lag_zero_one_n(
    execution_lag_steps: int,
) -> None:
    timestamps = pd.date_range("2026-02-01", periods=6, freq="D")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, 3),
            "asset_id": ["A", "B", "C"] * 6,
            "raw": [
                float((step + asset) % 5)
                for step in range(6)
                for asset in range(3)
            ],
            "forward_return": [0.01, -0.02, 0.03] * 6,
        }
    )
    spec = PortfolioSpec(
        portfolio_id=f"lag_{execution_lag_steps}",
        market_id=MarketId("a_share"),
        research_dataset_id="demo",
        signals=(
            SignalSpec(
                signal_id="signal",
                source_kind="factor",
                source_column="raw",
            ),
        ),
        return_source_column="forward_return",
        rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
        execution_lag_steps=execution_lag_steps,
    )

    _assert_portfolio_chunks_match_full(frame, spec, (2, 1, 3))


def test_portfolio_runner_joins_persisted_prediction_primary_key_with_explicit_gaps(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )

    expected_result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
    )
    expected_chunks = list(expected_result.output_chunks)
    expected_summary = expected_result.summary_factory()
    expected_result.close()

    original_to_dict = pd.DataFrame.to_dict

    def reject_return_row_graph(frame, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        orient = args[0] if args else kwargs.get("orient", "dict")
        if orient == "records" and "gross_portfolio_return" in frame.columns:
            raise AssertionError("portfolio returns converted to a Python row graph")
        return original_to_dict(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_dict", reject_return_row_graph)
    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
    )
    chunks = list(result.output_chunks)
    summary = result.summary_factory()
    returns = pd.concat(
        [chunk["portfolio_returns"] for chunk in chunks],
        ignore_index=True,
    )

    assert len(chunks) == len(expected_chunks)
    assert returns["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert returns["gross_portfolio_return"].tolist() == pytest.approx([0.0, 0.20])
    assert summary.keys() == expected_summary.keys()
    for name in summary:
        pd.testing.assert_frame_equal(summary[name], expected_summary[name])
    result.close()


def test_portfolio_prediction_join_preflights_index_allocation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**2,
        explicit_cap_bytes=1024**2,
    )

    with runtime_resource_policy_scope(policy):
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
            resource_policy=policy,
        )

        monkeypatch.setattr(
            portfolio_runner,
            "pandas_frame_payload_bytes",
            lambda _frame: policy.memory_budget_bytes,
        )

        def reject_prediction_index(*_args, **_kwargs):
            raise AssertionError("prediction index allocated before admission")

        monkeypatch.setattr(
            pd.MultiIndex,
            "from_frame",
            classmethod(reject_prediction_index),
        )
        with pytest.raises(
            ValueError,
            match="portfolio_prediction_join_working_set_exceeds_task_memory_budget",
        ):
            list(result.output_chunks)


def test_portfolio_result_explicit_close_owns_operation_root(tmp_path) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)
    result = build_portfolio_result_from_research_dataset(
        layout=layout,
        dataset_catalog=dataset_catalog,
        portfolio_catalog=portfolio_catalog,
        portfolio_id="demo_prediction_portfolio",
        time_window=TimeWindow("2024-01-01", "2024-01-02"),
        prediction_frame=prediction,
        prediction_metadata=manifest["metadata"],
        resource_policy=policy,
    )
    assert portfolio_runner.current_runtime_resource_policy() is policy
    result.close()
    result.close()
    assert portfolio_runner.current_runtime_resource_policy() is None
    with pytest.raises(RuntimeError, match="portfolio_result_closed"):
        list(result.output_chunks)
    with pytest.raises(RuntimeError, match="portfolio_result_closed"):
        result.summary_factory()


def test_portfolio_result_context_preserves_body_primary() -> None:
    primary = ValueError("body_primary")
    closed: list[bool] = []
    policy = capture_test_policy(
        available_memory_bytes=128 * 1024,
        explicit_cap_bytes=64 * 1024 + 768,
    )
    scope = portfolio_runner.runtime_resource_policy_scope(policy)
    scope.__enter__()

    def close() -> None:
        closed.append(True)
        scope.__exit__(None, None, None)
        raise MemoryError("cleanup_secondary")

    result = portfolio_runner.PortfolioResult(
        spec=object(),  # type: ignore[arg-type]
        output_chunks=(),
        summary_factory=dict,
        _close=close,
    )
    with pytest.raises(ValueError) as raised:
        with result:
            raise primary

    assert raised.value is primary
    assert raised.value.__notes__ == [
        "portfolio result cleanup also failed: MemoryError"
    ]
    assert closed == [True]
    assert portfolio_runner.current_runtime_resource_policy() is None


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("duplicate", "duplicate_persisted_prediction_keys"),
        ("missing", "missing_required_prediction_keys"),
        ("misaligned", "misaligned_persisted_prediction_keys"),
    ],
)
def test_portfolio_runner_rejects_invalid_persisted_prediction_coverage(
    tmp_path,
    corruption: str,
    error: str,
) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    if corruption == "duplicate":
        prediction = pd.concat([prediction, prediction.iloc[[0]]], ignore_index=True)
    elif corruption == "missing":
        prediction = prediction.iloc[1:].reset_index(drop=True)
    else:
        prediction = prediction.copy()
        prediction.loc[0, "asset_id"] = "misaligned"

    with pytest.raises(ValueError, match=error):
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
        )
        list(result.output_chunks)

    assert portfolio_runner.current_runtime_resource_policy() is None
    reopened = capture_test_policy(available_memory_bytes=64 * 1024 * 1024)
    with runtime_resource_policy_scope(reopened):
        assert portfolio_runner.current_runtime_resource_policy() is reopened


def test_portfolio_runner_rejects_parseable_numeric_prediction_text(tmp_path) -> None:
    layout, dataset_catalog, portfolio_catalog, prediction, manifest, _root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    prediction = prediction.assign(
        prediction_score=["0.0"] * len(prediction),
    )

    with pytest.raises(ValueError, match="persisted_prediction_values_invalid"):
        result = build_portfolio_result_from_research_dataset(
            layout=layout,
            dataset_catalog=dataset_catalog,
            portfolio_catalog=portfolio_catalog,
            portfolio_id="demo_prediction_portfolio",
            time_window=TimeWindow("2024-01-01", "2024-01-02"),
            prediction_frame=prediction,
            prediction_metadata=manifest["metadata"],
        )
        list(result.output_chunks)


def test_persisted_prediction_portfolio_uses_bounded_disk_lookup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, dataset_catalog, portfolio_catalog, _prediction, _manifest, root = (
        _prediction_portfolio_inputs(tmp_path)
    )
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    def reject_full_read(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("persisted prediction was materialized as one DataFrame")

    monkeypatch.setattr(pd, "read_parquet", reject_full_read)
    with runtime_resource_policy_scope(policy):
        with _read_prediction_output_bundle(
            root,
            expected_output_id="demo_prediction",
        ) as (manifest, prediction_lookup):
            result = build_portfolio_result_from_research_dataset(
                layout=layout,
                dataset_catalog=dataset_catalog,
                portfolio_catalog=portfolio_catalog,
                portfolio_id="demo_prediction_portfolio",
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
                prediction_frame=prediction_lookup,
                prediction_metadata=manifest["metadata"],
                resource_policy=policy,
            )
            chunks = list(result.output_chunks)
            summary = result.summary_factory()
            result.close()

    returns = pd.concat(
        [chunk["portfolio_returns"] for chunk in chunks],
        ignore_index=True,
    )
    assert returns["gross_portfolio_return"].tolist() == pytest.approx([0.0, 0.20])
    assert summary["performance_summary"].iloc[0]["periods"] == 2


def test_prediction_bundle_rejects_duplicate_outside_selected_window(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share"],
            "asset_id": ["a", "a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "model_input": [1.0, 2.0],
            "eligible": [True, True],
        }
    )
    spec = PredictionSpec(
        prediction_id="windowed_prediction",
        model_id="linear_regression_demo",
        research_dataset_id="research_demo",
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-03",
    )
    _prediction, root, _manifest = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        spec,
        np.array([0.0, 1.0]),
        ("model_input",),
    )
    path = root / "prediction.parquet"
    persisted = pd.read_parquet(path)
    excluded = persisted.loc[persisted["timestamp"] == pd.Timestamp("2024-01-02")].iloc[
        [0]
    ]
    pd.concat([persisted, excluded], ignore_index=True).to_parquet(path, index=False)
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024 * 1024,
        explicit_cap_bytes=4 * 1024 * 1024,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="duplicate_persisted_prediction_keys"):
            with _read_prediction_output_bundle(
                root,
                expected_output_id=spec.prediction_id,
                time_window=TimeWindow("2024-01-03", "2024-01-03"),
            ):
                pass


def _prediction_portfolio_inputs(tmp_path):
    layout = StorageLayout(project_root=tmp_path)
    dataset_spec = make_portfolio_research_dataset_spec(
        factor_columns=("model_input",),
        label_columns=("forward_return",),
        filter_columns=("eligible",),
    )
    dataset_catalog = make_research_dataset_catalog(dataset_spec)
    frame = pd.DataFrame(
        {
            "market_id": ["a_share"] * 8,
            "asset_id": ["a", "b", "c", "d"] * 2,
            "timestamp": [pd.Timestamp("2024-01-01")] * 4
            + [pd.Timestamp("2024-01-02")] * 4,
            "model_input": [1.0, 2.0, 3.0, 4.0, 4.0, 1.0, np.nan, 3.0],
            "forward_return": [0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.5, 0.2],
            "eligible": [True] * 7 + [False],
        }
    )
    write_dataset(
        layout.research_dataset_root(MarketId("a_share"), dataset_spec.data_id),
        frame,
        spec=dataset_spec,
    )
    prediction_spec = PredictionSpec(
        prediction_id="demo_prediction",
        model_id="linear_regression_demo",
        research_dataset_id=dataset_spec.data_id,
        train_end="2024-01-01",
        test_start="2024-01-02",
        test_end="2024-01-02",
    )
    prediction, root, manifest = write_prediction_output_and_read(
        tmp_path,
        (frame,),
        prediction_spec,
        np.array([0.0, 1.0]),
        ("model_input",),
    )
    portfolio_catalog = PortfolioCatalog(
        (
            PortfolioSpec(
            portfolio_id="demo_prediction_portfolio",
            market_id=MarketId("a_share"),
            research_dataset_id=dataset_spec.data_id,
            signals=(
                SignalSpec(
                    signal_id=prediction_spec.prediction_id,
                    source_column=prediction_spec.prediction_column,
                    source_kind="prediction",
                ),
            ),
            return_source_column="forward_return",
            rebalance_frequency=frequency_ids.DAILY_FREQUENCY_SLUG,
            execution_lag_steps=0,
            ),
        )
    )
    return layout, dataset_catalog, portfolio_catalog, prediction, manifest, root
