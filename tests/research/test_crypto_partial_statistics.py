"""Native performance-only statistics make observed-prefix scope explicit."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.crypto_hft import write_crypto_episode_artifact
from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    PortfolioEconomics,
    _record_execution_result,
)
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
def test_performance_only_observed_totals_have_unambiguous_scope(
    tmp_path, partial, mode
):
    dates = pd.date_range("2024-01-01", periods=3)
    capital = PortfolioCapital("scope", 1000.0, "USDT", "linear_perpetual")
    reason = "crypto_execution_book_coverage_incomplete"

    def produce(emit):
        account = PortfolioEconomics(capital, "btc_usdt", emit)
        account.mark(
            dates[0].value,
            100.0,
            diagnostics=(
                PortfolioDiagnostic("btc_usdt", None, "oos_window_start", number=0.0),
            ),
        )
        account.fill(
            timestamp_ns=dates[0].value,
            order_id="entry",
            quantity=1.0,
            price=100.0,
            mark_price=100.0,
            fee=1.0,
        )
        account.mark(
            dates[1].value,
            110.0,
            diagnostics=(
                PortfolioDiagnostic(
                    "btc_usdt",
                    None,
                    "utc_day_boundary",
                    number=float(dates[1].value // pd.Timedelta(days=1).value),
                ),
            ),
        )
        if not partial:
            account.mark(
                dates[2].value,
                120.0,
                diagnostics=(
                    PortfolioDiagnostic("btc_usdt", None, "oos_window_end", number=0.0),
                ),
            )
        _record_execution_result(
            account,
            {
                "simulation": "statistics_scope_fixture",
                "net_pnl": None if partial else 19.0,
                "created_orders": 1,
                "filled_orders": 1,
                "selection_status": "PARTIAL" if partial else "COMPLETE",
                "account_path_status": "PARTIAL" if partial else "COMPLETE",
                "account_stop_reason": reason if partial else None,
                "account_stop_ns": dates[1].value + 1 if partial else None,
            },
            dates[-1].value,
            120.0,
        )

    def produce_episodes(emit_episode):
        emit_episode(
            start_ns=dates[0].value,
            stop_ns=dates[-1].value + 1,
            termination="PLANNED_END",
            stop_reasons=["planned_oos_end"],
            produce=produce,
        )

    writer = (
        write_portfolio_ledger_artifact
        if mode == "continuous"
        else write_crypto_episode_artifact
    )
    source = writer(
        tmp_path / "ledger",
        capital=capital,
        metadata={
            "sample_role": "exploratory",
            "window": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
            "execution_mode": mode,
            "execution_style": "taker",
            "execution": {"position_size": 1.0, "taker_fee_bps": 0.0},
        },
        produce=produce if mode == "continuous" else produce_episodes,
    )
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(source),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    assert "crypto_execution_diagnostics.parquet" not in artifact._members
    observed = {
        "fees": 1.0,
        "turnover": 100.0,
        "fill_events": 1.0,
        "fill_quantity": 1.0,
    }
    account_id = (
        "continuous"
        if mode == "continuous"
        else f"{dates[0].value}_{dates[-1].value + 1}"
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as metrics:
        for episode_id in (account_id, "all"):
            rows = metrics.loc[metrics.episode_id.eq(episode_id)].set_index("metric_id")
            for name, expected in observed.items():
                metric_id = f"known_prefix_{name}" if partial else name
                assert rows.loc[metric_id, "value"] == pytest.approx(
                    expected, rel=2e-15, abs=0
                )
                assert rows.loc[metric_id, "status"] == "DEFINED"
                assert rows.loc[metric_id, "reason"] == ""
                assert (name if partial else f"known_prefix_{name}") not in rows.index
        rows = metrics.loc[metrics.episode_id.eq(account_id)].set_index("metric_id")
        for name in (
            "net_pnl",
            "terminal_nav",
            "continuous_max_drawdown"
            if mode == "continuous"
            else "episode_max_drawdown",
            "pnl_per_fill",
            "pnl_per_notional",
        ):
            assert rows.loc[name, "status"] == ("UNDEFINED" if partial else "DEFINED")
            if partial:
                assert np.isnan(rows.loc[name, "value"])
                assert rows.loc[name, "reason"] == reason
        assert rows.loc["planned_complete_utc_days", "value"] == 2
        assert rows.loc["defined_complete_utc_days", "value"] == (1 if partial else 2)
        if partial:
            assert rows.loc["oos_mean_daily_return", "status"] == "UNDEFINED"
            assert (
                rows.loc["oos_mean_daily_return", "reason"]
                == "planned_utc_day_valuation_undefined"
            )
            for name in ("mean", "variance", "max_abs"):
                assert rows.loc[f"oos_inventory_{name}", "status"] == "UNDEFINED"
                assert rows.loc[f"oos_inventory_{name}", "reason"] == reason
            if mode == "continuous":
                assert rows.loc["oos_net_pnl", "status"] == "UNDEFINED"
                assert rows.loc["oos_net_pnl", "reason"] == reason
    with artifact.table("crypto_portfolio_intervals.parquet") as intervals:
        assert len(intervals) == (0 if partial else 1)
    with artifact.table("crypto_portfolio_daily.parquet") as daily:
        assert daily.status.tolist() == [
            "DEFINED",
            "UNDEFINED" if partial else "DEFINED",
        ]
        assert daily.net_return.iloc[0] == pytest.approx(0.009, rel=2e-14, abs=0)
    with artifact.table("crypto_portfolio_coverage.parquet") as coverage:
        assert coverage.kind.tolist() == (
            ["covered", "missing"] if partial else ["covered"]
        )
        if partial:
            assert coverage.stop_at.iloc[0] == dates[1] + pd.Timedelta(1, "ns")
            assert coverage.reason.iloc[1] == reason
    costs_artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(source),
                "output": str(tmp_path / "costs"),
                "analyses": ["cost_sensitivity"],
                "statistics": {
                    "fee_scenarios_bps": [0.0, 1.0],
                    "additional_spread_scenarios_bps": [0.0],
                },
            }
        )
    )
    with costs_artifact.table("crypto_cost_sensitivity.parquet") as costs:
        assert costs.cost_delta.to_numpy() == pytest.approx(
            [0.0, 0.01], rel=2e-15, abs=0
        )
        if partial:
            assert costs.net_pnl.isna().all()
            assert costs.method.eq(
                "constant_known_prefix_turnover_additive_sensitivity"
            ).all()
        else:
            assert costs.net_pnl.to_numpy() == pytest.approx(
                [19.0, 18.99], rel=2e-15, abs=0
            )
            assert costs.method.eq(
                "constant_actual_turnover_additive_sensitivity"
            ).all()


def test_mixed_episode_aggregate_counts_known_observations_once(tmp_path):
    first = pd.Timestamp("2024-01-01")
    hour = pd.Timedelta(hours=1).value
    capital = PortfolioCapital("mixed", 1000.0, "USDT", "linear_perpetual")

    def produce(emit_episode):
        for offset, partial in ((0, False), (24 * hour, True)):
            start = first.value + offset
            quantity = 2.0 if partial else 1.0

            def ledger(emit, start=start, quantity=quantity, partial=partial):
                account = PortfolioEconomics(capital, "btc_usdt", emit)
                account.mark(start, 100.0)
                account.fill(
                    timestamp_ns=start,
                    order_id="entry",
                    quantity=quantity,
                    price=100.0,
                    mark_price=100.0,
                    fee=quantity,
                )
                account.mark(start + hour // 2, 100.0)
                _record_execution_result(
                    account,
                    {
                        "simulation": "mixed_scope_fixture",
                        "net_pnl": None if partial else -quantity,
                        "created_orders": 1,
                        "filled_orders": 1,
                        "account_path_status": "PARTIAL" if partial else "COMPLETE",
                        "account_stop_reason": (
                            "crypto_execution_funding_coverage_incomplete"
                            if partial
                            else None
                        ),
                        "account_stop_ns": start + hour // 2 + 1 if partial else None,
                    },
                    start + hour // 2,
                    100.0,
                )

            emit_episode(
                start_ns=start,
                stop_ns=start + hour,
                termination="TRUNCATED",
                stop_reasons=["book_coverage_end"],
                produce=ledger,
            )

    root = write_crypto_episode_artifact(
        tmp_path / "episodes",
        capital=capital,
        metadata={
            "sample_role": "exploratory",
            "window": {
                "start": first.isoformat(),
                "end": (first + pd.Timedelta(days=2)).isoformat(),
            },
            "execution_mode": "independent_segments",
            "execution_style": "taker",
            "execution": {"position_size": 2.0, "taker_fee_bps": 0.0},
        },
        produce=produce,
    )
    artifact = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "analyses": ["performance"],
            }
        )
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as frame:
        all_rows = frame.loc[frame.episode_id.eq("all")].set_index("metric_id")
        for name, expected in {
            "fees": 3.0,
            "turnover": 300.0,
            "fill_events": 2.0,
            "fill_quantity": 3.0,
        }.items():
            assert name not in all_rows.index
            result = all_rows.loc[f"known_prefix_{name}"]
            assert result.status == "DEFINED" and result.reason == ""
            assert result.value == pytest.approx(expected, rel=2e-15, abs=0)
        individual = frame.loc[~frame.episode_id.eq("all")]
        assert individual.loc[individual.metric_id.eq("fees"), "value"].tolist() == [
            1.0
        ]
        assert individual.loc[
            individual.metric_id.eq("known_prefix_fees"), "value"
        ].tolist() == [2.0]
    with artifact.table("crypto_portfolio_intervals.parquet") as intervals:
        assert len(intervals) == 1
        assert intervals.start_at.iloc[0] == first
        assert intervals[["end_nav", "gross_pnl", "net_pnl"]].notna().all().all()
