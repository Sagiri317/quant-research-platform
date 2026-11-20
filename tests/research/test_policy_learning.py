"""Learned decisions share real account, causality and persistence rules."""

import pickle
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.data import DataSpec, ProcessedDataSpec
from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioCandidateSpec,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.interfaces.queries import inspect
from quant_research_platform.research.policies.cost_aware import (
    backtest_cost_model,
    differentiable_returns,
    fit_cost_model,
)
from quant_research_platform.research.policies.environments import TradingEnv
from quant_research_platform.research.policies.reinforcement import (
    backtest_policy,
    train_policy,
)
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.portfolios.high_frequency import inventory_batches
from quant_research_platform.storage.datasets import read_dataset, write_dataset
from tests.helpers.benchmark_references import reference_input
from tests.helpers.high_frequency import (
    daily_view,
    parameters,
    quotes,
    schedule,
    write_daily_inputs,
)
from tests.helpers.portfolio_inputs import _interim_specs, closed_market_inputs
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.research.test_optional_statistics import _cash_ledger


def hft_episode(root, *, day="2024-01-02", task="execution", missing_features=False):
    panel = pd.concat(
        [quotes(date, assets=("a",)) for date in ("2024-01-02", "2024-01-03")],
        ignore_index=True,
    )
    daily = write_daily_inputs(root / "daily", panel)
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    if missing_features:
        panel["signal"] = [2.0, np.nan, 2.0, np.nan, 10000.0, 10000.0]
        roles["signal"] = "factor"
    spec = ProcessedDataSpec(
        data_id="hft_panel",
        parameters_json='{"max_gap_seconds": 6}',
        market_id="a_share",
        kind="panel",
        frequency="tick",
        grain="asset_timestamp",
        column_roles=roles,
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
    )
    write_dataset(root / "quotes", panel, spec=spec)
    fees = pd.DataFrame([asdict(rule) for rule in schedule().rules])
    fees["effective_date"] = pd.to_datetime(fees.effective_date)
    fees["end_date"] = pd.to_datetime(fees.end_date)
    fees["included_in"] = fees.included_in.astype("string")
    write_dataset(
        root / "fees", fees, spec=_interim_specs().get("a_share_fee_schedule")
    )
    params = parameters(("a",))
    del params["buy_threshold"], params["sell_threshold"]
    episode = {
        "kind": "hft",
        "task": task,
        "asset_ids": ["a"],
        "feature_columns": ["signal"] if missing_features else ["mid", "ask_vol1"],
        "window": {"start": day + "T01:30:00", "end": day + "T01:30:06"},
        "account": {
            "panel": str(root / "quotes"),
            "fee_schedule": str(root / "fees"),
            "daily_inputs": daily,
            "parameters": params,
        },
        "risk_aversion": 0.0,
        "unfinished_penalty_bps": 100.0,
    }
    if task == "execution":
        episode["order"] = {"asset_id": "a", "quantity": -100}
    else:
        episode["unfinished_penalty_bps"] = 0
    return episode


def test_interactive_quote_account_observes_then_executes_and_keeps_t1():
    panel = quotes("2024-01-02", assets=("a",))
    stream = inventory_batches(
        [(daily_view(panel), panel, pd.DataFrame())],
        schedule(),
        parameters(("a",)),
        TimeWindow("2024-01-02", "2024-01-02"),
        decision_columns=("mid",),
    )
    events = []
    event = next(stream)
    while not isinstance(event, PolicyDecision):
        events.append(event)
        event = next(stream)
    assert event.timestamp == panel.timestamp.iloc[0]
    assert not any(batch.trades for batch in events)
    event = stream.send(pd.Series({"a": 200.0}))
    bought = []
    while not isinstance(event, PolicyDecision):
        bought.extend(event.trades)
        event = next(stream)
    assert [trade.quantity for trade in bought] == [100]
    assert bought[0].timestamp_ns > int(bought[0].order_id)
    assert event.quantities["a"] == 200
    assert event.sellable["a"] == 100
    event = stream.send(pd.Series({"a": 0.0}))
    remaining = []
    try:
        while True:
            if isinstance(event, PolicyDecision):
                event = stream.send(event.quantities)
            else:
                remaining.append(event)
                event = next(stream)
    except StopIteration:
        pass
    assert sum(trade.quantity for batch in remaining for trade in batch.trades) == -100
    assert [
        row.quantity
        for row in remaining[-1].valuations
        if row.phase == "end" and row.asset_id == "a"
    ] == [100]


def test_execution_env_reward_equals_hand_shortfall_and_native_ledger(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    env = TradingEnv([episode])
    try:
        observation, _ = env.reset(seed=1)
        assert env.observation_space.contains(observation)
        total_reward = 0
        done = False
        while not done:
            _, reward, done, truncated, info = env.step(np.array([1.0]))
            total_reward += reward
            assert not truncated
        assert info["completion_rate"] == 1
        assert info["implementation_shortfall_bps"] == pytest.approx(10)
        assert total_reward == pytest.approx(-0.001)
        assert info["economic_reward"] == pytest.approx(total_reward)
        assert info["reward_penalty"] == 0
    finally:
        env.close()
    output = backtest_policy(
        {
            "episode": episode,
            "strategy": "immediate",
            "output": str(tmp_path / "comparison"),
        }
    )
    ledger = read_portfolio_ledger_artifact(output / "ledger")
    summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
    assert summary["net_pnl"] == pytest.approx(-1)
    assert summary["nav"] == pytest.approx(4999)
    assert inspect(output)["nav"] == pytest.approx(summary["nav"])
    artifact = read_statistics_artifact(
        portfolio_statistics(
            {
                "ledgers": [str(output / "ledger")],
                "window": episode["window"],
                "output": str(tmp_path / "statistics"),
            }
        )
    )
    assert artifact.metadata["observation_clock"] == "trading_day_close"
    assert artifact.metadata["candidate_ids"] == ["learned_policy"]
    with artifact.table("portfolio_performance.parquet") as performance:
        values = performance.set_index("metric_id").value
        assert values["net_pnl"] == pytest.approx(-1)
        assert values["terminal_nav"] == pytest.approx(4999)


def test_mother_order_penalty_stops_at_actual_partial_and_final_fills(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    episode["order"]["quantity"] = -200
    episode["account"]["parameters"].update(
        initial_inventory={"a": 200}, participation_rate=0.2
    )
    episode.update(risk_aversion=1.0, unfinished_penalty_bps=0.0)
    episode["window"]["end"] = "2024-01-02T01:30:09"
    env = TradingEnv([episode])
    try:
        env.reset(seed=0)
        done = False
        while not done:
            _, _, done, _, info = env.step(np.array([1.0]))
        assert info["filled_quantity"] == 200
        # Sell 100 shares each at +3s and +6s. The final 3 seconds carry no risk.
        assert info["reward_penalty"] == pytest.approx((3 + 3 * 0.5**2) / 9)
    finally:
        env.close()


def test_execution_strategies_keep_distinct_native_candidates(tmp_path):
    episode = hft_episode(tmp_path / "inputs")
    ledgers = []
    for strategy in ("immediate", "twap"):
        branch = deepcopy(episode)
        branch["account"]["candidate_id"] = strategy
        output = backtest_policy(
            {
                "episode": branch,
                "strategy": strategy,
                "output": str(tmp_path / strategy),
            }
        )
        ledgers.append(str(output / "ledger"))
        assert (
            read_portfolio_ledger_artifact(output / "ledger").capital.candidate_id
            == strategy
        )
    config = {
        "ledgers": ledgers,
        "window": episode["window"],
        "output": str(tmp_path / "comparison"),
        "analyses": ["performance"],
    }
    artifact = read_statistics_artifact(portfolio_statistics(config))
    assert artifact.metadata["candidate_ids"] == ["immediate", "twap"]
    with artifact.table("portfolio_performance.parquet") as performance:
        values = (
            performance.loc[performance.metric_id.eq("net_pnl")]
            .set_index("candidate_id")
            .value
        )
        # Immediate sells 100 shares at the 0.01 CNY half-spread. TWAP has no
        # executable whole-lot order before this six-second window ends.
        assert values.to_dict() == pytest.approx({"immediate": -1.0, "twap": 0.0})
    with pytest.raises(ValueError, match="portfolio_statistics_candidate_set_invalid"):
        portfolio_statistics(
            {
                **config,
                "ledgers": [ledgers[0], ledgers[0]],
                "output": str(tmp_path / "duplicate"),
            }
        )
    assert not (tmp_path / "duplicate").exists()


def test_policy_statistics_reject_mixed_account_clocks(tmp_path):
    episode = hft_episode(tmp_path / "hft-inputs")
    daily = tmp_path / "daily"
    window = _cash_ledger(daily)
    episode["window"] = window
    output = backtest_policy(
        {
            "episode": episode,
            "strategy": "immediate",
            "output": str(tmp_path / "hft"),
        }
    )
    with pytest.raises(ValueError, match="portfolio_statistics_account_clocks_mixed"):
        portfolio_statistics(
            {
                "ledgers": [str(daily), str(output / "ledger")],
                "window": window,
                "output": str(tmp_path / "mixed"),
            }
        )
    assert not (tmp_path / "mixed").exists()


@pytest.mark.parametrize("n_envs", [1, 2])
def test_real_ppo_roundtrip_and_later_native_backtest(tmp_path, n_envs):
    episode = hft_episode(tmp_path / "inputs", task="inventory", missing_features=True)
    model = train_policy(
        {
            "episodes": [episode],
            "output": str(tmp_path / "ppo"),
            "fit_cutoff": "2024-01-02T07:00:00",
            "timesteps": 8,
            "seed": 1,
            "device": "cpu",
            "n_envs": n_envs,
            "threads": 1,
            "ppo": {
                "n_steps": 4,
                "batch_size": 4,
                "n_epochs": 1,
                "policy_kwargs": {"net_arch": [8]},
            },
        }
    )
    facts = inspect(model)
    assert facts["actual_timesteps"] == 8
    progress = pd.read_csv(model.with_name(model.name + ".trial") / "progress.csv")
    assert progress["train/n_updates"].max() == facts["actual_updates"]
    assert progress["train/approx_kl"].notna().any()
    assert progress["train/value_loss"].notna().any()
    assert progress["rollout/ep_rew_mean"].notna().any()
    for index in range(n_envs):
        episodes = pd.read_csv(
            model.with_name(model.name + ".trial")
            / f"training-env-{index}.monitor.csv",
            comment="#",
        )
        assert len(episodes)
        assert set(episodes.episode_index) == {0}
        np.testing.assert_allclose(
            episodes.economic_reward - episodes.reward_penalty, episodes.r, atol=5e-7
        )
    frozen = (model / "feature_scaler.pkl").read_bytes()
    scaler = pickle.loads(frozen)
    assert scaler.medians == (2.0,)
    assert scaler.columns == ("signal",)
    later = {
        **episode,
        "window": {"start": "2024-01-03T01:30:00", "end": "2024-01-03T01:30:06"},
    }
    output = backtest_policy(
        {
            "episode": later,
            "output": str(tmp_path / "later"),
            "strategy": "ppo",
            "policy": str(model),
            "device": "cpu",
        }
    )
    assert list(read_portfolio_ledger_artifact(output / "ledger").iter_chunks())
    diagnostic = backtest_policy(
        {
            "episode": episode,
            "strategy": "ppo",
            "policy": str(model),
            "sample_role": "diagnostic",
            "checkpoint": "final",
            "output": str(tmp_path / "training-diagnostic"),
        }
    )
    diagnostic_ledger = read_portfolio_ledger_artifact(diagnostic / "ledger")
    assert diagnostic_ledger.metadata["sample_role"] == "diagnostic"
    assert diagnostic_ledger.metadata["policy"]["checkpoint"] == "final"
    assert list(diagnostic_ledger.iter_chunks())
    env = TradingEnv([later], feature_scaler=scaler)
    try:
        observation, _ = env.reset(seed=1)
        assert pd.isna(env.decision.features.signal.iloc[0])
        assert observation[0] == 2.0
    finally:
        env.close()
    with pytest.raises(ValueError, match="not_out_of_sample"):
        backtest_policy(
            {
                "episode": episode,
                "output": str(tmp_path / "invalid"),
                "strategy": "ppo",
                "policy": str(model),
            }
        )


@pytest.mark.parametrize("fail_after_learning", [False, True])
def test_ppo_execution_keeps_completed_validation_economics_on_failure(
    tmp_path, monkeypatch, fail_after_learning
):
    from stable_baselines3 import PPO

    from quant_research_platform.storage.experiments import read_trial_ledger

    episode = hft_episode(tmp_path / "inputs")
    episode["unfinished_penalty_bps"] = 0.0
    later = {
        **episode,
        "window": {"start": "2024-01-03T01:30:00", "end": "2024-01-03T01:30:06"},
    }
    output = tmp_path / "execution-policy"
    config = {
        "episodes": [episode],
        "validation_episodes": [later],
        "early_stopping": {"eval_freq": 4, "patience": 1, "min_evals": 1},
        "fit_cutoff": "2024-01-03T07:00:00",
        "output": str(output),
        "timesteps": 4,
        "seed": 1,
        "device": "cpu",
        "n_envs": 1,
        "threads": 1,
        "ppo": {
            "n_steps": 4,
            "batch_size": 4,
            "n_epochs": 1,
            "policy_kwargs": {"net_arch": [8]},
        },
    }
    if fail_after_learning:
        learn = PPO.learn

        def fail_after_validation(self, *args, **kwargs):
            learn(self, *args, **kwargs)
            raise RuntimeError("failure after completed validation")

        monkeypatch.setattr(PPO, "learn", fail_after_validation)
        with pytest.raises(RuntimeError, match="failure after completed validation"):
            train_policy(config)
        assert not output.exists()
    else:
        train_policy(config)
        facts = parse_qrp_json_object((output / "policy.json").read_bytes())
        for point in facts["validation"]["evaluations"]:
            metrics = point["episode_metrics"][0]
            assert metrics["unfilled_quantity"] == pytest.approx(
                100 * (1 - metrics["completion_rate"])
            )
            assert metrics["opportunity_cost_bps"] == 0
    trial = output.with_name(output.name + ".trial")
    assert read_trial_ledger(trial)[-1]["status"] == (
        "failed" if fail_after_learning else "completed"
    )
    progress = pd.read_csv(trial / "progress.csv")
    validation = progress.loc[progress["eval/mean_reward"].notna()]
    assert len(validation)
    assert validation["eval/n_updates"].notna().all()
    prefix = "eval/episode_0/"
    np.testing.assert_allclose(
        validation[prefix + "filled_quantity"]
        + validation[prefix + "unfilled_quantity"],
        100,
    )
    np.testing.assert_allclose(
        validation[prefix + "economic_reward"],
        -validation[prefix + "implementation_shortfall_bps"] / 10000,
    )
    np.testing.assert_allclose(
        validation[prefix + "total_implementation_shortfall_bps"],
        validation[prefix + "implementation_shortfall_bps"]
        + validation[prefix + "opportunity_cost_bps"],
    )


def test_cost_objective_uses_drifted_weights_and_two_sided_cost():
    class Equal(torch.nn.Module):
        def forward(self, features, previous):
            return torch.tensor([0.5, 0.5], dtype=features.dtype)

    returns, turnover = differentiable_returns(
        Equal(), torch.zeros(2, 2, 1), torch.tensor([[0.1, -0.1], [0.0, 0.0]]), 0.01
    )
    # First all-cash entry trades 1; the 55/45 drift then requires 0.1 turnover.
    assert turnover.tolist() == pytest.approx([1, 0.1])
    assert returns.tolist() == pytest.approx([-0.01, -0.001], abs=1e-7)


def test_delayed_mother_order_cannot_reverse_or_overfill_after_partial_fills():
    panel = quotes(
        "2024-01-02", assets=("a",), clocks=tuple(f"01:30:0{i}" for i in range(5))
    )
    panel["bid_vol1"] = 100.0
    params = parameters(("a",))
    params.update(initial_inventory={"a": 300}, latency_ms=2000)
    stream = inventory_batches(
        [(daily_view(panel), panel, pd.DataFrame())],
        schedule(),
        params,
        TimeWindow("2024-01-02", "2024-01-02"),
        decision_columns=("mid",),
        execution_order={"asset_id": "a", "quantity": -200},
    )
    targets = iter((100.0, 300.0, 100.0))
    trades = []
    try:
        event = next(stream)
        while True:
            if isinstance(event, PolicyDecision):
                event = stream.send(
                    pd.Series({"a": next(targets, event.quantities["a"])})
                )
            else:
                trades.extend(event.trades)
                event = next(stream)
    except StopIteration:
        pass
    finally:
        stream.close()
    assert [trade.quantity for trade in trades] == [-100, -100]
    assert all(
        trade.timestamp_ns - int(trade.order_id) >= 2_000_000_000 for trade in trades
    )


def test_daily_policy_and_cost_model_use_native_account_roundtrip(
    tmp_path, monkeypatch
):
    inputs, _, _, _ = closed_market_inputs(
        tmp_path / "daily-inputs", fault="normal_open", open_assets=0
    )
    inputs.pop("risk_context")
    assets = [f"a{i}" for i in range(6)]
    candidate = PortfolioCandidateSpec(
        "equal_weight",
        "global",
        "quantile_equal_weight",
        "champion",
        "daily",
        0.5,
        None,
    )
    assumptions = ExecutionAssumptions(100_000.0, 0.0, 0.0, 1.0, (1.0,), 0.0)
    episode = {
        "kind": "daily",
        "task": "portfolio",
        "asset_ids": assets,
        "feature_columns": ["signal"],
        "window": {"start": "2022-12-30", "end": "2023-01-06"},
        "account": {
            "inputs": {key: str(value) for key, value in inputs.items()},
            "candidate": asdict(candidate),
            "assumptions": asdict(assumptions),
            "input_window": {"start": "2022-12-01", "end": "2023-01-06"},
            "formal_dataset_window": {"start": "2022-12-30", "end": "2023-01-06"},
        },
        "risk_aversion": 0.0,
        "unfinished_penalty_bps": 0.0,
    }
    output = backtest_policy(
        {
            "episode": episode,
            "strategy": "equal_weight",
            "output": str(tmp_path / "equal"),
        }
    )
    ledger = read_portfolio_ledger_artifact(output / "ledger")
    assert any(batch.trades for batch in ledger.iter_chunks())
    assert ledger.capital.candidate_id == candidate.candidate_id
    policy_ledger = output / "ledger"
    daily_model = train_policy(
        {
            "episodes": [
                {**episode, "window": {"start": "2022-12-30", "end": "2023-01-03"}}
            ],
            "validation_episodes": [
                {**episode, "window": {"start": "2023-01-04", "end": "2023-01-06"}}
            ],
            "early_stopping": {"eval_freq": 4, "patience": 1, "min_evals": 1},
            "output": str(tmp_path / "daily-ppo"),
            "fit_cutoff": "2023-01-06T23:59:59.999999999",
            "timesteps": 4,
            "seed": 1,
            "device": "cpu",
            "n_envs": 1,
            "threads": 1,
            "ppo": {"n_steps": 4, "batch_size": 4, "n_epochs": 1},
        }
    )
    assert (daily_model / "policy.zip").is_file()
    stamps = pd.bdate_range("2022-12-20", periods=5)
    frame = pd.DataFrame(
        {
            "timestamp": stamps[:-1].repeat(6),
            "asset_id": assets * 4,
            "eligible": True,
            "signal": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 4,
            "label": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006] * 4,
            "label_available_at": stamps[1:].repeat(6),
            "return_end_at": stamps[1:].repeat(6),
        }
    )
    spec = DataSpec(
        data_id="cost_learning",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "eligible": "mask",
            "signal": "factor",
            "label": "label",
            "label_available_at": "context",
            "return_end_at": "context",
        },
        column_units={"label": "decimal_return"},
        label_availability_columns={"label": "label_available_at"},
    )
    frame.loc[frame.signal.eq(2.0), "signal"] = np.nan
    write_dataset(tmp_path / "training", frame, spec=spec)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    model = fit_cost_model(
        {
            "dataset": str(tmp_path / "training"),
            "feature_columns": ["signal"],
            "asset_ids": assets,
            "label": "label",
            "return_end_column": "return_end_at",
            "window": {"start": "2022-12-20", "end": "2022-12-23"},
            "cutoff": "2022-12-28",
            "cost_bps": 10.0,
            "risk_aversion": 1.0,
            "epochs": 2,
            "learning_rate": 0.01,
            "hidden_sizes": [4],
            "device": "cuda",
            "threads": 1,
            "seed": 1,
            "output": str(tmp_path / "cost-model"),
        }
    )
    facts = parse_qrp_json_object((model / "model.json").read_bytes())
    assert facts["device"] == "cpu"
    assert [point["epoch"] for point in facts["learning_curve"]] == [1, 2]
    assert all(np.isfinite(point["objective"]) for point in facts["learning_curve"])
    frozen = (model / "feature_scaler.pkl").read_bytes()
    assert pickle.loads(frozen).medians == (4.0,)
    cost_episode = deepcopy(episode)
    cost_episode["account"]["candidate"]["candidate_id"] = "cost_allocator"
    output = backtest_cost_model(
        {
            "model": str(model),
            "episode": cost_episode,
            "output": str(tmp_path / "cost-ledger"),
            "device": "cuda",
        }
    )
    ledger = read_portfolio_ledger_artifact(output)
    assert (
        ledger.metadata["episode"]["account"]["inputs"]
        == cost_episode["account"]["inputs"]
    )
    assert (
        ledger.metadata["episode"]["account"]["candidate"]["candidate_id"]
        == "cost_allocator"
    )
    assert any(batch.trades for batch in ledger.iter_chunks())
    assert np.isfinite(
        summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())["nav"]
    )
    roots = [policy_ledger, output]
    expected = {}
    for root in roots:
        native = read_portfolio_ledger_artifact(root)
        batches = list(native.iter_chunks())
        terminal = sum(
            row.value for row in batches[-1].valuations if row.phase == "end"
        )
        expected[native.capital.candidate_id] = {
            "terminal_nav": terminal,
            "net_pnl": terminal - native.capital.initial_capital,
            "cost": -sum(
                flow.amount
                for batch in batches
                for flow in batch.cashflows
                if flow.kind.endswith("_cost")
            ),
        }
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp(
                next(
                    row.timestamp_ns
                    for row in batches[0].valuations
                    if row.phase == "before"
                )
            )
        ]
        + [
            pd.Timestamp(
                next(row.timestamp_ns for row in batch.valuations if row.phase == "end")
            )
            for batch in batches
        ]
    )
    levels = pd.Series(100 + np.arange(len(dates)) ** 2, index=dates)
    reference = reference_input(
        tmp_path / "benchmark.parquet", levels.to_numpy(), dates=dates
    )
    config = {
        "ledgers": [str(root) for root in roots],
        "window": episode["window"],
        "inputs": {"calendar": str(inputs["calendar"])},
        "output": str(tmp_path / "statistics"),
        "analyses": [
            "performance",
            "capacity",
            "sharpe_inference",
            "cscv",
            "benchmark_comparison",
            "control_comparison",
            "market_exposure",
            "regime_comparison",
        ],
        "adv_sensitivities": [0.1],
        "statistics": {"cscv_blocks": 4},
        "comparison_benchmarks": [reference],
        "comparison_controls": [
            {
                **reference,
                "reference_id": "control",
                "return_basis": "strategy_net",
                "cost_basis": "execution_costs",
            }
        ],
        "market_exposure": {
            "hac_lag": 0,
            "confidence": 0.95,
            "minimum_observations": 3,
        },
        "regime_comparison": {"threshold": 0.0},
    }
    artifact = read_statistics_artifact(portfolio_statistics(config))
    ids = {"equal_weight", "cost_allocator"}
    assert set(artifact.metadata["candidate_ids"]) == ids
    with artifact.table("portfolio_performance.parquet") as performance:
        assert set(performance.candidate_id) == ids
        for candidate_id, values in expected.items():
            actual = performance.loc[
                performance.candidate_id.eq(candidate_id)
            ].set_index("metric_id")
            for metric, value in values.items():
                assert actual.loc[metric, "value"] == pytest.approx(value, abs=1e-9)
    for name in (
        "portfolio_daily",
        "portfolio_capacity_summary",
        "portfolio_benchmark_intervals",
        "portfolio_market_exposure",
        "portfolio_regime_comparison",
    ):
        with artifact.table(f"{name}.parquet") as rows:
            assert set(rows.candidate_id) == ids
            if name == "portfolio_benchmark_intervals":
                expected_returns = (
                    rows.end_at.map(levels) / rows.start_at.map(levels) - 1
                )
                assert rows.status.eq("DEFINED").all()
                assert rows.benchmark_return.tolist() == pytest.approx(expected_returns)
    with artifact.table("portfolio_robustness.parquet") as rows:
        assert set(rows.candidate_id) == ids | {"joint"}
    rejected = tmp_path / "attribution"
    with pytest.raises(
        ValueError, match="portfolio_statistics_candidate_metadata_required"
    ):
        portfolio_statistics(
            {
                **config,
                "risk_model": {},
                "benchmark_total_return_indices": {},
                "analyses": ["attribution"],
                "output": str(rejected),
            }
        )
    assert not rejected.exists()
    data = read_dataset(inputs["dataset"]).rename(columns={"signal": "feature_signal"})
    data.loc[data.asset_id.eq("a0"), "feature_signal"] = np.nan
    observed_spec = make_demo_research_dataset_spec(
        data_id="observed",
        factor_columns=(),
        feature_columns=("feature_signal",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )
    write_dataset(tmp_path / "observed", data, spec=observed_spec)
    episode["feature_columns"] = ["feature_signal"]
    episode["account"]["inputs"]["dataset"] = str(tmp_path / "observed")
    env = TradingEnv([episode])
    try:
        observation, _ = env.reset(seed=1)
        assert np.isnan(env.decision.features.loc["a0", "feature_signal"])
        assert np.isnan(observation[0])
        assert observation[1] == 1
    finally:
        env.close()


def test_hft_observation_rejects_nonquote_context(tmp_path):
    episode = hft_episode(tmp_path)
    panel = read_dataset(tmp_path / "quotes").assign(future_context=1.0)
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    write_dataset(
        tmp_path / "future",
        panel,
        spec=ProcessedDataSpec(
            data_id="hft_panel",
            market_id="a_share",
            kind="panel",
            frequency="tick",
            grain="asset_timestamp",
            column_roles=roles,
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
        ),
    )
    episode["account"]["panel"] = str(tmp_path / "future")
    episode["feature_columns"] = ["future_context"]
    env = TradingEnv([episode])
    try:
        with pytest.raises(ValueError, match="feature_role_required"):
            env.reset(seed=1)
    finally:
        env.close()


@pytest.mark.parametrize("action", [0.0, 1.0])
def test_execution_opportunity_cost_uses_last_observed_quote_not_daily_close(
    tmp_path, action
):
    episode = hft_episode(tmp_path / "inputs")
    panel = read_dataset(tmp_path / "inputs" / "quotes")
    panel.loc[2, ["mid", "bid_px1", "ask_px1"]] = [8.0, 7.99, 8.01]
    roles = dict.fromkeys(panel.columns, "context")
    roles.update(timestamp="timestamp", asset_id="key", eligible="mask")
    write_dataset(
        tmp_path / "observed",
        panel,
        spec=ProcessedDataSpec(
            data_id="hft_panel",
            market_id="a_share",
            kind="panel",
            frequency="tick",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=(),
            column_roles=roles,
        ),
    )
    episode["account"]["panel"] = str(tmp_path / "observed")
    episode["window"]["end"] = "2024-01-02T07:00:00"
    env = TradingEnv([episode])
    try:
        _, initial = env.reset(seed=1)
        assert initial["opportunity_cost_bps"] is None
        assert initial["opportunity_cost_reason"] == "episode_in_progress"
        done = False
        while not done:
            _, _, done, _, info = env.step([action])
        assert info["arrival_price"] == 10.0
        assert info["terminal_observed_mark"] == 8.0
        assert info["unfilled_quantity"] == (100.0 if action == 0 else 0.0)
        assert info["opportunity_cost_bps"] == pytest.approx(
            2000.0 if action == 0 else 0.0
        )
        assert info["total_implementation_shortfall_bps"] == pytest.approx(
            info["implementation_shortfall_bps"] + info["opportunity_cost_bps"]
        )
        assert info["unfilled_reason"] == (
            "execution_window_ended" if action == 0 else None
        )
        assert info["opportunity_cost_reason"] is None
    finally:
        env.close()
