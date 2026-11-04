"""Train PPO policies and compare frozen policies with explicit trading baselines.

Training saves policy weights, observation normalization, and episode facts.
Evaluation writes the native portfolio ledger and task-specific metrics without
updating the saved policy or normalization statistics.
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.policies.accounts import _episode
from quant_research_platform.research.policies.environments import TradingEnv
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    reject_if_exists,
)


def train_policy(config):
    """Train on explicitly bounded episodes; persist PPO and frozen normalization."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )

    config = _strict_config(
        config,
        {
            "episodes",
            "output",
            "fit_cutoff",
            "timesteps",
            "seed",
            "device",
            "n_envs",
            "threads",
            "ppo",
        },
        set(),
    )
    for key in ("timesteps", "n_envs", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("policy_training_budget_invalid")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("policy_seed_invalid")
    episodes = [_episode(value) for value in config["episodes"]]
    cutoff = pd.Timestamp(config["fit_cutoff"])
    if (
        pd.isna(cutoff)
        or not episodes
        or any(
            pd.Timestamp(
                time_window_timestamp_interval(TimeWindow(**item["window"]))[1]
            )
            > cutoff
            for item in episodes
        )
    ):
        raise ValueError("policy_training_after_cutoff")
    parameters = _strict_config(
        config["ppo"],
        set(),
        {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "policy_kwargs",
            "target_kl",
        },
    )
    output = Path(config["output"])
    reject_if_exists(output)
    trial = TrialRecorder(
        output.with_name(output.name + ".trial"),
        {"kind": "ppo_policy", "config": config},
    )
    trial.begin()
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["threads"])
        factories = [partial(TradingEnv, episodes) for _ in range(config["n_envs"])]
        with closing_iterator(
            VecNormalize(
                (DummyVecEnv if len(factories) == 1 else SubprocVecEnv)(factories),
                norm_reward=False,
            )
        ) as environment:
            model = PPO(
                "MlpPolicy",
                environment,
                seed=config["seed"],
                device=config["device"],
                **parameters,
            )
            model.learn(total_timesteps=config["timesteps"])
            facts = {
                "kind": "ppo_policy",
                "config": config,
                "actual_timesteps": model.num_timesteps,
                "device": str(model.device),
            }

            def write(staging):
                model.save(staging / "policy.zip")
                environment.save(staging / "normalization.pkl")
                atomic_write_json(staging / "policy.json", facts)

            atomic_write_directory(output, write)
        trial.finish(result={"actual_timesteps": model.num_timesteps})
    except BaseException as error:
        trial.finish(error=error)
        raise
    finally:
        torch.set_num_threads(old_threads)
    return output


def backtest_policy(config):
    """Replay a frozen policy or explicit execution baseline into a native ledger."""
    config = _strict_config(
        config, {"episode", "output", "strategy"}, {"policy", "device", "volume_curve"}
    )
    episode = _episode(config["episode"])
    strategy = config["strategy"]
    if strategy not in {"ppo", "immediate", "twap", "vwap", "hold", "equal_weight"}:
        raise ValueError("policy_strategy_invalid")
    env = TradingEnv([episode])
    model = normalization = None
    if strategy == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        root = Path(config["policy"])
        facts = parse_qrp_json_object((root / "policy.json").read_bytes())
        lower = pd.Timestamp(
            time_window_timestamp_interval(TimeWindow(**episode["window"]))[0]
        )
        if pd.Timestamp(facts["config"]["fit_cutoff"]) >= lower:
            raise ValueError("policy_backtest_not_out_of_sample")
        first = facts["config"]["episodes"][0]
        if any(
            first[key] != episode[key]
            for key in ("kind", "task", "asset_ids", "feature_columns")
        ):
            raise ValueError("policy_backtest_spaces_mismatch")
        normalization = VecNormalize.load(
            root / "normalization.pkl", DummyVecEnv([lambda: env])
        )
        normalization.training = False
        normalization.norm_reward = False
        model = PPO.load(root / "policy.zip", device=config.get("device", "cpu"))
    elif (strategy in {"immediate", "twap", "vwap"}) != (
        episode["task"] == "execution"
    ):
        raise ValueError("policy_baseline_task_mismatch")
    if strategy == "equal_weight" and episode["task"] != "portfolio":
        raise ValueError("policy_baseline_task_mismatch")
    curve = None
    if strategy == "vwap":
        curve = _strict_config(
            config.get("volume_curve"),
            {"as_of", "fractions", "cumulative_volume"},
            set(),
        )
        lower = pd.Timestamp(
            time_window_timestamp_interval(TimeWindow(**episode["window"]))[0]
        )
        if pd.Timestamp(curve["as_of"]) >= lower:
            raise ValueError("policy_volume_curve_not_historical")
        x, y = (
            np.asarray(curve["fractions"], dtype=float),
            np.asarray(curve["cumulative_volume"], dtype=float),
        )
        if (
            x.shape != y.shape
            or x.ndim != 1
            or len(x) < 2
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or x[0] != 0
            or x[-1] != 1
            or y[0] != 0
            or y[-1] != 1
            or (np.diff(x) <= 0).any()
            or (np.diff(y) < 0).any()
        ):
            raise ValueError("policy_volume_curve_invalid")
    capital_value = (
        episode["account"]["parameters"]["initial_capital"]
        if episode["kind"] == "hft"
        else episode["account"]["assumptions"]["initial_nav"]
    )
    capital = PortfolioCapital("learned_policy", capital_value, "CNY", "cash_equity")
    result = {}

    def produce(emit):
        env.on_batch = emit
        with closing_iterator(normalization if normalization is not None else env):
            observation, info = env.reset(seed=0)
            done = False
            while not done:
                if model is not None:
                    action, _ = model.predict(
                        normalization.normalize_obs(observation), deterministic=True
                    )
                elif episode["task"] == "execution":
                    lower, upper, _ = time_window_timestamp_interval(
                        TimeWindow(**episode["window"])
                    )
                    fraction = float(
                        (env.decision.timestamp - pd.Timestamp(lower))
                        / (pd.Timestamp(upper) - pd.Timestamp(lower))
                    )
                    desired = (
                        1.0
                        if strategy == "immediate"
                        else fraction
                        if strategy == "twap"
                        else float(np.interp(fraction, x, y))
                    )
                    completed = info["completion_rate"]
                    action = np.array(
                        [
                            np.clip(
                                (desired - completed) / max(1 - completed, 1e-30), 0, 1
                            )
                        ]
                    )
                elif episode["task"] == "inventory":
                    maximum = pd.Series(
                        episode["account"]["parameters"]["maximum_inventory"]
                    ).reindex(env.assets)
                    action = (
                        env.decision.quantities.reindex(env.assets)
                        .divide(maximum.where(maximum > 0))
                        .fillna(0)
                        .to_numpy()
                    )
                elif strategy == "equal_weight":
                    action = np.array([1.0] * len(env.assets) + [0.0])
                else:
                    holding = env.decision.quantities.reindex(
                        env.assets, fill_value=0
                    ) * env.decision.marks.reindex(env.assets)
                    action = np.r_[
                        holding.fillna(0).to_numpy(), env.decision.cash
                    ] / max(env.decision.nav, 1)
                observation, _, done, _, info = env.step(action)
            result.update(info)

    output = Path(config["output"])

    # The ledger is the sole accounting result; the report contains only its summary.
    def write(staging):
        write_portfolio_ledger_artifact(
            staging / "ledger",
            capital=capital,
            metadata={
                "kind": "a_share_hft" if episode["kind"] == "hft" else "a_share",
                "market_id": "a_share",
                "window": episode["window"],
                "policy": config,
                "execution": "native_account",
            },
            produce=produce,
        )
        atomic_write_json(staging / "summary.json", result)

    atomic_write_directory(output, write)
    return output
