"""Train PPO policies and compare frozen policies with explicit trading baselines.

Training saves policy weights, observation normalization, and episode facts.
Evaluation writes the native portfolio ledger and task-specific metrics without
updating the saved policy or normalization statistics.
"""

import pickle
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
from quant_research_platform.contracts.policies import PolicyDecision
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.risk import RiskInputsUnavailable
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.models.scaler import ModelMatrixScaler
from quant_research_platform.research.policies.accounts import _episode, _stream
from quant_research_platform.research.policies.environments import TradingEnv
from quant_research_platform.research.policies.execution_schedules import (
    execution_fraction,
    execution_schedule_parameters,
    validate_execution_as_of,
)
from quant_research_platform.research.policies.inventory_policy import (
    inventory_policy_parameters,
    inventory_targets,
)
from quant_research_platform.research.policies.portfolio_policy import (
    portfolio_policy_parameters,
    portfolio_targets,
)
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
        observations = []
        for episode in episodes:
            with _stream(episode) as stream:
                try:
                    event = next(stream)
                    while True:
                        if isinstance(event, PolicyDecision):
                            observations.append(
                                event.features.reindex(episode["asset_ids"])[
                                    episode["feature_columns"]
                                ]
                            )
                            target = (
                                event.quantities.reindex(
                                    event.features.index, fill_value=0
                                )
                                if episode["kind"] == "hft"
                                else pd.Series(0.0, index=event.features.index)
                            )
                            event = stream.send(target)
                        else:
                            event = next(stream)
                except StopIteration:
                    pass
        if not observations:
            raise ValueError("policy_episode_no_decisions")
        feature_scaler = ModelMatrixScaler.fit(
            pd.concat(observations, ignore_index=True), missing="impute", scaling=False
        )
        del observations
        factories = [
            partial(TradingEnv, episodes, feature_scaler=feature_scaler)
            for _ in range(config["n_envs"])
        ]
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
                (staging / "feature_scaler.pkl").write_bytes(
                    pickle.dumps(feature_scaler)
                )
                atomic_write_json(staging / "policy.json", facts)

            atomic_write_directory(output, write)
        trial.finish(result={"actual_timesteps": model.num_timesteps})
    except BaseException as error:
        trial.finish(error=error)
        raise
    finally:
        torch.set_num_threads(old_threads)
    return output


def backtest_policy(config, *, portfolio_context=None):
    """Replay a frozen policy or explicit execution baseline into a native ledger."""
    config = _strict_config(
        config,
        {"episode", "output", "strategy"},
        {
            "policy",
            "device",
            "volume_curve",
            "execution_parameters",
            "inventory_parameters",
            "portfolio_parameters",
            "portfolio_context",
        },
    )
    episode = _episode(config["episode"])
    strategy = config["strategy"]
    if strategy not in {
        "ppo",
        "immediate",
        "twap",
        "vwap",
        "almgren_chriss",
        "signal_ac",
        "hold",
        "equal_weight",
        "etf_inventory",
        "stock_portfolio",
    }:
        raise ValueError("policy_strategy_invalid")
    inventory_parameters = None
    if strategy == "etf_inventory":
        if "prediction" not in episode or episode["task"] != "inventory":
            raise ValueError("inventory_prediction_episode_required")
        inventory_parameters = inventory_policy_parameters(
            config.get("inventory_parameters")
        )
        config["inventory_parameters"] = inventory_parameters
    elif "inventory_parameters" in config:
        raise ValueError("inventory_parameters_unused")
    portfolio_parameters = None
    if strategy == "stock_portfolio":
        if episode["task"] != "portfolio" or "prediction" not in episode:
            raise ValueError("stock_portfolio_prediction_episode_required")
        portfolio_parameters = portfolio_policy_parameters(
            config.get("portfolio_parameters")
        )
        config["portfolio_parameters"] = portfolio_parameters
        if portfolio_context is not None and not callable(portfolio_context):
            raise ValueError("portfolio_context_provider_invalid")
        if (
            "portfolio_context" in config
            and portfolio_parameters["construction"] != "quantile_equal_weight"
        ):
            if portfolio_context is not None:
                raise ValueError("portfolio_context_source_conflict")
            from quant_research_platform.research.portfolios.risk_context import (
                portfolio_risk_context,
            )

            provider = None

            def portfolio_context(decision):
                nonlocal provider
                if provider is None:
                    provider = portfolio_risk_context(
                        config["portfolio_context"],
                        window=TimeWindow(**episode["window"]),
                        construction=portfolio_parameters["construction"],
                    )
                    config["portfolio_context"]["risk_model"] = (
                        provider.risk_model.to_dict()
                    )
                return provider(decision)

    elif (
        "portfolio_parameters" in config
        or "portfolio_context" in config
        or portfolio_context is not None
    ):
        raise ValueError("portfolio_parameters_unused")
    config["episode"] = episode
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
        # Like the policy itself, this is a trusted local fitted artifact.
        env.feature_scaler = pickle.loads((root / "feature_scaler.pkl").read_bytes())
        normalization = VecNormalize.load(
            root / "normalization.pkl", DummyVecEnv([lambda: env])
        )
        normalization.training = False
        normalization.norm_reward = False
        model = PPO.load(root / "policy.zip", device=config.get("device", "cpu"))
    elif (strategy in {"immediate", "twap", "vwap", "almgren_chriss", "signal_ac"}) != (
        episode["task"] == "execution"
    ):
        raise ValueError("policy_baseline_task_mismatch")
    if strategy == "equal_weight" and episode["task"] != "portfolio":
        raise ValueError("policy_baseline_task_mismatch")
    lower, upper, _ = time_window_timestamp_interval(TimeWindow(**episode["window"]))
    lower, upper = pd.Timestamp(lower), pd.Timestamp(upper)
    schedule = None
    if strategy in {"almgren_chriss", "signal_ac"}:
        schedule = execution_schedule_parameters(
            config.get("execution_parameters"),
            signal=strategy == "signal_ac",
            start=lower,
            feature_columns=episode["feature_columns"],
        )
    elif "execution_parameters" in config:
        raise ValueError("execution_schedule_parameters_unused")
    curve = None
    if strategy == "vwap":
        curve = _strict_config(
            config.get("volume_curve"),
            {"as_of", "fractions", "cumulative_volume"},
            set(),
        )
        validate_execution_as_of(
            curve["as_of"], start=lower, error="policy_volume_curve_not_historical"
        )
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
            done = env.done
            previous_desired = 0.0
            while not done:
                if model is not None:
                    action, _ = model.predict(
                        normalization.normalize_obs(observation), deterministic=True
                    )
                elif strategy == "etf_inventory":
                    target = inventory_targets(
                        env.decision,
                        episode["account"]["parameters"]["maximum_inventory"],
                        inventory_parameters,
                    )
                    observation, _, done, _, info = env.step_targets(target)
                    continue
                elif strategy == "stock_portfolio":
                    context = (
                        portfolio_context(env.decision)
                        if portfolio_context is not None
                        and portfolio_parameters["construction"]
                        != "quantile_equal_weight"
                        and env.decision.features.prediction_score.notna().any()
                        else None
                    )
                    try:
                        target, optimization = portfolio_targets(
                            env.decision,
                            episode["account"]["parameters"]["maximum_inventory"],
                            portfolio_parameters,
                            context=context,
                        )
                    except (RiskInputsUnavailable, PortfolioTargetBlocked):
                        if "portfolio_context" not in config:
                            raise
                        # Unknown dependent facts leave this target undefined;
                        # the native account still owns old orders and exits.
                        target = pd.Series(np.nan, index=env.decision.features.index)
                        optimization = None
                    if optimization is not None:
                        result["last_target_total_volatility"] = (
                            optimization.predicted_total_volatility
                        )
                        result["last_target_tracking_error"] = None
                        result["last_target_timestamp"] = (
                            env.decision.timestamp.isoformat()
                        )
                    observation, _, done, _, info = env.step_targets(target)
                    continue
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
                        if strategy == "vwap"
                        else execution_fraction(
                            fraction,
                            (pd.Timestamp(upper) - pd.Timestamp(lower)).total_seconds(),
                            schedule,
                            signal=float(
                                env.decision.features.loc[
                                    episode["order"]["asset_id"],
                                    schedule["signal_column"],
                                ]
                            )
                            if strategy == "signal_ac"
                            else 0.0,
                            side=np.sign(episode["order"]["quantity"]),
                        )
                    )
                    desired = previous_desired = max(desired, previous_desired)
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
                **(
                    {
                        "inventory_policy": {
                            "kind": "transparent_quadratic_candidate_baseline",
                            "forecast_horizon_seconds": episode["prediction"][
                                "horizon_seconds"
                            ],
                            "decision_interval_seconds": episode["prediction"][
                                "decision_interval_seconds"
                            ],
                            "holding_period": "realized_inventory_path",
                            "capital_per_etf": 1_000_000.0 * episode["capital_scale"],
                            "capital_scale": episode["capital_scale"],
                            "capital_basis": "cash_plus_opening_inventory",
                            "decision_cost": (
                                "candidate_fees_plus_visible_spread_or_explicit_bps"
                            ),
                            "realized_cost": "native_execution_fees_only",
                            "pending_assumption": "live_orders_not_inventory",
                            "missing_prediction": (
                                "no_new_target_existing_orders_continue"
                            ),
                        }
                    }
                    if strategy == "etf_inventory"
                    else {}
                ),
                **(
                    {
                        "stock_portfolio_policy": {
                            "construction": portfolio_parameters["construction"],
                            "forecast_horizon_seconds": episode["prediction"][
                                "horizon_seconds"
                            ],
                            "decision_interval_seconds": episode["prediction"][
                                "decision_interval_seconds"
                            ],
                            "holding_period": "continuous_cross_session_holdings",
                            "capital_scale": episode["capital_scale"],
                            "benchmark": None,
                            "risk_limits": portfolio_parameters.get("risk_limits"),
                            "risk_scope": "target_only_not_guaranteed_realized_fills",
                            "context_source": "native_point_in_time_daily_history"
                            if "portfolio_context" in config
                            and portfolio_parameters["construction"]
                            != "quantile_equal_weight"
                            else "caller_point_in_time_provider"
                            if portfolio_context is not None
                            else None,
                            "decision_cost": "explicit_bps"
                            if "trade_cost_bps" in portfolio_parameters
                            else (
                                "native_proportional_fee_plus_visible_spread_"
                                "excluding_minimum_commission"
                            )
                            if portfolio_parameters["construction"] == "absolute_risk"
                            else "ranking_only",
                            "realized_cost": "native_execution_fees_only",
                            "missing_prediction": (
                                "no_new_target_existing_orders_continue"
                            ),
                        }
                    }
                    if strategy == "stock_portfolio"
                    else {}
                ),
            },
            produce=produce,
        )
        atomic_write_json(staging / "summary.json", result)

    atomic_write_directory(output, write)
    return output
