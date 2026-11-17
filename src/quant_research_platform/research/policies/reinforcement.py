"""Train PPO policies and compare frozen policies with explicit trading baselines.

Training saves policy weights, observation normalization, and episode facts.
Evaluation writes the native portfolio ledger and task-specific metrics without
updating the saved policy or normalization statistics.
"""

import pickle
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.policies import PolicyDecision, PolicyExecution
from quant_research_platform.contracts.portfolios import (
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.risk import RiskInputsUnavailable
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_inclusive_bounds,
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
    reject_if_exists,
)
from quant_research_platform.storage.manifests import write_qrp_json_file


def train_policy(config):
    """Train on explicitly bounded episodes; persist PPO and frozen normalization."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor
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
        {"validation_episodes", "early_stopping"},
    )
    for key in ("timesteps", "n_envs", "threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("policy_training_budget_invalid")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("policy_seed_invalid")
    episodes = [_episode(value) for value in config["episodes"]]
    validation_episodes = [
        _episode(value) for value in config.get("validation_episodes", [])
    ]
    bounds = [
        time_window_inclusive_bounds(TimeWindow(**item["window"]))
        for item in (*episodes, *validation_episodes)
    ]
    cutoff = pd.Timestamp(config["fit_cutoff"])
    if pd.isna(cutoff) or not episodes or any(end > cutoff for _, end in bounds):
        raise ValueError("policy_training_after_cutoff")
    stopping = None
    if validation_episodes:
        stopping = _strict_config(
            config.get("early_stopping"),
            {"eval_freq", "patience", "min_evals"},
            set(),
        )
        if any(type(value) is not int or value < 1 for value in stopping.values()):
            raise ValueError("policy_early_stopping_invalid")
        train_end = max(end for _, end in bounds[: len(episodes)])
        validation_start = min(start for start, _ in bounds[len(episodes) :])
        if train_end >= validation_start:
            raise ValueError("policy_validation_not_after_training")
        probe = TradingEnv([*episodes, *validation_episodes])
        probe.close()
    elif "early_stopping" in config or episodes[0]["task"] == "dynamic_execution":
        raise ValueError("policy_historical_validation_required")
    config["episodes"] = episodes
    if validation_episodes:
        config["validation_episodes"] = validation_episodes
        config["early_stopping"] = stopping
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
        decision_start = decision_end = None
        decision_count = 0
        for episode in episodes:
            observed = False
            with _stream(episode) as stream:
                try:
                    event = next(stream)
                    while True:
                        if isinstance(event, PolicyDecision):
                            observed = True
                            decision_start = (
                                event.timestamp
                                if decision_start is None
                                else min(decision_start, event.timestamp)
                            )
                            decision_end = (
                                event.timestamp
                                if decision_end is None
                                else max(decision_end, event.timestamp)
                            )
                            decision_count += 1
                            observations.append(
                                event.features.reindex(episode["asset_ids"])[
                                    episode["feature_columns"]
                                ].to_numpy(dtype="float64")
                            )
                            target = (
                                event.quantities.reindex(
                                    event.features.index, fill_value=0
                                )
                                if episode["kind"] == "hft"
                                else pd.Series(0.0, index=event.features.index)
                            )
                            if episode["task"] == "dynamic_execution":
                                target = PolicyExecution(
                                    target,
                                    pd.Series("wait", index=target.index),
                                    target,
                                )
                            event = stream.send(target)
                        else:
                            event = next(stream)
                except StopIteration:
                    pass
            if not observed:
                raise ValueError("policy_episode_no_decisions")
        if not observations:
            raise ValueError("policy_episode_no_decisions")
        feature_scaler = ModelMatrixScaler.fit(
            pd.DataFrame(
                np.concatenate(observations), columns=episodes[0]["feature_columns"]
            ),
            missing="impute",
            scaling=False,
        )
        del observations
        diagnostic_columns = (
            "initial_nav",
            "nav",
            "fees",
            "economic_reward",
            "reward_penalty",
        )
        if episodes[0]["task"] == "execution":
            diagnostic_columns += (
                "completion_rate",
                "filled_quantity",
                "unfilled_quantity",
                "unfilled_reason",
                "last_block_reason",
                "arrival_price",
                "terminal_observed_mark",
                "implementation_shortfall_bps",
                "opportunity_cost_bps",
                "opportunity_cost_reason",
                "total_implementation_shortfall_bps",
            )

        def monitored_environment(episodes, filename=None):
            return Monitor(
                TradingEnv(episodes, feature_scaler=feature_scaler),
                filename=filename,
                info_keywords=("episode_index", *diagnostic_columns),
            )

        factories = [
            partial(
                monitored_environment,
                episodes,
                str(trial.root / f"training-env-{index}.monitor.csv"),
            )
            for index in range(config["n_envs"])
        ]
        with (
            TemporaryDirectory() as checkpoint_directory,
            closing_iterator(ExitStack()) as resources,
            closing_iterator(
                VecNormalize(
                    (DummyVecEnv if len(factories) == 1 else SubprocVecEnv)(factories),
                    norm_reward=True,
                )
            ) as environment,
            closing_iterator(configure(str(trial.root), ["csv"])) as logger,
        ):
            model = PPO(
                "MlpPolicy",
                environment,
                seed=config["seed"],
                device=config["device"],
                **parameters,
            )
            model.set_logger(logger)
            environment.gamma = model.gamma
            callback = None
            best = {}
            if validation_episodes:
                from stable_baselines3.common.callbacks import (
                    BaseCallback,
                    EvalCallback,
                    StopTrainingOnNoModelImprovement,
                )

                validation = resources.enter_context(
                    closing_iterator(
                        VecNormalize(
                            DummyVecEnv(
                                [
                                    partial(monitored_environment, [episode])
                                    for episode in validation_episodes
                                ]
                            ),
                            training=False,
                            norm_reward=False,
                        )
                    )
                )
                checkpoint = Path(checkpoint_directory)

                class SaveNormalization(BaseCallback):
                    def _on_step(self):
                        environment.save(checkpoint / "normalization.pkl")
                        best["selected_timesteps"] = self.num_timesteps
                        best["selected_updates"] = self.model._n_updates
                        return True

                class EvaluationDiagnostics(EvalCallback):
                    def _log_success_callback(self, locals_, globals_):
                        super()._log_success_callback(locals_, globals_)
                        if locals_["done"]:
                            info = locals_["info"]
                            index = int(locals_["i"])
                            metrics = {key: info[key] for key in diagnostic_columns}
                            self.logger.record("eval/n_updates", self.model._n_updates)
                            for key, value in {
                                "reward": info["episode"]["r"],
                                "length": info["episode"]["l"],
                                **metrics,
                            }.items():
                                self.logger.record(f"eval/episode_{index}/{key}", value)
                            self.episode_metrics.append(
                                {
                                    "episode_index": index,
                                    **metrics,
                                }
                            )

                    def _on_step(self):
                        self.episode_metrics = []
                        keep_training = super()._on_step()
                        if self.episode_metrics:
                            best.setdefault("evaluations", []).append(
                                {
                                    "timesteps": self.num_timesteps,
                                    "updates": self.model._n_updates,
                                    "episode_rewards": self.evaluations_results[-1],
                                    "episode_lengths": self.evaluations_length[-1],
                                    "episode_metrics": self.episode_metrics,
                                }
                            )
                        return keep_training

                stopper = StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=stopping["patience"],
                    min_evals=stopping["min_evals"],
                )
                callback = EvaluationDiagnostics(
                    validation,
                    callback_on_new_best=SaveNormalization(),
                    callback_after_eval=stopper,
                    n_eval_episodes=len(validation_episodes),
                    eval_freq=max(1, stopping["eval_freq"] // config["n_envs"]),
                    best_model_save_path=checkpoint_directory,
                    log_path=checkpoint_directory,
                    deterministic=True,
                    verbose=0,
                )
            model.learn(total_timesteps=config["timesteps"], callback=callback)
            # SB3 logs before each update; flush the final update as well.
            logger.record("time/total_timesteps", model.num_timesteps)
            logger.dump(model.num_timesteps)
            if callback is not None:
                stopped_early = (
                    stopper.no_improvement_evals > stopper.max_no_improvement_evals
                )
                if not stopped_early:
                    # Evaluate the last gradient update, also below eval_freq.
                    callback.eval_freq = 1
                    callback.on_step()
                best.update(
                    best_validation_reward=float(callback.best_mean_reward),
                    stopped_early=stopped_early,
                )
            separate_final = callback is not None and (
                best["selected_timesteps"],
                best["selected_updates"],
            ) != (model.num_timesteps, model._n_updates)
            facts = {
                "kind": "ppo_policy",
                "config": config,
                "actual_timesteps": model.num_timesteps,
                "actual_updates": model._n_updates,
                "device": str(model.device),
                "trial": str(trial.root),
                "validation": best or None,
                "final_checkpoint": {
                    "policy": "final_policy.zip" if separate_final else "policy.zip",
                    "normalization": "final_normalization.pkl"
                    if separate_final
                    else "normalization.pkl",
                },
                "training_decisions": {
                    "start": decision_start.isoformat(),
                    "end": decision_end.isoformat(),
                    "count": decision_count,
                },
            }

            def write(staging):
                if callback is None:
                    model.save(staging / "policy.zip")
                    environment.save(staging / "normalization.pkl")
                else:
                    copyfile(checkpoint / "best_model.zip", staging / "policy.zip")
                    copyfile(
                        checkpoint / "normalization.pkl", staging / "normalization.pkl"
                    )
                if separate_final:
                    model.save(staging / "final_policy.zip")
                    environment.save(staging / "final_normalization.pkl")
                (staging / "feature_scaler.pkl").write_bytes(
                    pickle.dumps(feature_scaler)
                )
                write_qrp_json_file(staging / "policy.json", facts)

            atomic_write_directory(output, write)
        trial.finish(result={"actual_timesteps": model.num_timesteps})
    except BaseException as error:
        trial.fail(error)
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
            "sample_role",
            "checkpoint",
        },
    )
    role = config.setdefault("sample_role", "out_of_sample")
    if role not in {"out_of_sample", "diagnostic"}:
        raise ValueError("policy_sample_role_invalid")
    checkpoint = config.get("checkpoint", "selected")
    if checkpoint not in {"selected", "final"} or (
        "checkpoint" in config and config["strategy"] != "ppo"
    ):
        raise ValueError("policy_checkpoint_invalid")
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
        if (
            role == "out_of_sample"
            and pd.Timestamp(facts["config"]["fit_cutoff"]) >= lower
        ):
            raise ValueError("policy_backtest_not_out_of_sample")
        first = facts["config"]["episodes"][0]
        if any(
            first[key] != episode[key]
            for key in ("kind", "task", "asset_ids", "feature_columns")
        ):
            raise ValueError("policy_backtest_spaces_mismatch")
        # Like the policy itself, this is a trusted local fitted artifact.
        env.feature_scaler = pickle.loads((root / "feature_scaler.pkl").read_bytes())
        members = (
            facts["final_checkpoint"]
            if checkpoint == "final"
            else {"policy": "policy.zip", "normalization": "normalization.pkl"}
        )
        normalization = VecNormalize.load(
            root / members["normalization"], DummyVecEnv([lambda: env])
        )
        normalization.training = False
        normalization.norm_reward = False
        model = PPO.load(root / members["policy"], device=config.get("device", "cpu"))
        config["checkpoint"] = checkpoint
    elif episode["task"] == "dynamic_execution":
        if strategy != "immediate":
            raise ValueError("policy_baseline_task_mismatch")
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
    capital = PortfolioCapital(
        episode["account"].get("candidate_id", "learned_policy")
        if episode["kind"] == "hft"
        else episode["account"]["candidate"]["candidate_id"],
        capital_value,
        "CNY",
        "cash_equity",
    )
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
                        env.decision.maximum_quantities,
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
                            env.decision.maximum_quantities,
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
                elif episode["task"] == "dynamic_execution":
                    action = np.full(len(env.assets), 7, dtype=np.int64)
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
                    observation, _, done, _, info = env.step_targets(
                        env.decision.quantities.copy()
                    )
                    continue
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
                "sample_role": role,
                "policy": config,
                "execution": "native_account",
                **(
                    {
                        "dynamic_execution": {
                            "target_policy": episode["inventory_parameters"],
                            "forecast_horizon_seconds": episode["prediction"][
                                "horizon_seconds"
                            ],
                            "decision_interval_seconds": episode["prediction"][
                                "decision_interval_seconds"
                            ],
                            "reward_unit": "bps_of_initial_nav",
                            "economic_reward": "native_nav_change_after_fees",
                            "target_deviation_bps": episode["target_deviation_bps"],
                            "inventory_risk_bps": episode["risk_aversion"],
                            "actions": [
                                "wait",
                                "cancel",
                                "maker_25pct",
                                "maker_50pct",
                                "maker_100pct",
                                "taker_25pct",
                                "taker_50pct",
                                "taker_100pct",
                            ],
                            "holding_period": "realized_inventory_path",
                        }
                    }
                    if episode["task"] == "dynamic_execution"
                    else {}
                ),
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
        write_qrp_json_file(staging / "summary.json", result)

    atomic_write_directory(output, write)
    return output
