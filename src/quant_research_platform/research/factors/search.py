"""Budgeted symbolic factor discovery without Python eval or holdout feedback."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import (
    _strict_config,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.research.datasets.learning import (
    finite_mean,
    mature_rows,
    research_frame,
    window_mask,
)
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.research.statistics import cross_sectional_rank_ic
from quant_research_platform.storage.atomic import atomic_write_json, reject_if_exists
from quant_research_platform.storage.datasets import write_dataset


def formula_values(frame, expression, columns):
    """Evaluate causal arithmetic, cross-sectional ranks and observed-row windows."""
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 256:
        raise ValueError("factor_formula_too_large")
    if not frame.index.is_unique or frame.duplicated(["timestamp", "asset_id"]).any():
        raise ValueError("factor_formula_keys_invalid")
    ordered = frame.sort_values(["asset_id", "timestamp"])
    groups = [ordered.asset_id]
    if "segment_id" in ordered:
        groups.extend((ordered.timestamp.dt.normalize(), ordered.segment_id))

    def visit(node):
        if isinstance(node, ast.Name) and node.id in columns:
            return ordered[node.id].astype(float)
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
            and np.isfinite(node.value)
        ):
            return pd.Series(float(node.value), index=ordered.index)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right.where(right.ne(0))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            name = node.func.id
            if name in {"abs", "sign", "logabs", "rank"} and len(node.args) == 1:
                values = visit(node.args[0])
                if name == "abs":
                    return values.abs()
                if name == "sign":
                    return np.sign(values)
                if name == "logabs":
                    return np.log1p(values.abs())
                return (
                    values.where(ordered.eligible)
                    .groupby(ordered.timestamp)
                    .rank(pct=True)
                    - 0.5
                )
            if name in {"delay", "mean", "std"} and len(node.args) == 2:
                arg = node.args[1]
                if (
                    not isinstance(arg, ast.Constant)
                    or type(arg.value) is not int
                    or arg.value <= 0
                ):
                    raise ValueError("factor_window_positive_integer_required")
                values = visit(node.args[0]).groupby(groups, sort=False)
                if name == "delay":
                    return values.shift(arg.value)
                return values.transform(
                    lambda x: getattr(
                        x.rolling(arg.value, min_periods=arg.value), name
                    )()
                )
        raise ValueError("factor_formula_not_allowed")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = visit(tree.body)
    return result.where(np.isfinite(result)).reindex(frame.index)


def _score(frame, values, label, baseline_columns):
    paired = frame.assign(candidate=values.reindex(frame.index))
    raw = finite_mean(
        cross_sectional_rank_ic(
            paired, value_columns=("candidate",), label_column=label
        ).candidate
    )
    if not baseline_columns:
        return raw, raw
    residual = pd.Series(np.nan, index=paired.index)
    for _, group in paired.groupby("timestamp", sort=False):
        sample = group.dropna(subset=["candidate", *baseline_columns])
        if len(sample) <= len(baseline_columns) + 1:
            continue
        ranked = sample[["candidate", *baseline_columns]].rank()
        x = np.column_stack(
            (np.ones(len(sample)), ranked[list(baseline_columns)].to_numpy())
        )
        if np.linalg.matrix_rank(x) != x.shape[1]:
            continue
        y = ranked.candidate.to_numpy()
        remainder = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        if np.linalg.norm(remainder) > 1e-10 * max(1, np.linalg.norm(y)):
            residual.loc[sample.index] = remainder
    paired["candidate"] = residual
    incremental = finite_mean(
        cross_sectional_rank_ic(
            paired, value_columns=("candidate",), label_column=label
        ).candidate
    )
    return raw, incremental


def search_factors(config):
    """Search a bounded formula space on training data, then select once on validation.

    Save every evaluated candidate, including undefined scores, and the selected
    expression. Validation scores are never fed back into adaptive proposals.
    """
    config = _strict_config(
        config,
        {
            "dataset",
            "columns",
            "baseline_columns",
            "label",
            "train_window",
            "validation_window",
            "train_cutoff",
            "selection_cutoff",
            "method",
            "budget",
            "max_depth",
            "windows",
            "seed",
            "output",
        },
        {"timesteps", "ppo", "device", "expressions"},
    )
    for name in ("budget", "max_depth"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError("factor_search_budget_invalid")
    if not config["windows"] or any(
        type(n) is not int or n < 1 for n in config["windows"]
    ):
        raise ValueError("factor_search_windows_invalid")
    if not config["columns"] or type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("factor_search_inputs_invalid")
    train_end = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**config["train_window"]))[1]
    )
    val_start = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**config["validation_window"]))[0]
    )
    if (
        train_end > pd.Timestamp(config["train_cutoff"])
        or pd.Timestamp(config["train_cutoff"]) >= val_start
        or val_start > pd.Timestamp(config["selection_cutoff"])
    ):
        raise ValueError("factor_search_windows_not_chronological")
    window = {
        "start": config["train_window"]["start"],
        "end": config["validation_window"]["end"],
    }
    columns = tuple(dict.fromkeys((*config["columns"], *config["baseline_columns"])))
    frame, session, availability = research_frame(
        config["dataset"], window, columns, label=config["label"]
    )
    spec = session.spec
    train = mature_rows(
        frame.loc[window_mask(frame, config["train_window"])],
        label=config["label"],
        availability=availability,
        cutoff=config["train_cutoff"],
    )
    validation = mature_rows(
        frame.loc[window_mask(frame, config["validation_window"])],
        label=config["label"],
        availability=availability,
        cutoff=config["selection_cutoff"],
    )
    if train.empty or validation.empty:
        raise ValueError("factor_search_mature_sample_empty")
    output = Path(config["output"])
    reject_if_exists(output)
    recorder = TrialRecorder(
        output.with_name(output.name + ".trial"),
        {"kind": "factor_search", "config": config},
    )
    recorder.begin()
    records, seen = [], set()
    rng = np.random.default_rng(config["seed"])

    def candidate(expression):
        identity = ast.dump(ast.parse(expression, mode="eval"))
        if identity in seen or len(records) >= config["budget"]:
            return 0.0
        seen.add(identity)
        # Adaptive proposals only receive training statistics.
        values = formula_values(
            frame.loc[frame.timestamp.le(train_end)], expression, columns
        )
        raw, increment = _score(
            train, values, config["label"], config["baseline_columns"]
        )
        records.append(
            {
                "expression": expression,
                "train_rank_ic": raw,
                "train_incremental_rank_ic": increment,
                "status": "DEFINED" if increment is not None else "UNDEFINED",
            }
        )
        return -1.0 if increment is None else increment

    unary = ["abs({})", "sign({})", "logabs({})", "rank({})", "-({})"] + [
        f"{op}({{}},{window})"
        for op in ("delay", "mean", "std")
        for window in config["windows"]
    ]

    def random_formula(depth):
        if depth == 0 or rng.random() < 0.3:
            return str(rng.choice(config["columns"]))
        if rng.random() < 0.5:
            return str(rng.choice(unary)).format(random_formula(depth - 1))
        left = random_formula(depth - 1)
        operator = rng.choice(["+", "-", "*", "/"])
        right = random_formula(depth - 1)
        return f"({left}{operator}{right})"

    try:
        for expression in config.get("expressions", config["columns"]):
            candidate(expression)
        if config["method"] == "random":
            # Attempts, including duplicates, have a finite declared budget.
            for _ in range(config["budget"]):
                candidate(random_formula(config["max_depth"]))
        elif config["method"] == "ppo":
            import gymnasium as gym
            from gymnasium import spaces
            from stable_baselines3 import PPO

            class FormulaEnv(gym.Env):
                def __init__(self):
                    self.action_space = spaces.Discrete(
                        len(config["columns"])
                        + len(unary)
                        + 4 * len(config["columns"])
                        + 1
                    )
                    self.observation_space = spaces.MultiDiscrete(
                        [self.action_space.n + 1] * (config["max_depth"] + 1)
                    )

                def reset(self, *, seed=None, options=None):
                    super().reset(seed=seed)
                    self.expression = config["columns"][0]
                    self.actions = np.zeros(
                        self.observation_space.shape, dtype=np.int64
                    )
                    self.depth = 0
                    return self.actions.copy(), {}

                def step(self, action):
                    action = int(action)
                    n = len(config["columns"])
                    if action < n:
                        self.expression = config["columns"][action]
                    elif action < n + len(unary):
                        self.expression = unary[action - n].format(self.expression)
                    elif action < self.action_space.n - 1:
                        offset = action - n - len(unary)
                        operator = ["+", "-", "*", "/"][offset // n]
                        column = config["columns"][offset % n]
                        self.expression = f"({self.expression}{operator}{column})"
                    self.actions[self.depth] = action + 1
                    self.depth += 1
                    done = (
                        self.depth == len(self.actions)
                        or action == self.action_space.n - 1
                    )
                    reward = candidate(self.expression) if done else 0.0
                    return self.actions.copy(), reward, done, False, {}

            model = PPO(
                "MlpPolicy",
                FormulaEnv(),
                seed=config["seed"],
                device=config.get("device", "cpu"),
                **config.get("ppo", {}),
            )
            with closing_iterator(model.get_env()):
                model.learn(total_timesteps=config["timesteps"])
        else:
            raise ValueError("factor_search_method_invalid")
        for record in records:
            values = formula_values(frame, record["expression"], columns)
            raw, incremental = _score(
                validation, values, config["label"], config["baseline_columns"]
            )
            record.update(
                validation_rank_ic=raw, validation_incremental_rank_ic=incremental
            )
        defined = [
            row for row in records if row["validation_incremental_rank_ic"] is not None
        ]
        best = (
            max(defined, key=lambda row: row["validation_incremental_rank_ic"])
            if defined
            else None
        )
        result = {
            "kind": "formula_search",
            "config": config,
            "frequency": spec["frequency"],
            "candidates": records,
            "selected_expression": best["expression"] if best else None,
            "validation_score": best["validation_incremental_rank_ic"]
            if best
            else None,
            "selection_cutoff": config["selection_cutoff"],
            "status": "DEFINED" if best else "UNDEFINED",
            "inference": "adaptive_development_search_no_holdout_evidence",
        }
        if config["method"] == "ppo":
            result["device"] = str(model.device)
        atomic_write_json(output, result)
        recorder.finish(
            result={"candidate_count": len(records), "status": result["status"]}
        )
    except BaseException as error:
        recorder.finish(error=error)
        raise
    return output


def apply_formula(config):
    """Apply the frozen selection to later rows, retaining declared warm-up history."""
    config = _strict_config(
        config, {"search", "dataset", "input_window", "output_window", "output"}, set()
    )
    selected = parse_qrp_json_object(Path(config["search"]).read_bytes())
    expression = selected["selected_expression"]
    if expression is None:
        raise ValueError("factor_search_no_defined_candidate")
    lower = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**config["output_window"]))[0]
    )
    if pd.Timestamp(selected["selection_cutoff"]) >= lower:
        raise ValueError("factor_selection_not_frozen")
    columns = tuple(
        dict.fromkeys(
            (*selected["config"]["columns"], *selected["config"]["baseline_columns"])
        )
    )
    frame, session, _ = research_frame(
        config["dataset"], config["input_window"], columns
    )
    spec = session.spec
    frame["discovered_factor"] = formula_values(frame, expression, columns)
    frame = frame.loc[
        window_mask(frame, config["output_window"]),
        ["timestamp", "asset_id", "eligible", "discovered_factor"],
    ]
    output_spec = DataSpec(
        data_id="discovered_factor",
        frequency=spec["frequency"],
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=(),
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "eligible": "mask",
            "discovered_factor": "factor",
        },
    )
    write_dataset(
        Path(config["output"]),
        frame,
        spec=output_spec,
        metadata={
            "parameters": {
                "expression": expression,
                "selection_cutoff": selected["selection_cutoff"],
                "search": config["search"],
                "window_semantics": "observed_rows",
            }
        },
    )
    return Path(config["output"])
