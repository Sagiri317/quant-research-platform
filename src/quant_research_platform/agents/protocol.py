"""Bind native research tools to one caller-owned scientific protocol."""

import os
import re
from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.storage.datasets.reading import open_dataset_session

OPERATIONS = {
    "prepare",
    "assemble",
    "fit",
    "predict",
    "evaluate",
    "score",
    "select",
    "backtest",
    "calibrate",
    "statistics-factors",
    "statistics-model",
    "statistics-portfolio",
    "plot",
    "train-policy",
    "backtest-policy",
    "search-factors",
    "apply-formula",
    "calibrate-uncertainty",
    "apply-uncertainty",
    "evaluate-uncertainty",
    "fit-cost-model",
    "backtest-cost-model",
    "normalize-book",
    "evaluate-time-series",
    "analyze-inventory",
    "fit-hawkes",
    "backtest-hft",
    "explain",
}
TUNABLE = re.compile(
    r"^(?:model\.(?:model_algorithm|input_factor_columns|input_feature_columns|"
    r"hyperparameters\.[A-Za-z_][A-Za-z_0-9]*)|"
    r"candidate_parameters\.[A-Za-z_][A-Za-z_0-9]*|"
    r"columns|expressions|feature_columns|baseline_columns|hidden_sizes|epochs|"
    r"learning_rate|ppo\.[A-Za-z_][A-Za-z_0-9]*|"
    r"parameters\.(?:buy_threshold|sell_threshold))$"
)


def resolve_inputs(value, inputs):
    """Resolve explicitly attached local inputs without exposing their paths."""
    if isinstance(value, str) and value.startswith("input:"):
        return inputs[value.removeprefix("input:")]
    if isinstance(value, dict):
        return {key: resolve_inputs(item, inputs) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_inputs(item, inputs) for item in value]
    return value


def bind_protocol(value, protocol):
    """Resolve protocol references without creating another copy of their meaning."""
    if isinstance(value, str) and value.startswith("protocol:"):
        selected = protocol
        for key in value.removeprefix("protocol:").split("."):
            selected = selected[key]
        return deepcopy(selected)
    if isinstance(value, list):
        return [bind_protocol(item, protocol) for item in value]
    if isinstance(value, dict):
        result = {key: bind_protocol(item, protocol) for key, item in value.items()}
        for key, item in result.items():
            if (
                key in {"label", "target_column", "primary_label"}
                and item != protocol["label"]
            ):
                raise ValueError("agent_protocol_label_conflict")
            if key in {"train_window", "validation_window"} and item != protocol[key]:
                raise ValueError("agent_protocol_window_conflict")
            if key in protocol["costs"] and item != protocol["costs"][key]:
                raise ValueError("agent_protocol_cost_conflict")
        return result
    return value


def set_argument(arguments, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        arguments = (
            arguments[int(part)] if isinstance(arguments, list) else arguments[part]
        )
    key = int(parts[-1]) if isinstance(arguments, list) else parts[-1]
    arguments[key] = value


def artifact_references(value):
    if isinstance(value, str) and value.startswith("artifact:"):
        yield value.removeprefix("artifact:").partition("/")[0]
    elif isinstance(value, dict):
        for item in value.values():
            yield from artifact_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from artifact_references(item)


def before_holdout(value, holdout_start):
    if isinstance(value, dict):
        if set(value) == {"start", "end"}:
            _, end, closed = time_window_timestamp_interval(TimeWindow(**value))
            if pd.Timestamp(end) > holdout_start or (
                closed and pd.Timestamp(end) == holdout_start
            ):
                raise ValueError("agent_tool_window_reaches_holdout")
        for key, item in value.items():
            if (
                key
                in {
                    "cutoff",
                    "fit_cutoff",
                    "selection_cutoff",
                    "train_cutoff",
                    "refit_cutoff",
                    "label_cutoff",
                    "evaluation_cutoff",
                }
                and pd.Timestamp(item) >= holdout_start
            ):
                raise ValueError("agent_tool_cutoff_reaches_holdout")
            before_holdout(item, holdout_start)
    elif isinstance(value, list):
        for item in value:
            before_holdout(item, holdout_start)


def dataset_context(inputs):
    """Expose native metadata without treating the whole table as a sample window."""
    result = {}
    for alias, root in inputs.items():
        session = open_dataset_session(Path(root))
        spec, manifest = session.spec, session.manifest
        result[alias] = {
            "columns": spec["column_roles"],
            "units": spec.get("column_units", {}),
            "frequency": spec["frequency"],
            "rows": manifest["row_count"],
            "date_min": manifest["date_min"],
            "date_max": manifest["date_max"],
            "label_availability_columns": spec.get("label_availability_columns", {}),
        }
    return result


def validate_budget(budget):
    budget = _strict_config(
        budget,
        {"max_requests", "max_experiments", "wall_seconds"},
        {"max_context_chars", "max_tokens"},
    )
    if any(
        type(value) is not int or value < 1
        for key, value in budget.items()
        if key != "max_context_chars" or value is not None
    ):
        raise ValueError("agent_budget_invalid")
    return budget


def validate_config(config):
    config = _strict_config(
        config,
        {"question", "protocol", "tools", "budget", "output", "llm"},
        {
            "sandbox",
            "documents",
            "memory",
            "final_evaluation",
            "web_search",
            "inputs",
            "history",
            "workspace",
        },
    )
    protocol = config["protocol"]
    if protocol is None:
        if config["tools"] or config.get("final_evaluation") or "sandbox" in config:
            raise ValueError("agent_research_protocol_required")
        test_start = None
    else:
        protocol = validate_protocol(protocol)
        config["protocol"] = protocol
        test_start = pd.Timestamp(
            time_window_timestamp_interval(TimeWindow(**protocol["test_window"]))[0]
        )
    config["budget"] = validate_budget(config["budget"])
    public_tools = {}
    final = config.get("final_evaluation", [])
    final_names = [item["name"] for item in final]
    if len(set(final_names)) != len(final_names):
        raise ValueError("agent_final_step_name_duplicate")
    resolved_final = []
    for phase, templates in (
        ("development", config["tools"].items()),
        ("final", [(item["name"], item) for item in final]),
    ):
        for name, template in templates:
            if not re.fullmatch(r"[A-Za-z][A-Za-z_0-9-]*", name):
                raise ValueError("agent_tool_name_invalid")
            template = _strict_config(
                template,
                {"operation", "description", "config", "parameters", "metrics"},
                {"bindings", "facts", "name"}
                if phase == "final"
                else {"bindings", "facts"},
            )
            if template["operation"] not in OPERATIONS:
                raise ValueError("agent_operation_not_allowed")
            template["config"] = bind_protocol(template["config"], protocol)
            if phase == "development":
                before_holdout(template["config"], test_start)
            elif template["parameters"] or template.get("bindings"):
                raise ValueError("agent_final_steps_must_be_fixed")
            if any(not TUNABLE.fullmatch(path) for path in template["parameters"]):
                raise ValueError("agent_tunable_not_allowed")
            if phase == "development":
                config["tools"][name] = template
                public_tools[name] = {
                    "operation": template["operation"],
                    "description": template["description"],
                    "parameters": template["parameters"],
                    "inputs": template.get("bindings", {}),
                    "metrics": list(template["metrics"]),
                    "facts": list(template.get("facts", {})),
                }
            else:
                resolved_final.append(template)
    if final:
        config["final_evaluation"] = resolved_final
    public = {
        "question": config["question"],
        "protocol": protocol,
        "tools": public_tools,
        "budget": config["budget"],
        "sandbox_inputs": dataset_context(config.get("sandbox", {}).get("inputs", {})),
        "documents": list(config.get("documents", {})),
        "final_evaluation": [
            {
                "name": item["name"],
                "description": item["description"],
                "inputs": sorted(set(artifact_references(item["config"]))),
            }
            for item in resolved_final
        ],
    }
    if "inputs" in config:
        public["inputs"] = dataset_context(config["inputs"])
        if "workspace" not in config:
            public["operations"] = sorted(OPERATIONS)
    if "workspace" in config:
        config["workspace"] = str(Path(config["workspace"]).resolve(strict=True))
        public["runtime"] = {
            "model": config["llm"].get("model"),
            "shell": "PowerShell" if os.name == "nt" else "Bash",
            "web_search": config.get("web_search", False),
            "workspace": ".",
            "source": "src/quant_research_platform/",
            "data": "data/",
            "python": "QRP_PYTHON",
            "outputs": "QRP_OUTPUT",
        }
    if evidence_json_value_unsafe_reason(public) or evidence_json_value_unsafe_reason(
        config.get("history", [])
    ):
        raise ValueError("agent_public_context_not_shareable")
    return config, public


def validate_protocol(protocol):
    protocol = _strict_config(
        protocol,
        {
            "label",
            "train_window",
            "validation_window",
            "test_window",
            "costs",
            "metric",
        },
        set(),
    )
    test_start = pd.Timestamp(
        time_window_timestamp_interval(TimeWindow(**protocol["test_window"]))[0]
    )
    _, train_end, train_closed = time_window_timestamp_interval(
        TimeWindow(**protocol["train_window"])
    )
    val_start, val_end, val_closed = time_window_timestamp_interval(
        TimeWindow(**protocol["validation_window"])
    )
    if (
        not pd.Timestamp(train_end)
        <= pd.Timestamp(val_start)
        <= pd.Timestamp(val_end)
        <= test_start
        or train_closed
        and pd.Timestamp(train_end) == pd.Timestamp(val_start)
        or val_closed
        and pd.Timestamp(val_end) == test_start
    ):
        raise ValueError("agent_protocol_windows_overlap")
    return protocol
