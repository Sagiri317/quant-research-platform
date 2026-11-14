"""Resolve explicit native tool inputs and validate Agent runtime configuration."""

import json
import os
import re
from pathlib import Path

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
    sanitize_shared_text,
)
from quant_research_platform.storage.datasets.reading import open_dataset_session

OPERATIONS = {
    "prepare",
    "assemble",
    "fit",
    "run-quarterly",
    "run-sessions",
    "sample-hft",
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


def resolve_inputs(value, inputs):
    """Resolve explicitly attached local inputs without exposing their paths."""
    if isinstance(value, str) and value.startswith("input:"):
        return inputs[value.removeprefix("input:")]
    if isinstance(value, dict):
        return {key: resolve_inputs(item, inputs) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_inputs(item, inputs) for item in value]
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
        {"question", "tools", "budget", "output", "llm"},
        {
            "sandbox",
            "documents",
            "memory",
            "web_search",
            "inputs",
            "history",
            "workspace",
        },
    )
    config["budget"] = validate_budget(config["budget"])
    public_tools = {}
    for name, template in config["tools"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z_0-9-]*", name):
            raise ValueError("agent_tool_name_invalid")
        template = _strict_config(
            template,
            {"operation", "description", "config", "parameters", "metrics"},
            {"bindings", "facts"},
        )
        if template["operation"] not in OPERATIONS:
            raise ValueError("agent_operation_not_allowed")
        config["tools"][name] = template
        public_tools[name] = {
            "operation": template["operation"],
            "description": template["description"],
            "config_json": sanitize_shared_text(
                json.dumps(template["config"], ensure_ascii=False)
            ),
            "parameters": template["parameters"],
            "inputs": template.get("bindings", {}),
            "metrics": list(template["metrics"]),
            "facts": list(template.get("facts", {})),
        }
    public = {
        "question": config["question"],
        "tools": public_tools,
        "budget": config["budget"],
        "sandbox_inputs": dataset_context(config.get("sandbox", {}).get("inputs", {})),
        "documents": list(config.get("documents", {})),
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
