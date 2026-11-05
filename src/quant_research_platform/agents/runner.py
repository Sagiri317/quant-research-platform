"""Run budgeted research loops under a fixed experimental protocol.

The caller selects tools, allowed parameter choices, windows, and feedback
metrics. Each attempt retains its action and result; final-test observations
are excluded from iterative feedback.
"""

import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_research_platform.agents.client import (
    AgentAction,
    OpenAIResponsesClient,
)
from quant_research_platform.agents.execution import (
    _metrics,
    _resolve_artifacts,
    _run_process,
    run_isolated_python,
)
from quant_research_platform.contracts.json_codec import (
    _strict_config,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.storage.atomic import (
    atomic_write_json,
    reject_if_exists,
)
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.storage.datasets.reading import (
    iter_dataset_chunks,
    open_dataset_session,
)

OPERATIONS = {
    "assemble",
    "fit",
    "predict",
    "evaluate",
    "score",
    "select",
    "backtest",
    "backtest-hft",
    "calibrate",
    "statistics-factors",
    "statistics-model",
    "statistics-portfolio",
    "train-policy",
    "backtest-policy",
    "search-factors",
    "apply-formula",
    "calibrate-uncertainty",
    "apply-uncertainty",
    "evaluate-uncertainty",
    "fit-cost-model",
    "backtest-cost-model",
}
_TUNABLE_PATH = re.compile(
    r"^(?:model\.(?:model_algorithm|input_factor_columns|input_feature_columns|hyperparameters\.[A-Za-z_][A-Za-z_0-9]*)|candidate_parameters\.[A-Za-z_][A-Za-z_0-9]*|columns|expressions|feature_columns|baseline_columns|hidden_sizes|epochs|learning_rate|ppo\.[A-Za-z_][A-Za-z_0-9]*)$"
)


def _protocol_config(value, protocol):
    """Resolve explicit protocol references and reject conflicting scientific fields."""
    if isinstance(value, str) and value.startswith("protocol:"):
        selected = protocol
        for key in value.removeprefix("protocol:").split("."):
            selected = selected[key]
        return deepcopy(selected)
    if isinstance(value, list):
        return [_protocol_config(item, protocol) for item in value]
    if isinstance(value, dict):
        result = {key: _protocol_config(item, protocol) for key, item in value.items()}
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


def _set_argument(arguments, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        arguments = (
            arguments[int(part)] if isinstance(arguments, list) else arguments[part]
        )
    key = int(parts[-1]) if isinstance(arguments, list) else parts[-1]
    arguments[key] = value


def _before_holdout(value, holdout_start):
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
                }
                and pd.Timestamp(item) >= holdout_start
            ):
                raise ValueError("agent_tool_cutoff_reaches_holdout")
            _before_holdout(item, holdout_start)
    elif isinstance(value, list):
        for item in value:
            _before_holdout(item, holdout_start)


def run_agent(config, *, client=None):
    """Run a single research loop; failed attempts remain immutable local evidence."""
    config = _strict_config(
        config,
        {"question", "protocol", "tools", "budget", "output", "llm"},
        {"sandbox"},
    )
    protocol = _strict_config(
        config["protocol"],
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
    train_end = pd.Timestamp(train_end)
    val_start, val_end, val_closed = time_window_timestamp_interval(
        TimeWindow(**protocol["validation_window"])
    )
    if (
        not train_end <= pd.Timestamp(val_start) <= pd.Timestamp(val_end) <= test_start
        or train_closed
        and train_end == pd.Timestamp(val_start)
        or val_closed
        and pd.Timestamp(val_end) == test_start
    ):
        raise ValueError("agent_protocol_windows_overlap")
    budget = _strict_config(
        config["budget"], {"max_requests", "max_experiments", "wall_seconds"}, set()
    )
    if any(type(value) is not int or value < 1 for value in budget.values()):
        raise ValueError("agent_budget_invalid")
    public_tools, templates = {}, {}
    for name, template in config["tools"].items():
        template = _strict_config(
            template, {"operation", "config", "vary", "metrics"}, {"bindings"}
        )
        if template["operation"] not in OPERATIONS:
            raise ValueError("agent_operation_not_allowed")
        template["config"] = _protocol_config(template["config"], protocol)
        _before_holdout(template["config"], test_start)
        templates[name] = template
        for path, choices in template["vary"].items():
            if (
                not _TUNABLE_PATH.fullmatch(path)
                or not isinstance(choices, list)
                or not choices
            ):
                raise ValueError("agent_tunable_not_allowed")
        public_tools[name] = {
            "operation": template["operation"],
            "choices": template["vary"],
            "inputs": list(template.get("bindings", {})),
        }
    sandbox_inputs = {}
    if "sandbox" in config:
        for alias, root in config["sandbox"]["inputs"].items():
            inspection = inspect_dataset(Path(root))
            if (
                inspection.date_max is None
                or pd.Timestamp(inspection.date_max) >= test_start
            ):
                raise ValueError("agent_sandbox_input_reaches_holdout")
            clocks = inspection.spec.get("label_availability_columns", {})
            labels = [
                name
                for name, role in inspection.spec["column_roles"].items()
                if role == "label"
            ]
            if set(labels) != set(clocks):
                raise ValueError("agent_sandbox_label_maturity_required")
            if labels:
                session = open_dataset_session(
                    Path(root),
                    columns=tuple(dict.fromkeys([*labels, *clocks.values()])),
                )
                with closing_iterator(iter_dataset_chunks(session)) as chunks:
                    for frame in chunks:
                        for label, clock in clocks.items():
                            available = pd.to_datetime(frame[clock], errors="raise")
                            if (
                                frame[label].notna()
                                & (available.isna() | available.ge(test_start))
                            ).any():
                                raise ValueError("agent_sandbox_labels_reach_holdout")
            sandbox_inputs[alias] = {
                "columns": inspection.spec["column_roles"],
                "units": inspection.spec.get("column_units", {}),
                "frequency": inspection.spec["frequency"],
                "rows": inspection.row_count,
            }
    public = {
        "question": config["question"],
        "protocol": protocol,
        "tools": public_tools,
        "budget": budget,
        "sandbox_inputs": sandbox_inputs,
    }
    if evidence_json_value_unsafe_reason(public):
        raise ValueError("agent_public_context_not_shareable")
    output = Path(config["output"])
    reject_if_exists(output)
    output.mkdir(parents=True)
    client = OpenAIResponsesClient(config["llm"]) if client is None else client
    messages = [
        {
            "role": "system",
            "content": (
                "You are a quantitative research agent. Fix the supplied label, "
                "windows, costs and metric. Explore only listed parameter choices. "
                "Never use the final test period for feedback. Return exactly one "
                "JSON action. Use run_tool with a tool name. Each change selects a "
                "parameter path and a zero-based choice index from that tool's "
                "choices. Leave unchanged parameters out of changes. Each input "
                "binds a declared path to a successful evidence ID such as e0001. "
                "Use empty lists and null for unused fields. Generated Python "
                "executes only in a networkless container with /inputs/<alias> and "
                "/outputs. Only supplied numerical summaries constitute evidence. "
                "Cite evidence IDs in a finish action. Negative and undefined "
                "results are valid. Do not claim unrun experiments succeeded."
            ),
        },
        {"role": "user", "content": json.dumps(public, ensure_ascii=False)},
    ]
    deadline = time.monotonic() + budget["wall_seconds"]
    artifacts, records = {}, []
    status, conclusion = "budget_exhausted", None
    try:
        for request in range(budget["max_requests"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            action = client.complete(messages, timeout=remaining)
            if time.monotonic() >= deadline:
                break
            action = (
                action
                if isinstance(action, AgentAction)
                else AgentAction.model_validate(action)
            )
            atomic_write_json(
                output / f"action-{request + 1:04d}.json", action.model_dump()
            )
            if action.kind == "finish":
                if not action.conclusion or any(
                    item not in artifacts for item in action.evidence
                ):
                    raise ValueError("agent_conclusion_evidence_invalid")
                conclusion = {"text": action.conclusion, "evidence": action.evidence}
                status = "completed"
                break
            if len(records) >= budget["max_experiments"]:
                break
            identifier = f"e{len(records) + 1:04d}"
            trial_root = output / identifier
            recorder = TrialRecorder(
                trial_root / "trial", {"kind": action.kind, "protocol": protocol}
            )
            recorder.begin()
            result = {"id": identifier}
            try:
                if action.kind == "run_python":
                    if "sandbox" not in config:
                        raise ValueError("agent_sandbox_not_configured")
                    report = run_isolated_python(
                        action.code,
                        config["sandbox"],
                        trial_root / "code",
                        timeout=max(0.001, deadline - time.monotonic()),
                    )
                    result.update(report)
                    artifact = trial_root / "code" / "results"
                    if report["status"] == "completed":
                        result["metrics"] = _metrics(
                            artifact, config["sandbox"].get("metrics", {})
                        )
                else:
                    template = templates[action.tool]
                    arguments = deepcopy(template["config"])
                    for change in action.changes:
                        if change.path not in template["vary"] or change.choice >= len(
                            template["vary"][change.path]
                        ):
                            raise ValueError("agent_change_outside_declared_choices")
                        _set_argument(
                            arguments,
                            change.path,
                            template["vary"][change.path][change.choice],
                        )
                    arguments = _protocol_config(arguments, protocol)
                    bindings = template.get("bindings", {})
                    if {item.path for item in action.inputs} != set(bindings):
                        raise ValueError("agent_input_bindings_mismatch")
                    for item in action.inputs:
                        path, prior_identifier = item.path, item.evidence
                        member = bindings[path]
                        allow_generated = path in {"dataset", "inputs"} and template[
                            "operation"
                        ] in {
                            "fit",
                            "predict",
                            "evaluate",
                            "search-factors",
                            "fit-cost-model",
                            "statistics-factors",
                            "statistics-model",
                        }
                        resolved = _resolve_artifacts(
                            f"artifact:{prior_identifier}/{member}",
                            artifacts,
                            allow_generated=allow_generated,
                        )
                        _set_argument(arguments, path, resolved)
                    _before_holdout(arguments, test_start)
                    arguments = _resolve_artifacts(arguments, artifacts)
                    artifact = trial_root / (
                        "result.json"
                        if template["operation"]
                        in {
                            "score",
                            "select",
                            "calibrate",
                            "search-factors",
                            "calibrate-uncertainty",
                            "evaluate-uncertainty",
                        }
                        else "result"
                    )
                    output_key = (
                        "output_root"
                        if template["operation"] == "assemble"
                        else "output"
                    )
                    arguments[output_key] = str(artifact.resolve())
                    arguments_file = atomic_write_json(
                        trial_root / "config.json", arguments
                    )
                    exit_code = _run_process(
                        [
                            sys.executable,
                            "-m",
                            "quant_research_platform.interfaces.cli",
                            template["operation"],
                            "--config",
                            str(arguments_file.resolve()),
                        ],
                        timeout=max(0.001, deadline - time.monotonic()),
                        cwd=Path.cwd(),
                        env={
                            **os.environ,
                            "PYTHONPATH": os.pathsep.join(
                                filter(
                                    None,
                                    (
                                        str(Path(__file__).resolve().parents[2]),
                                        os.environ.get("PYTHONPATH"),
                                    ),
                                )
                            ),
                        },
                        log=trial_root / "private.log",
                    )
                    result.update(
                        exit_code=exit_code,
                        status="completed" if exit_code == 0 else "failed",
                    )
                    if exit_code == 0:
                        result["metrics"] = _metrics(artifact, template["metrics"])
                if result["status"] == "completed":
                    artifacts[identifier] = {
                        "path": artifact,
                        "trusted": action.kind == "run_tool",
                    }
                    recorder.finish(result=result)
                else:
                    recorder.finish(
                        error=RuntimeError("experiment_failed"), result=result
                    )
            except (Exception, KeyboardInterrupt) as error:
                recorder.finish(error=error)
                result.update(status="failed", error=type(error).__name__)
                if isinstance(error, KeyboardInterrupt):
                    raise
            records.append(result)
            if evidence_json_value_unsafe_reason(result):
                raise ValueError("agent_tool_summary_not_shareable")
            messages.extend(
                (
                    {
                        "role": "assistant",
                        "content": json.dumps(action.model_dump(), ensure_ascii=False),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            result, ensure_ascii=False, allow_nan=False
                        ),
                    },
                )
            )
        atomic_write_json(
            output / "report.json",
            {
                "status": status,
                "protocol": protocol,
                "experiments": records,
                "conclusion": conclusion,
            },
        )
    except BaseException as error:
        atomic_write_json(
            output / "report.json",
            {
                "status": "failed",
                "error": type(error).__name__,
                "protocol": protocol,
                "experiments": records,
            },
        )
        raise
    return output
