"""Deterministic model driver and explicit research configs for harness tests."""

import json

from quant_research_platform.agents.client import AgentAction
from tests.helpers.learning import learning_panel


class ScriptedClient:
    """Local action driver for controller tests; it is not an external model service."""

    def __init__(self, actions):
        self.actions = iter(actions)
        self.messages = []

    def complete(self, messages, *, timeout):
        assert timeout > 0
        self.messages.append(json.dumps(messages))
        return AgentAction.model_validate(next(self.actions))


def agent_config(tmp_path):
    learning_panel(tmp_path / "input")
    protocol = {
        "label": "label",
        "train_window": {"start": "2020-01-01", "end": "2020-01-03"},
        "validation_window": {"start": "2020-01-05", "end": "2020-01-07"},
        "test_window": {"start": "2020-01-20", "end": "2020-01-30"},
        "costs": {},
        "metric": "validation_rank_ic",
    }
    search = {
        "dataset": str(tmp_path / "input"),
        "columns": ["x"],
        "baseline_columns": [],
        "label": "protocol:label",
        "train_window": "protocol:train_window",
        "validation_window": "protocol:validation_window",
        "train_cutoff": "2020-01-04",
        "selection_cutoff": "2020-01-08",
        "method": "random",
        "budget": 3,
        "max_depth": 1,
        "windows": [2],
        "seed": 1,
        "output": "assigned_by_runner",
    }
    tools = {
        "search": {
            "operation": "search-factors",
            "description": "Evaluate proposed causal formulas against a baseline.",
            "config": search,
            "parameters": {"expressions": "List of causal formula strings."},
            "metrics": {
                "score": {
                    "member": "",
                    "column": "validation_score",
                    "reduction": "mean",
                }
            },
        },
        "apply": {
            "operation": "apply-formula",
            "description": "Apply a frozen search result to later development rows.",
            "config": {
                "search": "bound_by_runner",
                "dataset": str(tmp_path / "input"),
                "input_window": {"start": "2020-01-08", "end": "2020-01-12"},
                "output_window": {"start": "2020-01-10", "end": "2020-01-12"},
                "output": "assigned_by_runner",
            },
            "parameters": {},
            "metrics": {},
            "bindings": {"search": ""},
        },
    }
    return {
        "question": "Compare declared causal formulas.",
        "protocol": protocol,
        "tools": tools,
        "budget": {"max_requests": 4, "max_experiments": 3, "wall_seconds": 60},
        "output": str(tmp_path / "agent"),
        "llm": {},
    }
