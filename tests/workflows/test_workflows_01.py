from __future__ import annotations

import pytest

from quant_research_platform.contracts.workflows import WorkflowNode, WorkflowSpec
from quant_research_platform.workflows.planner import plan_topological_steps


def test_workflow_planner_orders_contract_spec_dependencies() -> None:
    spec = WorkflowSpec(
        workflow_id="research_dataset_smoke",
        nodes=(
            WorkflowNode(node_id="factors", dependencies=("panel",)),
            WorkflowNode(node_id="labels", dependencies=("panel",)),
            WorkflowNode(node_id="dataset", dependencies=("factors", "labels")),
            WorkflowNode(node_id="panel"),
        ),
    )

    assert plan_topological_steps(spec) == ("panel", "factors", "labels", "dataset")


def test_workflow_planner_rejects_unknown_dependency() -> None:
    spec = WorkflowSpec(
        workflow_id="bad_dependency",
        nodes=(WorkflowNode(node_id="dataset", dependencies=("missing_panel",)),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_dependency: missing_panel"):
        plan_topological_steps(spec)


def test_workflow_planner_rejects_cycles() -> None:
    spec = WorkflowSpec(
        workflow_id="cycle",
        nodes=(
            WorkflowNode(node_id="panel", dependencies=("dataset",)),
            WorkflowNode(node_id="dataset", dependencies=("panel",)),
        ),
    )

    with pytest.raises(ValueError, match="workflow_cycle_detected"):
        plan_topological_steps(spec)
