from __future__ import annotations

from quant_research_platform.contracts.workflows import WorkflowNode, WorkflowSpec
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.executor import execute_workflow


def test_workflow_executor_runs_passed_handlers_in_topological_order() -> None:
    spec = WorkflowSpec(
        workflow_id="executor_smoke",
        nodes=(
            WorkflowNode(node_id="dataset", dependencies=("factors", "labels")),
            WorkflowNode(node_id="factors", dependencies=("panel",)),
            WorkflowNode(node_id="labels", dependencies=("panel",)),
            WorkflowNode(node_id="panel"),
        ),
    )
    calls: list[str] = []

    def handler(node_id: str) -> object:
        def run() -> str:
            calls.append(node_id)
            return f"{node_id}:done"

        return run

    results = execute_workflow(
        spec,
        handlers={node_id: handler(node_id) for node_id in spec.step_ids},
    )

    assert calls == ["panel", "factors", "labels", "dataset"]
    assert results == {
        "panel": "panel:done",
        "factors": "factors:done",
        "labels": "labels:done",
        "dataset": "dataset:done",
    }


def test_workflow_executor_records_memory_checkpoints_only() -> None:
    spec = WorkflowSpec(
        workflow_id="checkpoint_smoke",
        nodes=(
            WorkflowNode(node_id="panel"),
            WorkflowNode(node_id="factors", dependencies=("panel",)),
        ),
    )
    checkpoints = InMemoryCheckpointStore()

    execute_workflow(
        spec,
        handlers={
            "panel": lambda: "panel:done",
            "factors": lambda: "factors:done",
        },
        checkpoints=checkpoints,
    )

    assert checkpoints.list() == (
        ("panel", "completed"),
        ("factors", "completed"),
    )
    assert checkpoints.get("panel").status == "completed"
