"""Simple DAG workflow planner."""

from __future__ import annotations

from quant_research_platform.contracts.workflows import WorkflowNode, WorkflowSpec


def _dependency_map(
    workflow: WorkflowSpec | dict[str, tuple[str, ...]] | tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if isinstance(workflow, WorkflowSpec):
        return {node.node_id: node.dependencies for node in workflow.nodes}
    if isinstance(workflow, tuple):
        return dict.fromkeys(workflow, ())
    return workflow


def plan_topological_steps(
    workflow: WorkflowSpec | dict[str, tuple[str, ...]] | tuple[str, ...],
) -> tuple[str, ...]:
    dependencies = _dependency_map(workflow)

    ordered: list[str] = []
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in permanent:
            return
        if step_id in temporary:
            raise ValueError("workflow_cycle_detected")
        temporary.add(step_id)
        for dependency in dependencies.get(step_id, ()):
            if dependency not in dependencies:
                raise ValueError(f"unknown_workflow_dependency: {dependency}")
            visit(dependency)
        temporary.remove(step_id)
        permanent.add(step_id)
        ordered.append(step_id)

    for step_id in dependencies:
        visit(step_id)
    return tuple(ordered)


def plan_workflow_nodes(spec: WorkflowSpec) -> tuple[WorkflowNode, ...]:
    nodes_by_id = {node.node_id: node for node in spec.nodes}
    return tuple(nodes_by_id[node_id] for node_id in plan_topological_steps(spec))


__all__ = ["plan_topological_steps", "plan_workflow_nodes"]
