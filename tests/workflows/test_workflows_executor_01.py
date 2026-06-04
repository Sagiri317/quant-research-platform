from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import quant_research_platform.workflows.executor as executor_module
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowNode,
    WorkflowRunSummary,
    WorkflowSpec,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.executor import (
    execute_workflow,
    workflow_run_summary,
)
from quant_research_platform.workflows.registry import (
    WorkflowNodeDefinition,
    WorkflowNodeRegistry,
)
from quant_research_platform.workflows.services import execute_registered_workflow


def test_workflow_executor_runs_registered_nodes_in_topological_order() -> None:
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

    registry = WorkflowNodeRegistry()
    for node_id in spec.step_ids:
        registry.register(
            WorkflowNodeDefinition(node_id=node_id, handler=handler(node_id))
        )

    summary = execute_workflow(
        spec,
        registry=registry,
    )

    assert calls == ["panel", "factors", "labels", "dataset"]
    assert isinstance(summary, WorkflowRunSummary)
    assert summary.status == "executed"
    assert summary.completed_nodes == ("dataset", "factors", "labels", "panel")


def test_workflow_executor_records_memory_checkpoints_only() -> None:
    spec = WorkflowSpec(
        workflow_id="checkpoint_smoke",
        nodes=(
            WorkflowNode(node_id="panel"),
            WorkflowNode(node_id="factors", dependencies=("panel",)),
        ),
    )
    checkpoints = InMemoryCheckpointStore()
    registry = WorkflowNodeRegistry()
    registry.register(
        WorkflowNodeDefinition(node_id="panel", handler=lambda: "panel:done")
    )
    registry.register(
        WorkflowNodeDefinition(node_id="factors", handler=lambda: "factors:done")
    )

    execute_workflow(
        spec,
        registry=registry,
        checkpoints=checkpoints,
    )

    assert checkpoints.list() == (
        ("panel", "completed"),
        ("factors", "completed"),
    )
    assert checkpoints.get("panel").status == "completed"


def test_workflow_executor_rejects_unknown_spec_id() -> None:
    spec = WorkflowSpec(
        workflow_id="unknown_spec",
        input_spec_ids=("missing_spec",),
        nodes=(WorkflowNode(node_id="validate"),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_spec_id"):
        execute_workflow(
            spec,
            registry=WorkflowNodeRegistry.bootstrap_default(),
            allowed_spec_ids={"known_spec"},
        )


def test_workflow_executor_requires_output_spec_for_spec_driven_writer() -> None:
    spec = WorkflowSpec(
        workflow_id="missing_output_spec",
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="write"),),
    )

    with pytest.raises(ValueError, match="workflow_output_spec_required"):
        execute_workflow(spec, registry=WorkflowNodeRegistry.bootstrap_default())


def test_workflow_executor_does_not_accept_arbitrary_output_root() -> None:
    assert "output_root" not in inspect.signature(execute_workflow).parameters
    service_params = inspect.signature(execute_registered_workflow).parameters
    assert "output_root" not in service_params


def test_workflow_executor_rejects_arbitrary_handler_injection() -> None:
    assert "handlers" not in inspect.signature(execute_workflow).parameters
    assert "include_summary" not in inspect.signature(execute_workflow).parameters
    with pytest.raises(TypeError):
        execute_workflow(  # type: ignore[call-arg]
            WorkflowSpec(workflow_id="no_handlers"),
            handlers={"anything": lambda: "done"},
        )


def test_handler_map_execution_is_not_public_executor_surface() -> None:
    assert "execute_steps" not in executor_module.__all__
    assert not hasattr(executor_module, "execute_steps")


def test_workflow_summary_is_sanitized() -> None:
    spec = WorkflowSpec(
        workflow_id=r"C:\Users\DELL\private\workflow",
        market_id=MarketId("a_share"),
        output_spec_ids=("daily_panel",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="node"),),
    )

    summary = workflow_run_summary(
        spec,
        completed_nodes=("node",),
        execution_mode="execute",
        run_id="workflow-fixture-run",
    )

    assert isinstance(summary, WorkflowRunSummary)
    public_record = summary.to_public_record()
    assert "C:" not in str(public_record)
    assert "/mnt/" not in str(public_record)
    assert "000001" not in str(public_record)


def test_registered_workflow_rejects_unknown_node() -> None:
    spec = WorkflowSpec(
        workflow_id="unknown_node",
        output_spec_ids=("daily_panel",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="missing"),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_node"):
        execute_registered_workflow(
            spec,
            registry=WorkflowNodeRegistry(),
            allowed_spec_ids={"daily_panel"},
        )


def test_registered_workflow_returns_sanitized_summary_only() -> None:
    registry = WorkflowNodeRegistry()
    registry.register(
        WorkflowNodeDefinition(
            node_id="validate",
            handler=lambda: Path("/mnt/data/000001.SZ"),
        )
    )
    spec = WorkflowSpec(
        workflow_id="registered",
        market_id=MarketId("a_share"),
        output_spec_ids=("daily_panel",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="validate"),),
    )

    summary = execute_registered_workflow(
        spec,
        registry=registry,
        allowed_spec_ids={"daily_panel"},
    )

    assert summary["status"] == "executed"
    assert summary["execution_mode"] == "execute"
    assert "results" not in summary
    assert "C:" not in str(summary)
    assert "/mnt/" not in str(summary)
    assert "000001" not in str(summary)
