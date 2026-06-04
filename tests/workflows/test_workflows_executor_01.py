from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowNode,
    WorkflowSpec,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.executor import (
    execute_steps,
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

    results = execute_workflow(
        spec,
        registry=registry,
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


def test_workflow_executor_rejects_private_output_to_public_root(tmp_path) -> None:
    spec = WorkflowSpec(
        workflow_id="private_output",
        output_spec_ids=("daily_panel",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="write"),),
    )

    with pytest.raises(ValueError, match="workflow_output_outside_private_boundary"):
        execute_workflow(
            spec,
            registry=WorkflowNodeRegistry.bootstrap_default(),
            output_root=tmp_path / "public",
        )


def test_workflow_executor_rejects_arbitrary_handler_injection() -> None:
    assert "handlers" not in execute_workflow.__annotations__
    with pytest.raises(TypeError):
        execute_workflow(  # type: ignore[call-arg]
            WorkflowSpec(workflow_id="no_handlers"),
            handlers={"anything": lambda: "done"},
        )


def test_execute_steps_internal_helper_still_requires_registered_handler_map() -> None:
    with pytest.raises(ValueError, match="missing_workflow_handler"):
        execute_steps(("unknown",), handlers={})


def test_workflow_summary_is_sanitized() -> None:
    spec = WorkflowSpec(
        workflow_id=r"C:\Users\DELL\private\workflow",
        market_id=MarketId("a_share"),
        output_spec_ids=("daily_panel",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="node"),),
    )

    summary = workflow_run_summary(spec, {"node": Path("/mnt/data/000001.SZ")})

    assert "C:" not in str(summary)
    assert "/mnt/" not in str(summary)
    assert "000001" not in str(summary)


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
