from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import quant_research_platform.workflows.executor as executor_module
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowNode,
    WorkflowRunPolicy,
    WorkflowRunSummary,
    WorkflowSpec,
)
from quant_research_platform.runtime.workflow_records import (
    workflow_run_summary_to_public_record,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.executor import (
    run_workflow,
    workflow_run_summary,
)
from quant_research_platform.workflows.registry import (
    WorkflowNodeDefinition,
    WorkflowNodeRegistry,
    workflow_node_registry,
)
from quant_research_platform.workflows.services import run_registered_workflow


def test_workflow_node_registry_factory_supports_validate_only_modes() -> None:
    for run_policy in (None, "validate_only", WorkflowRunPolicy.VALIDATE_ONLY):
        registry = workflow_node_registry(run_policy=run_policy)

        assert isinstance(registry, WorkflowNodeRegistry)
        assert (
            registry.get("validate_inputs").handler()
            == "validate_inputs:validated"
        )


def test_workflow_node_registry_factory_write_artifacts_fails_closed() -> None:
    for run_policy in ("write_artifacts", WorkflowRunPolicy.WRITE_ARTIFACTS):
        with pytest.raises(
            NotImplementedError,
            match="workflow_write_artifacts_not_implemented",
        ):
            workflow_node_registry(run_policy=run_policy)


def test_workflow_runner_default_run_policy_validates_only() -> None:
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

    summary = run_workflow(
        spec,
        registry=registry,
    )

    assert calls == []
    assert isinstance(summary, WorkflowRunSummary)
    assert summary.run_policy is WorkflowRunPolicy.VALIDATE_ONLY
    assert summary.status == "validated"
    assert summary.completed_nodes == ()


def test_workflow_runner_records_memory_checkpoints_only() -> None:
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

    run_workflow(
        spec,
        registry=registry,
        checkpoints=checkpoints,
        run_policy="validate_only",
    )

    assert checkpoints.list() == ()


def test_workflow_validate_only_does_not_call_handlers_or_write_checkpoints() -> None:
    spec = WorkflowSpec(
        workflow_id="validate_only_no_side_effects",
        nodes=(WorkflowNode(node_id="panel"),),
    )
    calls: list[str] = []
    checkpoints = InMemoryCheckpointStore()
    registry = WorkflowNodeRegistry()
    registry.register(
        WorkflowNodeDefinition(
            node_id="panel",
            handler=lambda: calls.append("panel"),
        )
    )

    validate_only_summary = run_workflow(
        spec,
        registry=registry,
        checkpoints=checkpoints,
        run_policy="validate_only",
    )
    with pytest.raises(
        NotImplementedError,
        match="workflow_write_artifacts_not_implemented",
    ):
        run_workflow(
            spec,
            registry=registry,
            run_policy="write_artifacts",
        )

    assert calls == []
    assert checkpoints.list() == ()
    assert validate_only_summary.status == "validated"
    assert validate_only_summary.completed_nodes == ()


def test_workflow_validate_only_rejects_unknown_registered_node() -> None:
    spec = WorkflowSpec(
        workflow_id="validate_only_missing_node",
        nodes=(WorkflowNode(node_id="missing"),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_node"):
        run_workflow(
            spec,
            registry=WorkflowNodeRegistry(),
            run_policy="validate_only",
        )


def test_workflow_runner_rejects_unknown_spec_id() -> None:
    spec = WorkflowSpec(
        workflow_id="unknown_spec",
        input_spec_ids=("missing_spec",),
        nodes=(WorkflowNode(node_id="validate"),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_spec_id"):
        run_workflow(
            spec,
            registry=workflow_node_registry(),
            allowed_spec_ids={"known_spec"},
        )


def test_workflow_runner_requires_output_spec_for_spec_driven_writer() -> None:
    spec = WorkflowSpec(
        workflow_id="missing_output_spec",
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="write"),),
    )

    with pytest.raises(ValueError, match="workflow_output_spec_required"):
        run_workflow(spec, registry=workflow_node_registry())


def test_workflow_runner_does_not_accept_arbitrary_output_root() -> None:
    assert "output_root" not in inspect.signature(run_workflow).parameters
    service_params = inspect.signature(run_registered_workflow).parameters
    assert "output_root" not in service_params


def test_workflow_runner_rejects_arbitrary_handler_injection() -> None:
    assert "handlers" not in inspect.signature(run_workflow).parameters
    assert "include_summary" not in inspect.signature(run_workflow).parameters
    assert not hasattr(WorkflowNodeRegistry, "handlers_for")
    with pytest.raises(TypeError):
        run_workflow(  # type: ignore[call-arg]
            WorkflowSpec(workflow_id="no_handlers"),
            handlers={"anything": lambda: "done"},
        )


def test_handler_map_runner_is_not_public_workflow_surface() -> None:
    assert "run_steps" not in executor_module.__all__
    assert not hasattr(executor_module, "run_steps")


def test_workflow_summary_is_sanitized() -> None:
    spec = WorkflowSpec(
        workflow_id="vendor-source-marker-workflow",
        market_id=MarketId("a_share"),
        output_spec_ids=("market_panel_1d",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="node"),),
    )

    summary = workflow_run_summary(
        spec,
        completed_nodes=("node",),
        run_policy="write_artifacts",
        run_id="workflow-validation-run",
    )

    assert isinstance(summary, WorkflowRunSummary)
    public_record = workflow_run_summary_to_public_record(summary)
    assert "vendor" not in str(public_record).lower()
    assert "source" not in str(public_record).lower()


def test_registered_workflow_rejects_unknown_node() -> None:
    spec = WorkflowSpec(
        workflow_id="unknown_node",
        output_spec_ids=("market_panel_1d",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="missing"),),
    )

    with pytest.raises(ValueError, match="unknown_workflow_node"):
        run_registered_workflow(
            spec,
            registry=WorkflowNodeRegistry(),
            allowed_spec_ids={"market_panel_1d"},
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
        output_spec_ids=("market_panel_1d",),
        artifact_writer_policy=WorkflowArtifactWriterPolicy.SPEC_DRIVEN,
        nodes=(WorkflowNode(node_id="validate"),),
    )

    summary = run_registered_workflow(
        spec,
        registry=registry,
        allowed_spec_ids={"market_panel_1d"},
    )

    assert summary["status"] == "validated"
    assert summary["run_policy"] == "validate_only"
    assert "results" not in summary
    assert "C:" not in str(summary)
    assert "/mnt/" not in str(summary)
    assert "000001" not in str(summary)
