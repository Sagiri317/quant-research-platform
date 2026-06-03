from __future__ import annotations

import pytest

from quant_research_platform.contracts.workflows import WorkflowNode, WorkflowSpec
from quant_research_platform.markets.calendars import ChunkPlanner
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


def test_chunk_planner_uses_trading_sessions_not_calendar_days() -> None:
    chunks = ChunkPlanner.from_sessions(
        ["2014-01-03", "2014-01-06", "2014-01-07"],
        chunk_sessions=2,
    ).chunks()

    assert [
        (str(chunk.core_start.date()), str(chunk.core_end.date()))
        for chunk in chunks
    ] == [
        ("2014-01-03", "2014-01-06"),
        ("2014-01-07", "2014-01-07"),
    ]


def test_chunk_planner_supports_holiday_holes_and_windows() -> None:
    chunks = ChunkPlanner.from_sessions(
        [
            "2014-01-02",
            "2014-01-03",
            "2014-01-06",
            "2014-01-10",
            "2014-01-13",
        ],
        chunk_sessions=2,
        warmup_sessions=1,
        lookahead_sessions=1,
    ).chunks()

    assert str(chunks[0].core_start.date()) == "2014-01-02"
    assert str(chunks[0].read_start.date()) == "2014-01-02"
    assert str(chunks[0].read_end.date()) == "2014-01-06"
    assert str(chunks[1].core_start.date()) == "2014-01-06"
    assert str(chunks[1].read_start.date()) == "2014-01-03"
    assert str(chunks[1].read_end.date()) == "2014-01-13"
    assert str(chunks[-1].core_end.date()) == "2014-01-13"
    assert str(chunks[-1].read_end.date()) == "2014-01-13"


def test_chunk_planner_sorts_and_deduplicates_sessions() -> None:
    planner = ChunkPlanner.from_sessions(
        ["2014-01-06", "2014-01-03", "2014-01-03"],
        chunk_sessions=1,
    )

    assert [str(value.date()) for value in planner.sessions] == [
        "2014-01-03",
        "2014-01-06",
    ]
