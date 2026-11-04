"""Project native ledger valuations into NAV points for display.

Sampling retains whole transitions and the final endpoint; it does not recompute
returns or alter the underlying ledger."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator


def _ledger_curve(batches, *, stride):
    points = []
    total_points = 0
    last_points = []
    last_retained_sequence = -1
    with closing_iterator(iter(batches)) as batches:
        for batch in batches:
            last_points = [
                {
                    "sequence": batch.sequence,
                    "timestamp_ns": max(
                        row.timestamp_ns
                        for row in batch.valuations
                        if row.phase == phase
                    ),
                    "phase": phase,
                    "nav": sum(
                        row.value for row in batch.valuations if row.phase == phase
                    ),
                }
                for phase in ("before", "after", "end")
            ]
            total_points += len(last_points)
            if batch.sequence % stride == 0:
                points.extend(last_points)
                last_retained_sequence = batch.sequence
    if last_points and last_retained_sequence != last_points[-1]["sequence"]:
        points.extend(last_points)
    return {
        "status": "DEFINED",
        "observed_start_ns": points[0]["timestamp_ns"] if points else None,
        "observed_end_ns": last_points[-1]["timestamp_ns"] if last_points else None,
        "points": points,
        "source_point_count": total_points,
        "displayed_point_count": len(points),
        "stride": stride,
        "selection": "every_nth_ledger_transition_plus_last",
    }


def portfolio_curve(path: Path, *, stride: int = 1):
    """Project one native account NAV at a display stride."""
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )

    if type(stride) is not int or stride < 1:
        raise ValueError("portfolio_curve_stride_invalid")
    curve = {}
    artifact = read_portfolio_ledger_artifact(
        path,
        consume_batches=lambda _artifact, batches: curve.update(
            _ledger_curve(batches, stride=stride)
        ),
    )
    return {
        "source_reference": str(path),
        "window": artifact.metadata["window"],
        **(
            {"sample_role": artifact.metadata["sample_role"]}
            if "sample_role" in artifact.metadata
            else {}
        ),
        "selection_mode": artifact.metadata.get("selection_mode"),
        "valuation_method": artifact.capital.valuation_method,
        "kind": artifact.metadata.get("kind"),
        "candidate_id": artifact.capital.candidate_id,
        "valuation_currency": artifact.capital.valuation_currency,
        "initial_capital": artifact.capital.initial_capital,
        **curve,
    }
