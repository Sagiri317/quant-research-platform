"""Project native ledger valuations into NAV points for display.

Sampling retains whole transitions and the final endpoint; it does not recompute
returns or alter the underlying ledger."""

from __future__ import annotations

from itertools import groupby
from operator import itemgetter
from pathlib import Path

from quant_research_platform.policy.lifecycle import closing_iterator


def _nav_points(sequence, rows):
    return [
        {
            "sequence": sequence,
            "timestamp_ns": max(
                row["timestamp_ns"] for row in rows if row["phase"] == phase
            ),
            "phase": phase,
            "nav": sum(row["value"] for row in rows if row["phase"] == phase),
        }
        for phase in ("before", "after", "end")
    ]


def _ledger_curve(records, *, stride):
    points = []
    total_points = 0
    last_sequence = None
    last_rows = []
    with closing_iterator(records):
        for sequence, group in groupby(records, key=itemgetter("sequence")):
            last_sequence, last_rows = sequence, list(group)
            total_points += 3
            if sequence % stride == 0:
                points.extend(_nav_points(sequence, last_rows))
    if last_sequence is not None and last_sequence % stride != 0:
        points.extend(_nav_points(last_sequence, last_rows))
    return {
        "status": "DEFINED",
        "observed_start_ns": points[0]["timestamp_ns"] if points else None,
        "observed_end_ns": points[-1]["timestamp_ns"] if points else None,
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
    artifact = read_portfolio_ledger_artifact(path)
    curve = _ledger_curve(artifact.iter_nav_rows(), stride=stride)
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
