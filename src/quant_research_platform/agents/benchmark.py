"""Paired, source-backed comparisons of research-agent experiment results.

This is an analysis of existing evidence, not an alternate evaluator or
backtester. The native research operation remains the owner of each metric.
"""

from math import isfinite
from pathlib import Path
from statistics import mean, stdev

from quant_research_platform.agents.session import read_session, usage_summary
from quant_research_platform.contracts.json_codec import _strict_config, parse_qrp_json_object


def _score(run, metric):
    field = run.get("field", metric)
    if not isinstance(field, str) or not field:
        raise ValueError("agent_benchmark_metric_invalid")
    session = run.get("session")
    artifact = run.get("artifact")
    if (session is None) == (artifact is None):
        raise ValueError("agent_benchmark_source_ambiguous")
    if session is not None:
        if not isinstance(session, str) or not isinstance(run.get("evidence"), str):
            raise ValueError("agent_benchmark_evidence_missing")
        state = read_session(Path(session))
        event = next(
            (event for event in state["events"] if event["id"] == run["evidence"]),
            None,
        )
        if event is None or event["result"]["status"] != "completed":
            raise ValueError("agent_benchmark_evidence_not_completed")
        metrics = event["result"].get("metrics", {})
        if field not in metrics:
            raise ValueError("agent_benchmark_metric_missing")
        value = metrics[field]
        usage = usage_summary(state)
        source = {"session": session, "evidence": run["evidence"]}
    else:
        if not isinstance(artifact, str):
            raise ValueError("agent_benchmark_artifact_invalid")
        result = parse_qrp_json_object(Path(artifact).read_bytes())
        value = result
        for part in field.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ValueError("agent_benchmark_metric_missing")
            value = value[part]
        usage = None
        source = {"artifact": artifact}
    if value is not None and (
        type(value) not in (int, float) or not isfinite(value)
    ):
        raise ValueError("agent_benchmark_metric_not_finite")
    return value, usage, source


def compare_research_runs(config):
    """Compare arms on paired task/protocol/seed, retaining missing outcomes.

    Each run must point to a persisted native metric or completed Agent
    evidence. Results are descriptive; no claim of statistical significance or
    test-period independence follows from a successful comparison.
    """
    config = _strict_config(
        config, {"metric", "baseline_arm", "runs"}, set()
    )
    metric, baseline = config["metric"], config["baseline_arm"]
    if not isinstance(metric, str) or not metric:
        raise ValueError("agent_benchmark_metric_invalid")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("agent_benchmark_baseline_invalid")
    if not isinstance(config["runs"], list) or not config["runs"]:
        raise ValueError("agent_benchmark_runs_invalid")
    records = []
    seen = set()
    for raw in config["runs"]:
        run = _strict_config(
            raw,
            {"task", "protocol", "seed", "arm"},
            {"session", "evidence", "artifact", "field"},
        )
        if any(
            not isinstance(run[key], str) or not run[key].strip()
            for key in ("task", "protocol", "arm")
        ) or type(run["seed"]) is not int:
            raise ValueError("agent_benchmark_run_invalid")
        identity = (run["task"], run["protocol"], run["seed"], run["arm"])
        if identity in seen:
            raise ValueError("agent_benchmark_duplicate_run")
        seen.add(identity)
        value, usage, source = _score(run, metric)
        records.append(
            {
                "task": run["task"],
                "protocol": run["protocol"],
                "seed": run["seed"],
                "arm": run["arm"],
                "score": value,
                "usage": usage,
                "source": source,
            }
        )
    if baseline not in {row["arm"] for row in records}:
        raise ValueError("agent_benchmark_baseline_missing")
    by_identity = {
        (row["task"], row["protocol"], row["seed"], row["arm"]): row
        for row in records
    }
    arms = sorted({row["arm"] for row in records})
    summaries = {}
    for arm in arms:
        rows = [row for row in records if row["arm"] == arm]
        values = [row["score"] for row in rows if row["score"] is not None]
        summaries[arm] = {
            "runs": len(rows),
            "defined_runs": len(values),
            "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else None,
        }
    comparisons = {}
    for arm in arms:
        if arm == baseline:
            continue
        pairs, unpaired = [], 0
        for row in records:
            if row["arm"] != arm:
                continue
            other = by_identity.get(
                (row["task"], row["protocol"], row["seed"], baseline)
            )
            if (
                other is None
                or row["score"] is None
                or other["score"] is None
            ):
                unpaired += 1
                continue
            pairs.append(
                {
                    "task": row["task"],
                    "protocol": row["protocol"],
                    "seed": row["seed"],
                    "difference": row["score"] - other["score"],
                }
            )
        comparisons[arm] = {
            "baseline": baseline,
            "paired_runs": len(pairs),
            "unpaired_runs": unpaired,
            "mean_difference": mean(p["difference"] for p in pairs)
            if pairs else None,
            "pairs": pairs,
        }
    return {
        "metric": metric,
        "baseline_arm": baseline,
        "runs": records,
        "summary": summaries,
        "paired_comparisons": comparisons,
        "limitations": (
            "Protocol IDs are caller-declared; verify input data, label maturity, "
            "research budget, selection history, and OOS status independently. "
            "Differences are descriptive, not significance tests."
        ),
    }
