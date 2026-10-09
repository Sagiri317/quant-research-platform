"""Review-safe report and evidence helpers."""

from quant_research_platform.contracts.targets import RunStatus


def report_status(source_status, issues):
    if source_status is RunStatus.FAILED or any(
        issue.get("status") == "FAILED" for issue in issues
    ):
        return RunStatus.FAILED
    if source_status is RunStatus.BLOCKED or any(
        issue.get("status") == "BLOCKED" for issue in issues
    ):
        return RunStatus.BLOCKED
    return RunStatus.COMPLETE


def diagnostic_status(rows):
    states = {row["status"] for row in rows}
    if not states.issubset({"DEFINED", "UNDEFINED", "BLOCKED", "FAILED"}):
        raise ValueError("diagnostic_status_invalid")
    for state in ("FAILED", "BLOCKED", "UNDEFINED"):
        if state in states:
            return state
    return "DEFINED" if states else "UNDEFINED"
