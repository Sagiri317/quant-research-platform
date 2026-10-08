"""Review-safe report and evidence helpers."""

from collections.abc import Mapping
from html import escape

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


def render_research_report(payload: dict[str, object], *, sections, title) -> str:
    lines = [
        f"# {title}",
        "",
        f"Run: {payload['run_id']}",
        f"Scenario: {payload['scenario_id']}",
        f"Access: {payload['access_window']}",
        f"Source run status: {payload['source_run_status']}",
        f"Report status: {payload['report_status']}",
        "",
        "完整机器证据：[report.json](report.json)。"
        "表中的 — 表示未提供值；状态与原因见对应列。",
        "",
    ]
    content = payload["sections"]
    if type(content) is not dict:
        raise ValueError("research_report_payload_invalid")
    for name in sections:
        lines.extend((f"## {_markdown_cell(name)}", ""))
        _markdown_content(lines, content[name], depth=3)
    return "\n".join(lines)


def _markdown_cell(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{value:,}"
    return escape(str(value), quote=False).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(lines, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        return
    lines.extend(
        (
            "| " + " | ".join((_markdown_cell(key) for key in columns)) + " |",
            "| " + " | ".join("---" for _key in columns) + " |",
        )
    )
    lines.extend(
        "| " + " | ".join(_markdown_cell(row.get(key)) for key in columns) + " |"
        for row in rows
    )
    lines.append("")


def _markdown_content(lines, value, *, depth):
    if isinstance(value, Mapping):
        scalars = [
            {"指标": key, "值": item}
            for key, item in value.items()
            if not isinstance(item, (Mapping, list, tuple))
        ]
        _markdown_table(lines, scalars)
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)):
                lines.extend((f"{'#' * min(depth, 6)} {_markdown_cell(key)}", ""))
                _markdown_content(lines, item, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        if not value:
            lines.extend(("无记录。", ""))
        elif all(isinstance(item, Mapping) for item in value):
            _markdown_table(
                lines,
                [
                    {
                        key: item
                        for key, item in row.items()
                        if not isinstance(item, (Mapping, list, tuple))
                    }
                    for row in value
                ],
            )
            for index, row in enumerate(value, start=1):
                label = next(
                    (
                        row[key]
                        for key in ("candidate_id", "model_id", "target_id", "fold_id")
                        if key in row
                    ),
                    str(index),
                )
                for key, item in row.items():
                    if isinstance(item, (Mapping, list, tuple)):
                        lines.extend(
                            (
                                f"{'#' * min(depth, 6)} {_markdown_cell(label)} / "
                                f"{_markdown_cell(key)}",
                                "",
                            )
                        )
                        _markdown_content(lines, item, depth=depth + 1)
        else:
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    _markdown_content(lines, item, depth=depth + 1)
                else:
                    lines.append(f"- {_markdown_cell(item)}")
            lines.append("")
    else:
        lines.extend((_markdown_cell(value), ""))
