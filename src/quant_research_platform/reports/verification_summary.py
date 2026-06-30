"""Local verification run summary renderers."""

from __future__ import annotations

from typing import Any

from quant_research_platform.policy.share_safety import evidence_text_unsafe_reason


def render_run_summary_markdown(payload: dict[str, Any]) -> str:
    status = payload["status"]
    lines = [
        "# Local Verification Run Summary",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- overall: `{status['overall']}`",
        f"- validation: `{status['validation']}`",
        f"- smoke: `{status['smoke']}`",
        f"- compare: `{status['compare']}`",
        f"- share_safety: `{status['share_safety']}`",
        f"- full_horizon: `{status['full_horizon']}`",
        f"- validation_matrix: `{status['validation_matrix']}`",
        f"- data_layers: `{len(payload['data']['layers'])}`",
        f"- outputs: `{len(payload['outputs'])}`",
        "",
    ]
    return "\n".join(lines)


def render_review_safe_aggregate_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the review-safe aggregate evidence payload as Markdown."""

    lines = [
        "# QRP Review-Safe Aggregate Evidence",
        "",
        (
            "This evidence records review-safe aggregate validation for the "
            "current QRP research contract. Review identifiers are "
            "`artifact_id=review_safe_aggregate_evidence`, "
            "`market_id=a_share`, and "
            "`schema_id=qrp-a-share-aggregate-evidence`."
        ),
        "",
        (
            "Machine-readable run facts are recorded in run.json by the "
            "current local verification run."
        ),
        "",
        "| Area | Rows | Columns | Period | Status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for area in payload["areas"]:
        status = (
            "human approval required"
            if area["human_approval_required"]
            else "verified aggregate"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(area["area"]).replace("_", " "),
                    str(area["row_count"]),
                    str(area["column_count"]),
                    str(area["period"]),
                    status,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Review-safe status:",
            "",
            f"- `output_scope | {payload['output_scope']}`",
            f"- `schema_id | {payload['schema_id']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `verification_scope | {payload['verification_scope']}`",
            f"- `evidence_generation_mode | {payload['evidence_generation_mode']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `handoff_ready | {str(payload['handoff_ready']).lower()}`",
            ("- local aggregate run completed without committed row-level outputs"),
        ]
    )
    if not payload["handoff_ready"]:
        lines.append("- human approval required before final review status")
    return _render_review_safe_markdown(lines)


def render_local_verification_inventory_markdown(payload: dict[str, Any]) -> str:
    """Render the local dataset inventory summary as review-safe Markdown."""

    lines = [
        "# QRP Local Verification Inventory",
        "",
        (
            "This summary records aggregate-only sealed dataset inventory "
            "for the current QRP research contract."
        ),
        "",
        (
            "Machine-readable run facts are recorded in run.json by the "
            "current local verification run."
        ),
        "",
        "| Logical Root | Area | Rows | Columns | Period |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in payload["dataset_inventory"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["logical_root"],
                    item["covered_by_evidence_area"] or "uncovered",
                    str(item["row_count"]),
                    str(item["column_count"]),
                    str(item["period"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Review-safe status:",
            "",
            f"- `output_scope | {payload['output_scope']}`",
            f"- `schema_id | {payload['schema_id']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `covered_roots | {len(payload['covered_roots'])}`",
            f"- `uncovered_roots | {len(payload['uncovered_roots'])}`",
            (
                "- `allowed_unverified_roots | "
                f"{len(payload['allowed_unverified_roots'])}`"
            ),
            f"- `handoff_ready | {str(payload['handoff_ready']).lower()}`",
        ]
    )
    if payload["uncovered_roots"]:
        lines.append("- uncovered dataset roots require review before final status")
    return _render_review_safe_markdown(lines)


def _render_review_safe_markdown(lines: list[str]) -> str:
    text = "\n".join(lines) + "\n"
    reason = evidence_text_unsafe_reason(text)
    if reason is not None:
        raise ValueError(f"unsafe_review_safe_markdown: {reason}")
    return text


__all__ = [
    "render_local_verification_inventory_markdown",
    "render_review_safe_aggregate_evidence_markdown",
    "render_run_summary_markdown",
]
