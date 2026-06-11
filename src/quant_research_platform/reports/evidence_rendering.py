"""Public-safe evidence rendering helpers."""

from __future__ import annotations

from typing import Any


def render_public_safe_evidence_markdown(payload: dict[str, Any]) -> str:
    """Render the public-safe evidence payload as Markdown."""

    lines = [
        "# Clean Rewrite 0.1 Public-Safe Parity Evidence",
        "",
        (
            "This evidence records public-safe aggregate validation for QRP "
            "Clean Rewrite `0.1.0`. Public identifiers are "
            "`data_product_id=a_share_daily_bar`, `market_id=a_share`, and "
            "contract version `0.1`."
        ),
        "",
        (
            "Machine-readable evidence lives in "
            "`docs/reports/clean-rewrite-0.1-parity-evidence.json`."
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
            "Public-safe status:",
            "",
            f"- `privacy_level | {payload['privacy_level']}`",
            f"- `contract_version | {payload['contract_version']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `verification_scope | {payload['verification_scope']}`",
            f"- `evidence_generation_mode | {payload['evidence_generation_mode']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `merge_ready | {str(payload['merge_ready']).lower()}`",
            (
                "- private local aggregate run completed without committed "
                "row-level artifacts"
            ),
        ]
    )
    if not payload["merge_ready"]:
        lines.append("- human approval required before merge-ready status")
    return "\n".join(lines) + "\n"


def render_private_verification_summary_markdown(payload: dict[str, Any]) -> str:
    """Render the sealed dataset inventory summary as public-safe Markdown."""

    lines = [
        "# Clean Rewrite 0.1 Private Verification Summary",
        "",
        (
            "This summary records aggregate-only sealed dataset inventory "
            "for QRP Clean Rewrite `0.1.0`."
        ),
        "",
        (
            "Machine-readable summary lives in "
            "`docs/reports/clean-rewrite-0.1-private-verification-summary.json`."
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
            "Public-safe status:",
            "",
            f"- `privacy_level | {payload['privacy_level']}`",
            f"- `contract_version | {payload['contract_version']}`",
            f"- `verification_kind | {payload['verification_kind']}`",
            f"- `source_data_visibility | {payload['source_data_visibility']}`",
            "- `row_level_output_committed | false`",
            f"- `covered_roots | {len(payload['covered_roots'])}`",
            f"- `uncovered_roots | {len(payload['uncovered_roots'])}`",
            (
                "- `allowed_unverified_roots | "
                f"{len(payload['allowed_unverified_roots'])}`"
            ),
            f"- `merge_ready | {str(payload['merge_ready']).lower()}`",
        ]
    )
    if payload["uncovered_roots"]:
        lines.append("- uncovered dataset roots require review before merge-ready")
    return "\n".join(lines) + "\n"


__all__ = [
    "render_private_verification_summary_markdown",
    "render_public_safe_evidence_markdown",
]
