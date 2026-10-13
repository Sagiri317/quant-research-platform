"""Append-only facts for one explicit fitting attempt."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.atomic import atomic_write_json


def append_trial_record(root: Path, record: dict) -> Path:
    """Never replace an attempted or terminal record, including failed attempts."""
    assert_internal_artifact_metadata_safe(record)
    status = record.get("status")
    if status not in {"attempted", "completed", "failed", "abandoned"}:
        raise ValueError("trial_status_invalid")
    root = Path(root)
    if status != "attempted":
        attempted = read_trial_ledger(root)[0]
        if canonical_identity_bytes(
            {
                key: value
                for key, value in record.items()
                if key not in {"status", "result", "error"}
            }
        ) != canonical_identity_bytes(
            {key: value for key, value in attempted.items() if key != "status"}
        ):
            raise ValueError("trial_completion_mismatch")
    return atomic_write_json(
        root / ("attempted.json" if status == "attempted" else "terminal.json"), record
    )


def read_trial_ledger(root: Path) -> tuple[dict, ...]:
    root = Path(root)
    attempted = parse_qrp_json_object((root / "attempted.json").read_bytes())
    assert_internal_artifact_metadata_safe(attempted)
    if attempted.get("status") != "attempted":
        raise ValueError("trial_attempt_required")
    terminal_path = root / "terminal.json"
    if not terminal_path.exists():
        return (attempted,)
    terminal = parse_qrp_json_object(terminal_path.read_bytes())
    assert_internal_artifact_metadata_safe(terminal)
    if terminal.get("status") not in {"completed", "failed", "abandoned"}:
        raise ValueError("trial_terminal_invalid")
    if canonical_identity_bytes(
        {k: v for k, v in terminal.items() if k not in {"status", "result", "error"}}
    ) != canonical_identity_bytes(
        {k: v for k, v in attempted.items() if k != "status"}
    ):
        raise ValueError("trial_completion_mismatch")
    return attempted, terminal
