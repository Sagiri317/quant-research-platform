"""Append-only facts for one explicit fitting attempt."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
)
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.atomic import atomic_write_json


def append_trial_record(root: Path, record: dict) -> Path:
    """Never replace a record; persist terminal changes without copying facts."""
    assert_internal_artifact_metadata_safe(record)
    status = record.get("status")
    if status not in {"attempted", "completed", "failed", "abandoned"}:
        raise ValueError("trial_status_invalid")
    root = Path(root)
    return atomic_write_json(
        root / ("attempted.json" if status == "attempted" else "terminal.json"), record
    )


def read_trial_ledger(root: Path) -> tuple[dict, ...]:
    """Combine the attempted facts with the terminal outcome."""
    root = Path(root)
    attempted = parse_qrp_json_object((root / "attempted.json").read_bytes())
    terminal_path = root / "terminal.json"
    if not terminal_path.exists():
        return (attempted,)
    terminal = parse_qrp_json_object(terminal_path.read_bytes())
    return attempted, {**attempted, **terminal}
