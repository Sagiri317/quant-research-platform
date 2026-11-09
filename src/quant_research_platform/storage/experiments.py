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

_TERMINAL_FIELDS = {"trial_id", "status", "result", "error"}


def append_trial_record(root: Path, record: dict) -> Path:
    """Never replace a record; persist terminal changes without copying facts."""
    assert_internal_artifact_metadata_safe(record)
    status = record.get("status")
    if status not in {"attempted", "completed", "failed", "abandoned"}:
        raise ValueError("trial_status_invalid")
    root = Path(root)
    if status != "attempted":
        _restore_trial_facts(read_trial_ledger(root)[0], record)
        record = {
            key: value for key, value in record.items() if key in _TERMINAL_FIELDS
        }
    return atomic_write_json(
        root / ("attempted.json" if status == "attempted" else "terminal.json"), record
    )


def _restore_trial_facts(attempted: dict, terminal: dict) -> dict:
    """Accept compact terminal changes or a legacy full-facts terminal."""
    if terminal.get("status") not in {"completed", "failed", "abandoned"}:
        raise ValueError("trial_terminal_invalid")
    facts = {
        key: value
        for key, value in terminal.items()
        if key not in {"status", "result", "error"}
    }
    expected = {key: value for key, value in attempted.items() if key != "status"}
    if set(facts) <= {"trial_id"}:
        expected = {key: value for key, value in expected.items() if key == "trial_id"}
    if canonical_identity_bytes(facts) != canonical_identity_bytes(expected):
        raise ValueError("trial_completion_mismatch")
    return {**attempted, **terminal}


def read_trial_ledger(root: Path) -> tuple[dict, ...]:
    """Return full record views for both compact and legacy persisted terminals."""
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
    return attempted, _restore_trial_facts(attempted, terminal)
