"""Record a real learning attempt before preprocessing or fitting starts."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.storage.experiments import append_trial_record


class TrialRecorder:
    def __init__(self, root: Path, facts: dict) -> None:
        self.root = Path(root)
        self.record = {
            "trial_id": f"trial_{uuid4().hex}",
            **qrp_json_mapping_snapshot(facts),
            "status": "attempted",
        }

    def begin(self) -> dict:
        append_trial_record(self.root, self.record)
        return qrp_json_mapping_snapshot(self.record)

    def finish(
        self, *, result: dict | None = None, error: BaseException | None = None
    ) -> dict:
        terminal = {
            **self.record,
            "status": "completed"
            if error is None
            else "failed"
            if isinstance(error, Exception)
            else "abandoned",
        }
        if result is not None:
            terminal["result"] = result
        if error is not None:
            terminal["error"] = type(error).__name__
        append_trial_record(self.root, terminal)
        return terminal
