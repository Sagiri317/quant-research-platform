"""Read JSON metadata and write new records without replacing earlier results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    _qrp_json_snapshot_text,
    parse_qrp_json_object,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    open_filesystem_file,
)


def read_qrp_json_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    candidate = Path(path)
    try:
        with closing_iterator(open_filesystem_file(candidate)) as handle:
            raw = handle.read()
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    return raw, parse_qrp_json_object(raw)


def write_qrp_json_file(path: Path, payload: dict[str, Any]) -> Path:
    """Encode one complete JSON object and write a new exclusive file."""
    snapshot, text = _qrp_json_snapshot_text(payload)
    assert_internal_artifact_metadata_safe(snapshot)
    path = Path(path)
    filesystem_mkdir(path.parent, parents=True, exist_ok=True)
    try:
        handle = open_filesystem_file(path, "x", encoding="utf-8", newline="")
    except FileExistsError as exc:
        raise OutputAlreadyExistsError() from exc
    with closing_iterator(handle):
        handle.write(text + "\n")
    return path
