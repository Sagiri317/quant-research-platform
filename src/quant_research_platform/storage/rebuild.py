"""Storage-owned scratch lifecycle for deterministic rebuild candidates."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_name,
    remove_owned_scratch,
)
from quant_research_platform.storage.identity import require_unaliased_path
from quant_research_platform.storage.layout import StorageLayout


@dataclass(slots=True)
class RebuildCandidateWorkspace:
    project_root: Path
    token: str
    _preserve: bool = False

    @property
    def layout(self) -> StorageLayout:
        return StorageLayout(project_root=self.project_root)

    def preserve(self) -> None:
        self._preserve = True


@contextmanager
def rebuild_candidate_workspace(
    layout: StorageLayout,
) -> Iterator[RebuildCandidateWorkspace]:
    parent = require_unaliased_path(
        Path(layout.project_root) / ".run",
        error="invalid_rebuild_scratch_path",
    )
    parent.mkdir(parents=True, exist_ok=True)
    require_unaliased_path(parent, error="invalid_rebuild_scratch_path")
    token = uuid4().hex
    root = _extended_path(parent / owned_scratch_name(".r-", token))
    require_unaliased_path(root, error="invalid_rebuild_scratch_path")
    os.mkdir(root)
    workspace = RebuildCandidateWorkspace(project_root=root, token=token)
    try:
        yield workspace
    except BaseException:
        workspace.preserve()
        raise
    finally:
        if not workspace._preserve:
            primary = sys.exception()
            try:
                remove_owned_scratch(root)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note(
                    "rebuild candidate scratch cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )


def _extended_path(path: Path) -> Path:
    if os.name != "nt":
        return Path(path)
    value = os.path.abspath(path)
    return Path(value if value.startswith("\\\\?\\") else "\\\\?\\" + value)
