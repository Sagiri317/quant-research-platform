"""Stage profile output writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.artifacts._output_common_validation import (
    derive_output_bundle_root,
)
from quant_research_platform.artifacts._output_manifest import (
    stage_profile_output_manifest_payload,
)
from quant_research_platform.contracts.outputs import OutputSpec
from quant_research_platform.policy.share_safety import (
    sanitize_shared_object,
    sanitize_shared_text,
)
from quant_research_platform.storage.atomic import (
    atomic_write_csv,
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest


def write_stage_profile_outputs(
    frame: pd.DataFrame,
    *,
    spec: OutputSpec,
    markdown_text: str,
    layout: StorageLayout,
) -> dict[str, Path]:
    """Write local stage-profile outputs."""

    destination = derive_output_bundle_root(
        output_spec=spec,
        layout=layout,
        expected_definition_id="stage_profile_diagnostic",
    )

    def _write(temp_root: Path) -> None:
        safe_rows = sanitize_shared_object(frame.to_dict(orient="records"))
        safe_frame = pd.DataFrame.from_records(safe_rows)
        atomic_write_csv(safe_frame, temp_root / "stage_profile.csv", index=False)
        atomic_write_json(
            temp_root / "stage_profile.json",
            {
                "schema_id": "qrp-stage-profile",
                "rows": safe_rows,
            },
        )
        atomic_write_text(
            temp_root / "stage_profile.md",
            sanitize_shared_text(markdown_text),
            encoding="utf-8",
        )
        write_manifest(
            temp_root / "manifest.json",
            stage_profile_output_manifest_payload(spec=spec),
        )

    atomic_write_directory(destination, _write)
    csv_path = destination / "stage_profile.csv"
    json_path = destination / "stage_profile.json"
    markdown_path = destination / "stage_profile.md"
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}
