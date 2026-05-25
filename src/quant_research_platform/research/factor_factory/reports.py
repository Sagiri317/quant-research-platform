"""写出 Factor Factory 候选汇总和失败日志。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.research.factor_factory.candidate import (
    CandidateEvaluationRecord,
)

FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION = "0.1"


def write_candidate_reports(
    output_dir: str | Path,
    records: tuple[CandidateEvaluationRecord, ...],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """写出 candidate_summary.csv、failed_candidates.json 和 manifest。"""

    destination = Path(output_dir)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in records]
    summary_path = destination / "candidate_summary.csv"
    pd.DataFrame.from_records(rows).to_csv(summary_path, index=False)
    failed_path = destination / "failed_candidates.json"
    failed_rows = [row for row in rows if row["status"] == "failed"]
    failed_path.write_text(
        json.dumps(failed_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            _factor_factory_manifest(records=records),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary": summary_path, "failed": failed_path, "manifest": manifest_path}


def _factor_factory_manifest(
    *,
    records: tuple[CandidateEvaluationRecord, ...],
) -> dict[str, object]:
    candidate_ids = {record.candidate_id for record in records}
    successful_candidate_ids = {
        record.candidate_id for record in records if record.status == "success"
    }
    failed_candidate_ids = {
        record.candidate_id for record in records if record.status == "failed"
    }
    return {
        "artifact_schema_version": FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "factor_factory_run",
        "candidate_count": len(candidate_ids),
        "success_count": len(successful_candidate_ids),
        "failed_count": len(failed_candidate_ids),
        "artifacts": {
            "candidate_summary": "candidate_summary.csv",
            "failed_candidates": "failed_candidates.json",
        },
        "display": {
            "schema_version": FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION,
            "tables": [
                {
                    "key": "candidate_summary",
                    "label": "候选因子汇总",
                    "group": "candidate",
                    "path": "candidate_summary.csv",
                    "download_item": "candidate_summary.csv",
                }
            ],
            "downloads": {
                "items": [
                    {
                        "item": "candidate_summary.csv",
                        "label": "候选因子汇总",
                        "kind": "table",
                        "table": "candidate_summary",
                    },
                    {
                        "item": "failed_candidates.json",
                        "label": "失败候选日志",
                        "kind": "metadata",
                    },
                ],
                "presets": [
                    {
                        "key": "all",
                        "label": "全部 Factor Factory 产物",
                        "items": [
                            "candidate_summary.csv",
                            "failed_candidates.json",
                        ],
                    }
                ],
            },
        },
    }


__all__ = ["FACTOR_FACTORY_ARTIFACT_SCHEMA_VERSION", "write_candidate_reports"]
