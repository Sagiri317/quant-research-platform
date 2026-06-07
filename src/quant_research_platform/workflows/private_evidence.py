"""Minimal removable private evidence verification harness."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.catalog.defaults import (
    default_evaluation_catalog,
    default_model_catalog,
    default_prediction_catalog,
    default_processed_catalog,
    default_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as a_share_materializer,
)
from quant_research_platform.markets.implementations.a_share.source_bindings import (
    csmar as a_share_csmar_binding,
)
from quant_research_platform.reports.private_evidence import (
    build_private_evidence_summary,
    build_validation_summary,
    write_private_evidence_summary_json,
    write_private_evidence_summary_md,
    write_run_summary_json,
    write_run_summary_md,
    write_validation_summary,
)
from quant_research_platform.research.datasets import (
    assemble_research_dataset_from_processed,
)
from quant_research_platform.research.evaluations.runner import (
    run_evaluation_from_research_dataset,
)
from quant_research_platform.research.predictions.handoff import (
    run_prediction_from_research_dataset,
)
from quant_research_platform.runtime.execution_metrics import (
    ExecutionMetricsRecorder,
    write_execution_metrics_csv,
    write_execution_metrics_json,
)
from quant_research_platform.storage.atomic import atomic_write_json
from quant_research_platform.storage.layout import PrivateEvidenceLayout, StorageLayout
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_layer_manifest,
    write_run_manifest,
)

CSMAR_SOURCE_ID = a_share_csmar_binding.CSMAR_SOURCE_ID
a_share_csmar_import_interim_data = (
    a_share_csmar_binding.a_share_csmar_import_interim_data
)
build_a_share_processed_layers = a_share_materializer.build_a_share_processed_layers


@dataclass(frozen=True)
class PrivateEvidenceRunConfig:
    output_root: Path
    start_date: str
    end_date: str
    replace_output: bool = False
    run_id: str | None = None


@dataclass(frozen=True)
class PrivateEvidenceRunResult:
    status: str
    run_id: str
    output_root: str
    summary: dict[str, Any]


def run_private_evidence(config: PrivateEvidenceRunConfig) -> PrivateEvidenceRunResult:
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    if start > end:
        raise ValueError("start_date_after_end_date")
    output_root = Path(config.output_root)
    _assert_output_root_under_run(output_root)
    _prepare_output_root(output_root, replace_output=config.replace_output)
    run_id = config.run_id or output_root.name
    layout = PrivateEvidenceLayout.from_output_root(output_root)
    _create_layout_directories(layout)
    atomic_write_json(
        layout.config_root / "run_config.json",
        {
            "run_id": run_id,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "output_root": _logical_path(output_root),
            "replace_output": config.replace_output,
            "created_at": _utc_now(),
        },
    )

    recorder = ExecutionMetricsRecorder()
    storage_layout = StorageLayout(project_root=output_root)
    start_year = int(start.year)
    end_year = int(end.year)
    with recorder.record("interim_generation", logical_layer="interim"):
        a_share_csmar_import_interim_data(
            layout=storage_layout,
            raw_root=Path("data") / "raw" / CSMAR_SOURCE_ID,
            start_date=config.start_date,
            end_date=config.end_date,
        )
    processed_market_root = layout.processed_root / "a_share"
    with recorder.record("processed_generation", logical_layer="processed") as record:
        processed_result = build_a_share_processed_layers(
            interim_root=layout.interim_root / CSMAR_SOURCE_ID,
            processed_root=processed_market_root,
            start_year=start_year,
            end_year=end_year,
            end_date=config.end_date,
            metrics_recorder=recorder,
        )
        record.output_rows = sum(processed_result.row_counts.values())
    with recorder.record("research_generation", logical_layer="research"):
        assemble_research_dataset_from_processed(
            processed_root=processed_market_root,
            dataset_catalog=default_research_dataset_catalog(),
            processed_catalog=default_processed_catalog(),
            dataset_id=A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
            years=tuple(range(start_year, end_year + 1)),
        )
    with recorder.record("evaluation_generation", logical_layer="evidence"):
        run_evaluation_from_research_dataset(
            processed_root=processed_market_root,
            dataset_catalog=default_research_dataset_catalog(),
            evaluation_catalog=default_evaluation_catalog(),
            run_id="a_share_mom_20d_forward_return_20d_2014",
            years=(start_year,),
        )
    with recorder.record("prediction_generation", logical_layer="evidence"):
        run_prediction_from_research_dataset(
            processed_root=processed_market_root,
            dataset_catalog=default_research_dataset_catalog(),
            model_catalog=default_model_catalog(),
            prediction_catalog=default_prediction_catalog(),
            prediction_id="a_share_linear_mom_20d_prediction_2014",
            years=(start_year,),
        )

    dataset_roots = _dataset_roots(layout)
    dataset_manifest_paths = tuple(
        dataset_root / "_manifest" / "manifest.json"
        for dataset_root in dataset_roots
    )
    layer_manifest_paths = (
        write_layer_manifest(
            layer_root=layout.interim_root,
            logical_layer="interim",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "interim"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.processed_root,
            logical_layer="processed",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "processed"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.research_root,
            logical_layer="research",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "research"
            ),
        ),
        write_layer_manifest(
            layer_root=layout.evidence_root,
            logical_layer="evidence",
            run_id=run_id,
            dataset_roots=tuple(
                root
                for root in dataset_roots
                if _layer_for_root(root, layout) == "evidence"
            ),
        ),
    )
    with recorder.record("manifest_generation") as record:
        run_manifest_path = write_run_manifest(
            manifests_root=layout.manifests_root,
            run_id=run_id,
            output_root=output_root,
            layer_manifest_paths=layer_manifest_paths,
        )
        record.file_count = 1
    timings_json = write_execution_metrics_json(
        layout.timings_root / "timings.json",
        recorder.records,
        run_id=run_id,
        memory_sampler_unavailable=recorder.memory_sampler.unavailable,
    )
    write_execution_metrics_csv(layout.timings_root / "timings.csv", recorder.records)
    run_manifest = read_manifest(run_manifest_path)
    generated_paths = (
        layout.config_root / "run_config.json",
        run_manifest_path,
        *layer_manifest_paths,
        timings_json,
    )
    with recorder.record("validation_summary"):
        validation_summary = build_validation_summary(
            run_id=run_id,
            output_root=output_root,
            dataset_manifest_paths=dataset_manifest_paths,
            layer_manifest_paths=layer_manifest_paths,
            generated_text_paths=generated_paths,
        )
    write_validation_summary(
        layout.validations_root / "validation_summary.json",
        validation_summary,
    )
    timings_payload = read_manifest(timings_json)
    evidence_summary = build_private_evidence_summary(
        run_id=run_id,
        date_range={"start_date": config.start_date, "end_date": config.end_date},
        manifests={
            "dataset_count": len(dataset_manifest_paths),
            "layer_count": len(layer_manifest_paths),
            "run_validation_status": run_manifest["run_validation_status"],
        },
        timings=timings_payload,
        validation_summary=validation_summary,
        output_root=output_root,
    )
    write_private_evidence_summary_json(
        layout.evidence_root / "evidence_summary.json",
        evidence_summary,
    )
    write_private_evidence_summary_md(
        layout.evidence_root / "evidence_summary.md",
        evidence_summary,
    )
    summary = {
        **evidence_summary,
        "validation_summary_path": _logical_path(
            layout.validations_root / "validation_summary.json"
        ),
        "run_manifest_path": _logical_path(run_manifest_path),
    }
    write_run_summary_json(layout.summary_json_path, summary)
    write_run_summary_md(layout.summary_md_path, summary)
    return PrivateEvidenceRunResult(
        status=str(validation_summary["status"]),
        run_id=run_id,
        output_root=output_root.as_posix(),
        summary=summary,
    )


def _assert_output_root_under_run(path: Path) -> None:
    if ".run" not in Path(path).parts:
        raise ValueError("output_root_must_be_under_run")


def _prepare_output_root(path: Path, *, replace_output: bool) -> None:
    if path.exists() and not replace_output:
        raise ValueError("output_root_already_exists")
    if path.exists() and replace_output:
        _assert_output_root_under_run(path)
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _create_layout_directories(layout: PrivateEvidenceLayout) -> None:
    for root in (
        layout.config_root,
        layout.interim_root,
        layout.processed_root,
        layout.research_root,
        layout.evidence_root,
        layout.manifests_root,
        layout.validations_root,
        layout.timings_root,
        layout.logs_root,
    ):
        root.mkdir(parents=True, exist_ok=True)


def _dataset_roots(layout: PrivateEvidenceLayout) -> tuple[Path, ...]:
    roots: set[Path] = set()
    for data_root in sorted(layout.data_root.rglob("data")):
        if data_root.is_dir() and data_root.parts[-1] == "data":
            roots.add(data_root.parent)
    return tuple(sorted(roots, key=lambda item: item.as_posix()))


def _logical_layer(dataset_root: Path, output_root: Path) -> str:
    relative = dataset_root.relative_to(output_root).as_posix()
    if relative.startswith("data/interim/"):
        return "interim"
    if "/research_datasets/" in relative:
        return "research"
    if relative.startswith("reports/") or relative.startswith("data/evidence/"):
        return "evidence"
    return "processed"


def _layer_for_root(dataset_root: Path, layout: PrivateEvidenceLayout) -> str:
    return _logical_layer(dataset_root, layout.output_root)


def _dataset_id(dataset_root: Path) -> str:
    parts = dataset_root.parts
    if "research_datasets" in parts:
        return parts[parts.index("research_datasets") + 1]
    return dataset_root.name


def _logical_path(path: Path) -> str:
    parts = Path(path).parts
    for marker in (".run", "data", "reports", "manifests", "validations", "timings"):
        if marker in parts:
            return Path(*parts[parts.index(marker) :]).as_posix()
    return Path(path).as_posix()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "PrivateEvidenceRunConfig",
    "PrivateEvidenceRunResult",
    "run_private_evidence",
]
