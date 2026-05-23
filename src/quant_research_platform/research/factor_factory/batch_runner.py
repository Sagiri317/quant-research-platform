"""批量评估 processed-existing factor candidates。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    run_processed_existing_pipeline,
)
from quant_research_platform.research.factor_factory.candidate import (
    CandidateEvaluationRecord,
    FactorCandidate,
)
from quant_research_platform.research.factor_factory.reports import (
    write_candidate_reports,
)


def evaluate_processed_existing_candidates(
    candidates: tuple[FactorCandidate, ...],
    *,
    base_config: PipelineConfig,
    output_dir: str | Path,
) -> tuple[CandidateEvaluationRecord, ...]:
    """逐个评估 candidates，一个失败不影响其他 candidate。"""

    records: list[CandidateEvaluationRecord] = []
    for candidate in candidates:
        config = replace(
            base_config,
            run_id=_candidate_run_id(base_config.run_id, candidate.candidate_id),
            date_start=candidate.sample_start,
            date_end=candidate.sample_end,
            factor_expression=candidate.expression,
            factor_template_id=candidate.candidate_id.split("@", 1)[0],
            processed_factor_column=candidate.output_column,
            label_columns=candidate.label_columns,
            output_root=Path(output_dir),
        )
        if candidate.output_column is None:
            records.append(
                _failed_record(
                    candidate,
                    failure_code="missing_processed_factor",
                    failure_reason=(
                        "candidate cannot be mapped to an existing processed "
                        "factor column"
                    ),
                )
            )
            continue
        try:
            result = run_processed_existing_pipeline(config)
        except PipelineRunnerError as exc:
            records.append(
                _failed_record(
                    candidate,
                    failure_code=_pipeline_failure_code(exc),
                    failure_reason=str(exc),
                )
            )
            continue
        records.append(
            CandidateEvaluationRecord(
                candidate_id=candidate.candidate_id,
                version=candidate.version,
                expression=candidate.expression,
                dependencies=candidate.dependencies,
                output_column=candidate.output_column,
                sample_start=candidate.sample_start,
                sample_end=candidate.sample_end,
                label_columns=candidate.label_columns,
                inputs=candidate.inputs,
                outputs=candidate.outputs,
                capabilities=candidate.capabilities,
                parameters=candidate.parameters,
                availability_rule=candidate.availability_rule,
                time_semantics=candidate.time_semantics,
                status="success",
                run_dir=result.run_dir.as_posix(),
            )
        )
    output_path = Path(output_dir)
    write_candidate_reports(
        output_path / "factor_factory" / _factory_run_id(base_config.run_id),
        tuple(records),
        overwrite=False,
    )
    return tuple(records)


def _failed_record(
    candidate: FactorCandidate,
    *,
    failure_code: str,
    failure_reason: str,
) -> CandidateEvaluationRecord:
    return CandidateEvaluationRecord(
        candidate_id=candidate.candidate_id,
        version=candidate.version,
        expression=candidate.expression,
        dependencies=candidate.dependencies,
        output_column=candidate.output_column,
        sample_start=candidate.sample_start,
        sample_end=candidate.sample_end,
        label_columns=candidate.label_columns,
        inputs=candidate.inputs,
        outputs=candidate.outputs,
        capabilities=candidate.capabilities,
        parameters=candidate.parameters,
        availability_rule=candidate.availability_rule,
        time_semantics=candidate.time_semantics,
        status="failed",
        failure_code=failure_code,
        failure_reason=failure_reason,
    )


def _candidate_run_id(prefix: str, candidate_id: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in candidate_id
    )
    return f"{prefix}_{normalized}"


def _factory_run_id(prefix: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in prefix
    )
    return f"{normalized}_factory"


def _pipeline_failure_code(exc: PipelineRunnerError) -> str:
    if exc.failure_code is not None:
        return exc.failure_code
    message = str(exc).lower()
    if "missing" in message or "required column" in message:
        return "missing_processed_field"
    if "empty" in message or "insufficient" in message or "sample" in message:
        return "insufficient_sample"
    return "cannot_compute"


__all__ = ["evaluate_processed_existing_candidates"]
