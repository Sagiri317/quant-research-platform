"""Batch-evaluate A-share factor candidates from existing factor tables."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.runner import (
    PipelineRunnerError,
    run_a_share_pipeline,
)
from quant_research_platform.research.factor_factory.candidate import (
    CandidateEvaluationRecord,
    FactorCandidate,
)
from quant_research_platform.research.factor_factory.reports import (
    write_candidate_reports,
)


def evaluate_existing_factor_candidates(
    candidates: tuple[FactorCandidate, ...],
    *,
    base_config: PipelineConfig,
    output_dir: str | Path,
) -> tuple[CandidateEvaluationRecord, ...]:
    """Evaluate candidates independently; one failure does not stop the batch."""

    output_path = Path(output_dir)
    factory_run_dir = output_path / "factor_factory" / _factory_run_id(
        base_config.run_id
    )
    if factory_run_dir.exists():
        raise FileExistsError(factory_run_dir)

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
            start_from="existing_factors",
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
            result = run_a_share_pipeline(config)
        except PipelineRunnerError as exc:
            records.append(
                _failed_record(
                    candidate,
                    failure_code=_pipeline_failure_code(exc),
                    failure_reason=str(exc),
                )
            )
            continue
        try:
            records.extend(
                _success_records(
                    candidate,
                    result.run_dir,
                    artifact_dir=_evaluation_artifact_dir(result),
                )
            )
        except ValueError as exc:
            records.append(
                _failed_record(
                    candidate,
                    failure_code="invalid_evaluation_artifact",
                    failure_reason=str(exc),
                )
            )
    write_candidate_reports(
        factory_run_dir,
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


def _success_records(
    candidate: FactorCandidate,
    run_dir: Path,
    *,
    artifact_dir: Path | None = None,
) -> list[CandidateEvaluationRecord]:
    summary = _candidate_combination_summary(artifact_dir or run_dir)
    return [
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
            run_dir=run_dir.as_posix(),
            combination_id=str(row["combination_id"]),
            feature=str(row["feature"]),
            label=str(row["label"]),
            kernel=str(row["kernel"]),
            portfolio_mode=str(row["portfolio_mode"]),
            mean_ic=_optional_float(row.get("mean_ic")),
            final_cumulative_net_return=_optional_float(
                row.get("final_cumulative_net_return")
            ),
            mean_turnover=_optional_float(row.get("mean_turnover")),
            total_cost=_optional_float(row.get("total_cost")),
        )
        for row in summary.to_dict(orient="records")
    ]


def _evaluation_artifact_dir(result: object) -> Path | None:
    manifest_path = getattr(result, "evaluation_artifact_path", None)
    if manifest_path is None:
        return None
    return Path(manifest_path).parent


def _candidate_combination_summary(run_dir: Path) -> pd.DataFrame:
    combinations_path = run_dir / "dimensions" / "combinations.csv"
    evaluation_summary_path = run_dir / "summaries" / "evaluation_summary.csv"
    if not combinations_path.is_file():
        raise ValueError("missing dimensions/combinations.csv")
    if not evaluation_summary_path.is_file():
        raise ValueError("missing summaries/evaluation_summary.csv")
    combinations = pd.read_csv(combinations_path)
    evaluation_summary = pd.read_csv(evaluation_summary_path)
    _require_columns(
        combinations,
        "dimensions/combinations.csv",
        {"combination_id", "feature", "label", "kernel", "portfolio_mode"},
    )
    _require_columns(
        evaluation_summary,
        "summaries/evaluation_summary.csv",
        {
            "combination_id",
            "mean_ic",
            "final_cumulative_net_return",
            "mean_turnover",
            "total_cost",
        },
    )
    if combinations.empty:
        raise ValueError("evaluation artifact has no combinations")
    if evaluation_summary.empty:
        raise ValueError("evaluation artifact has no evaluation_summary rows")
    missing_summary = sorted(
        set(combinations["combination_id"].astype(str)).difference(
            evaluation_summary["combination_id"].astype(str)
        )
    )
    if missing_summary:
        raise ValueError(
            "summaries/evaluation_summary.csv missing combinations: "
            f"{missing_summary}"
        )
    metric_columns = [
        "combination_id",
        "mean_ic",
        "final_cumulative_net_return",
        "mean_turnover",
        "total_cost",
    ]
    merged = combinations.merge(
        evaluation_summary[metric_columns],
        on="combination_id",
        how="left",
        validate="one_to_one",
    )
    return merged


def _require_columns(frame: pd.DataFrame, source: str, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


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


__all__ = ["evaluate_existing_factor_candidates"]
