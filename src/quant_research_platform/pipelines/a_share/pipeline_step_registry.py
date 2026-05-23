"""注册 A 股 pipeline step contract。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStepContract:
    """描述一个可执行或可记录的 pipeline step。"""

    name: str
    version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    capabilities: tuple[str, ...]


PIPELINE_STEP_CONTRACTS = {
    "snapshot": PipelineStepContract(
        name="snapshot",
        version="1",
        inputs=("PipelineConfig",),
        outputs=("run_config.yaml", "warnings.json", "summary.md"),
        capabilities=("validate", "write_snapshot"),
    ),
    "processed_evaluation": PipelineStepContract(
        name="processed_evaluation",
        version="1",
        inputs=("PipelineConfig", "processed_factor_table"),
        outputs=(
            "evaluation/evaluation_manifest.json",
            "evaluation/label_availability.csv",
            "evaluation/evaluation_runs/",
        ),
        capabilities=("evaluate", "checkpoint"),
    ),
}


def get_pipeline_step_contract(name: str) -> PipelineStepContract:
    """返回已注册 pipeline step 的 contract。"""

    try:
        return PIPELINE_STEP_CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported pipeline step: {name!r}") from exc


__all__ = [
    "PIPELINE_STEP_CONTRACTS",
    "PipelineStepContract",
    "get_pipeline_step_contract",
]
