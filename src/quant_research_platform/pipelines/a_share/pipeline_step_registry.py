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
    "build_interim": PipelineStepContract(
        name="build_interim",
        version="1",
        inputs=("raw_dataset",),
        outputs=("interim_dataset",),
        capabilities=("build", "checkpoint", "private_data_boundary"),
    ),
    "build_panel": PipelineStepContract(
        name="build_panel",
        version="1",
        inputs=("interim_dataset",),
        outputs=("research_panel",),
        capabilities=("build", "checkpoint"),
    ),
    "build_factors": PipelineStepContract(
        name="build_factors",
        version="1",
        inputs=("research_panel",),
        outputs=("factor_table",),
        capabilities=("build", "checkpoint"),
    ),
    "evaluate": PipelineStepContract(
        name="evaluate",
        version="1",
        inputs=(
            "processed_factor_table",
            "factor_table",
            "PipelineConfig",
        ),
        outputs=(
            "evaluation/manifest.json",
            "evaluation/run_context.json",
            "evaluation/summary.json",
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
