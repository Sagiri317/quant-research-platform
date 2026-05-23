from __future__ import annotations


def test_package_root_exports_stable_evaluation_api() -> None:
    from quant_research_platform import (
        EvaluationSpec,
        WeightConfig,
        make_parquet_evaluation_spec,
        run_evaluation,
    )

    assert EvaluationSpec.__name__ == "EvaluationSpec"
    assert WeightConfig.__name__ == "WeightConfig"
    assert callable(make_parquet_evaluation_spec)
    assert callable(run_evaluation)


def test_artifacts_package_exports_current_artifact_writer() -> None:
    import quant_research_platform.artifacts as artifacts
    import quant_research_platform.artifacts.artifact_writer as writer

    assert "write_evaluation_set_artifacts" in artifacts.__all__
    assert callable(writer.write_evaluation_set_artifacts)
