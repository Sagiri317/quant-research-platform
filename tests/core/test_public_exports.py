from __future__ import annotations


def test_package_root_exports_stable_core_api() -> None:
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


def test_artifacts_package_does_not_export_legacy_writer() -> None:
    import quant_research_platform.artifacts as artifacts
    import quant_research_platform.artifacts.writer as writer

    assert "write_evaluation_set_artifacts" in artifacts.__all__
    assert "write_run_artifacts" not in artifacts.__all__
    assert not hasattr(writer, "write_run_artifacts")
