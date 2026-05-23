from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from quant_research_platform.pipelines.a_share.config import PipelineConfig
from quant_research_platform.pipelines.a_share.runner import PipelineRunnerError
from quant_research_platform.research.factor_factory.batch_runner import (
    evaluate_processed_existing_candidates,
)
from quant_research_platform.research.factor_factory.candidate import FactorCandidate


def _base_config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        market="a_share",
        run_id="factory_run",
        date_start="2024-01-01",
        date_end="2024-12-31",
        factor_expression="neg(return(close, 20))",
        factor_template_id="reversal_20d",
        portfolio_mode="long_only",
        cost_bps=5.0,
        pnl_return_column="forward_return_1d",
        label_columns=("forward_return_1d",),
        output_root=tmp_path,
    )


def _candidate(
    candidate_id: str,
    *,
    output_column: str | None,
) -> FactorCandidate:
    return FactorCandidate(
        candidate_id=candidate_id,
        name=candidate_id,
        version="1",
        expression="neg(return(close, 20))",
        dependencies=("close",),
        output_column=output_column,
        sample_start="2024-02-01",
        sample_end="2024-11-30",
        label_columns=("forward_return_1d", "forward_return_20d"),
        inputs=("close",),
        outputs=(output_column,) if output_column else (),
        capabilities=("processed_existing", "evaluate"),
    )


def test_batch_runner_records_success_and_unmappable_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.research.factor_factory.batch_runner as runner_module

    calls: list[PipelineConfig] = []

    def fake_run_processed_existing_pipeline(config):
        calls.append(config)
        run_dir = tmp_path / "runs" / config.run_id
        run_dir.mkdir(parents=True)
        return SimpleNamespace(run_dir=run_dir)

    monkeypatch.setattr(
        runner_module,
        "run_processed_existing_pipeline",
        fake_run_processed_existing_pipeline,
    )
    candidates = (
        _candidate("reversal_20d@1", output_column="rev_20d"),
        _candidate("unmapped@1", output_column=None),
    )

    records = evaluate_processed_existing_candidates(
        candidates,
        base_config=_base_config(tmp_path),
        output_dir=tmp_path,
    )

    assert [record.status for record in records] == ["success", "failed"]
    assert records[0].run_dir == (
        tmp_path / "runs" / "factory_run_reversal_20d_1"
    ).as_posix()
    assert records[1].failure_code == "missing_processed_factor"
    assert len(calls) == 1
    assert calls[0].run_id == "factory_run_reversal_20d_1"
    assert calls[0].date_start == "2024-02-01"
    assert calls[0].date_end == "2024-11-30"
    assert calls[0].factor_expression == "neg(return(close, 20))"
    assert calls[0].factor_template_id == "reversal_20d"
    assert calls[0].processed_factor_column == "rev_20d"
    assert calls[0].label_columns == ("forward_return_1d", "forward_return_20d")

    factory_run_dir = tmp_path / "factor_factory" / "factory_run_factory"
    summary_path = factory_run_dir / "candidate_summary.csv"
    failed_path = factory_run_dir / "failed_candidates.json"
    manifest_path = factory_run_dir / "manifest.json"
    assert summary_path.is_file()
    summary = summary_path.read_text(encoding="utf-8")
    assert "inputs" in summary
    assert "capabilities" in summary
    assert "availability_rule" in summary
    assert "run_ref" in summary
    assert "run_dir" not in summary
    assert str(tmp_path) not in summary
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    assert [row["candidate_id"] for row in failed] == ["unmapped@1"]
    assert "run_dir" not in failed[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "factor_factory_run"
    assert manifest["display"]["tables"][0]["path"] == "candidate_summary.csv"
    assert manifest["display"]["downloads"]["items"][1]["item"] == (
        "failed_candidates.json"
    )

    with pytest.raises(FileExistsError):
        evaluate_processed_existing_candidates(
            candidates,
            base_config=_base_config(tmp_path),
            output_dir=tmp_path,
        )


def test_batch_runner_records_pipeline_failure_and_continues(
    tmp_path,
    monkeypatch,
) -> None:
    import quant_research_platform.research.factor_factory.batch_runner as runner_module

    calls: list[str] = []

    def fake_run_processed_existing_pipeline(config):
        calls.append(config.run_id)
        if config.run_id.endswith("bad_1"):
            raise PipelineRunnerError("missing processed factor table")
        if config.run_id.endswith("empty_1"):
            raise PipelineRunnerError("empty sample after filtering")
        run_dir = tmp_path / "runs" / config.run_id
        run_dir.mkdir(parents=True)
        return SimpleNamespace(run_dir=run_dir)

    monkeypatch.setattr(
        runner_module,
        "run_processed_existing_pipeline",
        fake_run_processed_existing_pipeline,
    )
    candidates = (
        _candidate("bad@1", output_column="bad_factor"),
        _candidate("empty@1", output_column="empty_factor"),
        _candidate("good@1", output_column="rev_20d"),
    )

    records = evaluate_processed_existing_candidates(
        candidates,
        base_config=_base_config(tmp_path),
        output_dir=tmp_path,
    )

    assert [record.status for record in records] == ["failed", "failed", "success"]
    assert records[0].failure_code == "missing_processed_field"
    assert records[1].failure_code == "insufficient_sample"
    assert "missing processed factor table" in (records[0].failure_reason or "")
    assert calls == [
        "factory_run_bad_1",
        "factory_run_empty_1",
        "factory_run_good_1",
    ]


def test_batch_runner_sanitizes_private_failure_reason(tmp_path, monkeypatch) -> None:
    import quant_research_platform.research.factor_factory.batch_runner as runner_module

    def fake_run_processed_existing_pipeline(config):
        raise PipelineRunnerError(
            f"missing data/processed_private/csmar/{config.run_id}.parquet"
        )

    monkeypatch.setattr(
        runner_module,
        "run_processed_existing_pipeline",
        fake_run_processed_existing_pipeline,
    )
    candidates = (_candidate("bad@1", output_column="bad_factor"),)

    evaluate_processed_existing_candidates(
        candidates,
        base_config=_base_config(tmp_path),
        output_dir=tmp_path,
    )

    summary = (
        tmp_path / "factor_factory" / "factory_run_factory" / "candidate_summary.csv"
    ).read_text(encoding="utf-8")
    failed = (
        tmp_path / "factor_factory" / "factory_run_factory" / "failed_candidates.json"
    ).read_text(encoding="utf-8")
    assert "processed_private" not in summary
    assert "processed_private" not in failed
    assert "[private-path]" in summary
