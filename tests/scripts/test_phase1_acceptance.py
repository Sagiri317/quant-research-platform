from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "phase1_acceptance.py"
SPEC = importlib.util.spec_from_file_location("phase1_acceptance", SCRIPT_PATH)
assert SPEC is not None
phase1_acceptance = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["phase1_acceptance"] = phase1_acceptance
SPEC.loader.exec_module(phase1_acceptance)


def test_compare_run_tree_matches_combined_old_run_to_split_new_runs(tmp_path):
    baseline = tmp_path / "old"
    current = tmp_path / "new"
    old_run = baseline / "core_runs" / "csmar_2024_small"
    new_run_5d = current / "evaluation_runs" / "csmar_2024_forward_return_5d"
    new_run_20d = current / "evaluation_runs" / "csmar_2024_forward_return_20d"

    _write_run(
        old_run,
        labels=["forward_return_5d", "forward_return_20d"],
        rows=[
            [
                "mom_20d__forward_return_5d__w0__c0",
                "mom_20d",
                "forward_return_5d",
                "0.1",
            ],
            [
                "mom_20d__forward_return_20d__w0__c0",
                "mom_20d",
                "forward_return_20d",
                "0.2",
            ],
        ],
    )
    _write_run(
        new_run_5d,
        labels=["forward_return_5d"],
        rows=[
            [
                "mom_20d__forward_return_5d__w0__c0",
                "mom_20d",
                "forward_return_5d",
                "0.1",
            ],
        ],
    )
    _write_run(
        new_run_20d,
        labels=["forward_return_20d"],
        rows=[
            [
                "mom_20d__forward_return_20d__w0__c0",
                "mom_20d",
                "forward_return_20d",
                "0.2",
            ],
        ],
    )

    result = phase1_acceptance.compare_run_tree(baseline, current)

    assert result["status"] == "pass"
    assert result["old_run_count"] == 1
    assert result["new_run_count"] == 2
    assert result["old_summary_rows"]["metrics/performance_summary.csv"] == 2
    assert result["new_summary_rows"]["metrics/performance_summary.csv"] == 2
    assert result["differences"] == []


def test_run_acceptance_passes_src_pythonpath_to_child_process(monkeypatch):
    calls = []

    def fake_run(command, *, cwd, check, env):
        calls.append({"command": command, "cwd": cwd, "check": check, "env": env})

    monkeypatch.setattr(phase1_acceptance.subprocess, "run", fake_run)
    monkeypatch.setattr(phase1_acceptance, "PROJECT_ROOT", Path("C:/repo"))

    phase1_acceptance.run_acceptance("acceptance_2024_dataframe_long_only")

    assert calls
    pythonpath = calls[0]["env"]["PYTHONPATH"].split(os.pathsep)
    assert str(Path("C:/repo") / "src") in pythonpath
    assert calls[0]["cwd"] == Path("C:/repo")
    assert calls[0]["check"] is True


def test_module_import_adds_project_src_to_sys_path():
    assert str(phase1_acceptance.PROJECT_ROOT / "src") in sys.path


def test_factory_smoke_output_root_is_private_reports(monkeypatch, tmp_path):
    captured = {}

    class FakeRegistry:
        def candidates(self, **_kwargs):
            return (
                _candidate("momentum_60d__window_20@1"),
                _candidate("reversal_20d__window_20@1"),
            )

    class FakeRegistryFactory:
        @staticmethod
        def from_directory(_path):
            return FakeRegistry()

    def fake_evaluate(candidates, *, base_config, output_dir):
        captured["candidate_count"] = len(candidates)
        captured["output_dir"] = output_dir
        captured["base_output_root"] = base_config.output_root
        factory_run = output_dir / "factor_factory" / "f_factory"
        factory_run.mkdir(parents=True)
        return [
            _record("momentum_60d__window_20@1", factory_run),
            _record("reversal_20d__window_20@1", factory_run),
        ]

    monkeypatch.setattr(phase1_acceptance, "PROJECT_ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "quant_research_platform.research.factor_factory.factor_registry",
        type(
            "FakeFactorRegistryModule",
            (),
            {"FactorTemplateRegistry": FakeRegistryFactory},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "quant_research_platform.research.factor_factory.batch_runner",
        type(
            "FakeBatchRunnerModule",
            (),
            {"evaluate_existing_factor_candidates": fake_evaluate},
        ),
    )

    summary_path = phase1_acceptance.run_factory_smoke("unit_stamp")

    expected_root = tmp_path / "reports/private_real/ff/unit_stamp"
    assert captured["output_dir"] == expected_root
    assert captured["base_output_root"] == expected_root
    assert summary_path == expected_root / "phase1_factory_smoke_summary.json"
    assert expected_root.is_relative_to(tmp_path / "reports/private_real/ff")


def _write_run(
    run_dir: Path,
    *,
    labels: list[str],
    rows: list[list[str]],
) -> None:
    run_dir.mkdir(parents=True)
    summary = {
        "artifact_schema_version": "0.5",
        "artifact_type": "evaluation_set",
        "data_stage": "approved_offline_real_data",
        "uses_external_data": True,
        "n_groups": 5,
        "schema": {
            "feature_columns": ["mom_20d"],
            "label_columns": labels,
            "pnl_return_column": "forward_return_1d",
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    metrics = run_dir / "metrics"
    metrics.mkdir()
    with (metrics / "performance_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["combination_id", "feature", "label", "value"])
        writer.writerows(rows)


def _candidate(candidate_id: str):
    return type(
        "FakeCandidate",
        (),
        {
            "candidate_id": candidate_id,
            "expression": "neg(return(close, 20))",
            "template_id": "unit",
        },
    )()


def _record(candidate_id: str, factory_run: Path):
    return type(
        "FakeRecord",
        (),
        {
            "candidate_id": candidate_id,
            "status": "success",
            "to_row": lambda self: {"run_ref": factory_run.as_posix()},
        },
    )()
