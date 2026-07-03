# ruff: noqa: F403,F405
from __future__ import annotations

from tests.architecture.governance_scan_helpers import *


def test_business_logic_does_not_bypass_persistent_writers() -> None:
    hits: list[str] = []
    for path in _iter_python_files():
        if _is_approved_writer_path(path):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            hits.append(f"{path}:{exc.lineno}: syntax_error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _call_name(node.func)
            if call_name in FORBIDDEN_WRITER_METHODS:
                hits.append(f"{path}:{node.lineno}: {call_name}")
                continue
            if _is_open_write_call(node):
                hits.append(f"{path}:{node.lineno}: open(...).write")
                continue
            if _is_open_call(node.func):
                mode = _open_mode(node)
                if mode is None:
                    hits.append(f"{path}:{node.lineno}: open with dynamic mode")
                elif _is_write_mode(mode):
                    hits.append(f"{path}:{node.lineno}: open mode={mode!r}")

    assert hits == []

def test_retired_handoff_receipt_writers_are_not_approved_writer_modules() -> None:
    retired = {
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "verify_governance.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "run_" + "package_" + "smoke.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "collect_handoff_receipts.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/"
            "run_check.py"
        ),
        Path(
            ".agents/skills/quant-research-platform-maintenance/scripts/handoff/"
            "proof_contract.py"
        ),
    }

    assert APPROVED_WRITER_MODULES.isdisjoint(retired)

def test_direct_write_gate_detects_chained_open_write() -> None:
    tree = ast.parse("open('artifact.txt', 'w').write('unsafe')\n")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert any(_is_open_write_call(call) for call in calls)

def test_approved_writer_signatures_do_not_accept_arbitrary_physical_roots() -> None:
    writer_arg_sets = (
        _function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_stage_profile_outputs"),
        _function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_prediction_outputs"),
        _function_arg_names(RUN_OUTPUT_ARTIFACTS, "write_evaluation_outputs"),
    )

    assert "output_dir" not in _class_field_names(RUNTIME_PROFILING, "ProfileConfig")
    for arg_names in writer_arg_sets:
        assert "".join(("output_", "root")) not in arg_names
        assert "output_dir" not in arg_names
