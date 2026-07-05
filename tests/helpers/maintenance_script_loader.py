from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

MAINTENANCE_SCRIPT_ROOT = (
    Path(".agents")
    / "skills"
    / "quant-research-platform-maintenance"
    / "scripts"
)
HANDOFF_SCRIPT_ROOT = MAINTENANCE_SCRIPT_ROOT / "handoff"


def load_maintenance_script_module(name: str) -> ModuleType:
    return _load_script_module(name, MAINTENANCE_SCRIPT_ROOT / f"{name}.py")


def load_handoff_script_module(name: str) -> ModuleType:
    script = HANDOFF_SCRIPT_ROOT / f"{name}.py"
    return _load_script_module(name, script)


def _load_script_module(name: str, script: Path) -> ModuleType:
    for candidate in (HANDOFF_SCRIPT_ROOT, MAINTENANCE_SCRIPT_ROOT):
        candidate_text = str(candidate.resolve())
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
    for module_name in _maintenance_module_names(name):
        module = sys.modules.get(module_name)
        module_file = Path(str(getattr(module, "__file__", "")))
        if module is not None and not module_file.is_relative_to(Path(".").resolve()):
            sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _maintenance_module_names(name: str) -> tuple[str, ...]:
    if name in {
        "build_handoff_package",
        "handoff_contract",
        "receipt_contract",
        "replay_evidence",
        "verify_handoff_package",
    }:
        return (
            "build_handoff_package",
            "handoff_contract",
            "receipt_contract",
            "replay_evidence",
            "verify_handoff_package",
        )
    return (name,)
