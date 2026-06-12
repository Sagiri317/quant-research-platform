from __future__ import annotations

import importlib
from pathlib import Path

PACKAGE_ROOT = Path("src/quant_research_platform")
PACKAGE_PREFIX = "quant_research_platform"


def test_active_source_modules_import_without_side_effect_failures() -> None:
    modules = tuple(_active_module_names())
    failures: list[str] = []

    assert (
        "quant_research_platform.markets.implementations.a_share.validation"
        in modules
    )
    assert (
        "quant_research_platform.markets.implementations.a_share.validation.research_panel"
        not in modules
    )

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - failure path reports all modules.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []


def _active_module_names() -> tuple[str, ...]:
    names: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if any(part == "__pycache__" for part in path.parts):
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.name == "__init__.py":
            module_parts = relative.parent.parts
        else:
            module_parts = (*relative.parent.parts, relative.stem)
        module_name = ".".join((PACKAGE_PREFIX, *module_parts))
        names.append(module_name)
    return tuple(dict.fromkeys(names))
