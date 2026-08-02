from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ImportEdge,
    _import_edges_from_trees,
    import_edges_under,
)

SOURCE_ROOT = Path("src/quant_research_platform")
PACKAGE_PREFIX = "quant_research_platform"
LAYER_BOUNDARIES = {
    "contracts": (
        "quant_research_platform.compute",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.runtime",
        "quant_research_platform.sources",
        "quant_research_platform.storage",
        "quant_research_platform.workflows",
    ),
    "sources": (
        "quant_research_platform.artifacts",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.research",
        "quant_research_platform.workflows",
    ),
    "policy": (
        "quant_research_platform.artifacts",
        "quant_research_platform.catalog",
        "quant_research_platform.compute",
        "quant_research_platform.contracts",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets",
        "quant_research_platform.reports",
        "quant_research_platform.research",
        "quant_research_platform.runtime",
        "quant_research_platform.sources",
        "quant_research_platform.storage",
        "quant_research_platform.workflows",
    ),
    "catalog": (
        "quant_research_platform.markets.implementations",
        "quant_research_platform.reports",
        "quant_research_platform.sources",
    ),
    "runtime": ("quant_research_platform.reports",),
    "research": (
        "quant_research_platform.artifacts",
        "quant_research_platform.markets.implementations",
        "quant_research_platform.reports",
        "quant_research_platform.sources",
    ),
    "reports": (
        "quant_research_platform.compute",
        "quant_research_platform.research",
        "quant_research_platform.artifacts",
        "quant_research_platform.workflows",
        "quant_research_platform.interfaces",
        "quant_research_platform.markets.implementations",
        "quant_research_platform.storage",
    ),
    "artifacts": (
        "quant_research_platform.research",
        "quant_research_platform.reports",
        "quant_research_platform.markets.implementations",
    ),
    "compute": (
        "quant_research_platform.reports",
        "quant_research_platform.artifacts",
        "quant_research_platform.workflows",
        "quant_research_platform.interfaces",
    ),
    "interfaces": (
        "quant_research_platform.sources",
        "quant_research_platform.markets.implementations",
    ),
}


def test_active_source_modules_import_without_side_effect_failures() -> None:
    failures: list[str] = []
    for module_name in _active_module_names():
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - reports every failed import.
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    assert failures == []


def test_owner_layer_import_boundaries_are_enforced(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    violations: list[str] = []
    for root, forbidden_prefixes in LAYER_BOUNDARIES.items():
        for edge in import_edges_under(import_edges, SOURCE_ROOT / root):
            if any(
                edge.module.startswith(prefix)
                for prefix in forbidden_prefixes
            ):
                violations.append(
                    f"{edge.path.as_posix()}:{edge.lineno}:{edge.module}"
                )

    assert violations == []


@pytest.mark.parametrize(
    ("path", "source", "module"),
    (
        (
            SOURCE_ROOT / "contracts/example.py",
            "from quant_research_platform.storage import datasets\n",
            "quant_research_platform.storage.datasets",
        ),
        (
            SOURCE_ROOT / "runtime/example.py",
            "from quant_research_platform.reports import verification_summary\n",
            "quant_research_platform.reports.verification_summary",
        ),
        (
            SOURCE_ROOT / "catalog/example.py",
            "from quant_research_platform import reports\n",
            "quant_research_platform.reports",
        ),
        (
            SOURCE_ROOT / "research/example.py",
            "from ..markets.implementations import a_share\n",
            "quant_research_platform.markets.implementations.a_share",
        ),
        (
            SOURCE_ROOT / "reports/example.py",
            "from quant_research_platform.storage import layout\n",
            "quant_research_platform.storage.layout",
        ),
    ),
)
def test_owner_layer_boundaries_reject_real_dependency_directions(
    path: Path,
    source: str,
    module: str,
) -> None:
    edge = _import_edges_from_trees({path: ast.parse(source)})[0]
    root = path.relative_to(SOURCE_ROOT).parts[0]

    assert edge.module == module
    assert any(module.startswith(prefix) for prefix in LAYER_BOUNDARIES.get(root, ()))


def test_status_aggregation_rejects_pass_with_not_run_child() -> None:
    from quant_research_platform.policy.check_status import aggregate_check_status

    assert aggregate_check_status(["PASS", "NOT_RUN"]) == "FAIL"


def test_physical_writes_require_storage_or_artifacts_owner() -> None:
    violations = _physical_writer_violations(
        {
            SOURCE_ROOT / "contracts" / "bad.py": "\n".join(
                (
                    "import builtins",
                    "from pathlib import Path",
                    "from pyarrow import parquet as pq",
                    "from zipfile import ZipFile",
                    "import shutil",
                    "builtins.open('output', 'w')",
                    "open('output', 'a')",
                    "Path('output').open('x')",
                    "Path('output').write_text('x')",
                    "Path('output').write_bytes(b'x')",
                    "pq.write_table(None, 'output')",
                    "frame.to_parquet('output')",
                    "pq.ParquetWriter('output')",
                    "ZipFile('output', 'w')",
                    "shutil.copyfile('input', 'output')",
                    "open('input', 'rb')",
                    "Path('input').open('r')",
                )
            ),
            SOURCE_ROOT / "storage" / "allowed.py": "open('output', 'w')\n",
        }
    )

    assert [target for _path, _line, target in violations] == [
        "open",
        "open",
        "open",
        "write_text",
        "write_bytes",
        "write_table",
        "to_parquet",
        "ParquetWriter",
        "ZipFile",
        "copyfile",
    ]


def test_live_source_physical_writes_stay_in_writer_owners() -> None:
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in SOURCE_ROOT.rglob("*.py")
    }

    assert _physical_writer_violations(sources) == []


def _physical_writer_violations(
    sources: dict[Path, str],
) -> list[tuple[str, int, str]]:
    owner_roots = (SOURCE_ROOT / "storage", SOURCE_ROOT / "artifacts")
    always_writes = {
        "ParquetWriter",
        "copyfile",
        "write_bytes",
        "write_table",
        "write_text",
    }
    violations: list[tuple[str, int, str]] = []
    for path, source in sorted(sources.items()):
        if any(path == root or path.is_relative_to(root) for root in owner_roots):
            continue
        tree = ast.parse(source, filename=path.as_posix())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            target = (
                call.func.id
                if isinstance(call.func, ast.Name)
                else call.func.attr
                if isinstance(call.func, ast.Attribute)
                else None
            )
            writes = target in always_writes
            if target == "to_parquet":
                writes = bool(call.args) or any(
                    keyword.arg == "path"
                    and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is None
                    )
                    for keyword in call.keywords
                )
            elif target == "open":
                position = (
                    1
                    if isinstance(call.func, ast.Name)
                    or (
                        isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "builtins"
                    )
                    else 0
                )
                writes = _static_write_mode(call, position)
            elif target == "ZipFile":
                writes = _static_write_mode(call, 1)
            if writes:
                violations.append((path.as_posix(), call.lineno, str(target)))
    return violations


def _static_write_mode(call: ast.Call, position: int) -> bool:
    mode: ast.AST | None = call.args[position] if len(call.args) > position else None
    if mode is None:
        mode = next(
            (
                keyword.value
                for keyword in call.keywords
                if keyword.arg == "mode"
            ),
            None,
        )
    return (
        isinstance(mode, ast.Constant)
        and isinstance(mode.value, str)
        and any(flag in mode.value for flag in "wax+")
    )


def _active_module_names() -> tuple[str, ...]:
    names: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT)
        module_parts = (
            relative.parent.parts
            if relative.name == "__init__.py"
            else (*relative.parent.parts, relative.stem)
        )
        names.append(".".join((PACKAGE_PREFIX, *module_parts)))
    return tuple(dict.fromkeys(names))
