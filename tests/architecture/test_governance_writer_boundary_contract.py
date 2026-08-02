from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.maintenance_script_loader import load_maintenance_script_module

governance_contract = load_maintenance_script_module("governance_contract")


def test_live_source_has_no_direct_writer_owner_bypass() -> None:
    assert governance_contract._writer_boundary_violations(Path(".")) == []


@pytest.mark.parametrize(
    ("source", "line", "target"),
    (
        ("open('output', 'w')\n", 1, "builtins.open"),
        (
            "from pathlib import Path\nPath('output').write_text('x')\n",
            2,
            "pathlib.Path().write_text",
        ),
        (
            "from pyarrow import parquet as pq\npq.write_table(None, 'output')\n",
            2,
            "pyarrow.parquet.write_table",
        ),
        (
            "import zipfile as archive\narchive.ZipFile('output', 'w')\n",
            2,
            "zipfile.ZipFile",
        ),
    ),
)
def test_statically_explicit_physical_writes_require_a_writer_owner(
    source: str,
    line: int,
    target: str,
) -> None:
    path = Path("src/quant_research_platform/contracts/example.py")

    assert governance_contract._writer_boundary_violations(
        Path("."),
        {path: source},
    ) == [
        f"{path}:{line}: direct writer outside owner: {target}"
    ]


def test_writer_gate_allows_owner_writes_and_unproven_dynamic_calls() -> None:
    owner = Path("src/quant_research_platform/storage/example.py")
    ordinary = Path("src/quant_research_platform/contracts/example.py")

    assert governance_contract._writer_boundary_violations(
        Path("."),
        {
            owner: "import pathlib\npathlib.Path('output').write_text('x')\n",
            ordinary: (
                "from pathlib import Path\n"
                "def use(Path, open, value):\n"
                "    Path('domain').write_text('domain value')\n"
                "    open('domain', 'w')\n"
                "    value.write_text('domain value')\n"
                "    return getattr(value, 'write_text')('domain value')\n"
            ),
        },
    ) == []


def test_read_only_file_calls_are_not_writer_bypasses() -> None:
    path = Path("src/quant_research_platform/contracts/example.py")
    source = (
        "import zipfile\n"
        "open('input', 'r')\n"
        "zipfile.ZipFile('input')\n"
    )

    assert governance_contract._writer_boundary_violations(
        Path("."),
        {path: source},
    ) == []
