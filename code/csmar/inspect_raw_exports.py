"""Inspect raw CSMAR export filenames, headers, and metadata sidecars.

This script reads only table headers with pandas ``nrows=0``. It does not load
full CSMAR data files.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from csmar.raw_exports import (  # noqa: E402
    inspect_raw_export_tree,
    iter_raw_export_files,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check CSMAR raw export filenames, headers, and metadata sidecars "
            "without reading full tables."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="data/raw_private/csmar",
        help="Private CSMAR raw directory to inspect.",
    )
    parser.add_argument(
        "--write-metadata",
        action="store_true",
        help="Generate or update .metadata.json sidecars during inspection.",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        return 2

    files = iter_raw_export_files(root)
    if not files:
        print(f"No CSMAR export files found under: {root}")
        return 0

    exit_code = 0
    results = inspect_raw_export_tree(root, write_metadata=args.write_metadata)
    for result in results:
        print(f"\n== {result.path} ==")
        if result.error:
            exit_code = 1
            print(f"ERROR: {result.error}")
            continue

        inspection = result.inspection
        assert inspection is not None
        print(f"dataset: {inspection.dataset}")
        print(f"table_code: {inspection.table_code}")
        print(f"columns: {len(inspection.columns)}")
        print(f"metadata: {'yes' if inspection.has_metadata else 'missing'}")
        if result.metadata_path is not None:
            print(f"metadata_written: {result.metadata_path}")
        if inspection.missing_required_columns:
            exit_code = 1
            print("missing_required_columns:")
            for column in inspection.missing_required_columns:
                print(f"  - {column}")
        else:
            print("required_columns: ok")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
