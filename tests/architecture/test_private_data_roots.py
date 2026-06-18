from __future__ import annotations

from pathlib import Path


def test_gitignore_excludes_generated_research_data() -> None:
    lines = {
        line.strip()
        for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()
    }

    assert "data/research/" in lines
