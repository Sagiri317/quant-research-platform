from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.import_graph import (
    ImportEdge,
    import_edges_for_roots,
)


@pytest.fixture(scope="session")
def import_edges() -> tuple[ImportEdge, ...]:
    return import_edges_for_roots((Path("src"),))
