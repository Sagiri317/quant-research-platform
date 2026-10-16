"""Stable, independent local research random streams."""

import pytest

from quant_research_platform.contracts.experiments import DEFAULT_ROOT_SEED
from quant_research_platform.research.randomness import derive_seed


def test_derived_seeds_are_stable_and_do_not_depend_on_call_order():
    assert DEFAULT_ROOT_SEED == 42
    assert derive_seed(42, "fit", "model-a", "fold-1") == 3362232625
    identities = [
        ("fit", "model-a", "fold-1"),
        ("fit", "model-b", "fold-1"),
        ("fit", "model-a", "fold-2"),
        ("explain", "model-a", "fold-1"),
    ]
    first = {parts: derive_seed(42, *parts) for parts in identities}
    second = {parts: derive_seed(42, *parts) for parts in reversed(identities)}
    assert first == second
    assert len(set(first.values())) == len(identities)
    assert all(0 <= seed < 2**32 for seed in first.values())
    assert derive_seed(42, "shared", {"a": 1, "b": 2}) == derive_seed(
        42, "shared", {"b": 2, "a": 1}
    )
    assert derive_seed(42, "shared") != derive_seed(1, "shared")


@pytest.mark.parametrize("seed", [None, True, -1, 1.0])
def test_invalid_root_seed_rejected(seed):
    with pytest.raises(ValueError, match="root_seed"):
        derive_seed(seed, "fit")
