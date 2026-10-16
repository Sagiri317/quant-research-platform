"""Deterministic randomness shared by every research protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Final

UINT32_MODULUS: Final[int] = 2**32


def derive_seed(root_seed: int, component_id: str, *parts: object) -> int:
    """Derive a stable uint32-compatible seed without Python's randomized hash."""

    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed_must_be_nonnegative_integer")
    if type(component_id) is not str or not component_id:
        raise ValueError("random_component_id_required")
    payload = json.dumps(
        [root_seed, component_id, *parts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    ).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % UINT32_MODULUS
