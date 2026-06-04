"""Thin compatibility facade for internal provider contracts.

The implementation lives in ``raw_schemas.py``, ``field_mappings.py``, and
``checksums.py``. This module is internal-only and is not imported by API/CLI
public data-product surfaces.
"""

from quant_research_platform.sources.csmar.raw_schemas import *  # noqa: F403
