"""Native scientific catalogs for explicit producer fixtures."""

from types import SimpleNamespace

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.markets.implementations.a_share.processed import catalog
from quant_research_platform.sources.choice.interim import interim_catalog as choice
from quant_research_platform.sources.csmar.interim import interim_catalog as csmar
from quant_research_platform.sources.reference.interim import (
    interim_catalog as reference,
)


def scientific_catalogs():
    specs = {
        spec.data_id: spec
        for source in (csmar(), choice(), reference())
        for spec in source.list()
    }
    return SimpleNamespace(
        processed=catalog.processed_catalog(),
        interim=InterimDataCatalog(specs.values()),
    )
