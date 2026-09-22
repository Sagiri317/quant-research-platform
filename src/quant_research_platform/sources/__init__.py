"""Source composition for immutable native market inputs."""

from contextlib import contextmanager

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.storage.identity import require_unaliased_path


@contextmanager
def open_market_source(root, resource_policy=None):
    """Select one immutable native format; never fall back after validation fails."""
    from quant_research_platform.sources.binance.replay import open_recorder_run
    from quant_research_platform.sources.tardis.replay import open_tardis_archive

    root = require_unaliased_path(root, error="crypto_source_path_invalid")
    official = (root / "run.json").is_file()
    historical = (root / "manifest.json").is_file()
    if official == historical:
        if official:
            raise ValueError("crypto_source_identity_ambiguous")
        raise FileNotFoundError("crypto_source_manifest_missing")
    opener = open_recorder_run if official else open_tardis_archive
    with opener(root, resource_policy) as source:
        yield source


def market_source_identity(root, _target_id):
    try:
        with open_market_source(root) as source:
            return canonical_json_hash(source.source_identity)
    except FileNotFoundError:
        return None


def market_funding_identity(root, target_id, *, symbol):
    from quant_research_platform.sources.binance.funding import funding_source_identity

    with open_market_source(root) as source:
        return funding_source_identity(
            source.funding_root(symbol), target_id, symbol=symbol
        )
