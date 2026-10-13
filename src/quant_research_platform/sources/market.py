"""Source composition for immutable native market inputs."""

from contextlib import contextmanager

from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_is_dir,
    filesystem_is_file,
    require_unaliased_path,
)


@contextmanager
def open_market_source(root, *, archive_paths=None, time_window=None):
    """Select one immutable native format; never fall back after validation fails."""
    from quant_research_platform.sources.binance.replay import open_recorder_run
    from quant_research_platform.sources.tardis.replay import open_tardis_archive

    root = require_unaliased_path(root, error="crypto_source_path_invalid")
    if filesystem_exists(root) and not filesystem_is_dir(root):
        raise ValueError("crypto_source_path_invalid")
    if any(
        filesystem_exists(root / name) and not filesystem_is_file(root / name)
        for name in ("run.json", "manifest.json")
    ):
        raise ValueError("crypto_source_manifest_invalid")
    if archive_paths is not None:
        with open_tardis_archive(
            root, archive_paths=archive_paths, time_window=time_window
        ) as source:
            yield source
        return
    official = filesystem_is_file(root / "run.json")
    historical = filesystem_is_file(root / "manifest.json")
    if official == historical:
        if official:
            raise ValueError("crypto_source_identity_ambiguous")
        raise FileNotFoundError("crypto_source_manifest_missing")
    opener = open_recorder_run if official else open_tardis_archive
    kwargs = {} if official else {"time_window": time_window}
    with opener(root, **kwargs) as source:
        yield source


def market_source_identity(root, _target_id, *, archive_paths=None, time_window=None):
    root = require_unaliased_path(root, error="crypto_source_path_invalid")
    if not filesystem_exists(root):
        return None
    if (
        archive_paths is None
        and filesystem_is_dir(root)
        and not any(
            filesystem_exists(root / name) for name in ("run.json", "manifest.json")
        )
    ):
        return None
    try:
        with open_market_source(
            root, archive_paths=archive_paths, time_window=time_window
        ) as source:
            return canonical_json_hash(source.source_identity)
    except FileNotFoundError as error:
        raise ValueError("crypto_source_member_missing") from error


def market_funding_identity(
    root, target_id, *, symbol, time_window, archive_paths=None
):
    from quant_research_platform.sources.binance.funding import funding_source_identity

    with open_market_source(
        root, archive_paths=archive_paths, time_window=time_window
    ) as source:
        return funding_source_identity(
            source.funding_roots(symbol, time_window=time_window),
            target_id,
            symbol=symbol,
        )
