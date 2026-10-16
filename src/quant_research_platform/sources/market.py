"""Source composition for immutable native market inputs."""

from contextlib import contextmanager

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
