"""Explicit source normalization with immutable output destinations."""

from collections.abc import Mapping
from pathlib import Path

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot


def prepare_source(config: Mapping):
    """Normalize selected vendor exports without resolving a project layout."""
    config = qrp_json_mapping_snapshot(config)
    required = {"source_id", "raw_root", "outputs"}
    if (
        not isinstance(config, Mapping)
        or not required <= set(config)
        or set(config) - required - {"start_date", "end_date", "chunksize"}
    ):
        raise ValueError("source_prepare_fields_invalid")
    if type(config["raw_root"]) is not str or not config["raw_root"]:
        raise ValueError("source_prepare_root_invalid")
    outputs = config["outputs"]
    if (
        not isinstance(outputs, Mapping)
        or not outputs
        or any(
            type(k) is not str or not k or type(v) is not str or not v
            for k, v in outputs.items()
        )
    ):
        raise ValueError("source_prepare_outputs_invalid")
    if len({Path(value).resolve() for value in outputs.values()}) != len(outputs):
        raise ValueError("source_prepare_output_paths_duplicate")
    source = config["source_id"]
    if source == "csmar":
        from quant_research_platform.sources.csmar.interim import (
            materialize_interim_data,
        )
    elif source == "choice":
        from quant_research_platform.sources.choice.interim import (
            materialize_interim_data,
        )
    elif source == "reference":
        from quant_research_platform.sources.reference.interim import (
            materialize_interim_data,
        )
    else:
        raise ValueError("source_prepare_source_invalid")
    if "chunksize" in config and source != "csmar":
        raise ValueError("source_prepare_chunksize_not_consumed")
    kwargs = {
        key: config[key]
        for key in ("start_date", "end_date", "chunksize")
        if key in config
    }
    return materialize_interim_data(
        raw_root=Path(config["raw_root"]),
        output_roots={key: Path(value) for key, value in outputs.items()},
        data_ids=tuple(outputs),
        **kwargs,
    )
