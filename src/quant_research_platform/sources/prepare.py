"""Explicit source normalization with immutable output destinations."""

from collections.abc import Mapping
from pathlib import Path

from quant_research_platform.contracts.json_codec import qrp_json_mapping_snapshot
from quant_research_platform.contracts.time import TimeWindow


def prepare_source(config: Mapping):
    """Normalize selected vendor exports without resolving a project layout."""
    config = qrp_json_mapping_snapshot(config)
    required = {"source_id", "raw_root", "outputs"}
    if (
        not isinstance(config, Mapping)
        or not required <= set(config)
        or set(config)
        - required
        - {
            "start_date",
            "end_date",
            "chunksize",
            "fee_csv",
            "fee_sidecar",
            "instruments",
            "revision",
            "release_csv",
            "manifests",
        }
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
    start, end = config.get("start_date"), config.get("end_date")
    if start is not None or end is not None:
        TimeWindow(
            start if start is not None else end, end if end is not None else start
        )
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
    elif source == "venvoo":
        from quant_research_platform.sources.venvoo import materialize_interim_data

        if not {"instruments", "revision", "start_date", "end_date"} <= set(config):
            raise ValueError("level2_source_scope_required")
    elif source == "macro":
        from quant_research_platform.sources.macro.interim import (
            materialize_interim_data,
        )
    elif source == "public_research":
        from quant_research_platform.sources.public_research.interim import (
            materialize_interim_data,
        )
    else:
        raise ValueError("source_prepare_source_invalid")
    if "chunksize" in config and source != "csmar":
        raise ValueError("source_prepare_chunksize_not_consumed")
    if source != "venvoo" and {"instruments", "revision"} & set(config):
        raise ValueError("source_prepare_level2_parameters_not_consumed")
    if "release_csv" in config and (
        source not in {"macro", "public_research"}
        or type(config["release_csv"]) is not str
        or not config["release_csv"]
    ):
        raise ValueError("source_prepare_release_csv_invalid")
    if "manifests" in config and (
        source != "public_research"
        or type(config["manifests"]) is not list
        or any(type(value) is not str or not value for value in config["manifests"])
    ):
        raise ValueError("source_prepare_manifests_invalid")
    for key in ("fee_csv", "fee_sidecar"):
        if key in config and (
            source != "reference" or type(config[key]) is not str or not config[key]
        ):
            raise ValueError("source_prepare_fee_files_invalid")
    kwargs = {
        key: config[key]
        for key in (
            "start_date",
            "end_date",
            "chunksize",
            "fee_csv",
            "fee_sidecar",
            "instruments",
            "revision",
            "release_csv",
            "manifests",
        )
        if key in config
    }
    return materialize_interim_data(
        raw_root=Path(config["raw_root"]),
        output_roots={key: Path(value) for key, value in outputs.items()},
        data_ids=tuple(outputs),
        **kwargs,
    )
