"""Read-only catalogs and explicitly selected published artifacts."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.json_codec import parse_qrp_json_object
from quant_research_platform.interfaces.models import (
    FactorCatalogResponse,
    LabelCatalogResponse,
    ResearchDatasetRequirementsResponse,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.storage.manifests import read_qrp_json_file


def list_factors(profile: ResearchProfile) -> FactorCatalogResponse:
    return FactorCatalogResponse(
        items=[item.to_manifest() for item in profile.factors.list()],
    )


def list_labels(profile: ResearchProfile) -> LabelCatalogResponse:
    return LabelCatalogResponse(
        items=[item.to_manifest() for item in profile.labels.list()],
    )


def describe_research_dataset_requirements(
    dataset_id: str,
    profile: ResearchProfile,
) -> ResearchDatasetRequirementsResponse:
    spec = profile.research_datasets.get(dataset_id)
    return ResearchDatasetRequirementsResponse(
        status="available",
        mode="requirements_query_only",
        dataset_id=spec.data_id,
        market_id=str(spec.market_id),
        input_data_ids=list(spec.input_data_ids),
        factor_count=len(spec.factor_columns),
        feature_count=len(spec.feature_columns),
        label_count=len(spec.label_columns),
    )


def _ledger_curve(batches, *, stride):
    points, boundaries = [], []
    total_points = 0
    last_points = []
    last_retained_sequence = -1
    execution = {}
    with closing_iterator(iter(batches)) as batches:
        for batch in batches:
            last_points = [
                {
                    "sequence": batch.sequence,
                    "timestamp_ns": max(
                        row.timestamp_ns
                        for row in batch.valuations
                        if row.phase == phase
                    ),
                    "phase": phase,
                    "nav": sum(
                        row.value for row in batch.valuations if row.phase == phase
                    ),
                }
                for phase in ("before", "after", "end")
            ]
            total_points += len(last_points)
            has_marker = any(
                row.name in {"oos_window_start", "oos_window_end"}
                for row in batch.diagnostics
            )
            if batch.sequence % stride == 0 or has_marker:
                points.extend(last_points)
                last_retained_sequence = batch.sequence
            for marker in batch.diagnostics:
                if marker.name == "execution_summary" and marker.text is not None:
                    summary = parse_qrp_json_object(marker.text.encode("utf-8"))
                    execution.update(
                        (name, summary[name])
                        for name in (
                            "account_path_status",
                            "account_stop_reason",
                            "account_stop_ns",
                        )
                        if name in summary
                    )
                if marker.name not in {"oos_window_start", "oos_window_end"}:
                    continue
                boundaries.append(
                    {
                        "timestamp_ns": last_points[-1]["timestamp_ns"],
                        "kind": marker.name,
                        "window_index": int(marker.number),
                    }
                )
    if last_points and last_retained_sequence != last_points[-1]["sequence"]:
        points.extend(last_points)
    partial = execution.get("account_path_status") == "PARTIAL"
    return {
        "status": "PARTIAL" if partial else "DEFINED",
        **execution,
        **(
            {
                "observed_totals_scope": "known_prefix",
                "reason": execution["account_stop_reason"],
            }
            if partial
            else {}
        ),
        "observed_start_ns": points[0]["timestamp_ns"] if points else None,
        "observed_end_ns": last_points[-1]["timestamp_ns"] if last_points else None,
        "points": points,
        "boundaries": boundaries,
        "source_point_count": total_points,
        "displayed_point_count": len(points),
        "stride": stride,
        "selection": "every_nth_ledger_transition_plus_boundaries_and_last",
    }


def inspect(path: Path) -> dict[str, object]:
    """Read only the explicitly selected artifact's native metadata."""
    root = Path(path)
    if root.is_file():
        if root.name not in {
            "manifest.json",
            "spec.json",
            "metadata.json",
            "model.json",
        }:
            raise ValueError("inspection_requires_native_json_metadata")
        return read_qrp_json_file(root)[1]
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in (
        "manifest.json",
        "_manifest/manifest.json",
        "model.json",
        "spec.json",
        "metadata.json",
    ):
        candidate = root / name
        if candidate.is_file():
            return read_qrp_json_file(candidate)[1]
    raise ValueError("published_metadata_missing")


def portfolio_curve(path: Path, *, stride: int = 1, execution_mode="continuous"):
    """Project aggregate NAV, preserving independent episode boundaries."""
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )

    if type(stride) is not int or stride < 1:
        raise ValueError("portfolio_curve_stride_invalid")
    if execution_mode == "continuous":
        curve = {}
        artifact = read_portfolio_ledger_artifact(
            path,
            consume_batches=lambda _artifact, batches: curve.update(
                _ledger_curve(batches, stride=stride)
            ),
        )
        return {
            "source_reference": str(path),
            "window": artifact.metadata["window"],
            **(
                {"sample_role": artifact.metadata["sample_role"]}
                if "sample_role" in artifact.metadata
                else {}
            ),
            "episode_id": "continuous",
            "selection_mode": artifact.metadata.get("selection_mode"),
            "valuation_method": artifact.capital.valuation_method,
            "candidate_id": artifact.capital.candidate_id,
            "valuation_currency": artifact.capital.valuation_currency,
            "initial_capital": artifact.capital.initial_capital,
            "execution_mode": execution_mode,
            **curve,
        }
    if execution_mode != "independent_segments":
        raise ValueError("portfolio_curve_execution_mode_invalid")
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
    )

    episodes = []

    def consume_episode(index, child, batches):
        episodes.append(
            {
                **{
                    key: index[key]
                    for key in (
                        "episode_id",
                        "start_ns",
                        "stop_ns",
                        "terminal_valuation_ns",
                        "termination",
                        "stop_reasons",
                    )
                },
                "window": child.metadata["window"],
                **_ledger_curve(batches, stride=stride),
            }
        )

    artifact = read_crypto_episode_artifact(path, consume_episode=consume_episode)
    return {
        "status": "PARTIAL"
        if any(item["status"] == "PARTIAL" for item in episodes)
        else "DEFINED",
        "source_reference": str(path),
        "window": artifact.metadata["window"],
        **(
            {"sample_role": artifact.metadata["sample_role"]}
            if "sample_role" in artifact.metadata
            else {}
        ),
        "candidate_id": artifact.capital.candidate_id,
        "valuation_currency": artifact.capital.valuation_currency,
        "initial_capital": artifact.capital.initial_capital,
        "valuation_method": artifact.capital.valuation_method,
        "execution_mode": execution_mode,
        "episodes": episodes,
        "episode_count": len(episodes),
        "source_point_count": sum(item["source_point_count"] for item in episodes),
        "displayed_point_count": sum(
            item["displayed_point_count"] for item in episodes
        ),
        "stride": stride,
    }
