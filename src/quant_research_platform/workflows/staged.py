"""Public staged workflow API used by both CLI and installed-package smoke."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import ResearchProfile
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.layout import StorageLayout


def data_materialize(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    raw_root: Path,
    start_date: str | None,
    end_date: str | None,
    data_ids: tuple[str, ...] | None = None,
    plan: bool = False,
) -> object:
    if plan:
        selected = data_ids or tuple(spec.data_id for spec in profile.interim.list())
        return {
            "stage": "data materialize",
            "plan": True,
            "data_ids": list(selected),
            "source_group_count": len(
                {str(spec.source_id) for spec in profile.interim.list()}
            ),
        }
    result = profile.materialize_interim_data(
        layout=layout,
        raw_root=raw_root,
        data_ids=data_ids,
        start_date=start_date,
        end_date=end_date,
    )
    if hasattr(result, "artifact_roots"):
        roots = result.artifact_roots
        skipped = tuple(result.skipped)
        return {
            "stage": "data materialize",
            "status": "BLOCKED_EXTERNAL_DATA" if skipped else "READY",
            "data_ids": sorted(str(data_id) for data_id in roots),
            "skipped": list(skipped),
        }
    return result


def dataset_build(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    dataset_id: str,
    start_date: str,
    end_date: str,
    plan: bool = False,
    rebuild: bool = False,
) -> object:
    if plan:
        spec = profile.research_datasets.get(dataset_id)
        return {
            "stage": "dataset build",
            "plan": True,
            "dataset_id": spec.data_id,
            "input_data_ids": list(spec.input_data_ids),
            "rebuild": rebuild,
        }
    from quant_research_platform.research.datasets.assembler import (
        assemble_research_dataset,
    )

    with runtime_resource_policy_scope(None) as policy:
        profile.materialize_processed_data(
            layout=layout,
            time_window=TimeWindow(start_date, end_date),
            data_ids=tuple(
                profile.processed.get(data_id).data_id
                for data_id in _processed_targets(profile, dataset_id)
            ),
            resource_policy=policy,
            rebuild=rebuild,
        )
        return assemble_research_dataset(
            layout=layout,
            dataset_catalog=profile.research_datasets,
            processed_catalog=profile.processed,
            dataset_id=dataset_id,
            time_window=TimeWindow(start_date, end_date),
            resource_policy=policy,
        )


def research_run(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    scenario_id: str,
    phase: str,
    start_date: str,
    end_date: str,
    plan: bool = False,
    rebuild: bool = False,
    protocol_frozen: bool = False,
) -> object:
    from quant_research_platform.research.protocols.freeze import require_phase_access

    require_phase_access(phase=phase, protocol_frozen=protocol_frozen)
    if plan:
        from quant_research_platform.research.protocols.a_share import protocol

        return {
            "stage": "research run",
            "plan": True,
            "scenario_id": scenario_id,
            "phase": phase,
            "rebuild": rebuild,
            "protocol": protocol(scenario_id, phase).manifest(),
        }
    dataset = profile.research_datasets.get(scenario_id)
    root = layout.research_dataset_root(dataset.market_id, dataset.data_id)
    if not root.exists():
        raise ValueError(f"missing_research_dataset: {scenario_id}")
    from quant_research_platform.reports.a_share import build_a_share_report

    report = build_a_share_report(
        scenario_id=scenario_id,
        phase=phase,
        data={"readiness/blockers": "derived from current artifact status"},
    )
    return {
        "stage": "research run",
        "status": "READY",
        "scenario_id": scenario_id,
        "phase": phase,
        "rebuild": rebuild,
        "window": {"start": start_date, "end": end_date},
        "dataset_root": str(root),
        "report": report,
    }


def report_build(
    *,
    profile: ResearchProfile,
    layout: StorageLayout,
    scenario_id: str,
    phase: str,
    plan: bool = False,
    rebuild: bool = False,
    protocol_frozen: bool = False,
) -> object:
    from quant_research_platform.research.protocols.freeze import require_phase_access

    require_phase_access(phase=phase, protocol_frozen=protocol_frozen)
    if plan:
        return {
            "stage": "report build",
            "plan": True,
            "scenario_id": scenario_id,
            "phase": phase,
            "rebuild": rebuild,
            "inputs": [scenario_id],
        }
    dataset = profile.research_datasets.get(scenario_id)
    root = layout.research_dataset_root(dataset.market_id, dataset.data_id)
    if not root.exists():
        raise ValueError(f"missing_research_dataset: {scenario_id}")
    from quant_research_platform.reports.a_share import build_a_share_report

    report = build_a_share_report(
        scenario_id=scenario_id,
        phase=phase,
        data={"readiness/blockers": "derived from current artifact status"},
    )
    return {
        "stage": "report build",
        "status": "READY",
        "scenario_id": scenario_id,
        "phase": phase,
        "rebuild": rebuild,
        "report_source": str(root),
        "report": report,
    }


def status(*, profile: ResearchProfile, layout: StorageLayout) -> dict[str, object]:
    return {
        "market_id": str(profile.market_id),
        "readiness": _readiness(profile, layout),
        "interim": {
            spec.data_id: (
                "READY"
                if layout.interim_data_root(spec.source_id, spec.data_id).exists()
                else "BLOCKED_EXTERNAL_DATA"
            )
            for spec in profile.interim.list()
        },
        "research_datasets": {
            spec.data_id: (
                "READY"
                if layout.research_dataset_root(spec.market_id, spec.data_id).exists()
                else "BLOCKED_DATA"
            )
            for spec in profile.research_datasets.list()
        },
    }


def _readiness(profile: ResearchProfile, layout: StorageLayout) -> dict[str, str]:
    available = {
        spec.data_id
        for spec in profile.interim.list()
        if layout.interim_data_root(spec.source_id, spec.data_id).exists()
    }
    return {
        "price_volume": (
            "READY"
            if {
                "market_bars_1d",
                "stock_master",
                "market_cap",
                "trading_calendar",
            }.issubset(available)
            else "BLOCKED_EXTERNAL_DATA"
        ),
        "financial_statements": (
            "READY"
            if {
                "financial_balance",
                "financial_income",
                "financial_cashflow",
            }.issubset(available)
            else "BLOCKED_EXTERNAL_DATA"
        ),
        "financial_first_disclosure": (
            "READY"
            if "financial_first_disclosure" in available
            else "BLOCKED_EXTERNAL_DATA"
        ),
        "financial_correction": (
            "READY"
            if {
                "financial_balance",
                "financial_income",
                "financial_cashflow",
            }.issubset(available)
            else "BLOCKED_EXTERNAL_DATA"
        ),
        "analyst_forecast": (
            "READY" if "analyst_forecast" in available else "BLOCKED_EXTERNAL_DATA"
        ),
        "benchmark_total_return": (
            "READY" if "index_market" in available else "BLOCKED_EXTERNAL_DATA"
        ),
        "benchmark_exact_weights": (
            "READY" if "index_weights" in available else "BLOCKED_EXTERNAL_DATA"
        ),
        "fee_schedule": "BLOCKED_EXTERNAL_DATA",
    }


def inspect(
    *, profile: ResearchProfile, layout: StorageLayout, dataset_id: str
) -> dict[str, object]:
    spec = profile.research_datasets.get(dataset_id)
    root = layout.research_dataset_root(spec.market_id, spec.data_id)
    return {
        "dataset_id": dataset_id,
        "root": str(root),
        "exists": root.exists(),
        "required_columns": list(spec.required_columns),
    }


def _processed_targets(profile: ResearchProfile, dataset_id: str) -> tuple[str, ...]:
    dataset = profile.research_datasets.get(dataset_id)
    return tuple(
        data_id
        for data_id in dataset.input_data_ids
        if data_id in {spec.data_id for spec in profile.processed.list()}
    )
