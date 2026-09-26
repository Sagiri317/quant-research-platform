"""Production experiment attempt semantics."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite
from uuid import uuid4

from quant_research_platform.contracts.experiments import (
    ExperimentPlan,
    TrialRecord,
    trial_ledger_semantic_identity,
)
from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.research.models.supervised import model_thread_count
from quant_research_platform.storage.experiments import (
    append_trial_record,
    read_trial_ledger,
)
from quant_research_platform.storage.layout import StorageLayout


class TrialRecorder:
    """One registered family/fold attempt; unstarted candidates have no events."""

    def __init__(
        self, layout: StorageLayout, run_id: str, plan: ExperimentPlan
    ) -> None:
        self.plan = plan
        self.layout = layout
        self.run_id = run_id
        self.root = layout.trial_ledger_root(run_id, plan.experiment_id)
        self.index = 0

    def begin(self, spec: ModelSpec, *, common_coverage_id: str) -> TrialRecord:
        record = TrialRecord(
            run_id=self.run_id,
            event_id=f"event_{uuid4().hex}",
            attempt_index=self.index + 1,
            experiment_id=self.plan.experiment_id,
            protocol_hash=self.plan.protocol_hash,
            candidate_family_id=self.plan.candidate_family_id,
            candidate_id=spec.model_id,
            trial_id=f"trial_{uuid4().hex}",
            fold_id=self.plan.folds[0].fold_id,
            market_id=str(self.plan.market_id),
            research_dataset_id=self.plan.research_dataset_id,
            target_id=self.plan.target_id,
            horizon_id=spec.label_column,
            common_coverage_id=common_coverage_id,
            factor_definition_ids=spec.input_factor_columns,
            model_definition_id=spec.model_id,
            normalized_parameters=model_spec_payload(spec),
            random_seed=spec.random_seed,
            backend_versions=self.plan.backend_versions,
            thread_settings={
                "blas_threads": 1,
                "native_threads": model_thread_count(spec),
            },
            status="attempted",
            artifact_lineage=self.plan.input_lineage,
        )
        append_trial_record(self.root, self.plan, record)
        self.index = record.attempt_index
        return record

    def begin_refit(self, spec: ModelSpec, *, common_coverage_id: str):
        """Bind a real prepared fit domain to its completed selection ledger."""
        plan = replace(
            self.plan,
            stage="refit",
            experiment_id=f"experiment_{uuid4().hex}",
            selection_metric=None,
            selection_direction=None,
            tie_break_order=(),
            candidate_ids=(spec.model_id,),
            trial_budget=1,
            random_seed=spec.random_seed,
            candidate_parameters={spec.model_id: model_spec_payload(spec)},
            selection_ledger_semantic_identity=self.reference()["semantic_identity"],
        )
        recorder = TrialRecorder(self.layout, self.run_id, plan)
        return recorder, recorder.begin(spec, common_coverage_id=common_coverage_id)

    def finish(
        self,
        record: TrialRecord,
        *,
        metrics: dict[str, object] | None = None,
        model_metadata: dict[str, object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        values = {}
        if metrics is not None:
            for key in (
                "validation_ic",
                "validation_rankic",
                "validation_mse",
                "validation_ic_reason",
                "validation_rankic_reason",
                "validation_rows",
            ):
                value = metrics[key]
                if hasattr(value, "item"):
                    value = value.item()
                if type(value) is float and not isfinite(value):
                    value = None
                values[key] = value
        threads = dict(record.thread_settings)
        if model_metadata is not None:
            threads["native_threads"] = model_metadata["resolved_hyperparameters"].get(
                "n_jobs", 1
            )
            values["n_train"] = model_metadata["rows_after_input_label_dropna"]
        status = (
            "completed"
            if error is None
            else "failed"
            if isinstance(error, Exception)
            else "abandoned"
        )
        terminal = replace(
            record,
            event_id=f"event_{uuid4().hex}",
            status=status,
            metrics=values,
            thread_settings=threads,
            undefined_reasons={}
            if error is None
            else {"failure": type(error).__name__},
        )
        append_trial_record(self.root, self.plan, terminal)

    def reference(self) -> dict[str, str]:
        with read_trial_ledger(self.root, self.plan) as (plan, records):
            return {
                "run_id": self.run_id,
                "experiment_id": self.plan.experiment_id,
                "protocol_hash": self.plan.protocol_hash,
                "semantic_identity": trial_ledger_semantic_identity(self.plan, records),
            }
