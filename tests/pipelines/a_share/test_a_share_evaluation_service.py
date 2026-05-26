from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.evaluation.service import (
    AShareEvaluationRequest,
    run_a_share_evaluation,
)


def test_a_share_evaluation_service_delegates_to_processed_adapter(
    monkeypatch,
    tmp_path,
) -> None:
    from quant_research_platform.markets.implementations.a_share.evaluation import (
        service as service_module,
    )

    calls: list[dict[str, object]] = []

    def fake_run_processed_factor_evaluation(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame([{"rows": 3}]), Path("manifest.json"), []

    monkeypatch.setattr(
        service_module,
        "run_processed_factor_evaluation",
        fake_run_processed_factor_evaluation,
    )

    result = run_a_share_evaluation(
        AShareEvaluationRequest(
            processed_root=tmp_path / "processed",
            report_root=tmp_path / "report",
            start_year=2024,
            end_year=2024,
            requested_labels=("forward_return_1d",),
            requested_factors=("rev_20d",),
            n_groups=3,
            start_date="2024-01-01",
            end_date="2024-12-31",
            return_source_column="forward_return_1d",
            cost_bps=5.0,
            portfolio_modes=("long_only",),
            kernels=("dataframe",),
            source_alias="pipeline",
            manifest_run_id="service_smoke",
        )
    )

    assert result.manifest_path == Path("manifest.json")
    assert int(result.label_summary["rows"].max()) == 3
    assert calls == [
        {
            "processed_root": tmp_path / "processed",
            "report_root": tmp_path / "report",
            "start_year": 2024,
            "end_year": 2024,
            "requested_labels": ("forward_return_1d",),
            "requested_factors": ("rev_20d",),
            "all_factors": False,
            "n_groups": 3,
            "input_file": None,
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "return_source_column": "forward_return_1d",
            "cost_bps": 5.0,
            "portfolio_modes": ("long_only",),
            "kernels": ("dataframe",),
            "segment_columns": (),
            "risk_column": None,
            "neutralization_method": "none",
            "neutralization_group_columns": (),
            "neutralization_exposure_columns": (),
            "source_alias": "pipeline",
            "manifest_run_id": "service_smoke",
        }
    ]
