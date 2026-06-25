from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.research_scenarios import (
    build_prediction_specs,
    default_prediction_smoke_id,
)
from quant_research_platform.markets.implementations.a_share.verification import (
    aggregates,
)
from tests.helpers import frequency_ids


def test_aggregates_summarizes_frames_without_row_level_output() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_b"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "value": [1.0, 1.0, 2.0],
            "flag": [True, False, True],
        }
    )

    summary = aggregates._summarize_frames([frame])

    assert summary.row_count == 3
    assert summary.key_count == 2
    assert summary.duplicate_key_count == 1
    assert summary.period == "2024-01-02 through 2024-01-03"
    assert summary.column_count == 5
    assert summary.dtype_summary_hash == aggregates._sha256_json(summary.dtype_summary)
    assert len(summary.aggregate_checksum) == 64
    assert "value" in summary.content_summary
    assert summary.content_summary["flag"]["numeric_max"] == 1.0


def test_aggregates_combines_summaries_and_area_records(monkeypatch) -> None:
    first = aggregates._summarize_frames(
        [
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-02"]),
                    "signal": ["mom_signal"],
                    "label": ["ret"],
                    "group": [1],
                    "value": [1.0],
                }
            )
        ]
    )
    second = aggregates._summarize_frames(
        [
            pd.DataFrame(
                {
                    "session_date": pd.to_datetime(["2024-01-03"]),
                    "value": [2.0],
                }
            )
        ]
    )
    combined = aggregates._combine_summaries([first, second])

    assert combined.row_count == 2
    assert combined.period == "2024-01-02 through 2024-01-03"

    monkeypatch.setattr(
        aggregates,
        "_summarize_area_input",
        lambda *args, **kwargs: first,
    )
    ok = aggregates._area_record(
        aggregates.AggregateInput(
            area="factor_value_parity",
            roots=(Path("unused"),),
            expected_period=first.period,
            explanation_name="factor value",
        )
    )
    gap = aggregates._area_record(
        aggregates.AggregateInput(
            area="factor_value_parity",
            roots=(Path("unused"),),
            expected_period="2024-01-01 through 2024-01-01",
            explanation_name="factor value",
        )
    )

    assert ok["human_approval_required"] is False
    assert ok["accepted_explained_delta"] is False
    assert gap["human_approval_required"] is True
    assert gap["max_abs_delta"] == 1.0


def test_review_safe_aggregate_evidence_orchestrates_area_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary = aggregates._summarize_frames(
        [
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-02"]),
                    "value": [1.0],
                }
            )
        ]
    )
    monkeypatch.setattr(
        aggregates,
        "_area_inputs",
        lambda root: (
            aggregates.AggregateInput(
                area="factor_value_parity",
                roots=(root / "factor",),
                expected_period=summary.period,
                explanation_name="factor value",
            ),
        ),
    )
    monkeypatch.setattr(
        aggregates,
        "_summarize_area_input",
        lambda *args, **kwargs: summary,
    )

    payload = aggregates.build_review_safe_aggregate_evidence_payload(tmp_path)

    assert payload["handoff_ready"] is True
    assert payload["verification_periods"] == [summary.period]
    assert payload["areas"][0]["area"] == "factor_value_parity"


def test_aggregates_area_inputs_and_inventory_helpers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    area_inputs = aggregates._area_inputs(tmp_path)
    assert {item.area for item in area_inputs} >= {
        "processed_panel_parity",
        "factor_value_parity",
        frequency_ids.full_historical_market_panel_status_area(),
    }
    run_output_area_inputs = {
        item.area: item.summary_kind
        for item in area_inputs
        if item.area in {"evaluation_parity", "prediction_oos_smoke_parity"}
    }
    assert run_output_area_inputs == {
        "evaluation_parity": "run_outputs",
        "prediction_oos_smoke_parity": "run_outputs",
    }
    prediction_spec = build_prediction_specs(
        train_end="2014-06-30",
        test_start="2014-07-01",
        model_algorithm_ids=("linear_regression",),
    )[0]
    assert prediction_spec.prediction_id == default_prediction_smoke_id(
        test_start="2014-07-01"
    )
    prediction_area = next(
        item for item in area_inputs if item.area == "prediction_oos_smoke_parity"
    )
    assert prediction_area.expected_period == (
        f"{prediction_spec.test_start} through 2014-12-31"
    )

    artifact_root = frequency_ids.market_panel_processed_root(tmp_path)
    artifact_root.mkdir(parents=True)
    monkeypatch.setattr(
        aggregates,
        "_official_dataset_artifact_roots",
        lambda _root: (artifact_root,),
    )
    monkeypatch.setattr(
        aggregates,
        "_summarize_dataset_root",
        lambda *args, **kwargs: aggregates._summarize_frames(
            [
                pd.DataFrame(
                    {
                        "market_id": ["a_share"],
                        "asset_id": ["asset_a"],
                        "timestamp": pd.to_datetime(["2024-01-02"]),
                    }
                )
            ]
        ),
    )

    inventory = aggregates._dataset_inventory(tmp_path)

    assert inventory[0]["kind"] == "panel"
    assert inventory[0]["artifact_id"] == frequency_ids.MARKET_PANEL_DATA_ID
    assert inventory[0]["covered_by_evidence_area"] == (
        frequency_ids.full_historical_market_panel_status_area()
    )
    assert aggregates._logical_dataset_root(tmp_path, artifact_root) == (
        frequency_ids.market_panel_logical_root()
    )
    assert aggregates._dataset_kind_and_artifact_id(
        "research/a_share/demo_dataset"
    ) == ("research_dataset", "demo_dataset")


def test_aggregates_official_roots_use_inspection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_root = frequency_ids.market_panel_processed_root(tmp_path)
    artifact_root.mkdir(parents=True)
    (artifact_root / "_manifest").mkdir()
    inspected: list[Path] = []

    def fake_inspect(path: Path) -> object:
        inspected.append(path)
        if path == artifact_root:
            return object()
        raise ValueError("not_dataset")

    monkeypatch.setattr(aggregates, "inspect_dataset", fake_inspect)

    assert aggregates._is_dataset_artifact_root(artifact_root) is True
    assert aggregates._is_dataset_artifact_root(tmp_path) is False
    assert aggregates._official_dataset_artifact_roots(tmp_path) == (artifact_root,)
    assert artifact_root in inspected


def test_aggregates_helpers_validate_paths_and_periods(tmp_path: Path) -> None:
    assert (
        aggregates._local_interim_root(tmp_path)
        .as_posix()
        .endswith("data/interim/csmar")
    )
    assert (
        aggregates._time_window_from_years((2023, 2025)).start.isoformat()
        == "2023-01-01"
    )
    assert aggregates._time_window_from_years(None) is None
    assert aggregates._parse_period("not_applicable") == (None, None)
    assert aggregates._format_period(None, pd.Timestamp("2024-01-02")) == (
        "not_applicable"
    )

    with pytest.raises(FileNotFoundError, match="local_roots_missing"):
        aggregates._official_dataset_artifact_roots(tmp_path)
    with pytest.raises(ValueError, match="invalid_dataset_logical_root"):
        aggregates._dataset_kind_and_artifact_id("bad")
    with pytest.raises(ValueError, match="invalid_dataset_logical_root"):
        aggregates._dataset_kind_and_artifact_id("raw/a_share/demo")


def test_aggregates_filtering_and_calendar_helpers() -> None:
    calendar = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "is_trading_day": [True, False],
        }
    )
    assert aggregates._trading_sessions_from_calendar(calendar) == (
        pd.Timestamp("2024-01-02"),
    )
    with pytest.raises(ValueError, match="trading_calendar_columns_missing"):
        aggregates._trading_sessions_from_calendar(pd.DataFrame())

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", pd.NaT]),
            "value": [1, 2, 3],
        }
    )
    filtered = aggregates._filter_output_sessions(
        frame,
        (pd.Timestamp("2024-01-03"),),
    )
    assert filtered["value"].tolist() == [2]


def test_local_verification_summary_uses_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    evidence = {"handoff_ready": True}
    monkeypatch.setattr(
        aggregates,
        "_dataset_inventory",
        lambda *args, **kwargs: [
            {
                "logical_root": frequency_ids.market_panel_logical_root(),
                "covered_by_evidence_area": "panel",
            },
            {
                "logical_root": "processed/a_share/other/demo",
                "covered_by_evidence_area": "",
            },
        ],
    )

    summary = aggregates.build_local_verification_inventory_payload(
        tmp_path,
        evidence_payload=evidence,
    )

    assert summary["handoff_ready"] is False
    assert summary["covered_roots"] == [frequency_ids.market_panel_logical_root()]
    assert summary["uncovered_roots"] == ["processed/a_share/other/demo"]
