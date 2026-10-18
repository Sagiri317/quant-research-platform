"""Operation windows constrain native discovery before evaluation starts."""

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.statistics_production import factor_statistics
from quant_research_platform.storage.datasets import inventory
from tests.research.crypto_evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


def _inputs(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02"] * 3 + ["2025-01-02"] * 3),
            "asset_id": list("abc") * 2,
            "factor": [1.0, 2.0, 3.0] * 2,
            "label": [0.01, 0.02, 0.03] * 2,
            "label_available_at": pd.Timestamp("2025-01-03"),
            "eligible": True,
        }
    )
    root = write_evaluation_input(
        tmp_path / "input", frame, factors=("factor",), labels=("label",)
    )
    peer = write_evaluation_input(
        tmp_path / "peer", frame, factors=("factor",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="window",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="factor", source_kind="factor", source_column="factor"
            ),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        n_groups=2,
    )
    config = evaluation_config(
        root,
        tmp_path / "evaluation",
        spec,
        start="2025-01-02",
        end="2025-01-02",
        cutoff="2025-01-03",
    )
    return root, peer, config


@pytest.mark.parametrize("scope", ["ordinary", "own", "common"])
def test_unselected_year_is_not_discovered_and_eval_feeds_statistics(
    tmp_path, monkeypatch, scope
):
    root, peer, config = _inputs(tmp_path)
    if scope == "ordinary":
        config["evaluation"]["filter_mode"] = "unfiltered"
    elif scope == "common":
        config.update(
            sample_scope="common",
            comparison_inputs=[{"inputs": str(peer), "signal_columns": ["factor"]}],
        )
    for source in (root, peer):
        (source / "data/year=2024/unselected.parquet").write_bytes(b"unselected defect")
    discovered = []
    original = inventory._directory_entries

    def scan(path):
        discovered.append(Path(path))
        assert Path(path).name != "year=2024"
        yield from original(path)

    monkeypatch.setattr(inventory, "_directory_entries", scan)
    output = evaluate(config)
    names = ("metrics", "grouped_label_values", "ic_summary", "group_spread_summary")
    if scope != "ordinary":
        names += ("daily_coverage", "daily_strata")
    manifest, tables = read_evaluation_outputs(output, tables=names)
    assert tables["metrics"].n_obs.tolist() == [3]
    assert tables["metrics"].timestamp.tolist() == [pd.Timestamp("2025-01-02")]
    assert tables["metrics"].ic.iloc[0] == pytest.approx(1.0)
    assert manifest["metadata"]["evaluation_diagnostics"]["rows_input"] == 3
    assert root / "data/year=2025" in discovered
    if scope == "common":
        assert peer / "data/year=2025" in discovered
    if scope == "ordinary":
        return
    assert tables["daily_coverage"].domain_rows.tolist() == [3]

    def forbidden(*args, **kwargs):
        raise AssertionError("statistics must consume the persisted evaluation")

    from quant_research_platform.research import statistics_production

    monkeypatch.setattr(statistics_production, "open_dataset_session", forbidden)
    monkeypatch.setattr(statistics_production, "read_dataset", forbidden)
    statistic = read_statistics_artifact(
        factor_statistics(
            {
                "evaluation_inputs": [str(output)],
                "output": str(tmp_path / "statistics"),
                "window": config["window"],
                "analyses": ["stratified_ic"],
                "statistics": {"hac_lag": 1, "fdr_method": "BY", "fdr_alpha": 0.05},
            }
        )
    )
    with statistic.table("factor_stratified_diagnostics.parquet") as frame:
        assert frame.observation_pairs.eq(3).all()
        assert frame.observation_count.eq(1).all()


@pytest.mark.parametrize("peer_input", [False, True])
@pytest.mark.parametrize(
    "selector", ["window", "disjoint", "disjoint_same_year", "years", "files", "rows"]
)
def test_input_selectors_intersect_the_operation_window(tmp_path, selector, peer_input):
    root, peer, config = _inputs(tmp_path)
    source = peer if peer_input else root
    selected = {"path": str(source)}
    if selector == "window":
        selected["time_window"] = {
            "start": "2025-01-02T00:00:00.000000000",
            "end": "2025-01-02T00:00:00.000000000",
        }
    elif selector in {"disjoint", "disjoint_same_year"}:
        date = "2024-01-02" if selector == "disjoint" else "2025-01-03"
        selected["time_window"] = {"start": date, "end": date}
        (source / "data/year=2024/unselected.parquet").write_bytes(b"unselected defect")
        (source / "data/year=2025/unselected.parquet").write_bytes(b"unselected defect")
    elif selector == "years":
        selected["years"] = [2024]
    elif selector == "files":
        selected["files"] = [
            str(path.relative_to(source))
            for path in (source / "data/year=2024").glob("*.parquet")
        ]
    else:
        selected["row_filters"] = [["asset_id", "in", ["a", "c"]]]
    if peer_input:
        config.update(
            sample_scope="common",
            comparison_inputs=[{"inputs": selected, "signal_columns": ["factor"]}],
        )
    else:
        config["inputs"] = selected
    _, tables = read_evaluation_outputs(
        evaluate(config), tables=("metrics", "daily_coverage")
    )
    expected = 3 if selector == "window" else 2 if selector == "rows" else 0
    assert int(tables["metrics"].n_obs.sum()) == expected
    assert int(tables["daily_coverage"].domain_rows.sum()) == expected


@pytest.mark.parametrize("peer_input", [False, True])
def test_selected_corrupt_member_still_blocks_atomic_publication(tmp_path, peer_input):
    root, peer, config = _inputs(tmp_path)
    source = peer if peer_input else root
    next((source / "data/year=2025").glob("*.parquet")).write_bytes(b"selected defect")
    if peer_input:
        config.update(
            sample_scope="common",
            comparison_inputs=[{"inputs": str(peer), "signal_columns": ["factor"]}],
        )
    with pytest.raises(ValueError, match="dataset_parquet_footer_invalid"):
        evaluate(config)
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("operation_date", [False, True])
def test_source_operation_intersection_preserves_calendar_end_nanosecond(
    tmp_path, operation_date
):
    _, _, config = _inputs(tmp_path)
    stamps = pd.to_datetime(
        ["2025-01-02T23:59:59.999999999", "2025-01-03T00:00:00.000000000"]
    )
    frame = pd.DataFrame(
        {
            "timestamp": stamps.repeat(3),
            "asset_id": list("abc") * 2,
            "factor": [1.0, 2.0, 3.0] * 2,
            "label": [0.01, 0.02, 0.03] * 2,
            "label_available_at": stamps.repeat(3),
            "eligible": True,
        }
    )
    root = write_evaluation_input(
        tmp_path / "intraday", frame, factors=("factor",), labels=("label",)
    )
    date_window = {"start": "2025-01-02", "end": "2025-01-02"}
    exact_window = {"start": stamps[0].isoformat(), "end": stamps[1].isoformat()}
    config["window"] = date_window if operation_date else exact_window
    config["inputs"] = {
        "path": str(root),
        "time_window": exact_window if operation_date else date_window,
    }
    _, tables = read_evaluation_outputs(evaluate(config))
    assert tables["metrics"].timestamp.tolist() == [stamps[0]]
    assert tables["metrics"].n_obs.tolist() == [3]
