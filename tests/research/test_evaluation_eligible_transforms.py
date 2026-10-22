"""Native regressions for eligible-domain EVAL transformations."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.runner import evaluate
from tests.research.crypto_evaluation_test_support import (
    evaluation_config,
    write_evaluation_input,
)


@pytest.mark.parametrize("transform", ("identity", "rank"))
def test_native_evaluation_transforms_only_selected_cross_section(tmp_path, transform):
    frame = pd.DataFrame(
        {
            "asset_id": list("abcdef"),
            "timestamp": pd.Timestamp("2023-01-02"),
            "x": [1.0, 2.0, 8.0, 1.5, 1.6, 1.7],
            "label": [1.0, 4.0, 2.0, 10.0, 10.0, 10.0],
            "available_at": pd.Timestamp("2023-01-02"),
            "eligible": [True] * 3 + [False] * 3,
        }
    )
    raw_ic = np.corrcoef([1.0, 2.0, 8.0], [1.0, 4.0, 2.0])[0, 1]
    rank_ic = np.corrcoef([1.0, 2.0, 3.0], [1.0, 4.0, 2.0])[0, 1]
    assert abs(rank_ic - raw_ic) > 0.1
    expected_ic = rank_ic if transform == "rank" else raw_ic
    for name, selected in (
        ("eligible_only", frame.iloc[:3]),
        ("with_ineligible", frame),
    ):
        inputs = write_evaluation_input(
            tmp_path / name / "inputs", selected, factors=("x",), labels=("label",)
        )
        spec = EvaluationSpec(
            evaluation_id=name,
            market_id="a_share",
            research_dataset_id="fixture",
            signals=(
                SignalSpec(
                    signal_id="score",
                    source_kind="factor",
                    source_column="x",
                    transform=transform,
                ),
            ),
            label_columns=("label",),
            label_availability_columns={"label": "available_at"},
            filter_mode="eligible",
            n_groups=3,
        )
        output = evaluate(
            evaluation_config(
                inputs,
                tmp_path / name / "evaluation",
                spec,
                start="2023-01-02",
                end="2023-01-03",
            )
        )
        manifest, tables = read_evaluation_outputs(output)
        assert tables["metrics"].ic.tolist() == pytest.approx(
            [expected_ic], rel=1e-12, abs=1e-15
        )
        assert tables["metrics"].rankic.tolist() == pytest.approx([0.5])
        assert tables["metrics"].n_obs.tolist() == [3]
        assert tables["metrics"].ic_reason.isna().all()
        diagnostics = manifest["metadata"]["evaluation_diagnostics"]
        assert diagnostics["rows_input"] == len(selected)
        assert diagnostics["rows_after_filter_mode"] == 3
        assert diagnostics["rows_after_signal_label_dropna"] == 3
        assert diagnostics["filter_columns_used"] == ["eligible"]
        assert diagnostics["filter_definition"] == "eligible == True"


def test_native_zscore_is_undefined_on_singleton_eligible_cross_section(tmp_path):
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b"],
            "timestamp": pd.Timestamp("2023-01-02"),
            "x": [1.0, 2.0],
            "label": [1.0, 2.0],
            "available_at": pd.Timestamp("2023-01-02"),
            "eligible": [True, False],
        }
    )
    inputs = write_evaluation_input(
        tmp_path / "inputs", frame, factors=("x",), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="singleton",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="score",
                source_kind="factor",
                source_column="x",
                transform="zscore",
            ),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "available_at"},
        filter_mode="eligible",
        n_groups=3,
    )
    output = evaluate(
        evaluation_config(
            inputs,
            tmp_path / "evaluation",
            spec,
            start="2023-01-02",
            end="2023-01-03",
        )
    )
    _, tables = read_evaluation_outputs(output)
    # A singleton has zero variance; an ineligible peer must not define its score.
    assert tables["metrics"].n_obs.tolist() == [0]
    assert tables["grouped_label_values"].empty


@pytest.mark.parametrize(
    "transform,values,peer,domain,observed,corrupt_signal,corrupt_counts",
    [
        ("zscore", [1.0, np.nan, np.nan], False, 1, 0, "plain", (0,)),
        ("rank", [1.0, np.nan, np.nan], False, 1, 1, "scaled", (0,)),
        ("zscore", [1.0, 2.0, 3.0], True, 1, 1, "scaled", ()),
        ("zscore", [1.0, 2.0, 3.0], False, 3, 3, "scaled", (0, 2)),
    ],
    ids=("undefined-zscore", "defined-rank", "normalize-before-peer", "full-domain"),
)
def test_common_source_domain_and_transformed_observation_counts(
    tmp_path, transform, values, peer, domain, observed, corrupt_signal, corrupt_counts
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.statistics_production import factor_statistics

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-02"),
            "asset_id": list("abc"),
            "x": values,
            "z": [1.0, 2.0, 3.0],
            "label": [1.0, 2.0, 3.0],
            "label_available_at": pd.Timestamp("2023-01-02"),
            "eligible": True,
            "industry_level1_code": ["A", "B", "B"],
        }
    )
    inputs = write_evaluation_input(
        tmp_path / "inputs", frame, factors=("x", "z"), labels=("label",)
    )
    spec = EvaluationSpec(
        evaluation_id="common-support",
        market_id="a_share",
        research_dataset_id="fixture",
        signals=(
            SignalSpec(
                signal_id="scaled",
                source_kind="factor",
                source_column="x",
                transform=transform,
            ),
            SignalSpec(signal_id="plain", source_kind="factor", source_column="z"),
        ),
        label_columns=("label",),
        label_availability_columns={"label": "label_available_at"},
        filter_mode="eligible",
        n_groups=3,
    )
    config = evaluation_config(
        inputs,
        tmp_path / "evaluation",
        spec,
        start="2023-01-02",
        end="2023-01-03",
    )
    config.update(
        sample_scope="common", strata={"primary_label": "label", "kinds": ["industry"]}
    )
    if peer:
        peer_inputs = write_evaluation_input(
            tmp_path / "peer", frame.iloc[:1], factors=("x", "z"), labels=("label",)
        )
        config["comparison_inputs"] = [
            {"inputs": str(peer_inputs), "signal_columns": ["x"]}
        ]
    output = evaluate(config)
    _, tables = read_evaluation_outputs(
        output,
        tables=("metrics", "daily_coverage", "daily_strata", "grouped_label_values"),
    )
    expected = {"scaled": observed, "plain": domain}
    assert tables["daily_coverage"].domain_rows.tolist() == [domain]
    assert tables["metrics"].set_index("signal").n_obs.to_dict() == expected
    for name in ("grouped_label_values", "daily_strata"):
        counts = tables[name].groupby("signal").n_obs.sum()
        assert counts.reindex(expected, fill_value=0).to_dict() == expected
    stats = factor_statistics(
        {
            "evaluation_inputs": [str(output)],
            "output": str(tmp_path / "statistics"),
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    artifact = read_statistics_artifact(stats)
    with artifact.table("factor_stratified_diagnostics.parquet") as saved:
        for statistic in ("ic", "rankic"):
            counts = (
                saved.loc[
                    saved.stratification.eq("all") & saved.statistic.eq(statistic)
                ]
                .set_index("signal")
                .observation_pairs
            )
            assert counts.to_dict() == expected

    path = output / "metrics.parquet"
    original = pq.read_table(path)
    for count in corrupt_counts:
        corrupted = original.to_pandas()
        selected = corrupted.signal.eq(corrupt_signal)
        corrupted.loc[selected, "n_obs"] = count
        if count < 2:
            corrupted.loc[selected, ["ic", "rankic"]] = np.nan
            corrupted.loc[selected, ["ic_reason", "rankic_reason"]] = (
                "insufficient_observations"
            )
        pq.write_table(
            pa.Table.from_pandas(
                corrupted, schema=original.schema, preserve_index=False
            ),
            path,
        )
        with pytest.raises(ValueError, match="evaluation_common_metric_count_mismatch"):
            read_evaluation_outputs(output, tables=("metrics", "daily_coverage"))
