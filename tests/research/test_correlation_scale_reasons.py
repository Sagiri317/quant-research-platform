import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.bundle_validation import (
    validate_output_bundle,
)
from quant_research_platform.artifacts.outputs.evaluation import (
    write_evaluation_outputs_from_chunks,
)
from quant_research_platform.artifacts.outputs.evaluation_validation import (
    validate_evaluation_artifact_structure,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)


def test_native_evaluation_persists_independent_std_reason(tmp_path):
    tiny = np.nextafter(0.0, 1.0)
    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"] * 2,
            "timestamp": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3),
            "x": [-1.0, 0.0, 1.0] * 2,
            "y": [0.0, 1.0, 0.0, 0.0, 1.0, tiny],
            "eligible": True,
        }
    )
    spec = EvaluationSpec(
        evaluation_id="probe",
        market_id="a_share",
        research_dataset_id="probe",
        signals=(SignalSpec(signal_id="x", source_kind="factor", source_column="x"),),
        label_columns=("y",),
        n_groups=2,
    )
    with build_evaluation_result_from_frames(
        (frame,),
        spec=spec,
        asset_column="asset_id",
        source_columns_by_kind={"factor": ("x",)},
    ) as result:
        root = write_evaluation_outputs_from_chunks(
            result.output_chunks,
            summary_factory=result.summaries,
            spec=spec,
            output_path=tmp_path / "evaluation",
        )
    validate_output_bundle(root)
    summary = pd.read_parquet(root / "ic_summary.parquet")
    row = summary.iloc[0]
    assert pd.isna(row.std_ic)
    assert row.std_ic_reason == "nonfinite_result"
    assert row.icir == 1.0
    assert pd.isna(row.icir_reason)
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )

    _, tables = read_evaluation_outputs(root, tables=("ic_summary",))
    assert tables["ic_summary"].std_ic_reason.tolist() == ["nonfinite_result"]
    for value, reason in [
        (np.nan, None),
        (0.0, "nonfinite_result"),
        (np.nan, "invalid"),
    ]:
        broken = summary.copy()
        broken.loc[0, ["std_ic", "std_ic_reason"]] = [value, reason]
        broken.to_parquet(root / "ic_summary.parquet", index=False)
        with pytest.raises(ValueError):
            validate_evaluation_artifact_structure(
                metrics=pd.read_parquet(root / "metrics.parquet"),
                grouped_label_values=pd.read_parquet(
                    root / "grouped_label_values.parquet"
                ),
                ic_summary=pd.read_parquet(root / "ic_summary.parquet"),
                group_spread_summary=pd.read_parquet(
                    root / "group_spread_summary.parquet"
                ),
                spec=spec,
            )


@pytest.mark.parametrize(
    "count,mean,m2,expected_std,expected_reason,expected_ratio_reason",
    [
        (0, 0, 0, np.nan, "no_valid_observations", "no_valid_observations"),
        (1, 1, 0, 0.0, None, "insufficient_observations"),
        (2, 1, 0, 0.0, None, "zero_variance"),
        (2, 0, 2, 1.0, None, None),
    ],
)
def test_aggregate_scale_reason_preserves_degenerate_and_healthy_cases(
    count, mean, m2, expected_std, expected_reason, expected_ratio_reason
):
    from fractions import Fraction

    from quant_research_platform.compute.dataframe.cross_section import (
        _summarize_correlation_aggregates,
    )

    result = _summarize_correlation_aggregates(
        count=count, mean=Fraction(mean), m2=Fraction(m2), positive_count=0
    )
    assert result["std"] == expected_std or (
        pd.isna(result["std"]) and pd.isna(expected_std)
    )
    assert result["std_reason"] == expected_reason
    assert result["ratio_reason"] == expected_ratio_reason
