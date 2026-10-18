"""Persisted numerical explanation plots without loading models or SHAP."""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_research_platform.artifacts.explanations import (
    attribution_additivity,
    attribution_relative_tolerance,
    read_explanation,
    write_explanation,
)
from quant_research_platform.reports import plots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def explanation_fixture(root, *, native_only=False, undefined=False):
    features = ["价值因子", "规模因子", "动量因子", "波动因子"]
    metadata = {
        "kind": "MODEL_EXPLANATION",
        "model": "frozen-model",
        "model_id": "analytical-rank",
        "model_algorithm": "equal_weight_rank_ensemble",
        "input_columns": features,
        "input_units": dict.fromkeys(features, "dimensionless"),
        "target_column": "future_rank_label",
        "label_unit": "dimensionless_rank_score",
        "methods": ["native_importance"],
    }
    tables = {
        "native_importance": pd.DataFrame(
            {
                "feature": features,
                "value": [0.5, 0.3, 0.2, None],
                "signed_value": [0.5, -0.3, 0.2, None],
                "kind": ["absolute_rank_weight"] * 4,
                "unit": ["dimensionless"] * 4,
                "status": ["DEFINED"] * 3 + ["NOT_APPLICABLE"],
                "reason": [""] * 3 + ["excluded_at_fit"],
            }
        )
    }
    if not native_only:
        metadata.update(
            methods=["native_importance", "rank"],
            sample_role="test",
            output_space="model_output",
            unit="dimensionless_rank_score",
            attribution={
                "method": "rank",
                "baseline": "zero_centered_cross_section_rank",
                "implementation": "qrp",
                "background": {"kind": "none"},
                "additivity_relative_tolerance": attribution_relative_tolerance(
                    "rank", len(features)
                ),
            },
            aggregation="equal_timestamp_means",
        )
        timestamp = pd.Timestamp("2026-01-02T09:30:00.000000123")
        values = [0.4, -0.3, 0.2, -0.1]
        actual = 0.45 - 2e-14
        residual, tolerance, reason = attribution_additivity(
            0.25,
            values,
            actual,
            metadata["attribution"]["additivity_relative_tolerance"],
        )
        assert reason == ""
        predictions = pd.DataFrame(
            {
                "timestamp": [timestamp],
                "asset_id": ["asset-A"],
                "base_value": [0.25],
                "explained_output": [actual],
                "prediction_score": [actual],
                "additivity_residual": [residual],
                "additivity_tolerance": [tolerance],
                "status": ["DEFINED"],
                "reason": [""],
            }
        )
        contributions = pd.DataFrame(
            {
                "timestamp": [timestamp] * 4,
                "asset_id": ["asset-A"] * 4,
                "feature": features,
                "input_value": [2.0, -3.0, 1.0, 4.0],
                "contribution": values,
            }
        )
        summary = pd.DataFrame(
            {
                "feature": features * 2,
                "mean_signed_contribution": values * 2,
                "mean_abs_contribution": [abs(value) for value in values] * 2,
                "observation_count": [1] * 8,
                "date_count": [1] * 8,
                "sample_count": [1] * 8,
                "sample_date_count": [1] * 8,
                "absolute_share": [abs(value) for value in values] * 2,
                "rank": [1, 2, 3, 4] * 2,
                "period": ["all"] * 4 + ["2026-01"] * 4,
                "status": ["DEFINED"] * 8,
                "reason": [""] * 8,
            }
        )
        if undefined:
            predictions[
                ["base_value", "additivity_residual", "additivity_tolerance"]
            ] = np.nan
            predictions["status"] = "UNDEFINED"
            predictions["reason"] = "nonfinite_contributions"
            contributions["contribution"] = np.nan
            summary[
                [
                    "mean_signed_contribution",
                    "mean_abs_contribution",
                    "absolute_share",
                    "rank",
                ]
            ] = np.nan
            summary[["observation_count", "date_count"]] = 0
            summary["status"] = "UNDEFINED"
            summary["reason"] = "nonfinite_contributions"
        tables.update(
            predictions=predictions, contributions=contributions, summary=summary
        )
    return write_explanation(root, metadata=metadata, tables=tables)


def _loaded(root, **kwargs):
    source = explanation_fixture(root, **kwargs)
    metadata, tables = read_explanation(source)
    return source, metadata, tables


def test_summary_preserves_stored_values_units_period_and_top_n(tmp_path):
    source, metadata, tables = _loaded(tmp_path / "explain")
    figure, axes = plots._plot_explanation_summary(
        metadata,
        tables,
        source_reference=str(source),
        sample_role=None,
        period="2026-01",
        top_n=2,
    )
    axis = axes[0]
    np.testing.assert_allclose(
        [patch.get_width() for patch in axis.patches], [0.4, 0.3]
    )
    assert "dimensionless_rank_score" in axis.get_xlabel()
    assert "%" not in axis.get_xlabel() and "收益率" not in axis.get_xlabel()
    assert "测试集" in figure._supxlabel.get_text()
    assert "2026-01" in axis.texts[0].get_text()
    assert "2/4" in axis.texts[0].get_text()
    assert "未重新归一化" in axis.texts[0].get_text()
    assert "观测 1/1" in axis.texts[0].get_text()
    figure.canvas.draw()
    with pytest.raises(ValueError, match="plot_explanation_period_unavailable"):
        plots._plot_explanation_summary(
            metadata,
            tables,
            source_reference=str(source),
            sample_role=None,
            period="2030-01",
        )


def test_waterfall_signed_remainder_residual_and_exact_timestamp(tmp_path):
    source, metadata, tables = _loaded(tmp_path / "explain")
    figure, axes = plots._plot_explanation_waterfall(
        metadata,
        tables,
        timestamp="2026-01-02T17:30:00.000000123+08:00",
        asset_id="asset-A",
        source_reference=str(source),
        sample_role="test",
        top_n=1,
    )
    patches = axes[0].patches
    assert [label.get_text() for label in axes[0].get_yticklabels()] == [
        "基准值",
        "价值因子",
        "其他 3 个特征（有符号合计）",
        "数值残差修正",
        "实际被解释输出",
    ]
    np.testing.assert_allclose(
        [p.get_width() for p in patches],
        [
            0.25,
            0.4,
            -0.2,
            -tables["predictions"].iloc[0].additivity_residual,
            tables["predictions"].iloc[0].explained_output,
        ],
        atol=np.finfo(float).eps,
    )
    assert patches[-2].get_width() < 0
    correction = -tables["predictions"].iloc[0].additivity_residual
    assert f"{correction:+.5g}" in {text.get_text() for text in axes[0].texts}
    assert patches[-2].get_x() + patches[-2].get_width() == pytest.approx(
        patches[-1].get_width(), abs=1e-15
    )
    assert "dimensionless_rank_score" in axes[0].get_xlabel()
    assert "000000123" in axes[0].get_title()
    tolerance = tables["predictions"].iloc[0].additivity_tolerance
    assert f"容差={tolerance:.3g}" in axes[0].texts[-1].get_text()
    assert axes[0].get_xlim()[1] > 0.65
    figure.canvas.draw()


def test_native_importance_is_distinct_and_excluded_features_are_not_zero(tmp_path):
    source, metadata, tables = _loaded(tmp_path / "native", native_only=True)
    figure, axes = plots._plot_explanation_summary(
        metadata,
        tables,
        source_reference=str(source),
        sample_role=None,
        measure="native_importance",
    )
    assert np.isnan(axes[0].patches[-1].get_width())
    assert "原生重要性" in figure._suptitle.get_text()
    assert "不是逐预测贡献或 SHAP" in axes[0].texts[0].get_text()
    assert "excluded_at_fit" in axes[0].texts[-1].get_text()
    assert "测试集" not in figure._supxlabel.get_text()
    figure.canvas.draw()
    with pytest.raises(ValueError, match="plot_explanation_measure_unavailable"):
        plots._plot_explanation_summary(
            metadata, tables, source_reference=str(source), sample_role=None
        )
    with pytest.raises(ValueError, match="plot_explanation_contributions_unavailable"):
        plots._plot_explanation_waterfall(
            metadata,
            tables,
            source_reference=str(source),
            sample_role=None,
            timestamp="2026-01-02",
            asset_id="asset-A",
        )


def test_undefined_explanation_keeps_reason_and_real_prediction(tmp_path):
    source, metadata, tables = _loaded(tmp_path / "undefined", undefined=True)
    figure, axes = plots._plot_explanation_summary(
        metadata, tables, source_reference=str(source), sample_role=None
    )
    assert all(np.isnan(patch.get_width()) for patch in axes[0].patches)
    assert "观测 0/1" in axes[0].texts[0].get_text()
    assert "nonfinite_contributions" in axes[0].texts[0].get_text()
    figure.canvas.draw()
    figure, axes = plots._plot_explanation_waterfall(
        metadata,
        tables,
        source_reference=str(source),
        sample_role=None,
        timestamp="2026-01-02T09:30:00.000000123",
        asset_id="asset-A",
    )
    assert not axes[0].patches
    assert "UNDEFINED: nonfinite_contributions" in axes[0].texts[0].get_text()
    assert "实际被解释输出=0.45" in axes[0].texts[0].get_text()
    figure.canvas.draw()


@pytest.mark.parametrize("suffix", ["png", "svg", "pdf"])
def test_artifact_to_images_atomic_and_no_predictor_or_shap_import(tmp_path, suffix):
    source = explanation_fixture(tmp_path / "explain")
    output = tmp_path / f"summary.{suffix}"
    code = """
import importlib.abc
import sys
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'shap'
            or fullname.startswith('quant_research_platform.research.models')):
            raise AssertionError('model_or_shap_imported')
sys.meta_path.insert(0, NoModel())
from quant_research_platform.reports.plots import plot
plot({'kind': 'explanation_summary', 'inputs': sys.argv[1], 'output': sys.argv[2]})
"""
    if suffix == "png":
        result = subprocess.run(
            [sys.executable, "-c", code, str(source), str(output)],
            env={**os.environ, "PYTHONPATH": str(Path(plots.__file__).parents[2])},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        plots.plot(
            {
                "kind": "explanation_summary",
                "inputs": str(source),
                "output": str(output),
            }
        )
    assert output.stat().st_size > 1000
    original = output.read_bytes()
    with pytest.raises(Exception, match="output_already_exists"):
        plots.plot(
            {
                "kind": "explanation_summary",
                "inputs": str(source),
                "output": str(output),
            }
        )
    assert output.read_bytes() == original
    assert plt.get_fignums() == []


def test_explicit_selection_destination_and_role_guards(tmp_path):
    source = explanation_fixture(tmp_path / "explain")
    config = {
        "kind": "explanation_waterfall",
        "inputs": str(source),
        "output": str(tmp_path / "waterfall.png"),
        "timestamp": "2026-01-02T09:30:00.000000123",
        "asset_id": "asset-A",
    }
    assert plots.plot(config).stat().st_size > 1000
    variants = [
        ({"timestamp": "today"}, "plot_timestamp_invalid"),
        ({"timestamp": "2026-01-02"}, "plot_explanation_observation_unavailable"),
        ({"asset_id": "absent"}, "plot_explanation_observation_unavailable"),
        ({"sample_role": "validation"}, "plot_sample_role_mismatch"),
        ({"output": str(source / "new.png")}, "plot_output_inside_input"),
    ]
    for index, (change, error) in enumerate(variants):
        with pytest.raises(ValueError, match=error):
            plots.plot(
                {**config, "output": str(tmp_path / f"bad-{index}.png"), **change}
            )
    assert plt.get_fignums() == []


@pytest.mark.parametrize("top_n", [True, 0, -1, 1.5, "2"])
def test_bad_top_n_rejected_before_read(tmp_path, top_n):
    with pytest.raises(ValueError, match="plot_top_n_invalid"):
        plots.plot(
            {
                "kind": "explanation_summary",
                "inputs": "missing",
                "output": str(tmp_path / "new.png"),
                "top_n": top_n,
            }
        )


def test_native_period_and_summary_override_rejected_before_read(tmp_path):
    config = {
        "kind": "explanation_summary",
        "inputs": "missing",
        "output": str(tmp_path / "new.png"),
    }
    with pytest.raises(ValueError, match="plot_explanation_unused_period"):
        plots.plot({**config, "measure": "native_importance", "period": "all"})
    for overrides in (
        {"units": "return"},
        {"row_filters": []},
        {"timestamp": "2026-01-01"},
    ):
        with pytest.raises(ValueError, match="plot_config_fields_invalid"):
            plots.plot({**config, **overrides})
    with pytest.raises(ValueError, match="plot_explanation_measure_invalid"):
        plots.plot({**config, "measure": "shap"})


def test_native_importance_from_mixed_artifact_is_not_labeled_as_test(tmp_path):
    source, metadata, tables = _loaded(tmp_path / "mixed")
    figure, _axes = plots._plot_explanation_summary(
        metadata,
        tables,
        source_reference=str(source),
        sample_role="test",
        measure="native_importance",
    )
    assert "测试集" not in figure._supxlabel.get_text()


@pytest.mark.parametrize(
    "measure,member",
    [("mean_abs_contribution", "summary"), ("native_importance", "native_importance")],
)
def test_selected_summary_reader_and_plot_ignore_unconsumed_members(
    tmp_path, measure, member
):
    source = explanation_fixture(tmp_path / "explain")
    _, original = read_explanation(source)
    for name in set(original) - {member}:
        (source / f"{name}.parquet").unlink()
    _, selected = read_explanation(source, members=[member])
    assert set(selected) == {member}
    pd.testing.assert_frame_equal(selected[member], original[member])
    config = {
        "kind": "explanation_summary",
        "measure": measure,
        "inputs": str(source),
        "output": str(tmp_path / "summary.png"),
    }
    assert plots.plot(config).stat().st_size > 1000
    with pytest.raises(FileNotFoundError):
        read_explanation(source)
    (source / f"{member}.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        read_explanation(source, members=[member])
    with pytest.raises(FileNotFoundError):
        plots.plot({**config, "output": str(tmp_path / "missing.png")})


def test_waterfall_reads_and_validates_pair_without_summary(tmp_path):
    source = explanation_fixture(tmp_path / "explain")
    (source / "summary.parquet").unlink()
    (source / "native_importance.parquet").unlink()
    _, tables = read_explanation(source, members=["contributions"])
    assert set(tables) == {"predictions", "contributions"}
    config = {
        "kind": "explanation_waterfall",
        "inputs": str(source),
        "output": str(tmp_path / "waterfall.png"),
        "timestamp": "2026-01-02T09:30:00.000000123",
        "asset_id": "asset-A",
    }
    assert plots.plot(config).stat().st_size > 1000
    tables["contributions"].loc[0, "contribution"] += 1.0
    member_path = source / "contributions.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            tables["contributions"], schema=pq.read_schema(member_path)
        ),
        member_path,
    )
    with pytest.raises(ValueError, match="explanation_additivity_failed"):
        read_explanation(source, members=["predictions"])
    with pytest.raises(ValueError, match="explanation_additivity_failed"):
        plots.plot({**config, "output": str(tmp_path / "invalid.png")})
    (source / "predictions.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        read_explanation(source, members=["contributions"])
    with pytest.raises(FileNotFoundError):
        plots.plot({**config, "output": str(tmp_path / "missing.png")})


def test_selected_summary_still_checks_own_coverage_and_publication_requires_all(
    tmp_path,
):
    source = explanation_fixture(tmp_path / "explain")
    metadata, tables = read_explanation(source, members=["summary"])
    with pytest.raises(ValueError, match="explanation_members_invalid"):
        write_explanation(tmp_path / "incomplete", metadata=metadata, tables=tables)
    tables["summary"].loc[0, "observation_count"] = 2
    member_path = source / "summary.parquet"
    pq.write_table(
        pa.Table.from_pandas(tables["summary"], schema=pq.read_schema(member_path)),
        member_path,
    )
    with pytest.raises(ValueError, match="explanation_summary_coverage_invalid"):
        read_explanation(source, members=["summary"])
