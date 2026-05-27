from __future__ import annotations

import pytest

from quant_research_platform.research.factor_factory.factor_registry import (
    FactorTemplateRegistry,
)


def test_factor_template_registry_loads_controlled_a_share_templates() -> None:
    registry = FactorTemplateRegistry.from_directory("config/a_share/factor_templates")

    candidates = registry.candidates(
        sample_start="2024-01-01",
        sample_end="2024-12-31",
        label_columns=("forward_return_1d",),
    )

    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    assert set(by_id) == {
        "liquidity_20d__window_20@1",
        "liquidity_20d__window_60@1",
        "momentum_60d__window_20@1",
        "momentum_60d__window_60@1",
        "reversal_20d__window_20@1",
        "reversal_20d__window_60@1",
        "reversal_20d__window_120@1",
    }
    reversal = by_id["reversal_20d__window_20@1"]
    assert reversal.dependencies == ("close",)
    assert reversal.output_column == "rev_20d"
    assert reversal.parameters == {"window": 20}
    assert reversal.time_semantics["decision_time_column"] == "trade_date"
    assert by_id["momentum_60d__window_60@1"].output_column == "mom_60d"
    assert by_id["reversal_20d__window_120@1"].output_column == "rev_120d"
    assert by_id["liquidity_20d__window_20@1"].output_column == "avg_amount_20d"
    assert by_id["liquidity_20d__window_20@1"].inputs == ("amount",)
    assert reversal.sample_start == "2024-01-01"
    assert reversal.sample_end == "2024-12-31"
    assert reversal.label_columns == ("forward_return_1d",)
    assert "existing_factors" in reversal.capabilities
    assert "evaluate" in reversal.capabilities
    assert "parameter_grid" in reversal.capabilities
    templates = registry.templates()
    assert [template.template_id for template in templates] == [
        "liquidity_20d",
        "momentum_60d",
        "reversal_20d",
    ]
    assert templates[-1].description is not None


def test_factor_template_registry_rejects_unsafe_expression(tmp_path) -> None:
    (tmp_path / "unsafe.yaml").write_text(
        "\n".join(
            [
                "id: unsafe_factor",
                "market: a_share",
                "expression: __import__('os').system('dir')",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid expression"):
        FactorTemplateRegistry.from_directory(tmp_path)


@pytest.mark.parametrize(
    "field,content",
    [
        (
            "market",
            "\n".join(["id: missing_market", "expression: return(close, 20)"]),
        ),
        (
            "expression",
            "\n".join(["id: missing_expression", "market: a_share"]),
        ),
    ],
)
def test_factor_template_registry_rejects_missing_required_fields(
    tmp_path,
    field,
    content,
) -> None:
    (tmp_path / "missing.yaml").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        FactorTemplateRegistry.from_directory(tmp_path)


def test_factor_template_registry_rejects_future_label_field(tmp_path) -> None:
    (tmp_path / "label_leak.yaml").write_text(
        "\n".join(
            [
                "id: label_leak",
                "market: a_share",
                "expression: return(forward_return_1d, 20)",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported field"):
        FactorTemplateRegistry.from_directory(tmp_path)


def test_factor_template_registry_rejects_bad_window(tmp_path) -> None:
    (tmp_path / "bad_window.yaml").write_text(
        "\n".join(
            [
                "id: bad_window",
                "market: a_share",
                "expression: return(close, 0)",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive integer"):
        FactorTemplateRegistry.from_directory(tmp_path)


def test_factor_template_registry_rejects_incomplete_parameter_grid(tmp_path) -> None:
    (tmp_path / "bad_grid.yaml").write_text(
        "\n".join(
            [
                "id: bad_grid",
                "market: a_share",
                "expression: return(close, 20)",
                "grid_parameter: window",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grid_parameter"):
        FactorTemplateRegistry.from_directory(tmp_path)
