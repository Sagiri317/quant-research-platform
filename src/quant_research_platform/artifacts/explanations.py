"""Numerical tables for an independently computed frozen-model explanation."""

from __future__ import annotations

from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts.models import _model_algorithm_facts
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import open_filesystem_file
from quant_research_platform.storage.manifests import read_qrp_json_file
from quant_research_platform.storage.parquet import (
    _opened_parquet,
    parquet_table_writer,
)

LINEAR_EXPLANATION_ALGORITHMS = frozenset(
    {"linear_regression", "ridge_regression", "lasso_regression", "huber_regression"}
)

ATTRIBUTION_METHODS = frozenset(
    {
        "affine",
        "linear_shap",
        "rank",
        "native_tree",
        "tree_shap_path",
        "tree_shap_interventional",
    }
)

_S = pa.string()
_F = pa.float64()
_I = pa.int64()
_T = pa.timestamp("ns")
_SCHEMAS = {
    "native_importance": pa.schema(
        [
            ("feature", _S),
            ("value", _F),
            ("signed_value", _F),
            ("kind", _S),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "predictions": pa.schema(
        [
            ("timestamp", _T),
            ("asset_id", _S),
            ("base_value", _F),
            ("explained_output", _F),
            ("prediction_score", _F),
            ("additivity_residual", _F),
            ("additivity_tolerance", _F),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "contributions": pa.schema(
        [
            ("timestamp", _T),
            ("asset_id", _S),
            ("feature", _S),
            ("input_value", _F),
            ("contribution", _F),
        ]
    ),
    "summary": pa.schema(
        [
            ("feature", _S),
            ("mean_signed_contribution", _F),
            ("mean_abs_contribution", _F),
            ("observation_count", _I),
            ("date_count", _I),
            ("sample_count", _I),
            ("sample_date_count", _I),
            ("absolute_share", _F),
            ("rank", _I),
            ("period", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "permutation_repeats": pa.schema(
        [
            ("group", _S),
            ("repeat", _I),
            ("metric", _S),
            ("baseline", _F),
            ("perturbed", _F),
            ("delta", _F),
            ("observation_count", _I),
            ("period_count", _I),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
        ]
    ),
    "permutation_summary": pa.schema(
        [
            ("group", _S),
            ("metric", _S),
            ("baseline", _F),
            ("observation_count", _I),
            ("period_count", _I),
            ("unit", _S),
            ("status", _S),
            ("reason", _S),
            ("delta_mean", _F),
            ("delta_std", _F),
            ("repeats", _I),
        ]
    ),
}
_KEYS = {
    "native_importance": ["feature"],
    "predictions": ["timestamp", "asset_id"],
    "contributions": ["timestamp", "asset_id", "feature"],
    "summary": ["period", "feature"],
    "permutation_repeats": ["group", "repeat", "metric"],
    "permutation_summary": ["group", "metric"],
}


def validate_attribution_method(method, algorithm):
    """Bind the requested attribution to the existing model capability facts."""
    facts = _model_algorithm_facts(algorithm)
    linear = algorithm in LINEAR_EXPLANATION_ALGORITHMS
    rank = facts.input_kind == "rank_ensemble_inputs"
    if method in {"affine", "linear_shap"} and not linear:
        raise ValueError("explanation_linear_model_required")
    if method == "rank" and not rank:
        raise ValueError("explanation_rank_model_required")
    if method == "native_tree" and facts.dependency_package not in {
        "xgboost",
        "lightgbm",
    }:
        raise ValueError("explanation_native_tree_not_supported")
    if method in {"tree_shap_path", "tree_shap_interventional"} and (linear or rank):
        raise ValueError("explanation_tree_model_required")


def attribution_relative_tolerance(method, feature_count):
    """Tree backends accumulate float32 paths; affine/rank paths use float64."""
    return float(
        5e-5
        if method in {"native_tree", "tree_shap_path", "tree_shap_interventional"}
        else 64 * np.finfo(float).eps * max(1, feature_count)
    )


def attribution_additivity(base, contributions, output, relative):
    """Require both backward stability and useful precision at the output scale."""
    terms = [float(base), *map(float, contributions)]
    if not all(np.isfinite(terms)):
        return np.nan, np.nan, "nonrepresentable_contribution_or_baseline"
    magnitude = max(abs(output), *map(abs, terms))
    floor = np.nextafter(0.0, 1.0) * max(1, len(contributions))
    backward = (
        (relative * magnitude) * max(1.0, fsum(abs(term) / magnitude for term in terms))
        if magnitude
        else 0.0
    )
    backward += floor
    tolerance = min(backward, relative * abs(output) + floor)
    try:
        residual = fsum([*terms, -float(output)])
    except OverflowError:
        return np.nan, np.nan, "nonrepresentable_additivity"
    if not np.isfinite(residual) or not np.isfinite(tolerance):
        return np.nan, np.nan, "nonrepresentable_additivity"
    if abs(residual) > tolerance:
        reason = (
            "cancellation_precision_loss"
            if abs(residual) <= backward
            else "additivity_mismatch"
        )
        return np.nan, np.nan, reason
    return residual, tolerance, ""


def _validate(metadata, tables, *, partial=False):
    if not isinstance(metadata, dict) or metadata.get("kind") != "MODEL_EXPLANATION":
        raise ValueError("explanation_metadata_invalid")
    for key in ("model", "model_id", "model_algorithm", "target_column", "label_unit"):
        if type(metadata.get(key)) is not str or not metadata[key].strip():
            raise ValueError("explanation_metadata_invalid")
    _model_algorithm_facts(metadata["model_algorithm"])
    columns = metadata.get("input_columns")
    methods = metadata.get("methods")
    if (
        type(columns) is not list
        or not columns
        or any(type(value) is not str or not value for value in columns)
        or len(set(columns)) != len(columns)
        or not isinstance(metadata.get("input_units"), dict)
        or type(methods) is not list
        or not methods
        or any(type(value) is not str for value in methods)
        or len(set(methods)) != len(methods)
        or set(methods) - ATTRIBUTION_METHODS - {"native_importance", "permutation"}
    ):
        raise ValueError("explanation_metadata_invalid")
    attribution = set(methods) & ATTRIBUTION_METHODS
    if len(attribution) > 1:
        raise ValueError("explanation_single_attribution_method_required")
    if attribution or "permutation" in methods:
        space = metadata.get("output_space")
        rank = metadata["model_algorithm"] in {
            "equal_weight_rank_ensemble",
            "ic_weighted_rank_ensemble",
        }
        unit = (
            "raw_margin"
            if space == "raw_margin"
            else "dimensionless_rank_score"
            if rank and space == "model_output"
            else metadata["label_unit"]
        )
        if (
            space not in {"model_output", "raw_margin", "calibrated_label"}
            or metadata.get("unit") != unit
            or type(metadata.get("sample_role")) is not str
            or not metadata["sample_role"].strip()
            or space == "calibrated_label"
            and not rank
            or space == "raw_margin"
            and metadata["model_algorithm"]
            not in {"xgboost_regressor", "lightgbm_regressor"}
        ):
            raise ValueError("explanation_output_metadata_invalid")
    if attribution:
        facts = metadata.get("attribution", {})
        method = next(iter(attribution))
        validate_attribution_method(method, metadata["model_algorithm"])
        baseline, background = {
            "affine": ("training_scaled_zero", "none"),
            "rank": ("zero_centered_cross_section_rank", "none"),
            "linear_shap": ("historical_reference_expectation", "historical_reference"),
            "tree_shap_interventional": (
                "historical_reference_expectation",
                "historical_reference",
            ),
            "native_tree": (
                "model_internal_tree_expectation",
                "model_internal_tree_statistics",
            ),
            "tree_shap_path": (
                "model_internal_tree_expectation",
                "model_internal_tree_statistics",
            ),
        }[method]
        tolerance = facts.get("additivity_relative_tolerance")
        if (
            facts.get("method") != method
            or facts.get("baseline")
            not in (
                (baseline, "unscaled_input_zero") if method == "affine" else (baseline,)
            )
            or type(facts.get("implementation")) is not str
            or not facts["implementation"]
            or not isinstance(facts.get("background"), dict)
            or facts["background"].get("kind") != background
            or type(tolerance) not in (int, float)
            or tolerance != attribution_relative_tolerance(method, len(columns))
        ):
            raise ValueError("explanation_attribution_metadata_invalid")
    expected = set()
    if "native_importance" in methods:
        expected.add("native_importance")
    if "permutation" in methods:
        expected.update(("permutation_repeats", "permutation_summary"))
    if set(methods) - {"native_importance", "permutation"}:
        expected.update(("predictions", "contributions", "summary"))
    if (
        not expected
        or not tables
        or (set(tables) - expected if partial else set(tables) != expected)
    ):
        raise ValueError("explanation_members_invalid")
    result = {}
    for name, frame in tables.items():
        schema = _SCHEMAS[name]
        if list(frame.columns) != schema.names or frame.empty:
            raise ValueError("explanation_table_schema_invalid")
        try:
            table = pa.Table.from_pandas(
                frame, schema=schema, preserve_index=False, safe=True
            )
        except (pa.ArrowException, ValueError, TypeError):
            raise ValueError("explanation_table_schema_invalid") from None
        converted = table.to_pandas()
        if (
            converted[_KEYS[name]].isna().any(axis=None)
            or converted.duplicated(_KEYS[name]).any()
        ):
            raise ValueError("explanation_table_keys_invalid")
        for field in schema:
            if pa.types.is_string(field.type) and (
                converted[field.name].isna().any()
                or field.name != "reason"
                and converted[field.name].str.strip().eq("").any()
            ):
                raise ValueError("explanation_table_string_invalid")
            if pa.types.is_floating(field.type):
                if not np.isfinite(converted[field.name].dropna()).all():
                    raise ValueError("explanation_table_nonfinite")
            if pa.types.is_integer(field.type) and (
                (field.name != "rank" and converted[field.name].isna().any())
                or converted[field.name].lt(0).any()
            ):
                raise ValueError("explanation_table_count_invalid")
        if (
            "feature" in converted
            and not converted.feature.isin(metadata["input_columns"]).all()
        ):
            raise ValueError("explanation_feature_binding_invalid")
        if "status" in converted:
            defined = converted.status.eq("DEFINED")
            measurements = {
                "native_importance": ["value"],
                "predictions": [
                    "base_value",
                    "additivity_residual",
                    "additivity_tolerance",
                ],
                "summary": [
                    "mean_abs_contribution",
                    "mean_signed_contribution",
                    "rank",
                ],
                "permutation_repeats": ["baseline", "perturbed", "delta"],
                "permutation_summary": ["baseline", "delta_mean", "delta_std"],
            }[name]
            if (
                not converted.status.isin(
                    ("DEFINED", "NOT_APPLICABLE")
                    if name == "native_importance"
                    else ("DEFINED", "UNDEFINED")
                ).all()
                or converted.loc[defined, measurements].isna().any(axis=None)
                or converted.loc[~defined, measurements].notna().any(axis=None)
                or converted.loc[defined, "reason"].ne("").any()
                or converted.loc[~defined, "reason"].str.strip().eq("").any()
            ):
                raise ValueError("explanation_table_status_invalid")
        if name in {"predictions", "contributions", "summary"}:
            nullable = {
                "predictions": {
                    "base_value",
                    "additivity_residual",
                    "additivity_tolerance",
                },
                "contributions": {"input_value", "contribution"},
                "summary": {
                    "absolute_share",
                    "mean_abs_contribution",
                    "mean_signed_contribution",
                    "rank",
                },
            }.get(name, set())
            if converted.drop(columns=list(nullable)).isna().any(axis=None):
                raise ValueError("explanation_table_null_invalid")
        result[name] = converted
    if "predictions" in result:
        predictions = result["predictions"].set_index(["timestamp", "asset_id"])
        contributions = result["contributions"]
        counts = contributions.groupby(["timestamp", "asset_id"]).size()
        if (
            not counts.index.equals(predictions.sort_index().index)
            or not counts.eq(len(metadata["input_columns"])).all()
        ):
            raise ValueError("explanation_row_binding_invalid")
        predictions = predictions.sort_index()
        matrix = contributions.pivot(
            index=["timestamp", "asset_id"], columns="feature", values="contribution"
        ).reindex(columns=metadata["input_columns"])
        defined = predictions.status.eq("DEFINED")
        if matrix.loc[defined].isna().any(axis=None) or matrix.loc[
            ~defined
        ].notna().any(axis=None):
            raise ValueError("explanation_row_status_invalid")
        for key, row in predictions.loc[defined].iterrows():
            residual, tolerance, reason = attribution_additivity(
                row.base_value,
                matrix.loc[key],
                row.explained_output,
                metadata["attribution"]["additivity_relative_tolerance"],
            )
            if (
                reason
                or residual != row.additivity_residual
                or tolerance != row.additivity_tolerance
            ):
                raise ValueError("explanation_additivity_failed")
    if "summary" in result:
        summary = result["summary"]
        if (
            summary.observation_count.gt(summary.sample_count).any()
            or summary.date_count.gt(summary.sample_date_count).any()
            or summary.date_count.gt(summary.observation_count).any()
            or summary.status.eq("DEFINED").ne(summary.observation_count.gt(0)).any()
            or summary.mean_abs_contribution.lt(0).any()
            or not summary.absolute_share.dropna().between(0, 1).all()
        ):
            raise ValueError("explanation_summary_coverage_invalid")
        if "predictions" in result and (
            not summary.loc[summary.period.eq("all"), "observation_count"]
            .eq(int(defined.sum()))
            .all()
            or not summary.loc[summary.period.eq("all"), "sample_count"]
            .eq(len(predictions))
            .all()
        ):
            raise ValueError("explanation_summary_coverage_invalid")
    return result


def write_explanation(path: Path, *, metadata: dict, tables: dict) -> Path:
    """Publish only selected tables, with fail-if-exists atomic storage."""
    tables = _validate(metadata, tables)
    path = Path(path)

    def write(staging):
        for name, frame in tables.items():
            with open_filesystem_file(
                staging / (name + ".parquet"), mode="xb"
            ) as handle:
                with parquet_table_writer(
                    handle, compression="zstd", schema=_SCHEMAS[name]
                ) as writer:
                    writer(
                        pa.Table.from_pandas(
                            frame, schema=_SCHEMAS[name], preserve_index=False
                        )
                    )
        atomic_write_json(
            staging / "metadata.json", {**metadata, "members": list(tables)}
        )

    atomic_write_directory(path, write)
    return path


def read_explanation(
    path: Path, *, members: list[str] | tuple[str, ...] | None = None
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Read selected members; row attributions also require their prediction pair.

    Omit members to validate the complete artifact. Summaries and native importance
    are independently consumable; no predictor is loaded or explanation rerun.
    """
    path = Path(path)
    metadata = read_qrp_json_file(path / "metadata.json")[1]
    declared = metadata.get("members")
    if (
        type(declared) is not list
        or any(type(name) is not str for name in declared)
        or len(declared) != len(set(declared))
        or set(declared) - set(_SCHEMAS)
    ):
        raise ValueError("explanation_members_invalid")
    selected = declared if members is None else members
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or any(type(name) is not str for name in selected)
        or len(selected) != len(set(selected))
        or set(selected) - set(declared)
    ):
        raise ValueError("explanation_members_invalid")
    needed = set(selected)
    if needed & {"predictions", "contributions"}:
        needed.update(("predictions", "contributions"))
    if needed - set(declared):
        raise ValueError("explanation_members_invalid")
    tables = {}
    for name in declared:
        if name not in needed:
            continue
        with _opened_parquet(path / (name + ".parquet")) as (parquet, _handle):
            if parquet.schema_arrow.remove_metadata() != _SCHEMAS[name]:
                raise ValueError("explanation_table_schema_invalid")
            tables[name] = parquet.read().to_pandas()
    return metadata, _validate(metadata, tables, partial=members is not None)
