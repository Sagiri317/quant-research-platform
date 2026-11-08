"""Training-only feedback for iterative formula proposals through the local shell."""

from quant_research_platform.agents.execution import error_feedback
from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.research.datasets.learning import (
    mature_rows,
    research_frame,
)
from quant_research_platform.research.factors.search import _score, formula_values


def training_feedback(config):
    """Evaluate a finite proposal batch without loading validation or test rows."""
    config = _strict_config(
        config,
        {
            "dataset",
            "columns",
            "baseline_columns",
            "label",
            "train_window",
            "train_cutoff",
            "expressions",
        },
        set(),
    )
    expressions = config["expressions"]
    if (
        not isinstance(expressions, list)
        or not expressions
        or any(not isinstance(expression, str) for expression in expressions)
    ):
        raise ValueError("factor_search_expressions_must_be_string_list")
    columns = tuple(dict.fromkeys((*config["columns"], *config["baseline_columns"])))
    frame, _, availability = research_frame(
        config["dataset"], config["train_window"], columns, label=config["label"]
    )
    train = mature_rows(
        frame,
        label=config["label"],
        availability=availability,
        cutoff=config["train_cutoff"],
    )
    if train.empty:
        raise ValueError("factor_search_mature_sample_empty")
    count = len(train)
    candidates = []
    for expression in expressions:
        row = {"expression": expression}
        try:
            values = formula_values(frame, expression, columns)
        except (SyntaxError, ValueError) as error:
            candidates.append({**row, "status": "failed", **error_feedback(error)})
            continue
        raw, increment = _score(
            train, values, config["label"], config["baseline_columns"]
        )
        covered = int(values.reindex(train.index).notna().sum())
        candidates.append(
            {
                **row,
                "status": "defined" if increment is not None else "undefined",
                "train_rank_ic": raw,
                "train_incremental_rank_ic": increment,
                "covered_rows": covered,
                "coverage": covered / count,
            }
        )
    return {
        "sample_role": "training_only",
        "eligible_rows": count,
        "candidates": candidates,
    }
