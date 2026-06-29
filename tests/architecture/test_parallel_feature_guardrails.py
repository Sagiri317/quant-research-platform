from __future__ import annotations

import re
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
PORTFOLIO_GUARDRAIL_ROOTS = (
    PROJECT_SRC / "compute",
    PROJECT_SRC / "research" / "portfolios",
    PROJECT_SRC / "contracts" / "portfolios.py",
)
AGENT_ROOT = PROJECT_SRC / "agents"
PORTFOLIO_TRADING_TERMS = (
    "br" + "oker",
    "br" + "okerage",
    "order" + "_id",
    "order" + "_type",
    "order" + "_book",
    "fill" + "_id",
    "fill" + "_price",
    "live" + "_trading",
    "paper" + "_trading",
)
AGENT_MODE_TERMS = (
    "br" + "oker",
    "order",
    "fill",
    "live",
    "paper",
)


def test_portfolio_cost_and_risk_layers_do_not_define_trading_execution_surface(
    repo_python_files: tuple[Path, ...],
) -> None:
    trading_terms = _word_regex(PORTFOLIO_TRADING_TERMS)
    hits: list[str] = []
    for path in repo_python_files:
        if not _is_under_any(path, PORTFOLIO_GUARDRAIL_ROOTS):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if trading_terms.search(line) or _has_joined_words(
                line,
                ("execution", "mode"),
            ):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert not hits, "portfolio_trading_surface_found: " + "; ".join(hits)


def test_agent_layer_keeps_proposals_out_of_execution_and_artifact_modes(
    repo_python_files: tuple[Path, ...],
) -> None:
    mode_terms = _word_regex(AGENT_MODE_TERMS)
    hits: list[str] = []
    for path in repo_python_files:
        if not path.as_posix().startswith(AGENT_ROOT.as_posix() + "/"):
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if (
                mode_terms.search(line)
                or _has_joined_words(line, ("execution", "mode"))
                or "REVIEW_SAFE" in line
            ):
                hits.append(f"{path}:{line_number}: {line.strip()}")

    assert not hits, "agent_execution_or_artifact_mode_found: " + "; ".join(hits)


def test_ml_core_has_no_deep_learning_or_tuning_runtime_surface() -> None:
    import quant_research_platform.research.models.supervised as supervised

    source = Path(supervised.__file__).read_text(encoding="utf-8")

    assert supervised.model_algorithm_ids()
    assert "import torch" not in source
    assert "import tensorflow" not in source
    assert "import keras" not in source
    assert not (PROJECT_SRC / "research" / "tuning").exists()
    assert not hasattr(supervised, "select_best_trial")
    assert not hasattr(supervised, "trial_sequence")


def _is_under_any(path: Path, roots: tuple[Path, ...]) -> bool:
    path_text = path.as_posix()
    return any(
        path == root or path_text.startswith(root.as_posix() + "/")
        for root in roots
    )


def _word_regex(terms: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
        re.IGNORECASE,
    )


def _has_joined_words(line: str, words: tuple[str, str]) -> bool:
    tokens = re.findall(r"[A-Za-z]+", line.casefold())
    return any(pair == words for pair in zip(tokens, tokens[1:], strict=False))
