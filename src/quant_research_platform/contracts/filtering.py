"""Research sample filtering contracts and helpers."""

from __future__ import annotations

from typing import Literal

EVALUATION_FILTER_MODES = ("eligible", "tradable", "unfiltered")
SAMPLE_FILTER_MODES = ("eligible", "unfiltered")
FILTER_COLUMNS = ("eligible", "can_enter", "can_exit")

EvaluationFilterMode = Literal["eligible", "tradable", "unfiltered"]
SampleFilterMode = Literal["eligible", "unfiltered"]


def validate_evaluation_filter_mode(mode: str) -> EvaluationFilterMode:
    if mode not in EVALUATION_FILTER_MODES:
        raise ValueError("unsupported_evaluation_filter_mode")
    return mode  # type: ignore[return-value]


def validate_sample_filter_mode(mode: str, *, owner: str) -> SampleFilterMode:
    if mode not in SAMPLE_FILTER_MODES:
        raise ValueError(f"unsupported_{owner}_sample_filter_mode")
    return mode  # type: ignore[return-value]


def reject_filter_input_columns(input_columns: tuple[str, ...]) -> None:
    invalid = [column for column in input_columns if column in FILTER_COLUMNS]
    if invalid:
        raise ValueError(f"filter_column_in_input_columns: {', '.join(invalid)}")


def required_filter_columns(mode: str) -> tuple[str, ...]:
    if mode == "eligible":
        return ("eligible",)
    if mode == "tradable":
        return ("eligible", "can_enter", "can_exit")
    if mode == "unfiltered":
        return ()
    raise ValueError("unsupported_filter_mode")


def filter_definition(mode: str) -> str:
    if mode == "eligible":
        return "eligible == True"
    if mode == "tradable":
        return "eligible == True and can_enter == True and can_exit == True"
    if mode == "unfiltered":
        return "no eligible/can_enter/can_exit filter"
    raise ValueError("unsupported_filter_mode")


__all__ = [
    "EvaluationFilterMode",
    "SampleFilterMode",
    "filter_definition",
    "reject_filter_input_columns",
    "required_filter_columns",
    "validate_evaluation_filter_mode",
    "validate_sample_filter_mode",
]
