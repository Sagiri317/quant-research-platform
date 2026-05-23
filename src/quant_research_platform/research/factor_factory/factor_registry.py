"""读取并校验受控 factor template registry。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.markets.implementations.a_share.factors import (
    expressions,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    registry as factor_registry,
)
from quant_research_platform.research.factor_factory.candidate import FactorCandidate

REQUIRED_TEMPLATE_FIELDS = ("id", "market", "expression")


@dataclass(frozen=True)
class RegisteredFactorTemplate:
    """注册后的 factor template contract。"""

    template_id: str
    name: str
    version: str
    market: str
    expression: str
    description: str | None
    dependencies: tuple[str, ...]
    output_column: str | None
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    capabilities: tuple[str, ...]
    grid_parameter: str | None = None
    grid_values: tuple[str, ...] = ()
    expression_template: str | None = None
    output_column_template: str | None = None
    availability_rule: str = "trade_date_close"

    def to_candidate(
        self,
        *,
        sample_start: str,
        sample_end: str,
        label_columns: tuple[str, ...],
    ) -> FactorCandidate:
        """从非参数化 template 构造单个 candidate。"""

        return FactorCandidate(
            candidate_id=f"{self.template_id}@{self.version}",
            name=self.name,
            version=self.version,
            expression=self.expression,
            dependencies=self.dependencies,
            output_column=self.output_column,
            sample_start=sample_start,
            sample_end=sample_end,
            label_columns=label_columns,
            inputs=self.inputs,
            outputs=self.outputs,
            capabilities=self.capabilities,
            parameters={},
            availability_rule=self.availability_rule,
            time_semantics=_time_semantics(self.availability_rule),
        )

    def to_candidates(
        self,
        *,
        sample_start: str,
        sample_end: str,
        label_columns: tuple[str, ...],
    ) -> tuple[FactorCandidate, ...]:
        """按受控参数网格展开 template candidates。"""

        if self.grid_parameter is None:
            return (
                self.to_candidate(
                    sample_start=sample_start,
                    sample_end=sample_end,
                    label_columns=label_columns,
                ),
            )
        if self.expression_template is None or self.output_column_template is None:
            raise ValueError(
                f"{self.template_id} grid requires expression/output templates"
            )

        candidates: list[FactorCandidate] = []
        for raw_value in self.grid_values:
            parameters = {self.grid_parameter: _grid_value(raw_value)}
            expression = _format_grid_template(self.expression_template, parameters)
            output_column = _format_grid_template(
                self.output_column_template,
                parameters,
            )
            parsed = expressions.parse_factor_expression(expression)
            value_id = _candidate_parameter_value_id(raw_value)
            candidates.append(
                FactorCandidate(
                    candidate_id=(
                        f"{self.template_id}__{self.grid_parameter}_{value_id}"
                        f"@{self.version}"
                    ),
                    name=f"{self.name} {raw_value}",
                    version=self.version,
                    expression=expression,
                    dependencies=tuple(sorted(parsed.dependencies)),
                    output_column=output_column,
                    sample_start=sample_start,
                    sample_end=sample_end,
                    label_columns=label_columns,
                    inputs=tuple(sorted(parsed.dependencies)),
                    outputs=(output_column,),
                    capabilities=(*self.capabilities, "parameter_grid"),
                    parameters=parameters,
                    availability_rule=self.availability_rule,
                    time_semantics=_time_semantics(self.availability_rule),
                )
            )
        return tuple(candidates)


class FactorTemplateRegistry:
    """只注册受控 YAML templates，不执行任意 Python。"""

    def __init__(self, templates: tuple[RegisteredFactorTemplate, ...]) -> None:
        self._templates = {template.template_id: template for template in templates}
        if len(self._templates) != len(templates):
            raise ValueError("duplicate factor template id")

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        market: str = "a_share",
    ) -> FactorTemplateRegistry:
        """读取目录中的 YAML templates 并返回 registry。"""

        templates = tuple(
            _template_from_yaml(path, market=market)
            for path in sorted(Path(root).glob("*.yaml"))
        )
        return cls(tuple(template for template in templates if template is not None))

    def get(self, template_id: str) -> RegisteredFactorTemplate:
        """返回指定 template。"""

        try:
            return self._templates[template_id]
        except KeyError as exc:
            raise KeyError(f"unknown factor template: {template_id}") from exc

    def candidates(
        self,
        *,
        sample_start: str,
        sample_end: str,
        label_columns: tuple[str, ...],
    ) -> tuple[FactorCandidate, ...]:
        """返回 registry 中所有 templates 对应的 candidates。"""

        candidates: list[FactorCandidate] = []
        for template in self._templates.values():
            candidates.extend(
                template.to_candidates(
                    sample_start=sample_start,
                    sample_end=sample_end,
                    label_columns=label_columns,
                )
            )
        return tuple(candidates)

    def templates(self) -> tuple[RegisteredFactorTemplate, ...]:
        """按 template id 返回已注册 templates。"""

        return tuple(
            self._templates[template_id] for template_id in sorted(self._templates)
        )


def _template_from_yaml(
    path: Path,
    *,
    market: str,
) -> RegisteredFactorTemplate | None:
    values = _read_simple_yaml_mapping(path)
    missing = [field for field in REQUIRED_TEMPLATE_FIELDS if not values.get(field)]
    if missing:
        raise ValueError(f"{path.name} missing required fields: {missing}")
    if values["market"] != market:
        return None

    expression = values["expression"]
    try:
        parsed = expressions.parse_factor_expression(expression)
    except expressions.FactorExpressionError as exc:
        raise ValueError(f"{path.name} has invalid expression: {exc}") from exc

    template_id = values["id"]
    output_column = factor_registry.processed_factor_column_for_template(template_id)
    outputs = (output_column,) if output_column else ()
    grid_parameter = values.get("grid_parameter")
    grid_values = _parse_grid_values(values.get("grid_values"))
    expression_template = values.get("expression_template")
    output_column_template = values.get("output_column_template")
    if bool(grid_parameter) != bool(grid_values):
        raise ValueError(f"{path.name} grid_parameter and grid_values must match")
    return RegisteredFactorTemplate(
        template_id=template_id,
        name=values.get("name", template_id),
        version=values.get("version", "1"),
        market=market,
        expression=expression,
        description=values.get("description"),
        dependencies=tuple(sorted(parsed.dependencies)),
        output_column=output_column,
        inputs=tuple(sorted(parsed.dependencies)),
        outputs=outputs,
        capabilities=("processed_existing", "evaluate"),
        grid_parameter=grid_parameter,
        grid_values=grid_values,
        expression_template=expression_template,
        output_column_template=output_column_template,
        availability_rule=values.get("availability_rule", "trade_date_close"),
    )


def _read_simple_yaml_mapping(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _parse_grid_values(raw_values: str | None) -> tuple[str, ...]:
    if raw_values is None:
        return ()
    values = tuple(value.strip() for value in raw_values.split(",") if value.strip())
    if not values:
        raise ValueError("grid_values must contain at least one value")
    return values


def _grid_value(raw_value: str) -> object:
    return int(raw_value) if raw_value.isdecimal() else raw_value


def _format_grid_template(template: str, parameters: dict[str, object]) -> str:
    try:
        return template.format(**parameters)
    except KeyError as exc:
        raise ValueError(f"grid template references unknown parameter: {exc}") from exc


def _candidate_parameter_value_id(raw_value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in raw_value
    )


def _time_semantics(availability_rule: str) -> dict[str, object]:
    return {
        "decision_time_column": "trade_date",
        "availability_rule": availability_rule,
        "frequency": "1d",
        "timezone": "Asia/Shanghai",
    }


__all__ = ["FactorTemplateRegistry", "RegisteredFactorTemplate"]
