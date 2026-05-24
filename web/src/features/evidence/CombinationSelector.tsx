import { useMemo, useState } from 'react'
import type { TablePayload } from '../../shared/api/api'

const MISSING_COMBINATION_ID = '__missing_combination__'
const COMBINATION_LABELS: Record<string, string> = {
  cost_bps: '成本',
  cost_id: '成本配置',
  direction: '方向',
  feature: '因子',
  label: '标签',
  portfolio_mode: '组合模式',
  weight_id: '权重配置',
  weight_method: '权重方法',
}
const COMBINATION_DIMENSION_COLUMNS = [
  'feature',
  'label',
  'weight_id',
  'cost_id',
  'weight_method',
  'direction',
  'portfolio_mode',
  'cost_bps',
]

type LocalSelection = {
  rowsSignature: string
  sourceCombinationId: string
  values: Record<string, string>
}

export function CombinationSelector({
  combinations,
  onChange,
  selectedCombinationId,
}: {
  combinations: TablePayload | undefined
  onChange: (combinationId: string) => void
  selectedCombinationId: string
}) {
  const rows = useMemo(() => combinations?.rows ?? [], [combinations])
  const dimensions = useMemo(
    () => combinationDimensions(combinations),
    [combinations],
  )
  const selectedRow = rows.find(
    (row) => String(row.combination_id ?? '') === selectedCombinationId,
  )
  const initialValues = useMemo(
    () => selectedValuesForRow(selectedRow ?? rows[0], dimensions),
    [dimensions, rows, selectedRow],
  )
  const rowsSignature = useMemo(
    () =>
      rows
        .map((row) =>
          [
            row.combination_id,
            ...dimensions.map((dimension) => row[dimension]),
          ].join(':'),
        )
        .join('|'),
    [dimensions, rows],
  )
  const [localSelection, setLocalSelection] = useState<LocalSelection>({
    rowsSignature,
    sourceCombinationId: selectedCombinationId,
    values: initialValues,
  })
  const canUseLocalSelection =
    localSelection.rowsSignature === rowsSignature &&
    (selectedCombinationId === MISSING_COMBINATION_ID ||
      localSelection.sourceCombinationId === selectedCombinationId)
  const selectedValues = canUseLocalSelection
    ? localSelection.values
    : initialValues

  if (rows.length <= 1 || dimensions.length === 0) {
    return null
  }

  const matchedRow = findMatchingCombination(rows, dimensions, selectedValues)
  const hasMatch = matchedRow !== undefined

  function handleDimensionChange(dimension: string, value: string) {
    const nextValues = { ...selectedValues, [dimension]: value }
    const nextMatch = findMatchingCombination(rows, dimensions, nextValues)
    const nextCombinationId =
      nextMatch === undefined
        ? MISSING_COMBINATION_ID
        : String(nextMatch.combination_id ?? '')
    setLocalSelection({
      rowsSignature,
      sourceCombinationId: nextCombinationId,
      values: nextValues,
    })
    onChange(nextCombinationId)
  }

  return (
    <section className="combination-selector" aria-label="组合选择">
      <div className="combination-head">
        <div>
          <h3>组合筛选</h3>
          <p>选择当前图表和摘要使用的因子组合。</p>
        </div>
        <span
          className={hasMatch ? 'combination-status' : 'combination-status warning'}
        >
          {hasMatch ? '已匹配' : '未匹配'}
        </span>
      </div>
      <div className="combination-grid">
        {dimensions.map((dimension) => (
          <label className="field" key={dimension}>
            <span className="field-label">
              <span className="label-title">选择</span>
              <code>{dimension}</code>
            </span>
            <select
              aria-label={`选择 ${dimension}`}
              onChange={(event) =>
                handleDimensionChange(dimension, event.target.value)
              }
              value={selectedValues[dimension] ?? ''}
            >
              {dimensionValues(rows, dimension).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <span
        className={
          hasMatch
            ? 'field-help current-combination'
            : 'field-help warning current-combination'
        }
      >
        {hasMatch
          ? currentCombinationText(matchedRow, dimensions)
          : '没有匹配的组合配置'}
      </span>
    </section>
  )
}

function currentCombinationText(
  row: Record<string, unknown>,
  dimensions: string[],
): string {
  const parts = dimensions
    .filter((dimension) => row[dimension] != null)
    .map(
      (dimension) =>
        `${COMBINATION_LABELS[dimension] ?? dimension} ${String(row[dimension])}`,
    )
  return `当前组合：${parts.join('，')}。`
}

function combinationDimensions(payload: TablePayload | undefined): string[] {
  const columns = payload?.columns ?? []
  return COMBINATION_DIMENSION_COLUMNS.filter((column) => columns.includes(column))
}

function dimensionValues(
  rows: Record<string, unknown>[],
  dimension: string,
): string[] {
  return Array.from(
    new Set(
      rows
        .map((row) => row[dimension])
        .filter((value): value is string | number => value != null)
        .map(String),
    ),
  )
}

function selectedValuesForRow(
  row: Record<string, unknown> | undefined,
  dimensions: string[],
): Record<string, string> {
  return Object.fromEntries(
    dimensions.map((dimension) => [dimension, String(row?.[dimension] ?? '')]),
  )
}

function findMatchingCombination(
  rows: Record<string, unknown>[],
  dimensions: string[],
  selectedValues: Record<string, string>,
) {
  return rows.find((row) =>
    dimensions.every(
      (dimension) => String(row[dimension] ?? '') === selectedValues[dimension],
    ),
  )
}
