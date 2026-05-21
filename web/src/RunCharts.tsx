import type { RunDisplayChart, TablePayload } from './api'
import { RunBarChartPanel, RunLineChartPanel } from './charts'
import { formatPercent } from './format'
import { useMemo, useState } from 'react'

const ALLOWED_CHART_KINDS = new Set(['line', 'bar'])
const MAX_DENSE_BAR_ROWS = 160

export function RunCharts({
  charts,
  mode = 'workbench',
  tables,
}: {
  charts: RunDisplayChart[]
  mode?: 'workbench' | 'report'
  tables: Record<string, TablePayload>
}) {
  const renderableCharts = useMemo(
    () =>
      charts
        .flatMap(expandChart)
        .filter((chart) => isRenderableChart(chart, tables)),
    [charts, tables],
  )
  const primaryCharts = renderableCharts.filter((chart) => !isDiagnosticChart(chart))
  const diagnosticCharts = renderableCharts.filter(isDiagnosticChart)
  const [selectedDiagnosticId, setSelectedDiagnosticId] = useState(
    diagnosticCharts[0]?.id ?? '',
  )
  const selectedDiagnostic =
    diagnosticCharts.find((chart) => chart.id === selectedDiagnosticId) ??
    diagnosticCharts[0]

  if (mode === 'workbench') {
    return (
      <section className="chart-grid workbench-chart-grid">
        <div className="primary-chart-row">
          {primaryCharts.map((chart) => renderChart(chart, tables))}
        </div>
        {selectedDiagnostic ? (
          <div className="diagnostic-row">
            <section className="diagnostic-panel">
              <div className="diagnostic-head">
                <div>
                  <h3>诊断图</h3>
                  <p>辅助检查单期收益、换手和成本，不抢占核心图权重。</p>
                </div>
                <div className="diagnostic-tabs" role="group" aria-label="诊断图选择">
                  {diagnosticCharts.map((chart) => (
                    <button
                      aria-pressed={chart.id === selectedDiagnostic.id}
                      key={chart.id}
                      onClick={() => setSelectedDiagnosticId(chart.id)}
                      type="button"
                    >
                      {chart.title}
                    </button>
                  ))}
                </div>
              </div>
              {renderChart(selectedDiagnostic, tables)}
            </section>
            <DiagnosticSummary tables={tables} />
          </div>
        ) : null}
      </section>
    )
  }

  return (
    <section className="chart-grid report-chart-grid">
      {renderableCharts.map((chart) => renderChart(chart, tables))}
    </section>
  )
}

function renderChart(
  chart: RunDisplayChart,
  tables: Record<string, TablePayload>,
) {
  const rows = tables[chart.table].rows
  const kind = effectiveChartKind(chart, rows.length)
  const props = {
    chartKind: kind,
    description: chart.description,
    includeZero: chart.include_zero,
    rows,
    series: chart.series,
    testId: `${chart.id.replaceAll('_', '-')}-chart`,
    title: chart.title,
    valueFormatter: chartUsesPercent(chart) ? formatPercent : undefined,
    xKey: chart.x,
  }
  return kind === 'bar' ? (
    <RunBarChartPanel key={chart.id} {...props} />
  ) : (
    <RunLineChartPanel key={chart.id} {...props} />
  )
}

function expandChart(chart: RunDisplayChart): RunDisplayChart[] {
  if (chart.id !== 'return_turnover_cost') {
    return [chart]
  }
  return chart.series.map((series) => ({
    ...chart,
    description: returnDiagnosticDescription(series.key),
    id: series.key,
    include_zero: true,
    series: [series],
    title: returnDiagnosticTitle(series.key),
  }))
}

function returnDiagnosticTitle(key: string): string {
  if (key === 'net_portfolio_return') {
    return '单期净收益'
  }
  if (key === 'turnover') {
    return '换手率'
  }
  if (key === 'cost') {
    return '单期成本'
  }
  return key
}

function isDiagnosticChart(chart: RunDisplayChart): boolean {
  return ['cost', 'net_portfolio_return', 'turnover'].includes(chart.id)
}

function DiagnosticSummary({ tables }: { tables: Record<string, TablePayload> }) {
  const rows = tables.returns?.rows ?? []
  const turnover = meanNumeric(rows, 'turnover')
  const cost = sumNumeric(rows, 'cost')
  const maxCost = maxNumeric(rows, 'cost')
  return (
    <section className="diagnostic-summary">
      <div>
        <h3>快速诊断摘要</h3>
        <p>快速解释交易摩擦和成本拖累。</p>
      </div>
      <div className="diagnostic-note">
        <strong>平均换手</strong>
        <span>{formatDiagnosticNumber(turnover)}</span>
      </div>
      <div className="diagnostic-note">
        <strong>累计成本</strong>
        <span>{formatDiagnosticNumber(cost)}</span>
      </div>
      <div className="diagnostic-note">
        <strong>最大单期成本</strong>
        <span>{formatDiagnosticNumber(maxCost)}</span>
      </div>
    </section>
  )
}

function meanNumeric(rows: Record<string, unknown>[], key: string): number | undefined {
  const values = numericColumn(rows, key)
  if (values.length === 0) {
    return undefined
  }
  return values.reduce((total, value) => total + value, 0) / values.length
}

function sumNumeric(rows: Record<string, unknown>[], key: string): number | undefined {
  const values = numericColumn(rows, key)
  if (values.length === 0) {
    return undefined
  }
  return values.reduce((total, value) => total + value, 0)
}

function maxNumeric(rows: Record<string, unknown>[], key: string): number | undefined {
  const values = numericColumn(rows, key)
  return values.length === 0 ? undefined : Math.max(...values)
}

function numericColumn(rows: Record<string, unknown>[], key: string): number[] {
  return rows
    .map((row) => row[key])
    .filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
}

function formatDiagnosticNumber(value: number | undefined): string {
  if (value === undefined) {
    return '--'
  }
  return value.toLocaleString('en-US', { maximumFractionDigits: 4 })
}

function returnDiagnosticDescription(key: string): string {
  if (key === 'net_portfolio_return') {
    return '每期组合净收益'
  }
  if (key === 'turnover') {
    return '每期换手强度'
  }
  if (key === 'cost') {
    return '每期成本拖累'
  }
  return '组合收益诊断'
}

function effectiveChartKind(
  chart: RunDisplayChart,
  rowCount: number,
): RunDisplayChart['kind'] {
  if (chart.kind === 'bar' && rowCount > MAX_DENSE_BAR_ROWS) {
    return 'line'
  }
  return chart.kind
}

function isRenderableChart(
  chart: RunDisplayChart,
  tables: Record<string, TablePayload>,
): boolean {
  if (!ALLOWED_CHART_KINDS.has(chart.kind)) {
    return false
  }
  const table = tables[chart.table]
  if (!table) {
    return false
  }
  if (table.rows.length === 0) {
    return true
  }
  if (!table.columns.includes(chart.x)) {
    return false
  }
  return chart.series.some(
    (series) =>
      table.columns.includes(series.key) &&
      table.rows.some((row) => typeof row[series.key] === 'number'),
  )
}

function chartUsesPercent(chart: RunDisplayChart): boolean {
  return chart.series.some((series) => series.format === 'percent')
}
