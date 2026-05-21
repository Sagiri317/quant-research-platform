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

function isDiagnosticChart(chart: RunDisplayChart): boolean {
  return chart.section === 'diagnostic'
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
