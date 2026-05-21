import { useMemo } from 'react'

import { COMPARISON_CSV_URL, type TablePayload } from './api'
import { DataTable } from './components'

const COMPARISON_PREVIEW_COLUMNS = [
  'public_run_id',
  'n_rows_evaluated',
  'mean_ic',
  'mean_rankic',
  'mean_final_return',
  'max_drawdown',
  'mean_turnover',
  'total_cost',
  'sharpe_like',
]

export function RunComparison({
  comparison,
}: {
  comparison: TablePayload | null
}) {
  const preview = useMemo(
    () => (comparison ? comparisonPreview(comparison) : null),
    [comparison],
  )
  if (!comparison) {
    return null
  }

  return (
    <section className="data-section" id="run-comparison">
      <div className="section-title">
        <h3>运行对比</h3>
        <div className="section-actions">
          <a className="template-link" href={COMPARISON_CSV_URL}>
            下载运行对比 CSV
          </a>
          <span>独立指标排名</span>
        </div>
      </div>
      {preview ? <DataTable label="运行对比表格" payload={preview} /> : null}
    </section>
  )
}

function comparisonPreview(comparison: TablePayload): TablePayload {
  const columns = COMPARISON_PREVIEW_COLUMNS.filter((column) =>
    comparison.columns.includes(column),
  )
  if (columns.length === 0) {
    return comparison
  }
  return {
    ...comparison,
    columns,
    rows: comparison.rows.map((row) =>
      Object.fromEntries(columns.map((column) => [column, row[column]])),
    ),
  }
}
