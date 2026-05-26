import type { TablePayload } from '../../shared/api/api'
import { formatNumber, formatPercent } from '../../shared/format'

type DiagnosticMetric = {
  label: string
  value: unknown
  detail?: unknown
  format: 'number' | 'percent'
}

export function TradabilityDiagnostics({
  latestDiagnostic,
}: {
  latestDiagnostic?: TablePayload
}) {
  const selectedRow = latestDiagnostic?.rows[0]
  if (!selectedRow) {
    return null
  }

  const metrics: DiagnosticMetric[] = [
    {
      detail: selectedRow.universe_count,
      format: 'number',
      label: '样本池',
      value: selectedRow.universe_count,
    },
    {
      detail: selectedRow.eligible_count,
      format: 'percent',
      label: 'eligible 覆盖',
      value: selectedRow.eligible_rate,
    },
    {
      detail: selectedRow.can_enter_count,
      format: 'percent',
      label: 'can_enter 覆盖',
      value: selectedRow.can_enter_rate,
    },
    {
      detail: selectedRow.can_exit_count,
      format: 'percent',
      label: 'can_exit 覆盖',
      value: selectedRow.can_exit_rate,
    },
    {
      detail: selectedRow.entry_blocked_count,
      format: 'percent',
      label: '入场阻断',
      value: selectedRow.entry_blocked_rate,
    },
    {
      detail: selectedRow.exit_blocked_count,
      format: 'percent',
      label: '出场阻断',
      value: selectedRow.exit_blocked_rate,
    },
  ]
  const fallbackReason = String(selectedRow.fallback_reason ?? '').trim()
  return (
    <section className="tradability-diagnostics" aria-label="可交易性诊断">
      <div className="section-title compact-title">
        <div>
          <h3>Eligibility / Tradability 诊断</h3>
          <p>
            来自 diagnostics/tradability_diagnostics.csv，仅展示 artifact/API
            返回的单个诊断切片；全交易日平均指标后续由 summary artifact 提供。
          </p>
        </div>
        <span>{String(selectedRow.timestamp ?? '最新切片')}</span>
      </div>
      <div className="tradability-grid">
        {metrics.map((metric) => (
          <div className="tradability-card" key={metric.label}>
            <span>{metric.label}</span>
            <strong>{formatMetric(metric.value, metric.format)}</strong>
            {metric.detail !== undefined ? (
              <small>count {formatNumber(numericValue(metric.detail))}</small>
            ) : null}
          </div>
        ))}
      </div>
      {fallbackReason ? (
        <div className="tradability-fallback">
          <span>fallback reason</span>
          <strong>{fallbackReason}</strong>
        </div>
      ) : null}
    </section>
  )
}

function formatMetric(value: unknown, format: DiagnosticMetric['format']) {
  const numeric = numericValue(value)
  if (format === 'percent') {
    return formatPercent(numeric)
  }
  return formatNumber(numeric)
}

function numericValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
