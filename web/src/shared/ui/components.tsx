import type { ReactNode } from 'react'

import type { TablePayload } from '../api/api'
import { formatCell, formatNumber } from '../format'

export function TextField({
  fieldName,
  helperText,
  label,
  onChange,
  required,
  value,
}: {
  fieldName: string
  helperText?: string
  label: string
  onChange: (value: string) => void
  required?: boolean
  value: string
}) {
  return (
    <label className="field">
      <FieldLabel fieldName={fieldName} label={label} required={required} />
      <input
        aria-label={`${label} ${fieldName}`}
        onChange={(event) => onChange(event.target.value)}
        type="text"
        value={value}
      />
      {helperText ? <span className="field-help">{helperText}</span> : null}
    </label>
  )
}

export function SelectField({
  allowEmpty = false,
  fieldName,
  helperText,
  label,
  onChange,
  options,
  required,
  value,
}: {
  allowEmpty?: boolean
  fieldName: string
  helperText?: string
  label: string
  onChange: (value: string) => void
  options: { value: string; label?: string }[] | string[]
  required?: boolean
  value: string
}) {
  const normalizedOptions = options.map((option) =>
    typeof option === 'string' ? { value: option } : option,
  )
  if (allowEmpty && normalizedOptions.length === 0) {
    return (
      <TextField
        fieldName={fieldName}
        helperText={helperText}
        label={label}
        onChange={onChange}
        required={required}
        value={value}
      />
    )
  }
  const values = normalizedOptions.map((option) => option.value)
  const renderedOptions =
    value && !values.includes(value)
      ? [{ value, label: value }, ...normalizedOptions]
      : normalizedOptions
  return (
    <label className="field">
      <FieldLabel fieldName={fieldName} label={label} required={required} />
      <select
        aria-label={`${label} ${fieldName}`}
        onChange={(event) => onChange(event.target.value)}
        value={value}
      >
        {allowEmpty ? <option value="">不使用</option> : null}
        {renderedOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label ?? option.value}
          </option>
        ))}
      </select>
      {helperText ? <span className="field-help">{helperText}</span> : null}
    </label>
  )
}

export function FieldLabel({
  fieldName,
  label,
  required = false,
}: {
  fieldName: string
  label: string
  required?: boolean
}) {
  return (
    <span className="field-label">
      <span className="label-title">{label}</span>
      <code>{fieldName}</code>
      <span className={required ? 'field-badge required' : 'field-badge optional'}>
        {required ? '必填' : '选填'}
      </span>
    </span>
  )
}

export function MetricGroup({
  title,
  children,
}: {
  title: string
  children: ReactNode
}) {
  return (
    <section className="metric-group">
      <h3>{title}</h3>
      <div className="metrics-grid">{children}</div>
    </section>
  )
}

export function MetricCard({
  label,
  value,
}: {
  label: string
  value: number | undefined
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{formatNumber(value)}</strong>
    </div>
  )
}

export function ChartPanel({
  title,
  children,
  chartKind,
  chartMarginLeft,
  dataRowCount,
  testId,
  yAxisWidth,
}: {
  title: string
  children: ReactNode
  chartKind?: string
  chartMarginLeft?: number
  dataRowCount?: number
  testId?: string
  yAxisWidth?: number
}) {
  return (
    <section
      className="chart-panel"
      data-chart-kind={chartKind}
      data-chart-margin-left={chartMarginLeft}
      data-row-count={dataRowCount}
      data-y-axis-width={yAxisWidth}
      data-testid={testId}
    >
      <h3>{title}</h3>
      {children}
    </section>
  )
}

export function DataTable({
  label,
  maxRows,
  payload,
}: {
  label: string
  maxRows?: number
  payload: TablePayload
}) {
  const visibleRows =
    maxRows === undefined ? payload.rows : payload.rows.slice(0, maxRows)
  const totalRows = payload.total_rows ?? payload.rows.length
  const hiddenRows =
    maxRows === undefined ? 0 : Math.max(totalRows - visibleRows.length, 0)
  return (
    <div className="table-preview">
      {hiddenRows > 0 ? (
        <div className="table-limit-note">
          仅预览前 {maxRows} 行，共 {totalRows} 行。
        </div>
      ) : null}
      <div className="table-wrap">
        <table aria-label={label}>
          <thead>
            <tr>
              {payload.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {payload.columns.map((column) => (
                  <td key={column}>{formatCell(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
