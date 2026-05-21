import type { CsvPreview } from './runForm'

export function CsvPreviewPanel({ csvPreview }: { csvPreview: CsvPreview | null }) {
  if (!csvPreview) {
    return null
  }
  return (
    <section className="csv-preview wide" aria-label="CSV 预览">
      <div className="preview-head">
        <strong>CSV 预览</strong>
        <span>{csvPreview.rowCount} 行数据</span>
      </div>
      <div className="preview-columns">
        {csvPreview.columns.map((column) => (
          <code key={column}>{column}</code>
        ))}
      </div>
    </section>
  )
}
