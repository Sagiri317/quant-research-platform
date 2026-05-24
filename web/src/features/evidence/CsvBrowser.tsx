import {
  type RunSummary,
  type RunTableSpec,
  type TablePayload,
  artifactDownloadUrl,
} from '../../shared/api/api'
import { DataTable } from '../../shared/ui/components'

const CSV_PREVIEW_ROW_LIMIT = 20

export function CsvBrowser({
  activePayload,
  activeTable,
  activeTableSpec,
  expanded,
  loading,
  onActiveTableChange,
  onToggleExpanded,
  run,
  tableSpecs,
}: {
  activePayload: TablePayload | undefined
  activeTable: string
  activeTableSpec: RunTableSpec | undefined
  expanded: boolean
  loading: boolean
  onActiveTableChange: (value: string) => void
  onToggleExpanded: () => void
  run: RunSummary
  tableSpecs: RunTableSpec[]
}) {
  return (
    <section className="data-section">
      <div className="section-title">
        <h3>CSV 明细浏览</h3>
        <div className="section-actions">
          {activeTableSpec ? (
            <a
              className="template-link"
              href={artifactDownloadUrl(run, activeTableSpec.downloadItem)}
            >
              下载当前 CSV
            </a>
          ) : null}
          <span>{activeTableSpec?.label ?? activeTable}</span>
        </div>
      </div>
      <div className="csv-browser-controls">
        <label className="csv-table-select">
          <span>选择明细表</span>
          <select
            aria-label="选择明细表"
            onChange={(event) => onActiveTableChange(event.target.value)}
            value={activeTable}
          >
            {tableSpecs.map((spec) => (
              <option key={spec.key} value={spec.key}>
                {spec.label}
              </option>
            ))}
          </select>
        </label>
        <button
          className="preset-button"
          onClick={onToggleExpanded}
          type="button"
        >
          {expanded ? '收起预览' : '展开预览'}
        </button>
      </div>
      {expanded && loading ? (
        <div className="csv-preview-placeholder">正在读取当前明细表预览...</div>
      ) : activePayload && expanded ? (
        <DataTable
          label={`${activeTableSpec?.label ?? activeTable} 表格`}
          maxRows={CSV_PREVIEW_ROW_LIMIT}
          payload={activePayload}
        />
      ) : (
        <div className="csv-preview-placeholder">
          默认不展开明细行。选择表格后点击“展开预览”，最多显示前{' '}
          {CSV_PREVIEW_ROW_LIMIT} 行；完整数据请下载 CSV。
        </div>
      )}
    </section>
  )
}
