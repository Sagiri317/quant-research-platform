import type { DownloadOption } from './downloads'

export function ArtifactDownloads({
  downloadError,
  downloadNotice,
  downloading,
  downloadOptions,
  onDownload,
  onPreset,
  onToggleItem,
  selectedItems,
}: {
  downloadError: string
  downloadNotice: string
  downloading: boolean
  downloadOptions: DownloadOption[]
  onDownload: () => void
  onPreset: (preset: 'all' | 'report' | 'tables' | 'repro') => void
  onToggleItem: (item: string) => void
  selectedItems: string[]
}) {
  return (
    <section className="data-section">
      <div className="section-title">
        <h3>结果文件下载</h3>
        <span>选择后打包</span>
      </div>
      <div className="download-panel">
        <div className="download-toolbar">
          <div className="download-copy">
            <strong>常用组合</strong>
            <span>选择一个预设，或在下面逐项勾选。</span>
          </div>
          <div className="download-actions" aria-label="下载预设">
            <button
              className="preset-button"
              onClick={() => onPreset('all')}
              type="button"
            >
              全选
            </button>
            <button
              className="preset-button"
              onClick={() => onPreset('report')}
              type="button"
            >
              仅报告
            </button>
            <button
              className="preset-button"
              onClick={() => onPreset('tables')}
              type="button"
            >
              仅表格
            </button>
            <button
              className="preset-button"
              onClick={() => onPreset('repro')}
              type="button"
            >
              复现包
            </button>
          </div>
        </div>
        <div className="download-picker-head">
          <strong>选择内容</strong>
            <span>{selectedItems.length} 项已选</span>
        </div>
        <div className="download-grid">
          {downloadOptions.map((option) => {
            const selected = selectedItems.includes(option.item)
            return (
              <label
                className={selected ? 'download-option selected' : 'download-option'}
                key={option.item}
              >
                <input
                  checked={selected}
                  onChange={() => onToggleItem(option.item)}
                  type="checkbox"
                />
                <span className="download-check" aria-hidden="true">
                  {selected ? '✓' : ''}
                </span>
                <span>{option.label}</span>
              </label>
            )
          })}
        </div>
        {downloadError ? <div className="alert">{downloadError}</div> : null}
        {downloadNotice ? (
          <div className="status download-status">{downloadNotice}</div>
        ) : null}
        <div className="download-submit">
          <span>
            {selectedItems.length} / {downloadOptions.length}
          </span>
          <button
            className="submit-run"
            disabled={downloading}
            onClick={onDownload}
            type="button"
          >
            {downloading ? '下载中...' : '下载所选'}
          </button>
        </div>
      </div>
    </section>
  )
}
