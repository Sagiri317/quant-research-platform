import type { RunSummary } from '../../shared/api/api'

export function RunHeader({
  canDelete,
  deleting,
  onDelete,
  onViewModeChange,
  run,
  viewMode,
}: {
  canDelete: boolean
  deleting: boolean
  onDelete: () => void
  onViewModeChange: (mode: 'workbench' | 'report') => void
  run: RunSummary
  viewMode: 'workbench' | 'report'
}) {
  return (
    <header className="evidence-hero">
      <div className="evidence-hero-main">
        <span className="eyebrow">{boundaryKicker(run)}</span>
        <h2>{run.public_run_id}</h2>
        <strong>可复现的因子评估工作台</strong>
        <p>
          展示组合收益、IC 稳定性、成本换手和结果文件证据链。核心指标优先呈现，
          详细表格和下载材料按需展开。
        </p>
        <div className="view-switch" aria-label="结果视图">
          <button
            aria-pressed={viewMode === 'workbench'}
            onClick={() => onViewModeChange('workbench')}
            type="button"
          >
            工作台
          </button>
          <button
            aria-pressed={viewMode === 'report'}
            onClick={() => onViewModeChange('report')}
            type="button"
          >
            完整报告
          </button>
        </div>
      </div>
      <div className="reviewer-summary">
        <span>运行摘要</span>
        <p>
          这个页面不是交易系统，而是研究证据链：本地数据、可复现结果文件、可解释指标和可下载报告。
        </p>
        <div className="header-chip-row">
          <span>{run.kind}</span>
          <span>{run.data_stage}</span>
          {run.label_columns?.length ? (
            <span>标签列：{run.label_columns.join(', ')}</span>
          ) : null}
          {run.n_combinations ? <span>{run.n_combinations} 个组合</span> : null}
          {run.return_source_column ? <span>收益来源: {run.return_source_column}</span> : null}
          <span>{run.uses_external_data ? '外部数据' : '合成数据'}</span>
        </div>
        {canDelete ? (
          <button
            className="danger-button"
            disabled={deleting}
            onClick={onDelete}
            type="button"
          >
            {deleting ? '删除中...' : '删除回测'}
          </button>
        ) : null}
      </div>
    </header>
  )
}

function boundaryKicker(run: RunSummary): string {
  if (run.data_stage === 'approved_offline_real_data') {
    return '已批准的本地离线数据 · 不连接外部 API'
  }
  if (run.data_stage === 'uploaded_panel') {
    return '用户上传本地面板 · 不连接外部 API'
  }
  return '合成本地结果文件 · 不连接外部 API'
}
