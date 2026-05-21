import type { RunSummary } from './api'

export type ViewMode = 'factor' | 'pipeline' | 'checkpoints' | 'results' | 'create'

const VIEW_LABELS: Record<ViewMode, string> = {
  checkpoints: '数据检查点',
  create: 'CSV 上传（开发）',
  factor: '因子工作台',
  pipeline: '流程配置',
  results: '证据工作台',
}

export function Sidebar({
  onSelectRun,
  onViewModeChange,
  runs,
  selectedId,
  viewMode,
}: {
  onSelectRun: (publicRunId: string) => void
  onViewModeChange: (mode: ViewMode) => void
  runs: RunSummary[]
  selectedId: string
  viewMode: ViewMode
}) {
  return (
    <aside className="sidebar" aria-label="平台导航">
      <div className="brand">
        <span className="eyebrow">量化研究平台</span>
        <h1>研究证据工作台</h1>
      </div>
      <div className="mode-actions" aria-label="页面模式">
        {(['factor', 'pipeline', 'checkpoints', 'results', 'create'] as ViewMode[]).map(
          (mode) => (
            <button
              aria-pressed={viewMode === mode}
              className={viewMode === mode ? 'mode active' : 'mode'}
              key={mode}
              onClick={() => onViewModeChange(mode)}
              type="button"
            >
              {VIEW_LABELS[mode]}
            </button>
          ),
        )}
      </div>
      {viewMode === 'results' ? (
        <div className="run-list">
          {runs.length === 0 ? (
            <div className="sidebar-note">暂无可用运行</div>
          ) : (
            runs.map((run) => (
              <button
                className={run.public_run_id === selectedId ? 'run active' : 'run'}
                key={run.public_run_id}
                onClick={() => onSelectRun(run.public_run_id)}
                type="button"
              >
                <span>{run.public_run_id}</span>
                <small>{run.domain}</small>
              </button>
            ))
          )}
        </div>
      ) : (
        <div className="sidebar-note">已有评估结果仍可在证据工作台查看。</div>
      )}
    </aside>
  )
}
