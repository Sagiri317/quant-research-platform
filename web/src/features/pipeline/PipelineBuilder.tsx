import { useState } from 'react'
import {
  createPipelineRun,
  createPipelineSnapshot,
  getPipelineRunCheckpoints,
  getPipelineSnapshotCheckpoints,
  type PipelineCheckpointSummary,
  type PipelineRunSummary,
} from '../../shared/api/api'
import type { FactorDraft } from '../../shared/factorDraft'

export function PipelineBuilder({
  factorDraft,
  onFactorDraftChange,
}: {
  factorDraft: FactorDraft
  onFactorDraftChange: (draft: FactorDraft) => void
}) {
  const [runId, setRunId] = useState('a_share_reversal_smoke')
  const [dateStart, setDateStart] = useState('2024-01-01')
  const [dateEnd, setDateEnd] = useState('2024-12-31')
  const [costBps, setCostBps] = useState('10')
  const [kernel, setKernel] = useState('dataframe')
  const [submissionMode, setSubmissionMode] = useState('snapshot')
  const [startFrom, setStartFrom] = useState('existing_factors')
  const [status, setStatus] = useState<PipelineRunSummary | null>(null)
  const [checkpoints, setCheckpoints] = useState<PipelineCheckpointSummary[]>([])
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async () => {
    setSubmitting(true)
    setError('')
    try {
      const payload = {
        cost_bps: Number(costBps),
        date_end: dateEnd,
        date_start: dateStart,
        factor_expression: factorDraft.expression,
        factor_template_id: factorDraft.templateId || 'custom',
        kernel,
        label_columns: ['forward_return_1d', 'forward_return_20d'],
        market: 'a_share',
        return_source_column: 'forward_return_1d',
        portfolio_mode: 'long_only',
        run_id: runId,
      }
      const created =
        submissionMode === 'run'
          ? await createPipelineRun({
              ...payload,
              start_from: startFrom as
                | 'existing_factors'
                | 'rebuild_factors'
                | 'rebuild_panel'
                | 'rebuild_interim',
            })
          : await createPipelineSnapshot(payload)
      setStatus(created)
      const checkpointRef = created.run_ref ?? created.run_id
      setCheckpoints(
        submissionMode === 'run'
          ? await getPipelineRunCheckpoints(checkpointRef)
          : await getPipelineSnapshotCheckpoints(checkpointRef),
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : '流程快照生成失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <section className="studio-grid" aria-label="流程配置">
      <div className="studio-hero">
        <div>
          <span className="eyebrow">流程配置</span>
          <h2>A 股运行快照</h2>
        </div>
        <span className="market enabled">只做多</span>
      </div>

      <div className="studio-panel current-factor-panel">
        <div className="form-section-title">
          <h3>当前因子</h3>
          <p>来自因子工作台，可在下方继续微调后生成快照。</p>
        </div>
        <div className="current-factor-grid">
          <span>
            <strong>{factorDraft.templateName}</strong>
            <small>{factorDraft.templateId}</small>
          </span>
          <code>{factorDraft.expression}</code>
        </div>
        <div className="dependency-row" aria-label="当前依赖字段">
          {factorDraft.dependencies.map((dependency) => (
            <span key={dependency}>{dependency}</span>
          ))}
        </div>
      </div>

      <div className="studio-panel pipeline-form">
        <label className="pipeline-field" htmlFor="pipeline-run-id">
          <span className="field-label">运行编号</span>
          <input
            id="pipeline-run-id"
            onChange={(event) => setRunId(event.target.value)}
            value={runId}
          />
        </label>
        <label className="pipeline-field" htmlFor="date-start">
          <span className="field-label">开始日期</span>
          <input
            id="date-start"
            onChange={(event) => setDateStart(event.target.value)}
            value={dateStart}
          />
        </label>
        <label className="pipeline-field" htmlFor="date-end">
          <span className="field-label">结束日期</span>
          <input
            id="date-end"
            onChange={(event) => setDateEnd(event.target.value)}
            value={dateEnd}
          />
        </label>
        <label className="pipeline-field" htmlFor="cost-bps">
          <span className="field-label">成本 bps</span>
          <input
            id="cost-bps"
            onChange={(event) => setCostBps(event.target.value)}
            value={costBps}
          />
        </label>
        <label className="pipeline-field" htmlFor="submission-mode">
          <span className="field-label">提交方式</span>
          <select
            id="submission-mode"
            onChange={(event) => setSubmissionMode(event.target.value)}
            value={submissionMode}
          >
            <option value="snapshot">配置快照</option>
            <option value="run">执行流程</option>
          </select>
        </label>
        <label className="pipeline-field" htmlFor="pipeline-kernel">
          <span className="field-label">计算内核</span>
          <select
            id="pipeline-kernel"
            onChange={(event) => setKernel(event.target.value)}
            value={kernel}
          >
            <option value="dataframe">dataframe</option>
            <option value="tensor">tensor</option>
          </select>
        </label>
        {submissionMode === 'run' ? (
          <label className="pipeline-field" htmlFor="pipeline-start-from">
            <span className="field-label">起始步骤</span>
            <select
              id="pipeline-start-from"
              onChange={(event) => setStartFrom(event.target.value)}
              value={startFrom}
            >
              <option value="existing_factors">existing_factors</option>
              <option value="rebuild_factors">rebuild_factors</option>
              <option value="rebuild_panel">rebuild_panel</option>
              <option value="rebuild_interim">rebuild_interim</option>
            </select>
          </label>
        ) : null}
        <label
          className="pipeline-field pipeline-expression-field"
          htmlFor="pipeline-factor-expression"
        >
          <span className="field-label">因子表达式</span>
          <textarea
            className="expression-editor compact"
            id="pipeline-factor-expression"
            onChange={(event) =>
              onFactorDraftChange({ ...factorDraft, expression: event.target.value })
            }
            value={factorDraft.expression}
          />
        </label>
        <button disabled={submitting} onClick={handleSubmit} type="button">
          {submitting
            ? submissionMode === 'run'
              ? '正在执行流程...'
              : '正在生成快照...'
            : submissionMode === 'run'
              ? '执行流程'
              : '生成快照'}
        </button>
        {error ? <p className="inline-error">{error}</p> : null}
      </div>

      <div className="studio-panel">
        <div className="form-section-title">
          <h3>输出目录</h3>
          <p>
            {status?.run_ref
              ? `pipeline/${status.run_ref}`
              : 'pipeline/(pending)'}
          </p>
        </div>
        <div className="checkpoint-list">
          {checkpoints.length === 0 ? (
            <>
              <span className="checkpoint ready">forward_return_1d</span>
              <span className="checkpoint ready">forward_return_20d</span>
              <span className="checkpoint ready">仅生成配置快照</span>
            </>
          ) : (
            checkpoints.map((checkpoint) => (
              <span className={`checkpoint ${checkpoint.status}`} key={checkpoint.key}>
                {checkpoint.artifact ?? checkpoint.label}
              </span>
            ))
          )}
        </div>
        {status?.message ? <p>{status.message}</p> : null}
      </div>
    </section>
  )
}
