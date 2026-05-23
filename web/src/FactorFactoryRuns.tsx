import { useEffect, useMemo, useState } from 'react'
import {
  factorFactoryArtifactDownloadUrl,
  fetchFactorFactoryRun,
  fetchFactorFactoryTable,
  listFactorFactoryRuns,
  type FactorFactoryDisplayTable,
  type FactorFactoryDownloadItem,
  type FactorFactoryRunDetail,
  type FactorFactoryRunSummary,
  type TablePayload,
} from './api'

const CANDIDATE_PAGE_SIZE = 50

export function FactorFactoryRuns() {
  const [runs, setRuns] = useState<FactorFactoryRunSummary[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [candidateOffset, setCandidateOffset] = useState(0)
  const [detail, setDetail] = useState<FactorFactoryRunDetail | null>(null)
  const [candidateTable, setCandidateTable] = useState<TablePayload | null>(null)
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [loadingTable, setLoadingTable] = useState(false)
  const [error, setError] = useState('')
  const [tableError, setTableError] = useState('')

  useEffect(() => {
    let cancelled = false
    listFactorFactoryRuns()
      .then((loaded) => {
        if (cancelled) {
          return
        }
        setRuns(loaded)
        setError('')
        if (loaded[0]?.run_id) {
          setLoadingDetail(true)
        }
        setSelectedRunId((current) => current || loaded[0]?.run_id || '')
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason))
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingRuns(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedRunId) {
      return
    }
    let cancelled = false
    fetchFactorFactoryRun(selectedRunId)
      .then((loadedDetail) => {
        if (cancelled) {
          return
        }
        const nextPreviewTable = selectPreviewTable(loadedDetail)
        setCandidateTable(null)
        setTableError('')
        setLoadingTable(Boolean(nextPreviewTable))
        setDetail(loadedDetail)
        setError('')
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason))
          setDetail(null)
          setCandidateTable(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingDetail(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [selectedRunId])

  const previewTable = useMemo(() => selectPreviewTable(detail), [detail])

  useEffect(() => {
    if (!detail || !previewTable) {
      return
    }
    let cancelled = false
    fetchFactorFactoryTable(
      detail.run.run_id,
      previewTable.key,
      CANDIDATE_PAGE_SIZE,
      candidateOffset,
    )
      .then((loadedTable) => {
        if (cancelled) {
          return
        }
        setCandidateTable(loadedTable)
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setTableError(reason instanceof Error ? reason.message : String(reason))
          setCandidateTable(null)
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingTable(false)
        }
      })
    return () => {
      cancelled = true
    }
  }, [detail, previewTable, candidateOffset])

  function selectRun(runId: string) {
    if (runId !== selectedRunId) {
      setDetail(null)
      setCandidateTable(null)
      setLoadingDetail(true)
      setLoadingTable(false)
      setError('')
      setTableError('')
    }
    setSelectedRunId(runId)
    setCandidateOffset(0)
  }

  function selectCandidateOffset(offset: number) {
    setCandidateTable(null)
    setLoadingTable(true)
    setTableError('')
    setCandidateOffset(offset)
  }

  const summary = detail?.run ?? runs.find((run) => run.run_id === selectedRunId)
  const visibleColumns = useMemo(
    () => candidateTable?.columns ?? [],
    [candidateTable?.columns],
  )
  const tableHasMore = candidateTable
    ? hasMoreRows(candidateTable, candidateOffset)
    : false

  return (
    <section className="studio-grid" aria-label="因子工厂">
      <div className="studio-hero">
        <div>
          <span className="eyebrow">Factor Factory</span>
          <h2>候选因子批量结果</h2>
        </div>
        <div className="market-strip" aria-label="候选状态">
          <span className="market enabled">A 股</span>
          <span className="market planned">仅展示公开运行引用</span>
        </div>
      </div>

      {error ? <div className="alert">{error}</div> : null}
      {loadingRuns ? <div className="status">正在读取因子工厂运行...</div> : null}
      {!loadingRuns && runs.length === 0 ? (
        <div className="status">暂无 Factor Factory 运行</div>
      ) : null}

      {runs.length > 0 ? (
        <div className="factory-layout">
          <div className="studio-panel factory-run-list" aria-label="因子工厂运行">
            {runs.map((run) => (
              <button
                className={
                  run.run_id === selectedRunId
                    ? 'factory-run active'
                    : 'factory-run'
                }
                key={run.run_id}
                onClick={() => selectRun(run.run_id)}
                type="button"
              >
                <strong>{run.run_id}</strong>
                <span>
                  {run.success_count}/{run.candidate_count} success
                </span>
              </button>
            ))}
          </div>

          <div className="studio-panel factory-detail">
            {summary ? (
              <>
                <div className="form-section-title">
                  <h3>{summary.run_id}</h3>
                  <p>schema {summary.artifact_schema_version}</p>
                </div>
                <div className="metrics-grid factory-metrics">
                  <Metric label="候选数" value={summary.candidate_count} />
                  <Metric label="成功" value={summary.success_count} />
                  <Metric label="失败" value={summary.failed_count} />
                </div>
              </>
            ) : null}

            {loadingDetail ? (
              <div className="status">正在读取候选明细...</div>
            ) : null}

            {!loadingDetail && loadingTable ? (
              <div className="status">Loading table preview...</div>
            ) : null}

            {!loadingDetail && detail && !previewTable ? (
              <div className="status">No preview table is exposed for this run.</div>
            ) : null}

            {tableError ? (
              <div className="status factory-table-empty">
                <strong>{tableError}</strong>
                <span>Table preview is unavailable. Downloads remain available.</span>
              </div>
            ) : null}

            {candidateTable && visibleColumns.length > 0 ? (
              <div className="data-section factory-table-section">
                <div className="factory-table-heading">
                  <div>
                    <h3>{previewTable?.label ?? 'Table preview'}</h3>
                    <p>
                      显示第 {candidateOffset + 1}-
                      {candidateOffset + candidateTable.rows.length} 行
                    </p>
                  </div>
                  <div className="table-pager" aria-label="候选分页">
                    <button
                      disabled={candidateOffset === 0 || loadingTable}
                      onClick={() =>
                        selectCandidateOffset(
                          Math.max(candidateOffset - CANDIDATE_PAGE_SIZE, 0),
                        )
                      }
                      type="button"
                    >
                      上一页
                    </button>
                    <button
                      disabled={!tableHasMore || loadingTable}
                      onClick={() =>
                        selectCandidateOffset(candidateOffset + CANDIDATE_PAGE_SIZE)
                      }
                      type="button"
                    >
                      下一页
                    </button>
                  </div>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {visibleColumns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {candidateTable.rows.map((row, index) => (
                        <tr key={`${row.candidate_id ?? index}`}>
                          {visibleColumns.map((column) => (
                            <td key={column}>{formatCell(row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            {detail ? (
              <div className="download-list factory-downloads">
                {detail.display.downloads.items.map((item) => (
                  <a
                    href={factorFactoryArtifactDownloadUrl(detail.run.run_id, item.item)}
                    key={item.item}
                  >
                    {item.label}
                  </a>
                ))}
                {detail.display.downloads.presets.map((preset) => (
                  <div className="factory-download-preset" key={preset.key}>
                    <strong>{preset.label}</strong>
                    <div>
                      {preset.items.map((item) => (
                        <a
                          href={factorFactoryArtifactDownloadUrl(detail.run.run_id, item)}
                          key={item}
                        >
                          {presetItemLabel(item, detail.display.downloads.items)}
                        </a>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </section>
  )
}

function selectPreviewTable(
  detail: FactorFactoryRunDetail | null,
): FactorFactoryDisplayTable | null {
  const tables = detail?.display.tables ?? []
  return (
    tables.find((table) => table.key === 'candidate_summary') ??
    tables.find((table) => table.group === 'candidate') ??
    tables[0] ??
    null
  )
}

function presetItemLabel(
  item: string,
  downloadItems: FactorFactoryDownloadItem[],
): string {
  return downloadItems.find((downloadItem) => downloadItem.item === item)?.item ?? item
}

function hasMoreRows(table: TablePayload, fallbackOffset: number): boolean {
  if (typeof table.has_more === 'boolean') {
    return table.has_more
  }
  if (typeof table.total_rows === 'number') {
    const offset = typeof table.offset === 'number' ? table.offset : fallbackOffset
    return offset + table.rows.length < table.total_rows
  }
  return false
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value.toLocaleString('zh-CN')}</strong>
    </div>
  )
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : ''
  }
  return String(value)
}
