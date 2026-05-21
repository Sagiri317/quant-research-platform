import { useEffect, useRef, useState } from 'react'
import './App.css'
import { CreateRunPanel } from './CreateRunPanel'
import { DataCheckpoints } from './DataCheckpoints'
import { FactorStudio } from './FactorStudio'
import { PipelineBuilder } from './PipelineBuilder'
import { ResultsView } from './ResultsView'
import { Sidebar, type ViewMode } from './Sidebar'
import { DEFAULT_FACTOR_DRAFT, type FactorDraft } from './factorDraft'
import { useArtifactDownloads } from './useArtifactDownloads'
import { useCreateRun } from './useCreateRun'
import { useDeleteRun } from './useDeleteRun'
import { useRunEvidence } from './useRunEvidence'
import { useRuns } from './useRuns'

function App() {
  const {
    loadingRuns,
    removeRun,
    runListError,
    runs,
    selectedId,
    selectedRun,
    setRunListError,
    setSelectedId,
    upsertRunAtTop,
  } = useRuns()
  const [runCacheVersion, setRunCacheVersion] = useState(0)
  const {
    activePayload,
    activeTable,
    activeTableSpec,
    chartTables,
    combinationSummaries,
    display,
    evidenceError,
    loadingEvidence,
    loadingTablePreview,
    runState,
    selectedTableSpecs,
    selectedCombinationId,
    setActiveTable,
    setEvidenceError,
    setSelectedCombinationId,
    setTablePreviewExpanded,
    tablePreviewExpanded,
  } = useRunEvidence(selectedRun, runCacheVersion)
  const [viewMode, setViewMode] = useState<ViewMode>('factor')
  const userSelectedView = useRef(false)
  const [successMessage, setSuccessMessage] = useState('')
  const [factorDraft, setFactorDraft] = useState<FactorDraft>(DEFAULT_FACTOR_DRAFT)
  const {
    createError,
    createForm,
    csvFileName,
    csvPreview,
    handleCreateSubmit,
    handleCsvFileChange,
    setCreateForm,
    submitting,
  } = useCreateRun({
    onCreated(created) {
      setSuccessMessage('')
      setRunCacheVersion((version) => version + 1)
      upsertRunAtTop(created)
      setViewMode('results')
      setSuccessMessage(`已生成回测：${created.public_run_id}`)
    },
  })
  const {
    applyDownloadPreset,
    downloadError,
    downloadNotice,
    downloadOptions,
    downloadPresets,
    downloading,
    handleSelectedDownload,
    selectedDownloadItems,
    toggleDownloadItem,
  } = useArtifactDownloads({
    displayDownloads: display?.downloads,
    selectedRun,
  })
  const canDeleteSelectedRun = selectedRun?.data_stage === 'uploaded_panel'
  const { deleting, handleDeleteSelectedRun } = useDeleteRun({
    canDelete: canDeleteSelectedRun,
    onDeleted(publicRunId) {
      setRunCacheVersion((version) => version + 1)
      removeRun(publicRunId)
      setSuccessMessage(`已删除回测：${publicRunId}`)
    },
    onError(message) {
      setRunListError(message)
    },
    onStart() {
      setRunListError('')
      setEvidenceError('')
      setSuccessMessage('')
    },
    selectedRun,
  })
  const error = runListError || evidenceError
  const loading = loadingRuns || loadingEvidence

  useEffect(() => {
    if (runs.length > 0 && viewMode === 'factor' && !userSelectedView.current) {
      setViewMode('results')
    }
  }, [runs.length, viewMode])

  const handleViewModeChange = (mode: ViewMode) => {
    userSelectedView.current = true
    setViewMode(mode)
  }

  const openPipelineFromFactorStudio = () => {
    userSelectedView.current = true
    setViewMode('pipeline')
  }

  return (
    <main className="workspace">
      <Sidebar
        onSelectRun={setSelectedId}
        onViewModeChange={handleViewModeChange}
        runs={runs}
        selectedId={selectedId}
        viewMode={viewMode}
      />

      <section className="content">
        {error ? <div className="alert">{error}</div> : null}
        {viewMode === 'factor' ? (
          <FactorStudio
            draft={factorDraft}
            onDraftChange={setFactorDraft}
            onOpenPipeline={openPipelineFromFactorStudio}
          />
        ) : null}
        {viewMode === 'pipeline' ? (
          <PipelineBuilder
            factorDraft={factorDraft}
            onFactorDraftChange={setFactorDraft}
          />
        ) : null}
        {viewMode === 'checkpoints' ? <DataCheckpoints /> : null}
        {viewMode === 'create' ? (
          <CreateRunPanel
            createError={createError}
            csvFileName={csvFileName}
            csvPreview={csvPreview}
            form={createForm}
            onFileChange={handleCsvFileChange}
            onFormChange={setCreateForm}
            onSubmit={handleCreateSubmit}
            submitting={submitting}
          />
        ) : null}
        {viewMode === 'results' && loading ? (
          <div className="status">正在读取结果文件...</div>
        ) : null}
        {viewMode === 'results' && successMessage ? (
          <div className="status">{successMessage}</div>
        ) : null}
        {viewMode === 'results' && !loading && !selectedRun ? (
          <div className="status">暂无可用运行</div>
        ) : null}
        {viewMode === 'results' && selectedRun && runState.detail ? (
          <ResultsView
            activePayload={activePayload}
            activeTable={activeTable}
            activeTableSpec={activeTableSpec}
            canDelete={canDeleteSelectedRun}
            combinationSummaries={combinationSummaries}
            deleting={deleting}
            downloadError={downloadError}
            downloadNotice={downloadNotice}
            downloading={downloading}
            downloadOptions={downloadOptions}
            downloadPresets={downloadPresets}
            onActiveTableChange={setActiveTable}
            onDelete={handleDeleteSelectedRun}
            onDownload={handleSelectedDownload}
            onDownloadPreset={applyDownloadPreset}
            onToggleDownloadItem={toggleDownloadItem}
            onToggleTablePreview={() =>
              setTablePreviewExpanded((expanded) => !expanded)
            }
            onSelectedCombinationChange={setSelectedCombinationId}
            reportMarkdown={runState.reportMarkdown}
            run={selectedRun}
            runBoundary={runState.detail.boundary}
            runComparison={runState.comparison}
            selectedCombinationId={selectedCombinationId}
            selectedDownloadItems={selectedDownloadItems}
            selectedTableSpecs={selectedTableSpecs}
            tablePreviewExpanded={tablePreviewExpanded}
            tablePreviewLoading={loadingTablePreview}
            tables={runState.tables}
            chartTables={chartTables}
            displayCharts={display?.charts ?? []}
            key={selectedRun.public_run_id}
          />
        ) : null}
      </section>
    </main>
  )
}


export default App
