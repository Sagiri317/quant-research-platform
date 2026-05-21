import { AnimatePresence, motion } from 'framer-motion'
import { useMemo, useState } from 'react'
import {
  type RunSummary,
  type RunTableSpec,
  type RunDisplayChart,
  type RunDisplayDownloadPreset,
  type TablePayload,
} from './api'
import { ArtifactDownloads } from './ArtifactDownloads'
import { CombinationSelector } from './CombinationSelector'
import { CsvBrowser } from './CsvBrowser'
import type { DownloadOption } from './downloads'
import {
  MetricDetails,
  MetricSummary,
  type SelectedCombinationMetrics,
} from './MetricSummary'
import { ReportView } from './reportView'
import { RunComparison } from './RunComparison'
import { RunCharts } from './RunCharts'
import { RunHeader } from './RunHeader'

type EvidenceViewMode = 'workbench' | 'report'
type DeliveryTab = 'metrics' | 'report' | 'csv' | 'comparison' | 'downloads'

export function ResultsView({
  activePayload,
  activeTable,
  activeTableSpec,
  canDelete,
  chartTables,
  combinationSummaries,
  displayCharts,
  deleting,
  downloadError,
  downloadNotice,
  downloading,
  downloadOptions,
  downloadPresets,
  onActiveTableChange,
  onDelete,
  onDownload,
  onDownloadPreset,
  onSelectedCombinationChange,
  onToggleDownloadItem,
  onToggleTablePreview,
  reportMarkdown,
  run,
  runBoundary,
  runComparison,
  selectedCombinationId,
  selectedTableSpecs,
  selectedDownloadItems,
  tablePreviewExpanded,
  tablePreviewLoading,
  tables,
}: {
  activePayload: TablePayload | undefined
  activeTable: string
  activeTableSpec: RunTableSpec | undefined
  canDelete: boolean
  chartTables: Record<string, TablePayload>
  combinationSummaries: TablePayload | null
  displayCharts: RunDisplayChart[]
  deleting: boolean
  downloadError: string
  downloadNotice: string
  downloading: boolean
  downloadOptions: DownloadOption[]
  downloadPresets: RunDisplayDownloadPreset[]
  onActiveTableChange: (table: string) => void
  onDelete: () => void
  onDownload: () => void
  onDownloadPreset: (preset: string) => void
  onSelectedCombinationChange: (combinationId: string) => void
  onToggleDownloadItem: (item: string) => void
  onToggleTablePreview: () => void
  reportMarkdown: string
  run: RunSummary
  runBoundary?: string
  runComparison: TablePayload | null
  selectedCombinationId: string
  selectedTableSpecs: RunTableSpec[]
  selectedDownloadItems: string[]
  tablePreviewExpanded: boolean
  tablePreviewLoading: boolean
  tables: Record<string, TablePayload>
}) {
  const [viewMode, setViewMode] = useState<EvidenceViewMode>('workbench')
  const [deliveryTab, setDeliveryTab] = useState<DeliveryTab>('metrics')
  const activeDeliveryTab =
    deliveryTab === 'comparison' && runComparison === null ? 'metrics' : deliveryTab
  const selectedMetrics = useMemo(
    () =>
      selectedCombinationMetrics({
        chartTables,
        combinationSummaries,
        selectedCombinationId,
      }),
    [chartTables, combinationSummaries, selectedCombinationId],
  )
  const boundary = boundaryText(run, runBoundary)
  return (
    <div className="result-workbench">
      <RunHeader
        canDelete={canDelete}
        deleting={deleting}
        onDelete={onDelete}
        onViewModeChange={setViewMode}
        run={run}
        viewMode={viewMode}
      />

      <AnimatePresence mode="wait">
        {viewMode === 'workbench' ? (
          <motion.div
            animate={{ opacity: 1, y: 0 }}
            className="evidence-view"
            exit={{ opacity: 0, y: 8 }}
            initial={{ opacity: 0, y: 8 }}
            key="workbench"
            transition={{ duration: 0.18, ease: 'easeOut' }}
          >
            <MetricSummary run={run} selectedMetrics={selectedMetrics} />

            <CombinationSelector
              combinations={tables.combinations}
              onChange={onSelectedCombinationChange}
              selectedCombinationId={selectedCombinationId}
            />

            <RunCharts charts={displayCharts} mode="workbench" tables={chartTables} />

            <DeliveryTabs
              activePayload={activePayload}
              activeTable={activeTable}
              activeTableSpec={activeTableSpec}
              boundary={boundary}
              canShowComparison={runComparison !== null}
              deliveryTab={activeDeliveryTab}
              downloadError={downloadError}
              downloadNotice={downloadNotice}
              downloading={downloading}
              downloadOptions={downloadOptions}
              downloadPresets={downloadPresets}
              onActiveTableChange={onActiveTableChange}
              onDownload={onDownload}
              onDownloadPreset={onDownloadPreset}
              onSelectTab={setDeliveryTab}
              onToggleDownloadItem={onToggleDownloadItem}
              onToggleTablePreview={onToggleTablePreview}
              reportMarkdown={reportMarkdown}
              run={run}
              runComparison={runComparison}
              selectedDownloadItems={selectedDownloadItems}
              selectedMetrics={selectedMetrics}
              selectedTableSpecs={selectedTableSpecs}
              tablePreviewExpanded={tablePreviewExpanded}
              tablePreviewLoading={tablePreviewLoading}
            />
          </motion.div>
        ) : (
          <motion.div
            animate={{ opacity: 1, y: 0 }}
            className="evidence-view report-view-mode"
            exit={{ opacity: 0, y: 8 }}
            initial={{ opacity: 0, y: 8 }}
            key="report"
            transition={{ duration: 0.18, ease: 'easeOut' }}
          >
            <ReportOverview run={run} />

            <MetricSummary run={run} selectedMetrics={selectedMetrics} />

            <CombinationSelector
              combinations={tables.combinations}
              onChange={onSelectedCombinationChange}
              selectedCombinationId={selectedCombinationId}
            />

            <RunCharts charts={displayCharts} mode="report" tables={chartTables} />

            <MetricDetails run={run} selectedMetrics={selectedMetrics} />

            <section className="boundary">{boundary}</section>

            <ReportSection reportMarkdown={reportMarkdown} />

            <CsvBrowser
              activePayload={activePayload}
              activeTable={activeTable}
              activeTableSpec={activeTableSpec}
              expanded={tablePreviewExpanded}
              loading={tablePreviewLoading}
              onActiveTableChange={onActiveTableChange}
              onToggleExpanded={onToggleTablePreview}
              run={run}
              tableSpecs={selectedTableSpecs}
            />

            <RunComparison comparison={runComparison} />

            <ArtifactDownloads
              downloadError={downloadError}
              downloadNotice={downloadNotice}
              downloading={downloading}
              downloadOptions={downloadOptions}
              downloadPresets={downloadPresets}
              onDownload={onDownload}
              onPreset={onDownloadPreset}
              onToggleItem={onToggleDownloadItem}
              selectedItems={selectedDownloadItems}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function ReportOverview({ run }: { run: RunSummary }) {
  return (
    <section className="report-overview" aria-label="完整报告说明">
      <div>
        <span className="eyebrow">完整报告</span>
        <h3>完整报告</h3>
        <p>
          以阅读模式展开全部图表、指标、表格预览、运行对比和下载入口，用于系统检查这次
          运行的输入、组合表现和输出文件。
        </p>
      </div>
      <div className="report-overview-grid">
        <span>{run.kind}</span>
        <span>{run.domain}</span>
        <span>{run.data_stage}</span>
        {run.n_combinations ? <span>{run.n_combinations} 个组合</span> : null}
      </div>
    </section>
  )
}

function DeliveryTabs({
  activePayload,
  activeTable,
  activeTableSpec,
  boundary,
  canShowComparison,
  deliveryTab,
  downloadError,
  downloadNotice,
  downloading,
  downloadOptions,
  downloadPresets,
  onActiveTableChange,
  onDownload,
  onDownloadPreset,
  onSelectTab,
  onToggleDownloadItem,
  onToggleTablePreview,
  reportMarkdown,
  run,
  runComparison,
  selectedDownloadItems,
  selectedMetrics,
  selectedTableSpecs,
  tablePreviewExpanded,
  tablePreviewLoading,
}: {
  activePayload: TablePayload | undefined
  activeTable: string
  activeTableSpec: RunTableSpec | undefined
  boundary: string
  canShowComparison: boolean
  deliveryTab: DeliveryTab
  downloadError: string
  downloadNotice: string
  downloading: boolean
  downloadOptions: DownloadOption[]
  downloadPresets: RunDisplayDownloadPreset[]
  onActiveTableChange: (table: string) => void
  onDownload: () => void
  onDownloadPreset: (preset: string) => void
  onSelectTab: (tab: DeliveryTab) => void
  onToggleDownloadItem: (item: string) => void
  onToggleTablePreview: () => void
  reportMarkdown: string
  run: RunSummary
  runComparison: TablePayload | null
  selectedDownloadItems: string[]
  selectedMetrics: SelectedCombinationMetrics
  selectedTableSpecs: RunTableSpec[]
  tablePreviewExpanded: boolean
  tablePreviewLoading: boolean
}) {
  const tabs: { key: DeliveryTab; label: string; disabled?: boolean }[] = [
    { key: 'metrics', label: '指标详情' },
    { key: 'report', label: '研究报告' },
    { key: 'csv', label: 'CSV 明细' },
    { key: 'comparison', disabled: !canShowComparison, label: '运行对比' },
    { key: 'downloads', label: '结果文件下载' },
  ]
  return (
    <section aria-label="数据与交付" className="delivery-workbench">
      <div className="delivery-head">
        <div>
          <h3>数据与交付</h3>
          <p>可复查结果文件默认收起，按需切换。</p>
        </div>
        <span>可复现证据包</span>
      </div>
      <div className="delivery-tabs" role="tablist" aria-label="数据与交付 tabs">
        {tabs.map((tab) => (
          <button
            aria-selected={deliveryTab === tab.key}
            disabled={tab.disabled}
            key={tab.key}
            onClick={() => onSelectTab(tab.key)}
            role="tab"
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="delivery-content">
        {deliveryTab === 'metrics' ? (
          <>
            <MetricDetails run={run} selectedMetrics={selectedMetrics} />
            <section className="boundary">{boundary}</section>
          </>
        ) : null}
        {deliveryTab === 'report' ? (
          <ReportSection reportMarkdown={reportMarkdown} />
        ) : null}
        {deliveryTab === 'csv' ? (
          <CsvBrowser
            activePayload={activePayload}
            activeTable={activeTable}
            activeTableSpec={activeTableSpec}
            expanded={tablePreviewExpanded}
            loading={tablePreviewLoading}
            onActiveTableChange={onActiveTableChange}
            onToggleExpanded={onToggleTablePreview}
            run={run}
            tableSpecs={selectedTableSpecs}
          />
        ) : null}
        {deliveryTab === 'comparison' ? (
          <RunComparison comparison={runComparison} />
        ) : null}
        {deliveryTab === 'downloads' ? (
          <ArtifactDownloads
            downloadError={downloadError}
            downloadNotice={downloadNotice}
            downloading={downloading}
            downloadOptions={downloadOptions}
            downloadPresets={downloadPresets}
            onDownload={onDownload}
            onPreset={onDownloadPreset}
            onToggleItem={onToggleDownloadItem}
            selectedItems={selectedDownloadItems}
          />
        ) : null}
      </div>
    </section>
  )
}

function ReportSection({
  reportMarkdown,
}: {
  reportMarkdown: string
}) {
  return (
    <section className="data-section">
      <div className="section-title">
        <h3>研究报告</h3>
        <span>在结果文件下载中选择 report.md</span>
      </div>
      <ReportView markdown={reportMarkdown} />
    </section>
  )
}

function boundaryText(run: RunSummary, runBoundary?: string): string {
  if (run.data_stage === 'uploaded_panel') {
    return '本次运行使用用户上传面板在本地评估；平台不下载数据、不连接外部 API、不保存凭据，也不提供投资建议。'
  }
  if (run.uses_external_data || run.data_stage === 'approved_offline_real_data') {
    return '本次运行使用已批准的本地离线真实数据；平台不下载数据、不连接外部 API、不保存凭据，也不提供投资建议。'
  }
  return runBoundary ?? '本地合成示例结果，仅用于只读展示。'
}

function selectedCombinationMetrics({
  chartTables,
  combinationSummaries,
  selectedCombinationId,
}: {
  chartTables: Record<string, TablePayload>
  combinationSummaries: TablePayload | null
  selectedCombinationId: string
}): SelectedCombinationMetrics {
  const summaryRow = combinationSummaries?.rows.find(
    (row) => String(row.combination_id ?? '') === selectedCombinationId,
  )
  const pnlRows = chartTables.pnl?.rows ?? []
  const finalPnlRow = pnlRows[pnlRows.length - 1]
  return {
    finalReturn:
      numericValue(summaryRow?.final_cumulative_net_return) ??
      numericValue(finalPnlRow?.cumulative_pnl),
    icir: numericValue(summaryRow?.icir),
    maxDrawdown: numericValue(summaryRow?.max_drawdown),
    meanIc: numericValue(summaryRow?.mean_ic),
    meanRankic: numericValue(summaryRow?.mean_rankic),
    meanTurnover: numericValue(summaryRow?.mean_turnover),
    rankicir: numericValue(summaryRow?.rankicir),
    sharpeLike: numericValue(summaryRow?.sharpe_like),
    sortinoLike: numericValue(summaryRow?.sortino_like),
    totalCost: numericValue(summaryRow?.total_cost),
    winRate: numericValue(summaryRow?.win_rate),
  }
}

function numericValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
