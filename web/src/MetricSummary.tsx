import type { RunSummary } from './api'
import { MetricCard } from './components'
import { interpretRun } from './insights'
import type { ReactNode } from 'react'

export type SelectedCombinationMetrics = {
  finalReturn?: number
  icir?: number
  maxDrawdown?: number
  meanIc?: number
  meanRankic?: number
  meanTurnover?: number
  rankicir?: number
  sharpeLike?: number
  sortinoLike?: number
  totalCost?: number
  winRate?: number
}

export function MetricSummary({
  run,
  selectedMetrics,
}: {
  run: RunSummary
  selectedMetrics?: SelectedCombinationMetrics
}) {
  return (
    <>
      <section className="metric-summary" aria-label="核心摘要">
        <div className="section-title compact-title">
          <h3>核心摘要</h3>
          <span>关键指标</span>
        </div>
        <div className="metrics-grid summary-grid">
          <MetricCard label="样本行数" value={run.n_rows_evaluated} />
          <MetricCard
            label="当前组合最终收益"
            value={selectedMetrics?.finalReturn}
          />
          <MetricCard
            label="类 Sharpe"
            value={selectedMetrics?.sharpeLike ?? run.sharpe_like}
          />
          <MetricCard
            label="最大回撤"
            value={selectedMetrics?.maxDrawdown ?? run.max_drawdown}
          />
          <MetricCard
            label="平均 IC"
            value={selectedMetrics?.meanIc ?? run.mean_ic}
          />
          <MetricCard
            label="平均 RankIC"
            value={selectedMetrics?.meanRankic ?? run.mean_rankic}
          />
        </div>
      </section>

      <section className="data-section interpretation-section">
        <div className="section-title">
          <h3>结果解读</h3>
          <span>只读说明</span>
        </div>
        <ul className="interpretation-list">
          {interpretRun(run, selectedMetrics).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

    </>
  )
}

export function MetricDetails({
  run,
  selectedMetrics,
}: {
  run: RunSummary
  selectedMetrics?: SelectedCombinationMetrics
}) {
  return (
    <section className="metric-detail-groups" aria-label="指标详情">
      <div className="section-title compact-title">
        <h3>指标详情</h3>
        <span>按主题展开</span>
      </div>
      <MetricDetailGroup title="收益风险">
        <MetricCard label="样本行数" value={run.n_rows_evaluated} />
        <MetricCard
          label="当前组合最终收益"
          value={selectedMetrics?.finalReturn}
        />
        <MetricCard
            label="类 Sharpe"
          value={selectedMetrics?.sharpeLike ?? run.sharpe_like}
        />
        <MetricCard
            label="类 Sortino"
          value={selectedMetrics?.sortinoLike ?? run.sortino_like}
        />
        <MetricCard
          label="最大回撤"
          value={selectedMetrics?.maxDrawdown ?? run.max_drawdown}
        />
        <MetricCard label="胜率" value={selectedMetrics?.winRate ?? run.win_rate} />
      </MetricDetailGroup>
      <MetricDetailGroup title="IC / RankIC">
        <MetricCard label="平均 IC" value={selectedMetrics?.meanIc ?? run.mean_ic} />
        <MetricCard label="ICIR" value={selectedMetrics?.icir ?? run.icir} />
        <MetricCard
          label="平均 RankIC"
          value={selectedMetrics?.meanRankic ?? run.mean_rankic}
        />
        <MetricCard
          label="RankICIR"
          value={selectedMetrics?.rankicir ?? run.rankicir}
        />
      </MetricDetailGroup>
      <MetricDetailGroup title="成本换手">
        <MetricCard
          label="平均换手"
          value={selectedMetrics?.meanTurnover ?? run.mean_turnover}
        />
        <MetricCard label="总成本" value={selectedMetrics?.totalCost ?? run.total_cost} />
      </MetricDetailGroup>
    </section>
  )
}

function MetricDetailGroup({
  children,
  title,
}: {
  children: ReactNode
  title: string
}) {
  return (
    <details className="metric-details" open>
      <summary>{title}</summary>
      <div className="metrics-grid">{children}</div>
    </details>
  )
}
