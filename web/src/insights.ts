import type { RunSummary } from './api'
import { formatNumber } from './format'
import type { SelectedCombinationMetrics } from './MetricSummary'

export function interpretRun(
  run: RunSummary,
  selectedMetrics?: SelectedCombinationMetrics,
): string[] {
  const rankic = selectedMetrics?.meanRankic ?? run.mean_rankic ?? Number.NaN
  const finalReturn = selectedMetrics?.finalReturn
  const cost = selectedMetrics?.totalCost ?? run.total_cost
  return [
    Number.isFinite(rankic) && rankic > 0
      ? `平均 RankIC 为 ${formatNumber(rankic)}，当前样本内排序信号为正。`
      : '平均 RankIC 未显示稳定正向，需要检查因子方向或样本质量。',
    `当前组合最终收益为 ${formatNumber(finalReturn)}，回撤和收益曲线需要一起看。`,
    `总 cost 为 ${formatNumber(cost)}，建议结合 turnover 判断成本拖累。`,
    boundaryInsight(run),
  ]
}

function boundaryInsight(run: RunSummary): string {
  if (run.data_stage === 'uploaded_panel') {
    return '结果来自用户上传面板的本地评估；平台不下载数据、不连接 API、不提供投资建议。'
  }
  if (run.uses_external_data || run.data_stage === 'approved_offline_real_data') {
    return '结果来自已批准的本地离线数据；平台不下载数据、不连接 API、不提供投资建议。'
  }
  return '结果来自 synthetic/toy 数据；平台不下载数据、不连接 API、不提供投资建议。'
}
