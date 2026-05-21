export function jsonResponse(value: unknown): Promise<Response> {
  return Promise.resolve({
    ok: true,
    json: async () => value,
  } as Response)
}

export function tableResponse(
  rows: Record<string, unknown>[],
): Promise<Response> {
  return jsonResponse(tablePayload(rows))
}

export function tablePayload(rows: Record<string, unknown>[]) {
  return {
    columns: rows.length === 0 ? [] : Object.keys(rows[0]),
    rows,
  }
}

export function evidencePayload({
  combinationSummaryRows,
  comparisonRows = [
    {
      public_run_id: 'core/demo',
      mean_ic: 0.53,
      mean_rankic: 0.51,
      mean_final_return: 1.6,
      sharpe_like: 1.2,
      max_drawdown: -0.12,
    },
  ],
  reportMarkdown = '# 报告\n\n合成样本证据。',
  summary = {
    run_id: 'demo',
    project: 'quant-research-platform',
    kind: 'factor',
    domain: 'core_synthetic',
    data_stage: 'synthetic_only',
    uses_external_data: false,
    n_rows_evaluated: 600,
    weight_config: { method: 'equal_weight_quantile' },
    boundary: '仅使用 Synthetic/toy 数据。',
  },
  tables,
  display,
}: {
  combinationSummaryRows?: Record<string, unknown>[]
  comparisonRows?: Record<string, unknown>[]
  display?: Record<string, unknown>
  reportMarkdown?: string
  summary?: Record<string, unknown>
  tables?: Record<string, ReturnType<typeof tablePayload>>
} = {}) {
  const evidenceTables = tables ?? {
    combinations: tablePayload([
      {
        combination_id: 'factor__forward_return__w0__c0',
        feature: 'factor',
        label: 'forward_return',
        weight_id: 'w0',
        cost_id: 'c0',
      },
    ]),
    ic: tablePayload(
      Array.from({ length: 30 }, (_, index) => ({
        combination_id: 'factor__forward_return__w0__c0',
        timestamp: `2026-01-${String(index + 1).padStart(2, '0')}`,
        ic: 0.4,
        rankic: 0.5,
      })),
    ),
    returns: tablePayload([
      {
        combination_id: 'factor__forward_return__w0__c0',
        timestamp: '2026-01-01',
        net_portfolio_return: 0.02,
        turnover: 1.1,
        cost: 0.001,
      },
    ]),
    pnl: tablePayload([
      {
        combination_id: 'factor__forward_return__w0__c0',
        timestamp: '2026-01-01',
        net_portfolio_return: 0.02,
        cumulative_pnl: 0.02,
        drawdown: 0,
      },
    ]),
  }
  return {
    summary,
    report: { markdown: reportMarkdown },
    comparison: tablePayload(comparisonRows),
    combination_summaries: tablePayload(combinationSummaryRows ?? []),
    display: display ?? displayPayload(evidenceTables, { includeDefaultTables: true }),
    tables: evidenceTables,
  }
}

export function displayPayload(
  tables: Record<string, ReturnType<typeof tablePayload>>,
  { includeDefaultTables = false }: { includeDefaultTables?: boolean } = {},
) {
  const tableKeys = includeDefaultTables
    ? Array.from(new Set([...DEFAULT_TABLE_KEYS, ...Object.keys(tables)]))
    : Object.keys(tables)
  const tableEntries = tableKeys.map((key) => ({
    download_item: tableDownloadItem(key),
    group: tableGroup(key),
    key,
    label: tableLabel(key),
    path: tableDownloadItem(key),
    supports_combination_filter: key !== 'combinations',
  }))
  const tableItems = tableEntries.map((table) => table.download_item)
  return {
    schema_version: '0.5',
    tables: tableEntries,
    charts: [
      {
        description: '净值曲线和回撤',
        id: 'pnl',
        include_zero: true,
        kind: 'line',
        series: [
          {
            color: '#2563eb',
            format: 'percent',
            key: 'cumulative_pnl',
            label: 'cumulative_pnl',
          },
          {
            color: '#dc2626',
            format: 'percent',
            key: 'drawdown',
            label: 'drawdown',
          },
        ],
        table: 'pnl',
        title: '累计组合 PnL',
        x: 'timestamp',
      },
      {
        description: 'IC 和 RankIC 的时间序列',
        id: 'ic_rankic',
        kind: 'line',
        series: [
          { color: '#2563eb', key: 'ic', label: 'ic' },
          { color: '#059669', key: 'rankic', label: 'rankic' },
        ],
        table: 'ic',
        title: 'IC / RankIC',
        x: 'timestamp',
      },
      {
        description: '单期收益、换手和成本',
        id: 'return_turnover_cost',
        include_zero: true,
        kind: 'bar',
        series: [
          {
            color: '#2563eb',
            key: 'net_portfolio_return',
            label: 'net_portfolio_return',
          },
          { color: '#f59e0b', key: 'turnover', label: 'turnover' },
          { color: '#dc2626', key: 'cost', label: 'cost' },
        ],
        table: 'returns',
        title: '收益 / 换手 / 成本',
        x: 'timestamp',
      },
    ],
    downloads: {
      items: [
        { item: 'report.md', kind: 'report', label: '研究报告 report.md' },
        { item: 'comparison.csv', kind: 'index', label: '运行对比 comparison.csv' },
        { item: 'manifest.json', kind: 'json', label: '清单 manifest.json' },
        { item: 'config.json', kind: 'json', label: '配置 config.json' },
        { item: 'summary.json', kind: 'json', label: '摘要 summary.json' },
        ...tableEntries.map((table) => ({
          item: table.download_item,
          kind: 'table',
          label: `${table.label} CSV`,
          table: table.key,
        })),
      ],
      presets: [
        {
          items: [
            'report.md',
            'comparison.csv',
            'manifest.json',
            'config.json',
            'summary.json',
            ...tableItems,
          ],
          key: 'all',
          label: '全选',
        },
        { items: ['report.md'], key: 'report', label: '仅报告' },
        {
          items: ['comparison.csv', ...tableItems],
          key: 'tables',
          label: '仅表格',
        },
        {
          items: [
            'report.md',
            'comparison.csv',
            'manifest.json',
            'config.json',
            'summary.json',
            ...tableItems,
          ],
          key: 'repro',
          label: '复现包',
        },
      ],
    },
  }
}

const DEFAULT_TABLE_KEYS = [
  'combinations',
  'ic',
  'ic_summary',
  'grouped_returns',
  'group_spread_summary',
  'performance_summary',
  'exposure_summary',
  'positions',
  'turnover',
  'returns',
  'pnl',
  'cost_breakdown',
  'weight_diagnostics',
  'neutralization_diagnostics',
]

function tableDownloadItem(key: string) {
  if (
    [
      'cost_breakdown',
      'exposure_summary',
      'pnl',
      'positions',
      'returns',
      'turnover',
    ].includes(key)
  ) {
    return `portfolio/${key}.csv`
  }
  if (key === 'weight_diagnostics' || key === 'neutralization_diagnostics') {
    return `diagnostics/${key}.csv`
  }
  if (key === 'combinations') {
    return 'combinations.csv'
  }
  return `metrics/${key}.csv`
}

function tableGroup(key: string) {
  if (
    [
      'cost_breakdown',
      'exposure_summary',
      'pnl',
      'positions',
      'returns',
      'turnover',
    ].includes(key)
  ) {
    return 'portfolio'
  }
  if (key === 'weight_diagnostics' || key === 'neutralization_diagnostics') {
    return 'diagnostics'
  }
  if (key === 'combinations') {
    return 'setup'
  }
  return 'metrics'
}

function tableLabel(key: string) {
  return (
    {
      combinations: '组合配置',
      custom_alpha: '自定义 Alpha',
      cost_breakdown: '成本拆分',
      exposure_summary: '敞口汇总',
      group_spread_summary: '分组价差汇总',
      grouped_returns: '分组收益',
      ic: 'IC / RankIC',
      ic_summary: 'IC / RankIC 汇总',
      neutralization_diagnostics: '中性化诊断',
      performance_summary: '绩效汇总',
      pnl: 'PnL 曲线',
      positions: '持仓明细',
      returns: '组合收益明细',
      turnover: '换手明细',
      weight_diagnostics: '权重诊断',
    }[key] ?? key
  )
}
