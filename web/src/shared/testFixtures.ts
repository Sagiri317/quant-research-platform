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
        combination_id: 'factor__forward_return__dataframe__long_only',
        feature: 'factor',
        label: 'forward_return',
        kernel: 'dataframe',
        portfolio_mode: 'long_only',
      },
    ]),
    ic: tablePayload(
      Array.from({ length: 30 }, (_, index) => ({
        combination_id: 'factor__forward_return__dataframe__long_only',
        timestamp: `2026-01-${String(index + 1).padStart(2, '0')}`,
        ic: 0.4,
        rankic: 0.5,
      })),
    ),
    returns: tablePayload([
      {
        combination_id: 'factor__forward_return__dataframe__long_only',
        timestamp: '2026-01-01',
        net_portfolio_return: 0.02,
        turnover: 1.1,
        cost: 0.001,
      },
    ]),
    cumulative_returns: tablePayload([
      {
        combination_id: 'factor__forward_return__dataframe__long_only',
        timestamp: '2026-01-01',
        net_portfolio_return: 0.02,
        cumulative_net_return: 0.02,
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
    format: key === 'weights' ? 'parquet' : 'csv',
    supports_combination_filter: tableSupportsCombinationFilter(key),
  }))
  const tableItems = tableEntries.map((table) => table.download_item)
  return {
    schema_version: '0.6',
    tables: tableEntries,
    charts: [
      {
        description: '净值曲线和回撤',
        id: 'cumulative_returns',
        include_zero: true,
        kind: 'line',
        section: 'primary',
        series: [
          {
            color: '#2563eb',
            format: 'percent',
            key: 'cumulative_net_return',
            label: 'cumulative_net_return',
          },
          {
            color: '#dc2626',
            format: 'percent',
            key: 'drawdown',
            label: 'drawdown',
          },
        ],
        table: 'cumulative_returns',
        title: '累计组合收益',
        x: 'timestamp',
      },
      {
        description: 'IC 和 RankIC 的时间序列',
        id: 'ic_rankic',
        kind: 'line',
        section: 'primary',
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
        id: 'net_portfolio_return',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [
          {
            color: '#2563eb',
            key: 'net_portfolio_return',
            label: 'net_portfolio_return',
          },
        ],
        table: 'returns',
        title: '收益 / 换手 / 成本',
        x: 'timestamp',
      },
      {
        description: '每期换手强度',
        id: 'turnover',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [
          { color: '#f59e0b', key: 'turnover', label: 'turnover' },
        ],
        table: 'returns',
        title: '换手率',
        x: 'timestamp',
      },
      {
        description: '每期成本拖累',
        id: 'cost',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [
          { color: '#dc2626', key: 'cost', label: 'cost' },
        ],
        table: 'returns',
        title: '单期成本',
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
  'returns',
  'cumulative_returns',
  'weights',
  'tradability_diagnostics',
  'label_availability',
  'factor_coverage',
  'tradability_coverage',
  'evaluation_summary',
]

function tableDownloadItem(key: string) {
  if (key === 'weights') {
    return 'portfolio/weights.parquet'
  }
  if (['cumulative_returns', 'returns'].includes(key)) {
    return `portfolio/${key}.csv`
  }
  if (key === 'tradability_diagnostics') {
    return `diagnostics/${key}.csv`
  }
  if (
    ['label_availability', 'factor_coverage', 'tradability_coverage'].includes(key)
  ) {
    return `input_quality/${key}.csv`
  }
  if (key === 'evaluation_summary') {
    return 'summaries/evaluation_summary.csv'
  }
  if (key === 'combinations') {
    return 'dimensions/combinations.csv'
  }
  return `metrics/${key}.csv`
}

function tableGroup(key: string) {
  if (['cumulative_returns', 'returns', 'weights'].includes(key)) {
    return 'portfolio'
  }
  if (key === 'tradability_diagnostics') {
    return 'diagnostics'
  }
  if (
    ['label_availability', 'factor_coverage', 'tradability_coverage'].includes(key)
  ) {
    return 'input_quality'
  }
  if (key === 'evaluation_summary') {
    return 'summaries'
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
      evaluation_summary: '评价组合汇总',
      factor_coverage: '因子覆盖',
      group_spread_summary: '分组价差汇总',
      grouped_returns: '分组收益',
      ic: 'IC / RankIC',
      ic_summary: 'IC / RankIC 汇总',
      label_availability: '标签可用性',
      performance_summary: '绩效汇总',
      cumulative_returns: '累计收益曲线',
      returns: '组合收益明细',
      tradability_coverage: '可交易覆盖',
      tradability_diagnostics: '可交易诊断',
      weights: '目标权重',
    }[key] ?? key
  )
}

function tableSupportsCombinationFilter(key: string) {
  return ![
    'combinations',
    'label_availability',
    'factor_coverage',
    'tradability_coverage',
  ].includes(key)
}
