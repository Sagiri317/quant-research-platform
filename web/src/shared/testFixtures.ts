import type { RunDisplay } from './api/api'

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
  reportMarkdown = '# Report\n\nSynthetic evidence.',
  summary = {
    run_id: 'demo',
    project: 'quant-research-platform',
    kind: 'factor',
    domain: 'core_synthetic',
    data_stage: 'synthetic_only',
    uses_external_data: false,
    n_rows_evaluated: 600,
    weight_config: { method: 'equal_weight_quantile' },
    boundary: 'Synthetic/toy adapter only.',
  },
  tables,
  display,
}: {
  combinationSummaryRows?: Record<string, unknown>[]
  comparisonRows?: Record<string, unknown>[]
  display?: Record<string, unknown>
  reportMarkdown?: string | null
  summary?: Record<string, unknown>
  tables?: Record<string, ReturnType<typeof tablePayload>>
} = {}) {
  const sourceAlias = typeof summary.source_alias === 'string' ? summary.source_alias : 'core'
  const runId = typeof summary.run_id === 'string' ? summary.run_id : 'demo'
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
        net_return: 0.02,
        turnover: 1.1,
        cost: 0.001,
      },
    ]),
  }
  return {
    summary,
    report: reportMarkdown == null ? null : { markdown: reportMarkdown },
    comparison: tablePayload(comparisonRows),
    combination_summaries: tablePayload(combinationSummaryRows ?? []),
    display:
      display ??
      displayPayload(evidenceTables, {
        includeDefaultTables: true,
        runId,
        sourceAlias,
      }),
    tables: evidenceTables,
  }
}

export function displayPayload(
  tables: Record<string, ReturnType<typeof tablePayload>>,
  {
    includeDefaultTables = false,
    runId = 'demo',
    sourceAlias = 'core',
  }: { includeDefaultTables?: boolean; runId?: string; sourceAlias?: string } = {},
): RunDisplay {
  const tableKeys = includeDefaultTables
    ? Array.from(new Set([...DEFAULT_TABLE_KEYS, ...Object.keys(tables)]))
    : Object.keys(tables)
  const tableEntries = tableKeys.map((key) => ({
    download_ref: { kind: 'table', key },
    group: tableGroup(key),
    key,
    label: tableLabel(key),
    path: tableDownloadItem(key),
    format: key === 'weights' ? 'parquet' : 'csv',
    preview: tablePreview(key),
    preview_url: `/api/runs/${sourceAlias}/${runId}/tables/${key}/preview`,
    supports_combination_filter: tableSupportsCombinationFilter(key),
  }))
  const tableRefs = tableEntries.map((table) => ({ kind: 'table', key: table.key }))
  return {
    schema_version: '0.7',
    tables: tableEntries,
    charts: [
      {
        description: 'Derived from returns.net_return by the display chart contract.',
        id: 'primary_cumulative_return',
        input_series: 'net_return',
        kind: 'derived_cumulative_return',
        output_series: 'cumulative_net_return',
        section: 'primary',
        source_table_key: 'returns',
        title: '累计组合收益',
        transform: 'compound_cumulative_return',
        x: 'timestamp',
      },
      {
        description: 'IC and RankIC over time.',
        id: 'ic_rankic',
        kind: 'line',
        section: 'primary',
        series: [
          { color: '#2563eb', key: 'ic', label: 'ic' },
          { color: '#059669', key: 'rankic', label: 'rankic' },
        ],
        source_table_key: 'ic',
        title: 'IC / RankIC',
        x: 'timestamp',
      },
      {
        description: 'Single-period return.',
        id: 'net_return',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [
          {
            color: '#2563eb',
            key: 'net_return',
            label: 'net_return',
          },
        ],
        source_table_key: 'returns',
        title: '收益 / 换手 / 成本',
        x: 'timestamp',
      },
      {
        description: 'Turnover by period.',
        id: 'turnover',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [{ color: '#f59e0b', key: 'turnover', label: 'turnover' }],
        source_table_key: 'returns',
        title: '换手率',
        x: 'timestamp',
      },
      {
        description: 'Cost by period.',
        id: 'cost',
        include_zero: true,
        kind: 'bar',
        section: 'diagnostic',
        series: [{ color: '#dc2626', key: 'cost', label: 'cost' }],
        source_table_key: 'returns',
        title: '单期成本',
        x: 'timestamp',
      },
    ],
    downloads: {
      items: [
        { kind: 'metadata_file', key: 'report', label: '研究报告 report.md' },
        { kind: 'dynamic', key: 'comparison', label: '运行对比 comparison.csv' },
        { kind: 'metadata_file', key: 'manifest', label: '清单 manifest.json' },
        {
          kind: 'metadata_file',
          key: 'run_context',
          label: '运行上下文 run_context.json',
        },
        { kind: 'metadata_file', key: 'summary', label: '摘要 summary.json' },
        ...tableEntries.map((table) => ({
          kind: 'table',
          key: table.key,
          label: `${table.label} CSV`,
          table: table.key,
        })),
      ],
      presets: [
        {
          items: [
            { kind: 'metadata_file', key: 'report' },
            { kind: 'metadata_file', key: 'manifest' },
            { kind: 'metadata_file', key: 'run_context' },
            { kind: 'metadata_file', key: 'summary' },
            ...tableRefs,
          ],
          key: 'all',
          label: '全选',
        },
        {
          items: [{ kind: 'metadata_file', key: 'report' }],
          key: 'report',
          label: '仅报告',
        },
        {
          items: [{ kind: 'dynamic', key: 'comparison' }, ...tableRefs],
          key: 'tables',
          label: '仅表格',
        },
        {
          items: [
            { kind: 'metadata_file', key: 'manifest' },
            { kind: 'metadata_file', key: 'summary' },
            { kind: 'metadata_file', key: 'run_context' },
            ...tableRefs,
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
  'grouped_returns',
  'evaluation_metrics_summary',
  'returns',
  'weights',
  'coverage_summary',
  'diagnostic_policies',
  'diagnostic_summary',
  'diagnostic_metrics',
  'portfolio_assumptions',
  'tradability_diagnostics',
  'tradability_summary',
  'scenario_metrics',
  'exposure_capacity_summary',
  'portfolio_risk_summary',
  'diagnostic_observations',
]

function tableDownloadItem(key: string) {
  if (key === 'weights') {
    return 'portfolio/weights.parquet'
  }
  if (key === 'returns') {
    return 'portfolio/returns.csv'
  }
  if (['coverage_summary'].includes(key)) {
    return `input_quality/${key}.csv`
  }
  if (key === 'combinations') {
    return 'dimensions/combinations.csv'
  }
  if (key.endsWith('_diagnostics') || DIAGNOSTIC_TABLE_KEYS.has(key)) {
    return `diagnostics/${key}.csv`
  }
  return `metrics/${key}.csv`
}

function tableGroup(key: string) {
  if (['returns', 'weights'].includes(key)) {
    return 'portfolio'
  }
  if (key === 'coverage_summary') {
    return 'input_quality'
  }
  if (['combinations', 'diagnostic_policies', 'portfolio_assumptions'].includes(key)) {
    return 'dimensions'
  }
  if (key.endsWith('_diagnostics') || DIAGNOSTIC_TABLE_KEYS.has(key)) {
    return 'diagnostics'
  }
  return 'metrics'
}

function tableLabel(key: string) {
  return (
    {
      combinations: '组合配置',
      coverage_summary: '覆盖汇总',
      custom_alpha: 'Custom alpha',
      diagnostic_metrics: '诊断指标',
      diagnostic_observations: '诊断观察',
      diagnostic_policies: '诊断策略',
      diagnostic_summary: '诊断总览',
      evaluation_metrics_summary: '评价指标汇总',
      exposure_capacity_summary: '暴露和容量代理',
      grouped_returns: '分组收益',
      ic: 'IC / RankIC',
      portfolio_assumptions: '组合假设',
      portfolio_risk_summary: '组合风险代理',
      returns: '组合收益明细',
      scenario_metrics: '场景指标',
      tradability_diagnostics: '可交易诊断',
      tradability_summary: '可交易汇总',
      weights: '目标权重',
    }[key] ?? key
  )
}

function tableSupportsCombinationFilter(key: string) {
  return ![
    'combinations',
    'coverage_summary',
    'diagnostic_policies',
    'portfolio_assumptions',
  ].includes(key)
}

function tablePreview(key: string) {
  const supportsCombinationFilter = tableSupportsCombinationFilter(key)
  return {
    kind: 'table',
    table_key: key,
    default_limit: 20,
    max_limit: 1000,
    supports_offset: true,
    supports_combination_filter: supportsCombinationFilter,
    filter_parameter: supportsCombinationFilter ? 'combination_id' : null,
  }
}

const DIAGNOSTIC_TABLE_KEYS = new Set([
  'diagnostic_summary',
  'diagnostic_metrics',
  'tradability_summary',
  'scenario_metrics',
  'exposure_capacity_summary',
  'portfolio_risk_summary',
  'diagnostic_observations',
])
