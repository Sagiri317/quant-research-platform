import type { RunDisplayDownloads, RunTableSpec } from './api'

export type DownloadOption = {
  label: string
  item: string
}

export const DEFAULT_DOWNLOAD_ITEMS = [
  'report.md',
  'comparison.csv',
]

export function buildDownloadOptions(
  tableSpecs: RunTableSpec[],
  displayDownloads?: RunDisplayDownloads,
): DownloadOption[] {
  if (displayDownloads) {
    return displayDownloads.items.map((option) => ({
      label: option.label,
      item: option.item,
    }))
  }
  return [
    { label: '研究报告 report.md', item: 'report.md' },
    { label: '运行对比 comparison.csv', item: 'comparison.csv' },
    { label: '清单 manifest.json', item: 'manifest.json' },
    { label: '配置 config.json', item: 'config.json' },
    { label: '摘要 summary.json', item: 'summary.json' },
    ...tableSpecs.map((spec) => ({
      label: `${spec.label} CSV`,
      item: spec.downloadItem,
    })),
  ]
}

export function defaultDownloadItems(
  displayDownloads?: RunDisplayDownloads,
): string[] {
  if (!displayDownloads) {
    return DEFAULT_DOWNLOAD_ITEMS
  }
  const availableItems = new Set(displayDownloads.items.map((option) => option.item))
  return DEFAULT_DOWNLOAD_ITEMS.filter((item) => availableItems.has(item))
}

export function presetDownloadItems(
  displayDownloads: RunDisplayDownloads | undefined,
  presetKey: 'all' | 'report' | 'tables' | 'repro',
): string[] | undefined {
  return displayDownloads?.presets.find((preset) => preset.key === presetKey)?.items
}

export function triggerBrowserDownloadUrl(url: string) {
  const link = document.createElement('a')
  link.href = url
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  link.remove()
}
