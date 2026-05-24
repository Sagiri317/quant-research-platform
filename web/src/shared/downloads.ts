import type { RunDisplayDownloadPreset, RunDisplayDownloads } from './api/api'

export type DownloadOption = {
  label: string
  item: string
}

export function buildDownloadOptions(displayDownloads?: RunDisplayDownloads): DownloadOption[] {
  if (!displayDownloads) {
    return []
  }
  return displayDownloads.items.map((option) => ({
    label: option.label,
    item: option.item,
  }))
}

export function defaultDownloadItems(
  displayDownloads?: RunDisplayDownloads,
): string[] {
  if (!displayDownloads) {
    return []
  }
  return (
    displayDownloads.presets.find((preset) => preset.key === 'report')?.items ??
    displayDownloads.presets[0]?.items ??
    []
  )
}

export function presetDownloadItems(
  displayDownloads: RunDisplayDownloads | undefined,
  presetKey: string,
): string[] {
  return displayDownloads?.presets.find((preset) => preset.key === presetKey)?.items ?? []
}

export function buildDownloadPresets(
  displayDownloads?: RunDisplayDownloads,
): RunDisplayDownloadPreset[] {
  return displayDownloads?.presets ?? []
}

export function triggerBrowserDownloadUrl(url: string) {
  const link = document.createElement('a')
  link.href = url
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  link.remove()
}
