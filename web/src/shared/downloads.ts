import {
  displayRefKey,
  type RunDisplayRef,
  type RunDisplayDownloadPreset,
  type RunDisplayDownloads,
} from './api/api'

export type DownloadOption = {
  label: string
  item: string
}

export function buildDownloadOptions(displayDownloads?: RunDisplayDownloads): DownloadOption[] {
  if (!displayDownloads) {
    return []
  }
  return displayDownloads.items.flatMap((option) => {
    if (!option.key) {
      return []
    }
    return [
      {
        label: option.label,
        item: displayRefKey({ kind: option.kind, key: option.key }),
      },
    ]
  })
}

export function defaultDownloadItems(
  displayDownloads?: RunDisplayDownloads,
): string[] {
  if (!displayDownloads) {
    return []
  }
  return (
    compactPresetItems(
      displayDownloads.presets.find((preset) => preset.key === 'report')?.items ??
        displayDownloads.presets[0]?.items,
    ) ??
    []
  )
}

export function presetDownloadItems(
  displayDownloads: RunDisplayDownloads | undefined,
  presetKey: string,
): string[] {
  return compactPresetItems(
    displayDownloads?.presets.find((preset) => preset.key === presetKey)?.items,
  ) ?? []
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

function compactPresetItems(
  items: RunDisplayDownloadPreset['items'] | undefined,
): string[] | undefined {
  return items?.filter(isDisplayRef).map(displayRefKey)
}

function isDisplayRef(item: string | RunDisplayRef): item is RunDisplayRef {
  return (
    typeof item !== 'string' &&
    typeof item.kind === 'string' &&
    item.kind.length > 0 &&
    typeof item.key === 'string' &&
    item.key.length > 0
  )
}
