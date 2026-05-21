import { useState } from 'react'
import {
  type RunSummary,
  type RunDisplayDownloads,
  type RunTableSpec,
  selectedArtifactsDownloadUrl,
} from './api'
import {
  buildDownloadOptions,
  defaultDownloadItems,
  presetDownloadItems,
  triggerBrowserDownloadUrl,
} from './downloads'
import { messageFrom } from './format'

type DownloadPreset = 'all' | 'report' | 'tables' | 'repro'
type DownloadSelectionByKey = Record<string, string[]>

export function useArtifactDownloads({
  displayDownloads,
  selectedRun,
  selectedTableSpecs,
}: {
  displayDownloads?: RunDisplayDownloads
  selectedRun: RunSummary | null
  selectedTableSpecs: RunTableSpec[]
}) {
  const [selectedDownloadItemsByKey, setSelectedDownloadItemsByKey] =
    useState<DownloadSelectionByKey>({})
  const [downloadError, setDownloadError] = useState('')
  const [downloadNotice, setDownloadNotice] = useState('')
  const [downloading, setDownloading] = useState(false)
  const downloadOptions = buildDownloadOptions(selectedTableSpecs, displayDownloads)
  const selectionKey = selectedRun
    ? `${selectedRun.public_run_id}:${downloadOptions.map((option) => option.item).join('|')}`
    : ''
  const selectedDownloadItems =
    selectedDownloadItemsByKey[selectionKey] ?? defaultDownloadItems(displayDownloads)

  function setSelectedDownloadItems(items: string[]) {
    setSelectedDownloadItemsByKey((previous) => ({
      ...previous,
      [selectionKey]: items,
    }))
  }

  function applyDownloadPreset(preset: DownloadPreset) {
    setDownloadNotice('')
    const manifestItems = presetDownloadItems(displayDownloads, preset)
    if (manifestItems) {
      setSelectedDownloadItems(manifestItems)
    } else if (preset === 'all') {
      setSelectedDownloadItems(downloadOptions.map((option) => option.item))
    } else if (preset === 'report') {
      setSelectedDownloadItems(['report.md'])
    } else if (preset === 'tables') {
      setSelectedDownloadItems([
        'comparison.csv',
        ...selectedTableSpecs.map((spec) => spec.downloadItem),
      ])
    } else {
      setSelectedDownloadItems([
        'report.md',
        'comparison.csv',
        'manifest.json',
        'config.json',
        'summary.json',
        ...selectedTableSpecs.map((spec) => spec.downloadItem),
      ])
    }
  }

  function toggleDownloadItem(item: string) {
    setDownloadError('')
    setDownloadNotice('')
    setSelectedDownloadItems(
      selectedDownloadItems.includes(item)
        ? selectedDownloadItems.filter((selected) => selected !== item)
        : [...selectedDownloadItems, item],
    )
  }

  function handleSelectedDownload() {
    if (!selectedRun) {
      return
    }
    if (selectedDownloadItems.length === 0) {
      setDownloadError('请选择至少一个下载项。')
      return
    }
    try {
      setDownloading(true)
      setDownloadError('')
      setDownloadNotice('')
      triggerBrowserDownloadUrl(
        selectedArtifactsDownloadUrl(selectedRun, selectedDownloadItems),
      )
      setDownloadNotice(
        '浏览器已接管下载。大文件的打包、保存和失败提示以浏览器下载栏为准。',
      )
    } catch (caught) {
      setDownloadError(messageFrom(caught))
      setDownloadNotice('')
    } finally {
      setDownloading(false)
    }
  }

  return {
    applyDownloadPreset,
    downloadError,
    downloadNotice,
    downloadOptions,
    downloading,
    handleSelectedDownload,
    selectedDownloadItems,
    toggleDownloadItem,
  }
}
