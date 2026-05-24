import { useState } from 'react'
import {
  type RunSummary,
  type RunDisplayDownloads,
  selectedArtifactsDownloadUrl,
} from '../../shared/api/api'
import {
  buildDownloadPresets,
  buildDownloadOptions,
  defaultDownloadItems,
  presetDownloadItems,
  triggerBrowserDownloadUrl,
} from '../../shared/downloads'
import { messageFrom } from '../../shared/format'

type DownloadSelectionByKey = Record<string, string[]>
type RunMessageState = {
  key: string
  message: string
}

export function useArtifactDownloads({
  displayDownloads,
  selectedRun,
}: {
  displayDownloads?: RunDisplayDownloads
  selectedRun: RunSummary | null
}) {
  const [selectedDownloadItemsByKey, setSelectedDownloadItemsByKey] =
    useState<DownloadSelectionByKey>({})
  const [downloadErrorState, setDownloadErrorState] = useState<RunMessageState>({
    key: '',
    message: '',
  })
  const [downloadNoticeState, setDownloadNoticeState] = useState<RunMessageState>({
    key: '',
    message: '',
  })
  const [downloading, setDownloading] = useState(false)
  const downloadOptions = buildDownloadOptions(displayDownloads)
  const downloadPresets = buildDownloadPresets(displayDownloads)
  const selectionKey = selectedRun
    ? `${selectedRun.public_run_id}:${downloadOptions.map((option) => option.item).join('|')}`
    : ''
  const messageKey = selectedRun?.public_run_id ?? ''
  const selectedDownloadItems =
    selectedDownloadItemsByKey[selectionKey] ?? defaultDownloadItems(displayDownloads)
  const downloadError =
    downloadErrorState.key === messageKey ? downloadErrorState.message : ''
  const downloadNotice =
    downloadNoticeState.key === messageKey ? downloadNoticeState.message : ''

  function setSelectedDownloadItems(items: string[]) {
    setSelectedDownloadItemsByKey((previous) => ({
      ...previous,
      [selectionKey]: items,
    }))
  }

  function setDownloadError(message: string) {
    setDownloadErrorState({ key: messageKey, message })
  }

  function setDownloadNotice(message: string) {
    setDownloadNoticeState({ key: messageKey, message })
  }

  function applyDownloadPreset(preset: string) {
    setDownloadNotice('')
    const manifestItems = presetDownloadItems(displayDownloads, preset)
    setSelectedDownloadItems(manifestItems.length > 0 ? manifestItems : [])
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
    downloadPresets,
    downloading,
    handleSelectedDownload,
    selectedDownloadItems,
    toggleDownloadItem,
  }
}
